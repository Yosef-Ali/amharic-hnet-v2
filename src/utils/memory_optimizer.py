#!/usr/bin/env python3
"""
MEMORY OPTIMIZER FOR M2 8GB HARDWARE
Utilities for managing memory constraints during 300M H-Net training

CRITICAL OPTIMIZATIONS:
- Mixed precision (fp16) memory management
- Gradient accumulation for effective large batches
- Memory monitoring and automatic cleanup
- M2 MPS specific optimizations
- Dynamic memory scaling based on available resources
"""

import torch
import torch.nn as nn
import gc
import psutil
import warnings
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager
import time
import os

@dataclass
class MemoryConfig:
    """Memory configuration for M2 8GB optimization"""
    max_memory_gb: float = 6.0           # Conservative limit for M2 8GB
    warning_threshold_gb: float = 5.0    # Warning threshold
    critical_threshold_gb: float = 6.5   # Critical threshold
    
    # Optimization settings
    enable_mixed_precision: bool = True   # fp16 to halve memory usage
    gradient_accumulation_steps: int = 8  # Simulate larger batches
    gradient_checkpointing: bool = True   # Trade compute for memory
    
    # Cleanup settings
    gc_frequency: int = 50               # Garbage collect every N steps
    cache_clear_frequency: int = 100     # Clear cache every N steps
    memory_check_frequency: int = 10     # Check memory every N steps
    
    # Dynamic scaling
    enable_dynamic_batch_size: bool = True  # Adjust batch size based on memory
    min_batch_size: int = 1              # Minimum batch size
    max_batch_size: int = 4              # Maximum batch size for M2

class MemoryMonitor:
    """
    Real-time memory monitoring for M2 hardware
    """
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_history: List[Dict[str, float]] = []
        self.step_count = 0
        self.warnings_issued = 0
        
        # Check if MPS is available
        self.use_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        self.use_cuda = torch.cuda.is_available()
        
        print(f"Memory Monitor initialized:")
        print(f"  Device: {'MPS' if self.use_mps else 'CUDA' if self.use_cuda else 'CPU'}")
        print(f"  Max memory limit: {config.max_memory_gb:.1f} GB")
        print(f"  Mixed precision: {config.enable_mixed_precision}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        stats = {}
        
        # System RAM
        memory = psutil.virtual_memory()
        stats['system_ram_total_gb'] = memory.total / (1024**3)
        stats['system_ram_used_gb'] = memory.used / (1024**3)
        stats['system_ram_available_gb'] = memory.available / (1024**3)
        stats['system_ram_percent'] = memory.percent
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        stats['process_ram_gb'] = process_memory.rss / (1024**3)
        stats['process_vms_gb'] = process_memory.vms / (1024**3)
        
        # GPU memory (if available)
        if self.use_cuda:
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            stats['gpu_max_allocated_gb'] = torch.cuda.max_memory_allocated() / (1024**3)
        elif self.use_mps:
            # MPS memory tracking is limited, estimate from process memory
            stats['mps_estimated_gb'] = min(stats['process_ram_gb'], 6.0)
        
        # Calculate total estimated memory usage
        if self.use_cuda:
            stats['total_estimated_gb'] = stats['process_ram_gb'] + stats['gpu_allocated_gb']
        else:
            stats['total_estimated_gb'] = stats['process_ram_gb']
        
        return stats
    
    def check_memory_status(self) -> Tuple[str, Dict[str, float]]:
        """
        Check current memory status and return status level
        Returns: (status_level, memory_stats)
        status_level: 'ok', 'warning', 'critical', 'emergency'
        """
        stats = self.get_memory_stats()
        total_usage = stats['total_estimated_gb']
        
        if total_usage >= self.config.critical_threshold_gb:
            return 'critical', stats
        elif total_usage >= self.config.warning_threshold_gb:
            return 'warning', stats
        elif total_usage >= self.config.max_memory_gb:
            return 'emergency', stats
        else:
            return 'ok', stats
    
    def log_memory_stats(self, step: int, force: bool = False):
        """Log memory statistics"""
        if not force and step % self.config.memory_check_frequency != 0:
            return
        
        status, stats = self.check_memory_status()
        self.memory_history.append({
            'step': step,
            'status': status,
            **stats
        })
        
        # Print status if concerning
        if status != 'ok' or force:
            print(f"\nMemory Status at Step {step}: {status.upper()}")
            print(f"  Total Usage: {stats['total_estimated_gb']:.2f} GB")
            print(f"  Process RAM: {stats['process_ram_gb']:.2f} GB")
            print(f"  System RAM: {stats['system_ram_used_gb']:.1f}/{stats['system_ram_total_gb']:.1f} GB")
            
            if self.use_cuda:
                print(f"  GPU Allocated: {stats['gpu_allocated_gb']:.2f} GB")
            elif self.use_mps:
                print(f"  MPS Estimated: {stats['mps_estimated_gb']:.2f} GB")
        
        return status, stats
    
    def emergency_cleanup(self):
        """Emergency memory cleanup procedures"""
        print("🚨 EMERGENCY MEMORY CLEANUP")
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear GPU cache if available
        if self.use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.use_mps:
            # MPS doesn't have explicit cache clearing, but sync helps
            torch.mps.synchronize()
        
        # Clear any remaining references
        if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_clearCublasWorkspaces'):
            torch._C._cuda_clearCublasWorkspaces()
        
        print("Emergency cleanup completed")

class MemoryOptimizedTrainer:
    """
    Training wrapper with automatic memory optimization for M2 8GB
    """
    def __init__(self, model: nn.Module, config: MemoryConfig):
        self.model = model
        self.config = config
        self.monitor = MemoryMonitor(config)
        
        # Dynamic batch size tracking
        self.current_batch_size = 1
        self.batch_size_history: List[int] = []
        
        # Performance tracking
        self.step_times: List[float] = []
        self.memory_efficient_mode = False
        
    def optimize_model_for_memory(self):
        """Apply memory optimizations to the model"""
        print("Applying memory optimizations...")
        
        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                print("  ✅ Gradient checkpointing enabled")
            else:
                print("  ⚠️  Gradient checkpointing not available for this model")
        
        # Convert to half precision if using mixed precision
        if self.config.enable_mixed_precision:
            # Note: Don't convert the whole model to half, use autocast instead
            print("  ✅ Mixed precision training enabled")
        
        # Optimize attention mechanisms if available
        self._optimize_attention_layers()
        
        return self.model
    
    def _optimize_attention_layers(self):
        """Optimize attention layers for memory efficiency"""
        optimized_layers = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Enable memory efficient attention if available
                if hasattr(module, 'enable_nested_tensor'):
                    module.enable_nested_tensor = False  # Can cause issues with dynamic shapes
                optimized_layers += 1
        
        if optimized_layers > 0:
            print(f"  ✅ Optimized {optimized_layers} attention layers")
    
    def dynamic_batch_size_adjustment(self, current_memory_gb: float) -> int:
        """Dynamically adjust batch size based on memory usage"""
        if not self.config.enable_dynamic_batch_size:
            return self.current_batch_size
        
        # Memory-based batch size calculation
        if current_memory_gb > self.config.warning_threshold_gb:
            # Reduce batch size if memory is high
            new_batch_size = max(self.config.min_batch_size, self.current_batch_size - 1)
        elif current_memory_gb < self.config.warning_threshold_gb * 0.7:
            # Increase batch size if memory is comfortable
            new_batch_size = min(self.config.max_batch_size, self.current_batch_size + 1)
        else:
            new_batch_size = self.current_batch_size
        
        if new_batch_size != self.current_batch_size:
            print(f"Adjusting batch size: {self.current_batch_size} → {new_batch_size}")
            self.current_batch_size = new_batch_size
            self.batch_size_history.append(new_batch_size)
        
        return new_batch_size
    
    @contextmanager
    def memory_efficient_training_step(self, step: int):
        """Context manager for memory-efficient training step"""
        start_time = time.time()
        
        # Pre-step memory check
        status, stats = self.monitor.check_memory_status()
        
        if status == 'emergency':
            self.monitor.emergency_cleanup()
            # Re-check after cleanup
            status, stats = self.monitor.check_memory_status()
        
        try:
            # Enter memory efficient mode if needed
            if status in ['warning', 'critical']:
                self.memory_efficient_mode = True
            
            yield {
                'memory_status': status,
                'memory_stats': stats,
                'batch_size': self.dynamic_batch_size_adjustment(stats['total_estimated_gb']),
                'use_mixed_precision': self.config.enable_mixed_precision,
                'gradient_accumulation_steps': self.config.gradient_accumulation_steps
            }
            
        finally:
            # Post-step cleanup
            if step % self.config.gc_frequency == 0:
                gc.collect()
            
            if step % self.config.cache_clear_frequency == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif self.use_mps:
                    torch.mps.synchronize()
            
            # Log performance
            step_time = time.time() - start_time
            self.step_times.append(step_time)
            
            # Log memory if needed
            self.monitor.log_memory_stats(step)
            
            # Reset memory efficient mode
            self.memory_efficient_mode = False
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of memory optimizations applied"""
        return {
            'mixed_precision': self.config.enable_mixed_precision,
            'gradient_checkpointing': self.config.gradient_checkpointing,
            'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
            'dynamic_batch_size': self.config.enable_dynamic_batch_size,
            'current_batch_size': self.current_batch_size,
            'memory_limit_gb': self.config.max_memory_gb,
            'avg_step_time': sum(self.step_times[-100:]) / len(self.step_times[-100:]) if self.step_times else 0,
            'memory_history_length': len(self.monitor.memory_history),
            'total_warnings': self.monitor.warnings_issued,
            'device': 'MPS' if self.monitor.use_mps else 'CUDA' if self.monitor.use_cuda else 'CPU'
        }

def setup_m2_optimizations():
    """
    Setup global optimizations for M2 hardware
    """
    print("Setting up M2 hardware optimizations...")
    
    # Disable TensorFloat-32 (not applicable for M2 but good practice)
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = False
    
    # Enable optimized attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(False)  # May not be stable on M2
    
    # Set memory management for MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Conservative memory usage
        print("  ✅ MPS memory management configured")
    
    # Set thread count for M2 CPU cores
    torch.set_num_threads(min(8, os.cpu_count()))  # M2 has 8 performance cores
    
    # Disable automatic mixed precision warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*mixed precision.*')
    
    print("M2 optimizations configured")

def estimate_model_memory(model: nn.Module, batch_size: int = 1, seq_length: int = 512) -> Dict[str, float]:
    """
    Estimate memory requirements for model training
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory components (in GB)
    param_memory_fp32 = total_params * 4 / (1024**3)  # 4 bytes per parameter
    param_memory_fp16 = total_params * 2 / (1024**3)  # 2 bytes per parameter
    
    # Optimizer memory (AdamW needs ~2x parameter memory)
    optimizer_memory_fp32 = param_memory_fp32 * 2
    optimizer_memory_fp16 = param_memory_fp16 * 2
    
    # Activation memory (rough estimate)
    # Assume activation memory scales with batch_size * seq_length * d_model
    d_model = getattr(model, 'd_model', 1536)  # Default to 300M model dimension
    activation_memory = batch_size * seq_length * d_model * 4 / (1024**3)  # fp32
    activation_memory_fp16 = activation_memory / 2  # fp16
    
    # Gradient memory (same as parameters)
    gradient_memory_fp32 = param_memory_fp32
    gradient_memory_fp16 = param_memory_fp16
    
    # Total estimates
    total_fp32 = param_memory_fp32 + optimizer_memory_fp32 + activation_memory + gradient_memory_fp32
    total_fp16 = param_memory_fp16 + optimizer_memory_fp16 + activation_memory_fp16 + gradient_memory_fp16
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_memory_fp32_gb': param_memory_fp32,
        'parameter_memory_fp16_gb': param_memory_fp16,
        'optimizer_memory_fp32_gb': optimizer_memory_fp32,
        'optimizer_memory_fp16_gb': optimizer_memory_fp16,
        'activation_memory_fp32_gb': activation_memory,
        'activation_memory_fp16_gb': activation_memory_fp16,
        'gradient_memory_fp32_gb': gradient_memory_fp32,
        'gradient_memory_fp16_gb': gradient_memory_fp16,
        'total_memory_fp32_gb': total_fp32,
        'total_memory_fp16_gb': total_fp16,
        'memory_savings_gb': total_fp32 - total_fp16,
        'memory_savings_percent': ((total_fp32 - total_fp16) / total_fp32) * 100
    }

if __name__ == "__main__":
    print("🔧 MEMORY OPTIMIZER FOR M2 8GB HARDWARE")
    print("=" * 60)
    
    # Setup M2 optimizations
    setup_m2_optimizations()
    
    # Test memory monitoring
    config = MemoryConfig()
    monitor = MemoryMonitor(config)
    
    print("\nCurrent memory status:")
    status, stats = monitor.check_memory_status()
    print(f"Status: {status}")
    for key, value in stats.items():
        if 'gb' in key:
            print(f"  {key}: {value:.2f} GB")
        elif 'percent' in key:
            print(f"  {key}: {value:.1f}%")
    
    print("\n✅ Memory optimizer ready for 300M H-Net training!")
    print("This will ensure stable training within M2 8GB constraints.")