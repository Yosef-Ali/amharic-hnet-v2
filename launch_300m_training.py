#!/usr/bin/env python3
"""
LAUNCH SCRIPT FOR 300M AMHARIC H-NET TRAINING
Simple interface to start optimized training on M2 8GB hardware

USAGE:
  python launch_300m_training.py              # Start training with defaults
  python launch_300m_training.py --test-only  # Test generation only
  python launch_300m_training.py --resume    # Resume from checkpoint
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check if we're in the right directory
    if not os.path.exists('src/models/hnet_300m_amharic.py'):
        print("❌ Error: Please run from the amharic-hnet-v2 directory")
        return False
    
    # Check if config exists
    config_path = 'configs/hnet_300m_config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Error: Configuration file not found: {config_path}")
        return False
    
    # Check if data directory exists
    data_dir = 'data/raw'
    if not os.path.exists(data_dir):
        print(f"⚠️  Warning: Data directory not found: {data_dir}")
        print("   Training will use demo data if no corpus is available")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required")
        return False
    
    print("✅ Requirements check passed")
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        'outputs/300m',
        'checkpoints/300m', 
        'logs',
        'evaluation_results/300m'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories setup complete")

def check_memory():
    """Check available memory"""
    print("💾 Checking system memory...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        
        print(f"   Total RAM: {total_gb:.1f} GB")
        print(f"   Available: {available_gb:.1f} GB")
        
        if available_gb < 4.0:
            print("⚠️  Warning: Low available memory. Consider closing other applications.")
        elif available_gb >= 6.0:
            print("✅ Sufficient memory available")
        else:
            print("⚠️  Warning: Memory may be tight for 300M model training")
        
    except ImportError:
        print("   Could not check memory (psutil not available)")

def check_device():
    """Check available compute device"""
    print("🖥️  Checking compute device...")
    
    try:
        import torch
        
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) available - M1/M2 Mac")
            return "mps"
        elif torch.cuda.is_available():
            print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name()}")
            return "cuda"
        else:
            print("⚠️  CPU only - training will be slow")
            return "cpu"
            
    except ImportError:
        print("❌ PyTorch not available")
        return None

def estimate_training_time(device_type: str, num_epochs: int = 50):
    """Provide training time estimates"""
    print("⏱️  Training time estimates:")
    
    estimates = {
        "mps": "2-4 hours",     # M1/M2 optimized
        "cuda": "1-2 hours",    # GPU accelerated
        "cpu": "8-12 hours"     # CPU fallback
    }
    
    print(f"   Device: {device_type.upper()}")
    print(f"   Estimated time for {num_epochs} epochs: {estimates.get(device_type, 'Unknown')}")
    print("   (Actual time depends on data size and batch size)")

def show_configuration_summary():
    """Show key configuration settings"""
    print("\n" + "="*60)
    print("300M AMHARIC H-NET CONFIGURATION SUMMARY")
    print("="*60)
    print("Model Architecture:")
    print("  • Parameters: ~300 Million")
    print("  • Hidden dimension: 1536")
    print("  • Attention heads: 24") 
    print("  • Backbone layers: 24")
    print("  • Vocabulary: 256 (byte-level)")
    print("")
    print("Memory Optimizations:")
    print("  • Mixed precision (fp16) training")
    print("  • Gradient accumulation (effective batch size: 8)")
    print("  • Gradient checkpointing")
    print("  • Dynamic memory management")
    print("")
    print("Training Features:")
    print("  • Transfer learning from Chinese H-Net")
    print("  • Progressive layer unfreezing")
    print("  • Cultural safety monitoring")
    print("  • Automatic checkpointing")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Launch 300M Amharic H-Net Training")
    parser.add_argument('--test-only', action='store_true',
                       help='Only test generation, no training')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--chinese-weights', type=str, default=None,
                       help='Path to Chinese H-Net weights for transfer learning')
    parser.add_argument('--data-dir', type=str, default='data/raw',
                       help='Directory containing training data')
    parser.add_argument('--config', type=str, default='configs/hnet_300m_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip requirement checks')
    
    args = parser.parse_args()
    
    print("🚀 300M AMHARIC H-NET TRAINING LAUNCHER")
    print("Optimized for M2 8GB Hardware")
    print("="*60)
    
    # Run checks unless skipped
    if not args.skip_checks:
        if not check_requirements():
            print("❌ Requirement checks failed. Exiting.")
            return 1
        
        setup_directories()
        check_memory()
        device_type = check_device()
        
        if device_type is None:
            print("❌ No compatible compute device found. Exiting.")
            return 1
        
        if not args.test_only:
            estimate_training_time(device_type)
    
    # Show configuration
    show_configuration_summary()
    
    # Confirm before starting
    if not args.test_only and not args.skip_checks:
        print("\n" + "⚠️  IMPORTANT NOTES:")
        print("• This will train a 300M parameter model")
        print("• Training may take several hours")
        print("• Ensure stable power connection for laptops")
        print("• Model checkpoints will be saved automatically")
        print("• You can interrupt training with Ctrl+C safely")
        
        response = input("\nStart training? (y/N): ").lower().strip()
        if response != 'y':
            print("Training cancelled.")
            return 0
    
    # Build command
    cmd = [
        sys.executable, 
        'train_hnet_300m_optimized.py',
        '--config', args.config,
        '--data-dir', args.data_dir
    ]
    
    if args.test_only:
        cmd.append('--test-only')
    
    if args.resume:
        cmd.extend(['--resume', args.resume])
    
    if args.chinese_weights:
        cmd.extend(['--chinese-weights', args.chinese_weights])
    
    # Launch training
    print("\n🔥 LAUNCHING 300M H-NET TRAINING...")
    print("Command:", ' '.join(cmd))
    print("="*60)
    
    try:
        # Run the training script
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Output files:")
        print("  • Best model: outputs/300m/hnet_300m_best.pt")
        print("  • Final model: outputs/300m/hnet_300m_final.pt")
        print("  • Training logs: training_300m.log")
        print("  • Generation results: outputs/300m/generation_results.json")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code: {e.returncode}")
        print("Check the logs for detailed error information.")
        return e.returncode
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")
        print("Checkpoint should be saved automatically")
        print("Use --resume to continue training later")
        return 130

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)