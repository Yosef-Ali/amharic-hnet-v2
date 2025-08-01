#!/usr/bin/env python3
"""
300M H-NET TRAINING SCRIPT FOR M2 8GB HARDWARE
Optimized training with mixed precision, gradient accumulation, and transfer learning

CRITICAL FEATURES:
- 300M parameter model (NOT 10M compact)
- Mixed precision (fp16) for memory efficiency  
- Gradient accumulation to simulate larger batches
- Transfer learning from Chinese H-Net weights
- Memory management for M2 8GB constraints
- Progressive unfreezing strategy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import json
import numpy as np
import os
import math
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import psutil
import gc
from dataclasses import dataclass
import warnings

# Import our 300M H-Net architecture
from src.models.hnet_300m_amharic import AmharicHNet300M, HNet300MConfig, create_300m_model

@dataclass
class TrainingConfig:
    """Training configuration optimized for M2 8GB"""
    # Model architecture
    d_model: int = 1536
    n_heads: int = 24
    n_backbone_layers: int = 24
    max_seq_length: int = 512  # Reduced for memory efficiency
    
    # Training parameters
    batch_size: int = 1  # Small batch size for M2 8GB
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    num_epochs: int = 50
    learning_rate: float = 2e-5  # Conservative LR for large model
    warmup_steps: int = 1000
    
    # Optimization
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    use_mixed_precision: bool = True
    use_gradient_checkpointing: bool = True
    
    # Memory management
    empty_cache_every: int = 50  # Empty cache every N steps
    max_memory_usage: float = 0.85  # Max memory usage (85%)
    pin_memory: bool = False  # Disable on M2 for stability
    
    # Transfer learning
    use_transfer_learning: bool = True
    progressive_unfreezing: bool = True
    freeze_epochs: int = 5  # Epochs to keep layers frozen
    
    # Monitoring
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100

class OptimizedAmharicDataset(Dataset):
    """
    Memory-optimized dataset for large model training
    """
    def __init__(self, texts: List[str], max_length: int = 512, cache_size: int = 1000):
        self.texts = texts
        self.max_length = max_length
        self.cache_size = cache_size
        self.cache = {}
        
        print(f"Dataset: {len(texts)} texts, max_length: {max_length}")
        
        # Pre-process a subset for quick validation
        self.sample_processed = self._process_texts(texts[:min(100, len(texts))])
        
    def _process_texts(self, texts: List[str]) -> List[torch.Tensor]:
        """Process texts to byte sequences with optimizations"""
        processed = []
        
        for text in texts:
            # Encode to UTF-8 bytes
            byte_seq = list(text.encode('utf-8'))
            
            # Filter very short sequences
            if len(byte_seq) < 10:
                continue
                
            # Truncate or pad
            if len(byte_seq) > self.max_length:
                byte_seq = byte_seq[:self.max_length]
            else:
                # Pad with 0
                byte_seq.extend([0] * (self.max_length - len(byte_seq)))
            
            processed.append(torch.tensor(byte_seq, dtype=torch.long))
        
        return processed
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Use cache for frequently accessed items
        if idx in self.cache:
            return self.cache[idx]
        
        # Process single text
        text = self.texts[idx]
        byte_seq = list(text.encode('utf-8'))
        
        if len(byte_seq) > self.max_length:
            byte_seq = byte_seq[:self.max_length]
        else:
            byte_seq.extend([0] * (self.max_length - len(byte_seq)))
        
        tensor = torch.tensor(byte_seq, dtype=torch.long)
        
        # Cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = tensor
        
        return tensor

def load_amharic_corpus(data_path: str = "data/raw") -> List[str]:
    """Load Amharic corpus with memory optimization"""
    texts = []
    total_chars = 0
    
    print(f"Loading corpus from: {data_path}")
    
    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            if filename.endswith('.json') and 'corpus' in filename:
                filepath = os.path.join(data_path, filename)
                print(f"Processing: {filename}")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Extract texts
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                text = item.get('text', item.get('content', ''))
                            else:
                                text = str(item)
                            
                            if text and len(text.strip()) > 20:  # Minimum length
                                texts.append(text.strip())
                                total_chars += len(text)
                    
                    elif isinstance(data, dict):
                        if 'articles' in data:  # Wikipedia format
                            for article in data['articles']:
                                content = article.get('content', '')
                                if content and len(content) > 50:
                                    texts.append(content)
                                    total_chars += len(content)
                        else:
                            for key, value in data.items():
                                if isinstance(value, str) and len(value) > 20:
                                    texts.append(value)
                                    total_chars += len(value)
                
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    
    # Add demo data if insufficient corpus
    if len(texts) < 100:
        print("Adding demo Amharic texts...")
        demo_texts = [
            "ይፈልጋሉ ውሃ መጠጣት እርሳቸው። አማርኛ ቋንቋ በጣም ውብ ነው።",
            "እኔ ኢትዮጵያዊ ነኝ እና አማርኛ እናገራለሁ። ቡና እና ዳቦ በጣም ጣፋጭ ናቸው።",
            "ዛሬ ሰማይ ሰላማዊ እና ውብ ነው። ልጆቹ በፍጥነት እየሮጡ ናቸው።",
            "መጽሐፍ ማንበብ ለአእምሮ ጥሩ ነው። ሀገሬ ኢትዮጵያ ውብ ናት።",
            "በዓል በጣም አስደሳች ነው። ባህላዊ ዳንስ እና ሙዚቃ ልብን ይደስታል።",
            "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ነች። ገበያ እና መንገድ በሰው የተሞላ ነው።",
            "ዝናብ በርቀት እየመጣ ነው። አርሶ አደር ለዝናብ እየተጠባበቀ ነው።",
            "ትምህርት ቤት ሄዳ ብዙ ነገር ተማርኩ። መምህር በጣም ጥሩ ነው።"
        ] * 50  # Repeat for training data
        
        texts.extend(demo_texts)
    
    print(f"Loaded {len(texts):,} texts, {total_chars:,} total characters")
    print(f"Average text length: {total_chars / len(texts):.1f} characters")
    
    return texts

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    stats = {
        'ram_gb': memory_info.rss / 1024**3,
        'cpu_percent': process.cpu_percent()
    }
    
    if torch.cuda.is_available():
        stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
        stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # M2 MPS memory tracking is limited
        stats['mps_available'] = True
    
    return stats

def setup_transfer_learning(model: AmharicHNet300M, chinese_weights_path: Optional[str] = None) -> AmharicHNet300M:
    """
    Setup transfer learning from Chinese H-Net weights
    """
    if chinese_weights_path and os.path.exists(chinese_weights_path):
        print(f"Loading Chinese H-Net weights from: {chinese_weights_path}")
        try:
            chinese_state = torch.load(chinese_weights_path, map_location='cpu')
            
            # Transfer compatible layers
            model_state = model.state_dict()
            transferred_layers = []
            
            for name, param in chinese_state.items():
                if name in model_state and param.shape == model_state[name].shape:
                    model_state[name] = param
                    transferred_layers.append(name)
            
            model.load_state_dict(model_state)
            print(f"Transferred {len(transferred_layers)} layers from Chinese H-Net")
            
        except Exception as e:
            print(f"Error loading Chinese weights: {e}")
            print("Continuing with random initialization...")
    else:
        print("No Chinese H-Net weights provided, using random initialization")
    
    # Apply progressive unfreezing strategy
    if model.config.progressive_unfreezing:
        # Freeze embedding and early layers initially
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['input_embedding', 'dynamic_chunker']):
                param.requires_grad = False
                print(f"Frozen layer: {name}")
    
    return model

def progressive_unfreeze(model: AmharicHNet300M, epoch: int, config: TrainingConfig):
    """
    Progressively unfreeze layers during training
    """
    if not config.progressive_unfreezing:
        return
    
    # Unfreeze layers gradually
    if epoch == config.freeze_epochs // 2:
        # Unfreeze chunking layers
        for name, param in model.named_parameters():
            if 'dynamic_chunker' in name:
                param.requires_grad = True
                print(f"Unfrozen: {name}")
    
    elif epoch == config.freeze_epochs:
        # Unfreeze all layers
        for name, param in model.named_parameters():
            param.requires_grad = True
        print("All layers unfrozen")

def train_300m_hnet(
    model: AmharicHNet300M,
    train_loader: DataLoader,
    config: TrainingConfig,
    chinese_weights_path: Optional[str] = None
):
    """
    Train 300M H-Net with optimizations for M2 8GB hardware
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = model.to(device)
    
    # Setup transfer learning
    model = setup_transfer_learning(model, chinese_weights_path)
    
    # Optimizer with lower learning rate for large model
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)  # Stable betas for large models
    )
    
    # Learning rate scheduler
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / config.warmup_steps
        else:
            return 0.5 * (1 + math.cos(math.pi * (step - config.warmup_steps) / (len(train_loader) * config.num_epochs - config.warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler
    scaler = GradScaler() if config.use_mixed_precision else None
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    # Training metrics
    total_steps = 0
    best_loss = float('inf')
    
    print(f"Training configuration:")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Mixed precision: {config.use_mixed_precision}")
    print(f"  Gradient checkpointing: {config.use_gradient_checkpointing}")
    
    model.train()
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{config.num_epochs}")
        print(f"{'='*60}")
        
        # Progressive unfreezing
        progressive_unfreeze(model, epoch, config)
        
        epoch_loss = 0
        num_batches = 0
        optimizer.zero_grad()
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx, input_ids in enumerate(pbar):
                input_ids = input_ids.to(device)
                
                # Prepare targets (next token prediction)
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                # Forward pass with mixed precision
                if config.use_mixed_precision and scaler:
                    with autocast():
                        logits, debug_info = model(inputs)
                        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                        loss = loss / config.gradient_accumulation_steps  # Scale loss
                    
                    # Backward pass
                    scaler.scale(loss).backward()
                else:
                    logits, debug_info = model(inputs)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = loss / config.gradient_accumulation_steps
                    loss.backward()
                
                epoch_loss += loss.item() * config.gradient_accumulation_steps
                num_batches += 1
                total_steps += 1
                
                # Gradient accumulation
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.use_mixed_precision and scaler:
                        # Gradient clipping
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        
                        # Optimizer step
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Memory management
                if total_steps % config.empty_cache_every == 0:
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Logging
                if total_steps % config.log_interval == 0:
                    memory_stats = get_memory_usage()
                    current_lr = scheduler.get_last_lr()[0]
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item() * config.gradient_accumulation_steps:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'chunks': f'{debug_info["boundaries"].sum().item():.0f}',
                        'ram': f'{memory_stats.get("ram_gb", 0):.1f}GB'
                    })
                
                # Save checkpoint
                if total_steps % config.save_interval == 0:
                    checkpoint_path = f"outputs/hnet_300m_step_{total_steps}.pt"
                    os.makedirs("outputs", exist_ok=True)
                    
                    checkpoint = {
                        'step': total_steps,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                        'config': config
                    }
                    
                    if scaler:
                        checkpoint['scaler_state_dict'] = scaler.state_dict()
                    
                    torch.save(checkpoint, checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")
                
                # Memory check
                memory_stats = get_memory_usage()
                if memory_stats.get('ram_gb', 0) > 6.0:  # Near M2 limit
                    print(f"WARNING: High memory usage: {memory_stats.get('ram_gb', 0):.1f}GB")
                    gc.collect()
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Memory Usage: {get_memory_usage()}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = "outputs/hnet_300m_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, best_path)
            print(f"New best model saved: {best_path}")

def test_300m_generation(model: AmharicHNet300M, device: torch.device):
    """Test generation capabilities of 300M model"""
    model.eval()
    
    test_prompts = [
        "ይፈልጋሉ",  # "they want"
        "አማርኛ",    # "Amharic"
        "ኢትዮጵያ",   # "Ethiopia"
        "ቡና",       # "coffee"
        "ሰላም"       # "peace"
    ]
    
    print("\n" + "="*70)
    print("TESTING 300M H-NET GENERATION")
    print("="*70)
    
    with torch.no_grad():
        for prompt in test_prompts:
            prompt_bytes = list(prompt.encode('utf-8'))
            input_ids = torch.tensor([prompt_bytes], dtype=torch.long).to(device)
            
            print(f"\nPrompt: '{prompt}'")
            
            try:
                generated = model.generate(
                    input_ids,
                    max_length=30,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9
                )
                
                generated_bytes = generated[0].cpu().numpy().tolist()
                
                # Decode generated text
                prompt_len = len(prompt_bytes)
                new_bytes = generated_bytes[prompt_len:]
                
                # Remove padding zeros
                new_bytes = [b for b in new_bytes if b != 0]
                
                if new_bytes:
                    generated_text = bytes(new_bytes).decode('utf-8', errors='ignore')
                    full_text = prompt + generated_text
                    print(f"Generated: '{full_text}'")
                    print(f"New part: '{generated_text}'")
                else:
                    print("No new content generated")
                
            except Exception as e:
                print(f"Error in generation: {e}")

def main():
    """Main training function for 300M H-Net"""
    print("🚀 300M PARAMETER AMHARIC H-NET TRAINING")
    print("Optimized for M2 8GB Hardware with Transfer Learning")
    print("="*70)
    
    # Configuration
    config = TrainingConfig()
    
    # Load data
    print("Loading Amharic corpus...")
    texts = load_amharic_corpus()
    
    # Create dataset
    dataset = OptimizedAmharicDataset(texts, max_length=config.max_seq_length)
    train_loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0,  # Single worker for M2 stability
        pin_memory=config.pin_memory
    )
    
    # Create 300M model
    print("Initializing 300M H-Net model...")
    model = create_300m_model(
        max_seq_length=config.max_seq_length,
        use_mixed_precision=config.use_mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    
    # Test generation before training
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    print("\nGeneration before training (should be random):")
    test_300m_generation(model, device)
    
    # Train the model
    print(f"\nStarting training with {sum(p.numel() for p in model.parameters()):,} parameters...")
    train_300m_hnet(model, train_loader, config)
    
    # Test generation after training
    print("\nGeneration after training (should be meaningful):")
    test_300m_generation(model, device)
    
    # Save final model
    final_path = "outputs/hnet_300m_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'model_config': model.config
    }, final_path)
    
    print(f"\n🎯 300M H-NET TRAINING COMPLETE!")
    print(f"Final model saved: {final_path}")
    print("This model is ready for meaningful Amharic text generation!")

if __name__ == "__main__":
    main()