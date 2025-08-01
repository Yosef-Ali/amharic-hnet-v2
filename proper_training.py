#!/usr/bin/env python3
"""
Proper training script for Amharic H-Net with corrected architecture and training configuration.
This script addresses the issues with undertrained model and UTF-8 generation problems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
from typing import List, Dict, Any
import sys
import os

# Add src to path
sys.path.append('src')

from models.hnet_amharic import AmharicHNet
from preprocessing.prepare_amharic import AmharicPreprocessor

class AmharicTextDataset(Dataset):
    """Dataset for Amharic text training with proper UTF-8 handling."""
    
    def __init__(self, texts: List[str], preprocessor: AmharicPreprocessor, max_length: int = 256):
        self.texts = texts
        self.preprocessor = preprocessor
        self.max_length = max_length
        
        # Prepare all sequences
        self.sequences = []
        for text in texts:
            # Clean and prepare text
            cleaned_text = preprocessor.clean_text(text)
            if len(cleaned_text) < 10:  # Skip very short texts
                continue
                
            # Convert to byte sequence
            byte_seq = preprocessor.extract_byte_sequences(cleaned_text, max_length)
            if len(byte_seq) >= 10:  # Ensure minimum length
                self.sequences.append(byte_seq)
        
        print(f"Prepared {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Create input and target sequences (for language modeling)
        if len(sequence) <= 1:
            # Handle edge case
            input_seq = [0]  # Padding token
            target_seq = [0]
        else:
            input_seq = sequence[:-1]
            target_seq = sequence[1:]
        
        # Pad sequences to max_length
        while len(input_seq) < self.max_length:
            input_seq.append(0)  # Padding
        while len(target_seq) < self.max_length:
            target_seq.append(0)  # Padding
            
        # Truncate if too long
        input_seq = input_seq[:self.max_length]
        target_seq = target_seq[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'target_ids': torch.tensor(target_seq, dtype=torch.long),
            'attention_mask': torch.tensor([1 if x != 0 else 0 for x in input_seq], dtype=torch.long)
        }

class ProperAmharicTrainer:
    """Proper trainer with corrected configuration for good convergence."""
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        self.model_config = model_config
        self.training_config = training_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Initialize model with proper architecture
        self.model = AmharicHNet(
            d_model=model_config['d_model'],
            n_encoder_layers=model_config['n_encoder_layers'],
            n_decoder_layers=model_config['n_decoder_layers'],
            n_main_layers=model_config['n_main_layers'],
            n_heads=model_config['n_heads'],
            compression_ratio=model_config['compression_ratio'],
            vocab_size=256  # Byte-level
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup optimizer with proper learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Setup learning rate scheduler
        self.scheduler = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_losses = []
        self.validation_losses = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        # Cosine annealing with warmup
        warmup_steps = int(0.1 * num_training_steps)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def compute_loss(self, batch):
        """Compute training loss with proper handling."""
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass
        logits, boundary_probs = self.model(input_ids, target_ids)
        
        # Language modeling loss
        lm_loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Boundary regularization (light regularization)
        boundary_reg_loss = torch.mean(torch.abs(boundary_probs - 0.5)) * 0.01
        
        # Total loss
        total_loss = lm_loss + boundary_reg_loss
        
        return total_loss, {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'boundary_reg_loss': boundary_reg_loss.item()
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            # Compute loss
            loss, loss_dict = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss']
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log periodically
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch + 1}, Step {self.global_step}: "
                    f"Loss={loss_dict['total_loss']:.4f}, LR={current_lr:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        self.training_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                loss, loss_dict = self.compute_loss(batch)
                total_loss += loss_dict['total_loss']
        
        avg_loss = total_loss / num_batches
        self.validation_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, output_dir: str, is_best: bool = False):
        """Save model checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = output_path / f'checkpoint_epoch_{self.current_epoch + 1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = output_path / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with loss: {self.best_loss:.4f}")
        
        # Save latest checkpoint
        latest_path = output_path / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        return str(checkpoint_path)
    
    def test_generation(self, preprocessor: AmharicPreprocessor):
        """Test model generation to check for UTF-8 issues."""
        self.model.eval()
        
        test_prompts = [
            "ኢትዮጵያ",
            "ቡና", 
            "አማርኛ",
        ]
        
        print("\n" + "="*50)
        print("Testing Model Generation")
        print("="*50)
        
        with torch.no_grad():
            for prompt in test_prompts:
                print(f"\nPrompt: {prompt}")
                
                # Convert to byte sequence
                byte_seq = preprocessor.extract_byte_sequences(prompt, max_length=50)
                input_ids = torch.tensor([byte_seq], dtype=torch.long).to(self.device)
                
                try:
                    # Generate
                    generated = self.model.generate(
                        input_ids,
                        max_length=30,
                        temperature=0.8,
                        top_k=50
                    )
                    
                    # Decode
                    generated_bytes = generated[0].cpu().numpy().tolist()
                    generated_text = preprocessor.decode_byte_sequence(generated_bytes)
                    
                    print(f"Generated: {generated_text}")
                    
                    # Check for Amharic content
                    amharic_chars = sum(1 for c in generated_text if 0x1200 <= ord(c) <= 0x137F)
                    print(f"Amharic chars: {amharic_chars}/{len(generated_text)}")
                    
                except Exception as e:
                    print(f"Generation error: {e}")
    
    def train(self, train_loader, val_loader, num_epochs: int, output_dir: str):
        """Main training loop."""
        self.logger.info("Starting proper Amharic H-Net training...")
        self.logger.info(f"Total epochs: {num_epochs}")
        self.logger.info(f"Steps per epoch: {len(train_loader)}")
        self.logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Setup scheduler
        total_steps = num_epochs * len(train_loader)
        self.setup_scheduler(total_steps)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Log epoch results
            self.logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
            )
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            checkpoint_path = self.save_checkpoint(output_dir, is_best=is_best)
            
            # Test generation every few epochs
            if (epoch + 1) % 2 == 0:
                preprocessor = AmharicPreprocessor()
                self.test_generation(preprocessor)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_loss:.4f}")
        
        return checkpoint_path

def load_texts(file_path: str) -> List[str]:
    """Load texts from file."""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts

def main():
    """Main training function."""
    
    # Model configuration for proper convergence
    model_config = {
        'd_model': 384,          # Smaller for faster training but still capable
        'n_encoder_layers': 3,   # Sufficient for byte-level processing
        'n_decoder_layers': 3,   # Balanced architecture
        'n_main_layers': 6,      # Main processing layers
        'n_heads': 6,           # Attention heads
        'compression_ratio': 4.0 # Reasonable compression
    }
    
    # Training configuration for good convergence
    training_config = {
        'num_epochs': 8,         # Proper training duration
        'batch_size': 8,         # Good balance for memory and gradient estimates
        'learning_rate': 2e-4,   # Higher LR for faster convergence
        'weight_decay': 0.01,    # Regularization
        'max_length': 256,       # Reasonable sequence length
    }
    
    print("Loading training data...")
    texts = load_texts('data/processed/amharic_texts.txt')
    print(f"Loaded {len(texts)} texts")
    
    if len(texts) == 0:
        print("ERROR: No training texts found!")
        return
    
    # Create preprocessor
    preprocessor = AmharicPreprocessor()
    
    # Create datasets
    print("Creating datasets...")
    
    # Split data
    split_idx = int(0.9 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    print(f"Training texts: {len(train_texts)}")
    print(f"Validation texts: {len(val_texts)}")
    
    train_dataset = AmharicTextDataset(train_texts, preprocessor, training_config['max_length'])
    val_dataset = AmharicTextDataset(val_texts, preprocessor, training_config['max_length'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create trainer
    trainer = ProperAmharicTrainer(model_config, training_config)
    
    # Train the model
    output_dir = "outputs/proper_training"
    checkpoint_path = trainer.train(
        train_loader, 
        val_loader, 
        training_config['num_epochs'],
        output_dir
    )
    
    print(f"\nTraining completed! Best checkpoint saved at: {checkpoint_path}")
    print(f"Final training loss: {trainer.training_losses[-1]:.4f}")
    print(f"Final validation loss: {trainer.validation_losses[-1]:.4f}")
    print(f"Best validation loss: {trainer.best_loss:.4f}")
    
    # Final generation test
    print("\nFinal generation test:")
    trainer.test_generation(preprocessor)

if __name__ == "__main__":
    main()