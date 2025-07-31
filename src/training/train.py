import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import logging
import wandb
from typing import Dict, List, Optional, Tuple
import argparse
import yaml
import os
import time

from ..models.hnet_amharic import AmharicHNet
from ..preprocessing.prepare_amharic import AmharicPreprocessor
from ..safety.cultural_guardrails import AmharicCulturalGuardrails
from .data_loader import create_data_loaders, load_data_from_directory


class AmharicHNetTrainer:
    """
    Trainer class for Amharic H-Net model with cultural safety integration.
    """
    
    def __init__(
        self,
        model: AmharicHNet,
        config: Dict,
        device: torch.device,
        output_dir: str = "outputs"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.preprocessor = AmharicPreprocessor()
        self.cultural_guardrails = AmharicCulturalGuardrails()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = None
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.boundary_criterion = nn.BCELoss()
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with different learning rates for different components."""
        # Separate parameters for different learning rates
        model_params = []
        chunker_params = []
        
        for name, param in self.model.named_parameters():
            if 'chunker' in name:
                chunker_params.append(param)
            else:
                model_params.append(param)
        
        optimizer_config = self.config.get('optimizer', {})
        lr = optimizer_config.get('learning_rate', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 0.01)
        
        # Use different learning rates for chunker vs main model
        param_groups = [
            {'params': model_params, 'lr': lr},
            {'params': chunker_params, 'lr': lr * 0.5}  # Lower LR for chunker
        ]
        
        optimizer_type = optimizer_config.get('type', 'adamw')
        
        if optimizer_type.lower() == 'adamw':
            return optim.AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adam':
            return optim.Adam(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
    
    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'onecycle':
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=scheduler_config.get('max_lr', 1e-3),
                total_steps=num_training_steps,
                pct_start=scheduler_config.get('pct_start', 0.1)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
    
    def compute_loss(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss with cultural safety considerations.
        """
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
        
        # Boundary regularization loss
        # Encourage meaningful chunking by penalizing extreme boundary probabilities
        boundary_reg_loss = torch.mean(
            (boundary_probs - 0.5) ** 2
        ) * self.config.get('boundary_reg_weight', 0.1)
        
        # Compression ratio loss
        # Encourage target compression ratio
        target_compression = self.config.get('target_compression', 4.5)
        actual_compression = input_ids.size(1) / (boundary_probs.sum(dim=1).mean() + 1e-8)
        compression_loss = torch.abs(
            actual_compression - target_compression
        ) * self.config.get('compression_loss_weight', 0.01)
        
        # Cultural safety loss (if texts are available)
        cultural_loss = torch.tensor(0.0, device=self.device)
        if 'texts' in batch:
            cultural_violations = 0
            for text in batch['texts']:
                is_safe, violations = self.cultural_guardrails.check_cultural_safety(text)
                if not is_safe:
                    cultural_violations += len(violations)
            
            if cultural_violations > 0:
                cultural_loss = torch.tensor(
                    cultural_violations * self.config.get('cultural_loss_weight', 0.1),
                    device=self.device
                )
        
        # Total loss
        total_loss = (
            lm_loss + 
            boundary_reg_loss + 
            compression_loss + 
            cultural_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'boundary_reg_loss': boundary_reg_loss.item(),
            'compression_loss': compression_loss.item(),
            'cultural_loss': cultural_loss.item(),
            'compression_ratio': actual_compression.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {
            'total_loss': 0.0,
            'lm_loss': 0.0,
            'boundary_reg_loss': 0.0,
            'compression_loss': 0.0,
            'cultural_loss': 0.0,
            'compression_ratio': 0.0
        }
        
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # Compute loss
            loss, loss_dict = self.compute_loss(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('max_grad_norm', 1.0)
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'compression': f"{loss_dict['compression_ratio']:.2f}"
            })
            
            # Log to tensorboard
            self.writer.add_scalar('train/batch_loss', loss_dict['total_loss'], self.global_step)
            self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
            
            # Periodic logging
            if self.global_step % self.config.get('log_interval', 100) == 0:
                self.logger.info(
                    f"Step {self.global_step}: Loss={loss_dict['total_loss']:.4f}, "
                    f"LM Loss={loss_dict['lm_loss']:.4f}, "
                    f"Compression={loss_dict['compression_ratio']:.2f}"
                )
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'lm_loss': 0.0,
            'boundary_reg_loss': 0.0,
            'compression_loss': 0.0,
            'cultural_loss': 0.0,
            'compression_ratio': 0.0
        }
        
        num_batches = 0
        cultural_violations_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                loss, loss_dict = self.compute_loss(batch)
                
                # Update metrics
                for key, value in loss_dict.items():
                    total_losses[key] += value
                num_batches += 1
                
                # Check cultural safety on validation texts
                if 'texts' in batch:
                    for text in batch['texts']:
                        is_safe, violations = self.cultural_guardrails.check_cultural_safety(text)
                        if not is_safe:
                            cultural_violations_total += len(violations)
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        avg_losses['cultural_violations_per_batch'] = cultural_violations_total / num_batches
        
        return avg_losses
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
        
        # Save epoch checkpoint
        epoch_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int,
        save_every: int = 1,
        eval_every: int = 1
    ):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {num_epochs}")
        self.logger.info(f"Steps per epoch: {len(train_loader)}")
        
        # Setup scheduler
        total_steps = num_epochs * len(train_loader)
        self._setup_scheduler(total_steps)
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Log training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, epoch)
            
            self.logger.info(
                f"Epoch {epoch}: Train Loss={train_metrics['total_loss']:.4f}, "
                f"Compression Ratio={train_metrics['compression_ratio']:.2f}"
            )
            
            # Validation
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.validate(val_loader)
                
                # Log validation metrics
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                
                self.logger.info(
                    f"Epoch {epoch}: Val Loss={val_metrics['total_loss']:.4f}, "
                    f"Cultural Violations={val_metrics['cultural_violations_per_batch']:.2f}"
                )
                
                # Check if best model
                is_best = val_metrics['total_loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['total_loss']
                
                # Save checkpoint
                if (epoch + 1) % save_every == 0:
                    self.save_checkpoint(is_best=is_best)
        
        self.logger.info("Training completed!")
        self.writer.close()


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train Amharic H-Net model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Path to training data directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project="amharic-hnet",
            config=config,
            name=f"train-{int(time.time())}"
        )
    
    # Load data
    print("Loading data...")
    train_texts, val_texts = load_data_from_directory(
        args.data_dir,
        train_ratio=config.get('data', {}).get('train_split', 0.9)
    )
    
    print(f"Training samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    
    # Create data loaders
    preprocessor = AmharicPreprocessor()
    train_loader, val_loader = create_data_loaders(
        train_texts=train_texts,
        val_texts=val_texts,
        preprocessor=preprocessor,
        batch_size=config.get('training', {}).get('batch_size', 16),
        max_length=config.get('data', {}).get('max_length', 512),
        num_workers=config.get('training', {}).get('num_workers', 4)
    )
    
    # Initialize model
    model_config = config.get('model', {})
    model = AmharicHNet(
        d_model=model_config.get('d_model', 768),
        n_encoder_layers=model_config.get('n_encoder_layers', 4),
        n_decoder_layers=model_config.get('n_decoder_layers', 4),
        n_main_layers=model_config.get('n_main_layers', 12),
        n_heads=model_config.get('n_heads', 12),
        compression_ratio=model_config.get('compression_ratio', 4.5)
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = AmharicHNetTrainer(
        model=model,
        config=config,
        device=device,
        output_dir=args.output_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train the model
    training_config = config.get('training', {})
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=training_config.get('num_epochs', 10),
        save_every=training_config.get('save_every', 1),
        eval_every=training_config.get('eval_every', 1)
    )


if __name__ == "__main__":
    main()