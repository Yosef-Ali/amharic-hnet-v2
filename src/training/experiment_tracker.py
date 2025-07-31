"""
Experiment tracking utilities for Amharic H-Net v2.
Supports Weights & Biases and TensorBoard logging.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    # Experiment metadata
    experiment_name: str
    project_name: str = "amharic-hnet-v2"
    description: str = ""
    tags: list = None
    
    # Tracking backends
    use_wandb: bool = True
    use_tensorboard: bool = True
    
    # Output directories
    output_dir: str = "./outputs"
    tensorboard_dir: str = "./outputs/tensorboard"
    wandb_dir: str = "./outputs/wandb"
    
    # Logging configuration
    log_frequency: int = 100
    save_frequency: int = 1000
    eval_frequency: int = 500
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ExperimentTracker:
    """Central experiment tracking manager."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tb_writer: Optional[SummaryWriter] = None
        self.wandb_run = None
        self.step = 0
        
        # Create output directories
        self._setup_directories()
        
        # Initialize tracking backends
        if config.use_tensorboard:
            self._init_tensorboard()
        
        if config.use_wandb and self._wandb_available():
            self._init_wandb()
    
    def _setup_directories(self):
        """Create necessary directories for experiment tracking."""
        for dir_path in [self.config.output_dir, self.config.tensorboard_dir, self.config.wandb_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _wandb_available(self) -> bool:
        """Check if W&B is available and configured."""
        try:
            # Check if WANDB_API_KEY is set or wandb is logged in
            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                return True
            
            # Try to get the current user (will fail if not logged in)
            import wandb.api
            wandb.api.Api()
            return True
        except Exception:
            print("W&B not configured. Skipping W&B logging.")
            return False
    
    def _init_tensorboard(self):
        """Initialize TensorBoard writer."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(self.config.tensorboard_dir) / f"{self.config.experiment_name}_{timestamp}"
        self.tb_writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logging to: {log_dir}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                notes=self.config.description,
                tags=self.config.tags,
                dir=self.config.wandb_dir,
                reinit=True
            )
            print(f"W&B run initialized: {self.wandb_run.url}")
        except Exception as e:
            print(f"Failed to initialize W&B: {e}")
            self.config.use_wandb = False
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        if self.config.use_wandb and self.wandb_run:
            wandb.config.update(config)
        
        if self.config.use_tensorboard and self.tb_writer:
            # Log config as text in TensorBoard
            config_text = json.dumps(config, indent=2)
            self.tb_writer.add_text("config", config_text, 0)
        
        # Save config to file
        config_file = Path(self.config.output_dir) / "experiment_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """Log training metrics."""
        if step is not None:
            self.step = step
        else:
            self.step += 1
        
        # Log to W&B
        if self.config.use_wandb and self.wandb_run:
            wandb.log(metrics, step=self.step)
        
        # Log to TensorBoard
        if self.config.use_tensorboard and self.tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(key, value, self.step)
    
    def log_model_summary(self, model: torch.nn.Module, input_size: tuple = None):
        """Log model architecture summary."""
        if self.config.use_wandb and self.wandb_run:
            wandb.watch(model, log="all", log_freq=self.config.log_frequency)
        
        # Create model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        }
        
        self.log_metrics(summary)
        
        # Log architecture as text
        if self.config.use_tensorboard and self.tb_writer:
            model_str = str(model)
            self.tb_writer.add_text("model_architecture", model_str, 0)
    
    def log_learning_curves(self, train_metrics: Dict[str, float], 
                           val_metrics: Dict[str, float] = None):
        """Log learning curves for training and validation."""
        # Prefix metrics for clarity
        train_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
        metrics = train_metrics.copy()
        
        if val_metrics:
            val_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
            metrics.update(val_metrics)
        
        self.log_metrics(metrics)
    
    def log_text_samples(self, samples: Dict[str, str], title: str = "text_samples"):
        """Log text samples for qualitative evaluation."""
        if self.config.use_wandb and self.wandb_run:
            # Create a W&B table for text samples
            table = wandb.Table(columns=["Type", "Text"])
            for sample_type, text in samples.items():
                table.add_data(sample_type, text)
            wandb.log({title: table}, step=self.step)
        
        if self.config.use_tensorboard and self.tb_writer:
            # Log as markdown text
            text_content = "\n\n".join([f"**{k}:**\n{v}" for k, v in samples.items()])
            self.tb_writer.add_text(title, text_content, self.step)
    
    def log_histogram(self, name: str, values: torch.Tensor):
        """Log histograms of model weights or gradients."""
        if self.config.use_wandb and self.wandb_run:
            wandb.log({name: wandb.Histogram(values.detach().cpu().numpy())}, step=self.step)
        
        if self.config.use_tensorboard and self.tb_writer:
            self.tb_writer.add_histogram(name, values, self.step)
    
    def log_gradients(self, model: torch.nn.Module):
        """Log gradient histograms for model parameters."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.log_histogram(f"gradients/{name}", param.grad)
                # Log gradient norms
                grad_norm = param.grad.norm().item()
                self.log_metrics({f"gradient_norms/{name}": grad_norm})
    
    def log_weights(self, model: torch.nn.Module):
        """Log weight histograms for model parameters."""
        for name, param in model.named_parameters():
            self.log_histogram(f"weights/{name}", param)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, metrics: Dict[str, float] = None):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "step": self.step,
            "experiment_config": asdict(self.config),
        }
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        # Save checkpoint
        checkpoint_path = Path(self.config.output_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Log artifact to W&B
        if self.config.use_wandb and self.wandb_run:
            artifact = wandb.Artifact(
                name=f"model_checkpoint_epoch_{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}"
            )
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
        
        return checkpoint_path
    
    def finish(self):
        """Clean up tracking resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.config.use_wandb and self.wandb_run:
            wandb.finish()


def create_experiment_tracker(experiment_name: str, config: Dict[str, Any] = None) -> ExperimentTracker:
    """Factory function to create experiment tracker with sensible defaults."""
    exp_config = ExperimentConfig(
        experiment_name=experiment_name,
        description=config.get("description", "") if config else "",
        tags=config.get("tags", []) if config else []
    )
    
    tracker = ExperimentTracker(exp_config)
    
    if config:
        tracker.log_config(config)
    
    return tracker


# Context manager for experiment tracking
class ExperimentContext:
    """Context manager for experiment tracking."""
    
    def __init__(self, experiment_name: str, config: Dict[str, Any] = None):
        self.experiment_name = experiment_name
        self.config = config
        self.tracker: Optional[ExperimentTracker] = None
    
    def __enter__(self) -> ExperimentTracker:
        self.tracker = create_experiment_tracker(self.experiment_name, self.config)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker:
            self.tracker.finish()


# Convenience functions
def log_training_step(tracker: ExperimentTracker, loss: float, 
                     learning_rate: float, step: int):
    """Log a single training step."""
    metrics = {
        "train/loss": loss,
        "train/learning_rate": learning_rate,
        "train/step": step
    }
    tracker.log_metrics(metrics, step=step)


def log_evaluation_results(tracker: ExperimentTracker, eval_results: Dict[str, float],
                          step: int):
    """Log evaluation results."""
    eval_metrics = {f"eval/{k}": v for k, v in eval_results.items()}
    tracker.log_metrics(eval_metrics, step=step)