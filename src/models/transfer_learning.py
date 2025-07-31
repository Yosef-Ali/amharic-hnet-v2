#!/usr/bin/env python3
"""
Transfer learning from Chinese H-Net to Amharic H-Net.

This module implements sophisticated transfer learning strategies to leverage
the space-free text processing capabilities of Chinese H-Net for Amharic.

Key transfer learning strategies:
1. Weight Initialization - Transfer core H-Net weights from Chinese model
2. Architecture Adaptation - Modify layers for Amharic-specific requirements
3. Progressive Unfreezing - Gradual adaptation of transferred layers
4. Morphological Alignment - Align Chinese character patterns with Amharic syllables
5. Cultural Safety Integration - Ensure transferred knowledge respects Amharic culture
6. Multi-task Learning - Joint training on Chinese and Amharic for better generalization

Based on insights that Chinese and Amharic share space-free text challenges
but differ in script type (logographic vs syllabic) and morphological complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
import json
from dataclasses import dataclass
from collections import OrderedDict
import math


@dataclass
class TransferConfig:
    """Configuration for transfer learning parameters."""
    chinese_model_path: str
    target_compression_ratio: float = 4.5
    amharic_morpheme_length: float = 3.2
    chinese_character_length: float = 1.0
    
    # Transfer strategy parameters
    freeze_encoder: bool = True
    freeze_chunker: bool = False
    progressive_unfreezing: bool = True
    unfreezing_schedule: List[int] = None  # Epochs at which to unfreeze layers
    
    # Adaptation parameters
    adapt_chunking_params: bool = True
    add_morphological_layers: bool = True
    cultural_safety_integration: bool = True
    
    # Training parameters
    transfer_learning_rate: float = 1e-5
    fine_tuning_learning_rate: float = 1e-4
    warmup_epochs: int = 3
    
    def __post_init__(self):
        if self.unfreezing_schedule is None:
            self.unfreezing_schedule = [2, 5, 8, 12]


class ChineseToAmharicTransferLearner:
    """
    Manages transfer learning from Chinese H-Net to Amharic H-Net.
    
    Handles weight transfer, architecture adaptation, and progressive fine-tuning
    while preserving the core dynamic chunking capabilities learned from Chinese.
    """
    
    def __init__(self, config: TransferConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Script mapping strategies
        self.script_mappings = self._create_script_mappings()
        
        # Layer compatibility mapping
        self.layer_compatibility = self._define_layer_compatibility()
        
        # Morphological adaptation parameters
        self.morphological_adapters = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for transfer learning process."""
        logger = logging.getLogger('amharic_transfer_learning')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _create_script_mappings(self) -> Dict[str, any]:
        """
        Create mappings between Chinese characters and Amharic syllables.
        
        While Chinese uses logographic characters and Amharic uses syllabic script,
        both represent meaningful units that can be mapped for transfer learning.
        """
        return {
            'character_to_syllable_ratio': self.config.chinese_character_length / self.config.amharic_morpheme_length,
            'chunking_adaptation_factor': self.config.target_compression_ratio / 4.0,  # Assume Chinese uses 4.0
            'embedding_dimension_mapping': {
                'preserve_dimensions': True,  # Keep same embedding dimensions initially
                'add_cultural_dimensions': 64,  # Extra dimensions for cultural context
                'morphological_dimensions': 32  # Extra dimensions for morphological features
            },
            'attention_pattern_transfer': {
                'preserve_self_attention': True,
                'adapt_cross_attention': True,
                'add_morphological_attention': True
            }
        }
    
    def _define_layer_compatibility(self) -> Dict[str, str]:
        """Define which layers can be directly transferred vs need adaptation."""
        return {
            # Directly transferable layers
            'byte_encoder.layers': 'direct_transfer',
            'chunk_encoder.layers': 'direct_transfer', 
            'main_transformer.layers': 'direct_transfer',
            
            # Layers needing adaptation
            'chunker': 'adaptation_required',
            'morpheme_classifier': 'new_layer',
            'cultural_safety_head': 'new_layer',
            
            # Layers requiring parameter scaling
            'embedding_layers': 'parameter_scaling',
            'position_encodings': 'parameter_scaling',
            'output_heads': 'architecture_change'
        }
    
    def load_chinese_hnet(self, model_path: str) -> Dict[str, torch.Tensor]:
        """
        Load pretrained Chinese H-Net weights.
        
        Args:
            model_path: Path to Chinese H-Net checkpoint
            
        Returns:
            Dictionary of loaded weights
        """
        self.logger.info(f"Loading Chinese H-Net from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                weights = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                weights = checkpoint['state_dict']
            else:
                weights = checkpoint
            
            self.logger.info(f"Loaded {len(weights)} parameter tensors from Chinese H-Net")
            
            # Log model architecture information
            self._log_model_architecture(weights)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Failed to load Chinese H-Net: {e}")
            raise
    
    def _log_model_architecture(self, weights: Dict[str, torch.Tensor]):
        """Log information about the loaded Chinese model architecture."""
        layer_counts = {}
        total_params = 0
        
        for name, tensor in weights.items():
            layer_type = name.split('.')[0]
            layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
            total_params += tensor.numel()
        
        self.logger.info(f"Chinese H-Net architecture summary:")
        for layer_type, count in layer_counts.items():
            self.logger.info(f"  {layer_type}: {count} parameters")
        self.logger.info(f"Total parameters: {total_params:,}")
    
    def adapt_weights_for_amharic(
        self, 
        chinese_weights: Dict[str, torch.Tensor],
        amharic_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt Chinese H-Net weights for Amharic model architecture.
        
        Args:
            chinese_weights: Loaded Chinese model weights
            amharic_model: Target Amharic model
            
        Returns:
            Adapted weights for Amharic model
        """
        self.logger.info("Adapting Chinese weights for Amharic model...")
        
        adapted_weights = {}
        amharic_state_dict = amharic_model.state_dict()
        
        # Process each parameter in the Amharic model
        for amharic_param_name, amharic_tensor in amharic_state_dict.items():
            adapted_weights[amharic_param_name] = self._adapt_single_parameter(
                amharic_param_name,
                amharic_tensor,
                chinese_weights
            )
        
        # Add morphological adaptation layers
        if self.config.add_morphological_layers:
            adapted_weights.update(self._initialize_morphological_layers(amharic_model))
        
        self.logger.info(f"Adapted {len(adapted_weights)} parameters for Amharic model")
        
        return adapted_weights
    
    def _adapt_single_parameter(
        self,
        param_name: str,
        target_tensor: torch.Tensor,
        source_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Adapt a single parameter from Chinese to Amharic model.
        
        Args:
            param_name: Name of the parameter in Amharic model
            target_tensor: Target tensor shape and type
            source_weights: Source Chinese model weights
            
        Returns:
            Adapted parameter tensor
        """
        # Try direct mapping first
        if param_name in source_weights:
            source_tensor = source_weights[param_name]
            if source_tensor.shape == target_tensor.shape:
                self.logger.debug(f"Direct transfer: {param_name}")
                return source_tensor.clone()
        
        # Try pattern matching for similar layers
        adapted_tensor = self._find_compatible_weights(param_name, target_tensor, source_weights)
        if adapted_tensor is not None:
            return adapted_tensor
        
        # Check if this is a chunker-related parameter that needs adaptation
        if 'chunker' in param_name or 'morpheme' in param_name:
            return self._adapt_chunking_parameter(param_name, target_tensor, source_weights)
        
        # Check if this is an embedding parameter that needs scaling
        if 'embedding' in param_name:
            return self._adapt_embedding_parameter(param_name, target_tensor, source_weights)
        
        # Initialize new parameter (for Amharic-specific layers)
        self.logger.debug(f"New parameter initialization: {param_name}")
        return self._initialize_new_parameter(target_tensor)
    
    def _find_compatible_weights(
        self,
        param_name: str,
        target_tensor: torch.Tensor,
        source_weights: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Find compatible weights using pattern matching and layer similarity."""
        # Extract layer type and sublayer information
        name_parts = param_name.split('.')
        
        # Look for similar layer names
        for source_name, source_tensor in source_weights.items():
            source_parts = source_name.split('.')
            
            # Check for similar layer structure
            if len(name_parts) >= 2 and len(source_parts) >= 2:
                if (name_parts[0] == source_parts[0] and  # Same major component
                    name_parts[-1] == source_parts[-1] and  # Same parameter type (weight/bias)
                    source_tensor.shape == target_tensor.shape):
                    
                    self.logger.debug(f"Compatible transfer: {source_name} -> {param_name}")
                    return source_tensor.clone()
        
        return None
    
    def _adapt_chunking_parameter(
        self,
        param_name: str,
        target_tensor: torch.Tensor,
        source_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Adapt chunking-related parameters to account for morphological differences.
        
        Chinese characters vs Amharic syllables require different chunking strategies.
        """
        # Look for similar chunking parameters in Chinese model
        chunking_params = {k: v for k, v in source_weights.items() if 'chunk' in k.lower()}
        
        if chunking_params:
            # Use the most similar chunking parameter as initialization
            similar_param = next(iter(chunking_params.values()))
            
            if similar_param.shape == target_tensor.shape:
                # Apply morphological adaptation scaling
                adaptation_factor = self.config.amharic_morpheme_length / self.config.chinese_character_length
                adapted_param = similar_param * adaptation_factor
                
                self.logger.debug(f"Chunking adaptation: {param_name} (factor: {adaptation_factor:.2f})")
                return adapted_param
        
        # If no compatible chunking parameter found, initialize with scaled random weights
        return self._initialize_new_parameter(target_tensor) * 0.1  # Smaller initial weights for chunking
    
    def _adapt_embedding_parameter(
        self,
        param_name: str,
        target_tensor: torch.Tensor,
        source_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Adapt embedding parameters to handle script differences.
        
        Chinese logographic embeddings need adaptation for Amharic syllabic script.
        """
        # Look for similar embedding parameters
        embedding_params = {k: v for k, v in source_weights.items() if 'embed' in k.lower()}
        
        for source_name, source_tensor in embedding_params.items():
            if source_tensor.shape == target_tensor.shape:
                # Apply script adaptation transformation
                adapted_embedding = self._apply_script_transformation(source_tensor)
                self.logger.debug(f"Embedding adaptation: {source_name} -> {param_name}")
                return adapted_embedding
        
        # Initialize new embedding if no compatible one found
        return self._initialize_new_parameter(target_tensor)
    
    def _apply_script_transformation(self, embedding_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation to adapt embeddings from Chinese script to Amharic.
        
        Uses learned orthogonal transformation to preserve semantic relationships
        while adapting to syllabic structure.
        """
        # Simple orthogonal transformation - in practice would be learned
        embedding_dim = embedding_tensor.shape[-1]
        
        # Create orthogonal transformation matrix
        transform_matrix = torch.randn(embedding_dim, embedding_dim)
        Q, _ = torch.qr(transform_matrix)
        
        # Apply transformation while preserving norms
        if len(embedding_tensor.shape) == 2:  # Standard embedding matrix
            transformed = torch.matmul(embedding_tensor, Q)
        else:
            # Handle other tensor shapes
            original_shape = embedding_tensor.shape
            flattened = embedding_tensor.view(-1, embedding_dim)
            transformed = torch.matmul(flattened, Q)
            transformed = transformed.view(original_shape)
        
        return transformed
    
    def _initialize_morphological_layers(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Initialize new Amharic-specific morphological layers."""
        morphological_params = {}
        
        # Add morphological feature embeddings
        if hasattr(model, 'morphological_features'):
            morphological_params.update(model.morphological_features.state_dict())
        
        # Add cultural safety parameters
        if hasattr(model, 'cultural_safety_head'):
            morphological_params.update(model.cultural_safety_head.state_dict())
        
        self.logger.info(f"Initialized {len(morphological_params)} morphological parameters")
        return morphological_params
    
    def _initialize_new_parameter(self, target_tensor: torch.Tensor) -> torch.Tensor:
        """Initialize new parameter using appropriate initialization strategy."""
        if len(target_tensor.shape) == 1:  # Bias
            return torch.zeros_like(target_tensor)
        elif len(target_tensor.shape) == 2:  # Linear layer weight
            return torch.randn_like(target_tensor) * math.sqrt(2.0 / target_tensor.shape[0])
        else:  # Other parameter types
            return torch.randn_like(target_tensor) * 0.02
    
    def create_progressive_training_schedule(
        self, 
        amharic_model: nn.Module,
        total_epochs: int
    ) -> Dict[int, List[str]]:
        """
        Create progressive unfreezing schedule for transfer learning.
        
        Args:
            amharic_model: Target Amharic model
            total_epochs: Total training epochs
            
        Returns:
            Dictionary mapping epochs to layers to unfreeze
        """
        schedule = {}
        
        if not self.config.progressive_unfreezing:
            # Unfreeze everything at epoch 0
            all_params = [name for name, _ in amharic_model.named_parameters()]
            schedule[0] = all_params
            return schedule
        
        # Progressive unfreezing schedule
        layer_groups = self._group_layers_by_importance(amharic_model)
        
        for epoch, group_name in zip(self.config.unfreezing_schedule, layer_groups.keys()):
            if epoch < total_epochs:
                schedule[epoch] = layer_groups[group_name]
        
        return schedule
    
    def _group_layers_by_importance(self, model: nn.Module) -> Dict[str, List[str]]:
        """Group model layers by training importance for progressive unfreezing."""
        layer_groups = {
            'morphological_layers': [],
            'chunking_layers': [],
            'attention_layers': [],
            'encoder_layers': [],
            'all_layers': []
        }
        
        for name, _ in model.named_parameters():
            layer_groups['all_layers'].append(name)
            
            if any(keyword in name.lower() for keyword in ['morpheme', 'cultural', 'amharic']):
                layer_groups['morphological_layers'].append(name)
            elif 'chunk' in name.lower():
                layer_groups['chunking_layers'].append(name)
            elif 'attention' in name.lower() or 'attn' in name.lower():
                layer_groups['attention_layers'].append(name)
            elif 'encoder' in name.lower():
                layer_groups['encoder_layers'].append(name)
        
        return layer_groups
    
    def apply_transfer_learning(
        self,
        amharic_model: nn.Module,
        chinese_model_path: str
    ) -> Tuple[nn.Module, Dict[str, any]]:
        """
        Complete transfer learning pipeline from Chinese to Amharic H-Net.
        
        Args:
            amharic_model: Target Amharic model
            chinese_model_path: Path to pretrained Chinese model
            
        Returns:
            Tuple of (adapted_model, transfer_info)
        """
        self.logger.info("Starting Chinese to Amharic H-Net transfer learning...")
        
        # Load Chinese model weights
        chinese_weights = self.load_chinese_hnet(chinese_model_path)
        
        # Adapt weights for Amharic architecture
        adapted_weights = self.adapt_weights_for_amharic(chinese_weights, amharic_model)
        
        # Load adapted weights into Amharic model
        amharic_model.load_state_dict(adapted_weights, strict=False)
        
        # Freeze appropriate layers based on configuration
        if self.config.freeze_encoder:
            self._freeze_layers(amharic_model, ['encoder', 'transformer'])
        
        if self.config.freeze_chunker:
            self._freeze_layers(amharic_model, ['chunker'])
        
        # Prepare transfer learning information
        transfer_info = {
            'source_model': chinese_model_path,
            'adapted_parameters': len(adapted_weights),
            'frozen_layers': self._get_frozen_layers(amharic_model),
            'unfreezing_schedule': self.create_progressive_training_schedule(amharic_model, 20),
            'morphological_adaptations': len(self.morphological_adapters),
        }
        
        self.logger.info("Transfer learning completed successfully!")
        self.logger.info(f"Adapted {transfer_info['adapted_parameters']} parameters")
        self.logger.info(f"Frozen {len(transfer_info['frozen_layers'])} layer groups")
        
        return amharic_model, transfer_info
    
    def _freeze_layers(self, model: nn.Module, layer_keywords: List[str]):
        """Freeze layers containing specified keywords."""
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(keyword in name.lower() for keyword in layer_keywords):
                param.requires_grad = False
                frozen_count += 1
        
        self.logger.info(f"Frozen {frozen_count} parameters in layers: {layer_keywords}")
    
    def _get_frozen_layers(self, model: nn.Module) -> List[str]:
        """Get list of frozen layer names."""
        frozen_layers = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_layers.append(name)
        return frozen_layers
    
    def create_joint_training_dataset(
        self,
        chinese_texts: List[str],
        amharic_texts: List[str],
        mixing_ratio: float = 0.3
    ) -> List[Dict[str, any]]:
        """
        Create joint Chinese-Amharic dataset for multi-task learning.
        
        Args:
            chinese_texts: Chinese text samples
            amharic_texts: Amharic text samples  
            mixing_ratio: Ratio of Chinese to Amharic samples
            
        Returns:
            Mixed dataset with language labels
        """
        joint_dataset = []
        
        # Add Chinese samples
        chinese_count = int(len(amharic_texts) * mixing_ratio)
        for i, text in enumerate(chinese_texts[:chinese_count]):
            joint_dataset.append({
                'text': text,
                'language': 'chinese',
                'script_type': 'logographic',
                'has_spaces': False,
                'sample_id': f'zh_{i}'
            })
        
        # Add Amharic samples
        for i, text in enumerate(amharic_texts):
            joint_dataset.append({
                'text': text,
                'language': 'amharic',
                'script_type': 'syllabic',
                'has_spaces': False,
                'sample_id': f'am_{i}'
            })
        
        # Shuffle the dataset
        np.random.shuffle(joint_dataset)
        
        self.logger.info(f"Created joint dataset: {chinese_count} Chinese + {len(amharic_texts)} Amharic samples")
        
        return joint_dataset


class TransferLearningTrainer:
    """
    Specialized trainer for transfer learning from Chinese to Amharic H-Net.
    """
    
    def __init__(self, model: nn.Module, config: TransferConfig):
        self.model = model
        self.config = config
        self.logger = logging.getLogger('transfer_trainer')
        
        # Setup optimizers with different learning rates for different layer groups
        self.optimizers = self._setup_optimizers()
        
    def _setup_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Setup different optimizers for transferred vs new layers."""
        optimizers = {}
        
        # Parameters for transferred layers (lower learning rate)
        transferred_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(keyword in name.lower() for keyword in ['morphological', 'cultural', 'amharic']):
                    new_params.append(param)
                else:
                    transferred_params.append(param)
        
        if transferred_params:
            optimizers['transferred'] = torch.optim.AdamW(
                transferred_params,
                lr=self.config.transfer_learning_rate,
                weight_decay=0.01
            )
        
        if new_params:
            optimizers['new'] = torch.optim.AdamW(
                new_params,
                lr=self.config.fine_tuning_learning_rate,
                weight_decay=0.01
            )
        
        return optimizers
    
    def train_with_progressive_unfreezing(
        self,
        train_dataloader,
        val_dataloader,
        num_epochs: int,
        unfreezing_schedule: Dict[int, List[str]]
    ):
        """
        Train model with progressive unfreezing schedule.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            num_epochs: Number of training epochs
            unfreezing_schedule: Schedule for unfreezing layers
        """
        self.logger.info("Starting transfer learning training with progressive unfreezing...")
        
        for epoch in range(num_epochs):
            # Check if we need to unfreeze layers at this epoch
            if epoch in unfreezing_schedule:
                self._unfreeze_layers(unfreezing_schedule[epoch])
                # Update optimizers after unfreezing
                self.optimizers = self._setup_optimizers()
            
            # Training phase
            train_loss = self._train_epoch(train_dataloader)
            
            # Validation phase
            val_loss = self._validate_epoch(val_dataloader)
            
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
        
        self.logger.info("Transfer learning training completed!")
    
    def _unfreeze_layers(self, layer_names: List[str]):
        """Unfreeze specified layers."""
        unfrozen_count = 0
        for name, param in self.model.named_parameters():
            if name in layer_names and not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        
        self.logger.info(f"Unfroze {unfrozen_count} parameters")
    
    def _train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            # Forward pass
            outputs = self.model(batch['input_ids'])
            loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            
            loss.backward()
            
            for optimizer in self.optimizers.values():
                optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _validate_epoch(self, dataloader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(batch['input_ids'])
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def _compute_loss(self, outputs, batch) -> torch.Tensor:
        """Compute loss with potential multi-task components."""
        # Standard language modeling loss
        lm_loss = F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            batch['labels'].view(-1),
            ignore_index=-100
        )
        
        # Add cultural safety loss if available
        total_loss = lm_loss
        
        if hasattr(outputs, 'cultural_safety_loss'):
            total_loss += 0.1 * outputs.cultural_safety_loss
        
        return total_loss


def load_chinese_hnet_for_transfer(
    chinese_model_path: str,
    amharic_model_class,
    model_config: Dict,
    transfer_config: TransferConfig = None
) -> Tuple[nn.Module, Dict[str, any]]:
    """
    Convenience function to load Chinese H-Net and transfer to Amharic.
    
    Args:
        chinese_model_path: Path to Chinese H-Net checkpoint
        amharic_model_class: Amharic H-Net model class
        model_config: Configuration for Amharic model
        transfer_config: Transfer learning configuration
        
    Returns:
        Tuple of (transferred_model, transfer_info)
    """
    if transfer_config is None:
        transfer_config = TransferConfig(chinese_model_path=chinese_model_path)
    
    # Initialize transfer learner
    transfer_learner = ChineseToAmharicTransferLearner(transfer_config)
    
    # Create Amharic model
    amharic_model = amharic_model_class(**model_config)
    
    # Apply transfer learning
    transferred_model, transfer_info = transfer_learner.apply_transfer_learning(
        amharic_model, chinese_model_path
    )
    
    return transferred_model, transfer_info


if __name__ == "__main__":
    # Example usage
    from ..models.hnet_amharic import AmharicHNet
    import tempfile
    
    # Mock Chinese model for demonstration
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        # Create mock Chinese weights
        mock_weights = {
            'encoder.layer.0.weight': torch.randn(768, 256),
            'encoder.layer.0.bias': torch.randn(768),
            'chunker.classifier.weight': torch.randn(1, 768),
            'chunker.classifier.bias': torch.randn(1),
        }
        torch.save(mock_weights, tmp_file.name)
        chinese_model_path = tmp_file.name
    
    # Transfer learning example
    config = TransferConfig(
        chinese_model_path=chinese_model_path,
        progressive_unfreezing=True,
        add_morphological_layers=True
    )
    
    model_config = {
        'd_model': 768,
        'n_heads': 12,
        'n_encoder_layers': 6,
        'n_decoder_layers': 6,
        'n_main_layers': 12,
        'compression_ratio': 4.5
    }
    
    try:
        transferred_model, transfer_info = load_chinese_hnet_for_transfer(
            chinese_model_path,
            AmharicHNet,
            model_config,
            config
        )
        
        print("Transfer learning successful!")
        print(f"Transfer info: {transfer_info}")
        
    except Exception as e:
        print(f"Transfer learning failed: {e}")
    
    # Clean up temporary file
    import os
    os.unlink(chinese_model_path)