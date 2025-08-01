#!/usr/bin/env python3
"""
TRANSFER LEARNING UTILITIES FOR AMHARIC H-NET
Support for loading Chinese H-Net weights and progressive fine-tuning

CRITICAL FEATURES:
- Compatible weight loading from Chinese H-Net repository
- Layer-wise progressive unfreezing strategy
- Architecture mapping between Chinese and Amharic models
- Fine-tuning optimization for cross-language transfer
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import warnings
from dataclasses import dataclass
import re

@dataclass
class TransferConfig:
    """Configuration for transfer learning from Chinese H-Net"""
    # Progressive unfreezing schedule
    freeze_embedding: bool = False       # Don't freeze embedding (different vocab)
    freeze_chunking_epochs: int = 3     # Freeze chunking layers for N epochs
    freeze_backbone_epochs: int = 5     # Freeze backbone layers for N epochs
    freeze_output_epochs: int = 2       # Freeze output layers for N epochs
    
    # Learning rate schedule for transfer
    base_lr: float = 2e-5              # Base learning rate
    finetune_lr_multiplier: float = 0.5 # LR multiplier for frozen layers
    warmup_epochs: int = 2             # Warmup epochs for transfer learning
    
    # Weight initialization
    init_std: float = 0.02             # Standard deviation for new weights
    layer_norm_eps: float = 1e-5       # Layer norm epsilon
    
    # Architecture compatibility
    strict_loading: bool = False        # Allow partial weight loading
    auto_resize_embeddings: bool = True # Auto-resize embedding layers

class ChineseHNetWeightLoader:
    """
    Utility to load and adapt Chinese H-Net weights for Amharic model
    """
    def __init__(self, config: TransferConfig):
        self.config = config
        self.layer_mapping = self._create_layer_mapping()
        self.loaded_layers = []
        self.skipped_layers = []
        self.adapted_layers = []
    
    def _create_layer_mapping(self) -> Dict[str, str]:
        """
        Create mapping between Chinese H-Net layer names and Amharic H-Net layer names
        Based on original H-Net architecture: https://github.com/goombalab/hnet
        """
        mapping = {
            # Embedding layers (will need adaptation for different vocab)
            'input_embedding.weight': 'input_embedding.weight',
            'pos_encoding': 'pos_encoding',
            
            # Dynamic chunker components
            'dynamic_chunker.boundary_detector.0.weight': 'dynamic_chunker.boundary_detector.0.0.weight',
            'dynamic_chunker.boundary_detector.0.bias': 'dynamic_chunker.boundary_detector.0.0.bias',
            'dynamic_chunker.boundary_detector.2.weight': 'dynamic_chunker.boundary_detector.0.2.weight',
            'dynamic_chunker.boundary_detector.2.bias': 'dynamic_chunker.boundary_detector.0.2.bias',
            'dynamic_chunker.boundary_detector.4.weight': 'dynamic_chunker.boundary_detector.0.4.weight',
            'dynamic_chunker.boundary_detector.4.bias': 'dynamic_chunker.boundary_detector.0.4.bias',
            
            # Chunk layer components
            'chunk_layer.chunk_attention.in_proj_weight': 'chunk_layer.chunk_attention.in_proj_weight',
            'chunk_layer.chunk_attention.in_proj_bias': 'chunk_layer.chunk_attention.in_proj_bias',
            'chunk_layer.chunk_attention.out_proj.weight': 'chunk_layer.chunk_attention.out_proj.weight',
            'chunk_layer.chunk_attention.out_proj.bias': 'chunk_layer.chunk_attention.out_proj.bias',
            
            # Hierarchical backbone (transformer layers)
            # These will be mapped dynamically based on layer count
            
            # DeChunk layer components
            'dechunk_layer.reconstruction_proj.weight': 'dechunk_layer.reconstruction_net.0.weight',
            'dechunk_layer.reconstruction_proj.bias': 'dechunk_layer.reconstruction_net.0.bias',
            
            # Language model head
            'lm_head.weight': 'lm_head.0.weight',  # Amharic model has larger head
            'lm_head.bias': 'lm_head.0.bias',
        }
        
        return mapping
    
    def load_chinese_weights(self, chinese_model_path: str) -> Dict[str, torch.Tensor]:
        """
        Load weights from Chinese H-Net checkpoint
        """
        if not os.path.exists(chinese_model_path):
            raise FileNotFoundError(f"Chinese H-Net weights not found: {chinese_model_path}")
        
        print(f"Loading Chinese H-Net weights from: {chinese_model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(chinese_model_path, map_location='cpu')
            
            # Extract state dict (handle different checkpoint formats)
            if 'model_state_dict' in checkpoint:
                chinese_state = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                chinese_state = checkpoint['state_dict']
            else:
                chinese_state = checkpoint
            
            print(f"Loaded {len(chinese_state)} layers from Chinese H-Net")
            
            # Print sample layer names for debugging
            sample_layers = list(chinese_state.keys())[:10]
            print("Sample Chinese layer names:")
            for layer in sample_layers:
                print(f"  {layer}: {chinese_state[layer].shape}")
            
            return chinese_state
            
        except Exception as e:
            print(f"Error loading Chinese weights: {e}")
            return {}
    
    def adapt_weights_for_amharic(
        self, 
        chinese_state: Dict[str, torch.Tensor], 
        amharic_model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt Chinese H-Net weights for Amharic model architecture
        """
        amharic_state = amharic_model.state_dict()
        adapted_state = {}
        
        print("Adapting Chinese weights for Amharic model...")
        
        # Process each layer in Amharic model
        for amharic_name, amharic_param in amharic_state.items():
            amharic_shape = amharic_param.shape
            
            # Try direct mapping first
            if amharic_name in chinese_state:
                chinese_param = chinese_state[amharic_name]
                if chinese_param.shape == amharic_shape:
                    adapted_state[amharic_name] = chinese_param
                    self.loaded_layers.append(amharic_name)
                    continue
            
            # Try mapped names
            mapped_name = self.layer_mapping.get(amharic_name)
            if mapped_name and mapped_name in chinese_state:
                chinese_param = chinese_state[mapped_name]
                if chinese_param.shape == amharic_shape:
                    adapted_state[amharic_name] = chinese_param
                    self.loaded_layers.append(amharic_name)
                    continue
            
            # Handle transformer layers with pattern matching
            if self._is_transformer_layer(amharic_name):
                adapted_param = self._adapt_transformer_layer(
                    amharic_name, amharic_shape, chinese_state
                )
                if adapted_param is not None:
                    adapted_state[amharic_name] = adapted_param
                    self.loaded_layers.append(amharic_name)
                    continue
            
            # Handle embedding layers with size adaptation
            if 'embedding' in amharic_name.lower():
                adapted_param = self._adapt_embedding_layer(
                    amharic_name, amharic_shape, chinese_state
                )
                if adapted_param is not None:
                    adapted_state[amharic_name] = adapted_param
                    self.adapted_layers.append(amharic_name)
                    continue
            
            # Handle output layers with size adaptation
            if 'lm_head' in amharic_name or 'output' in amharic_name:
                adapted_param = self._adapt_output_layer(
                    amharic_name, amharic_shape, chinese_state
                )
                if adapted_param is not None:
                    adapted_state[amharic_name] = adapted_param
                    self.adapted_layers.append(amharic_name)
                    continue
            
            # Skip layers that can't be adapted
            self.skipped_layers.append(amharic_name)
        
        # Fill remaining layers with original Amharic weights
        for name, param in amharic_state.items():
            if name not in adapted_state:
                adapted_state[name] = param
        
        self._print_transfer_summary()
        return adapted_state
    
    def _is_transformer_layer(self, layer_name: str) -> bool:
        """Check if layer is part of transformer backbone"""
        transformer_patterns = [
            r'hierarchical_backbone\.transformer_layers\.\d+',
            r'transformer_layers\.\d+',
            r'layers\.\d+',
            r'backbone\..*\.transformer'
        ]
        
        for pattern in transformer_patterns:
            if re.search(pattern, layer_name):
                return True
        return False
    
    def _adapt_transformer_layer(
        self, 
        amharic_name: str, 
        amharic_shape: Tuple[int, ...], 
        chinese_state: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Adapt transformer layer weights from Chinese to Amharic
        """
        # Extract layer index and component from name
        layer_match = re.search(r'(\d+)', amharic_name)
        if not layer_match:
            return None
        
        layer_idx = int(layer_match.group(1))
        
        # Try different Chinese naming patterns
        chinese_patterns = [
            amharic_name,  # Direct match
            amharic_name.replace('hierarchical_backbone.transformer_layers', 'transformer_layers'),
            amharic_name.replace('hierarchical_backbone.transformer_layers', 'layers'),
            amharic_name.replace('hierarchical_backbone.transformer_layers', 'backbone.layers'),
        ]
        
        for pattern in chinese_patterns:
            if pattern in chinese_state:
                chinese_param = chinese_state[pattern]
                if chinese_param.shape == amharic_shape:
                    return chinese_param
        
        return None
    
    def _adapt_embedding_layer(
        self, 
        amharic_name: str, 
        amharic_shape: Tuple[int, ...], 
        chinese_state: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Adapt embedding layers (vocab sizes may differ)
        """
        if not self.config.auto_resize_embeddings:
            return None
        
        # Find corresponding Chinese embedding
        chinese_embedding = None
        for chinese_name, chinese_param in chinese_state.items():
            if 'embedding' in chinese_name.lower() and 'weight' in chinese_name:
                chinese_embedding = chinese_param
                break
        
        if chinese_embedding is None:
            return None
        
        chinese_vocab_size, chinese_embed_dim = chinese_embedding.shape
        amharic_vocab_size, amharic_embed_dim = amharic_shape
        
        # If dimensions match, use directly
        if chinese_embedding.shape == amharic_shape:
            return chinese_embedding
        
        # If embedding dimension matches but vocab size differs
        if chinese_embed_dim == amharic_embed_dim:
            # Initialize new embedding with Chinese weights where possible
            new_embedding = torch.randn(amharic_shape) * self.config.init_std
            
            # Copy overlapping vocabulary (first N entries)
            min_vocab = min(chinese_vocab_size, amharic_vocab_size)
            new_embedding[:min_vocab] = chinese_embedding[:min_vocab]
            
            print(f"Adapted embedding: {chinese_embedding.shape} → {amharic_shape}")
            return new_embedding
        
        return None
    
    def _adapt_output_layer(
        self, 
        amharic_name: str, 
        amharic_shape: Tuple[int, ...], 
        chinese_state: Dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Adapt output layers (vocabulary sizes may differ)
        """
        # Find corresponding Chinese output layer
        chinese_patterns = ['lm_head', 'output', 'classifier']
        chinese_output = None
        
        for pattern in chinese_patterns:
            for chinese_name, chinese_param in chinese_state.items():
                if pattern in chinese_name.lower() and chinese_param.dim() == 2:
                    chinese_output = chinese_param
                    break
            if chinese_output is not None:
                break
        
        if chinese_output is None:
            return None
        
        # Initialize new output layer
        new_output = torch.randn(amharic_shape) * self.config.init_std
        
        # If compatible, copy partial weights
        if len(amharic_shape) == 2 and len(chinese_output.shape) == 2:
            chinese_hidden, chinese_vocab = chinese_output.shape
            amharic_hidden, amharic_vocab = amharic_shape
            
            if chinese_hidden == amharic_hidden:
                # Copy overlapping vocabulary outputs
                min_vocab = min(chinese_vocab, amharic_vocab)
                new_output[:, :min_vocab] = chinese_output[:, :min_vocab]
                
                print(f"Adapted output layer: {chinese_output.shape} → {amharic_shape}")
                return new_output
        
        return None
    
    def _print_transfer_summary(self):
        """Print summary of transfer learning"""
        print("\n" + "="*60)
        print("TRANSFER LEARNING SUMMARY")
        print("="*60)
        print(f"Loaded layers: {len(self.loaded_layers)}")
        print(f"Adapted layers: {len(self.adapted_layers)}")
        print(f"Skipped layers: {len(self.skipped_layers)}")
        
        if self.loaded_layers:
            print("\nDirect transfers:")
            for layer in self.loaded_layers[:10]:  # Show first 10
                print(f"  ✅ {layer}")
            if len(self.loaded_layers) > 10:
                print(f"  ... and {len(self.loaded_layers) - 10} more")
        
        if self.adapted_layers:
            print("\nAdapted layers:")
            for layer in self.adapted_layers:
                print(f"  🔄 {layer}")
        
        if self.skipped_layers:
            print("\nSkipped layers (random init):")
            for layer in self.skipped_layers[:5]:  # Show first 5
                print(f"  ❌ {layer}")
            if len(self.skipped_layers) > 5:
                print(f"  ... and {len(self.skipped_layers) - 5} more")
        
        print("="*60)

class ProgressiveUnfreezer:
    """
    Manages progressive unfreezing of model layers during fine-tuning
    """
    def __init__(self, model: nn.Module, config: TransferConfig):
        self.model = model
        self.config = config
        self.frozen_layers = []
        self.unfrozen_schedule = self._create_unfreezing_schedule()
        
    def _create_unfreezing_schedule(self) -> Dict[str, int]:
        """Create schedule for progressive unfreezing"""
        schedule = {}
        
        # Embedding layers (unfreeze early since vocab is different)
        if not self.config.freeze_embedding:
            for name, _ in self.model.named_parameters():
                if 'embedding' in name.lower():
                    schedule[name] = 0  # Unfreeze immediately
        
        # Chunking layers
        for name, _ in self.model.named_parameters():
            if 'chunker' in name.lower():
                schedule[name] = self.config.freeze_chunking_epochs
        
        # Backbone layers (progressive from bottom to top)
        backbone_layers = []
        for name, _ in self.model.named_parameters():
            if 'backbone' in name.lower() or 'transformer_layers' in name:
                layer_match = re.search(r'\.(\d+)\.', name)
                if layer_match:
                    layer_idx = int(layer_match.group(1))
                    backbone_layers.append((name, layer_idx))
        
        # Sort by layer index and assign progressive unfreezing
        backbone_layers.sort(key=lambda x: x[1])
        for i, (name, _) in enumerate(backbone_layers):
            # Gradually unfreeze backbone layers
            unfreeze_epoch = self.config.freeze_backbone_epochs + (i // 4)
            schedule[name] = unfreeze_epoch
        
        # Output layers
        for name, _ in self.model.named_parameters():
            if 'lm_head' in name or 'output' in name:
                schedule[name] = self.config.freeze_output_epochs
        
        return schedule
    
    def freeze_initial_layers(self):
        """Freeze layers according to initial schedule"""
        frozen_count = 0
        
        for name, param in self.model.named_parameters():
            freeze_until_epoch = self.unfrozen_schedule.get(name, 0)
            
            if freeze_until_epoch > 0:
                param.requires_grad = False
                self.frozen_layers.append(name)
                frozen_count += 1
            else:
                param.requires_grad = True
        
        print(f"Frozen {frozen_count} layers for progressive unfreezing")
        return frozen_count
    
    def unfreeze_layers_for_epoch(self, epoch: int) -> int:
        """Unfreeze layers scheduled for this epoch"""
        unfrozen_count = 0
        
        for name, param in self.model.named_parameters():
            freeze_until_epoch = self.unfrozen_schedule.get(name, 0)
            
            if freeze_until_epoch == epoch and not param.requires_grad:
                param.requires_grad = True
                if name in self.frozen_layers:
                    self.frozen_layers.remove(name)
                unfrozen_count += 1
        
        if unfrozen_count > 0:
            print(f"Unfroze {unfrozen_count} layers at epoch {epoch}")
        
        return unfrozen_count
    
    def get_frozen_layer_count(self) -> int:
        """Get current number of frozen layers"""
        return len(self.frozen_layers)
    
    def get_unfreezing_summary(self) -> Dict[str, Any]:
        """Get summary of unfreezing schedule"""
        schedule_summary = {}
        for epoch in range(max(self.unfrozen_schedule.values()) + 1):
            layers_to_unfreeze = [
                name for name, unfreeze_epoch in self.unfrozen_schedule.items()
                if unfreeze_epoch == epoch
            ]
            if layers_to_unfreeze:
                schedule_summary[f'epoch_{epoch}'] = len(layers_to_unfreeze)
        
        return {
            'total_scheduled_layers': len(self.unfrozen_schedule),
            'currently_frozen_layers': len(self.frozen_layers),
            'unfreezing_schedule': schedule_summary
        }

def load_chinese_hnet_for_amharic(
    amharic_model: nn.Module,
    chinese_weights_path: str,
    config: Optional[TransferConfig] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Main function to load Chinese H-Net weights into Amharic model
    """
    if config is None:
        config = TransferConfig()
    
    print("🔄 INITIALIZING TRANSFER LEARNING FROM CHINESE H-NET")
    print("="*70)
    
    # Load Chinese weights
    loader = ChineseHNetWeightLoader(config)
    chinese_state = loader.load_chinese_weights(chinese_weights_path)
    
    if not chinese_state:
        print("No Chinese weights loaded, using random initialization")
        return amharic_model, {'transfer_success': False}
    
    # Adapt weights for Amharic model
    adapted_state = loader.adapt_weights_for_amharic(chinese_state, amharic_model)
    
    # Load adapted weights
    try:
        amharic_model.load_state_dict(adapted_state, strict=config.strict_loading)
        print("✅ Successfully loaded adapted weights into Amharic model")
    except Exception as e:
        print(f"❌ Error loading adapted weights: {e}")
        return amharic_model, {'transfer_success': False, 'error': str(e)}
    
    # Setup progressive unfreezing
    unfreezer = ProgressiveUnfreezer(amharic_model, config)
    frozen_count = unfreezer.freeze_initial_layers()
    
    transfer_info = {
        'transfer_success': True,
        'loaded_layers': len(loader.loaded_layers),
        'adapted_layers': len(loader.adapted_layers),
        'skipped_layers': len(loader.skipped_layers),
        'initially_frozen_layers': frozen_count,
        'unfreezer': unfreezer,
        'unfreezing_schedule': unfreezer.get_unfreezing_summary()
    }
    
    print(f"🎯 Transfer learning setup complete!")
    print(f"   Loaded: {transfer_info['loaded_layers']} layers")
    print(f"   Adapted: {transfer_info['adapted_layers']} layers") 
    print(f"   Frozen: {transfer_info['initially_frozen_layers']} layers")
    
    return amharic_model, transfer_info

if __name__ == "__main__":
    print("🔄 TRANSFER LEARNING UTILITIES FOR AMHARIC H-NET")
    print("=" * 70)
    print("✅ Chinese H-Net weight loading")
    print("✅ Architecture adaptation")
    print("✅ Progressive unfreezing")
    print("✅ Cross-language transfer optimization")
    print("=" * 70)
    
    # Test configuration
    config = TransferConfig()
    print(f"\nDefault configuration:")
    print(f"  Freeze chunking epochs: {config.freeze_chunking_epochs}")
    print(f"  Freeze backbone epochs: {config.freeze_backbone_epochs}")
    print(f"  Auto resize embeddings: {config.auto_resize_embeddings}")
    print(f"  Base learning rate: {config.base_lr}")
    
    print("\n🎯 Transfer learning utilities ready!")
    print("Use load_chinese_hnet_for_amharic() to initialize transfer learning.")