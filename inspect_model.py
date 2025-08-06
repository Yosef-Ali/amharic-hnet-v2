#!/usr/bin/env python3
"""
Inspect the downloaded Kaggle model to understand its architecture
"""

import torch
from pathlib import Path

def inspect_kaggle_model():
    """Inspect the downloaded model structure."""
    model_path = "kaggle_gpu_production/best_model.pt"
    
    print("🔍 INSPECTING KAGGLE MODEL STRUCTURE")
    print("=" * 50)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"📋 Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"📋 Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    print(f"\n🏗️  MODEL ARCHITECTURE ANALYSIS:")
    print(f"=" * 40)
    
    # Analyze model structure
    embedding_weight = None
    transformer_layers = 0
    d_model = None
    vocab_size = None
    
    for name, param in state_dict.items():
        print(f"📊 {name}: {param.shape}")
        
        # Extract key architecture info
        if 'embedding.weight' in name and not 'pos_embedding' in name:
            vocab_size, d_model = param.shape
            embedding_weight = param
        elif 'layers.' in name and 'weight' in name:
            if 'layers.0.' in name:
                pass  # First layer
            # Count unique layer numbers
            layer_num = int(name.split('.')[1])
            transformer_layers = max(transformer_layers, layer_num + 1)
    
    print(f"\n🎯 DETECTED ARCHITECTURE:")
    print(f"   • Vocabulary Size: {vocab_size:,}")
    print(f"   • Model Dimension: {d_model}")
    print(f"   • Transformer Layers: {transformer_layers}")
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"   • Total Parameters: {total_params:,}")
    
    return {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_layers': transformer_layers,
        'total_params': total_params,
        'state_dict': state_dict
    }

if __name__ == "__main__":
    model_info = inspect_kaggle_model()