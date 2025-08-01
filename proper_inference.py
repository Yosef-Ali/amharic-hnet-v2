#!/usr/bin/env python3
"""
Proper inference script for the retrained Amharic H-Net model.
Tests UTF-8 generation and model quality.
"""

import torch
import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.append('src')

from models.hnet_amharic import AmharicHNet
from preprocessing.prepare_amharic import AmharicPreprocessor

def load_model_from_checkpoint(checkpoint_path: str) -> tuple:
    """Load model from checkpoint with proper architecture."""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model configuration from checkpoint
    model_config = checkpoint.get('model_config', {
        'd_model': 384,
        'n_encoder_layers': 3,
        'n_decoder_layers': 3,
        'n_main_layers': 6,
        'n_heads': 6,
        'compression_ratio': 4.0
    })
    
    print(f"Model config: {model_config}")
    
    # Initialize model with saved configuration
    model = AmharicHNet(
        d_model=model_config['d_model'],
        n_encoder_layers=model_config['n_encoder_layers'],
        n_decoder_layers=model_config['n_decoder_layers'],
        n_main_layers=model_config['n_main_layers'],
        n_heads=model_config['n_heads'],
        compression_ratio=model_config['compression_ratio'],
        vocab_size=256  # Byte-level
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get training info
    epoch = checkpoint.get('epoch', 'unknown')
    best_loss = checkpoint.get('best_loss', 'unknown')
    training_losses = checkpoint.get('training_losses', [])
    validation_losses = checkpoint.get('validation_losses', [])
    
    print(f"Model trained for {epoch} epochs")
    print(f"Best validation loss: {best_loss}")
    
    if training_losses:
        print(f"Final training loss: {training_losses[-1]:.4f}")
    if validation_losses:
        print(f"Final validation loss: {validation_losses[-1]:.4f}")
    
    training_info = {
        'epoch': epoch,
        'best_loss': best_loss,
        'final_train_loss': training_losses[-1] if training_losses else None,
        'final_val_loss': validation_losses[-1] if validation_losses else None
    }
    
    return model, training_info

def test_model_generation(model: AmharicHNet, preprocessor: AmharicPreprocessor):
    """Test model generation capabilities."""
    
    print("\n" + "="*60)
    print("AMHARIC H-NET GENERATION TEST")
    print("="*60)
    
    # Test prompts in Amharic
    test_prompts = [
        "ኢትዮጵያ",
        "ቡና ቤት", 
        "አማርኛ ቋንቋ",
        "መስቀል በዓል",
        "አዲስ አበባ ከተማ",
        "ሰላም ነው",
        "እንዴት ነዎት",
        "ጤና ይስጥልኝ"
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            
            # Convert prompt to byte sequence
            try:
                byte_seq = preprocessor.extract_byte_sequences(prompt, max_length=100)
                input_ids = torch.tensor([byte_seq], dtype=torch.long).to(device)
                
                print(f"Input bytes: {len(byte_seq)} bytes")
                
                # Generate with different parameters
                for temp, top_k in [(0.7, 30), (0.9, 50), (1.1, 40)]:
                    try:
                        generated = model.generate(
                            input_ids, 
                            max_length=40,
                            temperature=temp,
                            top_k=top_k
                        )
                        
                        # Convert back to text
                        generated_bytes = generated[0].cpu().numpy().tolist()
                        generated_text = preprocessor.decode_byte_sequence(generated_bytes)
                        
                        print(f"  T={temp}, K={top_k}: {generated_text}")
                        
                        # Analysis
                        amharic_chars = sum(1 for c in generated_text if 0x1200 <= ord(c) <= 0x137F)
                        total_chars = len(generated_text)
                        amharic_ratio = amharic_chars / max(total_chars, 1)
                        
                        print(f"    Amharic ratio: {amharic_ratio:.2%} ({amharic_chars}/{total_chars})")
                        
                        # Check for control characters
                        control_chars = sum(1 for c in generated_text if ord(c) < 32 and c not in '\n\t\r')
                        if control_chars > 0:
                            print(f"    WARNING: {control_chars} control characters detected!")
                        else:
                            print(f"    ✓ No problematic control characters")
                        
                    except Exception as e:
                        print(f"    Generation error (T={temp}, K={top_k}): {e}")
                        
            except Exception as e:
                print(f"Error processing prompt '{prompt}': {e}")

def analyze_model_quality(model: AmharicHNet, training_info: dict):
    """Analyze overall model quality."""
    
    print("\n" + "="*60)
    print("MODEL QUALITY ANALYSIS")
    print("="*60)
    
    # Model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Training metrics
    if training_info['final_val_loss'] is not None:
        val_loss = training_info['final_val_loss']
        print(f"Final validation loss: {val_loss:.4f}")
        
        if val_loss < 2.0:
            print("✓ Loss target achieved (< 2.0)")
        else:
            print("⚠ Loss target not achieved (>= 2.0)")
    
    # Architecture details
    print(f"\nArchitecture:")
    print(f"  d_model: {model.d_model}")
    print(f"  Compression ratio: {model.compression_ratio}")
    print(f"  Encoder layers: {len(model.hierarchical_encoder.byte_encoder.layers)}")
    print(f"  Main layers: {len(model.main_encoder.layers)}")
    print(f"  Decoder layers: {len(model.decoder.layers)}")

def main():
    """Main inference testing function."""
    
    # Try to find the best checkpoint
    possible_paths = [
        "outputs/proper_training/checkpoint_best.pt",
        "outputs/proper_training/checkpoint_latest.pt",
        "outputs/checkpoint_best.pt",
        "outputs/checkpoint_latest.pt"
    ]
    
    checkpoint_path = None
    for path in possible_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print("ERROR: No trained model checkpoint found!")
        print("Please run proper_training.py first to train the model.")
        print(f"Looked for checkpoints in: {possible_paths}")
        return
    
    try:
        # Load model
        model, training_info = load_model_from_checkpoint(checkpoint_path)
        
        # Create preprocessor
        preprocessor = AmharicPreprocessor()
        
        # Analyze model quality
        analyze_model_quality(model, training_info)
        
        # Test generation
        test_model_generation(model, preprocessor)
        
        print("\n" + "="*60)
        print("INFERENCE TEST COMPLETED")
        print("="*60)
        
        # Summary
        if training_info['final_val_loss'] and training_info['final_val_loss'] < 2.0:
            print("✓ Model appears well-trained (loss < 2.0)")
        else:
            print("⚠ Model may need more training")
        
    except Exception as e:
        print(f"ERROR during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()