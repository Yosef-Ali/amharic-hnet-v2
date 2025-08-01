#!/usr/bin/env python3
"""
Direct inference test for Amharic H-Net without main.py dependencies
"""

import torch
import sys
import os
sys.path.append('src')

from models.hnet_amharic import AmharicHNet

def test_model_inference():
    """Test actual model performance with real inference"""
    
    # Load the trained model
    print("🔄 Loading trained Amharic H-Net model...")
    
    # Initialize model architecture 
    model = AmharicHNet(
        d_model=768,
        n_encoder_layers=4,
        n_decoder_layers=4,
        n_main_layers=12,
        n_heads=12,
        vocab_size=256  # Byte-level
    )
    
    try:
        # Load checkpoint
        checkpoint_path = "outputs/checkpoint_epoch_2.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model from {checkpoint_path}")
                
                # Get training info
                epoch = checkpoint.get('epoch', 'unknown')
                loss = checkpoint.get('loss', 'unknown')
                print(f"📊 Model trained for {epoch} epochs, final loss: {loss}")
            else:
                print("❌ No model_state_dict found in checkpoint")
                return
        else:
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return
            
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    print("\n🎯 Testing Amharic H-Net Performance")
    print("=" * 50)
    
    # Test prompts in Amharic
    test_prompts = [
        "ኢትዮጵያ",
        "ቡና", 
        "አማርኛ",
        "መስቀል",
        "አዲስ አበባ"
    ]
    
    with torch.no_grad():
        for prompt in test_prompts:
            print(f"\n📝 Prompt: {prompt}")
            
            # Convert prompt to byte sequence
            prompt_bytes = prompt.encode('utf-8')
            input_ids = torch.tensor([[b for b in prompt_bytes]], dtype=torch.long)
            
            try:
                # Generate with the model
                generated = model.generate(
                    input_ids, 
                    max_length=20,  # Shorter for testing
                    temperature=0.8,
                    top_k=50
                )
                
                # Convert back to text
                generated_bytes = generated[0].cpu().numpy()
                try:
                    generated_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
                    print(f"🤖 Generated: {generated_text}")
                    
                    # Analysis
                    is_amharic = any(ord(c) >= 0x1200 and ord(c) <= 0x137F for c in generated_text)
                    print(f"📊 Contains Amharic: {'✅' if is_amharic else '❌'}")
                    print(f"📏 Length: {len(generated_text)} chars")
                    
                except UnicodeDecodeError as e:
                    print(f"⚠️ Decode error: {e}")
                    print(f"🔢 Raw bytes: {generated_bytes[:20]}...")
                    
            except Exception as e:
                print(f"❌ Generation error: {e}")
    
    # Test model architecture
    print(f"\n🏗️ Model Architecture:")
    print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • Model size: {sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024:.1f} MB")
    print(f"   • Embedding dim: {model.d_model}")
    print(f"   • Compression ratio: {model.compression_ratio}")
    
    # Test chunker
    print(f"\n🔍 Testing Amharic Morpheme Chunker:")
    test_input = torch.randn(1, 10, 768)  # Sample input
    try:
        boundary_probs = model.chunker(test_input)
        print(f"   • Boundary detection: ✅ Shape {boundary_probs.shape}")
        print(f"   • Average boundary prob: {boundary_probs.mean():.3f}")
    except Exception as e:
        print(f"   • Chunker error: {e}")

if __name__ == "__main__":
    test_model_inference()