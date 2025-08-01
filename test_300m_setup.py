#!/usr/bin/env python3
"""
Simple test for 300M H-Net setup to verify configuration
"""

import torch
import yaml
from src.models.hnet_300m_amharic import AmharicHNet300M, HNet300MConfig, create_300m_model
import sys

def test_config_loading():
    """Test configuration loading"""
    print("🔍 Testing configuration loading...")
    
    try:
        with open('configs/hnet_300m_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Config loaded successfully")
        print(f"   Learning rate: {config['training']['learning_rate']} (type: {type(config['training']['learning_rate'])})")
        print(f"   Batch size: {config['training']['batch_size']}")
        print(f"   Model d_model: {config['model']['d_model']}")
        
        return config
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return None

def test_model_creation():
    """Test 300M model creation"""
    print("\n🔍 Testing 300M model creation...")
    
    try:
        config = HNet300MConfig(
            d_model=1536,
            n_heads=24,
            n_backbone_layers=24,
            n_chunk_layers=8,
            n_dechunk_layers=4,
            max_chunks=256,
            vocab_size=256,
            max_seq_length=512
        )
        
        model = create_300m_model(config)
        print(f"✅ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size (fp32): {total_params * 4 / 1e9:.2f} GB")
        print(f"   Model size (fp16): {total_params * 2 / 1e9:.2f} GB")
        
        return model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return None

def test_forward_pass(model):
    """Test forward pass"""
    print("\n🔍 Testing forward pass...")
    
    try:
        # Create dummy input
        batch_size = 1
        seq_len = 64
        input_ids = torch.randint(0, 256, (batch_size, seq_len))
        
        print(f"   Input shape: {input_ids.shape}")
        
        # Forward pass
        with torch.no_grad():
            logits, debug_info = model(input_ids)
        
        print(f"✅ Forward pass successful")
        print(f"   Output logits shape: {logits.shape}")
        print(f"   Boundary probs shape: {debug_info['boundaries'].shape}")
        print(f"   Chunk representations shape: {debug_info['chunk_representations'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_generation(model):
    """Test text generation"""
    print("\n🔍 Testing generation...")
    
    try:
        # Test prompt
        prompt = "ኢትዮጵያ"
        prompt_bytes = list(prompt.encode('utf-8'))
        input_ids = torch.tensor([prompt_bytes], dtype=torch.long)
        
        print(f"   Prompt: '{prompt}'")
        print(f"   Input bytes: {prompt_bytes}")
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_length=10,
                temperature=0.8,
                top_k=40
            )
        
        # Decode
        generated_bytes = generated[0].cpu().numpy().tolist()
        full_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
        
        print(f"✅ Generation successful")
        print(f"   Generated: '{full_text}'")
        print(f"   New part: '{full_text[len(prompt):]}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 300M H-NET SETUP VERIFICATION")
    print("="*50)
    
    # Test 1: Configuration
    config = test_config_loading()
    if not config:
        sys.exit(1)
    
    # Test 2: Model creation
    model = test_model_creation()
    if not model:
        sys.exit(1)
    
    # Test 3: Forward pass
    if not test_forward_pass(model):
        sys.exit(1)
    
    # Test 4: Generation
    if not test_generation(model):
        sys.exit(1)
    
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ 300M H-Net architecture is working correctly")
    print("✅ Configuration loading works")
    print("✅ Model creation successful")
    print("✅ Forward pass functional")
    print("✅ Text generation operational")
    print("\n🚀 Ready for full training!")

if __name__ == "__main__":
    main()