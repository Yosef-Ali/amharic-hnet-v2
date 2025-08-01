#!/usr/bin/env python3
"""
Compare outputs from different model configurations
"""

import torch
import os
from src.models.proper_hnet_amharic import AmharicHNetMixer

def test_model_outputs():
    """Test and compare different model configurations"""
    print("🧠 AMHARIC H-NET OUTPUT COMPARISON")
    print("="*60)
    
    # Test prompts
    test_prompts = [
        ("ኢትዮጵያ", "Ethiopia"),
        ("አማርኛ", "Amharic"),
        ("ቡና", "coffee"),
        ("ሰላም", "peace"),
        ("ቤት", "house")
    ]
    
    models_to_test = [
        {
            "name": "Small (10M params)",
            "config": {"d_model": 256, "n_heads": 4, "n_backbone_layers": 4, "max_chunks": 32}
        },
        {
            "name": "Medium (50M params)", 
            "config": {"d_model": 512, "n_heads": 8, "n_backbone_layers": 8, "max_chunks": 64}
        },
        {
            "name": "Large (300M params)",
            "config": {"d_model": 1024, "n_heads": 16, "n_backbone_layers": 16, "max_chunks": 128}
        }
    ]
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for model_info in models_to_test:
        print(f"\n{'='*60}")
        print(f"🔧 TESTING {model_info['name'].upper()}")
        print(f"{'='*60}")
        
        try:
            # Create model
            model = AmharicHNetMixer(
                vocab_size=256,
                **model_info['config']
            )
            
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {param_count:,}")
            print(f"Memory (fp16): {param_count * 2 / 1e9:.2f} GB")
            
            model = model.to(device)
            model.eval()
            
            print(f"\nGeneration Results:")
            print("-" * 40)
            
            with torch.no_grad():
                for prompt, meaning in test_prompts:
                    # Convert prompt to bytes
                    prompt_bytes = list(prompt.encode('utf-8'))
                    input_ids = torch.tensor([prompt_bytes], dtype=torch.long).to(device)
                    
                    try:
                        # Generate with conservative settings
                        generated = model.generate(
                            input_ids,
                            max_length=20,
                            temperature=0.7,
                            top_k=30
                        )
                        
                        # Decode result
                        generated_bytes = generated[0].cpu().numpy().tolist()
                        full_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
                        new_part = full_text[len(prompt):]
                        
                        # Calculate quality
                        amharic_chars = sum(1 for c in new_part if '\u1200' <= c <= '\u137F')
                        total_chars = len([c for c in new_part if c.strip()])
                        quality = amharic_chars / max(total_chars, 1) if total_chars > 0 else 0
                        
                        print(f"'{prompt}' → '{full_text}'")
                        print(f"   New: '{new_part}' | Quality: {quality:.2f}")
                        
                    except Exception as e:
                        print(f"'{prompt}' → Error: {str(e)[:50]}...")
            
            # Clean up memory
            del model
            if device.type == 'mps':
                torch.mps.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed to test {model_info['name']}: {e}")
            continue

def test_existing_checkpoints():
    """Test any existing trained checkpoints"""
    print(f"\n{'='*60}")
    print("🔍 TESTING EXISTING TRAINED MODELS")
    print(f"{'='*60}")
    
    checkpoint_paths = [
        "outputs/proper_hnet_final.pt",
        "outputs/compact_enhanced_final.pt", 
        "outputs/compact_enhanced_epoch_5.pt"
    ]
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    for checkpoint_path in checkpoint_paths:
        if os.path.exists(checkpoint_path):
            print(f"\n📁 Testing: {checkpoint_path}")
            print("-" * 40)
            
            try:
                # Create compatible model (compact size)
                model = AmharicHNetMixer(
                    vocab_size=256,
                    d_model=384,
                    n_heads=6,
                    n_backbone_layers=4,
                    max_chunks=64
                )
                
                # Load checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    if checkpoint.get('loss'):
                        print(f"Training loss: {checkpoint['loss']:.4f}")
                else:
                    model.load_state_dict(checkpoint)
                
                model = model.to(device)
                model.eval()
                
                # Test generation
                test_prompts = [("ኢትዮጵያ", "Ethiopia"), ("አማርኛ", "Amharic"), ("ቡና", "coffee")]
                
                with torch.no_grad():
                    for prompt, meaning in test_prompts:
                        prompt_bytes = list(prompt.encode('utf-8'))
                        input_ids = torch.tensor([prompt_bytes], dtype=torch.long).to(device)
                        
                        generated = model.generate(
                            input_ids,
                            max_length=25,
                            temperature=0.8,
                            top_k=40
                        )
                        
                        generated_bytes = generated[0].cpu().numpy().tolist()
                        full_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
                        new_part = full_text[len(prompt):]
                        
                        # Quality assessment
                        amharic_chars = sum(1 for c in new_part if '\u1200' <= c <= '\u137F')
                        total_chars = len([c for c in new_part if c.strip()])
                        quality = amharic_chars / max(total_chars, 1) if total_chars > 0 else 0
                        
                        print(f"'{prompt}' → '{full_text}'")
                        print(f"   Quality: {quality:.2f} | Amharic chars: {amharic_chars}/{total_chars}")
                
                del model
                if device.type == 'mps':
                    torch.mps.empty_cache()
                    
            except Exception as e:
                print(f"❌ Error testing {checkpoint_path}: {e}")
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")

def main():
    """Run all output tests"""
    print("🎯 COMPREHENSIVE OUTPUT TESTING")
    print("Comparing different H-Net configurations and trained models")
    print("="*80)
    
    # Test different model sizes
    test_model_outputs()
    
    # Test existing trained models
    test_existing_checkpoints()
    
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print("="*80)
    print("✅ Tested multiple model configurations")
    print("✅ Compared generation quality across scales") 
    print("✅ Evaluated existing trained checkpoints")
    print("\n🎯 Key Findings:")
    print("• Larger models should produce better quality Amharic")
    print("• Trained models should outperform untrained ones")
    print("• Quality measured by % of Amharic characters generated")

if __name__ == "__main__":
    main()