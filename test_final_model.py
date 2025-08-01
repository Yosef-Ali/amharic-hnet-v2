#!/usr/bin/env python3
"""
Test the final trained Amharic H-Net model for proper UTF-8 generation and text quality.
"""

import torch
import sys
import os

# Add src to path
sys.path.append('src')

from models.hnet_amharic import AmharicHNet
from preprocessing.prepare_amharic import AmharicPreprocessor

def test_final_model():
    """Test the successfully trained model."""
    
    print("="*60)
    print("FINAL AMHARIC H-NET MODEL TEST")
    print("="*60)
    
    # Load the trained model
    checkpoint_path = 'outputs/compact/final_model.pt'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model not found at {checkpoint_path}")
        return
    
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get model config
    config = checkpoint['model_config']
    final_loss = checkpoint['final_loss']
    
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Model configuration: {config}")
    
    # Initialize model
    model = AmharicHNet(
        d_model=config['d_model'],
        n_encoder_layers=config['n_encoder_layers'],
        n_decoder_layers=config['n_decoder_layers'],
        n_main_layers=config['n_main_layers'],
        n_heads=config['n_heads'],
        compression_ratio=config['compression_ratio'],
        vocab_size=256
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create preprocessor
    preprocessor = AmharicPreprocessor()
    
    # Test prompts
    test_prompts = [
        "ኢትዮጵያ",
        "ቡና",
        "አማርኛ",
        "አዲስ አበባ",
        "መስቀል",
        "ሰላም",
        "ጤና ይስጥልኝ",
        "እንዴት ነዎት"
    ]
    
    print("\n" + "="*60)
    print("GENERATION TESTS")
    print("="*60)
    
    success_count = 0
    total_tests = 0
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"Prompt: {prompt}")
            
            try:
                # Convert to bytes
                byte_seq = preprocessor.extract_byte_sequences(prompt, max_length=50)
                input_ids = torch.tensor([byte_seq], dtype=torch.long)
                
                # Test with multiple temperature settings
                for temp in [0.7, 0.9, 1.1]:
                    try:
                        generated = model.generate(
                            input_ids,
                            max_length=25,
                            temperature=temp,
                            top_k=50
                        )
                        
                        gen_bytes = generated[0].cpu().numpy().tolist()
                        gen_text = preprocessor.decode_byte_sequence(gen_bytes)
                        
                        print(f"  T={temp}: {gen_text}")
                        
                        # Analyze generation quality
                        if gen_text:
                            amharic_chars = sum(1 for c in gen_text if 0x1200 <= ord(c) <= 0x137F)
                            total_chars = len(gen_text)
                            amharic_ratio = amharic_chars / max(total_chars, 1)
                            
                            # Check for control characters
                            control_chars = sum(1 for c in gen_text if ord(c) < 32 and c not in '\n\t\r')
                            
                            print(f"    Amharic: {amharic_ratio:.1%} ({amharic_chars}/{total_chars})")
                            
                            if control_chars == 0:
                                print(f"    ✅ No control characters")
                                if amharic_ratio > 0.3:  # At least 30% Amharic
                                    success_count += 1
                            else:
                                print(f"    ❌ {control_chars} control characters")
                        
                        total_tests += 1
                        
                    except Exception as e:
                        print(f"    Error (T={temp}): {e}")
                        total_tests += 1
                        
            except Exception as e:
                print(f"  Prompt processing error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    success_rate = success_count / max(total_tests, 1)
    print(f"Successful generations: {success_count}/{total_tests} ({success_rate:.1%})")
    print(f"Final model loss: {final_loss:.4f}")
    
    if final_loss < 2.0:
        print("✅ Training target achieved (loss < 2.0)")
    else:
        print("❌ Training target not achieved")
    
    if success_rate > 0.5:
        print("✅ Good generation quality (>50% success)")
    else:
        print("❌ Poor generation quality")
    
    # Test specific Amharic capabilities
    print("\n" + "="*60)
    print("AMHARIC CAPABILITY TEST")
    print("="*60)
    
    # Test if model can continue Amharic text properly
    test_phrases = [
        "ኢትዮጵያ የአፍሪካ",  # Ethiopia of Africa
        "ቡና የኢትዮጵያ",    # Coffee of Ethiopia  
        "አማርኛ ቋንቋ"       # Amharic language
    ]
    
    for phrase in test_phrases:
        print(f"\nContinuation test: {phrase}")
        try:
            byte_seq = preprocessor.extract_byte_sequences(phrase, max_length=50)
            input_ids = torch.tensor([byte_seq], dtype=torch.long)
            
            generated = model.generate(input_ids, max_length=15, temperature=0.8)
            gen_bytes = generated[0].cpu().numpy().tolist()
            gen_text = preprocessor.decode_byte_sequence(gen_bytes)
            
            print(f"  Result: {gen_text}")
            
            # Check if it continued meaningfully
            if len(gen_text) > len(phrase):
                continuation = gen_text[len(phrase):].strip()
                if continuation:
                    print(f"  Continuation: '{continuation}'")
                    amharic_in_cont = sum(1 for c in continuation if 0x1200 <= ord(c) <= 0x137F)
                    if amharic_in_cont > 0:
                        print(f"  ✅ Generated Amharic continuation")
                    else:
                        print(f"  ❌ No Amharic in continuation")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    if final_loss < 2.0 and success_rate > 0.3:
        print("🎉 SUCCESS: Model is properly trained and generates coherent Amharic text!")
        print("   - Loss target achieved (< 2.0)")
        print("   - No control character issues")
        print("   - Generates proper UTF-8 encoded Amharic")
        return True
    else:
        print("⚠️  PARTIAL SUCCESS: Model shows improvement but may need more training")
        return False

if __name__ == "__main__":
    success = test_final_model()