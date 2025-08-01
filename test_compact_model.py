#!/usr/bin/env python3
"""
Test the compact enhanced H-Net model
"""

import torch
from src.models.proper_hnet_amharic import AmharicHNetMixer
import os

def load_and_test_compact_model():
    """Load and test the compact model"""
    print("🧠 TESTING COMPACT ENHANCED H-NET")
    print("="*50)
    
    # Initialize compact model (same config as training)
    model = AmharicHNetMixer(
        vocab_size=256,
        d_model=384,
        n_heads=6,
        n_backbone_layers=4,
        max_chunks=64
    )
    
    # Load the latest checkpoint
    checkpoint_path = "outputs/compact_enhanced_epoch_5.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
        print(f"✅ Training loss: {checkpoint['loss']:.4f}")
    else:
        print("❌ No checkpoint found")
        return
    
    model.eval()
    
    # Test generation
    test_prompts = [
        ("ኢትዮጵያ", "Ethiopia"),
        ("አማርኛ", "Amharic"),
        ("ቡና", "coffee"),
        ("አዲስ አበባ", "Addis Ababa"),
        ("ባህል", "culture"),
        ("ህዝብ", "people"),
        ("ቤት", "house"),
        ("ውሃ", "water")
    ]
    
    print("\n" + "="*60)
    print("🎯 GENERATION RESULTS")
    print("="*60)
    
    quality_scores = []
    
    with torch.no_grad():
        for i, (prompt, meaning) in enumerate(test_prompts, 1):
            print(f"\n{i}. '{prompt}' ({meaning})")
            print("-" * 40)
            
            # Convert to bytes
            prompt_bytes = list(prompt.encode('utf-8'))
            input_ids = torch.tensor([prompt_bytes], dtype=torch.long)
            
            # Test with different temperatures
            for temp in [0.5, 0.8, 1.2]:
                try:
                    generated = model.generate(
                        input_ids,
                        max_length=30,
                        temperature=temp,
                        top_k=40
                    )
                    
                    generated_bytes = generated[0].cpu().numpy().tolist()
                    full_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
                    new_part = full_text[len(prompt):]
                    
                    # Basic quality assessment
                    amharic_chars = sum(1 for c in new_part if '\u1200' <= c <= '\u137F')
                    total_chars = len([c for c in new_part if c.strip()])
                    quality = amharic_chars / max(total_chars, 1) if total_chars > 0 else 0
                    
                    print(f"  T={temp}: '{full_text}'")
                    print(f"         New: '{new_part}' (Quality: {quality:.2f})")
                    
                    if temp == 0.8:  # Track middle temperature
                        quality_scores.append(quality)
                        
                except Exception as e:
                    print(f"  T={temp}: Error - {e}")
                    if temp == 0.8:
                        quality_scores.append(0.0)
    
    # Overall assessment
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    meaningful_count = sum(1 for q in quality_scores if q > 0.3)
    
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Average Quality Score: {avg_quality:.3f}")
    print(f"Meaningful Generations: {meaningful_count}/{len(quality_scores)}")
    print(f"Training Progress: Loss reduced to {checkpoint['loss']:.4f}")
    
    if avg_quality > 0.5:
        print("✅ Model generating meaningful Amharic!")
    elif avg_quality > 0.2:
        print("🔄 Model showing improvement, needs more training")
    else:
        print("🔄 Model needs more training data or longer training")
    
    print("="*60)

if __name__ == "__main__":
    load_and_test_compact_model()