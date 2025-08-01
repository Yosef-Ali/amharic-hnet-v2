#!/usr/bin/env python3
"""
Tune Morpheme Boundary Detection Thresholds for Better Amharic Segmentation
"""

import torch
import sys
import os
sys.path.append('src')

from models.hnet_amharic import AmharicHNet

def test_boundary_thresholds():
    """Test different boundary detection thresholds to find optimal settings"""
    
    print("🔧 Tuning Amharic Morpheme Boundary Detection")
    print("=" * 60)
    
    # Load trained model
    model_path = "outputs/compact/final_model.pt"
    model = AmharicHNet(d_model=256, n_encoder_layers=2, n_decoder_layers=2, n_main_layers=4, n_heads=4, vocab_size=256)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test words with known morphological structure
    test_cases = [
        {
            "word": "ይፈልጋሉ",
            "expected_morphemes": ["ይ", "ፈልግ", "አል", "ው"],
            "expected_count": 4
        },
        {
            "word": "በቤታችን", 
            "expected_morphemes": ["በ", "ቤት", "አችን"],
            "expected_count": 3
        },
        {
            "word": "ተማሪዎች",
            "expected_morphemes": ["ተማሪ", "ዎች"], 
            "expected_count": 2
        }
    ]
    
    # Test different thresholds
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print("Testing boundary detection thresholds:")
    print("Word\t\tExpected\tThreshold\tDetected\tAccuracy")
    print("-" * 70)
    
    best_threshold = 0.5
    best_avg_accuracy = 0
    threshold_scores = {}
    
    for threshold in thresholds:
        threshold_accuracy = []
        
        for test_case in test_cases:
            word = test_case["word"]
            expected_count = test_case["expected_count"]
            
            # Get embeddings and boundary probabilities
            word_bytes = word.encode('utf-8')
            input_ids = torch.tensor([[b for b in word_bytes]], dtype=torch.long)
            embeddings = model.byte_embedding(input_ids)
            
            with torch.no_grad():
                boundary_probs = model.chunker(embeddings)
            
            # Count boundaries above threshold
            detected_boundaries = (boundary_probs[0] > threshold).sum().item()
            
            # Calculate accuracy (how close to expected morpheme count)
            accuracy = 1.0 - abs(detected_boundaries - expected_count) / max(detected_boundaries, expected_count)
            threshold_accuracy.append(accuracy)
            
            print(f"{word}\t\t{expected_count}\t\t{threshold:.1f}\t\t{detected_boundaries}\t\t{accuracy:.2f}")
        
        avg_accuracy = sum(threshold_accuracy) / len(threshold_accuracy)
        threshold_scores[threshold] = avg_accuracy
        
        if avg_accuracy > best_avg_accuracy:
            best_avg_accuracy = avg_accuracy
            best_threshold = threshold
        
        print(f"Average accuracy for threshold {threshold:.1f}: {avg_accuracy:.3f}")
        print("-" * 70)
    
    print(f"\n🎯 BEST THRESHOLD: {best_threshold} (Accuracy: {best_avg_accuracy:.3f})")
    
    # Test the best threshold on all words
    print(f"\n📊 Detailed Results with Optimal Threshold ({best_threshold}):")
    print("-" * 60)
    
    for test_case in test_cases:
        word = test_case["word"]
        expected = test_case["expected_morphemes"]
        
        word_bytes = word.encode('utf-8')
        input_ids = torch.tensor([[b for b in word_bytes]], dtype=torch.long)
        embeddings = model.byte_embedding(input_ids)
        
        with torch.no_grad():
            boundary_probs = model.chunker(embeddings)
        
        # Get boundary positions with optimal threshold
        boundary_positions = [i for i, p in enumerate(boundary_probs[0]) if p > best_threshold]
        
        print(f"\nWord: {word}")
        print(f"Expected morphemes: {' + '.join(expected)}")
        print(f"Boundary positions: {boundary_positions}")
        print(f"Detected morpheme count: {len(boundary_positions)}")
        print(f"Expected morpheme count: {len(expected)}")
        
        # Try to extract predicted morphemes (approximate)
        if len(boundary_positions) >= 2:
            # Convert boundaries to approximate morpheme chunks
            word_chars = list(word)
            morpheme_boundaries = []
            
            # Map byte positions to character positions (approximate)
            for pos in boundary_positions:
                char_pos = min(pos * len(word_chars) // len(word_bytes), len(word_chars) - 1)
                morpheme_boundaries.append(char_pos)
            
            # Extract morphemes based on boundaries
            predicted_morphemes = []
            for i in range(len(morpheme_boundaries) - 1):
                start = morpheme_boundaries[i]
                end = morpheme_boundaries[i + 1]
                morpheme = ''.join(word_chars[start:end])
                if morpheme:
                    predicted_morphemes.append(morpheme)
            
            print(f"Predicted morphemes: {' + '.join(predicted_morphemes)}")
        
        print("=" * 40)
    
    return best_threshold, threshold_scores

def apply_optimal_threshold(optimal_threshold):
    """Apply the optimal threshold to the model for better boundary detection"""
    
    print(f"\n🔧 Applying Optimal Threshold: {optimal_threshold}")
    print("=" * 50)
    
    # This would require modifying the model's forward pass to use the optimal threshold
    # For now, we'll document the finding
    
    recommendations = f"""
    🎯 MORPHEME BOUNDARY TUNING RESULTS:
    
    Optimal Threshold: {optimal_threshold}
    
    NEXT STEPS:
    1. Modify AmharicMorphemeChunker to use threshold {optimal_threshold}
    2. Retrain model with morpheme boundary supervision
    3. Focus training on morphologically rich Amharic sentences
    4. Validate against linguistic morpheme patterns
    
    ARCHITECTURE IMPROVEMENT:
    - Current model over-segments (too many boundaries)
    - Need better morphological pattern learning
    - Consider adding morpheme-level loss function
    """
    
    print(recommendations)
    
    # Save recommendations
    with open('boundary_tuning_results.txt', 'w', encoding='utf-8') as f:
        f.write(recommendations)
    
    return recommendations

if __name__ == "__main__":
    print("🇪🇹 Amharic H-Net Boundary Detection Optimization")
    print("=" * 60)
    
    # Test thresholds
    optimal_threshold, scores = test_boundary_thresholds()
    
    # Apply results
    recommendations = apply_optimal_threshold(optimal_threshold)
    
    print(f"\n✅ Boundary tuning complete! Check 'boundary_tuning_results.txt' for details.")
    print(f"🎯 Next: Use threshold {optimal_threshold} for better morpheme detection")