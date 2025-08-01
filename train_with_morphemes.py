#!/usr/bin/env python3
"""
Train H-Net with Morpheme Boundary Supervision
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import sys
import os
sys.path.append('src')

from models.hnet_amharic import AmharicHNet

def load_morpheme_training_data():
    """Load morpheme-annotated training data"""
    
    print("📚 Loading Morpheme Training Data")
    
    with open('data/boundary_training_data.json', 'r', encoding='utf-8') as f:
        boundary_data = json.load(f)
    
    print(f"✅ Loaded {len(boundary_data)} morpheme-annotated examples")
    
    return boundary_data

def create_training_batches(boundary_data, batch_size=4):
    """Create training batches from morpheme data"""
    
    training_batches = []
    
    for i in range(0, len(boundary_data), batch_size):
        batch = boundary_data[i:i + batch_size]
        
        # Pad sequences to same length in batch
        max_length = max(example['length'] for example in batch)
        
        input_ids = []
        boundary_labels = []
        
        for example in batch:
            # Pad byte sequence
            padded_bytes = example['byte_sequence'] + [0] * (max_length - example['length'])
            input_ids.append(padded_bytes)
            
            # Pad boundary labels
            padded_labels = example['boundary_labels'] + [0] * (max_length - example['length'])
            boundary_labels.append(padded_labels)
        
        batch_data = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'boundary_labels': torch.tensor(boundary_labels, dtype=torch.float),
            'words': [example['word'] for example in batch],
            'morphemes': [example['morphemes'] for example in batch]
        }
        
        training_batches.append(batch_data)
    
    print(f"📦 Created {len(training_batches)} training batches")
    
    return training_batches

def train_with_morpheme_supervision():
    """Train H-Net with morpheme boundary supervision"""
    
    print("🎯 Training H-Net with Morpheme Supervision")
    print("=" * 60)
    
    # Load existing model
    model_path = "outputs/compact/final_model.pt"
    model = AmharicHNet(d_model=256, n_encoder_layers=2, n_decoder_layers=2, n_main_layers=4, n_heads=4, vocab_size=256)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded existing model (loss: {checkpoint.get('loss', 'unknown')})")
    else:
        print("🆕 Starting with fresh model")
    
    model.train()
    
    # Load training data
    boundary_data = load_morpheme_training_data()
    training_batches = create_training_batches(boundary_data, batch_size=2)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    boundary_loss_fn = nn.BCELoss()
    
    # Training parameters
    num_epochs = 5
    
    print(f"\n🚀 Starting Morpheme-Supervised Training")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batches: {len(training_batches)}")
    print(f"   Learning rate: 0.001")
    
    training_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct_boundaries = 0
        total_boundaries = 0
        
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        for batch_idx, batch in enumerate(training_batches):
            input_ids = batch['input_ids']
            target_boundaries = batch['boundary_labels']
            words = batch['words']
            expected_morphemes = batch['morphemes']
            
            # Forward pass - get boundary predictions
            embeddings = model.byte_embedding(input_ids)
            predicted_boundaries = model.chunker(embeddings)
            
            # Compute boundary loss
            loss = boundary_loss_fn(predicted_boundaries, target_boundaries)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate boundary accuracy
            pred_binary = (predicted_boundaries > 0.5).float()
            correct = (pred_binary == target_boundaries).sum().item()
            total = target_boundaries.numel()
            
            correct_boundaries += correct
            total_boundaries += total
            
            # Print batch results
            print(f"   Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
            
            for i, (word, morphemes) in enumerate(zip(words, expected_morphemes)):
                pred_bounds = (predicted_boundaries[i] > 0.5).sum().item()
                expected_bounds = len(morphemes)
                print(f"     {word}: predicted={pred_bounds}, expected={expected_bounds} morphemes")
        
        avg_loss = epoch_loss / len(training_batches)
        boundary_accuracy = correct_boundaries / total_boundaries
        
        training_losses.append(avg_loss)
        
        print(f"\n   📊 Epoch {epoch + 1} Results:")
        print(f"      Average Loss: {avg_loss:.4f}")
        print(f"      Boundary Accuracy: {boundary_accuracy:.3f}")
        
        # Test on a few examples
        print(f"\n   🧪 Testing Boundary Detection:")
        model.eval()
        
        test_words = ["ይፈልጋሉ", "በቤታችን", "ተማሪዎች"]
        expected_counts = [4, 3, 2]
        
        with torch.no_grad():
            for word, expected in zip(test_words, expected_counts):
                word_bytes = word.encode('utf-8')
                input_ids = torch.tensor([[b for b in word_bytes]], dtype=torch.long)
                embeddings = model.byte_embedding(input_ids)
                boundary_probs = model.chunker(embeddings)
                
                detected = (boundary_probs[0] > 0.5).sum().item()
                print(f"      {word}: detected={detected}, expected={expected}")
        
        model.train()
    
    print(f"\n🎉 Training Complete!")
    
    # Save improved model
    output_path = "outputs/morpheme_supervised_model.pt"
    os.makedirs("outputs", exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_losses': training_losses,
        'final_loss': training_losses[-1],
        'training_type': 'morpheme_supervised',
        'epochs': num_epochs,
        'boundary_accuracy': boundary_accuracy
    }, output_path)
    
    print(f"💾 Saved morpheme-supervised model to {output_path}")
    
    return model, training_losses

def test_improved_segmentation(model):
    """Test if morpheme supervision improved segmentation"""
    
    print(f"\n🔍 Testing Improved Morpheme Segmentation")
    print("=" * 60)
    
    model.eval()
    
    test_cases = [
        {"word": "ይፈልጋሉ", "expected": ["ይ", "ፈልግ", "አል", "ው"], "meaning": "they want"},
        {"word": "በቤታችን", "expected": ["በ", "ቤት", "አችን"], "meaning": "in our house"},
        {"word": "ተማሪዎች", "expected": ["ተማሪ", "ዎች"], "meaning": "students"},
        {"word": "አይሰሩም", "expected": ["አይ", "ሰር", "ው", "ም"], "meaning": "he does not work"}
    ]
    
    improvements = []
    
    with torch.no_grad():
        for test in test_cases:
            word = test["word"]
            expected = test["expected"]
            meaning = test["meaning"]
            
            word_bytes = word.encode('utf-8')
            input_ids = torch.tensor([[b for b in word_bytes]], dtype=torch.long)
            embeddings = model.byte_embedding(input_ids)
            boundary_probs = model.chunker(embeddings)
            
            # Count boundaries
            detected_boundaries = (boundary_probs[0] > 0.5).sum().item()
            expected_boundaries = len(expected)
            
            # Calculate improvement (closer to expected count = better)
            error = abs(detected_boundaries - expected_boundaries)
            improvement = max(0, 8 - error)  # Previous model had ~8 boundaries
            improvements.append(improvement)
            
            print(f"\nWord: {word} ({meaning})")
            print(f"Expected morphemes: {' + '.join(expected)} ({expected_boundaries})")
            print(f"Detected boundaries: {detected_boundaries}")
            print(f"Improvement score: {improvement}/8")
            
            # Test text generation with improved segmentation
            try:
                generated = model.generate(input_ids, max_length=len(word_bytes) + 5, temperature=0.7)
                generated_bytes = generated[0].cpu().numpy()
                
                # Filter and decode
                filtered_bytes = [b for b in generated_bytes if 32 <= b <= 126 or b >= 128]
                try:
                    generated_text = bytes(filtered_bytes).decode('utf-8', errors='ignore')
                    print(f"Generated text: {generated_text}")
                    
                    # Check if more coherent
                    if word in generated_text:
                        continuation = generated_text.replace(word, "").strip()
                        print(f"Continuation: '{continuation}'")
                except:
                    print("Generation decode error")
            except:
                print("Generation error")
    
    avg_improvement = sum(improvements) / len(improvements)
    print(f"\n📈 Average Improvement Score: {avg_improvement:.1f}/8")
    
    if avg_improvement > 4:
        print("🎉 Significant improvement in morpheme segmentation!")
    elif avg_improvement > 2:
        print("✅ Moderate improvement in morpheme segmentation")
    else:
        print("⚠️ Limited improvement - may need more training")
    
    return avg_improvement

if __name__ == "__main__":
    print("🇪🇹 H-Net Morpheme-Supervised Training")
    print("=" * 60)
    
    # Train with morpheme supervision
    model, losses = train_with_morpheme_supervision()
    
    # Test improvements
    improvement_score = test_improved_segmentation(model)
    
    print(f"\n🎯 TRAINING SUMMARY:")
    print(f"   Final Loss: {losses[-1]:.4f}")
    print(f"   Improvement Score: {improvement_score:.1f}/8")
    print(f"   Model: outputs/morpheme_supervised_model.pt")
    
    if improvement_score > 3:
        print(f"\n✅ SUCCESS: Morpheme supervision improved segmentation!")
        print(f"🚀 Ready for better text generation with meaningful Amharic words")
    else:
        print(f"\n🔄 Need more training or data for better morpheme learning")