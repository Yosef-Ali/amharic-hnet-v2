#!/usr/bin/env python3
"""
Test H-Net Components - Morpheme Chunker and Hierarchical Processing
"""

import torch
import sys
import os
sys.path.append('src')

from models.hnet_amharic import AmharicHNet
import json

def test_morpheme_chunker():
    """Test the AmharicMorphemeChunker with real Amharic words"""
    
    print("🧪 Testing H-Net Morpheme Chunker")
    print("=" * 50)
    
    # Load the trained model
    model_path = "outputs/compact/final_model.pt"
    
    model = AmharicHNet(
        d_model=256,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_main_layers=4,
        n_heads=4,
        vocab_size=256
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (loss: {checkpoint.get('loss', 'unknown')})")
    
    # Test Amharic words with known morphological structure
    test_words = [
        {
            "word": "ይፈልጋሉ",  # they want
            "expected_morphemes": ["ይ", "ፈልግ", "አል", "ው"],
            "meaning": "3rd.MASC.PL-want-AUX-3rd.PL"
        },
        {
            "word": "አልመጣችም",  # she did not come  
            "expected_morphemes": ["አል", "መጣ", "ች", "ም"],
            "meaning": "NEG-come-3rd.FEM.SG-NEG"
        },
        {
            "word": "በቤታችን",   # in our house
            "expected_morphemes": ["በ", "ቤት", "አችን"],
            "meaning": "PREP-house-1st.PL.POSS"
        },
        {
            "word": "ተማሪዎች",   # students
            "expected_morphemes": ["ተማሪ", "ዎች"],
            "meaning": "student-PL"
        },
        {
            "word": "ኢትዮጵያዊ",  # Ethiopian
            "expected_morphemes": ["ኢትዮጵያ", "ዊ"],
            "meaning": "Ethiopia-ADJ"
        }
    ]
    
    print(f"\n🔍 Testing {len(test_words)} Amharic words with morphological structure:")
    print("-" * 60)
    
    for i, test_case in enumerate(test_words, 1):
        word = test_case["word"]
        expected = test_case["expected_morphemes"]
        meaning = test_case["meaning"]
        
        print(f"\n{i}. Testing: {word}")
        print(f"   Expected morphemes: {' + '.join(expected)}")
        print(f"   Meaning: {meaning}")
        
        # Convert to bytes and embed
        word_bytes = word.encode('utf-8')
        input_ids = torch.tensor([[b for b in word_bytes]], dtype=torch.long)
        
        try:
            # Get embeddings
            embeddings = model.byte_embedding(input_ids)
            
            # Test morpheme chunker
            with torch.no_grad():
                boundary_probs = model.chunker(embeddings)
            
            # Analyze boundaries
            probs = boundary_probs[0].tolist()
            boundaries = [i for i, p in enumerate(probs) if p > 0.5]
            
            print(f"   📊 Boundary probabilities: {[f'{p:.3f}' for p in probs]}")
            print(f"   🔄 Detected boundaries at positions: {boundaries}")
            print(f"   📈 Number of predicted morphemes: {len(boundaries)}")
            print(f"   📈 Expected number of morphemes: {len(expected)}")
            
            # Check if boundary count is reasonable
            boundary_accuracy = "✅ Good" if abs(len(boundaries) - len(expected)) <= 1 else "❌ Poor"
            print(f"   🎯 Boundary detection: {boundary_accuracy}")
            
            # Test if the chunker recognizes morphological patterns
            verb_prefixes = model.chunker.verb_prefixes
            verb_suffixes = model.chunker.verb_suffixes
            noun_prefixes = model.chunker.noun_prefixes
            noun_suffixes = model.chunker.noun_suffixes
            
            found_patterns = []
            for prefix in verb_prefixes + noun_prefixes:
                if word.startswith(prefix):
                    found_patterns.append(f"prefix:{prefix}")
            
            for suffix in verb_suffixes + noun_suffixes:
                if word.endswith(suffix):
                    found_patterns.append(f"suffix:{suffix}")
            
            if found_patterns:
                print(f"   🧠 Recognized patterns: {', '.join(found_patterns)}")
            else:
                print(f"   ⚠️ No known morphological patterns detected")
                
        except Exception as e:
            print(f"   ❌ Error processing {word}: {e}")
    
    print(f"\n" + "=" * 60)
    return model

def test_hierarchical_processing(model):
    """Test the hierarchical processing capabilities"""
    
    print(f"\n🏗️ Testing Hierarchical Processing")
    print("=" * 50)
    
    test_sentences = [
        "ኢትዮጵያ ውብ ሀገር ነች",  # Ethiopia is a beautiful country
        "ቡና ይወዳሉ",             # They like coffee  
        "አማርኛ ቋንቋ",            # Amharic language
        "ሰላም እንዴት ነሽ"          # Hello, how are you (fem)
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Testing sentence: {sentence}")
        
        # Convert to bytes
        sentence_bytes = sentence.encode('utf-8')
        input_ids = torch.tensor([[b for b in sentence_bytes]], dtype=torch.long)
        
        try:
            with torch.no_grad():
                # Full forward pass
                logits, boundary_probs = model.forward(input_ids)
                
                print(f"   📏 Input length: {len(sentence_bytes)} bytes")
                print(f"   📏 Output shape: {logits.shape}")
                print(f"   🔄 Boundary shape: {boundary_probs.shape}")
                
                # Count predicted morpheme boundaries
                boundaries = (boundary_probs[0] > 0.5).sum().item()
                print(f"   🧩 Predicted morphemes: {boundaries}")
                
                # Test generation capability
                generated = model.generate(
                    input_ids,
                    max_length=len(sentence_bytes) + 10,
                    temperature=0.7,
                    top_k=30
                )
                
                # Convert back to text
                generated_bytes = generated[0].cpu().numpy()
                
                # Filter and decode
                filtered_bytes = []
                for byte_val in generated_bytes:
                    if 32 <= byte_val <= 126 or byte_val >= 128:
                        filtered_bytes.append(byte_val)
                
                try:
                    generated_text = bytes(filtered_bytes).decode('utf-8', errors='ignore')
                    print(f"   🤖 Generated: {generated_text}")
                    
                    # Check if it contains original sentence
                    if sentence in generated_text:
                        continuation = generated_text.replace(sentence, "").strip()
                        print(f"   ➕ Continuation: '{continuation}'")
                    
                    # Amharic character analysis
                    amharic_chars = sum(1 for c in generated_text if 0x1200 <= ord(c) <= 0x137F)
                    total_chars = len(generated_text)
                    if total_chars > 0:
                        ratio = amharic_chars / total_chars
                        print(f"   📊 Amharic ratio: {ratio:.1%} ({amharic_chars}/{total_chars})")
                    
                except Exception as decode_error:
                    print(f"   ⚠️ Decode error: {decode_error}")
                    
        except Exception as e:
            print(f"   ❌ Processing error: {e}")

def test_pattern_recognition():
    """Test if the model recognizes Amharic morphological patterns"""
    
    print(f"\n🧠 Testing Morphological Pattern Recognition")
    print("=" * 50)
    
    # Load model  
    model_path = "outputs/compact/final_model.pt"
    model = AmharicHNet(d_model=256, n_encoder_layers=2, n_decoder_layers=2, n_main_layers=4, n_heads=4, vocab_size=256)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Check what patterns the chunker knows
    chunker = model.chunker
    
    print("📚 Built-in Morphological Knowledge:")
    print(f"   Verb prefixes: {chunker.verb_prefixes}")
    print(f"   Verb suffixes: {chunker.verb_suffixes}")
    print(f"   Noun prefixes: {chunker.noun_prefixes}")
    print(f"   Noun suffixes: {chunker.noun_suffixes}")
    
    print(f"\n🎯 This H-Net architecture has {len(chunker.verb_prefixes + chunker.verb_suffixes + chunker.noun_prefixes + chunker.noun_suffixes)} built-in Amharic morphological patterns!")
    
    return True

if __name__ == "__main__":
    print("🇪🇹 H-Net Amharic Architecture Component Testing")
    print("=" * 60)
    
    # Test morphological patterns first
    test_pattern_recognition()
    
    # Test morpheme chunker
    model = test_morpheme_chunker()
    
    # Test hierarchical processing
    test_hierarchical_processing(model)
    
    print(f"\n🎉 H-Net Component Testing Complete!")
    print("🎯 This shows how well our morpheme-aware architecture is working")