#!/usr/bin/env python3
"""
Create High-Quality Morpheme-Annotated Amharic Training Dataset
"""

import json
import torch
import sys
import os
sys.path.append('src')

def create_morpheme_annotated_dataset():
    """Create a dataset with proper morpheme boundary annotations"""
    
    print("🧠 Creating Morpheme-Annotated Amharic Dataset")
    print("=" * 60)
    
    # High-quality morpheme-annotated Amharic examples
    # Each entry has: word, morphemes, meanings, boundary positions
    morpheme_dataset = [
        # Verbs with complex conjugation
        {
            "word": "ይፈልጋሉ",
            "morphemes": ["ይ", "ፈልግ", "አል", "ው"],
            "meanings": ["3rd.MASC", "want", "AUX", "3rd.PL"],
            "gloss": "they want",
            "boundaries": [0, 3, 12, 15]  # byte positions
        },
        {
            "word": "አልመጣችም",
            "morphemes": ["አል", "መጣ", "ች", "ም"],
            "meanings": ["NEG", "come", "3rd.FEM.SG", "NEG"],
            "gloss": "she did not come",
            "boundaries": [0, 6, 12, 15]
        },
        {
            "word": "ሰማሁ",
            "morphemes": ["ሰማ", "ሁ"],
            "meanings": ["hear", "1st.SG"],
            "gloss": "I heard",
            "boundaries": [0, 9]
        },
        {
            "word": "ትሄዳለች",
            "morphemes": ["ት", "ሄድ", "አለ", "ች"],
            "meanings": ["2nd", "go", "AUX", "FEM"],
            "gloss": "she goes",
            "boundaries": [0, 3, 9, 15]
        },
        {
            "word": "አይሰሩም",
            "morphemes": ["አይ", "ሰር", "ው", "ም"],
            "meanings": ["NEG", "work", "3rd.MASC", "NEG"],
            "gloss": "he does not work",
            "boundaries": [0, 6, 12, 15]
        },
        
        # Nouns with possessives and plurals
        {
            "word": "በቤታችን",
            "morphemes": ["በ", "ቤት", "አችን"],
            "meanings": ["PREP", "house", "1st.PL.POSS"],
            "gloss": "in our house",
            "boundaries": [0, 3, 9]
        },
        {
            "word": "ተማሪዎች",
            "morphemes": ["ተማሪ", "ዎች"],
            "meanings": ["student", "PL"],
            "gloss": "students",
            "boundaries": [0, 12]
        },
        {
            "word": "መጽሐፍቱ",
            "morphemes": ["መጽሐፍ", "ቱ"],
            "meanings": ["book", "DEF"],
            "gloss": "the book",
            "boundaries": [0, 15]
        },
        {
            "word": "ልጆቻችን",
            "morphemes": ["ልጆች", "አችን"],
            "meanings": ["children", "1st.PL.POSS"],
            "gloss": "our children",
            "boundaries": [0, 12]
        },
        {
            "word": "መምህሩ",
            "morphemes": ["መምህር", "ው"],
            "meanings": ["teacher", "DEF.MASC"],
            "gloss": "the teacher",
            "boundaries": [0, 15]
        },
        
        # Adjectives and descriptors
        {
            "word": "ኢትዮጵያዊ",
            "morphemes": ["ኢትዮጵያ", "ዊ"],
            "meanings": ["Ethiopia", "ADJ"],
            "gloss": "Ethiopian",
            "boundaries": [0, 15]
        },
        {
            "word": "ትልቁ",
            "morphemes": ["ትልቅ", "ው"],
            "meanings": ["big", "DEF.MASC"],
            "gloss": "the big one",
            "boundaries": [0, 12]
        },
        
        # Common phrases and expressions
        {
            "word": "እንዴት",
            "morphemes": ["እንዴት"],
            "meanings": ["how"],
            "gloss": "how",
            "boundaries": [0]
        },
        {
            "word": "ልመጣ",
            "morphemes": ["ል", "መጣ"],
            "meanings": ["1st.SG", "come"],
            "gloss": "I will come",
            "boundaries": [0, 3]
        },
        {
            "word": "አላውቅም",
            "morphemes": ["አል", "አውቅ", "ም"],
            "meanings": ["NEG", "know", "NEG"],
            "gloss": "I don't know",
            "boundaries": [0, 6, 15]
        },
        
        # Complex compound words
        {
            "word": "መንግስት",
            "morphemes": ["መንግስት"],
            "meanings": ["government"],
            "gloss": "government",
            "boundaries": [0]
        },
        {
            "word": "ዩኒቨርሲቲ",
            "morphemes": ["ዩኒቨርሲቲ"],
            "meanings": ["university"],
            "gloss": "university", 
            "boundaries": [0]
        },
        {
            "word": "አስተማሪ",
            "morphemes": ["አስተማሪ"],
            "meanings": ["teacher"],
            "gloss": "teacher",
            "boundaries": [0]
        }
    ]
    
    print(f"📚 Created {len(morpheme_dataset)} morpheme-annotated examples")
    
    # Validate boundary positions
    validated_dataset = []
    for i, entry in enumerate(morpheme_dataset):
        word = entry["word"]
        morphemes = entry["morphemes"]
        boundaries = entry["boundaries"]
        
        # Convert word to bytes to check boundary positions
        word_bytes = word.encode('utf-8')
        
        print(f"\n{i+1}. Validating: {word}")
        print(f"   Morphemes: {' + '.join(morphemes)}")
        print(f"   Byte length: {len(word_bytes)}")
        print(f"   Boundaries: {boundaries}")
        
        # Validate boundaries make sense
        if boundaries[-1] <= len(word_bytes):
            validated_dataset.append(entry)
            print(f"   ✅ Valid")
        else:
            print(f"   ❌ Invalid boundaries")
    
    return validated_dataset

def save_morpheme_dataset(dataset, output_path):
    """Save morpheme dataset in training-ready format"""
    
    print(f"\n💾 Saving dataset to {output_path}")
    
    # Create training format
    training_data = {
        "metadata": {
            "description": "High-quality morpheme-annotated Amharic dataset for H-Net training",
            "language": "Amharic",
            "script": "Ge'ez",
            "total_examples": len(dataset),
            "annotation_type": "morpheme_boundaries"
        },
        "examples": dataset
    }
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(dataset)} examples")
    return training_data

def create_boundary_labels(dataset):
    """Create boundary label tensors for training"""
    
    print(f"\n🏷️ Creating Boundary Labels for Training")
    
    boundary_examples = []
    
    for entry in dataset:
        word = entry["word"]
        boundaries = entry["boundaries"]
        morphemes = entry["morphemes"]
        
        # Convert to bytes
        word_bytes = word.encode('utf-8')
        
        # Create binary boundary labels (1 = boundary, 0 = no boundary)
        boundary_labels = [0] * len(word_bytes)
        
        # Mark boundary positions
        for boundary_pos in boundaries:
            if boundary_pos < len(boundary_labels):
                boundary_labels[boundary_pos] = 1
        
        boundary_example = {
            "word": word,
            "morphemes": morphemes,
            "byte_sequence": list(word_bytes),
            "boundary_labels": boundary_labels,
            "length": len(word_bytes)
        }
        
        boundary_examples.append(boundary_example)
        
        print(f"Word: {word}")
        print(f"Boundaries: {boundary_labels}")
        print(f"Expected morpheme count: {sum(boundary_labels)}")
        print("-" * 40)
    
    return boundary_examples

def demonstrate_improved_segmentation(dataset):
    """Show how proper boundaries should improve segmentation"""
    
    print(f"\n🎯 Demonstrating Improved Segmentation")
    print("=" * 60)
    
    for entry in dataset[:5]:  # Show first 5 examples
        word = entry["word"]
        expected_morphemes = entry["morphemes"]
        gloss = entry["gloss"]
        
        print(f"\nWord: {word}")
        print(f"Expected: {' + '.join(expected_morphemes)}")
        print(f"Meaning: {gloss}")
        
        # Show what current model detects vs what we want
        print(f"Current model: over-segments into ~8 pieces")
        print(f"Target: {len(expected_morphemes)} proper morphemes")
        print("=" * 40)

if __name__ == "__main__":
    print("🇪🇹 Amharic Morpheme-Annotated Dataset Creation")
    print("=" * 60)
    
    # Create dataset
    dataset = create_morpheme_annotated_dataset()
    
    # Save dataset
    output_path = "data/morpheme_annotations.json"
    os.makedirs("data", exist_ok=True)
    training_data = save_morpheme_dataset(dataset, output_path)
    
    # Create boundary labels for training
    boundary_examples = create_boundary_labels(dataset)
    
    # Save boundary training data
    boundary_path = "data/boundary_training_data.json"
    with open(boundary_path, 'w', encoding='utf-8') as f:
        json.dump(boundary_examples, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Boundary training data saved to {boundary_path}")
    
    # Demonstrate expected improvements
    demonstrate_improved_segmentation(dataset)
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Use this data to train H-Net with morpheme boundary supervision")
    print("2. Add morpheme boundary loss to training objective") 
    print("3. Validate that model learns proper Amharic morpheme segmentation")
    print("4. Test improved text generation with meaningful words")
    
    print(f"\n📁 Files created:")
    print(f"   • {output_path} - Full morpheme annotations")
    print(f"   • {boundary_path} - Boundary training labels")