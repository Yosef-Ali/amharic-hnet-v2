#!/usr/bin/env python3
"""
Create Practice Competition Data
Generate realistic test scenarios for your 253M parameter model
"""

import pandas as pd
import numpy as np
import random

def create_practice_test_data():
    """Create realistic competition-style test data."""
    
    print("🏆 CREATING PRACTICE COMPETITION DATA")
    print("=" * 50)
    
    # Diverse Amharic texts for different classification tasks
    amharic_texts = [
        # News/Articles (Class 0)
        "የኢትዮጵያ መንግስት አዲስ የቴክኖሎጂ ፖሊሲ አወጣ።",
        "በአዲስ አበባ የትራፊክ ችግር እየጨመረ ነው።",
        "የገበሬዎች ምርታማነት በዓመቱ 15% ጨምሯል።",
        
        # Social Media (Class 1)  
        "ዛሬ በጣም ደስተኛ ነኝ! 😊",
        "ወዳጆቼ እንዴት ናችሁ? ሰላም ነው?",
        "የቡና ስነ ስርዓት በጣም ያምራል ❤️",
        
        # Educational (Class 2)
        "አማርኛ ቋንቋ መማር ለሁሉም ጠቃሚ ነው។",
        "ሂሳብ ትምህርት በዕለት ተዕለት ሕይወት አስፈላጊ ነው።",
        "ሳይንስ የወደፊቱን ቴክኖሎጂ ይገነባል។",
        
        # Business (Class 3)
        "አዲስ ንግድ ጀመርኩ። ደንበኞች እንኳን ደህና መጡ።",
        "የገበያ ዋጋ በዚህ ሳምንት መረጋጋት አሳይቷል។",
        "ኤክስፖርት ንግድ ለሀገር ልማት አስፈላጊ ነው።",
        
        # Culture/Entertainment (Class 4)
        "የኢትዮጵያ ባህላዊ ሙዚቃ በዓለም ታዋቂ ነው።",
        "እስክስታ እና ጉራጌ ሙዚቃ በጣም ያምራል።",
        "የባህላዊ ልብስ ፋሽን ሾው ትናንት ተካሄደ។",
        
        # Technology (Class 5)
        "አዲስ መተግበሪያ በአማርኛ ቋንቋ ተዘጋጀ።",
        "ስማርት ፎን ቴክኖሎጂ በገጠር ይስፋፋል።",
        "ክላውድ ማቆያ ለንግድ ቤቶች ጠቃሚ ነው।",
        
        # Health (Class 6)
        "የጤንነት መብት ለሁሉም ዜጋ እኩል ነው።",
        "ክትባት ለህጻናት ጤንነት አስፈላጊ ነው።",
        "የአካል ብቃት እንቅስቃሴ ጤናማ ሕይወት ያመጣል។",
        
        # Sports (Class 7)
        "የኢትዮጵያ አትሌቶች በአለም ሻምፕዮና ተሳትፈዋል።",
        "የእግር ኳስ ሊግ በሳምንቱ ይጀመራል።",
        "ስፖርት ለወጣቶች ጤናማ መዝናኛ ነው។",
        
        # Food (Class 8)
        "እንጀራ እና ወጥ የእለት ተእለት ምግባችን ነው។",
        "የኢትዮጵያ ቡና በዓለም አንደኛ ነው።",
        "ባህላዊ ምግብ ቤቶች በከተማዎች እየተከፈቱ ነው។",
        
        # Travel/Tourism (Class 9)
        "ላሊበላ አብያተ ክርስቲያናት መጎብኘት ያምራል።",
        "የሲሚያን ተራሮች ተፈጥሮ ድንቅ ነው።",
        "የቱሪዝም ዘርፍ ለሀገር ኢኮኖሚ አስፈላጊ ነው።"
    ]
    
    # Create larger dataset by combining and varying
    extended_texts = []
    labels = []
    
    # Generate 1000 samples
    for i in range(1000):
        # Pick base text and class
        base_idx = i % len(amharic_texts)
        base_text = amharic_texts[base_idx]
        base_class = base_idx // 3  # 3 samples per class
        
        # Add variations
        variations = [
            f"{base_text}",
            f"{base_text} እንዲሁም በጣም ጠቃሚ ነው።",
            f"በእርግጥ {base_text}",
            f"{base_text} ይህም በተመሳሳይ ጠቃሚ ነው።",
            f"እንደሚታወቀው {base_text}"
        ]
        
        final_text = random.choice(variations)
        extended_texts.append(final_text)
        labels.append(base_class)
    
    # Create practice test set (no labels)
    test_df = pd.DataFrame({
        'id': range(1, 1001),
        'text': extended_texts
    })
    
    # Create practice solution (with labels for validation)
    solution_df = pd.DataFrame({
        'id': range(1, 1001),
        'true_label': labels
    })
    
    # Save files
    test_df.to_csv('practice_test.csv', index=False)
    solution_df.to_csv('practice_solution.csv', index=False)
    
    print(f"✅ Created practice_test.csv: {len(test_df)} samples")
    print(f"✅ Created practice_solution.csv: ground truth labels")
    print(f"📊 Classes: 0-9 (10 different categories)")
    print(f"🎯 Ready for testing your 253M parameter model!")
    
    return test_df, solution_df

def evaluate_practice_results():
    """Evaluate your model results against practice solution."""
    
    print("\n🔍 PRACTICE EVALUATION GUIDE")
    print("=" * 40)
    print("After running your model:")
    print("1. Load your predictions: submission.csv")
    print("2. Load ground truth: practice_solution.csv") 
    print("3. Calculate accuracy and other metrics")
    print("4. Expected performance: 85th+ percentile")
    
    evaluation_script = '''
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Load results
predictions = pd.read_csv('your_submission.csv')
ground_truth = pd.read_csv('practice_solution.csv')

# Calculate metrics
accuracy = accuracy_score(ground_truth['true_label'], predictions['prediction'])
print(f"Accuracy: {accuracy:.3f}")
print(f"Expected for 85th percentile: >0.750")

# Detailed report
print(classification_report(ground_truth['true_label'], predictions['prediction']))
'''
    
    with open('evaluate_practice.py', 'w') as f:
        f.write(evaluation_script)
    
    print("✅ Created evaluate_practice.py for performance assessment")

if __name__ == "__main__":
    test_df, solution_df = create_practice_test_data()
    evaluate_practice_results()
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Run: python final_kaggle_inference.py (modify for practice_test.csv)")
    print(f"2. Evaluate: python evaluate_practice.py")
    print(f"3. Expected results: 85th+ percentile performance")
    print(f"4. Ready for real Kaggle competition! 🏆")