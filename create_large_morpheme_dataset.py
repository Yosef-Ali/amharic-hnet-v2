#!/usr/bin/env python3
"""
Create Large Morpheme-Annotated Dataset for Better Text Generation
"""

import json
import os

def create_expanded_morpheme_dataset():
    """Create a comprehensive morpheme-annotated dataset with sentence continuations"""
    
    print("🏗️ Creating Large Morpheme-Annotated Dataset")
    print("=" * 60)
    
    # Expanded dataset with meaningful continuations
    morpheme_data = [
        # Verbs - Present tense
        {"word": "ይፈልጋሉ", "morphemes": ["ይ", "ፈልግ", "አል", "ው"], "meaning": "they want", 
         "continuation": "ውሃ መጠጣት", "full_sentence": "ይፈልጋሉ ውሃ መጠጣት"},
        
        {"word": "ይሰራሉ", "morphemes": ["ይ", "ሰር", "አል", "ው"], "meaning": "they work", 
         "continuation": "በፋብሪካ ውስጥ", "full_sentence": "ይሰራሉ በፋብሪካ ውስጥ"},
        
        {"word": "ትሄዳለች", "morphemes": ["ት", "ሄድ", "አለ", "ች"], "meaning": "she goes", 
         "continuation": "ወደ ቤተ መጽሐፍት", "full_sentence": "ትሄዳለች ወደ ቤተ መጽሐፍት"},
        
        {"word": "ይመጣሉ", "morphemes": ["ይ", "መጣ", "አል", "ው"], "meaning": "they come", 
         "continuation": "ነገ ጠዋት", "full_sentence": "ይመጣሉ ነገ ጠዋት"},
        
        {"word": "ያስተምራል", "morphemes": ["ይ", "አስተምር", "አል"], "meaning": "he teaches", 
         "continuation": "አማርኛ ቋንቋ", "full_sentence": "ያስተምራል አማርኛ ቋንቋ"},
        
        # Verbs - Past tense
        {"word": "አልመጣችም", "morphemes": ["አል", "መጣ", "ች", "ም"], "meaning": "she did not come", 
         "continuation": "ዛሬ ምሽት", "full_sentence": "አልመጣችም ዛሬ ምሽት"},
        
        {"word": "ሰማሁ", "morphemes": ["ሰማ", "ሁ"], "meaning": "I heard", 
         "continuation": "ጥሩ ዜና", "full_sentence": "ሰማሁ ጥሩ ዜና"},
        
        {"word": "አየሁት", "morphemes": ["አየ", "ሁ", "ት"], "meaning": "I saw him", 
         "continuation": "በገበያ", "full_sentence": "አየሁት በገበያ"},
        
        # Prepositions and locations
        {"word": "በቤታችን", "morphemes": ["በ", "ቤት", "አችን"], "meaning": "in our house", 
         "continuation": "ብዙ መጽሐፍት አሉ", "full_sentence": "በቤታችን ብዙ መጽሐፍት አሉ"},
        
        {"word": "ወደቤት", "morphemes": ["ወደ", "ቤት"], "meaning": "to home", 
         "continuation": "በፍጥነት ሄዱ", "full_sentence": "ወደቤት በፍጥነት ሄዱ"},
        
        {"word": "ከኢትዮጵያ", "morphemes": ["ከ", "ኢትዮጵያ"], "meaning": "from Ethiopia", 
         "continuation": "መጣሁ", "full_sentence": "ከኢትዮጵያ መጣሁ"},
        
        # Nouns - Plural forms
        {"word": "ተማሪዎች", "morphemes": ["ተማሪ", "ዎች"], "meaning": "students", 
         "continuation": "ጠንክረው ይማራሉ", "full_sentence": "ተማሪዎች ጠንክረው ይማራሉ"},
        
        {"word": "መጽሐፍቶች", "morphemes": ["መጽሐፍ", "ቶች"], "meaning": "books", 
         "continuation": "በመደርደሪያ ላይ", "full_sentence": "መጽሐፍቶች በመደርደሪያ ላይ"},
        
        {"word": "ልጆቻችን", "morphemes": ["ልጆች", "አችን"], "meaning": "our children", 
         "continuation": "በሀገራቸው ይኮራሉ", "full_sentence": "ልጆቻችን በሀገራቸው ይኮራሉ"},
        
        # Adjectives and descriptors
        {"word": "ኢትዮጵያዊ", "morphemes": ["ኢትዮጵያ", "ዊ"], "meaning": "Ethiopian", 
         "continuation": "ባህል ውብ ነው", "full_sentence": "ኢትዮጵያዊ ባህል ውብ ነው"},
        
        {"word": "ሀገራዊ", "morphemes": ["ሀገር", "አዊ"], "meaning": "national", 
         "continuation": "ኩራት አለን", "full_sentence": "ሀገራዊ ኩራት አለን"},
        
        # Common phrases and expressions
        {"word": "እንዴት", "morphemes": ["እንዴት"], "meaning": "how", 
         "continuation": "ነህ?", "full_sentence": "እንዴት ነህ?"},
        
        {"word": "አላውቅም", "morphemes": ["አል", "አውቅ", "ም"], "meaning": "I don't know", 
         "continuation": "መልሱን", "full_sentence": "አላውቅም መልሱን"},
        
        {"word": "ልመጣ", "morphemes": ["ል", "መጣ"], "meaning": "let me come", 
         "continuation": "ወደ እናንተ", "full_sentence": "ልመጣ ወደ እናንተ"},
        
        # Professional and academic terms
        {"word": "መንግስት", "morphemes": ["መንግስት"], "meaning": "government", 
         "continuation": "ለህዝብ ያገለግላል", "full_sentence": "መንግስት ለህዝብ ያገለግላል"},
        
        {"word": "ዩኒቨርሲቲ", "morphemes": ["ዩኒቨርሲቲ"], "meaning": "university", 
         "continuation": "ተማሪዎች ይማራሉ", "full_sentence": "ዩኒቨርሲቲ ተማሪዎች ይማራሉ"},
        
        {"word": "አስተማሪ", "morphemes": ["አስተማሪ"], "meaning": "teacher", 
         "continuation": "ጥሩ ሰው ነው", "full_sentence": "አስተማሪ ጥሩ ሰው ነው"},
        
        # Family and relationships
        {"word": "እናቴ", "morphemes": ["እናት", "ኤ"], "meaning": "my mother", 
         "continuation": "ምግብ ታበስላለች", "full_sentence": "እናቴ ምግብ ታበስላለች"},
        
        {"word": "አባቴ", "morphemes": ["አባት", "ኤ"], "meaning": "my father", 
         "continuation": "በቢሮ ይሰራል", "full_sentence": "አባቴ በቢሮ ይሰራል"},
        
        {"word": "ወንድሜ", "morphemes": ["ወንድም", "ኤ"], "meaning": "my brother", 
         "continuation": "እኔን ይረዳኛል", "full_sentence": "ወንድሜ እኔን ይረዳኛል"},
        
        # Time expressions
        {"word": "ዛሬ", "morphemes": ["ዛሬ"], "meaning": "today", 
         "continuation": "ጥሩ ቀን ነው", "full_sentence": "ዛሬ ጥሩ ቀን ነው"},
        
        {"word": "ነገ", "morphemes": ["ነገ"], "meaning": "tomorrow", 
         "continuation": "ወደ ሥራ እሄዳለሁ", "full_sentence": "ነገ ወደ ሥራ እሄዳለሁ"},
        
        {"word": "ትናንትና", "morphemes": ["ትናንትና"], "meaning": "yesterday", 
         "continuation": "ዝናብ ነበር", "full_sentence": "ትናንትና ዝናብ ነበር"},
        
        # Actions and activities
        {"word": "አነብባለሁ", "morphemes": ["አ", "ነብብ", "አለሁ"], "meaning": "I read", 
         "continuation": "መጽሐፍ በየቀኑ", "full_sentence": "አነብባለሁ መጽሐፍ በየቀኑ"},
        
        {"word": "እፅፋለሁ", "morphemes": ["እ", "ፅፍ", "አለሁ"], "meaning": "I write", 
         "continuation": "ደብዳቤ ለወዳጄ", "full_sentence": "እፅፋለሁ ደብዳቤ ለወዳጄ"},
        
        {"word": "እጫወታለሁ", "morphemes": ["እ", "ጫወት", "አለሁ"], "meaning": "I play", 
         "continuation": "ኳስ ከጓደኞቼ ጋር", "full_sentence": "እጫወታለሁ ኳስ ከጓደኞቼ ጋር"},
        
        # Complex verbs with objects
        {"word": "አላውቀውም", "morphemes": ["አል", "አውቅ", "ው", "ም"], "meaning": "I don't know him", 
         "continuation": "በደንብ", "full_sentence": "አላውቀውም በደንብ"},
        
        {"word": "እመለሳለሁ", "morphemes": ["እ", "መለስ", "አለሁ"], "meaning": "I will return", 
         "continuation": "ከስራ በኋላ", "full_sentence": "እመለሳለሁ ከስራ በኋላ"},
        
        # Emotions and feelings
        {"word": "እወዳለሁ", "morphemes": ["እ", "ወድ", "አለሁ"], "meaning": "I love", 
         "continuation": "ሀገሬን በጣም", "full_sentence": "እወዳለሁ ሀገሬን በጣም"},
        
        {"word": "እሳቀቃለሁ", "morphemes": ["እ", "ሳቅ", "ቃለሁ"], "meaning": "I laugh", 
         "continuation": "በስቅስቅ ነገር", "full_sentence": "እሳቀቃለሁ በስቅስቅ ነገር"},
        
        # Questions and responses
        {"word": "ማንነህ", "morphemes": ["ማን", "ነህ"], "meaning": "who are you", 
         "continuation": "እባክህ?", "full_sentence": "ማንነህ እባክህ?"},
        
        {"word": "የትነህ", "morphemes": ["የት", "ነህ"], "meaning": "where are you", 
         "continuation": "አሁን?", "full_sentence": "የትነህ አሁን?"},
        
        # Colors and descriptions
        {"word": "ቀይ", "morphemes": ["ቀይ"], "meaning": "red", 
         "continuation": "አበባ ይመስላል", "full_sentence": "ቀይ አበባ ይመስላል"},
        
        {"word": "ሰማያዊ", "morphemes": ["ሰማይ", "አዊ"], "meaning": "blue", 
         "continuation": "ሰማይ ውብ ነው", "full_sentence": "ሰማያዊ ሰማይ ውብ ነው"},
        
        # Food and eating
        {"word": "እበላለሁ", "morphemes": ["እ", "በል", "አለሁ"], "meaning": "I eat", 
         "continuation": "ኢንጀራ በሶስት ጊዜ", "full_sentence": "እበላለሁ ኢንጀራ በሶስት ጊዜ"},
        
        {"word": "እጠጣለሁ", "morphemes": ["እ", "ጠጣ", "አለሁ"], "meaning": "I drink", 
         "continuation": "ሻይ ከጓደኞቼ ጋር", "full_sentence": "እጠጣለሁ ሻይ ከጓደኞቼ ጋር"},
        
        # Weather and nature
        {"word": "ዝናብ", "morphemes": ["ዝናብ"], "meaning": "rain", 
         "continuation": "ዛሬ ይዘንባል", "full_sentence": "ዝናብ ዛሬ ይዘንባል"},
        
        {"word": "ፀሀይ", "morphemes": ["ፀሀይ"], "meaning": "sun", 
         "continuation": "በሰማይ ላይ", "full_sentence": "ፀሀይ በሰማይ ላይ"},
        
        # Work and professions
        {"word": "ዶክተር", "morphemes": ["ዶክተር"], "meaning": "doctor", 
         "continuation": "ታማሚዎችን ያክማል", "full_sentence": "ዶክተር ታማሚዎችን ያክማል"},
        
        {"word": "ነርስ", "morphemes": ["ነርስ"], "meaning": "nurse", 
         "continuation": "በሆስፒታል ትሰራለች", "full_sentence": "ነርስ በሆስፒታል ትሰራለች"},
        
        # Technology and modern life
        {"word": "ኮምፒውተር", "morphemes": ["ኮምፒውተር"], "meaning": "computer", 
         "continuation": "በዘመናዊ ሕይወት", "full_sentence": "ኮምፒውተር በዘመናዊ ሕይወት"},
        
        {"word": "ስልክ", "morphemes": ["ስልክ"], "meaning": "phone", 
         "continuation": "ሁልጊዜ እይዛለሁ", "full_sentence": "ስልክ ሁልጊዜ እይዛለሁ"},
        
        # Transportation
        {"word": "መኪና", "morphemes": ["መኪና"], "meaning": "car", 
         "continuation": "ወደ ሥራ እገባለሁ", "full_sentence": "መኪና ወደ ሥራ እገባለሁ"},
        
        {"word": "አውቶቡስ", "morphemes": ["አውቶቡስ"], "meaning": "bus", 
         "continuation": "በየቀኑ እጓዛለሁ", "full_sentence": "አውቶቡስ በየቀኑ እጓዛለሁ"},
        
        # Sports and entertainment
        {"word": "እግርኳስ", "morphemes": ["እግር", "ኳስ"], "meaning": "football", 
         "continuation": "እወዳለሁ በጣም", "full_sentence": "እግርኳስ እወዳለሁ በጣም"},
        
        {"word": "ሙዚቃ", "morphemes": ["ሙዚቃ"], "meaning": "music", 
         "continuation": "እሰማለሁ በመዝናኛ", "full_sentence": "ሙዚቃ እሰማለሁ በመዝናኛ"}
    ]
    
    print(f"✅ Created {len(morpheme_data)} comprehensive examples")
    
    return morpheme_data

def convert_to_training_format(morpheme_data):
    """Convert morpheme data to training format with byte sequences"""
    
    print("🔄 Converting to Training Format")
    print("=" * 40)
    
    training_data = []
    
    for entry in morpheme_data:
        word = entry["word"]
        morphemes = entry["morphemes"]
        
        # Convert to bytes
        word_bytes = word.encode('utf-8')
        byte_sequence = [b for b in word_bytes]
        
        # Create boundary labels
        boundary_labels = [0] * len(byte_sequence)
        
        # Mark morpheme boundaries
        current_pos = 0
        for i, morpheme in enumerate(morphemes):
            if i == 0:
                boundary_labels[0] = 1  # Start of word
            else:
                # Find morpheme start in remaining text
                remaining_word = word[current_pos:]
                morpheme_start = remaining_word.find(morpheme)
                if morpheme_start >= 0:
                    # Convert character position to byte position
                    char_to_byte_pos = len(word[:current_pos + morpheme_start].encode('utf-8'))
                    if char_to_byte_pos < len(boundary_labels):
                        boundary_labels[char_to_byte_pos] = 1
            
            current_pos += len(morpheme)
        
        training_entry = {
            "word": word,
            "morphemes": morphemes,
            "meanings": entry.get("meaning", ""),
            "continuation": entry.get("continuation", ""),
            "full_sentence": entry.get("full_sentence", word),
            "byte_sequence": byte_sequence,
            "boundary_labels": boundary_labels,
            "length": len(byte_sequence)
        }
        
        training_data.append(training_entry)
    
    print(f"✅ Converted {len(training_data)} entries to training format")
    
    return training_data

def save_expanded_dataset(training_data):
    """Save the expanded dataset"""
    
    output_path = "data/expanded_boundary_training_data.json"
    os.makedirs("data", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved expanded dataset to {output_path}")
    print(f"📊 Dataset size: {len(training_data)} examples")
    
    # Print sample entries
    print(f"\n📝 Sample entries:")
    for i, entry in enumerate(training_data[:3]):
        print(f"   {i+1}. {entry['word']} → {entry['full_sentence']}")
    
    return output_path

if __name__ == "__main__":
    print("🏗️ Creating Large Morpheme-Annotated Dataset")
    print("=" * 70)
    
    # Create comprehensive dataset
    morpheme_data = create_expanded_morpheme_dataset()
    
    # Convert to training format
    training_data = convert_to_training_format(morpheme_data)
    
    # Save dataset
    dataset_path = save_expanded_dataset(training_data)
    
    print(f"\n🎯 DATASET SUMMARY:")
    print(f"   Total examples: {len(training_data)}")
    print(f"   File: {dataset_path}")
    print(f"   Ready for training!")
    
    print(f"\n🚀 Next step: Train with expanded dataset for meaningful continuations!")