#!/usr/bin/env python3
"""
Interactive Amharic Chatbot
Real-time conversation interface with improved generation
"""

import torch
import random
import re
from build_amharic_chatbot import AmharicChatbot

class ImprovedAmharicChatbot:
    """Improved Amharic chatbot with better response generation."""
    
    def __init__(self):
        print("🤖 IMPROVED AMHARIC CHATBOT")
        print("=" * 40)
        
        # Enhanced conversation database
        self.responses = self._build_enhanced_responses()
        self.context_memory = []
        
        print(f"📚 Loaded {len(self.responses)} response patterns")
        print("✅ Ready for Amharic conversations!")
    
    def _build_enhanced_responses(self):
        """Build comprehensive Amharic response database."""
        
        responses = {
            # Greetings and basic interactions
            "greetings": {
                "patterns": ["ሰላም", "hello", "hi", "ሃይ", "ጤና", "እንደምን"],
                "responses": [
                    "ሰላም! እንደምን ነዎት? በጣም ደስ ብሎኛል።",
                    "ሰላም ነው! ጤና ይስጥልኝ! እንዴት ናችሁ?",
                    "አሰላም ዓላይኩም! ደህና መጡ!",
                    "ሰላም! ደስ ብሎኛል ማግኘቶት።"
                ]
            },
            
            # How are you questions
            "wellbeing": {
                "patterns": ["እንደምን ነህ", "እንዴት ነው", "ደህና ነህ", "ሁኔታ", "አደር"],
                "responses": [
                    "በጣም ደህና ነኝ አመሰግናለሁ! እናንተስ እንዴት ናችሁ?",
                    "ጤና ይስጥልኝ፣ ደህና ነኝ። እርስዎስ?",
                    "በጣም ጥሩ ነኝ! ስለጠየቁኝ አመሰግናለሁ።",
                    "ደህና ነኝ፣ እግዚአብሔርን አመስግናለሁ። እናንተስ?"
                ]
            },
            
            # Food and culture
            "food_culture": {
                "patterns": ["ቡና", "እንጀራ", "ምግብ", "ወጥ", "coffee", "food"],
                "responses": [
                    "አህ! ቡና በጣም እወዳለሁ። የኢትዮጵያ ቡና በዓለም ታዋቂ ነው!",
                    "እንጀራ የባህላችን ትልቅ ክፍል ነው። በጣም ጣፋጭ ነው!",
                    "የኢትዮጵያ ምግብ በጣም ልዩ እና ጣፋጭ ነው። ምን ይወዳሉ?",
                    "ቡና ስነ ስርዓት በባህላችን በጣም ጠቃሚ ነው።"
                ]
            },
            
            # Location and geography
            "location": {
                "patterns": ["የት", "ኢትዮጵያ", "አዲስ አበባ", "ሀገር", "where"],
                "responses": [
                    "ኢትዮጵያ በጣም ቆንጆ ሀገር ናት! በአፍሪካ ቀንድ ትገኛለች።",
                    "አዲስ አበባ ቆንጆ ከተማ ናት። እኔም እወዳታለሁ!",
                    "የኢትዮጵያ ተፈጥሮ በጣም አስደናቂ ነው። ተራሮች፣ ሀይቆች...",
                    "በአዲስ አበባ ነኝ። እርስዎስ የት ናችሁ?"
                ]
            },
            
            # Language and communication
            "language": {
                "patterns": ["አማርኛ", "ቋንቋ", "amharic", "language", "መናገር"],
                "responses": [
                    "አማርኛ በጣም ቆንጆ ቋንቋ ነው! እወዳታለሁ።",
                    "አማርኛ የኢትዮጵያ ኦፊሴላዊ ቋንቋ ነው። በሚሊዮኖች ይነገራል።",
                    "በአማርኛ መናገር በጣም ያስደስተኛል! እርስዎም ይወዱታል?",
                    "የጌዝ ፊደል በጣም ቆንጆ ነው። ታሪካዊ ነው።"
                ]
            },
            
            # Gratitude and politeness
            "gratitude": {
                "patterns": ["አመሰግን", "thank", "ይቅርታ", "sorry", "በጣም"],
                "responses": [
                    "በፍቱ! ደስ ብሎኛል ማገዝ እንደቻልኩ።",
                    "አንዳንዴ! ሁሌም ደስተኛ ነኝ ለእርስዎ።",
                    "ችግር የለም! እኛ በጎ ናቸው።",
                    "እናመሰግናለን! እርስዎም ደህና ይሁኑ።"
                ]
            },
            
            # Time and activities
            "activities": {
                "patterns": ["ዛሬ", "ነገ", "ምን እየሰራ", "today", "tomorrow", "doing"],
                "responses": [
                    "ዛሬ በጣም ጥሩ ቀን ነው! እንዴት እያሳለፉት ነው?",
                    "አሁን ከእርስዎ ጋር እየተነጋገርኩ ነው። በጣም ደስ ብሎኛል!",
                    "ነገ ምን ማድረግ ይፈልጋሉ? የሚያስደስት ነገር?",
                    "እንዴት ያሳልፋሉ ጊዜዎን? ስፖርት? ሙዚቃ?"
                ]
            },
            
            # Feelings and emotions
            "emotions": {
                "patterns": ["ደስተኛ", "ጥሩ", "መጥፎ", "sad", "happy", "ተሰማ"],
                "responses": [
                    "በጣም ደስ ብሎኛል ደስተኛ እንደሆኑ!",
                    "ጥሩ ነው! ደስታ ማካፈል ይወዳል።",
                    "እኔም በጣም ደስተኛ ነኝ እርስዎን በማግኘቴ!",
                    "ይህ በጣም ያስደስተኛል! ተጨማሪ ንገረኝ።"
                ]
            },
            
            # Weather and environment
            "weather": {
                "patterns": ["የአየር ሁኔታ", "ዝናብ", "ፀሐይ", "weather", "rain"],
                "responses": [
                    "የአየር ሁኔታው ዛሬ እንዴት ነው እዚያ?",
                    "ዝናብ በኢትዮጵያ በጣም ጠቃሚ ነው። ለገበሬዎች።",
                    "ፀሐይ ስትወጣ በጣም ደስ ይላል። እርስዎም ይወዱታል?",
                    "የኢትዮጵያ አየር ንጹህ እና ቆንጆ ነው።"
                ]
            },
            
            # Learning and education
            "education": {
                "patterns": ["ትምህርት", "መማር", "ዩኒቨርሲቲ", "school", "learn"],
                "responses": [
                    "ትምህርት በጣም ጠቃሚ ነው! ሁሌም መማር ያስፈልጋል።",
                    "መማር የሕይወት ክፍል ነው። ምን መማር ይወዳሉ?",
                    "እኔም ሁሌም እመራለሁ፣ በተለይ ስለ አማርኛ ቋንቋ።",
                    "የትምህርት ጥናት በእድገት ይረዳል።"
                ]
            },
            
            # Default/fallback responses
            "default": {
                "patterns": [],
                "responses": [
                    "በጣም አስደሳች ነው! ተጨማሪ ንገረኝ።",
                    "እሺ፣ ተረድቻለሁ። ምን ላግዝዎት ችላለሁ?",
                    "በጣም ጥሩ ነው። እንዴት ላግዞት?",
                    "አዎ፣ እንደዚያ ነው። ተጨማሪ ንገረኝ።",
                    "በጣም ጠቃሚ ነው። ይህን ስለ ምን ያስባሉ?",
                    "ደስ ብሎኛል ይህን ሰምቶ! ተጨማሪ?",
                    "አንደምን ድንቅ ነው! እንዴት አየዎት?",
                    "በጣም ያስደስተኛል! ምን ማለት ነው?"
                ]
            }
        }
        
        return responses
    
    def chat(self, user_input):
        """Generate response to user input."""
        
        if not user_input or not user_input.strip():
            return "እባክዎ ይናገሩ! እንዴት ላግዞት?"
        
        # Clean input
        user_input = user_input.strip()
        user_input_lower = user_input.lower()
        
        # Add to context memory
        self.context_memory.append(user_input)
        if len(self.context_memory) > 5:  # Keep last 5 exchanges
            self.context_memory.pop(0)
        
        # Find best matching category
        best_category = "default"
        best_score = 0
        
        for category, data in self.responses.items():
            if category == "default":
                continue
            
            score = 0
            for pattern in data["patterns"]:
                if pattern.lower() in user_input_lower:
                    score += len(pattern)  # Longer matches get higher scores
            
            if score > best_score:
                best_score = score
                best_category = category
        
        # Get response from best category
        category_data = self.responses[best_category]
        response = random.choice(category_data["responses"])
        
        # Add some context awareness
        if len(self.context_memory) > 1:
            response = self._add_context_awareness(response, user_input)
        
        return response
    
    def _add_context_awareness(self, response, current_input):
        """Add context awareness to responses."""
        
        # Simple context modifications
        context_modifiers = [
            "እንደተነጋገርን፣ ",
            "እንደዚህ አድርጎ፣ ",
            "ያሳለፍነውን ተከትሎ፣ ",
            ""  # Sometimes no modifier
        ]
        
        if random.random() > 0.7:  # 30% chance to add context
            modifier = random.choice(context_modifiers)
            response = modifier + response
        
        return response
    
    def get_conversation_starters(self):
        """Get suggestions for starting conversations."""
        
        starters = [
            "ሰላም! እንደምን ነዎት?",
            "ዛሬ እንዴት አሳለፉ?",
            "ስለ ኢትዮጵያ ባህል ይወያዩ?",
            "የወዳጅነት ቋንቋ ነው አማርኛ!",
            "ቡና ጊዜ ነው!",
            "ስለ ምግብ እንነጋገር?",
            "የአየር ሁኔታው እንዴት ነው?",
            "ምን አዲስ ነገር አለ?"
        ]
        
        return starters
    
    def reset_conversation(self):
        """Reset conversation context."""
        self.context_memory = []
        return "🔄 አዲስ ውይይት ጀመርን! እንዴት ላግዞት?"


def interactive_chat():
    """Run interactive Amharic chat session."""
    
    print("🗣️ INTERACTIVE AMHARIC CHATBOT")
    print("=" * 50)
    print("እንኳን ደህና መጡ! (Welcome!)")
    print("Type 'quit' or 'በቃ' to exit")
    print("Type 'help' or 'እርዳታ' for suggestions")
    print("Type 'reset' or 'እንደገና' to start fresh")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = ImprovedAmharicChatbot()
    
    # Welcome message
    print("🤖 Bot: ሰላም! እንደምን ነዎት? ምን እንወያይ?")
    
    conversation_count = 0
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'በቃ', 'ውጣ']:
                print("🤖 Bot: ደህና ይሁኑ! (Goodbye!) በጣም ደስ ብሎኛል!")
                break
            
            # Check for help
            if user_input.lower() in ['help', 'እርዳታ']:
                starters = chatbot.get_conversation_starters()
                print("💡 Conversation suggestions:")
                for i, starter in enumerate(starters[:5], 1):
                    print(f"   {i}. {starter}")
                continue
            
            # Check for reset
            if user_input.lower() in ['reset', 'እንደገና']:
                response = chatbot.reset_conversation()
                print(f"🤖 Bot: {response}")
                conversation_count = 0
                continue
            
            # Empty input
            if not user_input:
                print("🤖 Bot: እባክዎ የሚፈልጉትን ይናገሩ!")
                continue
            
            # Generate response
            response = chatbot.chat(user_input)
            print(f"🤖 Bot: {response}")
            
            conversation_count += 1
            
            # Occasional encouragement
            if conversation_count % 5 == 0:
                encouragements = [
                    "በጣም ጥሩ ውይይት ነው!",
                    "አማርኛዎ በጣም ጥሩ ነው!",
                    "እንደዚህ ቀጥሉ!",
                    "በጣም ደስ ብሎኛል!"
                ]
                print(f"💫 {random.choice(encouragements)}")
        
        except KeyboardInterrupt:
            print("\n🤖 Bot: ደህና ይሁኑ! (Goodbye!)")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            print("🤖 Bot: ይቅርታ፣ ችግር ገጠመኝ። እንደገና ይሞክሩ።")


def main():
    """Main function."""
    
    print("🚀 AMHARIC CHATBOT - INTERACTIVE MODE")
    print("=" * 60)
    
    try:
        interactive_chat()
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        print("Please ensure all dependencies are installed.")


if __name__ == "__main__":
    main()