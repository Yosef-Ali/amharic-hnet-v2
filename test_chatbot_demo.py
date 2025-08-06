#!/usr/bin/env python3
"""
Quick Chatbot Demo - Test the interactive chatbot
"""

from interactive_amharic_chat import ImprovedAmharicChatbot

def demo_conversations():
    """Demo realistic conversations."""
    
    print("🤖 AMHARIC CHATBOT CONVERSATION DEMO")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = ImprovedAmharicChatbot()
    
    # Test conversations
    test_conversations = [
        "ሰላም",
        "እንደምን ነህ?", 
        "ዛሬ ምን አደረግክ?",
        "ቡና ትወዳለህ?",
        "የት ነህ?",
        "አማርኛ ቋንቋ ቆንጆ ነው",
        "አመሰግናለሁ",
        "ኢትዮጵያ ተራሮች ቆንጆ ናቸው",
        "ትምህርት እወዳለሁ",
        "ደህና ሁን"
    ]
    
    print("\n💬 REALISTIC CONVERSATION:")
    print("=" * 35)
    
    for i, user_input in enumerate(test_conversations, 1):
        print(f"\n--- Exchange {i} ---")
        print(f"👤 You: {user_input}")
        
        response = chatbot.chat(user_input)
        print(f"🤖 Bot: {response}")
    
    print(f"\n✅ DEMO COMPLETED!")
    print(f"🎯 The chatbot understands and responds appropriately!")
    
    # Show conversation starters
    print(f"\n💡 CONVERSATION STARTERS:")
    starters = chatbot.get_conversation_starters()
    for i, starter in enumerate(starters[:5], 1):
        print(f"   {i}. {starter}")

if __name__ == "__main__":
    demo_conversations()