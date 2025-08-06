#!/usr/bin/env python3
"""
Simple Amharic Chat - Easy to use chatbot interface
"""

from interactive_amharic_chat import ImprovedAmharicChatbot

def simple_chat_session():
    """Simple chat session - easy to use."""
    
    print("🤖 AMHARIC CHATBOT - SIMPLE CHAT")
    print("=" * 40)
    print("😊 Say something in Amharic!")
    print("💡 Try: ሰላም, እንደምን ነህ?, ቡና ትወዳለህ?")
    print("🛑 Type 'stop' to exit")
    print("=" * 40)
    
    # Initialize chatbot
    chatbot = ImprovedAmharicChatbot()
    
    # Start conversation
    print("🤖: ሰላም! እንደምን ነዎት? ምን እንወያይ?")
    
    while True:
        try:
            # Get input
            user_input = input("\n👤: ")
            
            # Check exit
            if user_input.lower() in ['stop', 'exit', 'quit']:
                print("🤖: ደህና ይሁኑ! (Goodbye!)")
                break
            
            # Get response
            if user_input.strip():
                response = chatbot.chat(user_input)
                print(f"🤖: {response}")
            else:
                print("🤖: እባክዎ ይናገሩ!")
                
        except KeyboardInterrupt:
            print("\n🤖: ደህና ይሁኑ!")
            break

if __name__ == "__main__":
    simple_chat_session()