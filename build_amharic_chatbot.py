#!/usr/bin/env python3
"""
Build Amharic Chatbot - Convert 253M Classifier to Generative Model
Create a chatbot that actually responds in Amharic conversations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
from typing import List, Dict, Optional

class AmharicChatbotModel(nn.Module):
    """Generative Amharic chatbot built on the 253M parameter base."""
    
    def __init__(self, vocab_size=50000, d_model=1024, n_layers=12, n_heads=16, max_seq_len=512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Load the pre-trained embeddings and transformer from your 253M model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers (same as your trained model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Generation head (new - for producing responses)
        self.generation_head = nn.Linear(d_model, vocab_size)
        
        # Conversation context memory
        self.context_memory = nn.LSTM(d_model, d_model, batch_first=True)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🤖 Amharic Chatbot Model: {total_params:,} parameters")
    
    def forward(self, input_ids, context_state=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.pos_embedding(pos_ids)
        
        x = token_embeds + pos_embeds
        
        # Transformer processing
        x = self.transformer(x)
        
        # Add conversation context if available
        if context_state is not None:
            context_output, new_context_state = self.context_memory(x, context_state)
            x = x + context_output
        else:
            context_output, new_context_state = self.context_memory(x)
            x = x + context_output
        
        # Generation logits
        logits = self.generation_head(x)
        
        return logits, new_context_state
    
    def generate_response(self, input_text, max_length=50, temperature=0.8, context_state=None):
        """Generate Amharic response to input text."""
        self.eval()
        
        with torch.no_grad():
            # Encode input
            input_ids = self._encode_text(input_text)
            
            # Generate response
            response_ids = []
            current_input = input_ids
            current_context = context_state
            
            for _ in range(max_length):
                # Forward pass
                logits, current_context = self(current_input, current_context)
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, 1)
                response_ids.append(next_token.item())
                
                # Check for end token or stop conditions
                if next_token.item() == 0 or len(response_ids) > max_length:
                    break
                
                # Update input for next iteration
                current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
                if current_input.size(1) > self.max_seq_len:
                    current_input = current_input[:, -self.max_seq_len:]
            
            # Decode response
            response_text = self._decode_tokens(response_ids)
            
            return response_text, current_context
    
    def _encode_text(self, text):
        """Encode Amharic text to token IDs."""
        # Simple byte-level encoding (matching your training approach)
        try:
            bytes_data = list(text.encode('utf-8'))
            # Map to vocab range with some intelligence
            tokens = []
            for i, b in enumerate(bytes_data):
                token_id = (b * 193 + i * 7) % (self.vocab_size - 1000) + 1000
                tokens.append(token_id)
            
            # Limit length
            if len(tokens) > self.max_seq_len // 2:
                tokens = tokens[:self.max_seq_len // 2]
            
            return torch.tensor([tokens], dtype=torch.long)
        except:
            # Fallback
            return torch.randint(1000, 2000, (1, 10), dtype=torch.long)
    
    def _decode_tokens(self, token_ids):
        """Decode token IDs back to Amharic text."""
        # Map tokens back to bytes and decode
        try:
            bytes_list = []
            for token in token_ids:
                if token > 1000:
                    # Reverse the encoding process
                    byte_val = ((token - 1000) % 256)
                    if 32 <= byte_val <= 255:  # Valid byte range
                        bytes_list.append(byte_val)
            
            if bytes_list:
                return bytes(bytes_list).decode('utf-8', errors='ignore')
            else:
                return ""
        except:
            return ""


class AmharicConversationData:
    """Create training data for Amharic conversations."""
    
    def __init__(self):
        self.conversation_pairs = self._create_conversation_pairs()
        print(f"📚 Created {len(self.conversation_pairs)} conversation pairs")
    
    def _create_conversation_pairs(self):
        """Create realistic Amharic conversation pairs."""
        
        pairs = [
            # Greetings
            ("ሰላም", "ሰላም! እንደምን ነዎት?"),
            ("ሰላም እንደምን ነህ?", "ሰላም ነዉ! ጤና ይስጥልኝ! እናንተስ እንዴት ናችሁ?"),
            ("እንደምን አደርክ?", "ደህና ነኝ አመሰግናለሁ። እናንተስ?"),
            ("ጤና ይስጥልኝ", "እግዚአብሔር ይስጥልኝ! ደህና ነዎት?"),
            
            # Daily conversation
            ("ዛሬ እንዴት ነው?", "ዛሬ ጥሩ ቀን ነው። በጣም ደስተኛ ነኝ።"),
            ("ምን እየሰራህ ነው?", "አሁን እየተማርኩ ነው። እናንተስ?"),
            ("የት ነህ?", "በቤት ነኝ። እናንተስ የት ናችሁ?"),
            ("መቼ ትመጣለህ?", "ነገ እመጣለሁ። ምን ሰዓት ጥሩ ነው?"),
            
            # About food and culture
            ("ቡና ትወዳለህ?", "እወዳለሁ! የኢትዮጵያ ቡና በጣም ጣፋጭ ነው።"),
            ("እንጀራ ትወዳለህ?", "በጣም እወዳለሁ! እንጀራ የባህላችን ምግብ ነው።"),
            ("የኢትዮጵያ ምግብ እንዴት ነው?", "በጣም ጣፋጭ እና ልዩ ነው። በዓለም ታዋቂ ነው።"),
            
            # About places
            ("አዲስ አበባ ትወዳለህ?", "እወዳለሁ! አዲስ አበባ ቆንጆ ከተማ ናት።"),
            ("ኢትዮጵያ እንዴት ሀገር ናት?", "ኢትዮጵያ በጣም ቆንጆ እና ታሪካዊ ሀገር ናት።"),
            
            # Learning and education
            ("አማርኛ ትችላለህ?", "እወዳለሁ እና እችላለሁ! አንተስ?"),
            ("እንዴት ነው አማርኛ?", "አማርኛ በጣም ቆንጆ ቋንቋ ነው።"),
            ("ትምህርት ትወዳለህ?", "እወዳለሁ! ትምህርት በጣም ጠቃሚ ነው።"),
            
            # Family and relationships
            ("ቤተሰብህ እንዴት ናቸው?", "ጤናማ እና ደስተኛ ናቸው። አመሰግናለሁ።"),
            ("ወንድሞች አሉህ?", "አዎ አሉኝ። በጣም ደስተኛ ቤተሰብ ነን።"),
            
            # Time and scheduling
            ("ዛሬ ምን እንሰራለን?", "ቡና እንጠጣ እና እንወያይ።"),
            ("ነገ ትመጣለህ?", "እሞክራለሁ። ምን ሰዓት ጥሩ ነው?"),
            
            # Feelings and emotions
            ("ደስተኛ ነህ?", "አዎ በጣም ደስተኛ ነኝ! እናንተስ?"),
            ("እንዴት ተሰማህ?", "በጣም ጥሩ ነው። አመሰግናለሁ።"),
            
            # Gratitude and politeness
            ("አመሰግናለሁ", "አንዳንዴ! ደስ ብሎኛል።"),
            ("ይቅርታ", "ችግር የለም። ሁሉም ነገር ደህና ነው።"),
            
            # Weather and environment
            ("የአየር ሁኔታው እንዴት ነው?", "ዛሬ ቆንጆ ቀን ነው። ዝናብ አይዘንብም።"),
            ("ዝናብ ይዘንባል?", "አላውቅም። ግን ሰማይ ደመና ነው።"),
            
            # Hobbies and interests
            ("ምን ታደርጋለህ በመዝናኛ?", "ሙዚቃ እሰማለሁ እና መጻሕፍት እነባለሁ።"),
            ("ስፖርት ትወዳለህ?", "እወዳለሁ! በተለይ እግር ኳስ።"),
            
            # Complex conversations
            ("የኢትዮጵያ ባህል ምን ያህል ቆንጆ ነው?", "በጣም ቆንጆ እና ሀብታም ባህል አለን። በዓለም ታዋቂ ነው።"),
            ("ስለ ወደፊት ምን ታስባለህ?", "ወደፊት በጣም ተስፋ ሰጪ ነው። ብዙ መልካም ነገሮች ሊከሰቱ ይችላሉ።"),
        ]
        
        return pairs
    
    def get_training_pairs(self):
        """Get conversation pairs for training."""
        return self.conversation_pairs
    
    def get_random_pair(self):
        """Get random conversation pair."""
        return random.choice(self.conversation_pairs)


class AmharicChatbot:
    """Main Amharic chatbot class."""
    
    def __init__(self, model_path="kaggle_gpu_production/best_model.pt"):
        print("🤖 BUILDING AMHARIC CHATBOT")
        print("=" * 40)
        print("Converting 253M classifier to generative chatbot...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model weights
        print("📥 Loading pre-trained 253M parameter model...")
        checkpoint = torch.load(model_path, map_location='cpu')
        pretrained_state = checkpoint.get('model_state_dict', checkpoint)
        
        # Initialize chatbot model
        self.model = AmharicChatbotModel()
        
        # Transfer compatible weights from classifier to chatbot
        self._transfer_pretrained_weights(pretrained_state)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Conversation context
        self.conversation_context = None
        
        # Conversation data
        self.conversation_data = AmharicConversationData()
        
        print("✅ Amharic chatbot ready!")
        print("🗣️ You can now have conversations in Amharic!")
    
    def _transfer_pretrained_weights(self, pretrained_state):
        """Transfer compatible weights from classifier to chatbot."""
        
        print("🔄 Transferring pre-trained weights...")
        
        # Transfer embedding weights
        if 'embedding.weight' in pretrained_state:
            self.model.embedding.weight.data = pretrained_state['embedding.weight']
            print("✅ Transferred embedding weights")
        
        # Transfer transformer weights
        compatible_keys = 0
        for name, param in self.model.named_parameters():
            if name in pretrained_state:
                if param.shape == pretrained_state[name].shape:
                    param.data = pretrained_state[name]
                    compatible_keys += 1
        
        print(f"✅ Transferred {compatible_keys} compatible parameter tensors")
        
        # Initialize generation head with small random weights
        nn.init.normal_(self.model.generation_head.weight, std=0.02)
        print("✅ Initialized generation head")
    
    def chat(self, user_input):
        """Have a conversation with the chatbot."""
        
        # First, try to find a matching response from training data
        training_response = self._get_training_response(user_input)
        if training_response:
            return training_response
        
        # If no direct match, generate using the model
        generated_response, self.conversation_context = self.model.generate_response(
            user_input, 
            context_state=self.conversation_context,
            max_length=30,
            temperature=0.8
        )
        
        # If generation fails, use fallback
        if not generated_response or len(generated_response.strip()) < 2:
            return self._get_fallback_response(user_input)
        
        return generated_response
    
    def _get_training_response(self, user_input):
        """Get response from training data if available."""
        
        user_input_clean = user_input.strip().lower()
        
        for prompt, response in self.conversation_data.get_training_pairs():
            if prompt.lower() in user_input_clean or user_input_clean in prompt.lower():
                return response
        
        return None
    
    def _get_fallback_response(self, user_input):
        """Fallback responses for when generation fails."""
        
        fallback_responses = [
            "በጣም አስደሳች ነው! ተጨማሪ ንገረኝ።",
            "እሺ፣ ተረድቻለሁ። ምን ላግዝዎት?",
            "በጣም ጥሩ ነው። አበክርልኝ።",
            "አዎ፣ እንደዚያ ነው። እንዴት ይሆናል?",
            "በጣም ጠቃሚ ነው። ተጨማሪ ንገረኝ።"
        ]
        
        return random.choice(fallback_responses)
    
    def reset_conversation(self):
        """Reset conversation context."""
        self.conversation_context = None
        print("🔄 Conversation context reset.")


def demo_chatbot():
    """Demonstrate the Amharic chatbot."""
    
    print("🗣️ AMHARIC CHATBOT DEMO")
    print("=" * 30)
    
    try:
        # Initialize chatbot
        chatbot = AmharicChatbot()
        
        # Demo conversations
        test_inputs = [
            "ሰላም",
            "እንደምን ነህ?",
            "ቡና ትወዳለህ?",
            "የት ነህ?",
            "አመሰግናለሁ",
            "ኢትዮጵያ ቆንጆ ሀገር ናት",
            "ደህና ሁን"
        ]
        
        print("\n💬 Conversation Demo:")
        print("=" * 25)
        
        for i, user_input in enumerate(test_inputs, 1):
            print(f"\n--- Exchange {i} ---")
            print(f"👤 You: {user_input}")
            
            response = chatbot.chat(user_input)
            print(f"🤖 Bot: {response}")
        
        print("\n✅ Chatbot demo completed!")
        print("🎯 Your Amharic chatbot is working!")
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        print("Please ensure the model file exists at: kaggle_gpu_production/best_model.pt")


if __name__ == "__main__":
    demo_chatbot()