#!/usr/bin/env python3
"""
PROPER H-NET TRAINING FOR AMHARIC
Uses the correct H-Net architecture following original Chinese implementation

This script demonstrates how TRUE H-Net should be trained for meaningful generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import numpy as np
from typing import List, Dict, Tuple
import os
from tqdm import tqdm

# Import our proper H-Net implementation
from src.models.proper_hnet_amharic import AmharicHNetMixer

class AmharicTextDataset(Dataset):
    """
    Dataset for Amharic text with proper UTF-8 byte encoding
    """
    def __init__(self, texts: List[str], max_length: int = 256):
        self.texts = texts
        self.max_length = max_length
        
        # Convert texts to byte sequences
        self.byte_sequences = []
        for text in texts:
            # Encode to UTF-8 bytes
            byte_seq = list(text.encode('utf-8'))
            
            # Pad or truncate
            if len(byte_seq) > max_length:
                byte_seq = byte_seq[:max_length]
            else:
                byte_seq.extend([0] * (max_length - len(byte_seq)))  # Pad with 0
            
            self.byte_sequences.append(byte_seq)
    
    def __len__(self):
        return len(self.byte_sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.byte_sequences[idx], dtype=torch.long)

def load_amharic_data(data_path: str = "data/raw") -> List[str]:
    """
    Load Amharic text data from corpus files
    """
    texts = []
    
    # Look for corpus files
    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            if filename.endswith('.json') and 'corpus' in filename:
                filepath = os.path.join(data_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        # Extract text from different formats
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    text = item.get('text', item.get('content', ''))
                                else:
                                    text = str(item)
                                
                                if text and len(text.strip()) > 10:  # Filter short texts
                                    texts.append(text.strip())
                        
                        elif isinstance(data, dict):
                            for key, value in data.items():
                                if isinstance(value, str) and len(value.strip()) > 10:
                                    texts.append(value.strip())
                
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
    
    # Fallback demo data if no corpus found
    if not texts:
        print("No corpus data found, using demo Amharic texts...")
        texts = [
            "ይፈልጋሉ ውሃ መጠጣት እርሳቸው።",
            "አማርኛ ቋንቋ በጣም ውብ ነው።",
            "እኔ ኢትዮጵያዊ ነኝ እና አማርኛ እናገራለሁ។",
            "ዛሬ ሰማይ ሰላማዊ እና ውብ ነው።",
            "ቡና እና ዳቦ በጣም ጣፋጭ ናቸው።",
            "ልጆቹ በፍጥነት እየሮጡ ናቸው።",
            "መጽሐፍ ማንበብ ለአእምሮ ጥሩ ነው።",
            "ሀገሬ ኢትዮጵያ ውብ ናት።"
        ] * 10  # Repeat for more training data
    
    print(f"Loaded {len(texts)} Amharic texts for training")
    return texts

def train_proper_hnet(
    model: AmharicHNetMixer,
    train_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the proper H-Net model
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, input_ids in enumerate(pbar):
                input_ids = input_ids.to(device)
                
                # Prepare input and targets for language modeling
                # Input: first n-1 tokens, Target: tokens 1 to n
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                # Forward pass through H-Net
                optimizer.zero_grad()
                logits, debug_info = model(inputs)
                
                # Calculate loss
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                avg_loss = total_loss / num_batches
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'chunks': f'{debug_info["boundaries"].sum().item():.0f}'
                })
        
        avg_epoch_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"outputs/proper_hnet_epoch_{epoch+1}.pt"
            os.makedirs("outputs", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

def test_generation(model: AmharicHNetMixer, device: str = 'cpu'):
    """
    Test the trained model's generation capabilities
    """
    model.eval()
    model = model.to(device)
    
    # Test prompts in Amharic
    test_prompts = [
        "ይፈልጋሉ",  # "they want"
        "አማርኛ",    # "Amharic"
        "ኢትዮጵያ",   # "Ethiopia"
        "ቡና"       # "coffee"
    ]
    
    print("\n" + "="*60)
    print("TESTING PROPER H-NET GENERATION")
    print("="*60)
    
    with torch.no_grad():
        for prompt in test_prompts:
            # Convert prompt to bytes
            prompt_bytes = list(prompt.encode('utf-8'))
            input_ids = torch.tensor([prompt_bytes], dtype=torch.long).to(device)
            
            print(f"\nPrompt: '{prompt}'")
            print(f"Input bytes: {prompt_bytes}")
            
            # Generate continuation
            generated = model.generate(
                input_ids,
                max_length=20,
                temperature=0.8,
                top_k=40
            )
            
            # Convert back to text
            generated_bytes = generated[0].cpu().numpy().tolist()
            
            try:
                # Find where generation starts (after original prompt)
                prompt_len = len(prompt_bytes)
                new_bytes = generated_bytes[prompt_len:]
                
                # Convert bytes back to text
                generated_text = bytes(new_bytes).decode('utf-8', errors='ignore')
                full_text = prompt + generated_text
                
                print(f"Generated: '{full_text}'")
                print(f"New part: '{generated_text}'")
                
            except Exception as e:
                print(f"Error decoding: {e}")
                print(f"Generated bytes: {generated_bytes}")
    
    print("="*60)

def main():
    """
    Main training function for proper H-Net
    """
    print("🔥 PROPER H-NET TRAINING FOR AMHARIC")
    print("Following TRUE H-Net architecture from Chinese implementation")
    print("="*70)
    
    # Configuration
    batch_size = 8
    max_length = 128
    num_epochs = 20
    learning_rate = 1e-4
    
    # Load data
    print("Loading Amharic data...")
    texts = load_amharic_data()
    
    # Create dataset and dataloader
    dataset = AmharicTextDataset(texts, max_length=max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize proper H-Net model
    print("Initializing proper H-Net model...")
    model = AmharicHNetMixer(
        vocab_size=256,  # Byte-level
        d_model=512,
        n_heads=8,
        n_backbone_layers=6,
        max_chunks=64
    )
    
    # Test generation before training
    print("\n" + "="*50)
    print("GENERATION BEFORE TRAINING (should be random)")
    test_generation(model)
    
    # Train the model
    print("\n" + "="*50)
    print("STARTING TRAINING")
    train_proper_hnet(model, train_loader, num_epochs, learning_rate)
    
    # Test generation after training
    print("\n" + "="*50)
    print("GENERATION AFTER TRAINING (should be meaningful)")
    test_generation(model)
    
    # Save final model
    final_path = "outputs/proper_hnet_final.pt"
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved: {final_path}")
    
    print("\n🎯 PROPER H-NET TRAINING COMPLETED!")
    print("This model follows the TRUE H-Net architecture and should generate meaningful Amharic text!")

if __name__ == "__main__":
    main()