#!/usr/bin/env python3
"""
Compact Enhanced H-Net Training
Optimized for faster training with smaller model and efficient data loading
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
import random

# Import our proper H-Net implementation
from src.models.proper_hnet_amharic import AmharicHNetMixer

class CompactAmharicDataset(Dataset):
    """
    Compact dataset with efficient loading
    """
    def __init__(self, corpus_path: str, max_length: int = 128, max_segments: int = 2000):
        self.max_length = max_length
        
        # Load and process corpus efficiently
        self.segments = self.load_and_process_corpus(corpus_path, max_segments)
        
        print(f"Created {len(self.segments)} training segments")
    
    def load_and_process_corpus(self, corpus_path: str, max_segments: int) -> List[str]:
        """Load and process corpus efficiently"""
        segments = []
        
        # Try to load the new large corpus
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'articles' in data:
                    articles = data['articles'][:100]  # Limit to first 100 articles for speed
                    
                    for article in articles:
                        if isinstance(article, dict):
                            text = article.get('content', article.get('text', ''))
                        else:
                            text = str(article)
                        
                        if text and len(text.strip()) > 50:
                            # Process text into segments
                            text = text.strip()
                            
                            # Split into sentences efficiently
                            sentences = []
                            current = ""
                            for char in text:
                                current += char
                                if char in ['።', '፣', '፤', '.', '!', '?']:
                                    if len(current.strip()) > 20:
                                        sentences.append(current.strip())
                                        current = ""
                            
                            if current.strip():
                                sentences.append(current.strip())
                            
                            # Add sentences as segments
                            for sentence in sentences:
                                if 30 <= len(sentence) <= self.max_length:
                                    segments.append(sentence)
                                    if len(segments) >= max_segments:
                                        break
                        
                        if len(segments) >= max_segments:
                            break
                
                print(f"Loaded {len(segments)} segments from large corpus")
                
            except Exception as e:
                print(f"Error loading large corpus: {e}")
        
        # Fallback if no segments loaded
        if not segments:
            segments = [
                "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ነች።",
                "አማርኛ ቋንቋ በኢትዮጵያ ውስጥ በስፋት የሚነገር ነው።",
                "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት።",
                "ቡና የኢትዮጵያ ህዝብ ተወዳጅ መጠጥ ነው።",
                "እንጀራ እና ወጥ የኢትዮጵያ ባህላዊ ምግቦች ናቸው።",
                "የኢትዮጵያ ታሪክ በጥንት ዘመን የጀመረ ነው።",
                "ኦሮሞ ህዝብ በኢትዮጵያ ውስጥ ብዙ ቁጥር ያላቸው ናቸው።",
                "አማራ ህዝብ የኢትዮጵያ ትልቅ ሕዝብ ናቸው።",
                "ትግራይ በሰሜን ኢትዮጵያ የሚገኝ ክልል ነው።",
                "የኢትዮጵያ ባንዲራ አረንጓዴ ቢጫ እና ቀይ ቀለሞች አሏት።"
            ] * 200  # Repeat for more training data
            
            print(f"Using {len(segments)} fallback segments")
        
        # Shuffle for better training
        random.shuffle(segments)
        return segments[:max_segments]
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        text = self.segments[idx]
        
        # Convert to UTF-8 bytes
        byte_seq = list(text.encode('utf-8'))
        
        # Pad or truncate
        if len(byte_seq) > self.max_length:
            byte_seq = byte_seq[:self.max_length]
        else:
            byte_seq.extend([0] * (self.max_length - len(byte_seq)))
        
        return torch.tensor(byte_seq, dtype=torch.long)

def compact_train_hnet(
    model: AmharicHNetMixer,
    train_loader: DataLoader,
    num_epochs: int = 15,
    learning_rate: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Compact efficient training
    """
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches: {len(train_loader)}")
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        boundary_count = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, input_ids in enumerate(pbar):
                input_ids = input_ids.to(device)
                
                # Prepare sequences
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                # Forward pass
                optimizer.zero_grad()
                logits, debug_info = model(inputs)
                
                # Calculate loss
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                boundary_count += debug_info['boundaries'].sum().item()
                
                # Update progress
                if batch_idx > 0:
                    avg_loss = total_loss / batch_idx
                    avg_boundaries = boundary_count / batch_idx
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'boundaries': f'{avg_boundaries:.1f}'
                    })
        
        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"outputs/compact_enhanced_epoch_{epoch+1}.pt"
            os.makedirs("outputs", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

def test_compact_generation(model: AmharicHNetMixer, device: str = 'cpu'):
    """Test the compact model generation"""
    model.eval()
    model = model.to(device)
    
    test_prompts = [
        ("ኢትዮጵያ", "Ethiopia"),
        ("አማርኛ", "Amharic"),
        ("አዲስ አበባ", "Addis Ababa"),
        ("ቡና", "coffee"),
        ("ባህል", "culture")
    ]
    
    print("\n" + "="*60)
    print("🧠 COMPACT H-NET GENERATION TEST")
    print("="*60)
    
    with torch.no_grad():
        for i, (prompt, meaning) in enumerate(test_prompts, 1):
            print(f"\n{i}. '{prompt}' ({meaning})")
            print("-" * 40)
            
            prompt_bytes = list(prompt.encode('utf-8'))
            input_ids = torch.tensor([prompt_bytes], dtype=torch.long).to(device)
            
            for temp in [0.7, 1.0]:
                generated = model.generate(
                    input_ids,
                    max_length=25,
                    temperature=temp,
                    top_k=40
                )
                
                try:
                    generated_bytes = generated[0].cpu().numpy().tolist()
                    full_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
                    new_part = full_text[len(prompt):]
                    
                    print(f"  T={temp}: '{full_text}'")
                    print(f"        New: '{new_part}'")
                    
                except Exception as e:
                    print(f"  T={temp}: Error - {e}")
    
    print("="*60)

def main():
    """Main compact training function"""
    print("🚀 COMPACT ENHANCED H-NET TRAINING")
    print("="*50)
    
    # Compact configuration
    batch_size = 8
    max_length = 128
    num_epochs = 15
    learning_rate = 1e-4
    
    # Load compact dataset
    corpus_path = "data/raw/supplemented_hnet_corpus_20250801_145849.json"
    dataset = CompactAmharicDataset(corpus_path, max_length=max_length, max_segments=2000)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize compact model
    print("Initializing compact H-Net model...")
    model = AmharicHNetMixer(
        vocab_size=256,
        d_model=384,    # Smaller model
        n_heads=6,      # Fewer heads
        n_backbone_layers=4,  # Fewer layers
        max_chunks=64   # Reasonable chunks
    )
    
    # Load previous weights if available
    checkpoint_path = "outputs/proper_hnet_final.pt"
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Load compatible weights
            model_dict = model.state_dict()
            compatible_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            if compatible_dict:
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                print(f"✅ Loaded {len(compatible_dict)} compatible weights")
            
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    
    # Test before training
    print("\n" + "="*40)
    print("BEFORE TRAINING")
    test_compact_generation(model)
    
    # Training
    print("\n" + "="*40)
    print("STARTING TRAINING")
    compact_train_hnet(model, train_loader, num_epochs, learning_rate)
    
    # Test after training
    print("\n" + "="*40)
    print("AFTER TRAINING")
    test_compact_generation(model)
    
    # Save final model
    final_path = "outputs/compact_enhanced_final.pt"
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Model saved: {final_path}")
    
    print("\n🎉 COMPACT ENHANCED TRAINING COMPLETED!")

if __name__ == "__main__":
    main()