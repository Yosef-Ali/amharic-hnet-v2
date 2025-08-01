#!/usr/bin/env python3
"""
Enhanced H-Net Training with Large Amharic Corpus
Uses the 500+ article corpus for significantly improved generation quality
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

class EnhancedAmharicDataset(Dataset):
    """
    Enhanced dataset using the large Amharic corpus
    """
    def __init__(self, corpus_path: str, max_length: int = 256, min_length: int = 32):
        self.max_length = max_length
        self.min_length = min_length
        
        # Load the large corpus
        self.texts = self.load_large_corpus(corpus_path)
        
        # Process texts into training segments
        self.segments = self.create_training_segments()
        
        print(f"Loaded {len(self.texts)} articles, created {len(self.segments)} training segments")
    
    def load_large_corpus(self, corpus_path: str) -> List[str]:
        """Load the comprehensive Amharic corpus"""
        texts = []
        
        # Try to load the new large corpus
        if os.path.exists(corpus_path):
            try:
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if 'articles' in data:
                    for article in data['articles']:
                        if isinstance(article, dict):
                            text = article.get('content', article.get('text', ''))
                        else:
                            text = str(article)
                        
                        if text and len(text.strip()) > 50:
                            # Clean text
                            text = text.strip()
                            # Remove excessive whitespace
                            text = ' '.join(text.split())
                            texts.append(text)
                
                print(f"Loaded {len(texts)} articles from large corpus")
                return texts
                
            except Exception as e:
                print(f"Error loading large corpus: {e}")
        
        # Fallback to smaller corpus files
        data_dir = "data/raw"
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.endswith('.json') and 'corpus' in filename:
                    filepath = os.path.join(data_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    text = item.get('text', item.get('content', ''))
                                else:
                                    text = str(item)
                                
                                if text and len(text.strip()) > 50:
                                    texts.append(text.strip())
                        
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        print(f"Total texts loaded: {len(texts)}")
        return texts
    
    def create_training_segments(self) -> List[str]:
        """Create training segments from texts"""
        segments = []
        
        for text in self.texts:
            # Split text into sentences
            sentences = []
            current_sentence = ""
            
            for char in text:
                current_sentence += char
                # End sentence on Amharic punctuation
                if char in ['።', '፣', '፤', '፥', '፦', '፧', '፨']:
                    if len(current_sentence.strip()) > 10:
                        sentences.append(current_sentence.strip())
                    current_sentence = ""
            
            # Add remaining text
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            
            # Create segments from sentences
            for sentence in sentences:
                if len(sentence) >= self.min_length:
                    # Create overlapping segments for more training data
                    if len(sentence) <= self.max_length:
                        segments.append(sentence)
                    else:
                        # Split long sentences into overlapping segments
                        step = self.max_length // 2
                        for i in range(0, len(sentence) - self.min_length, step):
                            segment = sentence[i:i + self.max_length]
                            if len(segment) >= self.min_length:
                                segments.append(segment)
        
        # Shuffle segments for better training
        random.shuffle(segments)
        return segments
    
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

def enhanced_train_hnet(
    model: AmharicHNetMixer,
    train_loader: DataLoader,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Enhanced training with larger dataset and better optimization
    """
    model = model.to(device)
    
    # Enhanced optimizer with learning rate scheduling
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"Enhanced training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training batches per epoch: {len(train_loader)}")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        boundary_stats = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, input_ids in enumerate(pbar):
                input_ids = input_ids.to(device)
                
                # Prepare input and targets
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
                
                # Track boundary statistics
                boundaries = debug_info['boundaries']
                avg_boundaries = boundaries.mean().item()
                boundary_stats.append(avg_boundaries)
                
                # Update progress
                if batch_idx > 0:
                    avg_loss = total_loss / batch_idx
                    avg_boundaries = np.mean(boundary_stats[-10:])  # Last 10 batches
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'boundaries': f'{avg_boundaries:.3f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
                    })
        
        avg_epoch_loss = total_loss / len(train_loader)
        avg_boundaries = np.mean(boundary_stats)
        
        print(f"Epoch {epoch+1}: Loss={avg_epoch_loss:.4f}, Boundaries={avg_boundaries:.3f}")
        
        # Learning rate scheduling
        scheduler.step(avg_epoch_loss)
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = "outputs/enhanced_hnet_best.pt"
            os.makedirs("outputs", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'boundary_stats': avg_boundaries
            }, best_path)
            print(f"✅ New best model saved: {best_path}")
        
        # Regular checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"outputs/enhanced_hnet_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

def test_enhanced_generation(model: AmharicHNetMixer, device: str = 'cpu'):
    """
    Test the enhanced model with comprehensive evaluation
    """
    model.eval()
    model = model.to(device)
    
    # Enhanced test prompts covering different domains
    test_prompts = [
        ("ኢትዮጵያ", "Ethiopia"),
        ("አማርኛ ቋንቋ", "Amharic language"),
        ("የኢትዮጵያ ታሪክ", "Ethiopian history"),
        ("ቡና ማብሰያ", "coffee brewing"),
        ("እስክንድር ነጋ", "Eskinder Nega"),
        ("አዲስ አበባ", "Addis Ababa"),
        ("የትምህርት ሥርዓት", "education system"),
        ("ባህላዊ ምግብ", "traditional food"),
        ("ኦሮሞ ህዝብ", "Oromo people"),
        ("መጽሐፍ ቅዱስ", "Holy Bible")
    ]
    
    print("\n" + "="*80)
    print("🧠 ENHANCED H-NET GENERATION EVALUATION")
    print("="*80)
    
    quality_scores = []
    
    with torch.no_grad():
        for i, (prompt, meaning) in enumerate(test_prompts, 1):
            print(f"\n{i}. Testing: '{prompt}' ({meaning})")
            print("-" * 60)
            
            # Convert to bytes
            prompt_bytes = list(prompt.encode('utf-8'))
            input_ids = torch.tensor([prompt_bytes], dtype=torch.long).to(device)
            
            # Test different temperatures
            for temp in [0.5, 0.8, 1.2]:
                generated = model.generate(
                    input_ids,
                    max_length=40,
                    temperature=temp,
                    top_k=50
                )
                
                try:
                    generated_bytes = generated[0].cpu().numpy().tolist()
                    full_text = bytes(generated_bytes).decode('utf-8', errors='ignore')
                    new_part = full_text[len(prompt):]
                    
                    # Basic quality assessment
                    amharic_chars = sum(1 for c in new_part if '\u1200' <= c <= '\u137F')
                    total_chars = len([c for c in new_part if c.strip()])
                    quality = amharic_chars / max(total_chars, 1) if total_chars > 0 else 0
                    
                    print(f"  T={temp}: '{full_text}'")
                    print(f"         Quality: {quality:.2f} ({amharic_chars}/{total_chars} Amharic)")
                    
                    if temp == 0.8:  # Use middle temperature for scoring
                        quality_scores.append(quality)
                        
                except Exception as e:
                    print(f"  T={temp}: Decode error - {e}")
                    if temp == 0.8:
                        quality_scores.append(0.0)
    
    # Overall quality assessment
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0
    print(f"\n🎯 OVERALL QUALITY SCORE: {avg_quality:.3f}")
    print(f"Generated meaningful Amharic in {sum(1 for q in quality_scores if q > 0.5)}/{len(quality_scores)} tests")
    print("="*80)

def main():
    """
    Main enhanced training function
    """
    print("🚀 ENHANCED H-NET TRAINING WITH LARGE AMHARIC CORPUS")
    print("="*80)
    
    # Configuration for enhanced training
    batch_size = 16  # Larger batch size
    max_length = 200  # Longer sequences
    num_epochs = 40   # More epochs
    learning_rate = 5e-5  # Lower learning rate for stability
    
    # Load enhanced dataset
    corpus_path = "data/raw/supplemented_hnet_corpus_20250801_145849.json"
    print(f"Loading enhanced dataset from: {corpus_path}")
    
    dataset = EnhancedAmharicDataset(corpus_path, max_length=max_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize enhanced model
    print("Initializing enhanced H-Net model...")
    model = AmharicHNetMixer(
        vocab_size=256,
        d_model=768,  # Larger model
        n_heads=12,   # More attention heads
        n_backbone_layers=8,  # Deeper backbone
        max_chunks=128  # More chunks
    )
    
    # Load previous checkpoint if available
    checkpoint_path = "outputs/proper_hnet_final.pt"
    if os.path.exists(checkpoint_path):
        try:
            # Load only compatible weights
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Try to load compatible weights
            model_dict = model.state_dict()
            compatible_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and v.shape == model_dict[k].shape}
            
            if compatible_dict:
                model_dict.update(compatible_dict)
                model.load_state_dict(model_dict)
                print(f"✅ Loaded {len(compatible_dict)} compatible weights from previous model")
            
        except Exception as e:
            print(f"Could not load previous checkpoint: {e}")
            print("Starting with fresh weights...")
    
    # Test before training
    print("\n" + "="*60)
    print("GENERATION BEFORE ENHANCED TRAINING")
    test_enhanced_generation(model)
    
    # Enhanced training
    print("\n" + "="*60)
    print("STARTING ENHANCED TRAINING")
    enhanced_train_hnet(model, train_loader, num_epochs, learning_rate)
    
    # Test after training
    print("\n" + "="*60)
    print("GENERATION AFTER ENHANCED TRAINING")
    test_enhanced_generation(model)
    
    # Save final enhanced model
    final_path = "outputs/enhanced_hnet_final.pt"
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Enhanced model saved: {final_path}")
    
    print("\n🎉 ENHANCED H-NET TRAINING COMPLETED!")
    print("Model trained on 500+ article Amharic corpus with improved architecture!")

if __name__ == "__main__":
    main()