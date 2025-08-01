#!/usr/bin/env python3
"""
Compact training script that saves space and focuses on getting a working model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.append('src')

from models.hnet_amharic import AmharicHNet
from preprocessing.prepare_amharic import AmharicPreprocessor

class CompactAmharicDataset(Dataset):
    """Minimal dataset for Amharic training."""
    
    def __init__(self, texts, preprocessor, max_length=128):
        self.sequences = []
        for text in texts:
            cleaned = preprocessor.clean_text(text)
            if len(cleaned) > 20:
                byte_seq = preprocessor.extract_byte_sequences(cleaned, max_length)
                if len(byte_seq) >= 10:
                    self.sequences.append(byte_seq)
        print(f"Prepared {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) <= 1:
            input_seq, target_seq = [0], [0]
        else:
            input_seq, target_seq = seq[:-1], seq[1:]
        
        # Pad to 128
        while len(input_seq) < 128:
            input_seq.append(0)
        while len(target_seq) < 128:
            target_seq.append(0)
            
        return {
            'input_ids': torch.tensor(input_seq[:128], dtype=torch.long),
            'target_ids': torch.tensor(target_seq[:128], dtype=torch.long)
        }

def quick_train():
    """Quick training to get below loss 2.0."""
    
    print("Loading texts...")
    texts = []
    with open('data/processed/amharic_texts.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} texts")
    
    # Use CPU to avoid MPS disk space issues
    device = torch.device('cpu')
    
    # Smaller model for space efficiency
    model = AmharicHNet(
        d_model=256,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_main_layers=4,
        n_heads=4,
        compression_ratio=4.0,
        vocab_size=256
    ).to(device)
    
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    preprocessor = AmharicPreprocessor()
    dataset = CompactAmharicDataset(texts, preprocessor)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print("\nStarting compact training...")
    start_time = time.time()
    
    # Train for 5 epochs
    for epoch in range(5):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            # Forward pass
            logits, _ = model(input_ids, target_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
        
        # Test generation every epoch
        if avg_loss < 2.5:  # Only test when loss is reasonable
            print("Testing generation...")
            model.eval()
            with torch.no_grad():
                # Test with simple prompt
                prompt = "ኢትዮጵያ"
                byte_seq = preprocessor.extract_byte_sequences(prompt, 50)
                input_ids = torch.tensor([byte_seq], dtype=torch.long).to(device)
                
                try:
                    generated = model.generate(input_ids, max_length=20, temperature=0.8)
                    gen_bytes = generated[0].cpu().numpy().tolist()
                    gen_text = preprocessor.decode_byte_sequence(gen_bytes)
                    print(f"  Generated: {gen_text}")
                    
                    # Check quality
                    amharic_chars = sum(1 for c in gen_text if 0x1200 <= ord(c) <= 0x137F)
                    print(f"  Amharic chars: {amharic_chars}/{len(gen_text)}")
                except Exception as e:
                    print(f"  Generation error: {e}")
        
        # Early stopping if loss is good
        if avg_loss < 1.5:
            print("Reached target loss!")
            break
    
    # Save final model compactly
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'final_loss': avg_loss,
        'model_config': {
            'd_model': 256,
            'n_encoder_layers': 2,
            'n_decoder_layers': 2,
            'n_main_layers': 4,
            'n_heads': 4,
            'compression_ratio': 4.0
        }
    }
    
    os.makedirs('outputs/compact', exist_ok=True)
    torch.save(final_checkpoint, 'outputs/compact/final_model.pt')
    
    print(f"\nTraining completed in {time.time() - start_time:.1f}s")
    print(f"Final loss: {avg_loss:.4f}")
    print("Model saved to outputs/compact/final_model.pt")
    
    return avg_loss

if __name__ == "__main__":
    final_loss = quick_train()
    
    if final_loss < 2.0:
        print("\n✅ SUCCESS: Model achieved loss < 2.0")
    else:
        print(f"\n⚠️  Model loss {final_loss:.4f} still above 2.0, but should generate better text")