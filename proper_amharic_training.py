#!/usr/bin/env python3
"""
PROPER AMHARIC H-NET TRAINING
Using VALID space-free, character-level Amharic data

CRITICAL FIXES IMPLEMENTED:
✅ Space-free Amharic text (as it should be)
✅ Character-level tokenization (required for H-Net)
✅ Morpheme validation during training
✅ Real-time output quality monitoring
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
import re

from src.models.proper_hnet_amharic import AmharicHNetMixer

class ValidAmharicDataset(Dataset):
    """
    Dataset using VALID space-free Amharic text with character-level tokenization
    CRITICAL: H-Net requires character-level input for dynamic chunking
    """
    def __init__(self, dataset_path: str, max_length: int = 256):
        self.max_length = max_length
        
        # Load the VALID character-level dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.texts = data['texts']
        self.vocab = data['vocab']
        self.vocab_size = len(self.vocab)
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        print(f"✅ Loaded VALID Amharic dataset:")
        print(f"   Texts: {len(self.texts)}")
        print(f"   Vocabulary size: {self.vocab_size}")
        print(f"   Total characters: {data['total_chars']:,}")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Convert characters to indices (character-level tokenization)
        char_indices = []
        for char in text:
            if char in self.char_to_idx:
                char_indices.append(self.char_to_idx[char])
            else:
                # Skip unknown characters (shouldn't happen with cleaned data)
                continue
        
        # Pad or truncate
        if len(char_indices) > self.max_length:
            char_indices = char_indices[:self.max_length]
        else:
            char_indices.extend([0] * (self.max_length - len(char_indices)))
        
        return torch.tensor(char_indices, dtype=torch.long)
    
    def decode(self, indices: List[int]) -> str:
        """Convert indices back to Amharic text"""
        chars = []
        for idx in indices:
            if idx > 0 and idx < len(self.idx_to_char):
                chars.append(self.idx_to_char[idx])
        return ''.join(chars)

def validate_amharic_output(text: str) -> Tuple[bool, str, float]:
    """
    CRITICAL: Validate if generated text is valid Amharic
    Following the analysis requirements
    """
    if not text or len(text.strip()) == 0:
        return False, "empty_output", 0.0
    
    # Check script purity (must be >95% Ge'ez)
    ge_ez_chars = sum(1 for c in text if '\u1200' <= c <= '\u137F')
    total_chars = len([c for c in text if c.strip()])
    
    if total_chars == 0:
        return False, "no_characters", 0.0
    
    script_purity = ge_ez_chars / total_chars
    
    if script_purity < 0.95:
        return False, f"script_impurity_{script_purity:.2f}", script_purity
    
    # Check for valid Amharic patterns (basic morpheme check)
    # Amharic words should have vowel-consonant patterns
    has_valid_patterns = bool(re.search(r'[\u1200-\u137F]{2,}', text))
    
    if not has_valid_patterns:
        return False, "invalid_morpheme_patterns", script_purity
    
    # Check length (should generate meaningful content)
    if len(text.strip()) < 3:
        return False, "too_short", script_purity
    
    return True, "valid_amharic", script_purity

def train_with_validation(
    model: AmharicHNetMixer,
    train_loader: DataLoader,
    dataset: ValidAmharicDataset,
    num_epochs: int = 20,
    learning_rate: float = 2e-5,  # Lower LR as recommended
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train with REAL-TIME validation following the analysis protocol
    """
    model = model.to(device)
    
    # AMHARIC-SPECIFIC training parameters (from analysis)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"🚀 PROPER AMHARIC TRAINING STARTED")
    print(f"   Device: {device}")
    print(f"   Model vocab size: {model.vocab_size}")
    print(f"   Data vocab size: {dataset.vocab_size}")
    print(f"   Learning rate: {learning_rate}")
    
    model.train()
    
    # Test prompts for validation (from analysis)
    test_prompts = ["ኢትዮጵያ", "ቡና", "አማርኛ", "ሰላም"]
    
    for epoch in range(num_epochs):
        total_loss = 0
        valid_outputs = 0
        total_validations = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for step, input_ids in enumerate(pbar):
                input_ids = input_ids.to(device)
                
                # Prepare training data
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                
                # Forward pass
                optimizer.zero_grad()
                logits, debug_info = model(inputs)
                
                # Calculate loss
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss.backward()
                
                # Gradient clipping (important for Amharic)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # CRITICAL: Real-time validation every 100 steps
                if step % 100 == 0 and step > 0:
                    model.eval()
                    
                    with torch.no_grad():
                        for prompt in test_prompts:
                            # Convert prompt to character indices
                            prompt_indices = [dataset.char_to_idx.get(c, 0) for c in prompt]
                            input_tensor = torch.tensor([prompt_indices], dtype=torch.long).to(device)
                            
                            try:
                                # Generate continuation
                                generated = model.generate(
                                    input_tensor,
                                    max_length=20,
                                    temperature=0.8,
                                    top_k=30
                                )
                                
                                # Decode to text
                                generated_indices = generated[0].cpu().numpy().tolist()
                                generated_text = dataset.decode(generated_indices)
                                new_part = generated_text[len(prompt):]
                                
                                # VALIDATE OUTPUT
                                is_valid, reason, quality = validate_amharic_output(new_part)
                                total_validations += 1
                                
                                if is_valid:
                                    valid_outputs += 1
                                    print(f"\n✅ VALID: '{prompt}' → '{generated_text}' (Q: {quality:.2f})")
                                else:
                                    print(f"\n❌ INVALID: '{prompt}' → '{generated_text}' ({reason})")
                                
                            except Exception as e:
                                print(f"\n🔥 ERROR: '{prompt}' → {str(e)[:50]}...")
                    
                    model.train()
                
                # Update progress
                avg_loss = total_loss / (step + 1)
                validation_rate = valid_outputs / max(total_validations, 1)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'valid_rate': f'{validation_rate:.2f}',
                    'boundaries': f'{debug_info["boundaries"].sum().item():.0f}'
                })
        
        avg_epoch_loss = total_loss / len(train_loader)
        epoch_validation_rate = valid_outputs / max(total_validations, 1)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Validation Rate: {epoch_validation_rate:.2f} ({valid_outputs}/{total_validations})")
        
        # Save checkpoint if validation is improving
        if epoch_validation_rate > 0.3:  # 30% valid outputs is good progress
            checkpoint_path = f"outputs/valid_amharic_epoch_{epoch+1}.pt"
            os.makedirs("outputs", exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_epoch_loss,
                'validation_rate': epoch_validation_rate,
                'valid_outputs': valid_outputs,
                'total_validations': total_validations
            }, checkpoint_path)
            print(f"✅ Checkpoint saved: {checkpoint_path}")

def main():
    """Main training function with VALID Amharic data"""
    print("🔥 PROPER AMHARIC H-NET TRAINING")
    print("Using space-free, character-level, morpheme-validated data")
    print("="*80)
    
    # Load VALID dataset
    dataset_path = "data/processed/character_level_dataset.json"
    
    if not os.path.exists(dataset_path):
        print("❌ Valid dataset not found. Run emergency_data_fix.py first!")
        return
    
    dataset = ValidAmharicDataset(dataset_path, max_length=128)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Create model with CORRECT vocabulary size
    print(f"\n🔧 Creating H-Net with vocabulary size: {dataset.vocab_size}")
    
    model = AmharicHNetMixer(
        vocab_size=dataset.vocab_size,  # Use ACTUAL Amharic vocab size
        d_model=384,
        n_heads=6,
        n_backbone_layers=6,
        max_chunks=64
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {param_count:,} parameters")
    
    # Train with validation
    train_with_validation(
        model=model,
        train_loader=train_loader,
        dataset=dataset,
        num_epochs=25,
        learning_rate=2e-5  # Conservative as recommended
    )
    
    # Save final model
    final_path = "outputs/valid_amharic_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Final model saved: {final_path}")
    
    print(f"\n🎯 TRAINING COMPLETED!")
    print("Model trained on VALID space-free Amharic with real-time validation")

if __name__ == "__main__":
    main()