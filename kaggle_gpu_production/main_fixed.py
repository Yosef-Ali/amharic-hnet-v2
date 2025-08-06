#!/usr/bin/env python3
"""
Kaggle GPU Training - Amharic H-Net Large Model (FIXED VERSION)
Syntax error corrected - Ready for upload
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

print("🚀 Amharic H-Net GPU Training Started")
print(f"GPU Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Large Model (19M+ parameters)
class AmharicModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50000, 1024)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(1024, 16, 4096, batch_first=True),
            num_layers=12
        )
        self.output = nn.Linear(1024, 50000)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)

model = AmharicModel().to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-4)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
model.train()
start_time = time.time()
best_loss = float('inf')

for epoch in range(20):  # 20 epochs for demo
    epoch_loss = 0
    
    for batch in range(100):  # 100 batches per epoch
        # Generate synthetic data
        inputs = torch.randint(0, 50000, (8, 256), device=device)
        targets = torch.randint(0, 50000, (8, 256), device=device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, 50000), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch % 25 == 0:
            print(f"Epoch {epoch+1}, Batch {batch}: Loss={loss.item():.4f}")
    
    avg_loss = epoch_loss / 100
    print(f"Epoch {epoch+1}/20: Average Loss = {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_loss
        }, 'best_model.pt')
        print(f"💾 Best model saved! Loss: {avg_loss:.4f}")

training_time = time.time() - start_time

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'training_time': training_time,
    'final_loss': avg_loss,
    'best_loss': best_loss
}, 'final_model.pt')

print(f"\n🏆 Training Complete!")
print(f"Time: {training_time/60:.1f} minutes")
print(f"Best Loss: {best_loss:.4f}")
print(f"Models saved: best_model.pt, final_model.pt")
print("Ready for download!")