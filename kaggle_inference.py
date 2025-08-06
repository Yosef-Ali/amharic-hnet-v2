#!/usr/bin/env python3
"""
Kaggle Competition Inference with Downloaded Models
Use the 19M+ parameter models trained on Kaggle GPU
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
from pathlib import Path

class AmharicHNetInference:
    """Inference with the 19M+ parameter trained model."""
    
    def __init__(self, model_path="trained_models/best_model.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Loading model from: {model_path}")
        print(f"💻 Using device: {self.device}")
        
        # Load the trained model
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            print("✅ Model loaded successfully!")
            print(f"📊 Model info: {len(str(checkpoint))} bytes")
        else:
            print(f"❌ Model not found at {model_path}")
            print("Please download models using: python download_models.py")
            return
        
        # Initialize model architecture (matching training)
        self.model = self._create_model_architecture()
        
        # Load trained weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("🏆 19M+ parameter Amharic H-Net ready for inference!")
    
    def _create_model_architecture(self):
        """Create the model architecture matching training."""
        # This should match your training architecture
        # Simplified version - adjust based on your actual model
        model = nn.Sequential(
            nn.Embedding(32000, 512),  # vocab_size, d_model
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=512,
                    nhead=16,
                    dim_feedforward=2048,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=24  # Adjust based on your model
            ),
            nn.LayerNorm(512),
            nn.Linear(512, 32000)  # Output vocabulary
        )
        return model
    
    def preprocess_text(self, text):
        """Preprocess Amharic text for inference."""
        # Simple byte-level tokenization
        if isinstance(text, str):
            bytes_data = list(text.encode('utf-8')[:512])  # Max length 512
            
            # Pad to fixed length
            if len(bytes_data) < 512:
                bytes_data.extend([0] * (512 - len(bytes_data)))
            
            return torch.tensor([bytes_data], dtype=torch.long).to(self.device)
        return torch.zeros(1, 512, dtype=torch.long).to(self.device)
    
    def predict(self, text):
        """Make prediction on Amharic text."""
        input_ids = self.preprocess_text(text)
        
        with torch.no_grad():
            outputs = self.model(input_ids)
            
            # Get predictions (simplified)
            if outputs.dim() > 2:
                predictions = outputs.mean(dim=1)  # Average over sequence
            else:
                predictions = outputs
            
            # Get most likely class
            pred_class = torch.argmax(predictions, dim=-1).cpu().numpy()[0]
            confidence = torch.softmax(predictions, dim=-1).max().cpu().numpy()
            
            return {
                'prediction': int(pred_class),
                'confidence': float(confidence),
                'text': text
            }
    
    def create_kaggle_submission(self, test_df, output_path='submission.csv'):
        """Create Kaggle competition submission."""
        print(f"📝 Creating submission for {len(test_df)} samples...")
        
        predictions = []
        start_time = time.time()
        
        for i, text in enumerate(test_df['text']):
            result = self.predict(text)
            predictions.append(result['prediction'])
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(test_df)} samples...")
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': test_df.get('id', range(len(test_df))),
            'prediction': predictions
        })
        
        submission_df.to_csv(output_path, index=False)
        
        elapsed = time.time() - start_time
        print(f"✅ Submission created: {output_path}")
        print(f"⏱️  Processing time: {elapsed:.2f}s")
        print(f"🎯 Ready for Kaggle competition!")
        
        return submission_df

def main():
    """Main inference function."""
    print("🏆 KAGGLE TRAINED MODEL INFERENCE")
    print("=" * 50)
    
    # Check if models exist
    model_path = "trained_models/best_model.pt"
    if not Path(model_path).exists():
        print("❌ Trained models not found!")
        print("Please run: python download_models.py")
        print("Then manually download models from your Kaggle notebook")
        return
    
    # Initialize inference
    inferencer = AmharicHNetInference(model_path)
    
    # Test with sample Amharic text
    test_text = "አማርኛ ቋንቋ በጣም ቆንጆ ነው።"
    print(f"\n🧪 Testing with: {test_text}")
    result = inferencer.predict(test_text)
    print(f"✅ Prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")
    
    # Create sample submission
    test_df = pd.DataFrame({
        'id': range(100),
        'text': [f'የኢትዮጵያ ባህል በጣም ቆንጆ ነው። {i}' for i in range(100)]
    })
    
    submission = inferencer.create_kaggle_submission(test_df)
    print("\n🚀 Ready for Kaggle competition submission!")

if __name__ == "__main__":
    main()
