#!/usr/bin/env python3
"""
Download Trained Models from Kaggle GPU Training
19M+ parameter Amharic H-Net models
"""

import subprocess
import os
from pathlib import Path

def download_models():
    """Download the trained models from Kaggle output."""
    print("🚀 DOWNLOADING KAGGLE TRAINED MODELS")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)
    
    print("📥 Downloading models from Kaggle notebook output...")
    
    # You'll need to replace 'your-notebook-url' with your actual Kaggle notebook
    # These commands will work once you have the specific dataset/notebook reference
    
    print("🎯 Manual Download Instructions:")
    print("1. Go to your Kaggle notebook that just completed training")
    print("2. Look for the 'Output' tab on the right side")
    print("3. You should see these files:")
    print("   • best_model.pt (19M+ parameters)")
    print("   • final_model.pt (19M+ parameters)")
    print("   • Training logs")
    print("4. Right-click each .pt file and select 'Download'")
    print("5. Save them to the 'trained_models/' directory")
    
    print("\n✅ Once downloaded, you can use:")
    print("   python kaggle_inference.py --model trained_models/best_model.pt")
    
    return True

if __name__ == "__main__":
    download_models()
