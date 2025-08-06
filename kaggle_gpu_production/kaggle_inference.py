#!/usr/bin/env python3
"""
Kaggle Inference Script - Amharic H-Net Large Model
Wrapper for production_inference.py
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from production_inference import main

if __name__ == "__main__":
    print("🚀 Starting Kaggle Inference for Amharic H-Net Large Model...")
    main()