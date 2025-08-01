#!/usr/bin/env python3
"""
Quick test to verify model loading works correctly
"""

import sys
import os
import asyncio
import torch

# Add paths
sys.path.append('/Users/mekdesyared/amharic-hnet-v2/deployment')
sys.path.append('/Users/mekdesyared/amharic-hnet-v2')

from app.model_service import ModelService
from app.cultural_safety import CulturalSafetyService

async def test_model_loading():
    """Test that we can load and use the model."""
    print("🧪 Testing Model Loading")
    print("=" * 30)
    
    model_path = "/Users/mekdesyared/amharic-hnet-v2/outputs/test_checkpoint.pt"
    
    try:
        # Test model service
        print("1. Loading model service...")
        model_service = ModelService(model_path)
        await model_service.load_model()
        print("✓ Model loaded successfully")
        
        # Test model info
        print("2. Getting model info...")
        model_info = await model_service.get_model_info()
        print(f"✓ Model info retrieved: {model_info['architecture']['type']}")
        print(f"  - Parameters: {model_info['architecture']['total_parameters']:,}")
        print(f"  - Device: {model_info['device']}")
        
        # Test generation
        print("3. Testing text generation...")
        generated_text, stats = await model_service.generate_text(
            prompt="ሰላም",
            max_length=20,
            temperature=1.0
        )
        print(f"✓ Generated text: '{generated_text}'")
        print(f"  - Inference time: {stats['inference_time']:.3f}s")
        print(f"  - Meets target: {stats['inference_time'] < 0.15}")
        
        # Test cultural safety
        print("4. Testing cultural safety...")
        safety_service = CulturalSafetyService()
        is_safe, violations = safety_service.check_input_safety("ሰላም እንዴት ነሽ")
        print(f"✓ Cultural safety check: {'SAFE' if is_safe else 'VIOLATIONS'}")
        
        # Test health check
        print("5. Testing health checks...")
        health = await model_service.health_check()
        print(f"✓ Health check: {health['status']}")
        
        print("\n🎉 All tests passed! Model is ready for deployment.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'model_service' in locals():
            await model_service.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_model_loading())
    exit(0 if success else 1)