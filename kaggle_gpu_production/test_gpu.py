#!/usr/bin/env python3
"""
Simple GPU Test Script for Kaggle
Upload this first to verify GPU access and PyTorch installation
"""

import torch
import time

print("🔍 Kaggle GPU Test Started")
print("=" * 40)

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu_props.name}")
        print(f"  Memory: {gpu_props.total_memory / 1e9:.1f} GB")
        print(f"  Compute: {gpu_props.major}.{gpu_props.minor}")
    
    # Set device
    device = torch.device('cuda')
    print(f"\n🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Simple GPU test
    print("\n🧪 Running GPU Test...")
    start_time = time.time()
    
    # Create tensors on GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Matrix multiplication
    z = torch.matmul(x, y)
    
    # Synchronize GPU
    torch.cuda.synchronize()
    
    end_time = time.time()
    print(f"✅ GPU Test Completed in {end_time - start_time:.3f} seconds")
    print(f"Result shape: {z.shape}")
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Test model creation
    print("\n🏗️ Testing Model Creation...")
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Test Model Created: {total_params:,} parameters")
    
    # Test forward pass
    test_input = torch.randn(32, 1000, device=device)
    output = model(test_input)
    print(f"✅ Forward Pass: {test_input.shape} → {output.shape}")
    
    print("\n🎉 All GPU Tests Passed!")
    print("✅ Ready for large model training")
    
else:
    print("❌ CUDA not available")
    print("Please enable GPU in Kaggle notebook settings:")
    print("Settings → Accelerator → GPU T4 x2")
    device = torch.device('cpu')
    print("🔄 Running on CPU instead...")
    
    # Simple CPU test
    x = torch.randn(100, 100)
    y = torch.randn(100, 100)
    z = torch.matmul(x, y)
    print(f"✅ CPU Test Completed: {z.shape}")

print("\n" + "=" * 40)
print("🏁 GPU Test Complete")
print("Ready to proceed with full training!")