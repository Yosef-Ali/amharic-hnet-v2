#!/bin/bash
# Amharic H-Net v2 Development Environment Setup
# Training Engineer Agent Implementation

set -e

echo "🇪🇹 Setting up Amharic H-Net v2 Development Environment..."

# 1. Python Environment Setup
echo "📦 Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

echo "⬆️ Upgrading pip and setuptools..."
pip install --upgrade pip setuptools wheel

# 2. Core ML Dependencies
echo "🧠 Installing core ML dependencies..."
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.30.0
pip install accelerate>=0.20.0
pip install datasets>=2.12.0

# 3. Training & Monitoring Tools
echo "📊 Installing training and monitoring tools..."
pip install wandb
pip install tensorboard
pip install tqdm
pip install einops

# 4. Data Processing & Augmentation
echo "🔧 Installing data processing libraries..."
pip install numpy pandas
pip install scikit-learn
pip install matplotlib seaborn
pip install pyyaml
pip install morfessor

# 5. Development Tools
echo "🛠️ Installing development tools..."
pip install pytest
pip install black
pip install flake8
pip install pre-commit

# 6. Create Directory Structure
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,augmented,morpheme_annotated}
mkdir -p outputs/{checkpoints,logs,evaluations}
mkdir -p experiments/{baseline,transfer_learning,optimized}
mkdir -p logs/{training,evaluation,deployment}
mkdir -p configs/{training,evaluation,deployment}

# 7. Create Requirements File
echo "📝 Generating requirements.txt..."
pip freeze > requirements.txt

# 8. Setup Git Hooks
echo "🔗 Setting up pre-commit hooks..."
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black
        language_version: python3.8
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
EOF

pre-commit install

# 9. Create Environment Validation Script
cat > validate_environment.py << 'EOF'
#!/usr/bin/env python3
"""Environment validation script for Amharic H-Net v2."""

import sys
import subprocess
import importlib

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_gpu_availability():
    """Check CUDA/GPU availability."""
    print("🖥️ Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ {gpu_count} GPU(s) available: {gpu_name}")
            return True
        else:
            print("⚠️ No GPU detected - training will be CPU-only")
            return True
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_required_packages():
    """Check if all required packages are installed."""
    print("📦 Checking required packages...")
    required_packages = [
        'torch', 'transformers', 'accelerate', 'datasets',
        'wandb', 'tensorboard', 'numpy', 'pandas', 'sklearn',
        'matplotlib', 'seaborn', 'yaml', 'morfessor'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn')
            elif package == 'yaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_directory_structure():
    """Check if directory structure is properly created."""
    print("📁 Checking directory structure...")
    import os
    
    required_dirs = [
        'data/raw', 'data/processed', 'data/augmented',
        'outputs/checkpoints', 'outputs/logs',
        'experiments/baseline', 'configs/training'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path}")
            missing_dirs.append(dir_path)
    
    return len(missing_dirs) == 0, missing_dirs

def main():
    """Run all validation checks."""
    print("🇪🇹 Amharic H-Net v2 Environment Validation")
    print("=" * 50)
    
    checks_passed = 0
    total_checks = 4
    
    # Python version check
    if check_python_version():
        checks_passed += 1
    
    # GPU availability check
    if check_gpu_availability():
        checks_passed += 1
    
    # Package installation check
    packages_ok, missing = check_required_packages()
    if packages_ok:
        checks_passed += 1
    else:
        print(f"Missing packages: {missing}")
    
    # Directory structure check
    dirs_ok, missing_dirs = check_directory_structure()
    if dirs_ok:
        checks_passed += 1
    else:
        print(f"Missing directories: {missing_dirs}")
    
    print("\n" + "=" * 50)
    print(f"Environment Status: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 Environment setup successful! Ready for Amharic H-Net training.")
        return 0
    else:
        print("⚠️ Environment setup incomplete. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# 10. Run Environment Validation
echo "🔍 Running environment validation..."
python validate_environment.py

# 11. Create Development Configuration
cat > configs/development.yaml << 'EOF'
# Development Configuration for Amharic H-Net v2
environment:
  name: "amharic-hnet-dev"
  python_version: "3.8+"
  cuda_version: "11.8"
  
directories:
  data_root: "data"
  output_root: "outputs"
  config_root: "configs"
  log_root: "logs"

training:
  device: "auto"  # auto-detect GPU/CPU
  mixed_precision: true
  gradient_checkpointing: false
  
logging:
  level: "INFO"
  wandb_project: "amharic-hnet-v2"
  tensorboard_dir: "logs/tensorboard"

development:
  code_style: "black"
  line_length: 88
  type_checking: "mypy"
  testing: "pytest"
EOF

echo "✅ Environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run validation: python validate_environment.py"
echo "3. Start data collection: python -m src.data_collection.collect_amharic"
echo "4. Begin training: python main.py train --config configs/config.yaml"
echo ""
echo "🇪🇹 Ready to develop world-class Amharic H-Net!"