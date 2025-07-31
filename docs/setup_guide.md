# Amharic H-Net v2 Development Environment Setup Guide

This comprehensive guide will help you set up a complete development environment for Amharic H-Net v2 training and development.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup](#quick-setup)
3. [Manual Setup](#manual-setup)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Development Workflow](#development-workflow)
7. [GPU Setup](#gpu-setup)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 16GB+ recommended (8GB minimum)
- **Storage**: 50GB+ free space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Software Prerequisites

- Git
- Python 3.8+
- pip
- CUDA 11.7+ (for GPU training)

## Quick Setup

The fastest way to set up your environment is using our automated setup script:

```bash
# Clone the repository
git clone https://github.com/your-org/amharic-hnet-v2.git
cd amharic-hnet-v2

# Run the automated setup script
./scripts/training/setup_training_env.sh
```

This script will:
- Create and configure a virtual environment
- Install all dependencies with pinned versions
- Set up experiment tracking (W&B, TensorBoard)
- Configure development tools
- Validate the installation

## Manual Setup

If you prefer manual setup or need to customize the installation:

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 2. Install PyTorch

Choose the appropriate PyTorch installation based on your system:

```bash
# CUDA 11.8 (recommended for most modern GPUs)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 11.7
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu117

# CPU only
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### 4. Setup Development Tools

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### 5. Create Directory Structure

```bash
# Create necessary directories
mkdir -p data/{raw,processed,morpheme_annotated,evaluation}
mkdir -p outputs/{models,checkpoints,tensorboard,wandb}
mkdir -p experiments/{configs,logs,results}
mkdir -p logs/{training,evaluation}
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Amharic H-Net v2 Environment Configuration

# Python path
PYTHONPATH=./src

# CUDA configuration
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false

# Experiment tracking
WANDB_PROJECT=amharic-hnet-v2
WANDB_ENTITY=your-username
WANDB_API_KEY=your-api-key

# Training configuration
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Paths
DATA_ROOT=./data
OUTPUT_ROOT=./outputs
CACHE_DIR=./cache

# Model settings
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=8

# Logging
LOG_LEVEL=INFO
```

### Weights & Biases Setup

1. Create account at [wandb.ai](https://wandb.ai)
2. Get your API key from [settings](https://wandb.ai/settings)
3. Set your API key:
   ```bash
   export WANDB_API_KEY=your_api_key_here
   # or
   wandb login
   ```

### TensorBoard Configuration

TensorBoard is automatically configured to use `outputs/tensorboard/` directory.

## Verification

### Automated Validation

Run the comprehensive environment validation:

```bash
python scripts/validate_environment.py
```

This will check:
- Python version and virtual environment
- All package installations and versions
- PyTorch and CUDA setup
- Directory structure
- Configuration files
- System resources
- Basic functionality

### Manual Verification

#### Test PyTorch Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

#### Test Key Packages

```python
# Test transformers
from transformers import AutoTokenizer
print("Transformers: OK")

# Test experiment tracking
import wandb
import tensorboard
print("Experiment tracking: OK")

# Test ML libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
print("ML libraries: OK")
```

## Development Workflow

### Daily Development

1. **Activate Environment**
   ```bash
   source venv/bin/activate
   ```

2. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

4. **Code Quality Checks**
   ```bash
   # Format code
   black src/ tests/

   # Check linting
   flake8 src/ tests/

   # Sort imports
   isort src/ tests/

   # Run all pre-commit hooks
   pre-commit run --all-files
   ```

### Training Workflow

1. **Prepare Data**
   ```bash
   python main.py preprocess --input data/raw/ --output data/processed/
   ```

2. **Start Training**
   ```bash
   # Basic training
   python main.py train --config configs/training/base.yaml

   # Large model
   python main.py train --config configs/training/large_model.yaml

   # Transfer learning
   python main.py train --config configs/training/transfer_learning.yaml
   ```

3. **Monitor Training**
   ```bash
   # TensorBoard
   tensorboard --logdir outputs/tensorboard

   # Check logs
   tail -f logs/training/train.log
   ```

4. **Evaluate Model**
   ```bash
   python main.py evaluate --model-path outputs/models/best_model.pt --data data/processed/test.jsonl
   ```

## GPU Setup

### NVIDIA GPU Setup

1. **Install NVIDIA Drivers**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install nvidia-driver-470

   # Check installation
   nvidia-smi
   ```

2. **Install CUDA Toolkit**
   ```bash
   # Download from https://developer.nvidia.com/cuda-downloads
   # Or use package manager
   sudo apt install nvidia-cuda-toolkit
   ```

3. **Verify CUDA Installation**
   ```bash
   nvcc --version
   ```

### Multi-GPU Setup

For multiple GPUs, configure the environment:

```bash
# Use specific GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Or in training config
python main.py train --config configs/training/large_model.yaml --gpus 0,1,2,3
```

### Memory Optimization

For limited GPU memory:

1. **Reduce Batch Size**
   ```yaml
   training:
     batch_size: 4  # Reduce from 8
     gradient_accumulation_steps: 8  # Increase to maintain effective batch size
   ```

2. **Enable Gradient Checkpointing**
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. **Use Mixed Precision**
   ```yaml
   training:
     use_fp16: true
     fp16_opt_level: "O2"
   ```

## Configuration Files

### Training Configurations

- **`configs/training/base.yaml`**: Standard training setup
- **`configs/training/large_model.yaml`**: Large model with optimizations
- **`configs/training/transfer_learning.yaml`**: Fine-tuning configuration

### Model Configurations

- **`configs/model/hnet_small.yaml`**: Small model for testing
- **`configs/model/hnet_base.yaml`**: Standard model configuration
- **`configs/model/hnet_large.yaml`**: Large model configuration

### Customization

Create custom configurations by copying existing files:

```bash
# Copy base config
cp configs/training/base.yaml configs/training/my_experiment.yaml

# Edit your configuration
vim configs/training/my_experiment.yaml

# Run with custom config
python main.py train --config configs/training/my_experiment.yaml
```

## Docker Setup (Optional)

For containerized development:

```bash
# Build Docker image
docker build -t amharic-hnet-v2 .

# Run container with GPU support
docker run --gpus all -v $(pwd):/workspace -it amharic-hnet-v2
```

## IDE Setup

### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- GitLens
- Jupyter

Settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.flake8Enabled": true,
    "python.linting.enabled": true
}
```

### PyCharm

1. Set interpreter to `./venv/bin/python`
2. Configure code formatter to use Black
3. Enable flake8 linting
4. Set source root to `src/`

## Best Practices

### Development

1. **Always use virtual environment**
2. **Run tests before committing**
3. **Use pre-commit hooks**
4. **Follow PEP 8 style guide**
5. **Write docstrings for functions**
6. **Use type hints**

### Training

1. **Start with small experiments**
2. **Monitor GPU memory usage**
3. **Save checkpoints frequently**
4. **Log hyperparameters**
5. **Use version control for configs**
6. **Document experiment results**

### Data Management

1. **Keep raw data unchanged**
2. **Version processed datasets**
3. **Use consistent file formats**
4. **Document data preprocessing steps**
5. **Backup important results**

## Performance Optimization

### CPU Optimization

```bash
# Set optimal thread counts
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

### Memory Optimization

```python
# In your training script
import torch
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic operations for speed
```

### Storage Optimization

```bash
# Use SSD for data storage
# Keep datasets on fastest storage
# Use data loading optimizations
```

## Next Steps

After setup completion:

1. **Read the training guide**: `docs/training_guide.md`
2. **Explore example notebooks**: `notebooks/`
3. **Review the API reference**: `docs/api_reference.md`
4. **Join the community**: Links in README.md
5. **Contribute**: See CONTRIBUTING.md

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: `docs/` directory
- **Examples**: `notebooks/` directory

---

**Note**: This setup guide is continuously updated. Please check for the latest version and report any issues you encounter.