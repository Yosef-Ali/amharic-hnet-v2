#!/bin/bash

# Amharic H-Net v2 Training Environment Setup
# Complete setup script for training environment with all optimizations

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_header() {
    echo -e "\n${PURPLE}${WHITE}$1${NC}"
    echo -e "${PURPLE}$(printf '=%.0s' {1..60})${NC}"
}

# Check if running in correct directory
check_project_root() {
    if [[ ! -f "requirements.txt" ]] || [[ ! -d "src" ]]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
    print_status "Running from project root directory"
}

# Create and activate virtual environment
setup_virtual_environment() {
    print_header "Setting up Virtual Environment"
    
    # Check if venv already exists
    if [[ -d "venv" ]]; then
        print_warning "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf venv
        else
            print_info "Using existing virtual environment"
            source venv/bin/activate
            return 0
        fi
    fi
    
    # Create new virtual environment
    print_info "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    print_status "Virtual environment created and activated"
    
    # Upgrade pip and setuptools
    print_info "Upgrading pip and setuptools..."
    pip install --upgrade pip setuptools wheel
    print_status "Package tools upgraded"
}

# Install PyTorch with appropriate CUDA support
install_pytorch() {
    print_header "Installing PyTorch"
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        # Check CUDA version
        CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+")
        print_info "NVIDIA GPU detected with CUDA $CUDA_VERSION"
        
        # Install CUDA-enabled PyTorch
        if [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l) -eq 1 ]]; then
            print_info "Installing PyTorch with CUDA 11.8 support..."
            pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
        elif [[ $(echo "$CUDA_VERSION >= 11.7" | bc -l) -eq 1 ]]; then
            print_info "Installing PyTorch with CUDA 11.7 support..."
            pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu117
        else
            print_warning "CUDA version $CUDA_VERSION not directly supported, installing CPU version"
            pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
        fi
    else
        print_info "No NVIDIA GPU detected, installing CPU-only PyTorch..."
        pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
    fi
    
    print_status "PyTorch installation completed"
}

# Install all other dependencies
install_dependencies() {
    print_header "Installing Python Dependencies"
    
    # Install from requirements.txt
    print_info "Installing packages from requirements.txt..."
    pip install -r requirements.txt
    
    # Install development dependencies
    print_info "Installing development dependencies..."
    pip install -e ".[dev]"
    
    print_status "All dependencies installed"
}

# Setup experiment tracking
setup_experiment_tracking() {
    print_header "Setting up Experiment Tracking"
    
    # Setup Weights & Biases
    print_info "Configuring Weights & Biases..."
    
    # Check if WANDB_API_KEY is set
    if [[ -z "${WANDB_API_KEY}" ]]; then
        print_warning "WANDB_API_KEY not set in environment"
        echo "To use Weights & Biases:"
        echo "1. Get your API key from https://wandb.ai/settings"
        echo "2. Set it: export WANDB_API_KEY=your_key_here"
        echo "3. Or login: wandb login"
    else
        print_status "WANDB_API_KEY found in environment"
    fi
    
    # Try to login to wandb
    if command -v wandb &> /dev/null; then
        wandb --version > /dev/null 2>&1 && print_status "Weights & Biases available"
    fi
    
    # Create TensorBoard directory
    mkdir -p outputs/tensorboard
    print_status "TensorBoard directory created"
}

# Setup data directories with proper structure
setup_data_directories() {
    print_header "Setting up Data Directories"
    
    # Create all necessary directories
    directories=(
        "data/raw"
        "data/processed"
        "data/morpheme_annotated"
        "data/evaluation"
        "outputs/models"
        "outputs/checkpoints"
        "outputs/tensorboard"
        "outputs/wandb"
        "experiments/configs"
        "experiments/logs"
        "experiments/results"
        "logs/training"
        "logs/evaluation"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    done
    
    print_status "Data directory structure created"
}

# Install and configure pre-commit hooks
setup_precommit_hooks() {
    print_header "Setting up Pre-commit Hooks"
    
    if [[ -f ".pre-commit-config.yaml" ]]; then
        print_info "Installing pre-commit hooks..."
        pre-commit install
        
        # Run pre-commit on all files to verify setup
        print_info "Running pre-commit on all files..."
        pre-commit run --all-files || print_warning "Some pre-commit checks failed (expected on first run)"
        
        print_status "Pre-commit hooks configured"
    else
        print_warning ".pre-commit-config.yaml not found, skipping pre-commit setup"
    fi
}

# Create environment configuration file
create_env_file() {
    print_header "Creating Environment Configuration"
    
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# Amharic H-Net v2 Environment Configuration

# Python path
PYTHONPATH=./src

# CUDA configuration
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false

# Experiment tracking
WANDB_PROJECT=amharic-hnet-v2
WANDB_ENTITY=your-username

# Training configuration
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Data paths
DATA_ROOT=./data
OUTPUT_ROOT=./outputs
CACHE_DIR=./cache

# Model configuration
MAX_SEQUENCE_LENGTH=512
BATCH_SIZE=8

# Logging
LOG_LEVEL=INFO
EOF
        print_status "Environment configuration file created (.env)"
    else
        print_warning ".env file already exists"
    fi
}

# Test the installation
test_installation() {
    print_header "Testing Installation"
    
    print_info "Running environment validation..."
    if python scripts/validate_environment.py; then
        print_status "Environment validation passed"
    else
        print_error "Environment validation failed"
        return 1
    fi
    
    # Test PyTorch installation
    print_info "Testing PyTorch installation..."
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')

# Test basic operations
x = torch.randn(2, 3)
y = torch.mm(x, x.t())
print('Basic tensor operations: OK')

if torch.cuda.is_available():
    x_cuda = x.cuda()
    y_cuda = torch.mm(x_cuda, x_cuda.t())
    print('CUDA tensor operations: OK')
"
    
    if [[ $? -eq 0 ]]; then
        print_status "PyTorch test passed"
    else
        print_error "PyTorch test failed"
        return 1
    fi
    
    # Test imports
    print_info "Testing package imports..."
    python -c "
import transformers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import tensorboard
print('All critical packages imported successfully')
"
    
    if [[ $? -eq 0 ]]; then
        print_status "Package import test passed"
    else
        print_error "Package import test failed"
        return 1
    fi
}

# Generate setup summary
generate_setup_summary() {
    print_header "Setup Summary"
    
    # Create summary file
    cat > SETUP_SUMMARY.md << EOF
# Amharic H-Net v2 Training Environment Setup Summary

## Setup Completed
- ✅ Virtual environment created and configured
- ✅ PyTorch installed with appropriate CUDA support
- ✅ All dependencies installed from requirements.txt
- ✅ Development tools configured (black, flake8, pytest)
- ✅ Pre-commit hooks installed
- ✅ Experiment tracking setup (W&B, TensorBoard)
- ✅ Directory structure created
- ✅ Environment configuration file created

## Quick Start

### Activate Environment
\`\`\`bash
source venv/bin/activate
\`\`\`

### Validate Environment
\`\`\`bash
python scripts/validate_environment.py
\`\`\`

### Start Training
\`\`\`bash
# Basic training
python main.py train --config configs/training/base.yaml

# Large model training
python main.py train --config configs/training/large_model.yaml

# Transfer learning
python main.py train --config configs/training/transfer_learning.yaml
\`\`\`

### Run Tests
\`\`\`bash
pytest tests/ -v
\`\`\`

### Code Quality Checks
\`\`\`bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Run pre-commit
pre-commit run --all-files
\`\`\`

## Environment Details
- Python: $(python --version)
- PyTorch: $(python -c "import torch; print(torch.__version__)")
- CUDA Available: $(python -c "import torch; print(torch.cuda.is_available())")
- Virtual Environment: $(echo $VIRTUAL_ENV)

## Next Steps
1. Prepare your training data in \`data/raw/\`
2. Configure experiment settings in \`configs/\`
3. Start training with your preferred configuration
4. Monitor experiments with TensorBoard or W&B

## Troubleshooting
- Run \`python scripts/validate_environment.py\` to check setup
- Check logs in \`logs/\` directory
- Refer to documentation in \`docs/\`

Setup completed on: $(date)
EOF

    print_status "Setup summary generated: SETUP_SUMMARY.md"
}

# Main setup function
main() {
    print_header "Amharic H-Net v2 Training Environment Setup"
    echo -e "${CYAN}Setting up complete development environment for Amharic H-Net v2 training${NC}\n"
    
    # Check prerequisites
    check_project_root
    
    # Setup virtual environment
    setup_virtual_environment
    
    # Install PyTorch first
    install_pytorch
    
    # Install other dependencies
    install_dependencies
    
    # Setup experiment tracking
    setup_experiment_tracking
    
    # Setup directories
    setup_data_directories
    
    # Setup pre-commit hooks
    setup_precommit_hooks
    
    # Create environment file
    create_env_file
    
    # Test installation
    if ! test_installation; then
        print_error "Installation test failed. Please check the errors above."
        exit 1
    fi
    
    # Generate summary
    generate_setup_summary
    
    print_header "Setup Complete!"
    echo -e "${GREEN}🎉 Amharic H-Net v2 training environment is ready!${NC}\n"
    
    echo -e "${WHITE}To get started:${NC}"
    echo -e "  ${CYAN}1. Activate environment:${NC} source venv/bin/activate"
    echo -e "  ${CYAN}2. Validate setup:${NC} python scripts/validate_environment.py"
    echo -e "  ${CYAN}3. Start training:${NC} python main.py train --config configs/training/base.yaml"
    echo -e "  ${CYAN}4. Monitor progress:${NC} tensorboard --logdir outputs/tensorboard"
    
    echo -e "\n${BLUE}Documentation and examples available in:${NC}"
    echo -e "  - SETUP_SUMMARY.md"
    echo -e "  - docs/"
    echo -e "  - notebooks/"
    
    echo -e "\n${PURPLE}Happy training! 🇪🇹${NC}"
}

# Run main function with all arguments
main "$@"