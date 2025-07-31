#!/bin/bash

# Amharic H-Net v2 Setup Script
# This script sets up the complete development environment for the Amharic H-Net project

set -e  # Exit on any error

echo "🇪🇹 Setting up Amharic H-Net v2 Development Environment"
echo "========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check if Python 3.8+ is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_status "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        return 1
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    print_status "Virtual environment created"
}

# Activate virtual environment and install dependencies
install_dependencies() {
    print_info "Installing Python dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install PyTorch (CPU version by default, user can switch to GPU version)
    if command -v nvidia-smi &> /dev/null; then
        print_info "NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_info "Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    pip install -r requirements.txt
    
    print_status "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_info "Creating project directories..."
    
    # Create missing directories
    mkdir -p data/raw
    mkdir -p data/processed
    mkdir -p data/morpheme_annotated
    mkdir -p models
    mkdir -p outputs
    mkdir -p logs
    mkdir -p evaluation_results
    mkdir -p notebooks
    
    print_status "Directories created"
}

# Set up Git hooks (if .git exists)
setup_git_hooks() {
    if [ -d ".git" ]; then
        print_info "Setting up Git hooks..."
        
        # Create pre-commit hook for code quality
        cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for Amharic H-Net
echo "Running pre-commit checks..."

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Please activate virtual environment before committing"
    exit 1
fi

# Run basic Python syntax check
find src -name "*.py" -exec python -m py_compile {} \;
if [ $? -ne 0 ]; then
    echo "Python syntax errors found"
    exit 1
fi

echo "Pre-commit checks passed"
EOF
        
        chmod +x .git/hooks/pre-commit
        print_status "Git hooks configured"
    else
        print_warning "Not a Git repository, skipping Git hooks setup"
    fi
}

# Create configuration files
create_configs() {
    print_info "Creating configuration files..."
    
    # Create .env file for environment variables
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# Amharic H-Net Environment Variables
PYTHONPATH=./src
CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
EOF
        print_status ".env file created"
    else
        print_warning ".env file already exists"
    fi
    
    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
outputs/
logs/
evaluation_results/
*.pt
*.pth
*.ckpt
wandb/
.env

# Data (uncomment if you don't want to track data files)
# data/raw/*
# data/processed/*
# !data/raw/sample_amharic.txt

# Jupyter
.ipynb_checkpoints/
*.ipynb

# TensorBoard
runs/
tb_logs/
EOF
        print_status ".gitignore created"
    else
        print_warning ".gitignore already exists"
    fi
}

# Test installation
test_installation() {
    print_info "Testing installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Run the test command
    python main.py test
    
    if [ $? -eq 0 ]; then
        print_status "Installation test passed"
    else
        print_error "Installation test failed"
        return 1
    fi
}

# Create sample data
create_sample_data() {
    print_info "Creating sample data..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Use the preprocessing script to create sample data
    python main.py preprocess --create-sample --output data/processed
    
    print_status "Sample data created"
}

# Main setup function
main() {
    echo
    print_info "Starting setup process..."
    echo
    
    # Check prerequisites
    if ! check_python; then
        print_error "Setup failed: Python requirements not met"
        exit 1
    fi
    
    # Create virtual environment
    create_venv
    
    # Install dependencies
    install_dependencies
    
    # Create directories
    create_directories
    
    # Setup Git hooks
    setup_git_hooks
    
    # Create configuration files
    create_configs
    
    # Create sample data
    create_sample_data
    
    # Test installation
    if ! test_installation; then
        print_error "Setup failed: Installation test failed"
        exit 1
    fi
    
    echo
    print_status "Setup completed successfully!"
    echo
    echo "🎉 Amharic H-Net v2 is ready!"
    echo
    echo "To get started:"
    echo "  1. Activate the virtual environment: source venv/bin/activate"
    echo "  2. Train a model: python main.py train --config configs/config.yaml"
    echo "  3. Generate text: python main.py generate --model-path outputs/checkpoint_best.pt --prompt 'አማርኛ'"
    echo
    echo "For more information, run: python main.py --help"
    echo
}

# Run main function
main "$@"