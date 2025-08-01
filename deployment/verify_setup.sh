#!/bin/bash

# Verification script for deployment setup
echo "🔍 Verifying Deployment Setup"
echo "=============================="

# Check model checkpoint
echo "1. Checking model checkpoint..."
if [ -f "../outputs/test_checkpoint.pt" ]; then
    size=$(ls -lh ../outputs/test_checkpoint.pt | awk '{print $5}')
    echo "✓ Model checkpoint found ($size)"
else
    echo "❌ Model checkpoint not found at ../outputs/test_checkpoint.pt"
    exit 1
fi

# Check Docker
echo "2. Checking Docker..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
    echo "✓ Docker installed (version $docker_version)"
else
    echo "❌ Docker not found"
    exit 1
fi

# Check Docker Compose
echo "3. Checking Docker Compose..."
if command -v docker-compose &> /dev/null; then
    compose_version=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
    echo "✓ Docker Compose installed (version $compose_version)"
else
    echo "❌ Docker Compose not found"
    exit 1
fi

# Check Python dependencies in main project
echo "4. Checking Python environment..."
cd ..
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
    source venv/bin/activate
    
    # Check key dependencies
    if python -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>/dev/null; then
        echo "✓ PyTorch available"
    else
        echo "❌ PyTorch not available"
    fi
    
    if python -c "import fastapi; print('✓ FastAPI available')" 2>/dev/null; then
        echo "✓ FastAPI dependencies available"
    else
        echo "❌ FastAPI not available - installing production requirements..."
        pip install -r deployment/requirements-production.txt
    fi
    
    deactivate
else
    echo "❌ Virtual environment not found"
    echo "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
fi

cd deployment

# Check deployment files
echo "5. Checking deployment files..."
required_files=(
    "Dockerfile"
    "docker-compose.yml"
    "requirements-production.txt"
    "app/main.py"
    "app/model_service.py"
    "app/cultural_safety.py"
    "deploy.sh"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

# Check monitoring configuration
echo "6. Checking monitoring setup..."
monitoring_files=(
    "monitoring/prometheus.yml"
    "monitoring/grafana/dashboards/hnet-dashboard.json"
    "monitoring/grafana/datasources/prometheus.yml"
)

for file in "${monitoring_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "❌ $file missing"
        exit 1
    fi
done

echo ""
echo "🎉 Deployment setup verification completed!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure"
echo "2. Run: ./deploy.sh"
echo "3. Test with: python test_deployment.py"
echo ""
echo "Quick start:"
echo "  cp .env.example .env"
echo "  # Edit .env with your configuration"
echo "  ./deploy.sh"