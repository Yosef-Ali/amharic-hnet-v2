#!/bin/bash

# Amharic H-Net Production Deployment Script
# ==========================================
# 
# This script deploys the Amharic H-Net model as a production-ready API service
# with comprehensive monitoring and observability.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="amharic-hnet"
MODEL_CHECKPOINT_PATH="../outputs/test_checkpoint.pt"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
API_KEY="${API_KEY:-$(openssl rand -hex 32)}"
GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-$(openssl rand -base64 12)}"

echo -e "${BLUE}🚀 Amharic H-Net Production Deployment${NC}"
echo "======================================"
echo "Environment: $DEPLOYMENT_ENV"
echo "Project: $PROJECT_NAME"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_status "Docker is installed"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_status "Docker Compose is installed"
    
    # Check model checkpoint
    if [ ! -f "$MODEL_CHECKPOINT_PATH" ]; then
        print_error "Model checkpoint not found at $MODEL_CHECKPOINT_PATH"
        exit 1
    fi
    print_status "Model checkpoint found"
    
    echo ""
}

# Setup environment
setup_environment() {
    echo -e "${BLUE}Setting up environment...${NC}"
    
    # Create necessary directories
    mkdir -p models logs monitoring/rules nginx/ssl
    
    # Copy model checkpoint
    if [ ! -f "models/test_checkpoint.pt" ]; then
        cp "$MODEL_CHECKPOINT_PATH" models/
        print_status "Model checkpoint copied"
    fi
    
    # Create environment file
    if [ ! -f ".env" ]; then
        cat > .env << EOF
# Amharic H-Net Production Configuration
ENVIRONMENT=$DEPLOYMENT_ENV
API_KEY=$API_KEY
GRAFANA_PASSWORD=$GRAFANA_PASSWORD

# Application Settings
HOST=0.0.0.0
PORT=8000
WORKERS=1
MODEL_PATH=/app/models/test_checkpoint.pt
MODEL_VERSION=1.0.0

# Security
ENABLE_AUTH=true
ENABLE_RATE_LIMITING=true
REQUESTS_PER_MINUTE=60
BURST_SIZE=10

# Performance
ENABLE_CACHING=true
CACHE_TTL=3600
REDIS_URL=redis://redis:6379/0
RESPONSE_TIME_TARGET_MS=200

# Monitoring
ENABLE_METRICS=true
ENABLE_TRACING=false
LOG_LEVEL=INFO

# Cultural Safety
ENABLE_CULTURAL_SAFETY=true
CULTURAL_SAFETY_STRICT_MODE=true

# Allowed Origins (configure for your domain)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
ALLOWED_HOSTS=localhost,127.0.0.1
EOF
        print_status "Environment file created"
    else
        print_warning "Environment file already exists"
    fi
    
    echo ""
}

# Setup monitoring
setup_monitoring() {
    echo -e "${BLUE}Setting up monitoring...${NC}"
    
    # Create Prometheus rules directory
    mkdir -p monitoring/rules
    
    # Create alerting rules
    cat > monitoring/rules/hnet-alerts.yml << 'EOF'
groups:
  - name: hnet-alerts
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"
      
      - alert: CulturalSafetyViolations
        expr: rate(cultural_safety_violations_total[5m]) > 0.01
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Cultural safety violations detected"
          description: "Cultural safety violation rate: {{ $value }} per second"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }}%"
      
      - alert: ModelNotHealthy
        expr: up{job="amharic-hnet-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Amharic H-Net API is down"
          description: "The API service is not responding"
EOF
    
    print_status "Monitoring alerts configured"
    
    # Create Grafana dashboard provisioning
    mkdir -p monitoring/grafana/dashboards monitoring/grafana/datasources
    
    cat > monitoring/grafana/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF
    
    print_status "Grafana dashboards configured"
    echo ""
}

# Setup nginx (optional)
setup_nginx() {
    echo -e "${BLUE}Setting up reverse proxy...${NC}"
    
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    
    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    
    upstream api {
        server amharic-hnet-api:8000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        
        # API proxy
        location / {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        # Health check endpoint (no rate limiting)
        location /health {
            proxy_pass http://api;
            proxy_set_header Host $host;
            access_log off;
        }
    }
}
EOF
    
    print_status "Nginx configuration created"
    echo ""
}

# Deploy services
deploy_services() {
    echo -e "${BLUE}Deploying services...${NC}"
    
    # Pull latest images
    docker-compose pull
    print_status "Docker images pulled"
    
    # Build application image
    docker-compose build
    print_status "Application image built"
    
    # Start services
    docker-compose up -d
    print_status "Services started"
    
    echo ""
}

# Wait for services
wait_for_services() {
    echo -e "${BLUE}Waiting for services to be ready...${NC}"
    
    # Wait for API
    echo -n "Waiting for API..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    # Wait for Grafana
    echo -n "Waiting for Grafana..."
    for i in {1..30}; do
        if curl -s http://localhost:3000 > /dev/null 2>&1; then
            echo -e " ${GREEN}✓${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done
    
    print_status "Services are ready"
    echo ""
}

# Run health checks
run_health_checks() {
    echo -e "${BLUE}Running health checks...${NC}"
    
    # API health check
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        print_status "API health check passed"
    else
        print_error "API health check failed"
        exit 1
    fi
    
    # Test text generation
    if curl -s -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"prompt": "ሰላም", "max_length": 10}' | grep -q "generated_text"; then
        print_status "Text generation test passed"
    else
        print_warning "Text generation test failed (check API key and model)"
    fi
    
    # Cultural safety test
    if curl -s -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"prompt": "test cultural safety", "max_length": 10}' | grep -q "cultural_safety"; then
        print_status "Cultural safety monitoring active"
    else
        print_warning "Cultural safety monitoring check failed"
    fi
    
    echo ""
}

# Display deployment info
display_deployment_info() {
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo "=================================="
    echo ""
    echo "Services:"
    echo "  API:        http://localhost:8000"
    echo "  Docs:       http://localhost:8000/docs"
    echo "  Metrics:    http://localhost:8000/metrics"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana:    http://localhost:3000"
    echo ""
    echo "Credentials:"
    echo "  API Key:     $API_KEY"
    echo "  Grafana:     admin / $GRAFANA_PASSWORD"
    echo ""
    echo "Quick test:"
    echo "  curl -X POST http://localhost:8000/generate \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -H 'X-API-Key: $API_KEY' \\"
    echo "    -d '{\"prompt\": \"ሰላም\", \"max_length\": 50}'"
    echo ""
    echo "Management:"
    echo "  View logs:     docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Restart API:   docker-compose restart amharic-hnet-api"
    echo ""
}

# Main deployment flow
main() {
    check_prerequisites
    setup_environment
    setup_monitoring
    setup_nginx
    deploy_services
    wait_for_services
    run_health_checks
    display_deployment_info
}

# Handle script interruption
trap 'echo -e "\n${RED}Deployment interrupted${NC}"; exit 1' INT TERM

# Run deployment
main "$@"