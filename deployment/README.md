# Amharic H-Net Production Deployment

This directory contains everything needed to deploy the Amharic H-Net model as a production-ready FastAPI service with comprehensive monitoring and observability.

## 🎯 Features

- **Sub-200ms Response Times**: Optimized inference pipeline with caching
- **Cultural Safety Monitoring**: Real-time validation and violation detection
- **Comprehensive Observability**: Prometheus metrics, Grafana dashboards, structured logging
- **Production Security**: Authentication, rate limiting, security headers
- **Auto-scaling Ready**: Container orchestration with health checks
- **High Availability**: Redis caching, circuit breakers, graceful error handling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx         │    │   FastAPI       │    │   H-Net Model   │
│   (Reverse      │────│   Application   │────│   Service       │
│    Proxy)       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │    │   Prometheus    │    │   Redis         │
│   (Dashboards)  │────│   (Metrics)     │    │   (Cache)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Model checkpoint at `../outputs/test_checkpoint.pt`
- At least 8GB RAM (4GB minimum)
- Optional: NVIDIA GPU for faster inference

### Deploy

```bash
cd deployment
./deploy.sh
```

The deployment script will:
1. Check prerequisites
2. Set up environment and configuration
3. Configure monitoring and alerting
4. Deploy all services with Docker Compose
5. Run health checks
6. Display access information

### Access Points

After deployment:

- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Metrics**: http://localhost:8000/metrics
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## 📊 Monitoring and Observability

### Metrics Collected

- **Application Metrics**:
  - HTTP request rates and response times
  - Model inference performance
  - Cultural safety violation rates
  - Cache hit/miss rates

- **System Metrics**:
  - CPU, memory, disk usage
  - GPU memory (if available)
  - Network connections

- **Business Metrics**:
  - Text generation quality
  - Cultural safety compliance
  - API usage patterns

### Grafana Dashboards

Pre-configured dashboards include:
- API Performance Overview
- Model Inference Metrics
- Cultural Safety Monitoring
- System Resource Usage
- Error Rate Analysis

### Alerting

Automated alerts for:
- Response time > 200ms (SLA violation)
- Cultural safety violations
- High error rates (>5%)
- Service unavailability
- Resource exhaustion

## 🔐 Security

### Authentication

- API key-based authentication
- Configurable rate limiting
- Request/response validation

### Security Headers

- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security (for HTTPS)

### Cultural Safety

- Real-time content validation
- Violation logging and monitoring
- Cultural context awareness
- Respectful treatment enforcement

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Application
ENVIRONMENT=production
API_KEY=your-secure-api-key
HOST=0.0.0.0
PORT=8000

# Performance
RESPONSE_TIME_TARGET_MS=200
ENABLE_CACHING=true
REDIS_URL=redis://redis:6379/0

# Security
ENABLE_AUTH=true
ENABLE_RATE_LIMITING=true
REQUESTS_PER_MINUTE=60

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO

# Cultural Safety
ENABLE_CULTURAL_SAFETY=true
CULTURAL_SAFETY_STRICT_MODE=true
```

### Model Configuration

The model service automatically loads configuration from the checkpoint:
- Model architecture parameters
- Cultural safety settings
- Performance optimizations

## 🧪 Testing

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Readiness probe
curl http://localhost:8000/ready
```

### API Testing

```bash
# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "prompt": "ሰላም",
    "max_length": 50,
    "temperature": 1.0,
    "enable_cultural_safety": true
  }'

# Get model info
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/model/info

# Cultural safety guidelines
curl http://localhost:8000/cultural-safety/guidelines
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Run load test
ab -n 1000 -c 10 -H "X-API-Key: your-api-key" \
   -p test_payload.json -T application/json \
   http://localhost:8000/generate
```

## 📈 Performance Optimization

### Response Time Targets

- **Text Generation**: < 200ms (including cultural safety)
- **Cultural Safety Check**: < 50ms
- **Health Checks**: < 10ms

### Optimization Features

1. **Model Optimization**:
   - PyTorch model compilation
   - GPU acceleration (when available)
   - Batch processing support

2. **Caching**:
   - Redis-based response caching
   - Intelligent cache keys
   - Configurable TTL

3. **Infrastructure**:
   - Connection pooling
   - Async processing
   - Resource monitoring

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Loading**:
   ```bash
   # Check model file exists
   ls -la models/test_checkpoint.pt
   
   # Check container logs
   docker-compose logs amharic-hnet-api
   ```

2. **High Response Times**:
   ```bash
   # Check system resources
   docker stats
   
   # Check cache status
   curl http://localhost:8000/health
   ```

3. **Cultural Safety Violations**:
   ```bash
   # Check violation reports
   curl http://localhost:8000/cultural-safety/guidelines
   
   # Monitor metrics
   curl http://localhost:8000/metrics | grep cultural_safety
   ```

### Log Analysis

```bash
# View all logs
docker-compose logs -f

# API logs only
docker-compose logs -f amharic-hnet-api

# Filter errors
docker-compose logs amharic-hnet-api | grep ERROR
```

## 🔄 Updates and Maintenance

### Model Updates

1. Replace model checkpoint in `models/` directory
2. Restart API service: `docker-compose restart amharic-hnet-api`
3. Verify health checks pass

### Configuration Updates

1. Modify `.env` file
2. Restart affected services: `docker-compose up -d`
3. Check service health

### Monitoring Updates

1. Update dashboard configurations in `monitoring/`
2. Restart monitoring stack: `docker-compose restart prometheus grafana`

## 📊 Production Checklist

Before deploying to production:

- [ ] Set secure API keys
- [ ] Configure allowed origins/hosts
- [ ] Set up HTTPS with valid certificates
- [ ] Configure external Redis (for scale)
- [ ] Set up log aggregation
- [ ] Configure backup procedures
- [ ] Test disaster recovery
- [ ] Set up external monitoring/alerting
- [ ] Document runbooks
- [ ] Train operations team

## 🤝 Support

For issues or questions:

1. Check the logs: `docker-compose logs`
2. Verify configuration: `cat .env`
3. Run health checks: `curl http://localhost:8000/health`
4. Check metrics: Visit Grafana dashboards
5. Review cultural safety guidelines

## 📄 License

This deployment configuration is part of the Amharic H-Net project.