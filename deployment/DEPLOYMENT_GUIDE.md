# Amharic H-Net v2 Production Deployment Guide

This guide covers the deployment of the retrained Amharic H-Net model (loss 0.2999, 91.7% success rate) as a production API with comprehensive monitoring and cultural safety validation.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available
- Python 3.8+ for testing scripts

### One-Command Deployment
```bash
cd deployment
python deploy_production.py
```

This will:
1. ✅ Validate all prerequisites
2. 📦 Prepare deployment files and copy the trained model
3. 🐳 Build Docker images and start services
4. ⏳ Wait for services to be ready
5. 🧪 Run comprehensive tests
6. 📊 Provide deployment summary

## 📋 API Endpoints

### Core Endpoints

#### 1. Text Generation
```http
POST /generate
Content-Type: application/json

{
  "prompt": "ሰላም እንዴት ነሽ",
  "max_length": 100,
  "temperature": 1.0,
  "top_k": 50,
  "enable_cultural_safety": true
}
```

**Response:**
```json
{
  "generated_text": "ሰላም እንዴት ነሽ? ደህና ነኝ አመሰግናለሁ...",
  "input_prompt": "ሰላም እንዴት ነሽ",
  "generation_stats": {
    "inference_time": 0.125,
    "input_length": 12,
    "generated_length": 45,
    "generation_speed": 360,
    "temperature": 1.0,
    "top_k": 50,
    "device": "cpu"
  },
  "cultural_safety": {
    "passed": true,
    "violations": [],
    "feedback": "Generated text is culturally appropriate and safe.",
    "check_duration": 0.032
  },
  "performance_metrics": {
    "total_duration": 0.157,
    "inference_duration": 0.125,
    "cultural_safety_duration": 0.032,
    "response_time_target_met": true
  }
}
```

#### 2. Morpheme Analysis
```http
POST /analyze-morphemes
Content-Type: application/json

{
  "text": "ሰላም እንዴት ነሽ? ቡና ትፈልጊ",
  "include_pos_tags": true,
  "include_features": true,
  "include_cultural_context": true
}
```

**Response:**
```json
{
  "original_text": "ሰላም እንዴት ነሽ? ቡና ትፈልጊ",
  "word_analyses": [
    {
      "word": "ሰላም",
      "morphemes": ["ሰላም"],
      "morpheme_types": ["root"],
      "pos_tag": "NOUN",
      "morphological_features": {
        "number": "singular",
        "definiteness": "indefinite"
      },
      "confidence_score": 0.85,
      "dialect_markers": ["standard"],
      "cultural_domain": "general"
    }
  ],
  "text_complexity": 0.42,
  "dialect_classification": "standard",
  "cultural_safety_score": 1.0,
  "linguistic_quality_score": 0.78,
  "readability_metrics": {
    "avg_words_per_sentence": 3.5,
    "avg_syllables_per_word": 2.1,
    "avg_morphemes_per_word": 1.2,
    "amharic_readability_score": 85.2
  },
  "performance_metrics": {
    "total_duration": 0.089,
    "analysis_duration": 0.076,
    "words_analyzed": 4,
    "analysis_rate": 52.6
  }
}
```

#### 3. Cultural Safety Validation
```http
POST /generate
Content-Type: application/json

{
  "prompt": "ቡና is addictive and dangerous",
  "max_length": 50,
  "enable_cultural_safety": true
}
```

**Response (Cultural Violation Detected):**
```http
HTTP 400 Bad Request

{
  "error": "Input violates cultural safety guidelines: ['Inappropriate association with cultural term']",
  "error_code": "HTTP_400",
  "timestamp": "2025-08-01T12:00:00Z"
}
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

#### Readiness Probe
```http
GET /ready
```

#### Model Information
```http
GET /model/info
```

#### Prometheus Metrics
```http
GET /metrics
```

#### Cultural Safety Guidelines
```http
GET /cultural-safety/guidelines
```

## 🐳 Docker Services

The deployment includes the following services:

### Main API Service (`amharic-hnet-api`)
- **Port:** 8000
- **Model:** Final trained model (48.5MB)
- **Features:** Text generation, morpheme analysis, cultural safety
- **Resources:** 8GB RAM limit, 4 CPU cores

### Redis Cache (`redis`)
- **Port:** 6379
- **Purpose:** Response caching, session storage
- **Configuration:** 512MB max memory, LRU eviction

### Prometheus Monitoring (`prometheus`)
- **Port:** 9090
- **Purpose:** Metrics collection and alerting
- **Retention:** 30 days

### Grafana Dashboard (`grafana`)
- **Port:** 3000
- **Purpose:** Metrics visualization
- **Default Login:** admin/admin123

### Nginx Reverse Proxy (`nginx`)
- **Ports:** 80, 443
- **Purpose:** Load balancing, SSL termination
- **Optional:** Enable for production

## 📊 Performance Characteristics

### Response Time Targets
- **Text Generation:** < 200ms
- **Morpheme Analysis:** < 100ms
- **Cultural Safety Check:** < 50ms
- **Health Check:** < 10ms

### Model Performance
- **Training Loss:** 0.2999
- **Success Rate:** 91.7%
- **Model Size:** 48.5MB
- **Inference Speed:** ~360 chars/second

### Throughput
- **Single Request:** Sub-200ms
- **Concurrent Requests:** 10+ requests/second
- **Batch Processing:** Up to 8 requests per batch

## 🔧 Configuration

### Environment Variables
```bash
# Core Configuration
ENVIRONMENT=production
MODEL_PATH=/app/models/final_model.pt
MODEL_VERSION=1.1.0

# Performance
MAX_BATCH_SIZE=8
RESPONSE_TIME_TARGET_MS=200
CACHE_TTL=3600

# Security
ENABLE_AUTH=true
API_KEY=your-secure-api-key
CULTURAL_SAFETY_STRICT_MODE=true

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
ENABLE_TRACING=true
```

### Production Recommendations
1. **Set secure API keys** in `.env` file
2. **Configure allowed origins** for CORS
3. **Enable SSL/TLS** with proper certificates
4. **Set up log aggregation** (ELK stack, Fluentd)
5. **Configure alerting** based on metrics
6. **Set up backup** for model and configuration

## 🧪 Testing

### Automated Testing
```bash
# Run all deployment tests
python test_deployment.py

# Run with custom API key
python test_deployment.py --api-key your-api-key

# Save test results
python test_deployment.py --output test_results.json
```

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# Text generation
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "ሰላም", "max_length": 50}'

# Morpheme analysis
curl -X POST http://localhost:8000/analyze-morphemes \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text": "ሰላም እንዴት ነሽ", "include_pos_tags": true}'
```

## 📈 Monitoring

### Key Metrics
- **Request Rate:** `http_requests_total`
- **Response Time:** `http_request_duration_seconds`
- **Model Inference Time:** `model_inference_seconds`
- **Cultural Safety Violations:** `cultural_safety_violations_total`
- **System Resources:** CPU, memory, GPU usage

### Alerting Rules
- Response time > 200ms for 5 minutes
- Error rate > 5% for 2 minutes
- Cultural safety violations > 10/hour
- System memory usage > 90%

### Dashboards
- **API Performance:** Request rates, response times, error rates
- **Model Performance:** Inference times, generation quality metrics
- **Cultural Safety:** Violation rates, safety check performance
- **System Health:** Resource usage, service status

## 🔒 Security

### Authentication
- API key based authentication
- Rate limiting (60 requests/minute)
- Request size limits (10MB max)

### Cultural Safety
- Real-time input validation
- Post-generation content checking
- Violation logging and alerting
- Cultural context awareness

### Infrastructure Security
- Non-root container execution
- Read-only model files
- Network isolation
- Security headers

## 🚦 Troubleshooting

### Common Issues

#### Model Loading Fails
```bash
# Check model file exists and has correct permissions
ls -la deployment/models/final_model.pt

# Check container logs
docker-compose logs amharic-hnet-api
```

#### Services Not Starting
```bash
# Check Docker resources
docker system df
docker system prune

# Restart services
docker-compose down
docker-compose up -d
```

#### Performance Issues
```bash
# Check system resources
docker stats

# Check Redis cache
docker-compose exec redis redis-cli info stats

# Check model memory usage
curl http://localhost:8000/model/info
```

### Log Analysis
```bash
# View API logs
docker-compose logs -f amharic-hnet-api

# View all service logs
docker-compose logs -f

# Export logs for analysis
docker-compose logs --no-color > deployment_logs.txt
```

## 📞 Support

### Health Checks
- **API Health:** `GET /health`
- **Service Status:** `docker-compose ps`
- **Resource Usage:** `docker stats`

### Metrics Endpoints
- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000
- **API Metrics:** http://localhost:8000/metrics

### Documentation
- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🎯 Next Steps

1. **Scale Horizontally:** Add more API instances behind load balancer
2. **Improve Caching:** Implement distributed caching with Redis Cluster
3. **Add Authentication:** Integrate with OAuth2/JWT for enterprise users
4. **Enhanced Monitoring:** Set up ELK stack for log aggregation
5. **CI/CD Pipeline:** Automate testing and deployment
6. **Model Versioning:** Implement A/B testing for model updates

For questions or issues, check the logs, monitoring dashboards, or create an issue in the project repository.