#!/usr/bin/env python3
"""
Production FastAPI Server for Amharic H-Net Model
================================================

High-performance, production-ready API server with comprehensive monitoring,
cultural safety validation, and sub-200ms response times.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

import structlog
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from pydantic import BaseModel, Field, validator

from app.model_service import ModelService
from app.cultural_safety import CulturalSafetyService
from app.monitoring import MetricsCollector, setup_logging
from app.config import get_settings
from app.auth import get_api_key
from app.cache import CacheService

# Import morphological analyzer
import sys
sys.path.append('/Users/mekdesyared/amharic-hnet-v2/src')
from linguistic_analysis.morphological_analyzer import AmharicMorphologicalAnalyzer


# Initialize structured logging
logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
MODEL_INFERENCE_TIME = Histogram('model_inference_seconds', 'Model inference time')
CULTURAL_SAFETY_VIOLATIONS = Counter('cultural_safety_violations_total', 'Cultural safety violations', ['severity'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
MODEL_MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Model memory usage in bytes')


# Request/Response Models
class TextGenerationRequest(BaseModel):
    """Request model for text generation."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Input prompt for text generation")
    max_length: int = Field(100, ge=1, le=500, description="Maximum generation length")
    temperature: float = Field(1.0, ge=0.1, le=2.0, description="Generation temperature")
    top_k: int = Field(50, ge=1, le=100, description="Top-k sampling parameter")
    enable_cultural_safety: bool = Field(True, description="Enable cultural safety checks")
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()


class TextGenerationResponse(BaseModel):
    """Response model for text generation."""
    generated_text: str
    input_prompt: str
    generation_stats: Dict[str, Union[str, int, float]]
    cultural_safety: Dict[str, Union[bool, str, List[str]]]
    performance_metrics: Dict[str, float]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    model_loaded: bool
    cultural_safety_active: bool
    system_metrics: Dict[str, Union[str, float]]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    error_code: str
    timestamp: str
    request_id: Optional[str] = None


class MorphemeAnalysisRequest(BaseModel):
    """Request model for morpheme analysis."""
    text: str = Field(..., min_length=1, max_length=2000, description="Amharic text to analyze")
    include_pos_tags: bool = Field(True, description="Include part-of-speech tags")
    include_features: bool = Field(True, description="Include morphological features")
    include_cultural_context: bool = Field(True, description="Include cultural context analysis")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v.strip()


class MorphemeAnalysisResponse(BaseModel):
    """Response model for morpheme analysis."""
    original_text: str
    word_analyses: List[Dict[str, Union[str, List[str], Dict[str, str], float]]]
    text_complexity: float
    dialect_classification: str
    cultural_safety_score: float
    linguistic_quality_score: float
    readability_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]


# Global services
model_service: Optional[ModelService] = None
cultural_safety_service: Optional[CulturalSafetyService] = None
metrics_collector: Optional[MetricsCollector] = None
cache_service: Optional[CacheService] = None
morphological_analyzer: Optional[AmharicMorphologicalAnalyzer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global model_service, cultural_safety_service, metrics_collector, cache_service, morphological_analyzer
    
    settings = get_settings()
    logger.info("Starting Amharic H-Net API server", version=settings.app_version)
    
    try:
        # Initialize services
        logger.info("Loading model service...")
        model_service = ModelService(settings.model_path)
        await model_service.load_model()
        
        logger.info("Initializing cultural safety service...")
        cultural_safety_service = CulturalSafetyService()
        
        logger.info("Setting up monitoring...")
        metrics_collector = MetricsCollector()
        
        logger.info("Initializing cache service...")
        cache_service = CacheService(settings.redis_url)
        await cache_service.initialize()
        
        logger.info("Initializing morphological analyzer...")
        morphological_analyzer = AmharicMorphologicalAnalyzer()
        
        # Update memory usage metric
        if torch.cuda.is_available():
            MODEL_MEMORY_USAGE.set(torch.cuda.memory_allocated())
        
        logger.info("All services initialized successfully")
        yield
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services...")
        if cache_service:
            await cache_service.close()
        logger.info("Services shut down successfully")


# Initialize FastAPI app
app = FastAPI(
    title="Amharic H-Net Language Model API",
    description="Production-ready API for Amharic text generation with cultural safety monitoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=get_settings().allowed_hosts
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware for collecting metrics and monitoring."""
    start_time = time.time()
    
    # Increment active connections
    ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration:.4f}s"
        response.headers["X-Model-Version"] = get_settings().model_version
        
        return response
        
    except Exception as e:
        # Record error metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        raise
    finally:
        ACTIVE_CONNECTIONS.dec()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error("Unhandled exception", error=str(exc), path=request.url.path)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check service health
        model_healthy = model_service is not None and model_service.is_loaded()
        safety_healthy = cultural_safety_service is not None
        cache_healthy = cache_service is not None and await cache_service.health_check()
        
        # System metrics
        import psutil
        system_metrics = {
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
        }
        
        if torch.cuda.is_available():
            system_metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated()
            system_metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved()
        
        overall_status = "healthy" if all([model_healthy, safety_healthy, cache_healthy]) else "unhealthy"
        
        return HealthResponse(
            status=overall_status,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            version=get_settings().app_version,
            model_loaded=model_healthy,
            cultural_safety_active=safety_healthy,
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Health check failed")


@app.get("/ready")
async def readiness_check():
    """Readiness probe for Kubernetes."""
    if not model_service or not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if not cultural_safety_service:
        raise HTTPException(status_code=503, detail="Cultural safety service not ready")
    
    return {"status": "ready"}


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(
    request: TextGenerationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Generate Amharic text with cultural safety monitoring.
    
    This endpoint provides high-performance text generation with:
    - Sub-200ms response times for typical requests
    - Cultural safety validation and monitoring
    - Comprehensive performance metrics
    - Caching for improved performance
    """
    start_time = time.time()
    
    try:
        # Check cache first
        cache_key = f"generate:{hash(request.prompt)}:{request.max_length}:{request.temperature}:{request.top_k}"
        cached_result = await cache_service.get(cache_key)
        
        if cached_result:
            logger.info("Serving cached result", prompt_hash=hash(request.prompt))
            return TextGenerationResponse(**cached_result)
        
        # Validate services are ready
        if not model_service or not model_service.is_loaded():
            raise HTTPException(status_code=503, detail="Model service not available")
        
        if not cultural_safety_service:
            raise HTTPException(status_code=503, detail="Cultural safety service not available")
        
        # Pre-generation cultural safety check
        if request.enable_cultural_safety:
            safety_start = time.time()
            is_safe, violations = cultural_safety_service.check_input_safety(request.prompt)
            safety_duration = time.time() - safety_start
            
            if not is_safe:
                # Record violation metrics
                for violation in violations:
                    CULTURAL_SAFETY_VIOLATIONS.labels(severity=violation.severity).inc()
                
                raise HTTPException(
                    status_code=400,
                    detail=f"Input violates cultural safety guidelines: {[v.context for v in violations]}"
                )
            
            logger.info("Input passed cultural safety check", duration=safety_duration)
        
        # Generate text
        inference_start = time.time()
        generated_text, generation_stats = await model_service.generate_text(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k
        )
        inference_duration = time.time() - inference_start
        
        # Record inference metrics
        MODEL_INFERENCE_TIME.observe(inference_duration)
        
        # Post-generation cultural safety check
        cultural_safety_result = {"passed": True, "violations": [], "feedback": ""}
        if request.enable_cultural_safety:
            safety_start = time.time()
            is_safe, feedback = cultural_safety_service.validate_generation(
                generated_text, request.prompt
            )
            safety_duration = time.time() - safety_start
            
            cultural_safety_result = {
                "passed": is_safe,
                "violations": [] if is_safe else ["Cultural safety violations detected"],
                "feedback": feedback,
                "check_duration": safety_duration
            }
            
            if not is_safe:
                logger.warning("Generated text failed cultural safety check", 
                             prompt_hash=hash(request.prompt), feedback=feedback)
        
        # Prepare response
        total_duration = time.time() - start_time
        
        response = TextGenerationResponse(
            generated_text=generated_text,
            input_prompt=request.prompt,
            generation_stats=generation_stats,
            cultural_safety=cultural_safety_result,
            performance_metrics={
                "total_duration": total_duration,
                "inference_duration": inference_duration,
                "cultural_safety_duration": cultural_safety_result.get("check_duration", 0),
                "response_time_target_met": total_duration < 0.2  # 200ms target
            }
        )
        
        # Cache the result asynchronously
        background_tasks.add_task(
            cache_service.set,
            cache_key,
            response.dict(),
            expire=get_settings().cache_ttl
        )
        
        # Log performance metrics
        logger.info(
            "Text generation completed",
            total_duration=total_duration,
            inference_duration=inference_duration,
            prompt_length=len(request.prompt),
            generated_length=len(generated_text),
            target_met=total_duration < 0.2
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text generation failed", error=str(e), prompt_hash=hash(request.prompt))
        raise HTTPException(status_code=500, detail="Text generation failed")


@app.post("/batch-generate")
async def batch_generate_text(
    requests: List[TextGenerationRequest],
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Batch text generation endpoint for processing multiple requests efficiently.
    """
    if len(requests) > get_settings().max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum allowed: {get_settings().max_batch_size}"
        )
    
    start_time = time.time()
    results = []
    
    try:
        # Process requests concurrently
        tasks = [
            generate_text(req, background_tasks, api_key)
            for req in requests
        ]
        
        # Wait for all generations to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "request_index": i
                })
            else:
                processed_results.append(result)
        
        total_duration = time.time() - start_time
        
        return {
            "results": processed_results,
            "batch_stats": {
                "total_requests": len(requests),
                "successful": len([r for r in results if not isinstance(r, Exception)]),
                "failed": len([r for r in results if isinstance(r, Exception)]),
                "total_duration": total_duration,
                "avg_duration_per_request": total_duration / len(requests)
            }
        }
        
    except Exception as e:
        logger.error("Batch generation failed", error=str(e), batch_size=len(requests))
        raise HTTPException(status_code=500, detail="Batch generation failed")


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest().decode('utf-8')


@app.get("/model/info")
async def get_model_info(api_key: str = Depends(get_api_key)):
    """Get model information and statistics."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model service not available")
    
    return await model_service.get_model_info()


@app.get("/cultural-safety/guidelines")
async def get_cultural_safety_guidelines():
    """Get cultural safety guidelines and information."""
    if not cultural_safety_service:
        raise HTTPException(status_code=503, detail="Cultural safety service not available")
    
    return cultural_safety_service.get_guidelines()


@app.post("/analyze-morphemes", response_model=MorphemeAnalysisResponse)
async def analyze_morphemes(
    request: MorphemeAnalysisRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Perform morphological analysis of Amharic text.
    
    This endpoint provides comprehensive linguistic analysis including:
    - Morpheme segmentation and classification
    - Part-of-speech tagging
    - Morphological feature extraction
    - Dialect classification
    - Cultural context analysis
    - Text complexity and readability metrics
    """
    start_time = time.time()
    
    try:
        # Check services availability
        if not morphological_analyzer:
            raise HTTPException(status_code=503, detail="Morphological analyzer not available")
        
        # Check cache first
        cache_key = f"morpheme:{hash(request.text)}:{request.include_pos_tags}:{request.include_features}:{request.include_cultural_context}"
        cached_result = await cache_service.get(cache_key)
        
        if cached_result:
            logger.info("Serving cached morphological analysis", text_hash=hash(request.text))
            return MorphemeAnalysisResponse(**cached_result)
        
        # Perform morphological analysis
        analysis_start = time.time()
        linguistic_annotation = morphological_analyzer.analyze_text(request.text)
        analysis_duration = time.time() - analysis_start
        
        # Convert word analyses to serializable format
        word_analyses_data = []
        for word_analysis in linguistic_annotation.word_analyses:
            analysis_dict = {
                "word": word_analysis.word,
                "morphemes": word_analysis.morphemes,
                "morpheme_types": [mt.value for mt in word_analysis.morpheme_types],
                "confidence_score": word_analysis.confidence_score,
                "dialect_markers": word_analysis.dialect_markers,
                "cultural_domain": word_analysis.cultural_domain
            }
            
            if request.include_pos_tags:
                analysis_dict["pos_tag"] = word_analysis.pos_tag.value
            
            if request.include_features:
                analysis_dict["morphological_features"] = word_analysis.morphological_features
            
            if not request.include_cultural_context:
                analysis_dict.pop("cultural_domain", None)
                analysis_dict.pop("dialect_markers", None)
            
            word_analyses_data.append(analysis_dict)
        
        # Prepare response
        total_duration = time.time() - start_time
        
        response = MorphemeAnalysisResponse(
            original_text=request.text,
            word_analyses=word_analyses_data,
            text_complexity=linguistic_annotation.text_complexity,
            dialect_classification=linguistic_annotation.dialect_classification,
            cultural_safety_score=linguistic_annotation.cultural_safety_score,
            linguistic_quality_score=linguistic_annotation.linguistic_quality_score,
            readability_metrics=linguistic_annotation.readability_metrics,
            performance_metrics={
                "total_duration": total_duration,
                "analysis_duration": analysis_duration,
                "words_analyzed": len(word_analyses_data),
                "analysis_rate": len(word_analyses_data) / analysis_duration if analysis_duration > 0 else 0
            }
        )
        
        # Cache the result asynchronously
        background_tasks.add_task(
            cache_service.set,
            cache_key,
            response.dict(),
            expire=get_settings().cache_ttl
        )
        
        # Log performance metrics
        logger.info(
            "Morphological analysis completed",
            total_duration=total_duration,
            analysis_duration=analysis_duration,
            text_length=len(request.text),
            words_analyzed=len(word_analyses_data),
            text_complexity=linguistic_annotation.text_complexity,
            dialect=linguistic_annotation.dialect_classification
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Morphological analysis failed", error=str(e), text_hash=hash(request.text))
        raise HTTPException(status_code=500, detail="Morphological analysis failed")


if __name__ == "__main__":
    setup_logging()
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_config=None,  # Use our custom logging
        access_log=False,  # Handle access logs with middleware
    )