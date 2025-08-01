#!/usr/bin/env python3
"""
Configuration Management for Production Deployment
=================================================

Centralized configuration with environment-specific settings,
security features, and production optimizations.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "amharic-hnet-api"
    app_version: str = "1.0.0"
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("production", env="ENVIRONMENT")
    
    # Server
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    workers: int = Field(1, env="WORKERS")  # Single worker for model consistency
    
    # Model
    model_path: str = Field("/Users/mekdesyared/amharic-hnet-v2/outputs/compact/final_model.pt", env="MODEL_PATH")
    model_version: str = Field("1.1.0", env="MODEL_VERSION")
    enable_model_compilation: bool = Field(True, env="ENABLE_MODEL_COMPILATION")
    
    # Performance
    max_batch_size: int = Field(8, env="MAX_BATCH_SIZE")
    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")
    max_generation_length: int = Field(500, env="MAX_GENERATION_LENGTH")
    response_time_target_ms: int = Field(200, env="RESPONSE_TIME_TARGET_MS")
    
    # Security
    api_key: Optional[str] = Field(None, env="API_KEY")
    allowed_origins: List[str] = Field(["*"], env="ALLOWED_ORIGINS")
    allowed_hosts: List[str] = Field(["*"], env="ALLOWED_HOSTS")
    enable_auth: bool = Field(False, env="ENABLE_AUTH")
    secret_key: str = Field("your-secret-key-change-in-production", env="SECRET_KEY")
    
    # Redis/Caching
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 hour
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    
    # Monitoring
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_port: int = Field(9090, env="METRICS_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_tracing: bool = Field(False, env="ENABLE_TRACING")
    
    # Cultural Safety
    enable_cultural_safety: bool = Field(True, env="ENABLE_CULTURAL_SAFETY")
    cultural_safety_strict_mode: bool = Field(True, env="CULTURAL_SAFETY_STRICT_MODE")
    max_violation_history: int = Field(1000, env="MAX_VIOLATION_HISTORY")
    
    # Rate Limiting
    enable_rate_limiting: bool = Field(True, env="ENABLE_RATE_LIMITING")
    requests_per_minute: int = Field(60, env="REQUESTS_PER_MINUTE")
    burst_size: int = Field(10, env="BURST_SIZE")
    
    # Health Checks
    health_check_interval: int = Field(30, env="HEALTH_CHECK_INTERVAL")
    readiness_probe_timeout: int = Field(5, env="READINESS_PROBE_TIMEOUT")
    
    # Resource Limits
    max_memory_gb: float = Field(8.0, env="MAX_MEMORY_GB")
    gpu_memory_fraction: float = Field(0.8, env="GPU_MEMORY_FRACTION")
    
    @validator('allowed_origins', pre=True)
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @validator('allowed_hosts', pre=True)
    def parse_allowed_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',')]
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of: {valid_envs}')
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    enable_auth: bool = False
    workers: int = 1
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    allowed_hosts: List[str] = ["localhost", "127.0.0.1"]


class StagingSettings(Settings):
    """Staging environment settings."""
    debug: bool = False
    log_level: str = "INFO"
    enable_auth: bool = True
    workers: int = 1
    enable_tracing: bool = True


class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    log_level: str = "WARNING"
    enable_auth: bool = True
    workers: int = 1  # Single worker for model consistency
    allowed_origins: List[str] = []  # Must be explicitly set
    allowed_hosts: List[str] = []    # Must be explicitly set
    enable_tracing: bool = True
    cultural_safety_strict_mode: bool = True


@lru_cache()
def get_settings() -> Settings:
    """Get settings based on environment."""
    environment = os.getenv("ENVIRONMENT", "production").lower()
    
    if environment == "development":
        return DevelopmentSettings()
    elif environment == "staging":
        return StagingSettings()
    else:
        return ProductionSettings()


# Environment-specific configurations
CORS_CONFIG = {
    "development": {
        "allow_origins": ["http://localhost:3000", "http://localhost:8080"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"]
    },
    "production": {
        "allow_origins": [],  # Must be explicitly configured
        "allow_credentials": True,
        "allow_methods": ["GET", "POST"],
        "allow_headers": ["Authorization", "Content-Type"]
    }
}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}

MONITORING_CONFIG = {
    "prometheus": {
        "metrics_path": "/metrics",
        "port": 9090
    },
    "logging": {
        "format": "json",
        "level": "INFO",
        "handlers": ["console", "file"]
    },
    "tracing": {
        "service_name": "amharic-hnet-api",
        "sample_rate": 0.1
    }
}

PERFORMANCE_CONFIG = {
    "response_time_targets": {
        "text_generation": 0.2,  # 200ms
        "cultural_safety_check": 0.05,  # 50ms
        "health_check": 0.01,  # 10ms
    },
    "cache_settings": {
        "default_ttl": 3600,
        "max_cache_size": "512MB",
        "eviction_policy": "LRU"
    },
    "resource_limits": {
        "max_request_size": "10MB",
        "max_response_size": "5MB",
        "connection_timeout": 30,
        "keep_alive_timeout": 5
    }
}