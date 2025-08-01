#!/usr/bin/env python3
"""
Monitoring and Observability Module
==================================

Comprehensive monitoring, logging, and observability for production deployment.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
import psutil
import structlog
import torch
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry


class MetricsCollector:
    """
    Comprehensive metrics collection for production monitoring.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        
        # Application metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'http_active_connections',
            'Number of active HTTP connections',
            registry=self.registry
        )
        
        # Model metrics
        self.model_inference_duration = Histogram(
            'model_inference_duration_seconds',
            'Model inference duration',
            ['model_version'],
            registry=self.registry
        )
        
        self.model_inference_count = Counter(
            'model_inference_total',
            'Total model inferences',
            ['model_version', 'status'],
            registry=self.registry
        )
        
        self.model_memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Model memory usage in bytes',
            registry=self.registry
        )
        
        self.model_generation_length = Histogram(
            'model_generation_length_chars',
            'Generated text length distribution',
            buckets=[10, 50, 100, 200, 500, 1000, 2000],
            registry=self.registry
        )
        
        # Cultural safety metrics
        self.cultural_safety_checks = Counter(
            'cultural_safety_checks_total',
            'Total cultural safety checks',
            ['check_type', 'result'],
            registry=self.registry
        )
        
        self.cultural_safety_violations = Counter(
            'cultural_safety_violations_total',
            'Cultural safety violations',
            ['violation_type', 'severity'],
            registry=self.registry
        )
        
        self.cultural_safety_duration = Histogram(
            'cultural_safety_check_duration_seconds',
            'Cultural safety check duration',
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            self.gpu_memory_allocated = Gauge(
                'gpu_memory_allocated_bytes',
                'GPU memory allocated in bytes',
                registry=self.registry
            )
            
            self.gpu_memory_reserved = Gauge(
                'gpu_memory_reserved_bytes',
                'GPU memory reserved in bytes',
                registry=self.registry
            )
            
            self.gpu_utilization = Gauge(
                'gpu_utilization_percent',
                'GPU utilization percentage',
                registry=self.registry
            )
        
        # Cache metrics
        self.cache_hits = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'errors_total',
            'Total errors',
            ['error_type', 'severity'],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'app_info',
            'Application information',
            registry=self.registry
        )
        
        # Performance SLA metrics
        self.sla_response_time_target = Gauge(
            'sla_response_time_target_seconds',
            'SLA response time target',
            registry=self.registry
        )
        
        self.sla_compliance_rate = Gauge(
            'sla_compliance_rate_percent',
            'SLA compliance rate percentage',
            registry=self.registry
        )
        
        # Initialize with default values
        self.sla_response_time_target.set(0.2)  # 200ms target
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self):
        """Start background system metrics collection."""
        asyncio.create_task(self._collect_system_metrics())
    
    async def _collect_system_metrics(self):
        """Collect system metrics periodically."""
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self.system_cpu_usage.set(cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.system_memory_usage.set(memory.percent)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.system_disk_usage.set(disk.percent)
                
                # GPU metrics
                if torch.cuda.is_available():
                    self.gpu_memory_allocated.set(torch.cuda.memory_allocated())
                    self.gpu_memory_reserved.set(torch.cuda.memory_reserved())
                    
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        self.gpu_utilization.set(utilization.gpu)
                    except:
                        pass  # pynvml not available
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except Exception as e:
                structlog.get_logger().error("System metrics collection failed", error=str(e))
                await asyncio.sleep(30)  # Wait longer on error
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_inference(self, model_version: str, duration: float, status: str, generation_length: int):
        """Record model inference metrics."""
        self.model_inference_duration.labels(model_version=model_version).observe(duration)
        self.model_inference_count.labels(model_version=model_version, status=status).inc()
        self.model_generation_length.observe(generation_length)
    
    def record_cultural_safety_check(self, check_type: str, result: str, duration: float, 
                                   violations: Optional[list] = None):
        """Record cultural safety check metrics."""
        self.cultural_safety_checks.labels(check_type=check_type, result=result).inc()
        self.cultural_safety_duration.observe(duration)
        
        if violations:
            for violation in violations:
                self.cultural_safety_violations.labels(
                    violation_type=violation.violation_type,
                    severity=violation.severity
                ).inc()
    
    def record_cache_access(self, cache_type: str, hit: bool):
        """Record cache access metrics."""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_error(self, error_type: str, severity: str):
        """Record error metrics."""
        self.error_count.labels(error_type=error_type, severity=severity).inc()
    
    def update_model_memory_usage(self, memory_bytes: int):
        """Update model memory usage."""
        self.model_memory_usage.set(memory_bytes)
    
    def increment_active_connections(self):
        """Increment active connections counter."""
        self.active_connections.inc()
    
    def decrement_active_connections(self):
        """Decrement active connections counter."""
        self.active_connections.dec()
    
    def set_app_info(self, version: str, model_version: str, environment: str):
        """Set application information."""
        self.app_info.info({
            'version': version,
            'model_version': model_version,
            'environment': environment,
            'python_version': f"{psutil.python_version()[0]}.{psutil.python_version()[1]}",
            'torch_version': torch.__version__,
            'cuda_available': str(torch.cuda.is_available())
        })


def setup_logging(log_level: str = "INFO", environment: str = "production"):
    """
    Setup structured logging for production.
    """
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
    ]
    
    if environment == "development":
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level)
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(message)s" if environment != "development" else None,
    )
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


class PerformanceMonitor:
    """
    Performance monitoring and SLA tracking.
    """
    
    def __init__(self, target_response_time: float = 0.2):
        self.target_response_time = target_response_time
        self.response_times = []
        self.max_history = 1000
        self.logger = structlog.get_logger(__name__)
    
    def record_response_time(self, duration: float, endpoint: str):
        """Record response time and check SLA compliance."""
        self.response_times.append(duration)
        
        # Maintain history size
        if len(self.response_times) > self.max_history:
            self.response_times = self.response_times[-self.max_history:]
        
        # Log SLA violations
        if duration > self.target_response_time:
            self.logger.warning(
                "SLA violation detected",
                endpoint=endpoint,
                response_time=duration,
                target=self.target_response_time,
                violation_percentage=(duration - self.target_response_time) / self.target_response_time * 100
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.response_times:
            return {"message": "No response times recorded"}
        
        response_times = self.response_times
        compliant_responses = len([t for t in response_times if t <= self.target_response_time])
        
        return {
            "total_requests": len(response_times),
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p50_response_time": sorted(response_times)[len(response_times) // 2],
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
            "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)],
            "sla_compliance_rate": compliant_responses / len(response_times) * 100,
            "target_response_time": self.target_response_time,
            "violations": len(response_times) - compliant_responses
        }


class HealthChecker:
    """
    Comprehensive health checking for all system components.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.health_checks = {}
        self.last_check_time = {}
    
    async def check_model_health(self, model_service) -> Dict[str, Any]:
        """Check model service health."""
        try:
            if not model_service or not model_service.is_loaded():
                return {"status": "unhealthy", "reason": "model_not_loaded"}
            
            # Perform quick inference test
            health_result = await model_service.health_check()
            self.health_checks["model"] = health_result
            self.last_check_time["model"] = time.time()
            
            return health_result
            
        except Exception as e:
            self.logger.error("Model health check failed", error=str(e))
            result = {"status": "unhealthy", "error": str(e)}
            self.health_checks["model"] = result
            return result
    
    async def check_cultural_safety_health(self, cultural_safety_service) -> Dict[str, Any]:
        """Check cultural safety service health."""
        try:
            if not cultural_safety_service:
                return {"status": "unhealthy", "reason": "service_not_available"}
            
            health_result = await cultural_safety_service.health_check()
            self.health_checks["cultural_safety"] = health_result
            self.last_check_time["cultural_safety"] = time.time()
            
            return health_result
            
        except Exception as e:
            self.logger.error("Cultural safety health check failed", error=str(e))
            result = {"status": "unhealthy", "error": str(e)}
            self.health_checks["cultural_safety"] = result
            return result
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Define thresholds
            cpu_threshold = 90
            memory_threshold = 90
            disk_threshold = 95
            
            issues = []
            if cpu_percent > cpu_threshold:
                issues.append(f"High CPU usage: {cpu_percent}%")
            if memory.percent > memory_threshold:
                issues.append(f"High memory usage: {memory.percent}%")
            if disk.percent > disk_threshold:
                issues.append(f"High disk usage: {disk.percent}%")
            
            status = "unhealthy" if issues else "healthy"
            
            result = {
                "status": status,
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "issues": issues
            }
            
            if torch.cuda.is_available():
                result["gpu_memory_allocated"] = torch.cuda.memory_allocated()
                result["gpu_memory_reserved"] = torch.cuda.memory_reserved()
            
            self.health_checks["system"] = result
            self.last_check_time["system"] = time.time()
            
            return result
            
        except Exception as e:
            self.logger.error("System health check failed", error=str(e))
            result = {"status": "unhealthy", "error": str(e)}
            self.health_checks["system"] = result
            return result
    
    async def check_cache_health(self, cache_service) -> Dict[str, Any]:
        """Check cache service health."""
        try:
            if not cache_service:
                return {"status": "unhealthy", "reason": "service_not_available"}
            
            health_result = await cache_service.health_check()
            self.health_checks["cache"] = health_result
            self.last_check_time["cache"] = time.time()
            
            return health_result
            
        except Exception as e:
            self.logger.error("Cache health check failed", error=str(e))
            result = {"status": "unhealthy", "error": str(e)}
            self.health_checks["cache"] = result
            return result
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.health_checks:
            return {"status": "unknown", "message": "No health checks performed"}
        
        unhealthy_services = [
            service for service, check in self.health_checks.items()
            if check.get("status") != "healthy"
        ]
        
        overall_status = "unhealthy" if unhealthy_services else "healthy"
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "services": self.health_checks.copy(),
            "unhealthy_services": unhealthy_services,
            "last_check_times": self.last_check_time.copy(),
            "summary": f"{len(self.health_checks) - len(unhealthy_services)}/{len(self.health_checks)} services healthy"
        }