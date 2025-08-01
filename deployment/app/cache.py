#!/usr/bin/env python3
"""
Caching Service for Production Deployment
========================================

High-performance caching service with Redis backend for improved response times.
"""

import asyncio
import json
import time
from typing import Any, Dict, Optional, Union
import hashlib

import aioredis
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)


class CacheService:
    """
    Production-ready caching service with Redis backend.
    
    Features:
    - Async Redis operations
    - Automatic serialization/deserialization
    - Connection pooling
    - Circuit breaker pattern
    - Performance monitoring
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.connection_pool: Optional[aioredis.ConnectionPool] = None
        self.is_connected = False
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0
        self.total_operations = 0
        
        # Circuit breaker state
        self.circuit_breaker_failures = 0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60  # seconds
        self.circuit_breaker_last_failure = 0
        self.circuit_breaker_open = False
        
        logger.info("Cache service initialized", redis_url=redis_url)
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            # Create connection pool
            self.connection_pool = aioredis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={},
                health_check_interval=30
            )
            
            # Create Redis client
            self.redis = aioredis.Redis(connection_pool=self.connection_pool)
            
            # Test connection
            await self.redis.ping()
            self.is_connected = True
            
            logger.info("Cache service connected to Redis")
            
        except Exception as e:
            logger.error("Failed to initialize cache service", error=str(e))
            self.is_connected = False
            raise
    
    async def close(self) -> None:
        """Close Redis connection."""
        try:
            if self.redis:
                await self.redis.close()
            if self.connection_pool:
                await self.connection_pool.disconnect()
            
            self.is_connected = False
            logger.info("Cache service disconnected")
            
        except Exception as e:
            logger.error("Error closing cache service", error=str(e))
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows operations."""
        if not self.circuit_breaker_open:
            return True
        
        # Check if timeout has passed
        if time.time() - self.circuit_breaker_last_failure > self.circuit_breaker_timeout:
            self.circuit_breaker_open = False
            self.circuit_breaker_failures = 0
            logger.info("Circuit breaker closed, retrying cache operations")
            return True
        
        return False
    
    def _record_failure(self):
        """Record cache operation failure."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        self.cache_errors += 1
        
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            self.circuit_breaker_open = True
            logger.warning("Circuit breaker opened due to cache failures",
                         failures=self.circuit_breaker_failures)
    
    def _record_success(self):
        """Record successful cache operation."""
        self.circuit_breaker_failures = 0
        if self.circuit_breaker_open:
            self.circuit_breaker_open = False
            logger.info("Circuit breaker closed after successful operation")
    
    def _generate_cache_key(self, key: str) -> str:
        """Generate cache key with namespace."""
        # Add namespace and hash long keys
        namespace = "amharic_hnet"
        full_key = f"{namespace}:{key}"
        
        if len(full_key) > 250:  # Redis key length limit
            # Hash long keys
            key_hash = hashlib.sha256(full_key.encode()).hexdigest()
            return f"{namespace}:hash:{key_hash}"
        
        return full_key
    
    def _serialize_value(self, value: Any) -> str:
        """Serialize value for storage."""
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize cache value", error=str(e))
            raise
    
    def _deserialize_value(self, value: str) -> Any:
        """Deserialize value from storage."""
        try:
            return json.loads(value)
        except (TypeError, ValueError) as e:
            logger.error("Failed to deserialize cache value", error=str(e))
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if not self.is_connected or not self._check_circuit_breaker():
            return None
        
        self.total_operations += 1
        cache_key = self._generate_cache_key(key)
        
        try:
            start_time = time.time()
            value = await self.redis.get(cache_key)
            operation_time = time.time() - start_time
            
            if value is not None:
                self.cache_hits += 1
                self._record_success()
                
                deserialized_value = self._deserialize_value(value.decode('utf-8'))
                
                logger.debug("Cache hit",
                           key=key,
                           operation_time=operation_time)
                
                return deserialized_value
            else:
                self.cache_misses += 1
                logger.debug("Cache miss", key=key)
                return None
                
        except Exception as e:
            self._record_failure()
            logger.error("Cache get operation failed",
                        key=key,
                        error=str(e))
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Expiration time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or not self._check_circuit_breaker():
            return False
        
        self.total_operations += 1
        cache_key = self._generate_cache_key(key)
        
        try:
            start_time = time.time()
            serialized_value = self._serialize_value(value)
            
            result = await self.redis.setex(
                cache_key,
                expire,
                serialized_value
            )
            
            operation_time = time.time() - start_time
            self._record_success()
            
            logger.debug("Cache set operation",
                        key=key,
                        expire=expire,
                        operation_time=operation_time,
                        value_size=len(serialized_value))
            
            return result
            
        except Exception as e:
            self._record_failure()
            logger.error("Cache set operation failed",
                        key=key,
                        error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected or not self._check_circuit_breaker():
            return False
        
        self.total_operations += 1
        cache_key = self._generate_cache_key(key)
        
        try:
            result = await self.redis.delete(cache_key)
            self._record_success()
            
            logger.debug("Cache delete operation", key=key, found=bool(result))
            return bool(result)
            
        except Exception as e:
            self._record_failure()
            logger.error("Cache delete operation failed",
                        key=key,
                        error=str(e))
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.is_connected or not self._check_circuit_breaker():
            return False
        
        cache_key = self._generate_cache_key(key)
        
        try:
            result = await self.redis.exists(cache_key)
            return bool(result)
            
        except Exception as e:
            logger.error("Cache exists operation failed",
                        key=key,
                        error=str(e))
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a numeric value in cache.
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value or None if failed
        """
        if not self.is_connected or not self._check_circuit_breaker():
            return None
        
        cache_key = self._generate_cache_key(key)
        
        try:
            result = await self.redis.incrby(cache_key, amount)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            logger.error("Cache increment operation failed",
                        key=key,
                        error=str(e))
            return None
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get time-to-live for a key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds or None if failed
        """
        if not self.is_connected or not self._check_circuit_breaker():
            return None
        
        cache_key = self._generate_cache_key(key)
        
        try:
            ttl = await self.redis.ttl(cache_key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            logger.error("Cache TTL operation failed",
                        key=key,
                        error=str(e))
            return None
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching a pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected or not self._check_circuit_breaker():
            return 0
        
        try:
            cache_pattern = self._generate_cache_key(pattern)
            keys = await self.redis.keys(cache_pattern)
            
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info("Cleared cache pattern",
                          pattern=pattern,
                          keys_deleted=deleted)
                return deleted
            
            return 0
            
        except Exception as e:
            self._record_failure()
            logger.error("Cache clear pattern operation failed",
                        pattern=pattern,
                        error=str(e))
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform cache health check.
        
        Returns:
            Health check results
        """
        try:
            if not self.is_connected:
                return {"status": "unhealthy", "reason": "not_connected"}
            
            # Test basic operations
            start_time = time.time()
            test_key = "health_check_test"
            test_value = {"timestamp": time.time(), "test": True}
            
            # Set test value
            set_success = await self.set(test_key, test_value, expire=60)
            
            if not set_success:
                return {"status": "unhealthy", "reason": "set_operation_failed"}
            
            # Get test value
            retrieved_value = await self.get(test_key)
            
            if retrieved_value != test_value:
                return {"status": "unhealthy", "reason": "get_operation_failed"}
            
            # Clean up
            await self.delete(test_key)
            
            health_check_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "health_check_time": health_check_time,
                "circuit_breaker_open": self.circuit_breaker_open,
                "connection_status": "connected"
            }
            
        except Exception as e:
            logger.error("Cache health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Cache statistics
        """
        hit_rate = self.cache_hits / max(self.total_operations, 1) * 100
        miss_rate = self.cache_misses / max(self.total_operations, 1) * 100
        error_rate = self.cache_errors / max(self.total_operations, 1) * 100
        
        return {
            "total_operations": self.total_operations,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_errors": self.cache_errors,
            "hit_rate_percent": hit_rate,
            "miss_rate_percent": miss_rate,
            "error_rate_percent": error_rate,
            "circuit_breaker": {
                "open": self.circuit_breaker_open,
                "failures": self.circuit_breaker_failures,
                "threshold": self.circuit_breaker_threshold,
                "last_failure": self.circuit_breaker_last_failure
            },
            "connection_status": "connected" if self.is_connected else "disconnected"
        }
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Cache server information
        """
        if not self.is_connected:
            return {"status": "not_connected"}
        
        try:
            info = await self.redis.info()
            
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits"),
                "keyspace_misses": info.get("keyspace_misses"),
                "uptime_in_seconds": info.get("uptime_in_seconds")
            }
            
        except Exception as e:
            logger.error("Failed to get cache info", error=str(e))
            return {"error": str(e)}