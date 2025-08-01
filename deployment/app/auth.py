#!/usr/bin/env python3
"""
Authentication and Authorization Module
======================================

Production-ready authentication with API key support and rate limiting.
"""

import time
from typing import Optional, Dict, Any
from functools import wraps

import structlog
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)


class RateLimiter:
    """
    Simple in-memory rate limiter for API requests.
    """
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.requests = {}  # client_id -> [(timestamp, count), ...]
        self.window_size = 60  # 1 minute window
    
    def _clean_old_requests(self, client_id: str):
        """Clean old requests outside the time window."""
        current_time = time.time()
        if client_id in self.requests:
            self.requests[client_id] = [
                (timestamp, count) for timestamp, count in self.requests[client_id]
                if current_time - timestamp < self.window_size
            ]
    
    def is_allowed(self, client_id: str, request_count: int = 1) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed based on rate limits.
        
        Args:
            client_id: Unique identifier for the client
            request_count: Number of requests (default: 1)
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        
        # Initialize client if not exists
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Clean old requests
        self._clean_old_requests(client_id)
        
        # Calculate current usage
        current_requests = sum(count for _, count in self.requests[client_id])
        
        # Check burst limit
        if current_requests + request_count > self.burst_size:
            return False, {
                "error": "Burst limit exceeded",
                "limit": self.burst_size,
                "current": current_requests,
                "reset_time": current_time + self.window_size
            }
        
        # Check rate limit
        requests_in_window = current_requests + request_count
        if requests_in_window > self.requests_per_minute:
            return False, {
                "error": "Rate limit exceeded",
                "limit": self.requests_per_minute,
                "current": current_requests,
                "reset_time": current_time + self.window_size
            }
        
        # Record the request
        self.requests[client_id].append((current_time, request_count))
        
        return True, {
            "allowed": True,
            "limit": self.requests_per_minute,
            "remaining": self.requests_per_minute - requests_in_window,
            "reset_time": current_time + self.window_size
        }


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=get_settings().requests_per_minute,
    burst_size=get_settings().burst_size
)


def get_client_id(request: Request) -> str:
    """
    Get client identifier for rate limiting.
    
    Uses IP address as default identifier.
    """
    # Try to get real IP from headers (for proxy setups)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    
    # Fallback to client IP
    return request.client.host if request.client else "unknown"


async def get_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """
    Validate API key authentication.
    
    Args:
        request: FastAPI request object
        credentials: HTTP authorization credentials
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If authentication fails
    """
    settings = get_settings()
    
    # Skip authentication if disabled
    if not settings.enable_auth:
        return "anonymous"
    
    # Check for API key in various locations
    api_key = None
    
    # 1. Authorization header (Bearer token)
    if credentials and credentials.scheme.lower() == "bearer":
        api_key = credentials.credentials
    
    # 2. X-API-Key header
    if not api_key:
        api_key = request.headers.get("X-API-Key")
    
    # 3. Query parameter (less secure, for testing only)
    if not api_key and settings.debug:
        api_key = request.query_params.get("api_key")
    
    if not api_key:
        logger.warning("Authentication failed - no API key provided",
                      client_ip=get_client_id(request))
        raise HTTPException(
            status_code=401,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Validate API key
    if not validate_api_key(api_key):
        logger.warning("Authentication failed - invalid API key",
                      client_ip=get_client_id(request),
                      api_key_prefix=api_key[:8] + "..." if len(api_key) > 8 else "short")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Rate limiting check
    client_id = get_client_id(request)
    
    if settings.enable_rate_limiting:
        is_allowed, rate_info = rate_limiter.is_allowed(client_id)
        
        if not is_allowed:
            logger.warning("Rate limit exceeded",
                          client_ip=client_id,
                          rate_info=rate_info)
            
            raise HTTPException(
                status_code=429,
                detail=rate_info["error"],
                headers={
                    "X-RateLimit-Limit": str(settings.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(rate_info["reset_time"])),
                    "Retry-After": str(int(rate_info["reset_time"] - time.time()))
                }
            )
        
        # Add rate limit headers to successful requests
        request.state.rate_limit_info = rate_info
    
    logger.info("Authentication successful",
               client_ip=client_id,
               api_key_prefix=api_key[:8] + "..." if len(api_key) > 8 else "short")
    
    return api_key


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key against configured keys.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    settings = get_settings()
    
    # Simple validation - in production, this would typically
    # check against a database or external service
    if settings.api_key and api_key == settings.api_key:
        return True
    
    # For development, allow any key starting with "dev_"
    if settings.debug and api_key.startswith("dev_"):
        return True
    
    # Check against environment-specific keys
    valid_keys = []
    
    if settings.environment == "development":
        valid_keys.extend([
            "dev_test_key_12345",
            "development_api_key",
            "test_key"
        ])
    elif settings.environment == "staging":
        valid_keys.extend([
            "staging_api_key_67890",
            "stg_test_key"
        ])
    
    return api_key in valid_keys


def require_admin_key(api_key: str = Depends(get_api_key)):
    """
    Dependency that requires admin-level API key.
    
    Args:
        api_key: API key from authentication
        
    Raises:
        HTTPException: If not admin key
    """
    if api_key == "anonymous":
        return api_key
    
    # Check if this is an admin key
    admin_keys = [
        "admin_key_super_secret",
        "dev_admin_key"  # For development
    ]
    
    if api_key not in admin_keys:
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required"
        )
    
    return api_key


def log_api_access(endpoint: str):
    """
    Decorator to log API access for security auditing.
    
    Args:
        endpoint: Endpoint name for logging
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from kwargs or args
            request = None
            for arg in args:
                if hasattr(arg, 'client'):
                    request = arg
                    break
            
            if not request:
                for value in kwargs.values():
                    if hasattr(value, 'client'):
                        request = value
                        break
            
            if request:
                client_id = get_client_id(request)
                logger.info("API access",
                           endpoint=endpoint,
                           client_ip=client_id,
                           user_agent=request.headers.get("User-Agent", "unknown"))
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


class SecurityHeaders:
    """
    Security headers middleware for production deployment.
    """
    
    @staticmethod
    def add_security_headers(response, request: Request):
        """Add security headers to response."""
        settings = get_settings()
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # HSTS for HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # CSP for API endpoints
        response.headers["Content-Security-Policy"] = "default-src 'none'"
        
        # Rate limit headers if available
        if hasattr(request.state, 'rate_limit_info'):
            rate_info = request.state.rate_limit_info
            response.headers["X-RateLimit-Limit"] = str(settings.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(rate_info.get("remaining", 0))
            response.headers["X-RateLimit-Reset"] = str(int(rate_info.get("reset_time", time.time())))
        
        return response


def get_api_key_info(api_key: str) -> Dict[str, Any]:
    """
    Get information about an API key for monitoring/auditing.
    
    Args:
        api_key: API key to analyze
        
    Returns:
        API key information
    """
    if api_key == "anonymous":
        return {"type": "anonymous", "level": "public"}
    
    # Analyze key type
    if api_key.startswith("admin_"):
        return {"type": "admin", "level": "admin"}
    elif api_key.startswith("dev_"):
        return {"type": "development", "level": "standard"}
    elif api_key.startswith("stg_"):
        return {"type": "staging", "level": "standard"}
    else:
        return {"type": "production", "level": "standard"}


def create_api_response_headers(request: Request) -> Dict[str, str]:
    """
    Create standard API response headers.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Dictionary of headers to add to response
    """
    headers = {
        "X-API-Version": get_settings().app_version,
        "X-Request-ID": request.headers.get("X-Request-ID", "unknown"),
        "X-Response-Time": "0",  # Will be updated by middleware
    }
    
    # Add rate limit headers if available
    if hasattr(request.state, 'rate_limit_info'):
        rate_info = request.state.rate_limit_info
        headers.update({
            "X-RateLimit-Limit": str(get_settings().requests_per_minute),
            "X-RateLimit-Remaining": str(rate_info.get("remaining", 0)),
            "X-RateLimit-Reset": str(int(rate_info.get("reset_time", time.time())))
        })
    
    return headers