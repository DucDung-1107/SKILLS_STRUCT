#!/usr/bin/env python3
"""
🔌 API Utilities
Các hàm tiện ích cho API operations và HTTP handling
"""

import json
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from functools import wraps
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Rate limiting storage
rate_limit_storage = defaultdict(list)

def create_response(success: bool, message: str, data: Any = None, 
                   error: str = None, status_code: int = 200) -> Dict[str, Any]:
    """
    Tạo standardized API response
    """
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    if error:
        response["error"] = error
    
    return response

def create_error_response(error_message: str, error_code: str = None, 
                         details: Dict[str, Any] = None, 
                         status_code: int = 400) -> Dict[str, Any]:
    """
    Tạo error response
    """
    response = {
        "success": False,
        "error": error_message,
        "timestamp": datetime.now().isoformat()
    }
    
    if error_code:
        response["error_code"] = error_code
    
    if details:
        response["details"] = details
    
    return response

def create_success_response(message: str, data: Any = None) -> Dict[str, Any]:
    """
    Tạo success response
    """
    return create_response(success=True, message=message, data=data)

def handle_api_error(error: Exception, message: str = "API error occurred") -> Dict[str, Any]:
    """
    Handle API errors and return formatted response
    
    Args:
        error: The exception that occurred
        message: Custom error message
        
    Returns:
        Error response dictionary
    """
    logger.error(f"{message}: {error}")
    
    # Determine error type and status code
    if isinstance(error, ValueError):
        return create_error_response(str(error), "VALIDATION_ERROR", status_code=400)
    elif isinstance(error, PermissionError):
        return create_error_response("Permission denied", "PERMISSION_DENIED", status_code=403)
    elif isinstance(error, FileNotFoundError):
        return create_error_response("Resource not found", "NOT_FOUND", status_code=404)
    else:
        return create_error_response(message, "INTERNAL_ERROR", status_code=500)

def handle_error(func: Callable) -> Callable:
    """
    Decorator để handle errors trong API endpoints
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"ValueError in {func.__name__}: {e}")
            return create_error_response(str(e), "VALIDATION_ERROR", status_code=400)
        except PermissionError as e:
            logger.warning(f"PermissionError in {func.__name__}: {e}")
            return create_error_response("Insufficient permissions", "PERMISSION_DENIED", status_code=403)
        except FileNotFoundError as e:
            logger.warning(f"FileNotFoundError in {func.__name__}: {e}")
            return create_error_response("Resource not found", "NOT_FOUND", status_code=404)
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return create_error_response("Internal server error", "INTERNAL_ERROR", status_code=500)
    
    return wrapper

def log_api_call(endpoint: str, method: str, user_id: str = None, 
                ip_address: str = None, user_agent: str = None,
                status_code: int = 200, response_time: float = None) -> None:
    """
    Log API call
    """
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "method": method,
        "user_id": user_id,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "status_code": status_code,
        "response_time_ms": round(response_time * 1000, 2) if response_time else None
    }
    
    logger.info(f"API Call: {json.dumps(log_data)}")

def rate_limit_check(user_id: str, endpoint: str, max_requests: int = 100, 
                    window_seconds: int = 3600) -> Dict[str, Any]:
    """
    Check rate limit for user/endpoint
    """
    current_time = time.time()
    key = f"{user_id}:{endpoint}"
    
    # Clean old requests outside window
    rate_limit_storage[key] = [
        req_time for req_time in rate_limit_storage[key]
        if current_time - req_time < window_seconds
    ]
    
    # Check if limit exceeded
    request_count = len(rate_limit_storage[key])
    if request_count >= max_requests:
        return {
            "allowed": False,
            "requests_made": request_count,
            "max_requests": max_requests,
            "window_seconds": window_seconds,
            "reset_time": min(rate_limit_storage[key]) + window_seconds
        }
    
    # Add current request
    rate_limit_storage[key].append(current_time)
    
    return {
        "allowed": True,
        "requests_made": request_count + 1,
        "max_requests": max_requests,
        "window_seconds": window_seconds,
        "remaining": max_requests - request_count - 1
    }

def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """
    Rate limiting decorator
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(request, *args, **kwargs):
            # Extract user info from request
            user_id = getattr(request.state, 'user_id', 'anonymous')
            endpoint = request.url.path
            
            # Check rate limit
            limit_result = rate_limit_check(user_id, endpoint, max_requests, window_seconds)
            
            if not limit_result["allowed"]:
                return create_error_response(
                    "Rate limit exceeded",
                    "RATE_LIMIT_EXCEEDED",
                    details=limit_result,
                    status_code=429
                )
            
            # Add rate limit headers to response
            response = await func(request, *args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Limit"] = str(max_requests)
                response.headers["X-RateLimit-Remaining"] = str(limit_result["remaining"])
                response.headers["X-RateLimit-Reset"] = str(int(time.time() + window_seconds))
            
            return response
        
        return wrapper
    return decorator

def validate_api_key(api_key: str, valid_keys: List[str] = None) -> Dict[str, Any]:
    """
    Validate API key
    """
    if not api_key:
        return {
            "valid": False,
            "error": "API key is required"
        }
    
    # Basic format validation
    if not api_key.startswith(('sk_', 'pk_')):
        return {
            "valid": False,
            "error": "Invalid API key format"
        }
    
    # Check against valid keys list (if provided)
    if valid_keys and api_key not in valid_keys:
        return {
            "valid": False,
            "error": "Invalid API key"
        }
    
    return {
        "valid": True,
        "key_type": "secret" if api_key.startswith('sk_') else "public"
    }

def paginate_results(data: List[Any], page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """
    Paginate list of results
    """
    total_items = len(data)
    total_pages = (total_items + page_size - 1) // page_size
    
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    
    paginated_data = data[start_index:end_index]
    
    return {
        "data": paginated_data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }

def filter_results(data: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Filter results based on criteria
    """
    if not filters:
        return data
    
    filtered_data = []
    
    for item in data:
        include_item = True
        
        for field, value in filters.items():
            if field not in item:
                continue
            
            item_value = item[field]
            
            # Handle different filter types
            if isinstance(value, dict):
                # Range filters (e.g., {"min": 1, "max": 10})
                if "min" in value and item_value < value["min"]:
                    include_item = False
                    break
                if "max" in value and item_value > value["max"]:
                    include_item = False
                    break
            elif isinstance(value, list):
                # Multiple choice filter
                if item_value not in value:
                    include_item = False
                    break
            elif isinstance(value, str):
                # String contains filter (case insensitive)
                if value.lower() not in str(item_value).lower():
                    include_item = False
                    break
            else:
                # Exact match filter
                if item_value != value:
                    include_item = False
                    break
        
        if include_item:
            filtered_data.append(item)
    
    return filtered_data

def sort_results(data: List[Dict[str, Any]], sort_by: str = "name", 
                sort_order: str = "asc") -> List[Dict[str, Any]]:
    """
    Sort results by field
    """
    if not data or sort_by not in data[0]:
        return data
    
    reverse = sort_order.lower() == "desc"
    
    try:
        return sorted(data, key=lambda x: x.get(sort_by, ""), reverse=reverse)
    except Exception as e:
        logger.warning(f"Error sorting data: {e}")
        return data

def search_results(data: List[Dict[str, Any]], query: str, 
                  search_fields: List[str] = None) -> List[Dict[str, Any]]:
    """
    Search through results
    """
    if not query or not data:
        return data
    
    if search_fields is None:
        search_fields = ["name", "description", "title"]
    
    query_lower = query.lower()
    results = []
    
    for item in data:
        for field in search_fields:
            if field in item:
                field_value = str(item[field]).lower()
                if query_lower in field_value:
                    results.append(item)
                    break  # Found match, no need to check other fields
    
    return results

def validate_request_data(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate request data against schema
    """
    errors = []
    validated_data = {}
    
    for field, rules in schema.items():
        value = data.get(field)
        
        # Check required fields
        if rules.get("required", False) and value is None:
            errors.append(f"Field '{field}' is required")
            continue
        
        # Skip validation if field is optional and not provided
        if value is None:
            continue
        
        # Type validation
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
            continue
        
        # Length validation for strings
        if isinstance(value, str):
            min_length = rules.get("min_length")
            max_length = rules.get("max_length")
            
            if min_length and len(value) < min_length:
                errors.append(f"Field '{field}' must be at least {min_length} characters")
                continue
            
            if max_length and len(value) > max_length:
                errors.append(f"Field '{field}' must be at most {max_length} characters")
                continue
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")
            
            if min_value is not None and value < min_value:
                errors.append(f"Field '{field}' must be at least {min_value}")
                continue
            
            if max_value is not None and value > max_value:
                errors.append(f"Field '{field}' must be at most {max_value}")
                continue
        
        # Enum validation
        allowed_values = rules.get("allowed_values")
        if allowed_values and value not in allowed_values:
            errors.append(f"Field '{field}' must be one of: {allowed_values}")
            continue
        
        # Custom validation function
        validator = rules.get("validator")
        if validator and not validator(value):
            error_message = rules.get("error_message", f"Field '{field}' is invalid")
            errors.append(error_message)
            continue
        
        validated_data[field] = value
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "data": validated_data
    }

def format_api_response(data: Any, meta: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Format API response with metadata
    """
    response = {
        "success": True,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    if meta:
        response["meta"] = meta
    
    return response

def calculate_response_time(start_time: float) -> float:
    """
    Calculate response time in seconds
    """
    return time.time() - start_time

def cors_headers() -> Dict[str, str]:
    """
    Get CORS headers
    """
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
        "Access-Control-Max-Age": "86400"
    }

def cache_response(key: str, data: Any, ttl: int = 300) -> None:
    """
    Cache API response (simple in-memory cache)
    """
    # This is a simple implementation
    # In production, use Redis or similar
    cache_storage = getattr(cache_response, 'storage', {})
    
    cache_storage[key] = {
        "data": data,
        "expires_at": time.time() + ttl
    }
    
    cache_response.storage = cache_storage

def get_cached_response(key: str) -> Optional[Any]:
    """
    Get cached API response
    """
    cache_storage = getattr(cache_response, 'storage', {})
    
    if key in cache_storage:
        cached = cache_storage[key]
        if time.time() < cached["expires_at"]:
            return cached["data"]
        else:
            # Remove expired cache
            del cache_storage[key]
    
    return None

def clear_cache() -> None:
    """
    Clear all cached responses
    """
    if hasattr(cache_response, 'storage'):
        cache_response.storage.clear()

def generate_request_id() -> str:
    """
    Generate unique request ID
    """
    import uuid
    return str(uuid.uuid4())

def extract_client_info(request) -> Dict[str, str]:
    """
    Extract client information from request
    """
    return {
        "ip_address": getattr(request.client, 'host', 'unknown'),
        "user_agent": request.headers.get("User-Agent", "unknown"),
        "referer": request.headers.get("Referer", ""),
        "accept_language": request.headers.get("Accept-Language", ""),
        "x_forwarded_for": request.headers.get("X-Forwarded-For", "")
    }

def health_check() -> Dict[str, Any]:
    """
    Basic health check response
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "uptime": time.time()  # This should be calculated from app start time
    }
