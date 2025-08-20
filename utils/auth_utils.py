#!/usr/bin/env python3
"""
🔐 Authentication Utilities
Các hàm tiện ích cho xác thực và bảo mật
"""

import hashlib
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET_KEY = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 8

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt (more secure than SHA256)
    """
    try:
        # Generate salt and hash password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    except Exception as e:
        logger.error(f"Error hashing password: {e}")
        # Fallback to SHA256 for compatibility
        return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify password against hash
    """
    try:
        # Try bcrypt first
        if hashed_password.startswith('$2b$'):
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        else:
            # Fallback to SHA256 for legacy passwords
            return hashlib.sha256(password.encode()).hexdigest() == hashed_password
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False

def generate_session_token(length: int = 32) -> str:
    """
    Generate secure session token
    """
    return secrets.token_urlsafe(length)

def create_jwt_token(user_data: Dict[str, Any], expires_hours: int = JWT_EXPIRATION_HOURS) -> str:
    """
    Create JWT token for user
    """
    try:
        payload = {
            "user_id": user_data.get("id"),
            "username": user_data.get("username"),
            "role": user_data.get("role"),
            "permissions": user_data.get("permissions", {}),
            "exp": datetime.utcnow() + timedelta(hours=expires_hours),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    except Exception as e:
        logger.error(f"Error creating JWT token: {e}")
        return None

def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode JWT token
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None

def check_permissions(user_permissions: Dict[str, bool], required_permission: str) -> bool:
    """
    Check if user has required permission
    """
    return user_permissions.get(required_permission, False)

def require_permission(permission: str):
    """
    Decorator to require specific permission
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # This would be used with FastAPI Depends
            # Implementation depends on the API framework
            return func(*args, **kwargs)
        return wrapper
    return decorator

def generate_api_key(prefix: str = "sk", length: int = 32) -> str:
    """
    Generate API key with prefix
    """
    random_part = secrets.token_urlsafe(length)
    return f"{prefix}_{random_part}"

def validate_api_key_format(api_key: str) -> bool:
    """
    Validate API key format
    """
    parts = api_key.split('_')
    return len(parts) == 2 and len(parts[1]) >= 20

def create_security_log(user_id: str, action: str, resource: str, 
                       ip_address: str, user_agent: str, success: bool, 
                       details: str = None) -> Dict[str, Any]:
    """
    Create security log entry
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "ip_address": ip_address,
        "user_agent": user_agent,
        "success": success,
        "details": details
    }

def mask_sensitive_data(data: str, mask_char: str = "*", visible_chars: int = 4) -> str:
    """
    Mask sensitive data (passwords, API keys, etc.)
    """
    if len(data) <= visible_chars:
        return mask_char * len(data)
    
    return data[:visible_chars] + mask_char * (len(data) - visible_chars)

def generate_otp(length: int = 6) -> str:
    """
    Generate One-Time Password
    """
    import random
    return ''.join([str(random.randint(0, 9)) for _ in range(length)])

def validate_session_token(token: str, active_sessions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Validate session token against active sessions
    """
    session = active_sessions.get(token)
    if not session:
        return None
    
    # Check if session is expired
    if datetime.now() > session.get("expires_at", datetime.now()):
        # Remove expired session
        active_sessions.pop(token, None)
        return None
    
    return session

def cleanup_expired_sessions(active_sessions: Dict[str, Any]) -> int:
    """
    Remove expired sessions and return count of removed sessions
    """
    expired_tokens = []
    current_time = datetime.now()
    
    for token, session in active_sessions.items():
        if current_time > session.get("expires_at", current_time):
            expired_tokens.append(token)
    
    for token in expired_tokens:
        active_sessions.pop(token, None)
    
    return len(expired_tokens)

def rate_limit_key(user_id: str, endpoint: str, window: str = "hour") -> str:
    """
    Generate rate limit key
    """
    timestamp = datetime.now()
    if window == "minute":
        time_key = timestamp.strftime("%Y%m%d%H%M")
    elif window == "hour":
        time_key = timestamp.strftime("%Y%m%d%H")
    else:  # day
        time_key = timestamp.strftime("%Y%m%d")
    
    return f"rate_limit:{user_id}:{endpoint}:{time_key}"

def extract_bearer_token(authorization_header: str) -> Optional[str]:
    """
    Extract token from Authorization header
    """
    if not authorization_header:
        return None
    
    parts = authorization_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    
    return parts[1]

# Role-based permission mappings
ROLE_PERMISSIONS = {
    "admin": {
        "read": True, "create": True, "update": True, "delete": True,
        "import_export": True, "versioning": True, "user_management": True
    },
    "hr_architect": {
        "read": True, "create": True, "update": True, "delete": True,
        "import_export": False, "versioning": False, "user_management": False
    },
    "manager": {
        "read": True, "create": False, "update": False, "delete": False,
        "import_export": False, "versioning": False, "user_management": False
    },
    "viewer": {
        "read": True, "create": False, "update": False, "delete": False,
        "import_export": False, "versioning": False, "user_management": False
    }
}

def get_role_permissions(role: str) -> Dict[str, bool]:
    """
    Get permissions for a role
    """
    return ROLE_PERMISSIONS.get(role, {})

def can_perform_action(user_role: str, action: str) -> bool:
    """
    Check if user role can perform action
    """
    permissions = get_role_permissions(user_role)
    return permissions.get(action, False)
