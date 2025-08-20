#!/usr/bin/env python3
"""
⚙️ SkillStruct Configuration
Centralized configuration for the entire platform
"""

import os
from typing import Dict, Any

# ===========================
#  ENVIRONMENT SETTINGS
# ===========================

ENV = os.getenv("ENVIRONMENT", "development")
DEBUG = ENV == "development"

# ===========================
#  DIRECTORIES
# ===========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
BACKUP_DIR = os.path.join(DATA_DIR, "backups")

# Ensure directories exist
for directory in [DATA_DIR, UPLOAD_DIR, BACKUP_DIR]:
    os.makedirs(directory, exist_ok=True)

# ===========================
#  API SETTINGS
# ===========================

API_SETTINGS = {
    "title": "SkillStruct Platform",
    "version": "1.0.0",
    "description": "AI-powered skill taxonomy and resume analysis platform",
    "cors_origins": ["*"],
    "cors_methods": ["*"],
    "cors_headers": ["*"]
}

# ===========================
#  SERVICE PORTS
# ===========================

SERVICES = {
    "ocr_api": {
        "port": 8000,
        "file": "services/ocr+clusteringAPI.py",
        "name": "OCR & Clustering API"
    },
    "json_api": {
        "port": 8001,
        "file": "services/genjsongraphAPI.py",
        "name": "JSON Generation API"
    },
    "graph_api": {
        "port": 8002,
        "file": "services/GraphAPI.py",
        "name": "Graph Management API"
    },
    "recommend_api": {
        "port": 8003,
        "file": "services/recommendapi.py",
        "name": "Recommendation API"
    },
    "rag_api": {
        "port": 8004,
        "file": "rag/services/rag_api.py",
        "name": "RAG Search API"
    }
}

STREAMLIT_PORT = 8503
STREAMLIT_FILE = "frontend_streamlit/streamlit_app.py"

# ===========================
#  AUTHENTICATION
# ===========================

AUTH_SETTINGS = {
    "secret_key": os.getenv("SECRET_KEY", "skillstruct-secret-key-2024"),
    "algorithm": "HS256",
    "access_token_expire_minutes": 30,
    "refresh_token_expire_days": 7,
    "session_expire_hours": 24
}

# ===========================
#  FILE PROCESSING
# ===========================

FILE_SETTINGS = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".pdf", ".doc", ".docx", ".txt"],
    "upload_path": UPLOAD_DIR,
    "temp_path": os.path.join(DATA_DIR, "temp")
}

# ===========================
#  AI SETTINGS
# ===========================

AI_SETTINGS = {
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "model_name": "gemini-pro",
    "max_tokens": 1000,
    "temperature": 0.7,
    "cache_results": True,
    "cache_ttl_hours": 24
}

# ===========================
#  RAG SYSTEM SETTINGS
# ===========================

RAG_SETTINGS = {
    "enabled": True,
    "milvus_host": os.getenv("MILVUS_HOST", "localhost"),
    "milvus_port": int(os.getenv("MILVUS_PORT", "19530")),
    "google_api_key": os.getenv("GOOGLE_API_KEY"),
    "collection_name": os.getenv("RAG_COLLECTION_NAME", "skillstruct_rag"),
    "embedding_model": "models/embedding-001",
    "default_top_k": 10,
    "score_threshold": 0.7,
    "enable_reranking": True,
    "cache_enabled": True
}

# ===========================
#  DATABASE SETTINGS
# ===========================

DATABASE_SETTINGS = {
    "type": "json",  # Using JSON files for simplicity
    "backup_enabled": True,
    "backup_interval_hours": 6,
    "auto_cleanup_days": 30
}

# ===========================
#  RATE LIMITING
# ===========================

RATE_LIMIT_SETTINGS = {
    "enabled": True,
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "burst_limit": 10
}

# ===========================
#  LOGGING
# ===========================

LOGGING_SETTINGS = {
    "level": "INFO" if not DEBUG else "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": os.path.join(DATA_DIR, "logs", "skillstruct.log"),
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# ===========================
#  DEFAULT USERS
# ===========================

DEFAULT_USERS = [
    {
        "id": "admin-001",
        "username": "admin",
        "password": "admin123",  # Will be hashed
        "email": "admin@skillstruct.com",
        "full_name": "System Administrator",
        "role": "admin",
        "is_active": True
    },
    {
        "id": "hr-001",
        "username": "hr_architect",
        "password": "hr123",
        "email": "hr@skillstruct.com",
        "full_name": "HR Architect",
        "role": "hr_architect",
        "is_active": True
    },
    {
        "id": "manager-001",
        "username": "manager",
        "password": "manager123",
        "email": "manager@skillstruct.com",
        "full_name": "Team Manager",
        "role": "manager",
        "is_active": True
    },
    {
        "id": "viewer-001",
        "username": "viewer",
        "password": "viewer123",
        "email": "viewer@skillstruct.com",
        "full_name": "Viewer User",
        "role": "viewer",
        "is_active": True
    }
]

# ===========================
#  PERMISSIONS MATRIX
# ===========================

PERMISSIONS = {
    "admin": {
        "read": True,
        "create": True,
        "update": True,
        "delete": True,
        "import_export": True,
        "user_management": True,
        "system_config": True
    },
    "hr_architect": {
        "read": True,
        "create": True,
        "update": True,
        "delete": True,
        "import_export": True,
        "user_management": False,
        "system_config": False
    },
    "manager": {
        "read": True,
        "create": True,
        "update": True,
        "delete": False,
        "import_export": True,
        "user_management": False,
        "system_config": False
    },
    "viewer": {
        "read": True,
        "create": False,
        "update": False,
        "delete": False,
        "import_export": False,
        "user_management": False,
        "system_config": False
    }
}

# ===========================
#  HELPER FUNCTIONS
# ===========================

def get_service_config(service_name: str) -> Dict[str, Any]:
    """Get configuration for specific service"""
    return SERVICES.get(service_name, {})

def get_all_service_ports() -> Dict[str, int]:
    """Get all service ports"""
    return {name: config["port"] for name, config in SERVICES.items()}

def get_upload_path() -> str:
    """Get upload directory path"""
    return UPLOAD_DIR

def get_data_path() -> str:
    """Get data directory path"""
    return DATA_DIR

def is_debug_mode() -> bool:
    """Check if running in debug mode"""
    return DEBUG

def get_allowed_file_extensions() -> list:
    """Get allowed file extensions"""
    return FILE_SETTINGS["allowed_extensions"]

def get_max_file_size() -> int:
    """Get maximum file size"""
    return FILE_SETTINGS["max_file_size"]

# ===========================
#  VALIDATION
# ===========================

def validate_config() -> Dict[str, Any]:
    """Validate configuration"""
    issues = []
    
    # Check required directories
    for directory in [DATA_DIR, UPLOAD_DIR, BACKUP_DIR]:
        if not os.path.exists(directory):
            issues.append(f"Directory not found: {directory}")
    
    # Check AI API key
    if not AI_SETTINGS["gemini_api_key"]:
        issues.append("GEMINI_API_KEY environment variable not set")
    
    # Check service files
    for service_name, config in SERVICES.items():
        service_file = config["file"]
        if not os.path.exists(service_file):
            issues.append(f"Service file not found: {service_file}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "config_summary": {
            "environment": ENV,
            "debug": DEBUG,
            "services_count": len(SERVICES),
            "data_dir": DATA_DIR
        }
    }

# ===========================
#  EXPORT CONFIG INFO
# ===========================

def get_config_info() -> Dict[str, Any]:
    """Get comprehensive configuration information"""
    return {
        "environment": ENV,
        "debug": DEBUG,
        "base_dir": BASE_DIR,
        "data_dir": DATA_DIR,
        "services": SERVICES,
        "streamlit": {
            "port": STREAMLIT_PORT,
            "file": STREAMLIT_FILE
        },
        "auth": {
            "token_expire_minutes": AUTH_SETTINGS["access_token_expire_minutes"],
            "session_expire_hours": AUTH_SETTINGS["session_expire_hours"]
        },
        "file_processing": {
            "max_size_mb": FILE_SETTINGS["max_file_size"] / (1024 * 1024),
            "allowed_extensions": FILE_SETTINGS["allowed_extensions"]
        },
        "validation": validate_config()
    }
