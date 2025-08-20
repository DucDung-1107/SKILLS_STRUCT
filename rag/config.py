#!/usr/bin/env python3
"""
⚙️ SkillStruct RAG Configuration
Centralized configuration cho RAG system
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

# Environment variables với defaults
RAG_CONFIG = {
    # Milvus configuration
    "MILVUS_HOST": os.getenv("MILVUS_HOST", "localhost"),
    "MILVUS_PORT": int(os.getenv("MILVUS_PORT", "19530")),
    "MILVUS_USER": os.getenv("MILVUS_USER", ""),
    "MILVUS_PASSWORD": os.getenv("MILVUS_PASSWORD", ""),
    "MILVUS_DB_NAME": os.getenv("MILVUS_DB_NAME", "skillstruct"),
    "MILVUS_COLLECTION_NAME": os.getenv("MILVUS_COLLECTION_NAME", "skillstruct_rag"),
    
    # Google AI configuration
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
    "GOOGLE_MODEL_NAME": os.getenv("GOOGLE_MODEL_NAME", "models/embedding-001"),
    
    # RAG API configuration
    "RAG_API_HOST": os.getenv("RAG_API_HOST", "0.0.0.0"),
    "RAG_API_PORT": int(os.getenv("RAG_API_PORT", "8004")),
    "RAG_API_PREFIX": os.getenv("RAG_API_PREFIX", "/api/v1/rag"),
    
    # Vector search configuration
    "DEFAULT_TOP_K": int(os.getenv("DEFAULT_TOP_K", "10")),
    "DEFAULT_SCORE_THRESHOLD": float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.7")),
    "MAX_TOP_K": int(os.getenv("MAX_TOP_K", "100")),
    
    # Embedding configuration
    "EMBEDDING_DIMENSION": int(os.getenv("EMBEDDING_DIMENSION", "768")),
    "EMBEDDING_BATCH_SIZE": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
    "MAX_TOKEN_LENGTH": int(os.getenv("MAX_TOKEN_LENGTH", "8192")),
    
    # Indexing configuration
    "INDEX_BATCH_SIZE": int(os.getenv("INDEX_BATCH_SIZE", "100")),
    "INDEX_WORKERS": int(os.getenv("INDEX_WORKERS", "4")),
    "AUTO_OPTIMIZE_INTERVAL": int(os.getenv("AUTO_OPTIMIZE_INTERVAL", "3600")),  # seconds
    
    # Cache configuration
    "ENABLE_CACHE": os.getenv("ENABLE_CACHE", "true").lower() == "true",
    "CACHE_TTL": int(os.getenv("CACHE_TTL", "300")),  # seconds
    "CACHE_SIZE": int(os.getenv("CACHE_SIZE", "1000")),
    
    # Reranking configuration
    "ENABLE_RERANKING": os.getenv("ENABLE_RERANKING", "true").lower() == "true",
    "RERANK_TOP_K": int(os.getenv("RERANK_TOP_K", "50")),
    "SKILL_WEIGHT": float(os.getenv("SKILL_WEIGHT", "0.3")),
    "FRESHNESS_WEIGHT": float(os.getenv("FRESHNESS_WEIGHT", "0.1")),
    "QUALITY_WEIGHT": float(os.getenv("QUALITY_WEIGHT", "0.2")),
    "DIVERSITY_WEIGHT": float(os.getenv("DIVERSITY_WEIGHT", "0.1")),
    
    # Backup configuration
    "BACKUP_ENABLED": os.getenv("BACKUP_ENABLED", "true").lower() == "true",
    "BACKUP_INTERVAL": int(os.getenv("BACKUP_INTERVAL", "86400")),  # seconds (24h)
    "BACKUP_RETENTION_DAYS": int(os.getenv("BACKUP_RETENTION_DAYS", "30")),
    "BACKUP_PATH": os.getenv("BACKUP_PATH", "./backups"),
    
    # Monitoring configuration
    "ENABLE_METRICS": os.getenv("ENABLE_METRICS", "true").lower() == "true",
    "METRICS_PORT": int(os.getenv("METRICS_PORT", "9090")),
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    
    # Security configuration
    "ENABLE_AUTH": os.getenv("ENABLE_AUTH", "false").lower() == "true",
    "JWT_SECRET": os.getenv("JWT_SECRET", "your-secret-key"),
    "API_KEY_HEADER": os.getenv("API_KEY_HEADER", "X-API-Key"),
    
    # Performance tuning
    "MAX_CONCURRENT_REQUESTS": int(os.getenv("MAX_CONCURRENT_REQUESTS", "100")),
    "REQUEST_TIMEOUT": int(os.getenv("REQUEST_TIMEOUT", "30")),
    "CONNECTION_POOL_SIZE": int(os.getenv("CONNECTION_POOL_SIZE", "10")),
}

class VectorIndexType(str, Enum):
    """Vector index types"""
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    IVF_PQ = "IVF_PQ"
    HNSW = "HNSW"
    ANNOY = "ANNOY"
    SCANN = "SCANN"

class MetricType(str, Enum):
    """Distance metrics"""
    L2 = "L2"
    IP = "IP"  # Inner Product
    COSINE = "COSINE"
    HAMMING = "HAMMING"
    JACCARD = "JACCARD"

@dataclass
class MilvusIndexConfig:
    """Milvus index configuration"""
    index_type: VectorIndexType = VectorIndexType.HNSW
    metric_type: MetricType = MetricType.COSINE
    index_params: Dict[str, Any] = field(default_factory=lambda: {
        "M": 16,
        "efConstruction": 200
    })
    search_params: Dict[str, Any] = field(default_factory=lambda: {
        "ef": 100
    })

@dataclass
class RAGSystemConfig:
    """Complete RAG system configuration"""
    
    # Milvus settings
    milvus_host: str = RAG_CONFIG["MILVUS_HOST"]
    milvus_port: int = RAG_CONFIG["MILVUS_PORT"]
    milvus_user: str = RAG_CONFIG["MILVUS_USER"]
    milvus_password: str = RAG_CONFIG["MILVUS_PASSWORD"]
    milvus_db_name: str = RAG_CONFIG["MILVUS_DB_NAME"]
    collection_name: str = RAG_CONFIG["MILVUS_COLLECTION_NAME"]
    
    # Google AI settings
    google_api_key: str = RAG_CONFIG["GOOGLE_API_KEY"]
    google_model_name: str = RAG_CONFIG["GOOGLE_MODEL_NAME"]
    
    # API settings
    api_host: str = RAG_CONFIG["RAG_API_HOST"]
    api_port: int = RAG_CONFIG["RAG_API_PORT"]
    api_prefix: str = RAG_CONFIG["RAG_API_PREFIX"]
    
    # Search settings
    default_top_k: int = RAG_CONFIG["DEFAULT_TOP_K"]
    default_score_threshold: float = RAG_CONFIG["DEFAULT_SCORE_THRESHOLD"]
    max_top_k: int = RAG_CONFIG["MAX_TOP_K"]
    
    # Embedding settings
    embedding_dimension: int = RAG_CONFIG["EMBEDDING_DIMENSION"]
    embedding_batch_size: int = RAG_CONFIG["EMBEDDING_BATCH_SIZE"]
    max_token_length: int = RAG_CONFIG["MAX_TOKEN_LENGTH"]
    
    # Indexing settings
    index_batch_size: int = RAG_CONFIG["INDEX_BATCH_SIZE"]
    index_workers: int = RAG_CONFIG["INDEX_WORKERS"]
    auto_optimize_interval: int = RAG_CONFIG["AUTO_OPTIMIZE_INTERVAL"]
    
    # Cache settings
    enable_cache: bool = RAG_CONFIG["ENABLE_CACHE"]
    cache_ttl: int = RAG_CONFIG["CACHE_TTL"]
    cache_size: int = RAG_CONFIG["CACHE_SIZE"]
    
    # Reranking settings
    enable_reranking: bool = RAG_CONFIG["ENABLE_RERANKING"]
    rerank_top_k: int = RAG_CONFIG["RERANK_TOP_K"]
    skill_weight: float = RAG_CONFIG["SKILL_WEIGHT"]
    freshness_weight: float = RAG_CONFIG["FRESHNESS_WEIGHT"]
    quality_weight: float = RAG_CONFIG["QUALITY_WEIGHT"]
    diversity_weight: float = RAG_CONFIG["DIVERSITY_WEIGHT"]
    
    # Backup settings
    backup_enabled: bool = RAG_CONFIG["BACKUP_ENABLED"]
    backup_interval: int = RAG_CONFIG["BACKUP_INTERVAL"]
    backup_retention_days: int = RAG_CONFIG["BACKUP_RETENTION_DAYS"]
    backup_path: str = RAG_CONFIG["BACKUP_PATH"]
    
    # Monitoring settings
    enable_metrics: bool = RAG_CONFIG["ENABLE_METRICS"]
    metrics_port: int = RAG_CONFIG["METRICS_PORT"]
    log_level: str = RAG_CONFIG["LOG_LEVEL"]
    
    # Security settings
    enable_auth: bool = RAG_CONFIG["ENABLE_AUTH"]
    jwt_secret: str = RAG_CONFIG["JWT_SECRET"]
    api_key_header: str = RAG_CONFIG["API_KEY_HEADER"]
    
    # Performance settings
    max_concurrent_requests: int = RAG_CONFIG["MAX_CONCURRENT_REQUESTS"]
    request_timeout: int = RAG_CONFIG["REQUEST_TIMEOUT"]
    connection_pool_size: int = RAG_CONFIG["CONNECTION_POOL_SIZE"]
    
    # Index configuration
    index_config: MilvusIndexConfig = field(default_factory=MilvusIndexConfig)
    
    def validate(self) -> List[str]:
        """
        Validate configuration
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Required fields
        if not self.google_api_key:
            errors.append("Google API key is required")
        
        if not self.milvus_host:
            errors.append("Milvus host is required")
        
        if not self.collection_name:
            errors.append("Collection name is required")
        
        # Range validations
        if self.milvus_port < 1 or self.milvus_port > 65535:
            errors.append("Milvus port must be between 1 and 65535")
        
        if self.api_port < 1 or self.api_port > 65535:
            errors.append("API port must be between 1 and 65535")
        
        if self.embedding_dimension < 1:
            errors.append("Embedding dimension must be positive")
        
        if self.default_top_k < 1 or self.default_top_k > self.max_top_k:
            errors.append(f"Default top_k must be between 1 and {self.max_top_k}")
        
        if self.default_score_threshold < 0 or self.default_score_threshold > 1:
            errors.append("Score threshold must be between 0 and 1")
        
        # Weight validations
        weights = [self.skill_weight, self.freshness_weight, self.quality_weight, self.diversity_weight]
        if any(w < 0 or w > 1 for w in weights):
            errors.append("All weights must be between 0 and 1")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            # Milvus
            "milvus": {
                "host": self.milvus_host,
                "port": self.milvus_port,
                "user": self.milvus_user,
                "password": self.milvus_password,
                "db_name": self.milvus_db_name,
                "collection_name": self.collection_name,
                "index_config": {
                    "index_type": self.index_config.index_type.value,
                    "metric_type": self.index_config.metric_type.value,
                    "index_params": self.index_config.index_params,
                    "search_params": self.index_config.search_params
                }
            },
            
            # Google AI
            "google_ai": {
                "api_key": "***" if self.google_api_key else "",
                "model_name": self.google_model_name
            },
            
            # API
            "api": {
                "host": self.api_host,
                "port": self.api_port,
                "prefix": self.api_prefix
            },
            
            # Search
            "search": {
                "default_top_k": self.default_top_k,
                "default_score_threshold": self.default_score_threshold,
                "max_top_k": self.max_top_k
            },
            
            # Embedding
            "embedding": {
                "dimension": self.embedding_dimension,
                "batch_size": self.embedding_batch_size,
                "max_token_length": self.max_token_length
            },
            
            # Indexing
            "indexing": {
                "batch_size": self.index_batch_size,
                "workers": self.index_workers,
                "auto_optimize_interval": self.auto_optimize_interval
            },
            
            # Cache
            "cache": {
                "enabled": self.enable_cache,
                "ttl": self.cache_ttl,
                "size": self.cache_size
            },
            
            # Reranking
            "reranking": {
                "enabled": self.enable_reranking,
                "top_k": self.rerank_top_k,
                "weights": {
                    "skill": self.skill_weight,
                    "freshness": self.freshness_weight,
                    "quality": self.quality_weight,
                    "diversity": self.diversity_weight
                }
            },
            
            # Backup
            "backup": {
                "enabled": self.backup_enabled,
                "interval": self.backup_interval,
                "retention_days": self.backup_retention_days,
                "path": self.backup_path
            },
            
            # Monitoring
            "monitoring": {
                "metrics_enabled": self.enable_metrics,
                "metrics_port": self.metrics_port,
                "log_level": self.log_level
            },
            
            # Security
            "security": {
                "auth_enabled": self.enable_auth,
                "api_key_header": self.api_key_header
            },
            
            # Performance
            "performance": {
                "max_concurrent_requests": self.max_concurrent_requests,
                "request_timeout": self.request_timeout,
                "connection_pool_size": self.connection_pool_size
            }
        }
    
    @classmethod
    def from_env(cls) -> "RAGSystemConfig":
        """Create config from environment variables"""
        return cls()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGSystemConfig":
        """Create config from dictionary"""
        config = cls()
        
        # Update from dictionary
        if "milvus" in data:
            milvus = data["milvus"]
            config.milvus_host = milvus.get("host", config.milvus_host)
            config.milvus_port = milvus.get("port", config.milvus_port)
            config.milvus_user = milvus.get("user", config.milvus_user)
            config.milvus_password = milvus.get("password", config.milvus_password)
            config.milvus_db_name = milvus.get("db_name", config.milvus_db_name)
            config.collection_name = milvus.get("collection_name", config.collection_name)
        
        if "google_ai" in data:
            google_ai = data["google_ai"]
            config.google_api_key = google_ai.get("api_key", config.google_api_key)
            config.google_model_name = google_ai.get("model_name", config.google_model_name)
        
        # Continue for other sections...
        
        return config

# Global config instance
rag_config = RAGSystemConfig.from_env()

# Configuration validation
def validate_rag_config() -> bool:
    """
    Validate RAG configuration
    
    Returns:
        True if valid, False otherwise
    """
    errors = rag_config.validate()
    
    if errors:
        print("❌ RAG Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ RAG Configuration is valid")
    return True

# Environment file generator
def generate_env_template(file_path: str = ".env.rag"):
    """
    Generate environment variables template
    
    Args:
        file_path: Path to save environment template
    """
    template = """# SkillStruct RAG System Configuration

# Milvus Vector Database
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=
MILVUS_PASSWORD=
MILVUS_DB_NAME=skillstruct
MILVUS_COLLECTION_NAME=skillstruct_rag

# Google AI API
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL_NAME=models/embedding-001

# RAG API Server
RAG_API_HOST=0.0.0.0
RAG_API_PORT=8004
RAG_API_PREFIX=/api/v1/rag

# Vector Search
DEFAULT_TOP_K=10
DEFAULT_SCORE_THRESHOLD=0.7
MAX_TOP_K=100

# Embeddings
EMBEDDING_DIMENSION=768
EMBEDDING_BATCH_SIZE=32
MAX_TOKEN_LENGTH=8192

# Indexing
INDEX_BATCH_SIZE=100
INDEX_WORKERS=4
AUTO_OPTIMIZE_INTERVAL=3600

# Cache
ENABLE_CACHE=true
CACHE_TTL=300
CACHE_SIZE=1000

# Reranking
ENABLE_RERANKING=true
RERANK_TOP_K=50
SKILL_WEIGHT=0.3
FRESHNESS_WEIGHT=0.1
QUALITY_WEIGHT=0.2
DIVERSITY_WEIGHT=0.1

# Backup
BACKUP_ENABLED=true
BACKUP_INTERVAL=86400
BACKUP_RETENTION_DAYS=30
BACKUP_PATH=./backups

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO

# Security
ENABLE_AUTH=false
JWT_SECRET=your-secret-key
API_KEY_HEADER=X-API-Key

# Performance
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
CONNECTION_POOL_SIZE=10
"""
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(template)
    
    print(f"✅ Environment template generated: {file_path}")

if __name__ == "__main__":
    # Validate configuration
    validate_rag_config()
    
    # Generate environment template
    generate_env_template()
    
    # Print current configuration
    print("\n📋 Current RAG Configuration:")
    import json
    print(json.dumps(rag_config.to_dict(), indent=2, ensure_ascii=False))
