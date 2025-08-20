#!/usr/bin/env python3
"""
🧠 SkillStruct RAG System
Advanced Retrieval-Augmented Generation với Milvus và Google AI
"""

__version__ = "1.0.0"
__author__ = "SkillStruct Team"
__description__ = "Advanced RAG system for SkillStruct platform"

# Core imports
from .core.rag_system import SkillStructRAG, create_skillstruct_rag
from .embeddings.skill_embedder import SkillStructEmbedder
from .vector_stores.milvus_store import MilvusVectorStore
from .retrievers.advanced_retriever import AdvancedRetriever

# Configuration
from .config import RAGSystemConfig, rag_config, validate_rag_config

# Schemas
from .schemas import (
    # Core models
    RAGDocument, DocumentMetadata, DocumentType,
    QueryRequest, QueryResponse, QueryType,
    
    # Search models
    SearchFilter, FilterType, RetrievedDocument,
    SearchMetrics, RerankingConfig, RetrievalMethod,
    
    # Embedding models
    EmbeddingConfig, EmbeddingModel, SkillEmbedding,
    
    # Indexing models
    BatchIndexRequest, BatchIndexResponse, IndexingStatus,
    
    # Configuration models
    RAGConfig, MilvusConfig, HealthResponse
)

__all__ = [
    # Core classes
    "SkillStructRAG",
    "SkillStructEmbedder", 
    "MilvusVectorStore",
    "AdvancedRetriever",
    
    # Factory functions
    "create_skillstruct_rag",
    
    # Configuration
    "RAGSystemConfig",
    "rag_config",
    "validate_rag_config",
    
    # Schemas - Core
    "RAGDocument",
    "DocumentMetadata", 
    "DocumentType",
    "QueryRequest",
    "QueryResponse",
    "QueryType",
    
    # Schemas - Search
    "SearchFilter",
    "FilterType",
    "RetrievedDocument",
    "SearchMetrics",
    "RerankingConfig",
    "RetrievalMethod",
    
    # Schemas - Embedding
    "EmbeddingConfig",
    "EmbeddingModel",
    "SkillEmbedding",
    
    # Schemas - Indexing
    "BatchIndexRequest",
    "BatchIndexResponse", 
    "IndexingStatus",
    
    # Schemas - Configuration
    "RAGConfig",
    "MilvusConfig",
    "HealthResponse",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__"
]
