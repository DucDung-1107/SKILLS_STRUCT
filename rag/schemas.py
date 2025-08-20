#!/usr/bin/env python3
"""
📋 RAG Schema - Pydantic models cho RAG system
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# ===========================
#  RAG ENUMS
# ===========================

class EmbeddingModel(str, Enum):
    """Supported embedding models"""
    GOOGLE_EMBEDDING = "models/embedding-001"
    OPENAI_ADA_002 = "text-embedding-ada-002"
    SENTENCE_TRANSFORMERS = "sentence-transformers/all-MiniLM-L6-v2"

class VectorStoreType(str, Enum):
    """Supported vector stores"""
    MILVUS = "milvus"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    FAISS = "faiss"

class QueryType(str, Enum):
    """Types of queries"""
    SKILL_SEARCH = "skill_search"
    RESUME_ANALYSIS = "resume_analysis"
    CANDIDATE_MATCHING = "candidate_matching"
    TAXONOMY_NAVIGATION = "taxonomy_navigation"
    GENERAL_QA = "general_qa"

class DocumentType(str, Enum):
    """Types of documents in RAG"""
    SKILL_NODE = "skill_node"
    RESUME = "resume"
    JOB_DESCRIPTION = "job_description"
    TAXONOMY_CONTEXT = "taxonomy_context"
    EMPLOYEE_PROFILE = "employee_profile"

class IndexingStatus(str, Enum):
    """Status of indexing operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

# ===========================
#  CORE RAG MODELS
# ===========================

class EmbeddingConfig(BaseModel):
    """Configuration for embedding models"""
    model_name: EmbeddingModel = EmbeddingModel.GOOGLE_EMBEDDING
    api_key: str
    vector_size: int = Field(default=768, ge=128, le=4096)
    batch_size: int = Field(default=100, ge=1, le=1000)
    max_tokens: int = Field(default=512, ge=1, le=8192)

class VectorStoreConfig(BaseModel):
    """Configuration for vector stores"""
    store_type: VectorStoreType = VectorStoreType.MILVUS
    connection_url: str = "http://localhost:19530"
    collection_name: str = "skillstruct_embeddings"
    metric_type: str = "COSINE"
    index_type: str = "IVF_FLAT"
    nlist: int = 1024

class MilvusConfig(BaseModel):
    """Specific configuration for Milvus vector store"""
    store_type: VectorStoreType = VectorStoreType.MILVUS
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    collection_name: str = "skillstruct_embeddings"
    metric_type: str = "COSINE"
    index_type: str = "IVF_FLAT"
    nlist: int = 1024
    
    @property
    def connection_url(self) -> str:
        return f"http://{self.host}:{self.port}"

class DocumentMetadata(BaseModel):
    """Metadata for documents in RAG"""
    doc_id: str
    doc_type: DocumentType
    source: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    tags: List[str] = []
    skill_ids: List[str] = []
    employee_ids: List[str] = []
    confidence_score: Optional[float] = None
    custom_fields: Dict[str, Any] = {}

class RAGDocument(BaseModel):
    """Document structure for RAG system"""
    id: str
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None
    chunk_index: Optional[int] = None
    chunk_count: Optional[int] = None

# ===========================
#  QUERY & RESPONSE MODELS
# ===========================

class RAGQuery(BaseModel):
    """RAG query request"""
    question: str = Field(..., min_length=1, max_length=2000)
    query_type: QueryType = QueryType.GENERAL_QA
    session_id: Optional[str] = None
    context_filters: Dict[str, Any] = {}
    top_k: int = Field(default=5, ge=1, le=50)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    include_metadata: bool = True
    clear_history: bool = False

class RAGContext(BaseModel):
    """Context retrieved for RAG"""
    document_id: str
    content: str
    score: float
    metadata: DocumentMetadata
    chunk_info: Optional[Dict[str, Any]] = None

class RAGResponse(BaseModel):
    """RAG system response"""
    answer: str
    question: str
    contexts: List[RAGContext]
    session_id: str
    query_type: QueryType
    response_time_ms: float
    total_contexts: int
    confidence_score: float
    timestamp: datetime
    
class RAGSession(BaseModel):
    """RAG conversation session"""
    session_id: str
    created_at: datetime
    updated_at: datetime
    query_count: int = 0
    total_contexts_used: int = 0
    avg_response_time: float = 0.0
    conversation_history: List[Dict[str, Any]] = []

class QueryResponse(BaseModel):
    results: list[Any] = []

# ===========================
#  SKILL-SPECIFIC MODELS
# ===========================

class SkillEmbedding(BaseModel):
    """Skill node embedding information"""
    skill_id: str
    skill_name: str
    skill_type: str
    level: int
    parent_skills: List[str] = []
    child_skills: List[str] = []
    related_skills: List[str] = []
    employees: List[str] = []
    context_text: str
    embedding_vector: List[float]
    last_updated: datetime

class ResumeEmbedding(BaseModel):
    """Resume embedding information"""
    resume_id: str
    file_name: str
    candidate_name: Optional[str] = None
    extracted_text: str
    skills_mentioned: List[str] = []
    experience_years: Optional[int] = None
    education_level: Optional[str] = None
    embedding_chunks: List[Dict[str, Any]] = []
    last_processed: datetime

class SkillMatchResult(BaseModel):
    """Skill matching result"""
    query_skill: str
    matched_skills: List[Dict[str, Any]]
    similarity_scores: List[float]
    context_explanation: str
    recommendation: str

# ===========================
#  ANALYTICS MODELS
# ===========================

class RAGAnalytics(BaseModel):
    """RAG system analytics"""
    total_documents: int
    total_embeddings: int
    avg_query_time: float
    top_queries: List[str]
    query_success_rate: float
    storage_size_mb: float
    last_updated: datetime

class VectorStoreStats(BaseModel):
    """Vector store statistics"""
    collection_name: str
    total_vectors: int
    vector_size: int
    storage_size_mb: float
    index_type: str
    query_performance: Dict[str, float]
    last_sync: datetime

# ===========================
#  BATCH OPERATIONS
# ===========================

class BatchEmbeddingRequest(BaseModel):
    """Batch embedding request"""
    documents: List[RAGDocument]
    overwrite_existing: bool = False
    batch_size: int = Field(default=100, ge=1, le=1000)
    priority: int = Field(default=5, ge=1, le=10)

class BatchEmbeddingResponse(BaseModel):
    """Batch embedding response"""
    total_processed: int
    successful: int
    failed: int
    processing_time_seconds: float
    failed_documents: List[str] = []
    errors: List[str] = []

class BatchIndexRequest(BaseModel):
    """Batch indexing request"""
    documents: List[RAGDocument]
    collection_name: Optional[str] = None
    overwrite_existing: bool = False
    batch_size: int = Field(default=100, ge=1, le=1000)
    priority: int = Field(default=5, ge=1, le=10)

class BatchIndexResponse(BaseModel):
    """Batch indexing response"""
    total_processed: int
    successful: int
    failed: int
    processing_time_seconds: float
    status: IndexingStatus
    failed_documents: List[str] = []
    errors: List[str] = []

# ===========================
#  SEARCH & RETRIEVAL
# ===========================

class FilterType(str, Enum):
    """Types of search filters"""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    RANGE = "range"
    IN_LIST = "in_list"
    REGEX = "regex"

class RetrievalMethod(str, Enum):
    """Document retrieval methods"""
    VECTOR_SIMILARITY = "vector_similarity"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID = "hybrid"
    FACETED = "faceted"

class RerankingConfig(BaseModel):
    """Configuration for result reranking"""
    enabled: bool = True
    method: str = "cross_encoder"
    top_k_before_rerank: int = 100
    top_k_after_rerank: int = 10
    boost_factors: Dict[str, float] = {}

class SearchMetrics(BaseModel):
    """Search performance metrics"""
    query_time_ms: float
    embedding_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    total_documents_found: int = 0
    returned_documents: int = 0
    precision: Optional[float] = None
    recall: Optional[float] = None

class RetrievedDocument(BaseModel):
    """Document retrieved from search"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    rank: int
    highlight_snippets: List[str] = []
    retrieval_method: RetrievalMethod
    relevance_explanation: Optional[str] = None

class SearchFilter(BaseModel):
    """Advanced search filters"""
    doc_types: List[DocumentType] = []
    date_range: Optional[Dict[str, datetime]] = None
    skill_filters: List[str] = []
    employee_filters: List[str] = []
    score_range: Optional[Dict[str, float]] = None
    custom_filters: Dict[str, Any] = {}

class HybridSearchRequest(BaseModel):
    """Hybrid search combining multiple methods"""
    query: str
    semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    filters: SearchFilter = SearchFilter()
    top_k: int = Field(default=10, ge=1, le=100)

class RetrievalStrategy(str, Enum):
    """Retrieval strategies"""
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximal Marginal Relevance
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    GRAPH_BASED = "graph_based"

# ===========================
#  SYSTEM HEALTH
# ===========================

class RAGSystemHealth(BaseModel):
    """RAG system health status"""
    status: str  # healthy, degraded, down
    vector_store_status: str
    embedding_service_status: str
    query_latency_ms: float
    error_rate: float
    last_health_check: datetime
    issues: List[str] = []

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    details: dict = {}

class RAGConfig(BaseModel):
    """Complete RAG system configuration"""
    embedding_config: EmbeddingConfig
    vector_store_config: VectorStoreConfig
    system_name: str = "SkillStruct RAG"
    version: str = "1.0.0"
    max_concurrent_queries: int = 10
    cache_ttl_seconds: int = 3600
    enable_analytics: bool = True
    debug_mode: bool = False

class QueryRequest(BaseModel):
    query: str

class CandidateSearchRequest(BaseModel):
    required_skills: list[str]
    top_k: int = 10
    collection_name: str = None

class SkillRecommendationRequest(BaseModel):
    current_skills: list[str]
    top_k: int = 5
    collection_name: str = None
