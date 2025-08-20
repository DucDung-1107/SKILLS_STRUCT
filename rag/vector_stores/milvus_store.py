#!/usr/bin/env python3

import logging
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
import json
import numpy as np

try:
    from pymilvus import (
        connections, 
        Collection, 
        CollectionSchema, 
        FieldSchema, 
        DataType,
        utility
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.warning("Milvus not installed. Install with: pip install pymilvus")

from ..schemas import (
    RAGDocument, VectorStoreConfig, DocumentMetadata, 
    VectorStoreStats, DocumentType, MilvusConfig
)

logger = logging.getLogger(__name__)

class MilvusVectorStore:
    """
    Advanced Milvus vector store for SkillStruct RAG system
    """
    
    def __init__(self, config: Union[VectorStoreConfig, MilvusConfig]):
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus is not installed. Please install pymilvus.")
        
        self.config = config
        self.collection_name = config.collection_name
        self.vector_dim = 768  # Default for Google embeddings
        self.metric_type = getattr(config, 'metric_type', 'COSINE')
        self.index_type = getattr(config, 'index_type', 'IVF_FLAT')
        self.nlist = getattr(config, 'nlist', 1024)
        
        # Handle both VectorStoreConfig and MilvusConfig
        if isinstance(config, MilvusConfig):
            self.host = config.host
            self.port = config.port
            self.user = config.user
            self.password = config.password
        else:
            # Parse from connection_url
            url_parts = config.connection_url.replace("http://", "").replace("https://", "")
            if ":" in url_parts:
                self.host, port_str = url_parts.split(":")
                self.port = int(port_str)
            else:
                self.host = url_parts
                self.port = 19530
            self.user = ""
            self.password = ""
        
        self._connect()
        self._setup_collection()
    
    def _connect(self) -> None:
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user if self.user else None,
                password=self.password if self.password else None
            )
            
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    def _setup_collection(self) -> None:
        """Setup Milvus collection with proper schema"""
        try:
            # Drop existing collection if exists
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped existing collection: {self.collection_name}")
            
            # Define schema
            fields = [
                FieldSchema(
                    name="id", 
                    dtype=DataType.VARCHAR, 
                    is_primary=True, 
                    max_length=256
                ),
                FieldSchema(
                    name="content", 
                    dtype=DataType.VARCHAR, 
                    max_length=65535
                ),
                FieldSchema(
                    name="doc_type", 
                    dtype=DataType.VARCHAR, 
                    max_length=50
                ),
                FieldSchema(
                    name="source", 
                    dtype=DataType.VARCHAR, 
                    max_length=512
                ),
                FieldSchema(
                    name="created_at", 
                    dtype=DataType.VARCHAR, 
                    max_length=50
                ),
                FieldSchema(
                    name="skill_ids", 
                    dtype=DataType.VARCHAR, 
                    max_length=2048
                ),
                FieldSchema(
                    name="employee_ids", 
                    dtype=DataType.VARCHAR, 
                    max_length=2048
                ),
                FieldSchema(
                    name="tags", 
                    dtype=DataType.VARCHAR, 
                    max_length=1024
                ),
                FieldSchema(
                    name="metadata_json", 
                    dtype=DataType.VARCHAR, 
                    max_length=8192
                ),
                FieldSchema(
                    name="confidence_score", 
                    dtype=DataType.FLOAT
                ),
                FieldSchema(
                    name="embedding", 
                    dtype=DataType.FLOAT_VECTOR, 
                    dim=self.vector_dim
                )
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description=f"SkillStruct RAG collection for {self.collection_name}"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            
            # Create index
            self._create_index()
            
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def _create_index(self) -> None:
        """Create vector index for fast similarity search"""
        try:
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": self.nlist}
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info(f"Created index with type: {self.index_type}")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    async def initialize(self) -> bool:
        """
        Initialize the vector store (async version)
        
        Returns:
            Success status
        """
        try:
            # Load collection into memory
            self.collection.load()
            logger.info(f"Collection {self.collection_name} loaded into memory")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            return False
    
    async def insert_document(self, document: RAGDocument, collection_name: Optional[str] = None) -> bool:
        """
        Insert a single document (async version)
        
        Args:
            document: RAGDocument to insert
            collection_name: Optional collection name override
            
        Returns:
            Success status
        """
        try:
            # Use specified collection or default
            target_collection = collection_name or self.collection_name
            
            # Prepare data
            data = self._prepare_document_data([document])
            
            # Insert
            self.collection.insert(data)
            
            logger.debug(f"Inserted document {document.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert document {document.id}: {e}")
            return False
    
    async def delete_document(self, document_id: str, collection_name: Optional[str] = None) -> bool:
        """
        Delete a document (async version)
        
        Args:
            document_id: ID of document to delete
            collection_name: Optional collection name override
            
        Returns:
            Success status
        """
        try:
            # Delete by ID
            expr = f'id == "{document_id}"'
            self.collection.delete(expr)
            
            logger.debug(f"Deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of vector store (async version)
        
        Returns:
            Health status information
        """
        try:
            # Check connection
            if not connections.has_connection("default"):
                return {
                    "status": "unhealthy",
                    "error": "No connection to Milvus"
                }
            
            # Check collection
            if not utility.has_collection(self.collection_name):
                return {
                    "status": "unhealthy", 
                    "error": f"Collection {self.collection_name} not found"
                }
            
            # Get collection stats
            stats = self.collection.num_entities
            
            return {
                "status": "healthy",
                "collection_name": self.collection_name,
                "document_count": stats,
                "host": self.host,
                "port": self.port
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def backup_collection(self, collection_name: str, backup_path: str) -> bool:
        """
        Backup collection (async version)
        
        Args:
            collection_name: Name of collection to backup
            backup_path: Path to save backup
            
        Returns:
            Success status
        """
        try:
            # This is a simplified backup - in production you'd want proper backup mechanisms
            logger.info(f"Backup functionality not implemented for Milvus in this version")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup collection: {e}")
            return False
    
    async def restore_collection(self, collection_name: str, backup_path: str) -> bool:
        """
        Restore collection (async version)
        
        Args:
            collection_name: Name of collection to restore
            backup_path: Path to backup file
            
        Returns:
            Success status
        """
        try:
            # This is a simplified restore - in production you'd want proper restore mechanisms
            logger.info(f"Restore functionality not implemented for Milvus in this version")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore collection: {e}")
            return False
    
    async def optimize_collection(self, collection_name: str) -> bool:
        """
        Optimize collection (async version)
        
        Args:
            collection_name: Name of collection to optimize
            
        Returns:
            Success status
        """
        try:
            # Compact collection
            self.collection.compact()
            logger.info(f"Optimized collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize collection: {e}")
            return False
    
    async def close(self) -> None:
        """
        Close connection (async version)
        """
        try:
            connections.disconnect("default")
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def _prepare_document_data(self, documents: List[RAGDocument]) -> Dict[str, List]:
        """
        Prepare document data for Milvus insertion
        
        Args:
            documents: List of RAGDocument objects
            
        Returns:
            Dictionary with lists for each field
        """
        data = {
            "id": [],
            "content": [],
            "doc_type": [],
            "source": [],
            "created_at": [],
            "skill_ids": [],
            "employee_ids": [],
            "tags": [],
            "metadata_json": [],
            "confidence_score": [],
            "embedding": []
        }
        
        for doc in documents:
            data["id"].append(doc.id)
            data["content"].append(doc.content[:65534])  # Truncate if too long
            data["doc_type"].append(doc.metadata.doc_type.value if doc.metadata else "unknown")
            data["source"].append(doc.metadata.source[:511] if doc.metadata else "unknown")
            data["created_at"].append(doc.metadata.created_at.isoformat() if doc.metadata else datetime.now().isoformat())
            data["skill_ids"].append(json.dumps(doc.metadata.skill_ids[:50]) if doc.metadata else "[]")  # Limit and serialize
            data["employee_ids"].append(json.dumps(doc.metadata.employee_ids[:50]) if doc.metadata else "[]")
            data["tags"].append(json.dumps(doc.metadata.tags[:20]) if doc.metadata else "[]")
            data["metadata_json"].append(json.dumps(doc.metadata.custom_fields) if doc.metadata and doc.metadata.custom_fields else "{}")
            data["confidence_score"].append(doc.metadata.confidence_score if doc.metadata and doc.metadata.confidence_score else 0.0)
            data["embedding"].append(doc.embedding if doc.embedding else [0.0] * self.vector_dim)
        
        return data
    
    def _prepare_document_data(self, documents: List[RAGDocument]) -> Dict[str, List]:
        """
        Prepare document data for Milvus insertion
        
        Args:
            documents: List of RAGDocument objects
            
        Returns:
            Dict with prepared data lists
        """
        data = {
            "id": [],
            "content": [],
            "doc_type": [],
            "source": [],
            "created_at": [],
            "skill_ids": [],
            "employee_ids": [],
            "tags": [],
            "metadata_json": [],
            "confidence_score": [],
            "embedding": []
        }
        
        for doc in documents:
            data["id"].append(doc.id)
            data["content"].append(doc.content[:65534])  # Truncate if too long
            data["doc_type"].append(doc.metadata.doc_type.value if doc.metadata else "unknown")
            data["source"].append(doc.metadata.source[:511] if doc.metadata else "unknown")
            data["created_at"].append(doc.metadata.created_at.isoformat() if doc.metadata else datetime.now().isoformat())
            data["skill_ids"].append(json.dumps(doc.metadata.skill_ids[:50]) if doc.metadata else "[]")  # Limit and serialize
            data["employee_ids"].append(json.dumps(doc.metadata.employee_ids[:50]) if doc.metadata else "[]")
            data["tags"].append(json.dumps(doc.metadata.tags[:20]) if doc.metadata else "[]")
            data["metadata_json"].append(json.dumps(doc.metadata.custom_fields) if doc.metadata and doc.metadata.custom_fields else "{}")
            data["confidence_score"].append(doc.metadata.confidence_score if doc.metadata and doc.metadata.confidence_score else 0.0)
            data["embedding"].append(doc.embedding if doc.embedding else [0.0] * self.vector_dim)
        
        return data
    
    def add_documents(self, documents: List[RAGDocument]) -> Dict[str, Any]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of RAGDocument objects
            
        Returns:
            Dict with operation results
        """
        try:
            if not documents:
                return {"success": True, "inserted": 0, "errors": []}
            
            # Prepare data for insertion
            data = {
                "id": [],
                "content": [],
                "doc_type": [],
                "source": [],
                "created_at": [],
                "skill_ids": [],
                "employee_ids": [],
                "tags": [],
                "metadata_json": [],
                "confidence_score": [],
                "embedding": []
            }
            
            errors = []
            
            for doc in documents:
                try:
                    if not doc.embedding:
                        errors.append(f"Document {doc.id} has no embedding")
                        continue
                    
                    data["id"].append(doc.id)
                    data["content"].append(doc.content[:65535])  # Truncate if too long
                    data["doc_type"].append(doc.metadata.doc_type.value)
                    data["source"].append(doc.metadata.source[:512])
                    data["created_at"].append(doc.metadata.created_at.isoformat())
                    data["skill_ids"].append(",".join(doc.metadata.skill_ids)[:2048])
                    data["employee_ids"].append(",".join(doc.metadata.employee_ids)[:2048])
                    data["tags"].append(",".join(doc.metadata.tags)[:1024])
                    data["metadata_json"].append(json.dumps(doc.metadata.custom_fields)[:8192])
                    data["confidence_score"].append(doc.metadata.confidence_score or 0.0)
                    data["embedding"].append(doc.embedding)
                    
                except Exception as e:
                    errors.append(f"Error processing document {doc.id}: {e}")
            
            if not data["id"]:
                return {"success": False, "inserted": 0, "errors": errors}
            
            # Insert data
            insert_result = self.collection.insert(data)
            self.collection.flush()
            
            logger.info(f"Inserted {len(data['id'])} documents into {self.collection_name}")
            
            return {
                "success": True,
                "inserted": len(data["id"]),
                "errors": errors,
                "insert_ids": insert_result.primary_keys
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return {"success": False, "inserted": 0, "errors": [str(e)]}
    
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        score_threshold: float = 0.7,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            filters: Additional filters
            
        Returns:
            List of search results
        """
        try:
            # Load collection
            self.collection.load()
            
            # Build filter expression
            filter_expr = None
            if filters:
                filter_conditions = []
                
                if "doc_types" in filters:
                    doc_types_str = "', '".join(filters["doc_types"])
                    filter_conditions.append(f"doc_type in ['{doc_types_str}']")
                
                if "skill_ids" in filters:
                    # This is a simple contains check - in production use more sophisticated filtering
                    skill_filter = " or ".join([f"skill_ids like '%{skill_id}%'" for skill_id in filters["skill_ids"]])
                    filter_conditions.append(f"({skill_filter})")
                
                if "confidence_min" in filters:
                    filter_conditions.append(f"confidence_score >= {filters['confidence_min']}")
                
                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)
            
            # Search parameters
            search_params = {
                "metric_type": self.metric_type,
                "params": {"nprobe": min(16, self.nlist)}
            }
            
            # Perform search
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=[
                    "content", "doc_type", "source", "created_at",
                    "skill_ids", "employee_ids", "tags", "metadata_json", "confidence_score"
                ]
            )
            
            # Process results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        result = {
                            "id": hit.id,
                            "score": float(hit.score),
                            "content": hit.entity.get("content"),
                            "doc_type": hit.entity.get("doc_type"),
                            "source": hit.entity.get("source"),
                            "created_at": hit.entity.get("created_at"),
                            "skill_ids": hit.entity.get("skill_ids", "").split(",") if hit.entity.get("skill_ids") else [],
                            "employee_ids": hit.entity.get("employee_ids", "").split(",") if hit.entity.get("employee_ids") else [],
                            "tags": hit.entity.get("tags", "").split(",") if hit.entity.get("tags") else [],
                            "confidence_score": hit.entity.get("confidence_score"),
                            "metadata": json.loads(hit.entity.get("metadata_json", "{}"))
                        }
                        formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results above threshold {score_threshold}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_documents(self, doc_ids: List[str]) -> Dict[str, Any]:
        """
        Delete documents by IDs
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Dict with operation results
        """
        try:
            if not doc_ids:
                return {"success": True, "deleted": 0}
            
            # Create delete expression
            ids_str = "', '".join(doc_ids)
            delete_expr = f"id in ['{ids_str}']"
            
            # Delete documents
            delete_result = self.collection.delete(delete_expr)
            self.collection.flush()
            
            logger.info(f"Deleted {len(doc_ids)} documents from {self.collection_name}")
            
            return {
                "success": True,
                "deleted": len(doc_ids),
                "delete_count": delete_result.delete_count
            }
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return {"success": False, "deleted": 0, "error": str(e)}
    
    def get_stats(self) -> VectorStoreStats:
        """
        Get vector store statistics
        
        Returns:
            VectorStoreStats object
        """
        try:
            # Get collection stats
            self.collection.load()
            num_entities = self.collection.num_entities
            
            # Get collection info
            collection_info = utility.describe_collection(self.collection_name)
            
            return VectorStoreStats(
                collection_name=self.collection_name,
                total_vectors=num_entities,
                vector_size=self.vector_dim,
                storage_size_mb=0.0,  
                index_type=self.index_type,
                query_performance={
                    "metric_type": self.metric_type,
                    "nlist": self.nlist
                },
                last_sync=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return VectorStoreStats(
                collection_name=self.collection_name,
                total_vectors=0,
                vector_size=self.vector_dim,
                storage_size_mb=0.0,
                index_type=self.index_type,
                query_performance={},
                last_sync=datetime.now()
            )
    
    def health_check(self) -> bool:
        """
        Check if vector store is healthy
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check connection
            if not connections.has_connection("default"):
                return False
            
            # Check collection
            if not utility.has_collection(self.collection_name):
                return False
            
            # Try a simple operation
            self.collection.load()
            _ = self.collection.num_entities
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """
        Reset (recreate) the collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Drop existing collection
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Dropped collection: {self.collection_name}")
            
            # Recreate collection
            self._setup_collection()
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False
    
    def backup_collection(self, backup_path: str) -> bool:
        """
        Backup collection data (simple export)
        
        Args:
            backup_path: Path to save backup
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # This is a simplified backup - in production use Milvus backup tools
            results = self.collection.query(
                expr="id != ''",
                output_fields=["*"]
            )
            
            backup_data = {
                "collection_name": self.collection_name,
                "vector_dim": self.vector_dim,
                "backup_time": datetime.now().isoformat(),
                "documents": results
            }
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Backed up {len(results)} documents to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
