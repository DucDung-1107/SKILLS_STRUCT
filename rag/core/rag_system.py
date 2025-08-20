#!/usr/bin/env python3
"""
🧠 SkillStruct RAG Core System
Complete RAG system tích hợp tất cả components
"""

import logging
from typing import List, Dict, Optional, Any, Union
import asyncio
from datetime import datetime
import json

from ..schemas import (
    RAGDocument, RAGQuery, RAGResponse, QueryType,
    EmbeddingConfig, EmbeddingModel, BatchEmbeddingRequest,
    BatchEmbeddingResponse, BatchIndexRequest, BatchIndexResponse,
    IndexingStatus, RAGConfig
)
from ..vector_stores.milvus_store import MilvusVectorStore
from ..embeddings.skill_embedder import SkillStructEmbedder
from ..retrievers.advanced_retriever import AdvancedRetriever
from utils.core_utils import clean_text
from database.core_schema import SkillNodeResponse

logger = logging.getLogger(__name__)

class SkillStructRAG:
    """
    Main RAG system cho SkillStruct
    Orchestrates all RAG components
    """
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._initialized = False
        
        # Components
        self.vector_store: Optional[MilvusVectorStore] = None
        self.embedder: Optional[SkillStructEmbedder] = None
        self.retriever: Optional[AdvancedRetriever] = None
        
        # State tracking
        self.indexed_documents = set()
        self.indexing_stats = {
            "total_documents": 0,
            "successful_indexing": 0,
            "failed_indexing": 0,
            "last_update": None
        }
    
    async def initialize(self) -> bool:
        """
        Initialize RAG system
        
        Returns:
            Success status
        """
        try:
            logger.info("Initializing SkillStruct RAG system...")
            
            # Initialize vector store
            self.vector_store = MilvusVectorStore(self.config.vector_store_config)
            await self.vector_store.initialize()
            
            # Initialize embedder
            self.embedder = SkillStructEmbedder(self.config.embedding_config)
            
            # Initialize retriever
            self.retriever = AdvancedRetriever(self.vector_store)
            
            self._initialized = True
            logger.info("SkillStruct RAG system initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def _ensure_initialized(self):
        """Ensure system is initialized"""
        if not self._initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
    
    async def index_skill_taxonomy(
        self, 
        taxonomy_data: Dict[str, Any],
        collection_name: Optional[str] = None
    ) -> BatchIndexResponse:
        """
        Index complete skill taxonomy
        
        Args:
            taxonomy_data: Complete skill taxonomy data
            collection_name: Optional custom collection name
            
        Returns:
            Batch indexing response
        """
        self._ensure_initialized()
        
        try:
            logger.info("Starting skill taxonomy indexing...")
            start_time = datetime.now()
            
            # Validate taxonomy data
            
            # Embed skill taxonomy
            documents = self.embedder.embed_skill_taxonomy(taxonomy_data)
            
            if not documents:
                return BatchIndexResponse(
                    batch_id=f"skill_taxonomy_{int(start_time.timestamp())}",
                    status=IndexingStatus.FAILED,
                    total_documents=0,
                    successful_documents=0,
                    failed_documents=0,
                    error="No documents generated from taxonomy"
                )
            
            # Index documents
            collection = collection_name or self.config.default_collection_name
            success_count = 0
            
            for doc in documents:
                try:
                    await self.vector_store.insert_document(doc, collection)
                    self.indexed_documents.add(doc.id)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to index document {doc.id}: {e}")
            
            # Update stats
            self.indexing_stats["total_documents"] += len(documents)
            self.indexing_stats["successful_indexing"] += success_count
            self.indexing_stats["failed_indexing"] += len(documents) - success_count
            self.indexing_stats["last_update"] = datetime.now().isoformat()
            
            end_time = datetime.now()
            
            return BatchIndexResponse(
                batch_id=f"skill_taxonomy_{int(start_time.timestamp())}",
                status=IndexingStatus.COMPLETED if success_count == len(documents) else IndexingStatus.PARTIAL,
                total_documents=len(documents),
                successful_documents=success_count,
                failed_documents=len(documents) - success_count,
                processing_time_ms=int((end_time - start_time).total_seconds() * 1000),
                metadata={
                    "taxonomy_nodes": len(taxonomy_data.get("nodes", [])),
                    "taxonomy_edges": len(taxonomy_data.get("edges", [])),
                    "collection_name": collection
                }
            )
            
        except Exception as e:
            logger.error(f"Error indexing skill taxonomy: {e}")
            return BatchIndexResponse(
                batch_id=f"skill_taxonomy_{int(datetime.now().timestamp())}",
                status=IndexingStatus.FAILED,
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                error=str(e)
            )
    
    async def index_resumes(
        self, 
        resume_data_list: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> BatchIndexResponse:
        """
        Index multiple resumes
        
        Args:
            resume_data_list: List of resume data
            collection_name: Optional custom collection name
            
        Returns:
            Batch indexing response
        """
        self._ensure_initialized()
        
        try:
            logger.info(f"Starting resume indexing for {len(resume_data_list)} resumes...")
            start_time = datetime.now()
            
            all_documents = []
            
            # Process resumes in batches
            batch_size = self.config.embedding_config.batch_size
            for i in range(0, len(resume_data_list), batch_size):
                batch_resumes = resume_data_list[i:i + batch_size]
                
                for resume_data in batch_resumes:
                    try:
                        documents = self.embedder.embed_resume(resume_data)
                        all_documents.extend(documents)
                    except Exception as e:
                        logger.error(f"Failed to embed resume {resume_data.get('id', 'unknown')}: {e}")
            
            if not all_documents:
                return BatchIndexResponse(
                    batch_id=f"resumes_{int(start_time.timestamp())}",
                    status=IndexingStatus.FAILED,
                    total_documents=0,
                    successful_documents=0,
                    failed_documents=len(resume_data_list),
                    error="No documents generated from resumes"
                )
            
            # Index documents
            collection = collection_name or self.config.default_collection_name
            success_count = 0
            
            for doc in all_documents:
                try:
                    await self.vector_store.insert_document(doc, collection)
                    self.indexed_documents.add(doc.id)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to index document {doc.id}: {e}")
            
            # Update stats
            self.indexing_stats["total_documents"] += len(all_documents)
            self.indexing_stats["successful_indexing"] += success_count
            self.indexing_stats["failed_indexing"] += len(all_documents) - success_count
            self.indexing_stats["last_update"] = datetime.now().isoformat()
            
            end_time = datetime.now()
            
            return BatchIndexResponse(
                batch_id=f"resumes_{int(start_time.timestamp())}",
                status=IndexingStatus.COMPLETED if success_count == len(all_documents) else IndexingStatus.PARTIAL,
                total_documents=len(all_documents),
                successful_documents=success_count,
                failed_documents=len(all_documents) - success_count,
                processing_time_ms=int((end_time - start_time).total_seconds() * 1000),
                metadata={
                    "resume_count": len(resume_data_list),
                    "document_count": len(all_documents),
                    "collection_name": collection
                }
            )
            
        except Exception as e:
            logger.error(f"Error indexing resumes: {e}")
            return BatchIndexResponse(
                batch_id=f"resumes_{int(datetime.now().timestamp())}",
                status=IndexingStatus.FAILED,
                total_documents=0,
                successful_documents=0,
                failed_documents=len(resume_data_list),
                error=str(e)
            )
    
    async def search(
        self, 
        query_request: RAGQuery,
        collection_name: Optional[str] = None
    ) -> RAGResponse:
        """
        Perform search query
        
        Args:
            query_request: Search query request
            collection_name: Optional custom collection name
            
        Returns:
            Query response with results
        """
        self._ensure_initialized()
        
        try:
            # Set collection if provided
            if collection_name:
                original_collection = self.vector_store.collection_name
                self.vector_store.collection_name = collection_name
            
            # Perform retrieval
            response = self.retriever.retrieve(
                query_request,
                reranking_config=self.config.reranking_config
            )
            
            # Restore original collection
            if collection_name:
                self.vector_store.collection_name = original_collection
            
            return response
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            return RAGResponse(
                query_id=query_request.query_id or "error",
                documents=[],
                metrics=None,
                error=str(e)
            )
    
    async def find_similar_skills(
        self, 
        skill_name: str, 
        top_k: int = 10,
        collection_name: Optional[str] = None
    ) -> RAGResponse:
        """
        Find skills similar to given skill
        
        Args:
            skill_name: Target skill name
            top_k: Number of results to return
            collection_name: Optional custom collection name
            
        Returns:
            Query response with similar skills
        """
        query_request = RAGQuery(
            query=f"Find skills similar to {skill_name}",
            query_type=QueryType.SKILL_SEARCH,
            top_k=top_k,
            include_facets=True
        )
        
        return await self.search(query_request, collection_name)
    
    async def find_candidates_with_skills(
        self, 
        required_skills: List[str], 
        top_k: int = 10,
        collection_name: Optional[str] = None
    ) -> RAGResponse:
        """
        Find candidates with specific skills
        
        Args:
            required_skills: List of required skills
            top_k: Number of results to return
            collection_name: Optional custom collection name
            
        Returns:
            Query response with matching candidates
        """
        skills_query = " AND ".join(required_skills)
        query_request = RAGQuery(
            query=f"Find candidates with skills: {skills_query}",
            query_type=QueryType.CANDIDATE_SEARCH,
            top_k=top_k,
            include_facets=True
        )
        
        return await self.search(query_request, collection_name)
    
    async def get_skill_recommendations(
        self, 
        current_skills: List[str], 
        top_k: int = 5,
        collection_name: Optional[str] = None
    ) -> RAGResponse:
        """
        Get skill recommendations based on current skills
        
        Args:
            current_skills: List of current skills
            top_k: Number of recommendations
            collection_name: Optional custom collection name
            
        Returns:
            Query response with skill recommendations
        """
        skills_context = ", ".join(current_skills)
        query_request = RAGQuery(
            query=f"Recommend complementary skills for someone with: {skills_context}",
            query_type=QueryType.RECOMMENDATION,
            top_k=top_k,
            include_facets=True
        )
        
        return await self.search(query_request, collection_name)
    
    async def update_document(
        self, 
        document_id: str, 
        updated_data: Dict[str, Any],
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Update existing document
        
        Args:
            document_id: Document ID to update
            updated_data: Updated document data
            collection_name: Optional custom collection name
            
        Returns:
            Success status
        """
        self._ensure_initialized()
        
        try:
            collection = collection_name or self.config.default_collection_name
            
            # Delete old document
            await self.vector_store.delete_document(document_id, collection)
            
            # Create new document
            if "skill_node" in updated_data:
                # Skill node update
                skill_node = SkillNodeResponse(**updated_data["skill_node"])
                taxonomy_data = updated_data.get("taxonomy_data", {})
                
                context = self.embedder.create_skill_context(skill_node, taxonomy_data)
                documents = [RAGDocument(
                    id=document_id,
                    content=context,
                    metadata=updated_data.get("metadata", {})
                )]
                
                # Embed and index
                embeddings = self.embedder.embedding_model.embed_documents([context])
                documents[0].embedding = embeddings[0]
                
            elif "resume_data" in updated_data:
                # Resume update
                documents = self.embedder.embed_resume(updated_data["resume_data"])
            else:
                raise ValueError("Invalid update data format")
            
            # Insert updated documents
            for doc in documents:
                await self.vector_store.insert_document(doc, collection)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
    
    async def delete_document(
        self, 
        document_id: str,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete document from index
        
        Args:
            document_id: Document ID to delete
            collection_name: Optional custom collection name
            
        Returns:
            Success status
        """
        self._ensure_initialized()
        
        try:
            collection = collection_name or self.config.default_collection_name
            success = await self.vector_store.delete_document(document_id, collection)
            
            if success:
                self.indexed_documents.discard(document_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status
        
        Returns:
            Health status information
        """
        try:
            if not self._initialized:
                return {
                    "status": "not_initialized",
                    "initialized": False,
                    "error": "System not initialized"
                }
            
            # Check vector store health
            vector_store_health = await self.vector_store.health_check()
            
            # Get component stats
            embedding_stats = self.embedder.get_embedding_stats()
            retrieval_stats = self.retriever.get_retrieval_stats()
            
            return {
                "status": "healthy" if vector_store_health["status"] == "healthy" else "unhealthy",
                "initialized": self._initialized,
                "components": {
                    "vector_store": vector_store_health,
                    "embedder": embedding_stats,
                    "retriever": retrieval_stats
                },
                "indexing_stats": self.indexing_stats,
                "indexed_documents_count": len(self.indexed_documents),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "error",
                "initialized": self._initialized,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def backup_index(
        self, 
        backup_path: str,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Backup vector index
        
        Args:
            backup_path: Path to save backup
            collection_name: Optional custom collection name
            
        Returns:
            Success status
        """
        self._ensure_initialized()
        
        try:
            collection = collection_name or self.config.default_collection_name
            return await self.vector_store.backup_collection(collection, backup_path)
            
        except Exception as e:
            logger.error(f"Error backing up index: {e}")
            return False
    
    async def restore_index(
        self, 
        backup_path: str,
        collection_name: Optional[str] = None
    ) -> bool:
        """
        Restore vector index from backup
        
        Args:
            backup_path: Path to backup file
            collection_name: Optional custom collection name
            
        Returns:
            Success status
        """
        self._ensure_initialized()
        
        try:
            collection = collection_name or self.config.default_collection_name
            return await self.vector_store.restore_collection(collection, backup_path)
            
        except Exception as e:
            logger.error(f"Error restoring index: {e}")
            return False
    
    async def optimize_index(self, collection_name: Optional[str] = None) -> bool:
        """
        Optimize vector index for better performance
        
        Args:
            collection_name: Optional custom collection name
            
        Returns:
            Success status
        """
        self._ensure_initialized()
        
        try:
            collection = collection_name or self.config.default_collection_name
            return await self.vector_store.optimize_collection(collection)
            
        except Exception as e:
            logger.error(f"Error optimizing index: {e}")
            return False
    
    async def get_analytics(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get RAG system analytics
        
        Args:
            collection_name: Optional custom collection name
            
        Returns:
            Analytics data
        """
        self._ensure_initialized()
        
        try:
            collection = collection_name or self.config.default_collection_name
            
            # Get collection stats
            collection_stats = self.vector_store.get_collection_stats(collection)
            
            # Calculate document type distribution
            # This would require querying the metadata, simplified here
            doc_type_distribution = {
                "skill_nodes": collection_stats.get("entity_count", 0) // 2,  # Rough estimate
                "resume_documents": collection_stats.get("entity_count", 0) // 2
            }
            
            return {
                "collection_stats": collection_stats,
                "indexing_stats": self.indexing_stats,
                "document_distribution": doc_type_distribution,
                "system_config": {
                    "embedding_model": self.config.embedding_config.model_name.value,
                    "vector_size": self.config.embedding_config.vector_size,
                    "default_collection": self.config.default_collection_name
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.vector_store:
                await self.vector_store.close()
            
            self._initialized = False
            logger.info("RAG system cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up RAG system: {e}")

# Factory function for easy initialization
async def create_skillstruct_rag(
    milvus_host: str = "localhost",
    milvus_port: int = 19530,
    google_api_key: str = "",
    collection_name: str = "skillstruct_rag"
) -> SkillStructRAG:
    """
    Factory function để tạo SkillStruct RAG system
    
    Args:
        milvus_host: Milvus server host
        milvus_port: Milvus server port
        google_api_key: Google AI API key
        collection_name: Default collection name
        
    Returns:
        Initialized SkillStruct RAG system
    """
    from ..schemas import MilvusConfig
    
    # Create config
    config = RAGConfig(
        vector_store_config=MilvusConfig(
            host=milvus_host,
            port=milvus_port,
            collection_name=collection_name
        ),
        embedding_config=EmbeddingConfig(
            model_name=EmbeddingModel.GOOGLE_EMBEDDING_001,
            api_key=google_api_key
        ),
        default_collection_name=collection_name
    )
    
    # Create and initialize RAG system
    rag_system = SkillStructRAG(config)
    await rag_system.initialize()
    
    return rag_system
