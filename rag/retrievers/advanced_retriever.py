#!/usr/bin/env python3
"""
🔍 SkillStruct Advanced Retriever
Intelligent retrieval system với skill-aware search và filtering
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json

from ..schemas import (
    RAGQuery, QueryType, SearchFilter, FilterType,
    RAGResponse, RetrievedDocument, SearchMetrics,
    RerankingConfig, RetrievalMethod
)
from ..vector_stores.milvus_store import MilvusVectorStore
from utils.core_utils import normalize_text, extract_skills_from_text

logger = logging.getLogger(__name__)

@dataclass
class RetrievalContext:
    """Context for retrieval operation"""
    original_query: str
    processed_query: str
    extracted_skills: List[str]
    query_type: QueryType
    user_context: Optional[Dict[str, Any]] = None
    search_history: Optional[List[str]] = None

class AdvancedRetriever:
    """
    Advanced retrieval system cho SkillStruct
    Tích hợp skill intelligence và contextual search
    """
    
    def __init__(self, vector_store: MilvusVectorStore):
        self.vector_store = vector_store
        self.skill_synonyms = self._load_skill_synonyms()
        self.retrieval_cache = {}
        
    def _load_skill_synonyms(self) -> Dict[str, List[str]]:
        """Load skill synonyms for query expansion"""
        return {
            "python": ["python", "py", "python programming", "python development"],
            "javascript": ["javascript", "js", "node.js", "nodejs", "ecmascript"],
            "java": ["java", "java programming", "jvm", "spring", "spring boot"],
            "react": ["react", "reactjs", "react.js", "react native"],
            "angular": ["angular", "angularjs", "angular.js", "ng"],
            "machine learning": ["machine learning", "ml", "artificial intelligence", "ai", "deep learning"],
            "sql": ["sql", "mysql", "postgresql", "database", "relational database"],
            "docker": ["docker", "containerization", "containers"],
            "kubernetes": ["kubernetes", "k8s", "container orchestration"],
            "aws": ["aws", "amazon web services", "cloud computing"],
            "azure": ["azure", "microsoft azure", "cloud platform"],
            "project management": ["project management", "pm", "agile", "scrum", "kanban"]
        }
    
    def retrieve(
        self, 
        query_request: RAGQuery,
        reranking_config: Optional[RerankingConfig] = None
    ) -> RAGResponse:
        """
        Main retrieval method với advanced features
        
        Args:
            query_request: Query request object
            reranking_config: Optional reranking configuration
            
        Returns:
            Query response with ranked results
        """
        try:
            start_time = datetime.now()
            
            # Prepare retrieval context
            context = self._prepare_retrieval_context(query_request)
            
            # Multi-stage retrieval
            if query_request.retrieval_method == RetrievalMethod.HYBRID:
                documents = self._hybrid_retrieval(context, query_request)
            elif query_request.retrieval_method == RetrievalMethod.SKILL_AWARE:
                documents = self._skill_aware_retrieval(context, query_request)
            else:
                documents = self._semantic_retrieval(context, query_request)
            
            # Apply filters
            if query_request.filters:
                documents = self._apply_filters(documents, query_request.filters)
            
            # Reranking
            if reranking_config and len(documents) > 1:
                documents = self._rerank_documents(documents, context, reranking_config)
            
            # Limit results
            if query_request.top_k:
                documents = documents[:query_request.top_k]
            
            # Calculate metrics
            end_time = datetime.now()
            metrics = SearchMetrics(
                total_documents=len(documents),
                search_time_ms=int((end_time - start_time).total_seconds() * 1000),
                avg_score=np.mean([doc.score for doc in documents]) if documents else 0.0,
                max_score=max([doc.score for doc in documents]) if documents else 0.0,
                min_score=min([doc.score for doc in documents]) if documents else 0.0,
                query_complexity=len(context.extracted_skills) + len(query_request.query.split()),
                filter_count=len(query_request.filters) if query_request.filters else 0
            )
            
            return RAGResponse(
                query_id=query_request.query_id or f"query_{int(datetime.now().timestamp())}",
                documents=documents,
                metrics=metrics,
                query_suggestions=self._generate_query_suggestions(context),
                facets=self._calculate_facets(documents) if query_request.include_facets else None
            )
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return RAGResponse(
                query_id=query_request.query_id or "error",
                documents=[],
                metrics=SearchMetrics(
                    total_documents=0,
                    search_time_ms=0,
                    avg_score=0.0,
                    max_score=0.0,
                    min_score=0.0,
                    query_complexity=0,
                    filter_count=0
                ),
                error=str(e)
            )
    
    def _prepare_retrieval_context(self, query_request: RAGQuery) -> RetrievalContext:
        """Prepare context for retrieval"""
        processed_query = normalize_text(query_request.query)
        extracted_skills = extract_skills_from_text(processed_query)
        
        # Expand skills with synonyms
        expanded_skills = []
        for skill in extracted_skills:
            expanded_skills.append(skill)
            if skill.lower() in self.skill_synonyms:
                expanded_skills.extend(self.skill_synonyms[skill.lower()])
        
        return RetrievalContext(
            original_query=query_request.query,
            processed_query=processed_query,
            extracted_skills=list(set(expanded_skills)),
            query_type=query_request.query_type,
            user_context=query_request.user_context
        )
    
    def _semantic_retrieval(
        self, 
        context: RetrievalContext, 
        query_request: RAGQuery
    ) -> List[RetrievedDocument]:
        """Semantic vector similarity search"""
        try:
            results = self.vector_store.search(
                query_vector=None,  # Will be embedded in vector store
                query_text=context.processed_query,
                top_k=query_request.top_k or 10,
                score_threshold=query_request.score_threshold
            )
            
            documents = []
            for result in results:
                doc = RetrievedDocument(
                    document_id=result["id"],
                    content=result["content"],
                    score=result["score"],
                    metadata=result.get("metadata", {}),
                    explanation="Semantic similarity match"
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {e}")
            return []
    
    def _skill_aware_retrieval(
        self, 
        context: RetrievalContext, 
        query_request: RAGQuery
    ) -> List[RetrievedDocument]:
        """Skill-aware retrieval với skill boosting"""
        try:
            # Get semantic results first
            semantic_docs = self._semantic_retrieval(context, query_request)
            
            # Boost scores based on skill matches
            for doc in semantic_docs:
                skill_boost = self._calculate_skill_boost(doc, context.extracted_skills)
                doc.score = min(1.0, doc.score + skill_boost)
                
                # Add skill match explanation
                if skill_boost > 0:
                    doc.explanation += f" | Skill boost: +{skill_boost:.3f}"
            
            # Sort by boosted scores
            semantic_docs.sort(key=lambda x: x.score, reverse=True)
            
            return semantic_docs
            
        except Exception as e:
            logger.error(f"Error in skill-aware retrieval: {e}")
            return []
    
    def _hybrid_retrieval(
        self, 
        context: RetrievalContext, 
        query_request: RAGQuery
    ) -> List[RetrievedDocument]:
        """Hybrid retrieval combining multiple approaches"""
        try:
            all_documents = {}
            
            # Semantic search
            semantic_docs = self._semantic_retrieval(context, query_request)
            for doc in semantic_docs:
                doc_id = doc.document_id
                if doc_id not in all_documents:
                    all_documents[doc_id] = doc
                    doc.retrieval_methods = ["semantic"]
                else:
                    # Combine scores
                    existing_doc = all_documents[doc_id]
                    existing_doc.score = max(existing_doc.score, doc.score)
                    existing_doc.retrieval_methods.append("semantic")
            
            # Skill-based search
            if context.extracted_skills:
                skill_docs = self._skill_based_search(context, query_request)
                for doc in skill_docs:
                    doc_id = doc.document_id
                    if doc_id not in all_documents:
                        all_documents[doc_id] = doc
                        doc.retrieval_methods = ["skill_based"]
                    else:
                        existing_doc = all_documents[doc_id]
                        # Weighted combination
                        existing_doc.score = 0.7 * existing_doc.score + 0.3 * doc.score
                        existing_doc.retrieval_methods.append("skill_based")
            
            # Convert to list and sort
            final_documents = list(all_documents.values())
            final_documents.sort(key=lambda x: x.score, reverse=True)
            
            return final_documents
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            return []
    
    def _skill_based_search(
        self, 
        context: RetrievalContext, 
        query_request: RAGQuery
    ) -> List[RetrievedDocument]:
        """Search based on skill matches"""
        try:
            if not context.extracted_skills:
                return []
            
            # Build skill filter
            skill_filter = {
                "skill_ids": {"$in": context.extracted_skills}
            }
            
            results = self.vector_store.search_with_metadata_filter(
                query_text=context.processed_query,
                metadata_filter=skill_filter,
                top_k=query_request.top_k or 10
            )
            
            documents = []
            for result in results:
                doc = RetrievedDocument(
                    document_id=result["id"],
                    content=result["content"],
                    score=result["score"],
                    metadata=result.get("metadata", {}),
                    explanation="Skill-based match"
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in skill-based search: {e}")
            return []
    
    def _calculate_skill_boost(self, document: RetrievedDocument, query_skills: List[str]) -> float:
        """Calculate skill boost for document"""
        try:
            doc_skills = document.metadata.get("skill_ids", [])
            if not doc_skills or not query_skills:
                return 0.0
            
            # Normalize skill names for comparison
            doc_skills_norm = [skill.lower().strip() for skill in doc_skills]
            query_skills_norm = [skill.lower().strip() for skill in query_skills]
            
            # Calculate intersection
            matching_skills = set(doc_skills_norm).intersection(set(query_skills_norm))
            skill_match_ratio = len(matching_skills) / len(query_skills_norm)
            
            # Boost based on match ratio
            return min(0.3, skill_match_ratio * 0.3)  # Max boost of 0.3
            
        except Exception:
            return 0.0
    
    def _apply_filters(
        self, 
        documents: List[RetrievedDocument], 
        filters: List[SearchFilter]
    ) -> List[RetrievedDocument]:
        """Apply search filters to documents"""
        try:
            filtered_docs = documents.copy()
            
            for filter_obj in filters:
                if filter_obj.filter_type == FilterType.SKILL_LEVEL:
                    filtered_docs = self._filter_by_skill_level(filtered_docs, filter_obj)
                elif filter_obj.filter_type == FilterType.EXPERIENCE_YEARS:
                    filtered_docs = self._filter_by_experience(filtered_docs, filter_obj)
                elif filter_obj.filter_type == FilterType.DOCUMENT_TYPE:
                    filtered_docs = self._filter_by_document_type(filtered_docs, filter_obj)
                elif filter_obj.filter_type == FilterType.DATE_RANGE:
                    filtered_docs = self._filter_by_date_range(filtered_docs, filter_obj)
                elif filter_obj.filter_type == FilterType.TAGS:
                    filtered_docs = self._filter_by_tags(filtered_docs, filter_obj)
                elif filter_obj.filter_type == FilterType.SCORE_THRESHOLD:
                    filtered_docs = self._filter_by_score(filtered_docs, filter_obj)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return documents
    
    def _filter_by_skill_level(
        self, 
        documents: List[RetrievedDocument], 
        filter_obj: SearchFilter
    ) -> List[RetrievedDocument]:
        """Filter by skill level"""
        if not filter_obj.values:
            return documents
        
        target_levels = [level for level in filter_obj.values if isinstance(level, str)]
        
        filtered = []
        for doc in documents:
            # Check if document has required skill levels
            doc_metadata = doc.metadata
            if "skill_levels" in doc_metadata:
                doc_levels = doc_metadata["skill_levels"]
                if any(level in doc_levels for level in target_levels):
                    filtered.append(doc)
        
        return filtered
    
    def _filter_by_experience(
        self, 
        documents: List[RetrievedDocument], 
        filter_obj: SearchFilter
    ) -> List[RetrievedDocument]:
        """Filter by experience years"""
        if not filter_obj.values or len(filter_obj.values) < 2:
            return documents
        
        min_exp = filter_obj.values[0]
        max_exp = filter_obj.values[1]
        
        filtered = []
        for doc in documents:
            exp_years = doc.metadata.get("experience_years")
            if exp_years and min_exp <= exp_years <= max_exp:
                filtered.append(doc)
        
        return filtered
    
    def _filter_by_document_type(
        self, 
        documents: List[RetrievedDocument], 
        filter_obj: SearchFilter
    ) -> List[RetrievedDocument]:
        """Filter by document type"""
        if not filter_obj.values:
            return documents
        
        target_types = filter_obj.values
        
        return [
            doc for doc in documents 
            if doc.metadata.get("doc_type") in target_types
        ]
    
    def _filter_by_date_range(
        self, 
        documents: List[RetrievedDocument], 
        filter_obj: SearchFilter
    ) -> List[RetrievedDocument]:
        """Filter by date range"""
        if not filter_obj.values or len(filter_obj.values) < 2:
            return documents
        
        start_date = datetime.fromisoformat(filter_obj.values[0])
        end_date = datetime.fromisoformat(filter_obj.values[1])
        
        filtered = []
        for doc in documents:
            doc_date_str = doc.metadata.get("created_at")
            if doc_date_str:
                try:
                    doc_date = datetime.fromisoformat(doc_date_str)
                    if start_date <= doc_date <= end_date:
                        filtered.append(doc)
                except ValueError:
                    continue
        
        return filtered
    
    def _filter_by_tags(
        self, 
        documents: List[RetrievedDocument], 
        filter_obj: SearchFilter
    ) -> List[RetrievedDocument]:
        """Filter by tags"""
        if not filter_obj.values:
            return documents
        
        target_tags = set(filter_obj.values)
        
        filtered = []
        for doc in documents:
            doc_tags = set(doc.metadata.get("tags", []))
            if target_tags.intersection(doc_tags):
                filtered.append(doc)
        
        return filtered
    
    def _filter_by_score(
        self, 
        documents: List[RetrievedDocument], 
        filter_obj: SearchFilter
    ) -> List[RetrievedDocument]:
        """Filter by score threshold"""
        if not filter_obj.values:
            return documents
        
        threshold = filter_obj.values[0]
        return [doc for doc in documents if doc.score >= threshold]
    
    def _rerank_documents(
        self, 
        documents: List[RetrievedDocument], 
        context: RetrievalContext,
        reranking_config: RerankingConfig
    ) -> List[RetrievedDocument]:
        """Rerank documents using advanced scoring"""
        try:
            for doc in documents:
                # Base score
                new_score = doc.score * reranking_config.base_weight
                
                # Skill relevance boost
                if context.extracted_skills:
                    skill_relevance = self._calculate_skill_relevance(doc, context.extracted_skills)
                    new_score += skill_relevance * reranking_config.skill_weight
                
                # Freshness boost
                if reranking_config.freshness_weight > 0:
                    freshness_score = self._calculate_freshness_score(doc)
                    new_score += freshness_score * reranking_config.freshness_weight
                
                # Quality boost
                if reranking_config.quality_weight > 0:
                    quality_score = self._calculate_quality_score(doc)
                    new_score += quality_score * reranking_config.quality_weight
                
                # Diversity penalty
                if reranking_config.diversity_weight > 0:
                    diversity_penalty = self._calculate_diversity_penalty(doc, documents)
                    new_score -= diversity_penalty * reranking_config.diversity_weight
                
                doc.score = min(1.0, max(0.0, new_score))
            
            # Sort by new scores
            documents.sort(key=lambda x: x.score, reverse=True)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error reranking documents: {e}")
            return documents
    
    def _calculate_skill_relevance(self, document: RetrievedDocument, query_skills: List[str]) -> float:
        """Calculate skill relevance score"""
        doc_skills = document.metadata.get("skill_ids", [])
        if not doc_skills or not query_skills:
            return 0.0
        
        doc_skills_norm = [skill.lower() for skill in doc_skills]
        query_skills_norm = [skill.lower() for skill in query_skills]
        
        intersection = set(doc_skills_norm).intersection(set(query_skills_norm))
        union = set(doc_skills_norm).union(set(query_skills_norm))
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_freshness_score(self, document: RetrievedDocument) -> float:
        """Calculate freshness score based on document age"""
        try:
            created_at_str = document.metadata.get("created_at")
            if not created_at_str:
                return 0.0
            
            created_at = datetime.fromisoformat(created_at_str)
            age_days = (datetime.now() - created_at).days
            
            # Exponential decay: newer documents get higher scores
            return max(0.0, 1.0 - (age_days / 365.0))  # Decay over 1 year
            
        except Exception:
            return 0.0
    
    def _calculate_quality_score(self, document: RetrievedDocument) -> float:
        """Calculate quality score based on metadata"""
        try:
            confidence = document.metadata.get("confidence_score", 0.5)
            employee_count = document.metadata.get("employee_count", 0)
            
            # Combine confidence and popularity
            quality = 0.7 * confidence + 0.3 * min(1.0, employee_count / 10.0)
            
            return quality
            
        except Exception:
            return 0.5
    
    def _calculate_diversity_penalty(
        self, 
        document: RetrievedDocument, 
        all_documents: List[RetrievedDocument]
    ) -> float:
        """Calculate diversity penalty to promote result diversity"""
        try:
            doc_type = document.metadata.get("doc_type")
            similar_count = sum(
                1 for doc in all_documents 
                if doc.metadata.get("doc_type") == doc_type and doc.document_id != document.document_id
            )
            
            # Penalty increases with similar documents
            return min(0.3, similar_count * 0.05)
            
        except Exception:
            return 0.0
    
    def _generate_query_suggestions(self, context: RetrievalContext) -> List[str]:
        """Generate query suggestions based on context"""
        suggestions = []
        
        # Skill-based suggestions
        for skill in context.extracted_skills[:3]:
            suggestions.append(f"Find experts in {skill}")
            suggestions.append(f"Projects using {skill}")
        
        # Query type suggestions
        if context.query_type == QueryType.SKILL_SEARCH:
            suggestions.extend([
                "Find similar skills",
                "Show skill requirements",
                "Related technologies"
            ])
        elif context.query_type == QueryType.CANDIDATE_SEARCH:
            suggestions.extend([
                "Find similar candidates",
                "Show candidate experience",
                "Related positions"
            ])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def _calculate_facets(self, documents: List[RetrievedDocument]) -> Dict[str, Dict[str, int]]:
        """Calculate facets for result filtering"""
        facets = {
            "document_types": {},
            "skills": {},
            "experience_levels": {},
            "tags": {}
        }
        
        for doc in documents:
            metadata = doc.metadata
            
            # Document types
            doc_type = metadata.get("doc_type", "unknown")
            facets["document_types"][doc_type] = facets["document_types"].get(doc_type, 0) + 1
            
            # Skills
            skills = metadata.get("skill_ids", [])
            for skill in skills[:5]:  # Limit to top 5 skills
                facets["skills"][skill] = facets["skills"].get(skill, 0) + 1
            
            # Experience levels
            exp_years = metadata.get("experience_years")
            if exp_years:
                exp_level = "Senior" if exp_years >= 5 else "Mid" if exp_years >= 2 else "Junior"
                facets["experience_levels"][exp_level] = facets["experience_levels"].get(exp_level, 0) + 1
            
            # Tags
            tags = metadata.get("tags", [])
            for tag in tags[:3]:  # Limit to top 3 tags
                facets["tags"][tag] = facets["tags"].get(tag, 0) + 1
        
        return facets
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "cache_size": len(self.retrieval_cache),
            "synonym_count": len(self.skill_synonyms),
            "vector_store_stats": self.vector_store.get_collection_stats()
        }
