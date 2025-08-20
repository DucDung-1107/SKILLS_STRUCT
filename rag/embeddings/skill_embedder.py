#!/usr/bin/env python3
"""
🧠 SkillStruct Embedder
Advanced embedding system cho skill taxonomy và resume data
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import json
import hashlib
from datetime import datetime
import re

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    logging.warning("Google AI not available")

from ..schemas import (
    RAGDocument, DocumentMetadata, DocumentType, 
    SkillEmbedding, ResumeEmbedding, EmbeddingConfig
)
from utils.core_utils import clean_text, extract_skills_from_text
from database.core_schema import SkillNodeResponse

logger = logging.getLogger(__name__)

class SkillStructEmbedder:
    """
    Advanced embedder cho SkillStruct RAG system
    Tối ưu hóa cho skill taxonomy và resume processing
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedding_model = self._initialize_embeddings()
        self.skill_context_cache = {}
        
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            if self.config.model_name.value == "models/embedding-001" and GOOGLE_AI_AVAILABLE:
                return GoogleGenerativeAIEmbeddings(
                    model=self.config.model_name.value,
                    google_api_key=self.config.api_key,
                    task_type="RETRIEVAL_DOCUMENT"
                )
            else:
                raise ValueError(f"Unsupported embedding model: {self.config.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def create_skill_context(
        self, 
        skill_node: SkillNodeResponse, 
        taxonomy_data: Dict[str, Any],
        include_relationships: bool = True
    ) -> str:
        """
        Tạo rich context cho skill node để embedding
        
        Args:
            skill_node: Skill node data
            taxonomy_data: Full taxonomy data for relationships
            include_relationships: Whether to include parent/child relationships
            
        Returns:
            Rich context string
        """
        try:
            context_parts = []
            
            # Basic skill information
            context_parts.append(f"Skill: {skill_node.name}")
            context_parts.append(f"Type: {skill_node.node_type.value}")
            context_parts.append(f"Level: {skill_node.level}")
            
            if skill_node.description:
                context_parts.append(f"Description: {skill_node.description}")
            
            # Employee information
            if hasattr(skill_node, 'employees') and skill_node.employees:
                context_parts.append(f"Employees with this skill: {', '.join(skill_node.employees)}")
                context_parts.append(f"Employee count: {len(skill_node.employees)}")
            
            # Relationship context
            if include_relationships and taxonomy_data:
                relationships = self._extract_skill_relationships(skill_node.id, taxonomy_data)
                
                if relationships["parents"]:
                    parent_names = [self._get_skill_name(pid, taxonomy_data) for pid in relationships["parents"]]
                    context_parts.append(f"Parent skills: {', '.join(filter(None, parent_names))}")
                
                if relationships["children"]:
                    child_names = [self._get_skill_name(cid, taxonomy_data) for cid in relationships["children"]]
                    context_parts.append(f"Sub-skills: {', '.join(filter(None, child_names))}")
                
                if relationships["siblings"]:
                    sibling_names = [self._get_skill_name(sid, taxonomy_data) for sid in relationships["siblings"]]
                    context_parts.append(f"Related skills: {', '.join(filter(None, sibling_names))}")
            
            # Proficiency context
            if hasattr(skill_node, 'proficiency_stats') and skill_node.proficiency_stats:
                stats = skill_node.proficiency_stats
                if stats:
                    context_parts.append(f"Proficiency statistics: {json.dumps(stats)}")
            
            # Industry/domain context
            context_parts.append(f"Skill domain: {self._infer_skill_domain(skill_node.name)}")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error creating skill context for {skill_node.id}: {e}")
            return f"Skill: {skill_node.name} | Type: {skill_node.node_type.value}"
    
    def _extract_skill_relationships(self, skill_id: str, taxonomy_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract parent, child, and sibling relationships"""
        relationships = {"parents": [], "children": [], "siblings": []}
        
        try:
            edges = taxonomy_data.get("edges", [])
            
            for edge in edges:
                if isinstance(edge, dict):
                    source = edge.get("source") or edge.get("from") or edge.get("parent")
                    target = edge.get("target") or edge.get("to") or edge.get("child")
                    
                    if target == skill_id:
                        relationships["parents"].append(source)
                    elif source == skill_id:
                        relationships["children"].append(target)
            
            # Find siblings (nodes with same parent)
            for parent in relationships["parents"]:
                for edge in edges:
                    if isinstance(edge, dict):
                        source = edge.get("source") or edge.get("from") or edge.get("parent")
                        target = edge.get("target") or edge.get("to") or edge.get("child")
                        
                        if source == parent and target != skill_id and target not in relationships["siblings"]:
                            relationships["siblings"].append(target)
            
        except Exception as e:
            logger.error(f"Error extracting relationships for {skill_id}: {e}")
        
        return relationships
    
    def _get_skill_name(self, skill_id: str, taxonomy_data: Dict[str, Any]) -> Optional[str]:
        """Get skill name by ID"""
        try:
            nodes = taxonomy_data.get("nodes", [])
            for node in nodes:
                if node.get("id") == skill_id:
                    return node.get("name")
        except Exception:
            pass
        return None
    
    def _infer_skill_domain(self, skill_name: str) -> str:
        """Infer skill domain from name"""
        skill_lower = skill_name.lower()
        
        domains = {
            "programming": ["python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "rust", "swift"],
            "web_development": ["html", "css", "react", "angular", "vue", "nodejs", "express", "django", "flask"],
            "data_science": ["sql", "nosql", "machine learning", "deep learning", "data analysis", "pandas", "numpy"],
            "cloud_computing": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform", "cloudformation"],
            "devops": ["jenkins", "ci/cd", "git", "docker", "kubernetes", "monitoring", "deployment"],
            "mobile": ["android", "ios", "react native", "flutter", "swift", "kotlin"],
            "database": ["mysql", "postgresql", "mongodb", "redis", "elasticsearch", "oracle"],
            "frontend": ["react", "angular", "vue", "html", "css", "javascript", "typescript"],
            "backend": ["api", "microservices", "server", "database", "authentication"],
            "testing": ["unit testing", "integration testing", "selenium", "cypress", "jest"],
            "project_management": ["agile", "scrum", "kanban", "project management", "communication"],
            "security": ["cybersecurity", "encryption", "authentication", "authorization", "secure coding"]
        }
        
        for domain, keywords in domains.items():
            if any(keyword in skill_lower for keyword in keywords):
                return domain
        
        return "general"
    
    def create_resume_context(self, resume_data: Dict[str, Any]) -> str:
        """
        Tạo context cho resume để embedding
        
        Args:
            resume_data: Resume data from processing
            
        Returns:
            Rich context string
        """
        try:
            context_parts = []
            
            # Basic info
            if "file_name" in resume_data:
                context_parts.append(f"Resume: {resume_data['file_name']}")
            
            # Extracted data
            extracted = resume_data.get("extracted_data", {})
            
            # Personal info
            personal = extracted.get("personal_info", {})
            if personal.get("name"):
                context_parts.append(f"Candidate: {personal['name']}")
            
            # Experience
            if extracted.get("experience_years"):
                context_parts.append(f"Experience: {extracted['experience_years']} years")
            
            # Skills
            skills = extracted.get("skills", [])
            if skills:
                context_parts.append(f"Skills: {', '.join(skills)}")
            
            # Technologies
            technologies = extracted.get("technologies", [])
            if technologies:
                context_parts.append(f"Technologies: {', '.join(technologies)}")
            
            # Education
            if extracted.get("highest_education"):
                context_parts.append(f"Education: {extracted['highest_education']}")
            
            # Work history
            work_history = extracted.get("work_history", [])
            if work_history:
                positions = [work.get("position", "") for work in work_history if work.get("position")]
                if positions:
                    context_parts.append(f"Positions: {', '.join(positions)}")
            
            # Current position
            if extracted.get("current_position"):
                context_parts.append(f"Current position: {extracted['current_position']}")
            
            # Location
            if extracted.get("location"):
                context_parts.append(f"Location: {extracted['location']}")
            
            # Certifications
            certifications = extracted.get("certifications", [])
            if certifications:
                context_parts.append(f"Certifications: {', '.join(certifications)}")
            
            # Languages
            languages = extracted.get("languages", [])
            if languages:
                context_parts.append(f"Languages: {', '.join(languages)}")
            
            # AI analysis
            ai_analysis = resume_data.get("ai_analysis", {})
            if ai_analysis.get("overall_score"):
                context_parts.append(f"AI Score: {ai_analysis['overall_score']}/100")
            
            return " | ".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error creating resume context: {e}")
            return "Resume document"
    
    def embed_skill_taxonomy(self, taxonomy_data: Dict[str, Any]) -> List[RAGDocument]:
        """
        Embed entire skill taxonomy
        
        Args:
            taxonomy_data: Full skill taxonomy data
            
        Returns:
            List of RAGDocument objects with embeddings
        """
        try:
            documents = []
            nodes = taxonomy_data.get("nodes", [])
            
            logger.info(f"Embedding {len(nodes)} skill nodes")
            
            # Process nodes in batches
            batch_size = self.config.batch_size
            for i in range(0, len(nodes), batch_size):
                batch_nodes = nodes[i:i + batch_size]
                
                # Prepare contexts and documents
                contexts = []
                batch_docs = []
                
                for node in batch_nodes:
                    try:
                        # Convert to SkillNodeResponse for consistency
                        skill_node = SkillNodeResponse(**node)
                        
                        # Create context
                        context = self.create_skill_context(skill_node, taxonomy_data)
                        contexts.append(context)
                        
                        # Create metadata
                        metadata = DocumentMetadata(
                            doc_id=node["id"],
                            doc_type=DocumentType.SKILL_NODE,
                            source="skill_taxonomy",
                            created_at=datetime.now(),
                            skill_ids=[node["id"]],
                            employee_ids=node.get("employees", []),
                            tags=[node.get("type", ""), f"level_{node.get('level', 0)}"],
                            confidence_score=1.0,
                            custom_fields={
                                "skill_type": node.get("type"),
                                "skill_level": node.get("level"),
                                "color": node.get("color"),
                                "employee_count": node.get("employee_count", 0)
                            }
                        )
                        
                        # Create document
                        doc = RAGDocument(
                            id=node["id"],
                            content=context,
                            metadata=metadata
                        )
                        batch_docs.append(doc)
                        
                    except Exception as e:
                        logger.error(f"Error processing skill node {node.get('id', 'unknown')}: {e}")
                        continue
                
                # Generate embeddings for batch
                if contexts:
                    try:
                        embeddings = self.embedding_model.embed_documents(contexts)
                        
                        # Assign embeddings to documents
                        for doc, embedding in zip(batch_docs, embeddings):
                            doc.embedding = embedding
                            documents.append(doc)
                            
                        logger.info(f"Embedded batch {i//batch_size + 1}/{(len(nodes)-1)//batch_size + 1}")
                        
                    except Exception as e:
                        logger.error(f"Error embedding batch {i//batch_size + 1}: {e}")
                        continue
            
            logger.info(f"Successfully embedded {len(documents)} skill nodes")
            return documents
            
        except Exception as e:
            logger.error(f"Error embedding skill taxonomy: {e}")
            return []
    
    def embed_resume(self, resume_data: Dict[str, Any]) -> List[RAGDocument]:
        """
        Embed resume data with chunking for large documents
        
        Args:
            resume_data: Resume data from processing
            
        Returns:
            List of RAGDocument objects with embeddings
        """
        try:
            documents = []
            
            # Create main resume document
            main_context = self.create_resume_context(resume_data)
            
            # Extract skills for metadata
            extracted_data = resume_data.get("extracted_data", {})
            skills = extracted_data.get("skills", [])
            
            metadata = DocumentMetadata(
                doc_id=resume_data["id"],
                doc_type=DocumentType.RESUME,
                source=resume_data.get("file_name", "unknown"),
                created_at=datetime.fromisoformat(resume_data.get("upload_date", datetime.now().isoformat())),
                skill_ids=skills,
                employee_ids=[extracted_data.get("personal_info", {}).get("name", "")],
                tags=["resume", f"experience_{extracted_data.get('experience_years', 0)}"],
                confidence_score=resume_data.get("ai_analysis", {}).get("overall_score", 0) / 100.0,
                custom_fields={
                    "file_name": resume_data.get("file_name"),
                    "candidate_name": extracted_data.get("personal_info", {}).get("name"),
                    "experience_years": extracted_data.get("experience_years"),
                    "education_level": extracted_data.get("highest_education"),
                    "current_position": extracted_data.get("current_position")
                }
            )
            
            # Create main document
            main_doc = RAGDocument(
                id=resume_data["id"],
                content=main_context,
                metadata=metadata
            )
            
            # Chunk large content if needed
            full_text = resume_data.get("extracted_text", "")
            if len(full_text) > self.config.max_tokens * 4:  # Rough token estimation
                chunks = self._chunk_text(full_text, self.config.max_tokens * 4)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.model_copy()
                    chunk_metadata.doc_id = f"{resume_data['id']}_chunk_{i}"
                    chunk_metadata.custom_fields["chunk_index"] = i
                    chunk_metadata.custom_fields["total_chunks"] = len(chunks)
                    
                    chunk_doc = RAGDocument(
                        id=f"{resume_data['id']}_chunk_{i}",
                        content=chunk,
                        metadata=chunk_metadata,
                        chunk_index=i,
                        chunk_count=len(chunks)
                    )
                    documents.append(chunk_doc)
            else:
                documents.append(main_doc)
            
            # Generate embeddings
            contexts = [doc.content for doc in documents]
            embeddings = self.embedding_model.embed_documents(contexts)
            
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
            
            logger.info(f"Embedded resume {resume_data['id']} into {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error embedding resume {resume_data.get('id', 'unknown')}: {e}")
            return []
    
    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        """
        Chunk text into smaller pieces
        
        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split("\n\n")
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chars:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # If paragraph itself is too long, split by sentences
                if len(paragraph) > max_chars:
                    sentences = re.split(r'[.!?]+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= max_chars:
                            temp_chunk += sentence + ". "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + ". "
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                else:
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query for search
        
        Args:
            query: Query string
            
        Returns:
            Query embedding vector
        """
        try:
            # Clean and enhance query
            cleaned_query = clean_text(query)
            
            # Extract potential skills from query
            potential_skills = extract_skills_from_text(cleaned_query)
            if potential_skills:
                enhanced_query = f"{cleaned_query} | Skills mentioned: {', '.join(potential_skills)}"
            else:
                enhanced_query = cleaned_query
            
            # Generate embedding
            embedding = self.embedding_model.embed_query(enhanced_query)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return []
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get embedding statistics
        
        Returns:
            Dict with embedding stats
        """
        return {
            "model_name": self.config.model_name.value,
            "vector_size": self.config.vector_size,
            "batch_size": self.config.batch_size,
            "max_tokens": self.config.max_tokens,
            "cache_size": len(self.skill_context_cache)
        }
