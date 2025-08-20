#!/usr/bin/env python3
"""
🔧 Utility Functions - Skill Struct Platform
Tập hợp các hàm tiện ích cho toàn bộ hệ thống
"""

from .auth_utils import *
from .data_utils import *
from .file_utils import *
from .graph_utils import *
from .validation_utils import *
from .ai_utils import *
from .db_utils import *
from .api_utils import *
from .export_utils import *

__version__ = "1.0.0"
__author__ = "Skill Struct Team"

# Export all utility functions
__all__ = [
    # Auth utilities
    "hash_password", "verify_password", "generate_session_token", 
    "create_jwt_token", "verify_jwt_token", "check_permissions",
    
    # Data utilities
    "clean_text", "extract_skills", "normalize_skill_name", 
    "calculate_similarity", "merge_duplicates",
    
    # File utilities
    "read_json_file", "write_json_file", "read_csv_file", 
    "extract_pdf_text", "extract_doc_text", "validate_file_type",
    
    # Graph utilities
    "find_node_by_id", "get_node_level", "calculate_graph_metrics", 
    "export_to_mermaid", "validate_graph_structure",
    
    # Validation utilities
    "validate_email", "validate_phone", "validate_hex_color", 
    "validate_skill_name", "sanitize_input",
    
    # AI utilities
    "call_gemini_api", "extract_features_with_ai", "generate_recommendations", 
    "analyze_skill_gaps", "cluster_skills",
    
    # Database utilities
    "init_database", "execute_query", "bulk_insert", 
    "backup_database", "migrate_schema",
    
    # API utilities
    "create_response", "handle_error", "log_api_call", 
    "rate_limit_check", "validate_api_key",
    
    # Export utilities
    "export_to_json", "export_to_csv", "export_to_excel", 
    "export_to_mermaid", "import_from_json"
]
