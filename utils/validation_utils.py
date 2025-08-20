#!/usr/bin/env python3
"""
✅ Validation Utilities
Các hàm tiện ích cho validation và sanitization
"""

import re
import html
from typing import Dict, List, Any, Optional, Union
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def validate_email(email: str) -> bool:
    """
    Validate email format
    """
    if not email or not isinstance(email, str):
        return False
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email.strip()))

def validate_phone(phone: str) -> bool:
    """
    Validate phone number format (supports Vietnamese and international)
    """
    if not phone or not isinstance(phone, str):
        return False
    
    # Remove spaces, hyphens, and parentheses
    cleaned_phone = re.sub(r'[-\s\(\)]', '', phone.strip())
    
    # Patterns for Vietnamese and international numbers
    patterns = [
        r'^\+84[1-9]\d{8,9}$',  # +84 format
        r'^0[1-9]\d{8,9}$',     # 0xx format (Vietnamese)
        r'^\+[1-9]\d{7,14}$',   # International format
        r'^[1-9]\d{6,14}$'      # General format
    ]
    
    for pattern in patterns:
        if re.match(pattern, cleaned_phone):
            return True
    
    return False

def validate_hex_color(color: str) -> bool:
    """
    Validate hex color format
    """
    if not color or not isinstance(color, str):
        return False
    
    pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$'
    return bool(re.match(pattern, color.strip()))

def validate_url(url: str) -> bool:
    """
    Validate URL format
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        result = urlparse(url.strip())
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def validate_skill_name(skill_name: str) -> bool:
    """
    Validate skill name
    """
    if not skill_name or not isinstance(skill_name, str):
        return False
    
    # Skill name should be 1-100 characters, allow letters, numbers, spaces, +, -, ., #
    pattern = r'^[a-zA-Z0-9\s\+\-\.#\u00C0-\u1EF9]{1,100}$'
    return bool(re.match(pattern, skill_name.strip()))

def validate_node_type(node_type: str) -> bool:
    """
    Validate node type
    """
    valid_types = ['root', 'skill_group', 'skill', 'sub_skill']
    return node_type in valid_types

def validate_proficiency_level(level: str) -> bool:
    """
    Validate proficiency level
    """
    valid_levels = ['beginner', 'intermediate', 'advanced', 'expert']
    return level.lower() in valid_levels

def validate_experience_years(years: Union[int, str]) -> bool:
    """
    Validate years of experience
    """
    try:
        years_int = int(years)
        return 0 <= years_int <= 50
    except (ValueError, TypeError):
        return False

def sanitize_input(input_str: str, max_length: int = 255) -> str:
    """
    Sanitize input string
    """
    if not isinstance(input_str, str):
        return ""
    
    # Remove HTML tags and entities
    sanitized = html.escape(input_str.strip())
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', sanitized)
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename
    """
    if not isinstance(filename, str):
        return "file"
    
    # Remove/replace invalid filename characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename.strip())
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "file"
    
    return sanitized

def validate_json_structure(data: Any, required_fields: List[str]) -> Dict[str, Any]:
    """
    Validate JSON structure has required fields
    """
    result = {
        "valid": True,
        "missing_fields": [],
        "invalid_fields": [],
        "errors": []
    }
    
    if not isinstance(data, dict):
        result["valid"] = False
        result["errors"].append("Data must be a dictionary")
        return result
    
    for field in required_fields:
        if field not in data:
            result["missing_fields"].append(field)
            result["valid"] = False
    
    return result

def validate_taxonomy_structure(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate skill taxonomy structure
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Check required top-level fields
    required_fields = ['metadata', 'nodes', 'edges']
    for field in required_fields:
        if field not in taxonomy:
            result["errors"].append(f"Missing required field: {field}")
            result["valid"] = False
    
    if not result["valid"]:
        return result
    
    # Validate nodes
    nodes = taxonomy.get('nodes', [])
    if not isinstance(nodes, list):
        result["errors"].append("Nodes must be a list")
        result["valid"] = False
    else:
        node_ids = set()
        for i, node in enumerate(nodes):
            node_validation = validate_node_structure(node, i)
            if not node_validation["valid"]:
                result["errors"].extend(node_validation["errors"])
                result["valid"] = False
            
            # Check for duplicate IDs
            node_id = node.get('id')
            if node_id in node_ids:
                result["errors"].append(f"Duplicate node ID: {node_id}")
                result["valid"] = False
            else:
                node_ids.add(node_id)
    
    # Validate edges
    edges = taxonomy.get('edges', [])
    if not isinstance(edges, list):
        result["errors"].append("Edges must be a list")
        result["valid"] = False
    else:
        for i, edge in enumerate(edges):
            edge_validation = validate_edge_structure(edge, i, node_ids)
            if not edge_validation["valid"]:
                result["errors"].extend(edge_validation["errors"])
                result["valid"] = False
    
    return result

def validate_node_structure(node: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Validate individual node structure
    """
    result = {
        "valid": True,
        "errors": []
    }
    
    if not isinstance(node, dict):
        result["errors"].append(f"Node {index} must be a dictionary")
        result["valid"] = False
        return result
    
    # Required fields
    required_fields = ['id', 'name', 'type', 'level', 'color']
    for field in required_fields:
        if field not in node:
            result["errors"].append(f"Node {index} missing required field: {field}")
            result["valid"] = False
    
    # Validate field types and values
    if 'id' in node and not isinstance(node['id'], str):
        result["errors"].append(f"Node {index} ID must be a string")
        result["valid"] = False
    
    if 'name' in node and not validate_skill_name(node['name']):
        result["errors"].append(f"Node {index} has invalid name")
        result["valid"] = False
    
    if 'type' in node and not validate_node_type(node['type']):
        result["errors"].append(f"Node {index} has invalid type")
        result["valid"] = False
    
    if 'level' in node and not isinstance(node['level'], int):
        result["errors"].append(f"Node {index} level must be an integer")
        result["valid"] = False
    
    if 'color' in node and not validate_hex_color(node['color']):
        result["errors"].append(f"Node {index} has invalid color format")
        result["valid"] = False
    
    return result

def validate_edge_structure(edge: Dict[str, Any], index: int = 0, 
                          valid_node_ids: set = None) -> Dict[str, Any]:
    """
    Validate individual edge structure
    """
    result = {
        "valid": True,
        "errors": []
    }
    
    if not isinstance(edge, dict):
        result["errors"].append(f"Edge {index} must be a dictionary")
        result["valid"] = False
        return result
    
    # Required fields
    required_fields = ['source', 'target', 'type']
    for field in required_fields:
        if field not in edge:
            result["errors"].append(f"Edge {index} missing required field: {field}")
            result["valid"] = False
    
    # Validate field values
    if 'source' in edge and 'target' in edge:
        source = edge['source']
        target = edge['target']
        
        if source == target:
            result["errors"].append(f"Edge {index} has self-reference")
            result["valid"] = False
        
        if valid_node_ids:
            if source not in valid_node_ids:
                result["errors"].append(f"Edge {index} references invalid source node: {source}")
                result["valid"] = False
            
            if target not in valid_node_ids:
                result["errors"].append(f"Edge {index} references invalid target node: {target}")
                result["valid"] = False
    
    return result

def validate_file_upload(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate uploaded file data
    """
    result = {
        "valid": True,
        "errors": []
    }
    
    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if 'size' in file_data and file_data['size'] > max_size:
        result["errors"].append("File size exceeds 10MB limit")
        result["valid"] = False
    
    # Check file type
    allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
    if 'filename' in file_data:
        filename = file_data['filename'].lower()
        if not any(filename.endswith(ext) for ext in allowed_extensions):
            result["errors"].append("File type not allowed")
            result["valid"] = False
    
    return result

def validate_search_query(query: str) -> bool:
    """
    Validate search query
    """
    if not query or not isinstance(query, str):
        return False
    
    # Query should be 1-100 characters
    query = query.strip()
    if not (1 <= len(query) <= 100):
        return False
    
    # Should not contain only special characters
    if re.match(r'^[^a-zA-Z0-9\u00C0-\u1EF9]+$', query):
        return False
    
    return True

def validate_pagination(page: int, page_size: int) -> Dict[str, Any]:
    """
    Validate pagination parameters
    """
    result = {
        "valid": True,
        "errors": []
    }
    
    if not isinstance(page, int) or page < 1:
        result["errors"].append("Page must be a positive integer")
        result["valid"] = False
    
    if not isinstance(page_size, int) or not (1 <= page_size <= 100):
        result["errors"].append("Page size must be between 1 and 100")
        result["valid"] = False
    
    return result

def validate_date_range(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Validate date range
    """
    result = {
        "valid": True,
        "errors": []
    }
    
    try:
        from datetime import datetime
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start >= end:
            result["errors"].append("Start date must be before end date")
            result["valid"] = False
            
    except (ValueError, AttributeError):
        result["errors"].append("Invalid date format")
        result["valid"] = False
    
    return result

def validate_proficiency_score(score: Union[int, float]) -> bool:
    """
    Validate proficiency score (1-10)
    """
    try:
        score_num = float(score)
        return 1 <= score_num <= 10
    except (ValueError, TypeError):
        return False

def clean_and_validate_skill_list(skills: List[str]) -> List[str]:
    """
    Clean and validate list of skills
    """
    if not isinstance(skills, list):
        return []
    
    cleaned_skills = []
    for skill in skills:
        if isinstance(skill, str) and validate_skill_name(skill):
            cleaned_skill = sanitize_input(skill, 100).strip()
            if cleaned_skill and cleaned_skill not in cleaned_skills:
                cleaned_skills.append(cleaned_skill)
    
    return cleaned_skills[:50]  # Limit to 50 skills

def validate_api_request(request_data: Dict[str, Any], 
                        schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate API request against schema
    """
    result = {
        "valid": True,
        "errors": [],
        "sanitized_data": {}
    }
    
    for field, field_schema in schema.items():
        value = request_data.get(field)
        
        # Check required fields
        if field_schema.get('required', False) and value is None:
            result["errors"].append(f"Missing required field: {field}")
            result["valid"] = False
            continue
        
        # Skip validation if field is optional and not provided
        if value is None:
            continue
        
        # Type validation
        expected_type = field_schema.get('type')
        if expected_type and not isinstance(value, expected_type):
            result["errors"].append(f"Field {field} must be of type {expected_type.__name__}")
            result["valid"] = False
            continue
        
        # Custom validation
        validator = field_schema.get('validator')
        if validator and not validator(value):
            result["errors"].append(f"Field {field} failed validation")
            result["valid"] = False
            continue
        
        # Sanitization
        sanitizer = field_schema.get('sanitizer')
        if sanitizer:
            value = sanitizer(value)
        
        result["sanitized_data"][field] = value
    
    return result
