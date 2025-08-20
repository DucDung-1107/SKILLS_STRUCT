#!/usr/bin/env python3
"""
🔧 Core Utils - Essential utilities chỉ những gì cần thiết
"""

import json
import hashlib
import secrets
import re
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ===========================
#  PASSWORD & AUTH UTILITIES
# ===========================

def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{pwd_hash.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    try:
        salt, pwd_hash = hashed.split(':')
        new_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return new_hash.hex() == pwd_hash
    except:
        return False

def generate_token() -> str:
    """Generate secure token"""
    return secrets.token_urlsafe(32)

# ===========================
#  FILE UTILITIES
# ===========================

def extract_text_from_file(file_path: str) -> str:
    """Extract text from common file formats"""
    try:
        if file_path.lower().endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            return extract_text_from_docx(file_path)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF"""
    try:
        import PyMuPDF as fitz
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        logger.warning("PyMuPDF not installed")
        return ""
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except ImportError:
        logger.warning("python-docx not installed")
        return ""
    except Exception as e:
        logger.error(f"Error reading DOCX: {e}")
        return ""

# ===========================
#  ENVIRONMENT UTILITIES
# ===========================

def get_environment_variable(key: str, default: str = "") -> str:
    """Get environment variable with default"""
    import os
    return os.getenv(key, default)

# ===========================
#  VALIDATION UTILITIES
# ===========================

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_input(text: str) -> str:
    """Basic input sanitization"""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove SQL injection patterns
    text = re.sub(r'(union|select|insert|update|delete|drop|create|alter)', '', text, flags=re.IGNORECASE)
    return text.strip()

def validate_file_type(filename: str, allowed_types: List[str] = None) -> bool:
    """Validate file type"""
    if allowed_types is None:
        allowed_types = ['.pdf', '.doc', '.docx', '.txt']
    
    return any(filename.lower().endswith(ext) for ext in allowed_types)

# ===========================
#  DATA UTILITIES
# ===========================

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s\-.,;:()!?]', '', text)
    return text.strip()

def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\-.,;:()!?]', '', text)
    
    return text

def extract_skills_from_text(text: str) -> List[str]:
    """Extract skills from text using basic patterns"""
    skills = []
    text_lower = text.lower()
    
    # Common programming languages
    prog_langs = ['python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 'swift']
    for lang in prog_langs:
        if lang in text_lower:
            skills.append(lang.title())
    
    # Common frameworks
    frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'laravel', 'express']
    for framework in frameworks:
        if framework in text_lower:
            skills.append(framework.title())
    
    # Common tools
    tools = ['git', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'terraform']
    for tool in tools:
        if tool in text_lower:
            skills.append(tool.upper() if tool in ['aws', 'gcp'] else tool.title())
    
    return list(set(skills))  # Remove duplicates

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate basic similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0

# ===========================
#  API UTILITIES
# ===========================

def create_response(success: bool, message: str, data: Any = None) -> Dict[str, Any]:
    """Create standardized API response"""
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    return response

def create_error_response(error_message: str, status_code: int = 400) -> Dict[str, Any]:
    """Create error response"""
    return {
        "success": False,
        "error": error_message,
        "timestamp": datetime.now().isoformat(),
        "status_code": status_code
    }

def paginate_results(data: List[Any], page: int = 1, page_size: int = 20) -> Dict[str, Any]:
    """Paginate list of results"""
    total_items = len(data)
    total_pages = (total_items + page_size - 1) // page_size
    
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    
    return {
        "data": data[start_index:end_index],
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        }
    }

# ===========================
#  DATABASE UTILITIES
# ===========================

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}

def save_json_file(data: Dict[str, Any], file_path: str) -> bool:
    """Save data to JSON file"""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False

# ===========================
#  EXPORT UTILITIES
# ===========================

def export_to_json(data: Any, pretty: bool = True) -> str:
    """Export data to JSON format"""
    try:
        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        else:
            return json.dumps(data, ensure_ascii=False, default=str)
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        return "{}"

def export_to_csv(data: List[Dict[str, Any]]) -> str:
    """Export data to CSV format"""
    if not data:
        return ""
    
    try:
        import csv
        from io import StringIO
        
        output = StringIO()
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            # Convert complex objects to strings
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    clean_row[key] = json.dumps(value)
                else:
                    clean_row[key] = str(value) if value is not None else ""
            writer.writerow(clean_row)
        
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return ""
