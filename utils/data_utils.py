#!/usr/bin/env python3
"""
📊 Data Processing Utilities
Các hàm tiện ích cho xử lý và phân tích dữ liệu
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Làm sạch text: loại bỏ ký tự đặc biệt, normalize spaces
    """
    if not text:
        return ""
    
    # Remove special characters but keep Vietnamese
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_skills(text: str, skill_keywords: List[str] = None) -> List[str]:
    """
    Trích xuất skills từ text
    """
    if not text:
        return []
    
    # Default skill keywords if not provided
    if skill_keywords is None:
        skill_keywords = [
            # Programming Languages
            'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'typescript', 'scala', 'kotlin', 'swift', 'dart', 'r', 'matlab',
            
            # Frameworks & Libraries
            'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask',
            'spring', 'laravel', 'rails', 'tensorflow', 'pytorch', 'scikit-learn',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle',
            'sqlite', 'cassandra', 'dynamodb',
            
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
            'terraform', 'ansible', 'prometheus', 'grafana',
            
            # Tools & Technologies
            'git', 'linux', 'windows', 'agile', 'scrum', 'jira', 'confluence',
            'photoshop', 'illustrator', 'figma', 'sketch'
        ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in skill_keywords:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return list(set(found_skills))  # Remove duplicates

def normalize_skill_name(skill: str) -> str:
    """
    Normalize tên skill: lowercase, remove spaces, etc.
    """
    if not skill:
        return ""
    
    # Convert to lowercase
    skill = skill.lower().strip()
    
    # Replace common variations
    replacements = {
        'javascript': 'javascript',
        'js': 'javascript',
        'node.js': 'nodejs',
        'node': 'nodejs',
        'react.js': 'react',
        'vue.js': 'vue',
        'c++': 'cpp',
        'c#': 'csharp',
        '.net': 'dotnet',
        'artificial intelligence': 'ai',
        'machine learning': 'ml',
        'deep learning': 'dl'
    }
    
    return replacements.get(skill, skill)

def calculate_similarity(str1: str, str2: str) -> float:
    """
    Tính độ tương tự giữa 2 chuỗi (0-1)
    """
    if not str1 or not str2:
        return 0.0
    
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def merge_duplicates(items: List[str], similarity_threshold: float = 0.8) -> List[str]:
    """
    Gộp các item trùng lặp dựa trên độ tương tự
    """
    if not items:
        return []
    
    merged = []
    used_indices = set()
    
    for i, item1 in enumerate(items):
        if i in used_indices:
            continue
        
        group = [item1]
        used_indices.add(i)
        
        for j, item2 in enumerate(items[i+1:], i+1):
            if j in used_indices:
                continue
            
            if calculate_similarity(item1, item2) >= similarity_threshold:
                group.append(item2)
                used_indices.add(j)
        
        # Use the most common or longest name in the group
        merged.append(max(group, key=len))
    
    return merged

def extract_experience_years(text: str) -> Optional[int]:
    """
    Trích xuất số năm kinh nghiệm từ text
    """
    if not text:
        return None
    
    # Patterns for extracting years of experience
    patterns = [
        r'(\d+)\s*(?:years?|năm)\s*(?:of\s*)?(?:experience|kinh nghiệm)',
        r'(\d+)\+?\s*(?:years?|năm)',
        r'(?:experience|kinh nghiệm).*?(\d+)\s*(?:years?|năm)',
        r'(\d+)\s*(?:years?|năm).*?(?:experience|kinh nghiệm)'
    ]
    
    text_lower = text.lower()
    
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        if matches:
            try:
                years = int(matches[0])
                # Reasonable bounds for years of experience
                if 0 <= years <= 50:
                    return years
            except ValueError:
                continue
    
    return None

def extract_education_level(text: str) -> Optional[str]:
    """
    Trích xuất trình độ học vấn từ text
    """
    if not text:
        return None
    
    education_patterns = {
        'phd': ['phd', 'doctorate', 'tiến sĩ', 'ph.d'],
        'master': ['master', 'thạc sĩ', 'mba', 'ms', 'ma', 'msc'],
        'bachelor': ['bachelor', 'đại học', 'cử nhân', 'bs', 'ba', 'bsc'],
        'diploma': ['diploma', 'cao đẳng', 'associate'],
        'certificate': ['certificate', 'chứng chỉ', 'cert']
    }
    
    text_lower = text.lower()
    
    for level, patterns in education_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                return level
    
    return None

def extract_emails(text: str) -> List[str]:
    """
    Trích xuất email addresses từ text
    """
    if not text:
        return []
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    return list(set(emails))  # Remove duplicates

def extract_phone_numbers(text: str) -> List[str]:
    """
    Trích xuất phone numbers từ text
    """
    if not text:
        return []
    
    # Patterns for Vietnamese and international phone numbers
    phone_patterns = [
        r'\+84[-.\s]?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # +84 format
        r'0\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # 0xx format
        r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}',  # (xxx) xxx-xxxx
        r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'   # xxx-xxx-xxxx
    ]
    
    phones = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        phones.extend(matches)
    
    # Clean and normalize phone numbers
    cleaned_phones = []
    for phone in phones:
        cleaned = re.sub(r'[-.\s()]', '', phone)
        if len(cleaned) >= 10:  # Minimum valid phone length
            cleaned_phones.append(cleaned)
    
    return list(set(cleaned_phones))

def analyze_skill_distribution(skills_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phân tích phân bố skills trong dataset
    """
    if not skills_data:
        return {}
    
    # Count skills
    skill_counts = Counter()
    proficiency_dist = defaultdict(Counter)
    department_skills = defaultdict(set)
    
    for record in skills_data:
        skill = record.get('skill_name', '').lower()
        proficiency = record.get('proficiency_level', 'beginner')
        department = record.get('department', 'unknown')
        
        if skill:
            skill_counts[skill] += 1
            proficiency_dist[skill][proficiency] += 1
            department_skills[department].add(skill)
    
    # Calculate statistics
    total_skills = len(skill_counts)
    most_common_skills = skill_counts.most_common(10)
    
    # Calculate average proficiency
    avg_proficiency = {}
    proficiency_weights = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
    
    for skill, prof_counts in proficiency_dist.items():
        total_weight = sum(count * proficiency_weights.get(prof, 1) 
                          for prof, count in prof_counts.items())
        total_count = sum(prof_counts.values())
        avg_proficiency[skill] = total_weight / total_count if total_count > 0 else 1
    
    return {
        'total_unique_skills': total_skills,
        'most_common_skills': most_common_skills,
        'proficiency_distribution': dict(proficiency_dist),
        'average_proficiency': avg_proficiency,
        'skills_by_department': {dept: list(skills) for dept, skills in department_skills.items()},
        'total_records': len(skills_data)
    }

def identify_skill_gaps(required_skills: List[str], available_skills: List[str]) -> Dict[str, Any]:
    """
    Xác định khoảng trống kỹ năng
    """
    required_set = set(skill.lower() for skill in required_skills)
    available_set = set(skill.lower() for skill in available_skills)
    
    missing_skills = required_set - available_set
    extra_skills = available_set - required_set
    matching_skills = required_set & available_set
    
    coverage_percent = (len(matching_skills) / len(required_set)) * 100 if required_set else 0
    
    return {
        'missing_skills': list(missing_skills),
        'extra_skills': list(extra_skills),
        'matching_skills': list(matching_skills),
        'coverage_percent': round(coverage_percent, 2),
        'gap_count': len(missing_skills),
        'total_required': len(required_set),
        'total_available': len(available_set)
    }

def cluster_similar_skills(skills: List[str], similarity_threshold: float = 0.7) -> List[List[str]]:
    """
    Gom nhóm các skills tương tự nhau
    """
    if not skills:
        return []
    
    clusters = []
    used_skills = set()
    
    for skill in skills:
        if skill in used_skills:
            continue
        
        cluster = [skill]
        used_skills.add(skill)
        
        for other_skill in skills:
            if other_skill not in used_skills:
                similarity = calculate_similarity(skill, other_skill)
                if similarity >= similarity_threshold:
                    cluster.append(other_skill)
                    used_skills.add(other_skill)
        
        clusters.append(cluster)
    
    return clusters

def calculate_skill_importance(skill_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Tính toán độ quan trọng của skills dựa trên frequency và proficiency
    """
    skill_stats = defaultdict(lambda: {'count': 0, 'total_proficiency': 0})
    proficiency_weights = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
    
    for record in skill_data:
        skill = record.get('skill_name', '').lower()
        proficiency = record.get('proficiency_level', 'beginner')
        
        if skill:
            skill_stats[skill]['count'] += 1
            skill_stats[skill]['total_proficiency'] += proficiency_weights.get(proficiency, 1)
    
    # Calculate importance score
    importance_scores = {}
    max_count = max(stats['count'] for stats in skill_stats.values()) if skill_stats else 1
    
    for skill, stats in skill_stats.items():
        frequency_score = stats['count'] / max_count
        avg_proficiency = stats['total_proficiency'] / stats['count']
        proficiency_score = avg_proficiency / 4  # Normalize to 0-1
        
        # Weighted combination
        importance_scores[skill] = (frequency_score * 0.6) + (proficiency_score * 0.4)
    
    return importance_scores

def generate_skill_recommendations(current_skills: List[str], target_role: str = None) -> List[str]:
    """
    Đề xuất skills cần học dựa trên skills hiện tại
    """
    # Skill progression paths
    skill_paths = {
        'python': ['django', 'flask', 'pandas', 'numpy', 'scikit-learn', 'tensorflow'],
        'javascript': ['react', 'nodejs', 'express', 'mongodb', 'typescript'],
        'java': ['spring', 'hibernate', 'maven', 'junit', 'microservices'],
        'react': ['redux', 'next.js', 'react-native', 'graphql'],
        'nodejs': ['express', 'mongodb', 'socket.io', 'jest', 'docker'],
        'sql': ['postgresql', 'mysql', 'data modeling', 'etl', 'tableau'],
    }
    
    # Role-based skill requirements
    role_skills = {
        'data_scientist': ['python', 'r', 'sql', 'machine learning', 'statistics', 'tableau'],
        'web_developer': ['html', 'css', 'javascript', 'react', 'nodejs', 'sql'],
        'devops': ['docker', 'kubernetes', 'aws', 'terraform', 'jenkins', 'monitoring'],
        'mobile_developer': ['react-native', 'flutter', 'swift', 'kotlin', 'firebase']
    }
    
    recommendations = set()
    current_skills_lower = [skill.lower() for skill in current_skills]
    
    # Add skills from progression paths
    for skill in current_skills_lower:
        if skill in skill_paths:
            for recommended_skill in skill_paths[skill]:
                if recommended_skill not in current_skills_lower:
                    recommendations.add(recommended_skill)
    
    # Add role-specific skills
    if target_role and target_role in role_skills:
        for skill in role_skills[target_role]:
            if skill.lower() not in current_skills_lower:
                recommendations.add(skill)
    
    return list(recommendations)[:10]  # Return top 10 recommendations
