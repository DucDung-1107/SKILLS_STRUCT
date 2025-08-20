#!/usr/bin/env python3
"""
🤖 AI Integration Utilities
Các hàm tiện ích cho tích hợp AI/ML và NLP
"""

import json
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# AI API Configuration
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-1.5-flash"

def call_gemini_api(prompt: str, api_key: str, model: str = DEFAULT_MODEL, 
                   **kwargs) -> Optional[Dict[str, Any]]:
    """
    Gọi Gemini API
    """
    if not api_key:
        logger.error("API key is required")
        return None
    
    url = f"{GEMINI_API_BASE}/models/{model}:generateContent"
    
    # Default generation config
    generation_config = {
        "temperature": kwargs.get("temperature", 0.1),
        "top_p": kwargs.get("top_p", 0.95),
        "top_k": kwargs.get("top_k", 40),
        "max_output_tokens": kwargs.get("max_output_tokens", 8192),
        "response_mime_type": kwargs.get("response_mime_type", "application/json")
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": generation_config
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return {
                        "success": True,
                        "text": parts[0]["text"],
                        "full_response": result
                    }
        
        return {
            "success": False,
            "error": "Invalid response format",
            "full_response": result
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Gemini API: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error in Gemini API call: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def extract_features_with_ai(text_content: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    Trích xuất features từ CV bằng AI
    """
    prompt = f"""
    Extract the following information from this resume text and return it as a JSON object. 
    If any information is not found, use null or empty array.

    Required fields:
    - name: Full name of the candidate
    - email: Email address
    - phone: Phone number
    - address: Full address or location
    - linkedin: LinkedIn profile URL
    - skills: List of technical skills (as array)
    - experience_years: Total years of experience (as number)
    - education: Highest education degree
    - university: Name of university/college
    - certifications: List of certifications (as array)
    - languages: List of languages spoken (as array)
    - job_titles: List of job titles from experience (as array)
    - companies: List of companies worked at (as array)
    - summary: Brief professional summary (2-3 sentences)

    Resume text:
    {text_content}

    Return only valid JSON format without any additional text:
    """
    
    result = call_gemini_api(prompt, api_key)
    
    if result and result.get("success"):
        try:
            # Parse JSON response
            features = json.loads(result["text"])
            return features
        except json.JSONDecodeError:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', result["text"], re.DOTALL)
            if json_match:
                try:
                    features = json.loads(json_match.group())
                    return features
                except json.JSONDecodeError:
                    logger.error("Could not parse JSON from AI response")
                    return None
    
    return None

def generate_skill_taxonomy_with_ai(employee_data: List[Dict[str, Any]], 
                                  api_key: str) -> Optional[Dict[str, Any]]:
    """
    Tạo skill taxonomy từ dữ liệu nhân viên bằng AI
    """
    # Convert data to JSON string
    data_json = json.dumps(employee_data, ensure_ascii=False, indent=2)
    
    prompt = f"""
    Hãy phân tích dữ liệu nhân viên sau và tạo một skill taxonomy graph dạng cây phân cấp từ tổng quát đến chi tiết.

    Dữ liệu đầu vào:
    {data_json}

    Yêu cầu:
    1. Tạo cấu trúc cây skill taxonomy từ department/team → skill group → skill → sub-skill
    2. Mỗi node có thông tin: id, name, type, level, employees, proficiency_stats
    3. Mỗi edge thể hiện quan hệ parent-child
    4. Bao gồm thông tin màu sắc và độ đậm dựa trên proficiency level
    5. Kết quả có thể import vào Mermaid, D3.js hoặc graph editor

    Trả về JSON với cấu trúc:
    {{
        "metadata": {{
            "title": "Employee Skill Taxonomy",
            "description": "Hierarchical skill structure", 
            "created_date": "{datetime.now().isoformat()}",
            "total_employees": <number>,
            "total_skills": <number>
        }},
        "nodes": [
            {{
                "id": "unique_id",
                "name": "Node Name", 
                "type": "root|skill_group|skill|sub_skill",
                "level": <number>,
                "color": "#hexcolor",
                "employees": ["emp1", "emp2"],
                "employee_count": <number>,
                "proficiency_stats": {{
                    "beginner": <count>,
                    "intermediate": <count>, 
                    "advanced": <count>,
                    "expert": <count>
                }}
            }}
        ],
        "edges": [
            {{
                "id": "edge_id",
                "source": "parent_node_id",
                "target": "child_node_id", 
                "type": "contains|includes|specializes",
                "weight": 1.0
            }}
        ],
        "color_scheme": {{
            "root": "#808080",
            "skill_group": "#ff7f0e",
            "skill": "#2ca02c", 
            "sub_skill": "#d62728"
        }}
    }}

    Chỉ trả về JSON hợp lệ, không có text thêm:
    """
    
    result = call_gemini_api(prompt, api_key, max_output_tokens=8192)
    
    if result and result.get("success"):
        try:
            taxonomy = json.loads(result["text"])
            return taxonomy
        except json.JSONDecodeError:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', result["text"], re.DOTALL)
            if json_match:
                try:
                    taxonomy = json.loads(json_match.group())
                    return taxonomy
                except json.JSONDecodeError:
                    logger.error("Could not parse taxonomy JSON from AI response")
    
    return None

def generate_recommendations(taxonomy: Dict[str, Any], api_key: str, 
                           max_recommendations: int = 8) -> List[Dict[str, Any]]:
    """
    Tạo skill recommendations bằng AI
    """
    # Prepare current taxonomy summary
    nodes_summary = []
    for node in taxonomy.get("nodes", []):
        nodes_summary.append({
            "name": node.get("name"),
            "type": node.get("type"),
            "employee_count": node.get("employee_count", 0)
        })
    
    prompt = f"""
    Dựa trên skill taxonomy hiện tại, hãy đề xuất {max_recommendations} skills mới để mở rộng taxonomy.

    Taxonomy hiện tại:
    {json.dumps(nodes_summary, ensure_ascii=False, indent=2)}

    Yêu cầu đề xuất:
    1. Skills phù hợp với xu hướng công nghệ hiện tại
    2. Skills bổ sung cho các skill groups đã có
    3. Skills quan trọng cho sự phát triển nghề nghiệp
    4. Ưu tiên skills có nhu cầu cao trên thị trường

    Trả về JSON array với format:
    [
        {{
            "name": "Skill Name",
            "type": "skill_group|skill|sub_skill",
            "parent_name": "Parent Skill Name (if applicable)",
            "description": "Brief description of the skill",
            "category": "technology|soft_skill|management|security|design|business",
            "level": 1-5,
            "priority": "high|medium|low"
        }}
    ]

    Chỉ trả về JSON array hợp lệ:
    """
    
    result = call_gemini_api(prompt, api_key)
    
    if result and result.get("success"):
        try:
            recommendations = json.loads(result["text"])
            if isinstance(recommendations, list):
                return recommendations[:max_recommendations]
        except json.JSONDecodeError:
            logger.error("Could not parse recommendations JSON")
    
    return []

def analyze_skill_gaps(required_skills: List[str], available_skills: List[str], 
                      api_key: str) -> Dict[str, Any]:
    """
    Phân tích skill gaps bằng AI
    """
    prompt = f"""
    Phân tích khoảng cách kỹ năng dựa trên yêu cầu và kỹ năng hiện có:

    Kỹ năng yêu cầu:
    {json.dumps(required_skills, ensure_ascii=False)}

    Kỹ năng hiện có:
    {json.dumps(available_skills, ensure_ascii=False)}

    Hãy phân tích và đưa ra:
    1. Kỹ năng còn thiếu (gaps)
    2. Kỹ năng thừa 
    3. Mức độ matching
    4. Đề xuất training/hiring
    5. Độ ưu tiên cho từng skill gap

    Trả về JSON:
    {{
        "missing_skills": ["skill1", "skill2"],
        "extra_skills": ["skill3", "skill4"],  
        "matching_skills": ["skill5", "skill6"],
        "coverage_percent": <percentage>,
        "gap_analysis": [
            {{
                "skill": "Missing Skill Name",
                "priority": "high|medium|low",
                "reason": "Why this skill is important",
                "recommendation": "Specific action to address gap"
            }}
        ],
        "training_recommendations": ["recommendation1", "recommendation2"],
        "hiring_recommendations": ["recommendation1", "recommendation2"]
    }}

    Chỉ trả về JSON hợp lệ:
    """
    
    result = call_gemini_api(prompt, api_key)
    
    if result and result.get("success"):
        try:
            analysis = json.loads(result["text"])
            return analysis
        except json.JSONDecodeError:
            logger.error("Could not parse gap analysis JSON")
    
    return {}

def cluster_skills(skills: List[str], api_key: str) -> List[List[str]]:
    """
    Cluster skills theo semantic similarity bằng AI
    """
    prompt = f"""
    Hãy phân nhóm các skills sau đây theo semantic similarity và domain:

    Skills:
    {json.dumps(skills, ensure_ascii=False)}

    Yêu cầu:
    1. Nhóm các skills có liên quan với nhau
    2. Mỗi nhóm nên có 2-8 skills
    3. Đặt tên cho mỗi nhóm
    4. Sắp xếp từ nhóm quan trọng nhất

    Trả về JSON:
    {{
        "clusters": [
            {{
                "name": "Cluster Name",
                "skills": ["skill1", "skill2", "skill3"],
                "description": "Brief description of this cluster"
            }}
        ]
    }}

    Chỉ trả về JSON hợp lệ:
    """
    
    result = call_gemini_api(prompt, api_key)
    
    if result and result.get("success"):
        try:
            clusters_data = json.loads(result["text"])
            clusters = []
            for cluster in clusters_data.get("clusters", []):
                cluster_skills = cluster.get("skills", [])
                if cluster_skills:
                    clusters.append(cluster_skills)
            return clusters
        except json.JSONDecodeError:
            logger.error("Could not parse clusters JSON")
    
    return []

def generate_mermaid_code(taxonomy: Dict[str, Any], api_key: str) -> str:
    """
    Tạo Mermaid diagram code từ taxonomy bằng AI
    """
    prompt = f"""
    Tạo Mermaid diagram code từ skill taxonomy này:

    {json.dumps(taxonomy, ensure_ascii=False, indent=2)}

    Yêu cầu:
    1. Sử dụng graph TD (top-down)
    2. Tạo các node với shape phù hợp với type
    3. Thêm màu sắc cho các node
    4. Tạo connections giữa parent-child
    5. Code phải chạy được trên Mermaid

    Chỉ trả về Mermaid code thuần túy, không có markdown wrapper:
    """
    
    result = call_gemini_api(prompt, api_key, response_mime_type="text/plain")
    
    if result and result.get("success"):
        return result["text"].strip()
    
    return ""

def extract_job_requirements(job_description: str, api_key: str) -> Dict[str, Any]:
    """
    Trích xuất requirements từ job description bằng AI
    """
    prompt = f"""
    Trích xuất thông tin yêu cầu từ job description này:

    {job_description}

    Trả về JSON:
    {{
        "position": "Job title",
        "required_skills": ["skill1", "skill2"],
        "preferred_skills": ["skill3", "skill4"],
        "experience_years": <number>,
        "education_level": "bachelor|master|phd|diploma",
        "certifications": ["cert1", "cert2"],
        "soft_skills": ["skill1", "skill2"],
        "responsibilities": ["task1", "task2"],
        "company_info": "Brief company description",
        "salary_range": "If mentioned",
        "location": "Work location"
    }}

    Chỉ trả về JSON hợp lệ:
    """
    
    result = call_gemini_api(prompt, api_key)
    
    if result and result.get("success"):
        try:
            requirements = json.loads(result["text"])
            return requirements
        except json.JSONDecodeError:
            logger.error("Could not parse job requirements JSON")
    
    return {}

def match_candidate_to_job(candidate_skills: List[str], job_requirements: Dict[str, Any], 
                          api_key: str) -> Dict[str, Any]:
    """
    Matching candidate với job requirements bằng AI
    """
    prompt = f"""
    Đánh giá mức độ phù hợp của candidate với job requirements:

    Candidate skills:
    {json.dumps(candidate_skills, ensure_ascii=False)}

    Job requirements:
    {json.dumps(job_requirements, ensure_ascii=False)}

    Trả về JSON:
    {{
        "match_score": <0-100>,
        "matching_skills": ["skill1", "skill2"],
        "missing_skills": ["skill3", "skill4"],
        "extra_skills": ["skill5", "skill6"],
        "skill_gaps": [
            {{
                "skill": "Missing skill",
                "importance": "high|medium|low",
                "can_be_trained": true/false
            }}
        ],
        "recommendations": [
            "Specific recommendation for improvement"
        ],
        "interview_questions": [
            "Suggested interview question"
        ],
        "decision": "hire|interview|reject",
        "reasoning": "Brief explanation of the decision"
    }}

    Chỉ trả về JSON hợp lệ:
    """
    
    result = call_gemini_api(prompt, api_key)
    
    if result and result.get("success"):
        try:
            match_result = json.loads(result["text"])
            return match_result
        except json.JSONDecodeError:
            logger.error("Could not parse matching result JSON")
    
    return {}

def generate_learning_path(current_skills: List[str], target_role: str, 
                         api_key: str) -> List[Dict[str, Any]]:
    """
    Tạo learning path bằng AI
    """
    prompt = f"""
    Tạo learning path từ skills hiện tại đến target role:

    Current skills: {json.dumps(current_skills, ensure_ascii=False)}
    Target role: {target_role}

    Trả về JSON array với learning steps theo thứ tự:
    [
        {{
            "step": 1,
            "skill": "Skill to learn",
            "type": "technical|soft_skill|certification",
            "difficulty": "beginner|intermediate|advanced",
            "estimated_weeks": <number>,
            "prerequisites": ["prerequisite1", "prerequisite2"],
            "resources": [
                {{
                    "type": "course|book|practice|project",
                    "name": "Resource name",
                    "url": "URL if available",
                    "cost": "free|paid"
                }}
            ],
            "milestone": "What you should achieve",
            "assessment": "How to verify you learned it"
        }}
    ]

    Chỉ trả về JSON array hợp lệ:
    """
    
    result = call_gemini_api(prompt, api_key)
    
    if result and result.get("success"):
        try:
            learning_path = json.loads(result["text"])
            if isinstance(learning_path, list):
                return learning_path
        except json.JSONDecodeError:
            logger.error("Could not parse learning path JSON")
    
    return []

def validate_ai_response(response_text: str, expected_format: str = "json") -> bool:
    """
    Validate AI response format
    """
    if not response_text:
        return False
    
    if expected_format == "json":
        try:
            json.loads(response_text)
            return True
        except json.JSONDecodeError:
            return False
    
    return True

def clean_ai_response(response_text: str) -> str:
    """
    Clean AI response (remove markdown, extra formatting)
    """
    if not response_text:
        return ""
    
    # Remove markdown code blocks
    response_text = re.sub(r'```json\s*|\s*```', '', response_text)
    response_text = re.sub(r'```\s*|\s*```', '', response_text)
    
    # Remove extra whitespace
    response_text = response_text.strip()
    
    return response_text
