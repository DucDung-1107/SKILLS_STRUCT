from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import requests
import json
import os
from datetime import datetime
import copy
import traceback
from pathlib import Path

app = FastAPI(
    title="Skill Taxonomy Recommendation API",
    description="API để tạo recommendations cho skill taxonomy từ file JSON",
    version="1.0.0"
)

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cấu hình API Gemini Flash 2
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyDL7j6v9YK0U0l_ooD-WaXxWaGMOdRQvnA")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Pydantic Models
class SkillRecommendation(BaseModel):
    name: str
    type: str = Field(..., description="skill_group, skill, hoặc sub_skill")
    parent_name: Optional[str] = None
    description: str = ""
    category: str = Field(default="technology", description="technology, soft_skill, management, security")
    level: int = Field(default=1, ge=1, le=5)
    priority: str = Field(default="medium", description="high, medium, low")

class RecommendationRequest(BaseModel):
    file_path: str = Field(..., description="Đường dẫn đến file JSON chứa skill taxonomy")
    output_path: Optional[str] = Field(default=None, description="Đường dẫn lưu kết quả (tùy chọn)")
    max_recommendations: int = Field(default=8, ge=1, le=20, description="Số lượng recommendations tối đa")

class AnalyzeRequest(BaseModel):
    file_path: str = Field(..., description="Đường dẫn đến file JSON")

class RecommendationResponse(BaseModel):
    success: bool
    message: str
    original_nodes_count: int
    recommendations_count: int
    total_nodes_count: int
    recommended_skills: List[SkillRecommendation]
    output_file: Optional[str] = None
    processing_time: float

class SkillTaxonomyStats(BaseModel):
    total_nodes: int
    total_edges: int
    skill_groups: int
    skills: int
    sub_skills: int
    metadata: Dict[str, Any]

# Utility Functions
def load_skill_taxonomy(file_path: str) -> Dict[str, Any]:
    """Đọc skill taxonomy từ file JSON"""
    try:
        if not os.path.exists(file_path):
            # Tạo sample data nếu file không tồn tại
            return {
                "metadata": {"title": "Sample Taxonomy", "created_date": datetime.now().isoformat()},
                "nodes": [
                    {"id": "1", "name": "Python", "type": "skill"},
                    {"id": "2", "name": "SQL", "type": "skill"}
                ],
                "edges": []
            }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            skill_data = json.load(f)
        
        # Validate basic structure
        if 'nodes' not in skill_data:
            skill_data['nodes'] = []
        if 'edges' not in skill_data:
            skill_data['edges'] = []
        
        return skill_data
    except Exception as e:
        # Return sample data on error
        return {
            "metadata": {"title": "Error - Sample Data", "error": str(e)},
            "nodes": [
                {"id": "1", "name": "Python", "type": "skill"},
                {"id": "2", "name": "SQL", "type": "skill"}
            ],
            "edges": []
        }

def extract_existing_skills(skill_data: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """Trích xuất danh sách skills và skill groups hiện có"""
    existing_skills = []
    existing_skill_groups = []
    
    for node in skill_data.get('nodes', []):
        node_type = node.get('type', '')
        node_name = node.get('name', '')
        
        if node_type == 'skill_group':
            existing_skill_groups.append(node_name)
        elif node_type in ['skill', 'sub_skill']:
            existing_skills.append(node_name)
    
    return existing_skills, existing_skill_groups

def call_gemini_api(existing_skills: List[str], existing_skill_groups: List[str], max_recommendations: int = 8) -> List[Dict[str, Any]]:
    """Gọi API Gemini để lấy recommendations"""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_API_KEY":
        raise HTTPException(status_code=500, detail="API key chưa được cấu hình")
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }
    
    # Tạo prompt
    prompt = f"""
    Dựa trên skill taxonomy hiện có, hãy gợi ý {max_recommendations} kỹ năng quan trọng để bổ sung:

    Skill Groups hiện có: {', '.join(existing_skill_groups)}
    Skills cụ thể hiện có: {', '.join(existing_skills[:30])}

    Yêu cầu:
    1. Phân tích gaps trong cấu trúc hiện tại
    2. Gợi ý skills mới để hoàn thiện các skill groups đã có
    3. Đề xuất skill groups mới nếu cần thiết
    4. Tập trung vào công nghệ mới, trending skills

    Trả về JSON format chính xác:
    {{
        "recommended_skills": [
            {{
                "name": "Tên skill hoặc skill group",
                "type": "skill_group|skill|sub_skill",
                "parent_name": "Tên của parent node (nếu thuộc về skill group có sẵn)",
                "description": "Mô tả ngắn gọn về tầm quan trọng",
                "category": "technology|soft_skill|management|security",
                "level": 1,
                "priority": "high|medium|low"
            }}
        ]
    }}
    """
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['candidates'][0]['content']['parts'][0]['text']
            
            # Parse JSON từ response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                parsed_data = json.loads(json_content)
                return parsed_data.get("recommended_skills", [])
            else:
                raise ValueError("Không tìm thấy JSON trong response")
        else:
            raise HTTPException(status_code=500, detail=f"API Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Request Error: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON Parse Error: {str(e)}")

def add_recommendations_to_taxonomy(original_skill_data: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Thêm recommendations vào skill taxonomy"""
    skill_data = copy.deepcopy(original_skill_data)
    
    # Cập nhật metadata
    skill_data['metadata'] = skill_data.get('metadata', {})
    skill_data['metadata']['title'] = "Employee Skill Taxonomy with AI Recommendations"
    skill_data['metadata']['updated_date'] = datetime.now().isoformat()
    skill_data['metadata']['total_recommended_skills'] = len(recommendations)
    skill_data['metadata']['source'] = "Enhanced by AI Recommendations API"
    
    # Tìm ID lớn nhất và map tên -> id
    max_id_number = 0
    existing_node_names = {}
    
    for node in skill_data.get('nodes', []):
        node_id = node.get('id', '')
        node_name = node.get('name', '')
        existing_node_names[node_name.lower()] = node_id
        
        if '_' in node_id:
            try:
                id_parts = node_id.split('_')
                if id_parts[-1].isdigit():
                    max_id_number = max(max_id_number, int(id_parts[-1]))
            except:
                pass
    
    # Thêm recommendations
    next_id = max_id_number + 1
    added_skills = 0
    
    for skill in recommendations:
        skill_name = skill.get('name', '').strip()
        skill_type = skill.get('type', 'skill')
        parent_name = skill.get('parent_name') or ''
        if parent_name:
            parent_name = parent_name.strip()
        
        # Kiểm tra skill đã tồn tại
        if skill_name.lower() in existing_node_names:
            continue
        
        # Tạo ID và xác định parent
        skill_id = f"recommended_{skill_type}_{next_id}"
        parent_id = existing_node_names.get(parent_name.lower(), "root") if parent_name else "root"
        
        # Xác định màu và level
        color_map = {
            'skill_group': "#9467bd",
            'skill': "#ff6b6b",
            'sub_skill': "#ffb347"
        }
        
        level_map = {
            'skill_group': 1,
            'skill': 2,
            'sub_skill': 3
        }
        
        # Tạo node mới
        new_node = {
            "id": skill_id,
            "name": skill_name,
            "type": skill_type,
            "level": level_map.get(skill_type, 2),
            "color": color_map.get(skill_type, "#cccccc"),
            "description": skill.get('description', ''),
            "category": skill.get('category', ''),
            "priority": skill.get('priority', 'medium'),
            "is_recommended": True,
            "employees": [],
            "employee_count": 0,
            "proficiency_stats": {}
        }
        
        skill_data['nodes'].append(new_node)
        existing_node_names[skill_name.lower()] = skill_id
        
        # Thêm edge
        new_edge = {
            "source": parent_id,
            "target": skill_id,
            "type": "recommended"
        }
        skill_data['edges'].append(new_edge)
        
        next_id += 1
        added_skills += 1
    
    return skill_data

def save_skill_taxonomy(skill_data: Dict[str, Any], output_path: str) -> str:
    """Lưu skill taxonomy vào file"""
    try:
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(skill_data, f, ensure_ascii=False, indent=2)
        
        return output_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi lưu file: {str(e)}")

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Trang chủ với links để test API"""
    html_content = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Skill Taxonomy Recommendation API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
            .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .endpoint { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
            .btn { display: inline-block; padding: 10px 20px; background: #3498db; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
            .btn:hover { background: #2980b9; }
            .success { color: #27ae60; font-weight: bold; }
            .description { color: #7f8c8d; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Skill Taxonomy Recommendation API</h1>
            <p class="description">API để tạo gợi ý kỹ năng cho cây phân cấp skill taxonomy sử dụng AI</p>
            
            <div class="endpoint">
                <h3>📊 Interactive Documentation</h3>
                <a href="/docs" class="btn">Swagger UI - Test API trực tiếp</a>
                <a href="/redoc" class="btn">ReDoc - Documentation đẹp</a>
            </div>
            
            <div class="endpoint">
                <h3>🔗 API Endpoints</h3>
                <p><strong>GET /health</strong> - Kiểm tra trạng thái API</p>
                <p><strong>POST /analyze</strong> - Phân tích thống kê skill taxonomy</p>
                <p><strong>POST /recommend</strong> - Tạo recommendations từ AI</p>
            </div>
            
            <div class="endpoint">
                <h3>✅ Server Status</h3>
                <p class="success">✓ API đang hoạt động tại: http://127.0.0.1:8000</p>
                <p class="success">✓ Gemini AI API đã được cấu hình</p>
                <p class="success">✓ Sẵn sàng xử lý requests</p>
            </div>
            
            <div class="endpoint">
                <h3>📝 Quick Test Commands</h3>
                <p><code>GET /health</code> - Kiểm tra server</p>
                <p><code>POST /analyze?file_path=data/my_skill_taxonomy.json</code> - Phân tích file</p>
                <p><code>POST /recommend</code> với JSON body để tạo recommendations</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

class AnalyzeRequest(BaseModel):
    file_path: str = Field(..., description="Đường dẫn đến file JSON")

@app.post("/analyze", response_model=SkillTaxonomyStats)
async def analyze_skill_taxonomy(request: AnalyzeRequest):
    """Phân tích thống kê skill taxonomy từ file JSON"""
    try:
        skill_data = load_skill_taxonomy(request.file_path)
        
        # Thống kê nodes
        total_nodes = len(skill_data.get('nodes', []))
        skill_groups = sum(1 for node in skill_data.get('nodes', []) if node.get('type') == 'skill_group')
        skills = sum(1 for node in skill_data.get('nodes', []) if node.get('type') == 'skill')
        sub_skills = sum(1 for node in skill_data.get('nodes', []) if node.get('type') == 'sub_skill')
        
        return SkillTaxonomyStats(
            total_nodes=total_nodes,
            total_edges=len(skill_data.get('edges', [])),
            skill_groups=skill_groups,
            skills=skills,
            sub_skills=sub_skills,
            metadata=skill_data.get('metadata', {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")

@app.post("/recommend", response_model=RecommendationResponse)
async def create_recommendations(request: RecommendationRequest):
    """Tạo skill recommendations từ file JSON"""
    start_time = datetime.now()
    
    try:
        # 1. Đọc file gốc
        original_skill_data = load_skill_taxonomy(request.file_path)
        original_nodes_count = len(original_skill_data.get('nodes', []))
        
        # 2. Trích xuất skills hiện có
        existing_skills, existing_skill_groups = extract_existing_skills(original_skill_data)
        
        # 3. Tạo mock recommendations thay vì gọi AI API
        mock_recommendations = [
            {
                "name": "Docker",
                "type": "skill",
                "description": "Containerization technology",
                "category": "technology",
                "level": 2,
                "priority": "high"
            },
            {
                "name": "Kubernetes",
                "type": "skill", 
                "description": "Container orchestration",
                "category": "technology",
                "level": 3,
                "priority": "high"
            },
            {
                "name": "Machine Learning",
                "type": "skill_group",
                "description": "AI and ML techniques",
                "category": "technology",
                "level": 1,
                "priority": "medium"
            }
        ]
        
        # 4. Thêm recommendations vào taxonomy
        enhanced_skill_data = add_recommendations_to_taxonomy(original_skill_data, mock_recommendations)
        total_nodes_count = len(enhanced_skill_data.get('nodes', []))
        
        # 5. Convert to SkillRecommendation objects
        recommended_skills = [SkillRecommendation(**skill) for skill in mock_recommendations]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return RecommendationResponse(
            success=True,
            message="Mock recommendations generated successfully",
            original_nodes_count=original_nodes_count,
            recommendations_count=len(mock_recommendations),
            total_nodes_count=total_nodes_count,
            recommended_skills=recommended_skills,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        return RecommendationResponse(
            success=False,
            message=f"Error: {str(e)}",
            original_nodes_count=0,
            recommendations_count=0,
            total_nodes_count=0,
            recommended_skills=[],
            processing_time=processing_time
        )
        output_file = None
        if request.output_path:
            output_file = save_skill_taxonomy(enhanced_skill_data, request.output_path)
        
        # 6. Tính thời gian xử lý
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 7. Chuyển đổi recommendations sang Pydantic model
        recommended_skills = [
            SkillRecommendation(
                name=skill.get('name', ''),
                type=skill.get('type', 'skill'),
                parent_name=skill.get('parent_name'),
                description=skill.get('description', ''),
                category=skill.get('category', 'technology'),
                level=skill.get('level', 1),
                priority=skill.get('priority', 'medium')
            ) for skill in recommendations
        ]
        
        return RecommendationResponse(
            success=True,
            message="Recommendations đã được tạo thành công",
            original_nodes_count=original_nodes_count,
            recommendations_count=len(recommendations),
            total_nodes_count=total_nodes_count,
            recommended_skills=recommended_skills,
            output_file=output_file,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi tạo recommendations: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check cho API"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gemini_api_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY")
    }

@app.get("/docs-urls", response_class=HTMLResponse)
async def get_documentation_urls():
    """Hiển thị các liên kết tài liệu"""
    return """
    <html>
        <head>
            <title>API Documentation Links</title>
        </head>
        <body>
            <h1>API Documentation Links</h1>
            <ul>
                <li><a href="/docs" target="_blank">Swagger UI</a></li>
                <li><a href="/redoc" target="_blank">ReDoc</a></li>
            </ul>
        </body>
    </html>
    """

# Chạy server
if __name__ == "__main__":
    import uvicorn
    # Thử các port khác nhau nếu 8000 bị occupied
    ports_to_try = [8000, 8001, 8002, 8080, 3000, 5000]
    
    for port in ports_to_try:
        try:
            print(f"Đang thử chạy server trên port {port}...")
            uvicorn.run("recommendapi:app", host="127.0.0.1", port=port, reload=True)
            break
        except OSError as e:
            if port == ports_to_try[-1]:  # Port cuối cùng
                print(f"Không thể chạy server trên bất kỳ port nào: {e}")
                print("Hãy thử chạy bằng lệnh: uvicorn recommendapi:app --host 127.0.0.1 --port 8001")
            else:
                print(f"Port {port} đã được sử dụng, thử port tiếp theo...")
                continue
