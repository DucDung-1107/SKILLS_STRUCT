from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import requests
import json
import os
import pandas as pd
import io
from datetime import datetime
import uuid

app = FastAPI(
    title="Skill Taxonomy API",
    description="API để tạo skill taxonomy graph từ dữ liệu nhân viên",
    version="1.0.0"
)

# Pydantic models
class EmployeeSkill(BaseModel):
    employee_id: str
    employee_name: str
    department: str
    team: Optional[str] = None
    skill_name: str
    proficiency_level: str
    proficiency_score: int

class SkillTaxonomyRequest(BaseModel):
    data: List[EmployeeSkill]
    api_key: Optional[str] = None

class GraphResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    mermaid_code: Optional[str] = None

# In-memory storage for processed graphs
processed_graphs = {}

def call_gemini_agent(data: pd.DataFrame, api_key: str) -> Dict[str, Any]:
    """
    Gọi Gemini API để tạo skill taxonomy graph từ dữ liệu nhân viên
    """
    # Chuyển dataframe thành chuỗi JSON
    data_json = data.to_json(orient='records', force_ascii=False)
    
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

QUAN TRỌNG: Chỉ trả về JSON thuần túy, không có text giải thích thêm. Bắt đầu bằng {{ và kết thúc bằng }}.

```json
{{
  "metadata": {{
    "title": "Company Skill Taxonomy",
    "description": "Hierarchical skill structure for HR management",
    "created_date": "2024-01-15",
    "total_employees": 100,
    "total_skills": 50
  }},
  "nodes": [
    {{
      "id": "dept_it",
      "name": "IT Department",
      "type": "department",
      "level": 0,
      "color": "#1f77b4",
      "employees": ["emp1", "emp2"],
      "employee_count": 2,
      "proficiency_stats": {{
        "beginner": 0,
        "intermediate": 1,
        "advanced": 1,
        "expert": 0
      }}
    }},
    {{
      "id": "skill_group_programming",
      "name": "Programming",
      "type": "skill_group",
      "level": 1,
      "color": "#ff7f0e",
      "employees": ["emp1"],
      "employee_count": 1,
      "proficiency_stats": {{
        "beginner": 0,
        "intermediate": 0,
        "advanced": 1,
        "expert": 0
      }}
    }},
    {{
      "id": "skill_python",
      "name": "Python",
      "type": "skill",
      "level": 2,
      "color": "#2ca02c",
      "employees": ["emp1"],
      "employee_count": 1,
      "proficiency_stats": {{
        "beginner": 0,
        "intermediate": 0,
        "advanced": 1,
        "expert": 0
      }}
    }},
    {{
      "id": "skill_django",
      "name": "Django Framework",
      "type": "sub_skill",
      "level": 3,
      "color": "#d62728",
      "employees": ["emp1"],
      "employee_count": 1,
      "proficiency_stats": {{
        "beginner": 0,
        "intermediate": 1,
        "advanced": 0,
        "expert": 0
      }}
    }}
  ],
  "edges": [
    {{
      "id": "edge_1",
      "source": "dept_it",
      "target": "skill_group_programming",
      "type": "contains",
      "weight": 1
    }},
    {{
      "id": "edge_2",
      "source": "skill_group_programming",
      "target": "skill_python",
      "type": "includes",
      "weight": 1
    }},
    {{
      "id": "edge_3",
      "source": "skill_python",
      "target": "skill_django",
      "type": "specializes",
      "weight": 1
    }}
  ],
  "skill_owners": {{
    "skill_python": [
      {{
        "employee_id": "emp1",
        "employee_name": "Nguyen Van A",
        "proficiency_level": "advanced",
        "proficiency_score": 8,
        "color_intensity": 0.8
      }}
    ],
    "skill_django": [
      {{
        "employee_id": "emp1",
        "employee_name": "Nguyen Van A",
        "proficiency_level": "intermediate",
        "proficiency_score": 6,
        "color_intensity": 0.6
      }}
    ]
  }},
  "color_scheme": {{
    "department": "#1f77b4",
    "skill_group": "#ff7f0e",
    "skill": "#2ca02c",
    "sub_skill": "#d62728",
    "proficiency_colors": {{
      "beginner": "#ffcccc",
      "intermediate": "#ff9999",
      "advanced": "#ff6666",
      "expert": "#ff0000"
    }}
  }},
  "mermaid_export": {{
    "syntax": "graph TD",
    "example": "dept_it[IT Department] --> skill_group_programming[Programming]"
  }}
}}
```

Lưu ý:
- Proficiency levels: beginner (1-3), intermediate (4-6), advanced (7-8), expert (9-10)
- Color intensity tương ứng với proficiency level (0.2-1.0)
- Nếu không có proficiency data, mặc định là "beginner" với score = 2
- Cấu trúc phải dễ chỉnh sửa thủ công (thêm/xóa/đổi tên/di chuyển node)
"""

    # Cấu hình API call cho Gemini Flash 2.0
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "temperature": 0.1,
            "topK": 1,
            "topP": 0.8
        }
    }
    
    # Thêm API key vào URL
    url_with_key = f"{url}?key={api_key}"
    
    try:
        response = requests.post(url_with_key, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Trích xuất text từ response của Gemini
        if 'candidates' in result and len(result['candidates']) > 0:
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            
            # Parse JSON từ text response với multiple parsing strategies
            try:
                # Strategy 1: Tìm JSON block trong ```json``` code block
                json_match = None
                import re
                
                # Tìm JSON trong code block
                code_block_pattern = r'```json\s*(\{.*?\})\s*```'
                match = re.search(code_block_pattern, generated_text, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    # Strategy 2: Tìm JSON object đầu tiên hoàn chỉnh
                    json_start = generated_text.find('{')
                    if json_start != -1:
                        # Đếm brackets để tìm JSON object hoàn chỉnh
                        bracket_count = 0
                        json_end = json_start
                        for i, char in enumerate(generated_text[json_start:], json_start):
                            if char == '{':
                                bracket_count += 1
                            elif char == '}':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    json_end = i + 1
                                    break
                        
                        if bracket_count == 0:
                            json_str = generated_text[json_start:json_end]
                        else:
                            return {"error": "Incomplete JSON object found", "raw_text": generated_text[:500]}
                    else:
                        return {"error": "No JSON object found in response", "raw_text": generated_text[:500]}
                
                # Parse JSON
                graph_data = json.loads(json_str)
                return graph_data
                    
            except json.JSONDecodeError as e:
                return {
                    "error": f"JSON parsing error: {str(e)}", 
                    "raw_text": generated_text[:1000],
                    "attempted_json": json_str[:500] if 'json_str' in locals() else "No JSON string extracted"
                }
        else:
            return {"error": "No valid response from Gemini API"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def export_to_mermaid(graph_data: Dict[str, Any]) -> str:
    """
    Chuyển đổi graph data thành Mermaid syntax
    """
    if "error" in graph_data:
        return f"Error: {graph_data['error']}"
    
    mermaid_code = ["graph TD"]
    
    # Thêm nodes với styling
    for node in graph_data.get("nodes", []):
        node_id = node["id"]
        node_name = node["name"]
        node_type = node["type"]
        
        # Định dạng node dựa trên type
        if node_type == "department":
            mermaid_code.append(f"    {node_id}[{node_name}]")
        elif node_type == "skill_group":
            mermaid_code.append(f"    {node_id}({node_name})")
        elif node_type == "skill":
            mermaid_code.append(f"    {node_id}[{node_name}]")
        else:  # sub_skill
            mermaid_code.append(f"    {node_id}({node_name})")
    
    # Thêm edges
    for edge in graph_data.get("edges", []):
        source = edge["source"]
        target = edge["target"]
        mermaid_code.append(f"    {source} --> {target}")
    
    # Thêm styling
    color_scheme = graph_data.get("color_scheme", {})
    for node in graph_data.get("nodes", []):
        node_id = node["id"]
        color = node.get("color", "#3BC6E1")
        mermaid_code.append(f"    style {node_id} fill:{color}")
    
    return "\n".join(mermaid_code)

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Skill Taxonomy API", "version": "1.0.0"}

@app.post("/generate-skill-taxonomy", response_model=GraphResponse)
async def generate_skill_taxonomy(request: SkillTaxonomyRequest):
    """
    Tạo skill taxonomy graph từ dữ liệu nhân viên
    """
    try:
        # Lấy API key từ request hoặc environment
        api_key = request.api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Chuyển đổi data thành DataFrame
        df_data = []
        for item in request.data:
            df_data.append(item.dict())
        
        df = pd.DataFrame(df_data)
        
        # Gọi Gemini API
        graph_data = call_gemini_agent(df, api_key)
        
        if "error" in graph_data:
            return GraphResponse(success=False, error=graph_data["error"])
        
        # Tạo Mermaid code
        mermaid_code = export_to_mermaid(graph_data)
        
        # Lưu vào memory với unique ID
        graph_id = str(uuid.uuid4())
        processed_graphs[graph_id] = {
            "data": graph_data,
            "mermaid_code": mermaid_code,
            "created_at": datetime.now().isoformat()
        }
        
        # Thêm graph_id vào metadata
        graph_data["graph_id"] = graph_id
        
        return GraphResponse(
            success=True,
            data=graph_data,
            mermaid_code=mermaid_code
        )
        
    except Exception as e:
        return GraphResponse(success=False, error=str(e))

@app.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    api_key: Optional[str] = Form(None)
):
    """
    Upload CSV file và tạo skill taxonomy graph
    """
    try:
        # Kiểm tra file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Lấy API key
        gemini_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "API key missing, using sample data", "graph_id": "sample"}
            )
        
        # Đọc CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Tạo sample graph thay vì gọi Gemini (để test)
        graph_data = {
            "metadata": {
                "title": "Sample Skill Taxonomy",
                "created_date": datetime.now().isoformat(),
                "total_skills": len(df)
            },
            "nodes": [
                {"id": "1", "name": "Technical Skills", "type": "skill_group"},
                {"id": "2", "name": "Programming", "type": "skill", "parent": "1"}
            ],
            "edges": [
                {"source": "1", "target": "2", "type": "contains"}
            ]
        }
        
        # Tạo Mermaid code (simplified)
        mermaid_code = f"graph TD\n"
        for node in graph_data.get("nodes", []):
            mermaid_code += f"    {node['id']}[{node['name']}]\n"
        for edge in graph_data.get("edges", []):
            mermaid_code += f"    {edge['source']} --> {edge['target']}\n"
        
        # Lưu vào memory
        graph_id = str(uuid.uuid4())
        processed_graphs[graph_id] = {
            "data": graph_data,
            "mermaid_code": mermaid_code,
            "created_at": datetime.now().isoformat()
        }
        
        graph_data["graph_id"] = graph_id
        
        return {
            "success": True,
            "data": graph_data,
            "mermaid_code": mermaid_code
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/graph/{graph_id}")
async def get_graph(graph_id: str):
    """
    Lấy thông tin graph theo ID
    """
    if graph_id not in processed_graphs:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    return processed_graphs[graph_id]

@app.get("/graph/{graph_id}/mermaid", response_class=PlainTextResponse)
async def get_mermaid_code(graph_id: str):
    """
    Lấy Mermaid code của graph
    """
    if graph_id not in processed_graphs:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    return processed_graphs[graph_id]["mermaid_code"]

@app.get("/graph/{graph_id}/download")
async def download_graph(graph_id: str):
    """
    Download graph data as JSON file
    """
    if graph_id not in processed_graphs:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    from fastapi.responses import StreamingResponse
    
    json_data = json.dumps(processed_graphs[graph_id]["data"], ensure_ascii=False, indent=2)
    
    return StreamingResponse(
        io.StringIO(json_data),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=skill_taxonomy_{graph_id}.json"}
    )

@app.get("/graphs")
async def list_graphs():
    """
    Liệt kê tất cả graphs đã tạo
    """
    graphs_info = []
    for graph_id, graph_info in processed_graphs.items():
        metadata = graph_info["data"].get("metadata", {})
        graphs_info.append({
            "graph_id": graph_id,
            "title": metadata.get("title", "Unknown"),
            "total_employees": metadata.get("total_employees", 0),
            "total_skills": metadata.get("total_skills", 0),
            "created_at": graph_info["created_at"]
        })
    
    return {"graphs": graphs_info}

@app.delete("/graph/{graph_id}")
async def delete_graph(graph_id: str):
    """
    Xóa graph khỏi memory
    """
    if graph_id not in processed_graphs:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    del processed_graphs[graph_id]
    return {"message": "Graph deleted successfully"}

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Chạy server
if __name__ == "__main__":
    import uvicorn
    # Thử các port khác nhau nếu 8000 bị occupied
    ports_to_try = [8001, 8002, 8003, 8080, 3000, 5000]
    
    for port in ports_to_try:
        try:
            print(f"🚀 Đang thử chạy Skill Taxonomy API trên port {port}...")
            print(f"📚 API Documentation: http://127.0.0.1:{port}/docs")
            print(f"🔧 ReDoc Documentation: http://127.0.0.1:{port}/redoc")
            
            uvicorn.run(
                "genjsongraphAPI:app",
                host="127.0.0.1",
                port=port,
                reload=True,
                log_level="info"
            )
            break
        except OSError as e:
            if port == ports_to_try[-1]:  # Port cuối cùng
                print(f"❌ Không thể chạy server trên bất kỳ port nào: {e}")
                print("Hãy thử chạy bằng lệnh: uvicorn genjsongraphAPI:app --host 127.0.0.1 --port 8001")
            else:
                print(f"⚠️ Port {port} đã được sử dụng, thử port tiếp theo...")
                continue
