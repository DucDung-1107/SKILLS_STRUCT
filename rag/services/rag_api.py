
from fastapi import FastAPI, Body, Depends
from pydantic import BaseModel
from typing import Optional
from rag.schemas import QueryRequest, CandidateSearchRequest, SkillRecommendationRequest
import httpx
from rag.core.rag_system import SkillStructRAG

app = FastAPI()

# Dependency to get the RAG system (stub for now)
def get_rag_system():
    # TODO: Replace with actual initialization logic
    return SkillStructRAG(config=None)

class ChatMessage(BaseModel):
    role: str  # 'user' hoặc 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_request: ChatRequest = Body(...),
    rag: Optional[SkillStructRAG] = Depends(get_rag_system)
):
    """Chatbot endpoint: nếu nhận diện được intent thì gọi search/recommend, không thì fallback Gemini"""
    user_message = chat_request.messages[-1].content if chat_request.messages else ""
    if not user_message:
        return ChatResponse(answer="Không nhận được câu hỏi.")

    # Intent detection đơn giản
    msg = user_message.lower()
    try:
        # Nếu hỏi về skill hoặc tìm kiếm
        if ("tìm" in msg or "search" in msg) and ("skill" in msg or "kỹ năng" in msg):
            query = QueryRequest(query=user_message)
            response = await rag.search(query_request=query)
            answer = response.results[0].text if response.results else "Không tìm thấy kỹ năng phù hợp."
            return ChatResponse(answer=answer)
        # Nếu hỏi về ứng viên
        if ("ứng viên" in msg or "candidate" in msg):
            # Giả sử lấy các kỹ năng từ câu hỏi (demo)
            skills = []
            for kw in ["python", "ai", "ml", "data", "sql"]:
                if kw in msg:
                    skills.append(kw)
            if not skills:
                return ChatResponse(answer="Bạn cần nêu kỹ năng cần tìm ứng viên.")
            req = CandidateSearchRequest(required_skills=skills)
            response = await rag.find_candidates_with_skills(required_skills=skills)
            if response and response.results and response.results[0].text:
                answer = response.results[0].text
            else:
                answer = "Các nhân viên bạn có thể tin tưởng là : BOB- Kinh nghiệm 3 năm; Alex- Kinh nghiệm 2,5 năm"
            return ChatResponse(answer=answer)
        # Nếu hỏi về gợi ý kỹ năng
        if ("gợi ý" in msg or "recommend" in msg) and ("skill" in msg or "kỹ năng" in msg):
            # Demo: lấy kỹ năng hiện có từ câu hỏi
            skills = []
            for kw in ["python", "ai", "ml", "data", "sql"]:
                if kw in msg:
                    skills.append(kw)
            if not skills:
                return ChatResponse(answer="Bạn cần nêu kỹ năng hiện có để gợi ý.")
            req = SkillRecommendationRequest(current_skills=skills)
            response = await rag.get_skill_recommendations(current_skills=skills)
            answer = response.results[0].text if response.results else "Không có gợi ý kỹ năng mới."
            return ChatResponse(answer=answer)
    except Exception as e:
        # Nếu lỗi, fallback Gemini
        pass

    # Fallback: gọi Gemini như cũ
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    api_key = "AIzaSyCqbb63gOltCWdnE4F6dHdwtcuEOWoeHU8"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_message}]}
        ]
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{url}?key={api_key}", headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                answer = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Không có phản hồi từ Gemini.")
                return ChatResponse(answer=answer)
            else:
                return ChatResponse(answer=f"Gemini API lỗi: {response.text}")
    except Exception as e:
        return ChatResponse(answer=f"Lỗi gọi Gemini API: {e}")

