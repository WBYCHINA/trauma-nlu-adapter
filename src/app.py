import json
import os
import sys
import ssl
import torch

ssl._create_default_https_context = ssl._create_unverified_context

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

from src.core.nlu_service import NLUService
from src.core.knowledge_base import KnowledgeBase

app = FastAPI(title="智能外伤急救指导 AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[INFO] Initializing NLU service...")
nlu_service = NLUService()
print("[INFO] NLU service initialized")

print("[INFO] Loading knowledge base...")
knowledge_base = KnowledgeBase()
print("[INFO] Knowledge base loaded")

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    intent: Optional[Dict[str, Any]] = None
    guide: Optional[Dict[str, Any]] = None
    is_emergency: bool = False

@app.get("/")
async def root():
    return {"message": "智能外伤急救指导 AI 服务已启动"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        intent = nlu_service.classify(request.message)
        guide = knowledge_base.search(intent)
        
        reply = f"已识别：大类型={intent['type']}，小类型={intent['type_digital']}，部位={intent['location']}"
        
        if guide.get("steps"):
            reply += "\n\n急救方案："
            for i, step in enumerate(guide["steps"], 1):
                reply += f"\n{i}. {step}"
        
        response = ChatResponse(
            reply=reply,
            intent=intent,
            guide=guide,
            is_emergency=intent.get("type") == "开放式"
        )
        
        return response
    except Exception as e:
        print(f"Chat error: {e}")
        return ChatResponse(
            reply="抱歉，我现在有点忙，请稍后再试。",
            intent=None,
            guide=None,
            is_emergency=False
        )

@app.post("/api/classify")
async def classify(request: ChatRequest):
    try:
        intent = nlu_service.classify(request.message)
        return {"intent": intent}
    except Exception as e:
        print(f"Classify error: {e}")
        return {"intent": None}

@app.post("/api/detect/type")
async def detect_type(request: ChatRequest):
    try:
        injury_type = nlu_service.detect_type(request.message)
        return {"type": injury_type}
    except Exception as e:
        print(f"Detect type error: {e}")
        return {"type": "未知"}

@app.post("/api/detect/subtype")
async def detect_subtype(request: ChatRequest):
    try:
        subtype = nlu_service.detect_subtype(request.message)
        return {"type_digital": subtype}
    except Exception as e:
        print(f"Detect subtype error: {e}")
        return {"type_digital": "未知"}

@app.post("/api/detect/location")
async def detect_location(request: ChatRequest):
    try:
        location = nlu_service.detect_location(request.message)
        return {"location": location}
    except Exception as e:
        print(f"Detect location error: {e}")
        return {"location": "未知"}

@app.post("/api/guide")
async def get_guide(request: Dict[str, Any]):
    try:
        guide = knowledge_base.search(request)
        return {"guide": guide}
    except Exception as e:
        print(f"Guide error: {e}")
        return {"guide": None}

@app.post("/api/reset")
async def reset_session():
    return {"message": "会话已重置"}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_text()
            message = json.loads(data).get("message", "")
            
            intent = nlu_service.classify(message)
            guide = knowledge_base.search(intent)
            
            response = {
                "reply": f"已识别：大类型={intent['type']}，小类型={intent['type_digital']}，部位={intent['location']}",
                "intent": intent,
                "guide": guide,
                "is_emergency": intent.get("type") == "开放式"
            }
            
            await websocket.send_text(json.dumps(response))
        except Exception as e:
            print(f"WebSocket error: {e}")
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)