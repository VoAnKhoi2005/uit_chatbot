from __future__ import annotations

from fastapi import FastAPI

from backend.llm.orchestrator import ChatPipeline
from .schemas import ChatRequest, ChatResponse, Source


from fastapi.middleware.cors import CORSMiddleware
# ... các import khác


app = FastAPI(title="UIT Chatbot API")


# ===== CORS cho FE Vercel + local dev =====
origins = [
    "https://uitchatbotfe.vercel.app",  # FE trên Vercel
    "http://localhost:5173",            # FE local (Vite)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # cho phép tất cả method: GET, POST, OPTIONS, ...
    allow_headers=["*"],   # cho phép tất cả headers
)
# ===== hết phần CORS =====git 

# Initialize shared pipeline
pipeline = ChatPipeline()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = await pipeline.answer_question(req.question)
    sources = [
        Source(article_id=s.get("article_id"), clause_id=s.get("clause_id"), text=s.get("text", ""))
        for s in result.get("sources", [])
    ]
    return ChatResponse(answer=result["answer"], question_type=result["question_type"], sources=sources)

