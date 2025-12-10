from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.llm.orchestrator import ChatPipeline
from .schemas import ChatRequest, ChatResponse, Source

app = FastAPI(title="UIT Chatbot API")

# CORS middleware: allow all origins for demo mode
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared pipeline
pipeline = ChatPipeline()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Convert ChatMessage to dict format expected by pipeline
    history = None
    if req.conversation_history:
        history = [{"role": msg.role, "content": msg.content} for msg in req.conversation_history]
    
    result = await pipeline.answer_question(req.question, conversation_history=history)
    sources = [
        Source(article_id=s.get("article_id"), clause_id=s.get("clause_id"), text=s.get("text", ""))
        for s in result.get("sources", [])
    ]
    return ChatResponse(answer=result["answer"], question_type=result["question_type"], sources=sources)

