from __future__ import annotations

from fastapi import FastAPI

from backend.llm.orchestrator import ChatPipeline
from .schemas import ChatRequest, ChatResponse, Source

app = FastAPI(title="UIT Chatbot API")

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

