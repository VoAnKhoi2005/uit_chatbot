from __future__ import annotations

import csv
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
    expose_headers=["Content-Disposition", "Content-Type"],
)

# Initialize shared pipeline
pipeline = ChatPipeline()

# Load doc_id to filename mapping
DOC_MAPPING = {}
csv_path = Path(__file__).parent.parent.parent / "data" / "doc_sources.csv"
if csv_path.exists():
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("doc_id") or row.get("\ufeffdoc_id")
            file_name = row.get("file_name")
            if doc_id and file_name:
                DOC_MAPPING[doc_id.strip()] = file_name.strip()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/document/{doc_id}")
async def get_pdf(doc_id: str):
    """Serve PDF file by doc_id for inline viewing"""
    print(f"[PDF API] Received request for doc_id: {doc_id}")
    
    if doc_id not in DOC_MAPPING:
        print(f"[PDF API] Error: doc_id '{doc_id}' not found in mapping")
        print(f"[PDF API] Available doc_ids: {list(DOC_MAPPING.keys())[:10]}...")
        return {"error": "Document not found"}, 404
    
    filename = DOC_MAPPING[doc_id]
    print(f"[PDF API] Mapped to filename: {filename}")
    
    pdf_path = Path(__file__).parent.parent.parent / "pdfs"
    print(f"[PDF API] Searching in path: {pdf_path}")
    
    # Search for the file in all subdirectories
    for pdf_file in pdf_path.rglob("*.pdf"):
        if pdf_file.name == filename:
            print(f"[PDF API] Found PDF file at: {pdf_file}")
            print(f"[PDF API] Serving file with Content-Disposition: inline")
            return FileResponse(
                pdf_file,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"inline; filename={filename}"
                }
            )
    
    print(f"[PDF API] Error: PDF file '{filename}' not found in {pdf_path}")
    return {"error": "PDF file not found"}, 404


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Convert ChatMessage to dict format expected by pipeline
    history = None
    if req.conversation_history:
        history = [{"role": msg.role, "content": msg.content} for msg in req.conversation_history]
    
    result = await pipeline.answer_question(req.question, conversation_history=history)
    sources = [
        Source(
            article_id=s.get("article_id"),
            clause_id=s.get("clause_id"),
            text=s.get("text", ""),
            doc_id=s.get("doc_id"),
            doc_title=s.get("doc_title"),
            so_hieu=s.get("so_hieu")
        )
        for s in result.get("sources", [])
    ]
    return ChatResponse(answer=result["answer"], question_type=result["question_type"], sources=sources)

