from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from llm.gpt_client import GPTLLMClient
from llm.orchestrator import ChatPipeline
from api.schemas import ChatRequest, ChatResponse, Source

logging.basicConfig(
    level=logging.INFO,  # change to DEBUG if you want more detail
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("uitchatbot.api")

app = FastAPI(title="UIT Chatbot API (GPT)")

# CORS middleware: allow all origins for demo mode
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type"],
)

logger.info("Initializing GPT client and chat pipeline")
logger.warning("PIPELINE_VERSION=diagram-aligned-v1")
gpt_client = GPTLLMClient()
pipeline = ChatPipeline(llm_client=gpt_client)

DOC_MAPPING: dict[str, str] = {}

csv_path = Path(__file__).parent / "data" / "doc_sources.csv"
logger.info("Loading document mapping from %s", csv_path)

if csv_path.exists():
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row.get("doc_id") or row.get("\ufeffdoc_id")
            file_name = row.get("file_name")
            if doc_id and file_name:
                DOC_MAPPING[doc_id.strip()] = file_name.strip()

logger.info("Loaded %d document mappings", len(DOC_MAPPING))

@app.get("/health")
async def health():
    logger.debug("Health check requested")
    return {"status": "ok"}

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """Serve PDF file by doc_id for inline viewing"""

    logger.info("[PDF API] Request received: doc_id=%s", doc_id)

    if doc_id not in DOC_MAPPING:
        logger.error(
            "[PDF API] doc_id not found: %s (available=%s...)",
            doc_id,
            list(DOC_MAPPING.keys())[:10],
        )
        raise HTTPException(status_code=404, detail="Document not found")

    filename = DOC_MAPPING[doc_id]
    logger.info("[PDF API] Mapped doc_id=%s → filename=%s", doc_id, filename)

    pdf_path = Path(__file__).parent / "pdfs"
    logger.debug("[PDF API] Searching PDFs in path: %s", pdf_path)

    for pdf_file in pdf_path.rglob("*.pdf"):
        if pdf_file.name == filename:
            logger.info("[PDF API] Found PDF file: %s", pdf_file)
            logger.info("[PDF API] Serving PDF inline")
            return FileResponse(
                pdf_file,
                media_type="application/pdf",
                headers={"Content-Disposition": f"inline; filename={filename}"},
            )

    logger.error(
        "[PDF API] PDF file not found: filename=%s path=%s",
        filename,
        pdf_path,
    )
    raise HTTPException(status_code=404, detail="PDF file not found")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    force_intent: Optional[str] = Query(
        default=None,
        description="Optional override for intent routing. One of: EXACT_RULE, NEAR_RULE, OUT_OF_SCOPE",
    ),
):
    logger.info("Chat request received")

    history = None
    if req.conversation_history:
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in req.conversation_history
        ]
        logger.debug("Conversation history length: %d", len(history))

    result = await pipeline.answer_question(
        req.question,
        conversation_history=history,
        debug=True,
        force_intent=force_intent,
    )

    sources = [
        Source(
            article_id=s.get("article_id"),
            title=s.get("title"),
            clause_id=s.get("clause_id"),
            text=s.get("text", ""),
            doc_id=s.get("doc_id"),
            doc_title=s.get("doc_title"),
            so_hieu=s.get("so_hieu"),
        )
        for s in result.get("sources", [])
    ]

    # Handle different response types (answer vs reply for clarification)
    answer_text = result.get("answer") or result.get("reply", "")

    # Attach debug info if available
    debug_info = result.get("debug") or {}
    if force_intent:
        debug_info["intent_forced"] = True
        debug_info["forced_intent"] = force_intent

    logger.info(
        "Chat response generated (question_type=%s, sources=%d)",
        result.get("question_type"),
        len(sources),
    )

    return ChatResponse(
        answer=answer_text,
        question_type=result["question_type"],
        sources=sources,
        debug=debug_info or None,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

# Example curl commands for demo-stable routing:
# curl -X POST "http://127.0.0.1:10000/chat?force_intent=EXACT_RULE" -H "Content-Type: application/json" -d "{\"question\":\"điều kiện tốt nghiệp là gì\"}"
# curl -X POST "http://127.0.0.1:10000/chat?force_intent=NEAR_RULE" -H "Content-Type: application/json" -d "{\"question\":\"em rớt 1 môn có sao không\"}"
# curl -X POST "http://127.0.0.1:10000/chat?force_intent=OUT_OF_SCOPE" -H "Content-Type: application/json" -d "{\"question\":\"canteen hôm nay bán gì\"}"
