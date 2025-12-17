from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "bot"
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_history: List[ChatMessage] | None = None


class Source(BaseModel):
    article_id: Optional[str] = None
    title: Optional[str] = None
    clause_id: Optional[str] = None
    text: str
    doc_id: str
    doc_title: Optional[str] = None
    so_hieu: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    question_type: str
    sources: List[Source]
    # Optional debug information for tracing routing and retrieval decisions.
    debug: Optional[dict] = None

