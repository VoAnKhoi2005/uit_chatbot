from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    article_id: Optional[str] = None
    clause_id: Optional[str] = None
    text: str


class ChatResponse(BaseModel):
    answer: str
    question_type: str
    sources: List[Source]

