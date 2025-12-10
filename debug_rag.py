"""
Debug helper to inspect RAG retrieval before EXACT_RULE filtering.

Usage:
    python debug_rag.py
"""

from __future__ import annotations

import asyncio

from backend.llm.orchestrator import ChatPipeline
from backend.llm.question_types import QuestionType


async def main() -> None:
    pipeline = ChatPipeline()
    question = "Điều kiện để bị cảnh báo học vụ là gì?"
    print("Question:", question)

    # Build retrieval query (includes rewriting & multi-turn logic; no history here)
    retrieval_query = await pipeline._build_retrieval_query_async(
        question, QuestionType.EXACT_RULE, conversation_history=None
    )
    print("Retrieval query:", retrieval_query)

    # Retrieve raw chunks (hybrid vector + lexical) before any EXACT_RULE selection
    chunks = pipeline._hybrid_retrieve(retrieval_query, top_k=10)
    print(f"Retrieved {len(chunks)} chunks")
    for i, ch in enumerate(chunks[:10], start=1):
        print(f"--- Chunk #{i} ---")
        print("article_id:", ch.get("article_id"))
        print("clause_id :", ch.get("clause_id"))
        md = ch.get("metadata")
        if isinstance(md, dict):
            print("title    :", md.get("title") or md.get("heading"))
        print("score    :", ch.get("score"), "lexical_score:", ch.get("lexical_score"), "combined:", ch.get("combined_score"))
        text = ch.get("text") or ch.get("content") or ""
        print("excerpt  :", text[:200].replace("\n", " "))
        print()


if __name__ == "__main__":
    asyncio.run(main())

