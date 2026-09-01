"""Adapter over ChatPipeline, in place of SchemaGraph's LibreChat/workspace clients.

We don't have a remote chat frontend or a per-run workspace to ingest into -
the corpus is already indexed (retrieval.text_rag.build_index /
retrieval.src.retrieval.build_graph_index), and ChatPipeline is importable
directly. So `collect` calls it in-process rather than over HTTP: no chat
API, no admin API, and no RAG-trace-file polling, since the retrieval
evidence (`debug.text_hits` / `debug.graph_hits`) comes back in the same
call that produced the answer.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Make `backend/` importable (same convention as tests/conftest.py at the
# repo root: bare `llm`, `retrieval`, `ontology` resolve against backend/'s
# packages, matching how the app is actually served).
_BACKEND_DIR = Path(__file__).resolve().parents[3] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


@dataclass
class ChatCompletion:
    """One question's outcome - enough for collect to build a ResponseRow."""

    answer: str | None
    question_type: str | None
    sources: list[dict[str, Any]]
    latency_ms: int
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class UitChatbotClient:
    """Thin wrapper around a shared ChatPipeline instance.

    One instance per collect run (loading embedders/graph/ontology is not
    cheap), reused across every question.
    """

    def __init__(self, pipeline: Any = None) -> None:
        if pipeline is None:
            from llm.orchestrator import ChatPipeline

            pipeline = ChatPipeline()
        self.pipeline = pipeline

    async def complete(
        self, question: str, conversation_history: list[dict[str, str]] | None = None
    ) -> ChatCompletion:
        started = time.monotonic()
        try:
            result = await self.pipeline.answer_question(
                question, conversation_history=conversation_history, debug=True
            )
        except Exception as exc:  # noqa: BLE001 - recorded per question, run continues
            return ChatCompletion(
                answer=None,
                question_type=None,
                sources=[],
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        debug = result.get("debug", {}) or {}
        sources = self._flatten_sources(debug.get("text_hits") or [], debug.get("graph_hits") or [])
        return ChatCompletion(
            answer=result.get("answer"),
            question_type=result.get("question_type"),
            sources=sources,
            latency_ms=latency_ms,
            raw=result,
        )

    @staticmethod
    def _flatten_sources(text_hits: list[dict], graph_hits: list[dict]) -> list[dict[str, Any]]:
        """Text chunks and graph facts both become RAGAS "retrieved_contexts" -
        RAGAS just wants the text that was available to ground the answer."""
        sources: list[dict[str, Any]] = []
        for rank, hit in enumerate(text_hits, start=1):
            sources.append(
                {
                    "chunk_id": hit.get("chunk_id"),
                    "article_id": hit.get("article_id"),
                    "content": hit.get("text") or "",
                    "score": hit.get("score"),
                    "rank": rank,
                    "route": "text",
                }
            )
        for rank, hit in enumerate(graph_hits, start=1):
            fact = f"{hit.get('subject', '')} — {hit.get('predicate', '')} — {hit.get('object', '')}"
            sources.append(
                {
                    "chunk_id": hit.get("chunk_id"),
                    "article_id": hit.get("article_id"),
                    "content": (hit.get("text") or fact),
                    "score": hit.get("score"),
                    "rank": rank,
                    "route": "graph",
                }
            )
        return sources
