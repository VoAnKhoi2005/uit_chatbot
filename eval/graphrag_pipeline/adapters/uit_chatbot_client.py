"""Adapter over the backend's HTTP `/chat` API, in place of SchemaGraph's
LibreChat client - same idea (a chat API over HTTP), pointed at our own
backend instead. `collect` runs entirely outside Docker/the backend's own
process: it never imports anything from `backend/`, so this venv only ever
needs eval/graphrag_pipeline/requirements.txt installed, not the backend's
(fastapi, faiss-cpu, rdflib, ...) - the backend just needs to be reachable
over HTTP (e.g. the Docker Compose service on localhost:10000).

The backend's POST /chat always returns both `sources` (the evidence the
answer was actually grounded on - empty for OUT_OF_SCOPE, since none was
used) and `debug.text_hits`/`debug.graph_hits` (everything Path 1/Path 2
retrieved *before* routing/domain-gating filtered or discarded it, which can
be nonempty even when `sources` is empty). RAGAS's contexts come from
`sources` - what the LLM actually saw - not the broader debug hits, so an
OUT_OF_SCOPE item is correctly recorded with no contexts instead of
irrelevant retrieval noise the pipeline itself decided not to use.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_BACKEND_URL = "http://localhost:10000"


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
    """Thin async httpx wrapper over the backend's `POST /chat`."""

    def __init__(self, base_url: str | None = None, timeout_s: float = 120.0) -> None:
        self.base_url = (base_url or os.getenv("EVAL_BACKEND_URL", DEFAULT_BACKEND_URL)).rstrip("/")
        # Connect timeout short, read long: an answer that takes a while to
        # generate is slow, not broken, and a cut-off call would be recorded
        # as an error on a question that was never actually answered.
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=httpx.Timeout(10.0, read=timeout_s)
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self, question: str, conversation_history: list[dict[str, str]] | None = None
    ) -> ChatCompletion:
        started = time.monotonic()
        payload: dict[str, Any] = {"question": question}
        if conversation_history:
            payload["conversation_history"] = conversation_history

        try:
            response = await self._client.post("/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - recorded per question, run continues
            return ChatCompletion(
                answer=None,
                question_type=None,
                sources=[],
                latency_ms=int((time.monotonic() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

        latency_ms = int((time.monotonic() - started) * 1000)
        sources = self._flatten_sources(data.get("sources") or [])
        return ChatCompletion(
            answer=data.get("answer"),
            question_type=data.get("question_type"),
            sources=sources,
            latency_ms=latency_ms,
            raw=data,
        )

    @staticmethod
    def _flatten_sources(sources: list[dict]) -> list[dict[str, Any]]:
        """`sources` is the API's `List[Source]` (article_id, title, clause_id,
        text, doc_id, doc_title, so_hieu) - the evidence actually used to
        ground the answer, which is what RAGAS's "retrieved_contexts" means."""
        out: list[dict[str, Any]] = []
        for rank, hit in enumerate(sources, start=1):
            out.append(
                {
                    "article_id": hit.get("article_id"),
                    "content": hit.get("text") or "",
                    "rank": rank,
                }
            )
        return out
