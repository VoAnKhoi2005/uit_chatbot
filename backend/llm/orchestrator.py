from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ontology.loader import get_article_by_id, load_ontology
from retrieval.text_rag.embeddings import TextEmbedder
from retrieval.text_rag.vector_store import ChunkVectorStore

from .client import LLMClient
from .prompts import ANSWER_SYSTEM_PROMPT, OUT_OF_SCOPE_SYSTEM_PROMPT
from .question_classifier import classify_question
from .question_types import QuestionType


class ChatPipeline:
    def __init__(
        self,
        ontology_path: str | None = None,
        vector_db_path: str | None = None,
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[TextEmbedder] = None,
        vector_store: Optional[ChunkVectorStore] = None,
        ontology_graph=None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.ontology_graph = ontology_graph or load_ontology(
            ontology_path or os.getenv("UIT_TTL_PATH", "ontology/uit_regulations.ttl")
        )
        self.embedder = embedder or TextEmbedder()
        self.vector_store = vector_store or ChunkVectorStore(
            vector_db_path or os.getenv("UIT_VECTOR_DB", "retrieval/text_rag/vector_store.db")
        )
        self.top_k = int(os.getenv("UIT_RAG_TOP_K", "5"))

    async def answer_question(self, question: str) -> Dict[str, Any]:
        q_type = await classify_question(question, self.llm_client)
        if q_type == QuestionType.OUT_OF_SCOPE:
            return await self._handle_out_of_scope(question)

        normalized_question = question  # placeholder for future rewrite if needed
        chunks = self.vector_store.search(normalized_question, self.embedder, top_k=self.top_k)
        ontology_facts = self._fetch_ontology_facts(chunks)
        context = self._build_context(chunks, ontology_facts)

        answer_text = await self.llm_client.generate(
            ANSWER_SYSTEM_PROMPT, user_prompt=normalized_question, context=context
        )

        return {
            "answer": answer_text,
            "question_type": q_type.value,
            "sources": [
                {
                    "article_id": c.get("article_id"),
                    "clause_id": c.get("clause_id"),
                    "text": c.get("text"),
                }
                for c in chunks
            ],
            "ontology_facts": ontology_facts,
        }

    async def _handle_out_of_scope(self, question: str) -> Dict[str, Any]:
        answer_text = await self.llm_client.generate(
            OUT_OF_SCOPE_SYSTEM_PROMPT, user_prompt=question, context=""
        )
        return {
            "answer": answer_text,
            "question_type": QuestionType.OUT_OF_SCOPE.value,
            "sources": [],
            "ontology_facts": [],
        }

    def _fetch_ontology_facts(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for c in chunks:
            art = c.get("article_id")
            if not art or art in seen:
                continue
            seen.add(art)
            try:
                rows = get_article_by_id(self.ontology_graph, art)
            except Exception:
                rows = []
            for r in rows:
                facts.append({"article_id": art, "title": r.get("title"), "text": r.get("text")})
        return facts

    def _build_context(self, chunks: List[Dict[str, Any]], ontology_facts: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        if chunks:
            lines.append("Các đoạn trích liên quan:")
            for c in chunks:
                lines.append(
                    f"- Article: {c.get('article_id')}, Clause: {c.get('clause_id')}, Text: {c.get('text')}"
                )
        if ontology_facts:
            lines.append("Dữ kiện ontology:")
            for f in ontology_facts:
                lines.append(f"- Article: {f.get('article_id')}, Title: {f.get('title')}, Text: {f.get('text')}")
        return "\n".join(lines)

