from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ontology.loader import get_article_by_id, load_ontology
from retrieval.text_rag.config import UIT_DISABLE_LOCAL_EMBEDDER
from retrieval.text_rag.vector_store import ChunkVectorStore

from .client import LLMClient
from .prompts import ANSWER_SYSTEM_PROMPT, OUT_OF_SCOPE_SYSTEM_PROMPT
from .question_classifier import classify_question
from .question_types import QuestionType

# Only import TextEmbedder when local embedding is enabled
if not UIT_DISABLE_LOCAL_EMBEDDER:
    from retrieval.text_rag.embeddings import TextEmbedder


class ChatPipeline:
    def __init__(
        self,
        ontology_path: str | None = None,
        vector_db_path: str | None = None,
        llm_client: Optional[LLMClient] = None,
        embedder=None,
        vector_store: Optional[ChunkVectorStore] = None,
        ontology_graph=None,
    ) -> None:
        self.llm_client = llm_client or LLMClient()
        self.ontology_graph = ontology_graph or load_ontology(
            ontology_path or os.getenv("UIT_TTL_PATH", "ontology/uit_regulations.ttl")
        )

        # Only create embedder if local embedding is enabled
        if UIT_DISABLE_LOCAL_EMBEDDER:
            self.embedder = None
        else:
            if embedder is None:
                from retrieval.text_rag.embeddings import TextEmbedder

                embedder = TextEmbedder()
            self.embedder = embedder

        self.vector_store = vector_store or ChunkVectorStore(
            vector_db_path or os.getenv("UIT_VECTOR_DB", "retrieval/text_rag/vector_store.db"),
            disable_local_embedder=UIT_DISABLE_LOCAL_EMBEDDER,
        )
        self.top_k = int(os.getenv("UIT_RAG_TOP_K", "5"))

    async def answer_question(self, question: str) -> Dict[str, Any]:
        q_type = await classify_question(question, self.llm_client)
        if q_type == QuestionType.OUT_OF_SCOPE:
            return await self._handle_out_of_scope(question)

        normalized_question = question  # placeholder for future rewrite if needed
        # Pass embedder only if it exists (when local embedding is enabled)
        chunks = self.vector_store.search(normalized_question, self.embedder, top_k=self.top_k)
        ontology_facts = self._fetch_ontology_facts(chunks)

        # For EXACT_RULE: bypass LLM and answer directly from retrieved rules
        if q_type == QuestionType.EXACT_RULE:
            answer_text = self._answer_exact_rule(question, chunks)
        else:
            # For NEAR_RULE and others: use LLM with context
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

    def _answer_exact_rule(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a direct answer for EXACT_RULE questions using retrieved rules only.
        Do NOT call the LLM here. Returns a fallback message if no rules found.
        """
        if not chunks:
            return "Không tìm thấy quy định phù hợp trong dữ liệu hiện có."

        # Take the best rule (first chunk, usually highest score)
        best_chunk = chunks[0]
        article_id = best_chunk.get("article_id") or ""
        clause_id = best_chunk.get("clause_id")
        text = best_chunk.get("text", "").strip()
        metadata = best_chunk.get("metadata", {})
        title = metadata.get("title") if isinstance(metadata, dict) else None

        # Build reference string
        ref_parts = []
        if title:
            ref_parts.append(title)
        elif article_id:
            # Try to extract "Điều X" from article_id if it's a readable format
            # Otherwise use article_id as-is
            if "Điều" in str(article_id) or article_id.startswith("Điều"):
                ref_parts.append(str(article_id))
            else:
                ref_parts.append(f"Điều {article_id}")
        if clause_id:
            if "Khoản" in str(clause_id) or clause_id.startswith("Khoản"):
                ref_parts.append(str(clause_id))
            else:
                ref_parts.append(f"Khoản {clause_id}")

        ref_str = ", ".join(ref_parts) if ref_parts else "quy định"

        # Build answer: use the rule text directly, optionally truncated if too long
        if len(text) > 500:
            # Truncate to first 500 chars and add ellipsis
            truncated = text[:500].rsplit(".", 1)[0] + "."
            answer = f"Theo {ref_str}: {truncated}"
        else:
            answer = f"Theo {ref_str}: {text}"

        # Add note about consulting full regulation
        answer += "\n\nBạn nên tham khảo đầy đủ điều khoản này trong quy chế chính thức để nắm rõ hơn."

        return answer

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

