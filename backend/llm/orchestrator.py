from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

from ontology.loader import get_article_by_id, load_ontology
from retrieval.text_rag.config import UIT_DISABLE_LOCAL_EMBEDDER
from retrieval.text_rag.vector_store import ChunkVectorStore

from .client import LLMClient
from .prompts import (
    ANSWER_SYSTEM_PROMPT,
    NEAR_RULE_QUERY_REWRITE_PROMPT,
    OUT_OF_SCOPE_SYSTEM_PROMPT,
)
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

    async def answer_question(
        self, question: str, conversation_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any]:
        """
        Answer a question, optionally with conversation history for multi-turn context.
        
        Args:
            question: The current question
            conversation_history: List of previous messages in format [{"role": "user|bot", "content": "..."}]
        """
        # Build full context for classification (include recent history if available)
        classification_input = question
        if conversation_history:
            # Include last 2-3 turns for context
            recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            classification_input = f"{history_text}\n\nCurrent question: {question}"
        
        q_type = await classify_question(classification_input, self.llm_client)
        if q_type == QuestionType.OUT_OF_SCOPE:
            return await self._handle_out_of_scope(question)

        # Build retrieval query: handle multi-turn and query rewriting (async)
        retrieval_query = await self._build_retrieval_query_async(question, q_type, conversation_history)
        
        # Perform hybrid retrieval (vector + lexical)
        chunks = self._hybrid_retrieve(retrieval_query, top_k=self.top_k)
        ontology_facts = self._fetch_ontology_facts(chunks)

        # For EXACT_RULE: re-rank chunks using keyword-aware scoring before selecting best rule
        if q_type == QuestionType.EXACT_RULE:
            chunks = self._rerank_chunks_by_keywords(question, chunks)
            answer_text = self._answer_exact_rule(question, chunks)
        else:
            # For NEAR_RULE and others: use LLM with context
            context = self._build_context(chunks, ontology_facts)
            # Include conversation history in context for multi-turn
            if conversation_history:
                history_context = "\n\nLịch sử hội thoại trước đó:\n"
                history_context += "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-3:]])
                context = f"{context}\n{history_context}"
            
            # Use original question for answer generation, not rewritten query
            answer_text = await self.llm_client.generate(
                ANSWER_SYSTEM_PROMPT, user_prompt=question, context=context
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
        Format: "Theo {title}{optional_path_info}: {full_rule_content}"
        """
        if not chunks:
            return "Không tìm thấy quy định phù hợp trong dữ liệu hiện có."

        # Take the best rule (first chunk, usually highest score)
        best_chunk = chunks[0]
        article_id = best_chunk.get("article_id") or ""
        clause_id = best_chunk.get("clause_id")
        text = best_chunk.get("text", "").strip()
        metadata = best_chunk.get("metadata", {})
        title = None
        if isinstance(metadata, dict):
            title = metadata.get("title")
            # Also check if title is in article_id format
            if not title and article_id:
                # Try to construct title from article_id if it looks like "Điều X"
                if "Điều" in str(article_id) or article_id.startswith("Điều"):
                    title = str(article_id)
                else:
                    # Use article_id as fallback
                    title = f"Điều {article_id}" if article_id else None

        # Build reference string: prefer title from metadata, fallback to article_id/clause_id
        ref_parts = []
        if title:
            ref_parts.append(title)
        elif article_id:
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

        # Build answer: use the FULL rule text directly (no truncation, no extra commentary)
        answer = f"Theo {ref_str}: {text}"

        return answer

    def _extract_keywords(self, question: str) -> tuple[List[str], List[str]]:
        """
        Extract domain keywords and numbers from question for keyword-aware ranking.
        Returns: (domain_keywords, numbers)
        """
        question_lower = question.lower()
        
        # Domain keywords by category
        credit_keywords = ["tín chỉ", "học kỳ", "học kỳ chính", "đăng ký", "tối đa", "tối thiểu"]
        warning_keywords = ["cảnh báo", "học vụ", "rớt môn", "học lại", "thi lại", "gpa", "điểm trung bình", "đtbhk", "đtbc"]
        graduation_keywords = ["điểm rèn luyện", "tốt nghiệp", "xét tốt nghiệp", "điều kiện tốt nghiệp"]
        conduct_keywords = ["kỷ luật", "vi phạm", "hành vi", "đạo đức"]
        
        all_keywords = credit_keywords + warning_keywords + graduation_keywords + conduct_keywords
        
        # Find matching keywords
        found_keywords = [kw for kw in all_keywords if kw in question_lower]
        
        # Extract numbers (including decimals like 3.0, 8.0)
        numbers = re.findall(r'\d+\.?\d*', question)
        
        return found_keywords, numbers
    
    def _compute_lexical_score(self, chunk_text: str, keywords: List[str], numbers: List[str]) -> float:
        """
        Compute lexical overlap score for a chunk based on keywords and numbers.
        Returns a score where higher = better match.
        """
        chunk_lower = chunk_text.lower()
        score = 0.0
        
        # Score for keyword matches
        for keyword in keywords:
            if keyword in chunk_lower:
                score += 1.0
                # Bonus for exact phrase match
                if f" {keyword} " in chunk_lower or chunk_lower.startswith(keyword) or chunk_lower.endswith(keyword):
                    score += 0.5
        
        # Score for number matches (important for exact rules)
        for num in numbers:
            if num in chunk_text:  # Case-sensitive for numbers
                score += 2.0  # Numbers are very important for exact rules
        
        return score
    
    def _rerank_chunks_by_keywords(
        self, question: str, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks for EXACT_RULE questions using keyword-aware scoring.
        Combines lexical score with original similarity score.
        """
        if not chunks:
            return chunks
        
        keywords, numbers = self._extract_keywords(question)
        
        # If no keywords found, return original order
        if not keywords and not numbers:
            return chunks
        
        # Compute lexical scores for all chunks
        scored_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            lexical_score = self._compute_lexical_score(chunk_text, keywords, numbers)
            
            # Get original similarity score (default to 0.5 if missing)
            original_score = chunk.get("score", 0.5)
            
            # Combine scores: lexical_score is weighted more heavily for EXACT_RULE
            # alpha = 1.0 means lexical score has equal weight to similarity
            combined_score = original_score + 1.0 * lexical_score
            
            scored_chunks.append({
                **chunk,
                "lexical_score": lexical_score,
                "combined_score": combined_score,
            })
        
        # Sort by combined_score descending, then by lexical_score descending
        scored_chunks.sort(key=lambda x: (x["combined_score"], x["lexical_score"]), reverse=True)
        
        return scored_chunks
    
    def _normalize_vietnamese(self, text: str) -> str:
        """
        Normalize Vietnamese text: lowercase, remove accents (NFD + filter Mn), strip spaces.
        Example: "Cảnh báo học vụ" → "canh bao hoc vu"
        """
        # Lowercase
        text = text.lower()
        # Remove accents using NFD decomposition and filter out combining marks
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        # Strip spaces
        text = " ".join(text.split())
        return text
    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """
        Extract keywords from query by removing stopwords.
        Returns list of normalized keywords.
        """
        # Vietnamese stopwords (common function words)
        stopwords = {
            "la", "co", "thi", "em", "anh", "toi", "ban", "the", "mot", "cua", "va", "hoac", "nhung",
            "duoc", "se", "da", "dang", "ve", "cho", "voi", "trong", "tren", "duoi", "sao", "gi",
            "nao", "khi", "neu", "thi", "vay", "the", "con", "hay", "ma", "de", "nen", "khong",
            "khong", "chua", "chung", "nhieu", "it", "rat", "qua", "nhu", "nhu", "cung", "cac"
        }
        
        normalized = self._normalize_vietnamese(query)
        tokens = normalized.split()
        keywords = [t for t in tokens if t not in stopwords and len(t) > 1]
        return keywords
    
    def _compute_hybrid_lexical_score(
        self, chunk_text: str, keywords: List[str]
    ) -> float:
        """
        Compute lexical overlap score for hybrid retrieval.
        Includes bonuses for common bigrams.
        """
        normalized_content = self._normalize_vietnamese(chunk_text)
        score = 0.0
        
        # Basic keyword matching
        for kw in keywords:
            if kw in normalized_content:
                score += 1.0
        
        # Bigram bonuses
        normalized_lower = normalized_content.lower()
        if "canh bao" in normalized_lower and "hoc vu" in normalized_lower:
            score += 2.0
        if ("khoa luan" in normalized_lower or "kltn" in normalized_lower) and (
            "tot nghiep" in normalized_lower or "nop" in normalized_lower
        ):
            score += 2.0
        if "tin chi" in normalized_lower:
            score += 1.0
        if "diem ren luyen" in normalized_lower:
            score += 2.0
        
        return score
    
    def _hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: vector search + lexical ranking.
        Generic implementation that works for any question.
        """
        # Step 1: Vector search (or text search if embedding disabled)
        chunks = self.vector_store.search(query, self.embedder, top_k=top_k * 2)  # Get more candidates
        
        if not chunks:
            return []
        
        # Step 2: Extract keywords from query
        keywords = self._extract_keywords_from_query(query)
        
        # Step 3: Compute lexical scores and combine with embedding scores
        scored_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            lexical_score = self._compute_hybrid_lexical_score(chunk_text, keywords)
            embedding_score = chunk.get("score", 0.5)
            
            # Combine scores: alpha = 0.75 (lexical has significant weight)
            combined_score = embedding_score + 0.75 * lexical_score
            
            scored_chunks.append({
                **chunk,
                "lexical_score": lexical_score,
                "combined_score": combined_score,
            })
        
        # Step 4: Ranking rule
        # If at least one candidate has lexical_score > 0, sort by combined_score
        # Otherwise, keep original embedding ranking
        has_lexical_matches = any(c["lexical_score"] > 0 for c in scored_chunks)
        
        if has_lexical_matches:
            scored_chunks.sort(key=lambda x: (x["combined_score"], x["lexical_score"]), reverse=True)
        else:
            # Keep original embedding ranking
            scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_chunks[:top_k]
    
    async def _rewrite_query_for_regulations(self, user_question: str) -> str:
        """
        Use LLM to rewrite an informal student question into a formal, regulation-oriented query.
        The result will be used as the retrieval query. Must be generic, no hard-coded content.
        
        Example (for illustration only, not hard-coded):
        - Input: "Em rớt 3 môn thì có bị sao không ạ?"
        - Output: "Quy định về cảnh báo học vụ và xử lý khi sinh viên rớt nhiều môn, điểm trung bình học kỳ thấp."
        """
        rewritten = await self.llm_client.generate(
            NEAR_RULE_QUERY_REWRITE_PROMPT,
            user_prompt=user_question,
            context=""
        )
        # Clean up: remove quotes, extra whitespace
        rewritten = rewritten.strip().strip('"').strip("'")
        return rewritten
    
    async def _build_retrieval_query_async(
        self,
        question: str,
        q_type: QuestionType,
        conversation_history: List[Dict[str, str]] | None,
    ) -> str:
        """
        Build retrieval query with query rewriting (async) and multi-turn context.
        """
        # Step 1: Query rewriting for NEAR_RULE
        if q_type == QuestionType.NEAR_RULE:
            rewritten = await self._rewrite_query_for_regulations(question)
            current_query = rewritten
        else:
            current_query = question
        
        # Step 2: Multi-turn query building
        normalized_question = self._normalize_vietnamese(question)
        discourse_markers = ["vay", "the", "con", "neu vay", "vay thi"]
        starts_with_discourse = any(
            normalized_question.startswith(marker) for marker in discourse_markers
        )
        
        if starts_with_discourse and conversation_history:
            # Find last user question
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    prev_question = msg.get("content", "")
                    if prev_question:
                        # Combine previous question with current query
                        return f"{prev_question} {current_query}"
        
        return current_query
    
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

