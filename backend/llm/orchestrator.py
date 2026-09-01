from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.client import LLMClient
from llm.question_classifier import classify_question
from llm.question_types import QuestionType
from llm.prompts import (
    ANSWER_SYSTEM_PROMPT,
    OUT_OF_SCOPE_SYSTEM_PROMPT,
    NEAR_RULE_QUERY_REWRITE_PROMPT, EXACT_RULE_ANSWER_SYSTEM_PROMPT,
)
from llm.citations import build_citations
from llm.state import ConversationStateStore
from ontology.loader import load_ontology, get_article_by_id
from retrieval.src.registry.metadata_registry import MetadataRegistry
from retrieval.src.retrieval.hybrid_orchestrator import HybridOrchestrator
from retrieval.src.retrieval.graph_retriever import GraphRetriever
from retrieval.text_rag.chunker import iter_all_chunks
from retrieval.text_rag.vector_store import ChunkVectorStore

# Check if local embedder is disabled
UIT_DISABLE_LOCAL_EMBEDDER = os.getenv("UIT_DISABLE_LOCAL_EMBEDDER", "false").lower() == "true"
# Disables Path 2 (knowledge-graph retrieval) of the dual-path pipeline; text-only
# retrieval still works when this is set. Name kept for backward-compat with existing deploys.
UIT_DISABLE_TRIPLET_RAG = os.getenv("UIT_DISABLE_TRIPLET_RAG", "false").lower() in ("true", "1")


class ChatPipeline:
    # Shared state store (in-memory, per-process)
    state_store = ConversationStateStore()
    registry = MetadataRegistry()
    
    # Routing thresholds (can be overridden via env for tuning)
    OUT_SCORE_DEFAULT = 0.18
    EXACT_SCORE_DEFAULT = 0.33
    EXACT_DOM_DEFAULT = 0.45
    # New thresholds for diagram-aligned routing
    EXACT_GOOD_SCORE = 0.45
    NEAR_MIN_GOOD_SCORE = 0.25

    # Generous domain keywords for study/training questions (accent-insensitive)
    STUDY_KEYWORDS = [
        "hoc",
        "mon",
        "hoc phan",
        "tin chi",
        "dang ky",
        "rut",
        "huy",
        "bao luu",
        "tam dung",
        "thoi hoc",
        "nghi hoc",
        "thi",
        "diem",
        "diem ren luyen",
        "ket qua",
        "hoc phi",
        "mien",
        "hoan",
        "canh cao",
        "dinh chi",
        "tot nghiep",
        "chuan dau ra",
        "dao tao",
        "quy che",
        "quy dinh",
        "phong dao tao",
        "phuc khao",
        "hoc lai",
        "cai thien",
        "thuc tap",
        "khoa luan",
        "do an",
        "lich hoc",
        "thoi khoa bieu",
        # Institution / context hints
        "uit",
        "dhcntt",
        "truong",
        "dtdh",
        "pttpc",
        "chuong trinh",
        "cttn",
        "clc",
        "cvht",
        "cbct",
        "nckh",
    ]

    # Obvious non-study / smalltalk keywords (accent-insensitive)
    NON_STUDY_KEYWORDS = [
        # "mua",
        # "nang",
        # "thoi tiet",
        # "du bao",
        # "bao nhieu do",
        # "nhiet do",
        # "bao",
        # "troi",
        # "canteen",
        # "can tin",
        # "phim",
        # "nhac",
        # "tinh yeu",
        # "bong da",
    ]
    
    def __init__(
        self,
        ontology_path: str | None = None,
        vector_db_path: str | None = None,
        llm_client: Any = None,
        embedder=None,
        vector_store: Any = None,
        ontology_graph=None,
        **kwargs
    ):
        self.logger = logging.getLogger("uit_chatbot.chat_pipeline")
        self.logger.setLevel(logging.DEBUG)

        self.llm_client = llm_client or LLMClient()
        self.ontology_graph = ontology_graph or load_ontology(
            ontology_path or os.getenv("UIT_TTL_PATH", "backend/ontology/uit_regulations.ttl")
        )

        # Only create embedder if local embedding is enabled
        if UIT_DISABLE_LOCAL_EMBEDDER:
            self.embedder = None
        else:
            if embedder is None:
                try:
                    from retrieval.text_rag.embeddings import TextEmbedder
                    embedder = TextEmbedder()
                except Exception:
                    embedder = None
            self.embedder = embedder

        try:
            # Resolve vector DB path relative to backend directory
            default_vector_db = Path(__file__).parent.parent / "retrieval" / "text_rag" / "vector_store.db"
            self.vector_store = vector_store or ChunkVectorStore(
                vector_db_path or os.getenv("UIT_VECTOR_DB", str(default_vector_db)),
                disable_local_embedder=UIT_DISABLE_LOCAL_EMBEDDER,
            )
            self.logger.info("VectorStore initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize VectorStore: %s", e, exc_info=True)
            self.vector_store = None

        # Path 2 of the dual-path pipeline: knowledge-graph retriever (seed + expand).
        # Much lighter than the old triplet extractor - no NLP models are loaded at
        # query time, only the exported (subject, relation, object) triples - so a
        # failure here no longer needs to take down text retrieval (Path 1) with it.
        if UIT_DISABLE_TRIPLET_RAG:
            self.graph_retriever = None
            self.logger.info("GraphRetriever disabled via UIT_DISABLE_TRIPLET_RAG")
        else:
            try:
                self.graph_retriever = GraphRetriever(embedder=self.embedder)
                self.logger.info(
                    "GraphRetriever initialized successfully (%d nodes, %d triples)",
                    len(self.graph_retriever.nodes),
                    len(self.graph_retriever.triples),
                )
            except Exception as e:
                self.logger.error("Failed to initialize GraphRetriever: %s", e, exc_info=True)
                self.graph_retriever = None

        try:
            if self.vector_store:
                self.hybrid = HybridOrchestrator(self.vector_store, self.graph_retriever)
                self.logger.info("HybridOrchestrator initialized successfully")
            else:
                self.hybrid = None
                self.logger.warning("HybridOrchestrator NOT initialized: VectorStore is None")
        except Exception as e:
            self.logger.error("Failed to initialize HybridOrchestrator: %s", e, exc_info=True)
            self.hybrid = None

        # Ensure the vector store is populated with the proper KB (JSONL items)
        try:
            if self.vector_store:
                self._ensure_vector_store_loaded()
        except Exception:
            pass

        self.top_k = int(os.getenv("UIT_RAG_TOP_K", "5"))
        # Larger candidate pool for EXACT_RULE questions (before reranking)
        self.exact_rule_candidate_k = int(
            os.getenv("UIT_RAG_EXACT_RULE_CANDIDATES", str(self.top_k * 6))
        )
        # Lazy-loaded cache of raw KB items for lexical fallback
        self._kb_items_cache: list[dict] | None = None


    async def answer_question(
        self,
        question: str,
        conversation_history: List[Dict[str, str]] | None = None,
        debug: bool = False,
        user_id: str = "default",
        force_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question, optionally with conversation history for multi-turn context.
        
        Args:
            question: The current question
            conversation_history: List of previous messages in format [{"role": "user|bot", "content": "..."}]
        """
        self.logger.info("[PIPELINE] Starting answer_question for user=%s, question=%r", user_id, question)

        # --- Hard OUT for obviously non-study/smalltalk questions ---
        if self.is_obviously_non_study(question):
            self.logger.info("[PIPELINE] Hard OUT_OF_SCOPE for obvious non-study question")
            debug_info: Dict[str, Any] = {
                "is_obviously_non_study": True,
                "final_intent": QuestionType.OUT_OF_SCOPE.value,
                "model_intent_suggestion": None,
                "demo_override": False,
                "intent_forced": False,
                "answer_style": "helpful_out_of_scope",
                "intent_decision_reason": "Hard OUT for obvious non-study/smalltalk question.",
            }
            base = await self._handle_out_of_scope(question)
            # Ensure no KB bullets / sources
            base["sources"] = []
            base["question_type"] = QuestionType.OUT_OF_SCOPE.value
            base["debug"] = debug_info
            return base

        # --- Intent classification (model label only, routing adjusted by thresholds) ---
        classification_input = question
        if conversation_history:
            recent_history = (
                conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            )
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            classification_input = f"{history_text}\n\nCurrent question: {question}"

        self.logger.info("[PIPELINE] Classifying question type via LLM...")
        q_type = await classify_question(classification_input, self.llm_client)
        self.logger.info("[PIPELINE] Question classified as: %s", q_type.value)

        # --- Build retrieval query (rewrite for NEAR_RULE) ---
        self.logger.info("[PIPELINE] Building retrieval query...")
        retrieval_query = await self._build_retrieval_query_async(question, q_type, conversation_history or [])
        self.logger.info("[PIPELINE] Retrieval query: %r", retrieval_query)
        
        # --- HYBRID RETRIEVAL ---
        if not self.hybrid:
            self.logger.error("[PIPELINE] HybridOrchestrator is None, cannot proceed")
            raise RuntimeError(
                "HybridOrchestrator not initialized. This likely means VectorStore failed to initialize. "
                "Check that: (1) vector_store.db exists, (2) graph NLP models are available, "
                "(3) UIT_DISABLE_TRIPLET_RAG environment variable is not set if you need triplet retrieval."
            )
        
        self.logger.info(
            "[PIPELINE] Running hybrid retrieval (text_top_k=%d, graph_top_k=%d)...",
            self.top_k,
            self.top_k,
        )
        # Always request debug info so we can compute routing metrics
        hybrid_result = self.hybrid.run(
            retrieval_query,
            text_top_k=self.top_k,
            graph_top_k=self.top_k,
            debug=True,
        )
        context = hybrid_result["context"]
        grounding = hybrid_result["grounding"]
        selected_article_id = grounding.get("article_id")
        dominance = grounding.get("dominance", 0.0)
        candidates = grounding.get("candidates", [])
        top_evidence = hybrid_result.get("top_evidence_for_debug", []) or []
        text_hits = hybrid_result.get("text_hits", []) or []
        graph_hits = hybrid_result.get("graph_hits", []) or []

        num_evidence = len(top_evidence)
        max_score = max((ev.get("score", 0.0) for ev in top_evidence), default=0.0)
        anchor_count = len(candidates)

        # --- C) Calibration & Confidence: evidence-quality filter ---
        # C2: Compute good_hits (filter out TOC/heading noise)
        good_hits = [h for h in text_hits if not self.is_toc_like(h.get("text", ""))]
        good_count = len(good_hits)
        toc_hits_count = len(text_hits) - good_count
        good_max_score = max((h.get("score", 0.0) for h in good_hits), default=0.0) if good_hits else 0.0

        # Domain detection (D0: Domain Gate)
        # Use enhanced domain detection with LLM fallback if needed
        is_study_domain, domain_reason = await self.is_study_domain_enhanced(
            question, retrieval_query, []
        )

        # Thresholds (allow override via env)
        out_score = float(os.getenv("UIT_ROUTING_OUT_SCORE", str(self.OUT_SCORE_DEFAULT)))
        exact_score = float(os.getenv("UIT_ROUTING_EXACT_SCORE", str(self.EXACT_SCORE_DEFAULT)))
        exact_dom = float(os.getenv("UIT_ROUTING_EXACT_DOM", str(self.EXACT_DOM_DEFAULT)))
        exact_good_score = float(os.getenv("UIT_ROUTING_EXACT_GOOD_SCORE", str(self.EXACT_GOOD_SCORE)))
        near_min_good_score = float(os.getenv("UIT_ROUTING_NEAR_MIN_GOOD_SCORE", str(self.NEAR_MIN_GOOD_SCORE)))

        # --- Intent decision ---
        intent_forced = False
        final_intent = q_type.value
        intent_reason = ""

        # Normalize force_intent
        force_intent_normalized = None
        if force_intent:
            fi = force_intent.strip().upper()
            if fi in {QuestionType.EXACT_RULE.value, QuestionType.NEAR_RULE.value, QuestionType.OUT_OF_SCOPE.value}:
                force_intent_normalized = fi

        if force_intent_normalized:
            intent_forced = True
            final_intent = force_intent_normalized
            intent_reason = f"Forced via API parameter force_intent={force_intent_normalized}"
        else:
            # --- D) Intent Decision (diagram-aligned) ---
            # D1: Intent policy using good_hits and new thresholds
            if not is_study_domain:
                # Non-study domain: always OUT_OF_SCOPE
                final_intent = QuestionType.OUT_OF_SCOPE.value
                intent_reason = "OUT_OF_SCOPE because question is not in study domain."
            else:
                # Study domain: use good_hits for routing
                if good_count == 0 or good_max_score < near_min_good_score:
                    # No good evidence or very weak evidence
                    final_intent = QuestionType.NEAR_RULE.value
                    intent_reason = (
                        "NEAR_RULE (in_domain_no_evidence) because good_count=0 or good_max_score < NEAR_MIN_GOOD_SCORE."
                    )
                elif good_max_score >= exact_good_score and (dominance >= exact_dom or max_score >= exact_score):
                    # Strong evidence and high confidence
                    final_intent = QuestionType.EXACT_RULE.value
                    intent_reason = (
                        "EXACT_RULE because good_max_score >= EXACT_GOOD_SCORE and dominance/max_score thresholds passed."
                    )
                else:
                    # Moderate evidence
                    final_intent = QuestionType.NEAR_RULE.value
                    intent_reason = (
                        "NEAR_RULE (grounded_with_clarification) because study-domain question has moderate evidence."
                    )

        # Log intent decision (D spec)
        self.logger.info(
            "[INTENT] study=%s, good_count=%d, good_max_score=%.3f, dominance=%.3f, final=%s",
            is_study_domain,
            good_count,
            good_max_score,
            dominance,
            final_intent,
        )

        debug_info: Dict[str, Any] = {
            "max_score": max_score,
            "dominance": dominance,
            "num_evidence": num_evidence,
            "anchor_count": anchor_count,
            "has_domain_signal": is_study_domain,
            "is_study_domain": is_study_domain,
            "domain_reason": domain_reason,
            "good_count": good_count,
            "good_max_score": good_max_score,
            "toc_hits_count": toc_hits_count,
            # For backward compatibility, keep classifier_label and chosen_intent,
            # but also expose clearer names.
            "classifier_label": q_type.value,
            "model_intent_suggestion": q_type.value,
            "chosen_intent": final_intent,
            "final_intent": final_intent,
            "intent_forced": intent_forced,
            "forced_intent": force_intent_normalized,
            "intent_decision_reason": intent_reason,
            }
        if debug:
            debug_info["text_hits"] = text_hits
            debug_info["graph_hits"] = graph_hits

        # --- OUT_OF_SCOPE branch ---
        if final_intent == QuestionType.OUT_OF_SCOPE.value:
            self.logger.info("[PIPELINE] Handling OUT_OF_SCOPE intent")
            debug_info["answer_style"] = "helpful_out_of_scope"
            base = await self._handle_out_of_scope(question)
            base["debug"] = debug_info
            # Ensure question_type in response matches final intent
            base["question_type"] = final_intent
            return base

        # --- Normal answer flow (EXACT_RULE / NEAR_RULE) ---
        self.logger.info(
            "[PIPELINE] Generating answer for article_id=%s with intent=%s",
            selected_article_id,
            final_intent,
        )
        
        # --- E) Compose Context + Citations (diagram box) ---
        # Use good_hits only for context composition
        evidence_for_context = good_hits if good_hits else []
        
        # Determine answer style for NEAR
        answer_style = "grounded_with_clarification"
        if final_intent == QuestionType.NEAR_RULE.value:
            if good_count == 0 or good_max_score < near_min_good_score:
                answer_style = "in_domain_no_evidence"
                # For in_domain_no_evidence: do not compose KB context
                evidence_for_context = []
            else:
                answer_style = "grounded_with_clarification"
                # For grounded_with_clarification: use top 3 good_hits to keep concise
                evidence_for_context = good_hits[:3]
        
        # EXACT: compose context using good_hits only (not TOC hits)
        if final_intent == QuestionType.EXACT_RULE.value:
            evidence_for_context = good_hits
        
        # Build context from evidence_for_context
        if evidence_for_context:
            context_lines = ["Các đoạn trích liên quan:"]
            for ev in evidence_for_context:
                text = ev.get("text", "").strip()
                article_id_ev = ev.get("article_id") or ""
                clause_id_ev = ev.get("clause_id") or ""
                if text:
                    context_lines.append(f"- Article: {article_id_ev}, Clause: {clause_id_ev}, Text: {text}")
            context = "\n".join(context_lines)
        else:
            context = ""

        # Path 2 of the dual-path pipeline: fold in knowledge-graph facts grounded to
        # the selected article, so relationally-connected evidence (computed above but
        # previously only logged for debug) actually reaches the LLM's context.
        graph_facts_block = self._graph_facts_block(graph_hits, selected_article_id)
        if graph_facts_block:
            context = f"{context}\n\n{graph_facts_block}".strip()

        # Enrich ontology facts chỉ theo grounding
        ontology_facts = []
        if selected_article_id:
            try:
                ontology_facts = get_article_by_id(self.ontology_graph, selected_article_id)
                self.logger.debug("[PIPELINE] Loaded %d ontology facts for article_id=%s", len(ontology_facts), selected_article_id)
            except Exception as e:
                self.logger.warning("[PIPELINE] Failed to load ontology facts: %s", e)
                ontology_facts = []
        # Add ontology facts to context
        if ontology_facts:
            context = f"{context}\n\nONTOLOGY FACTS:\n" + "\n".join([f.get("text", "") for f in ontology_facts])
            self.logger.debug("[PIPELINE] Added ontology facts to context")

        # --- F9: Citation builder ---
        # Use evidence_for_context for citations (good_hits only)
        citations = build_citations(evidence_for_context, self.registry)
        citation_text = "; ".join([c["display"] for c in citations]) if citations else None
        self.logger.debug("[PIPELINE] Built %d citations", len(citations) if citations else 0)

        # --- F) LLM Generate Answer (diagram box) ---
        self.logger.info("[PIPELINE] Calling LLM to generate answer...")
        if final_intent == QuestionType.EXACT_RULE.value:
            system_prompt = EXACT_RULE_ANSWER_SYSTEM_PROMPT
            # Add rule about numeric thresholds (F spec)
            system_prompt += "\n\nQUAN TRỌNG: Không được xuất ra các ngưỡng số (ví dụ: điểm, tín chỉ, thời gian) trừ khi chúng xuất hiện nguyên văn trong bằng chứng được cung cấp."
        else:
            # NEAR_RULE or any other in-domain fallback
            system_prompt = ANSWER_SYSTEM_PROMPT

        llm_answer = await self.llm_client.generate(
            system_prompt,
            user_prompt=question,
            context=context,
        )
        self.logger.info("[PIPELINE] LLM response received, length=%d chars", len(llm_answer))

        # --- G) Postprocess / Response shaping (diagram box) ---
        if final_intent == QuestionType.EXACT_RULE.value:
            debug_info["answer_style"] = "grounded"
            answer_text = self.render_exact_answer(
                question=question,
                evidence=evidence_for_context,
                ontology_facts=ontology_facts,
                citations=citations,
                llm_answer=llm_answer,
            )
        elif final_intent == QuestionType.NEAR_RULE.value:
            debug_info["answer_style"] = answer_style
            if answer_style == "in_domain_no_evidence":
                # G2(b): in_domain_no_evidence rendering
                answer_text = await self.render_near_no_evidence_answer(
                    question=question,
                    llm_answer=llm_answer,
                )
            else:
                # G2(a): grounded_with_clarification rendering
                clar_q = await self.choose_clarifying_question_llm(question, evidence_for_context)
                debug_info["clarifying_question"] = clar_q
                answer_text = self.render_near_answer(
                    question=question,
                    evidence=evidence_for_context,
                    ontology_facts=ontology_facts,
                    citations=citations,
                    llm_answer=llm_answer,
                    clarifying_question=clar_q,
                )
        else:
            # Should not reach here, but fallback
            answer_text = llm_answer
        
        result = {
            "answer": answer_text,
            "question_type": final_intent,
            "sources": evidence_for_context,  # Use good_hits only
            "grounding": grounding,
            "citations": citations,
            "citation_text": citation_text,
            "debug": debug_info,
        }
        # Update state
        self.state_store.update(user_id, question, selected_article_id)
        self.logger.info("[PIPELINE] Answer generation complete")
        return result

    async def get_context(
            self,
            question: str,
            conversation_history: List[Dict[str, str]] | None = None,
            debug: bool = False,
            force_intent: Optional[str] = None,
    ):
        """
        Answer a question, optionally with conversation history for multi-turn context.

        Args:
            question: The current question
            conversation_history: List of previous messages in format [{"role": "user|bot", "content": "..."}]
        """
        self.logger.info("[PIPELINE] Starting answer_question for user=%s, question=%r", question)

        # --- Hard OUT for obviously non-study/smalltalk questions ---
        if self.is_obviously_non_study(question):
            self.logger.info("[PIPELINE] Hard OUT_OF_SCOPE for obvious non-study question")
            debug_info: Dict[str, Any] = {
                "is_obviously_non_study": True,
                "final_intent": QuestionType.OUT_OF_SCOPE.value,
                "model_intent_suggestion": None,
                "demo_override": False,
                "intent_forced": False,
                "answer_style": "helpful_out_of_scope",
                "intent_decision_reason": "Hard OUT for obvious non-study/smalltalk question.",
            }
            base = await self._handle_out_of_scope(question)
            # Ensure no KB bullets / sources
            base["sources"] = []
            base["question_type"] = QuestionType.OUT_OF_SCOPE.value
            base["debug"] = debug_info
            return base

        # --- Intent classification (model label only, routing adjusted by thresholds) ---
        classification_input = question
        if conversation_history:
            recent_history = (
                conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            )
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
            classification_input = f"{history_text}\n\nCurrent question: {question}"

        self.logger.info("[PIPELINE] Classifying question type via LLM...")
        q_type = await classify_question(classification_input, self.llm_client)
        self.logger.info("[PIPELINE] Question classified as: %s", q_type.value)

        # --- Build retrieval query (rewrite for NEAR_RULE) ---
        self.logger.info("[PIPELINE] Building retrieval query...")
        retrieval_query = await self._build_retrieval_query_async(question, q_type, conversation_history or [])
        self.logger.info("[PIPELINE] Retrieval query: %r", retrieval_query)

        # --- HYBRID RETRIEVAL ---
        if not self.hybrid:
            self.logger.error("[PIPELINE] HybridOrchestrator is None, cannot proceed")
            raise RuntimeError(
                "HybridOrchestrator not initialized. This likely means VectorStore failed to initialize. "
                "Check that: (1) vector_store.db exists, (2) graph NLP models are available, "
                "(3) UIT_DISABLE_TRIPLET_RAG environment variable is not set if you need triplet retrieval."
            )

        self.logger.info(
            "[PIPELINE] Running hybrid retrieval (text_top_k=%d, graph_top_k=%d)...",
            self.top_k,
            self.top_k,
        )
        # Always request debug info so we can compute routing metrics
        hybrid_result = self.hybrid.run(
            retrieval_query,
            text_top_k=self.top_k,
            graph_top_k=self.top_k,
            debug=True,
        )
        context = hybrid_result["context"]
        grounding = hybrid_result["grounding"]
        selected_article_id = grounding.get("article_id")
        dominance = grounding.get("dominance", 0.0)
        candidates = grounding.get("candidates", [])
        top_evidence = hybrid_result.get("top_evidence_for_debug", []) or []
        text_hits = hybrid_result.get("text_hits", []) or []
        graph_hits = hybrid_result.get("graph_hits", []) or []

        num_evidence = len(top_evidence)
        max_score = max((ev.get("score", 0.0) for ev in top_evidence), default=0.0)
        anchor_count = len(candidates)

        # --- C) Calibration & Confidence: evidence-quality filter ---
        # C2: Compute good_hits (filter out TOC/heading noise)
        good_hits = [h for h in text_hits if not self.is_toc_like(h.get("text", ""))]
        good_count = len(good_hits)
        toc_hits_count = len(text_hits) - good_count
        good_max_score = max((h.get("score", 0.0) for h in good_hits), default=0.0) if good_hits else 0.0

        # Domain detection (D0: Domain Gate)
        # Use enhanced domain detection with LLM fallback if needed
        is_study_domain, domain_reason = await self.is_study_domain_enhanced(
            question, retrieval_query, []
        )

        # Thresholds (allow override via env)
        out_score = float(os.getenv("UIT_ROUTING_OUT_SCORE", str(self.OUT_SCORE_DEFAULT)))
        exact_score = float(os.getenv("UIT_ROUTING_EXACT_SCORE", str(self.EXACT_SCORE_DEFAULT)))
        exact_dom = float(os.getenv("UIT_ROUTING_EXACT_DOM", str(self.EXACT_DOM_DEFAULT)))
        exact_good_score = float(os.getenv("UIT_ROUTING_EXACT_GOOD_SCORE", str(self.EXACT_GOOD_SCORE)))
        near_min_good_score = float(os.getenv("UIT_ROUTING_NEAR_MIN_GOOD_SCORE", str(self.NEAR_MIN_GOOD_SCORE)))

        # --- Intent decision ---
        intent_forced = False
        final_intent = q_type.value
        intent_reason = ""

        # Normalize force_intent
        force_intent_normalized = None
        if force_intent:
            fi = force_intent.strip().upper()
            if fi in {QuestionType.EXACT_RULE.value, QuestionType.NEAR_RULE.value, QuestionType.OUT_OF_SCOPE.value}:
                force_intent_normalized = fi

        if force_intent_normalized:
            intent_forced = True
            final_intent = force_intent_normalized
            intent_reason = f"Forced via API parameter force_intent={force_intent_normalized}"
        else:
            # --- D) Intent Decision (diagram-aligned) ---
            # D1: Intent policy using good_hits and new thresholds
            if not is_study_domain:
                # Non-study domain: always OUT_OF_SCOPE
                final_intent = QuestionType.OUT_OF_SCOPE.value
                intent_reason = "OUT_OF_SCOPE because question is not in study domain."
            else:
                # Study domain: use good_hits for routing
                if good_count == 0 or good_max_score < near_min_good_score:
                    # No good evidence or very weak evidence
                    final_intent = QuestionType.NEAR_RULE.value
                    intent_reason = (
                        "NEAR_RULE (in_domain_no_evidence) because good_count=0 or good_max_score < NEAR_MIN_GOOD_SCORE."
                    )
                elif good_max_score >= exact_good_score and (dominance >= exact_dom or max_score >= exact_score):
                    # Strong evidence and high confidence
                    final_intent = QuestionType.EXACT_RULE.value
                    intent_reason = (
                        "EXACT_RULE because good_max_score >= EXACT_GOOD_SCORE and dominance/max_score thresholds passed."
                    )
                else:
                    # Moderate evidence
                    final_intent = QuestionType.NEAR_RULE.value
                    intent_reason = (
                        "NEAR_RULE (grounded_with_clarification) because study-domain question has moderate evidence."
                    )

        # Log intent decision (D spec)
        self.logger.info(
            "[INTENT] study=%s, good_count=%d, good_max_score=%.3f, dominance=%.3f, final=%s",
            is_study_domain,
            good_count,
            good_max_score,
            dominance,
            final_intent,
        )

        debug_info: Dict[str, Any] = {
            "max_score": max_score,
            "dominance": dominance,
            "num_evidence": num_evidence,
            "anchor_count": anchor_count,
            "has_domain_signal": is_study_domain,
            "is_study_domain": is_study_domain,
            "domain_reason": domain_reason,
            "good_count": good_count,
            "good_max_score": good_max_score,
            "toc_hits_count": toc_hits_count,
            # For backward compatibility, keep classifier_label and chosen_intent,
            # but also expose clearer names.
            "classifier_label": q_type.value,
            "model_intent_suggestion": q_type.value,
            "chosen_intent": final_intent,
            "final_intent": final_intent,
            "intent_forced": intent_forced,
            "forced_intent": force_intent_normalized,
            "intent_decision_reason": intent_reason,
        }

        if debug:
            debug_info["text_hits"] = text_hits
            debug_info["graph_hits"] = graph_hits

        # --- OUT_OF_SCOPE branch ---
        if final_intent == QuestionType.OUT_OF_SCOPE.value:
            self.logger.info("[PIPELINE] Handling OUT_OF_SCOPE intent")
            debug_info["answer_style"] = "helpful_out_of_scope"
            base = await self._handle_out_of_scope(question)
            base["debug"] = debug_info
            # Ensure question_type in response matches final intent
            base["question_type"] = final_intent
            return base

        # --- Normal answer flow (EXACT_RULE / NEAR_RULE) ---
        self.logger.info(
            "[PIPELINE] Generating answer for article_id=%s with intent=%s",
            selected_article_id,
            final_intent,
        )

        # --- E) Compose Context + Citations (diagram box) ---
        # Use good_hits only for context composition
        evidence_for_context = good_hits if good_hits else []

        # Determine answer style for NEAR
        answer_style = "grounded_with_clarification"
        if final_intent == QuestionType.NEAR_RULE.value:
            if good_count == 0 or good_max_score < near_min_good_score:
                answer_style = "in_domain_no_evidence"
                # For in_domain_no_evidence: do not compose KB context
                evidence_for_context = []
            else:
                answer_style = "grounded_with_clarification"
                # For grounded_with_clarification: use top 3 good_hits to keep concise
                evidence_for_context = good_hits[:3]

        # EXACT: compose context using good_hits only (not TOC hits)
        if final_intent == QuestionType.EXACT_RULE.value:
            evidence_for_context = good_hits

        # Build context from evidence_for_context
        if evidence_for_context:
            context_lines = ["Các đoạn trích liên quan:"]
            for ev in evidence_for_context:
                text = ev.get("text", "").strip()
                article_id_ev = ev.get("article_id") or ""
                clause_id_ev = ev.get("clause_id") or ""
                if text:
                    context_lines.append(f"- Article: {article_id_ev}, Clause: {clause_id_ev}, Text: {text}")
            context = "\n".join(context_lines)
        else:
            context = ""

        # Enrich ontology facts chỉ theo grounding
        ontology_facts = []
        if selected_article_id:
            try:
                ontology_facts = get_article_by_id(self.ontology_graph, selected_article_id)
                self.logger.debug("[PIPELINE] Loaded %d ontology facts for article_id=%s", len(ontology_facts),
                                  selected_article_id)
            except Exception as e:
                self.logger.warning("[PIPELINE] Failed to load ontology facts: %s", e)
                ontology_facts = []
        # Add ontology facts to context
        if ontology_facts:
            context = f"{context}\n\nONTOLOGY FACTS:\n" + "\n".join([f.get("text", "") for f in ontology_facts])
            self.logger.debug("[PIPELINE] Added ontology facts to context")

        # --- F9: Citation builder ---
        # Use evidence_for_context for citations (good_hits only)
        citations = build_citations(evidence_for_context, self.registry)
        citation_text = "; ".join([c["display"] for c in citations]) if citations else None
        self.logger.debug("[PIPELINE] Built %d citations", len(citations) if citations else 0)

        # --- F) LLM Generate Answer (diagram box) ---
        self.logger.info("[PIPELINE] Calling LLM to generate answer...")
        if final_intent == QuestionType.EXACT_RULE.value:
            system_prompt = EXACT_RULE_ANSWER_SYSTEM_PROMPT
            # Add rule about numeric thresholds (F spec)
            system_prompt += "\n\nQUAN TRỌNG: Không được xuất ra các ngưỡng số (ví dụ: điểm, tín chỉ, thời gian) trừ khi chúng xuất hiện nguyên văn trong bằng chứng được cung cấp."
        else:
            # NEAR_RULE or any other in-domain fallback
            system_prompt = ANSWER_SYSTEM_PROMPT

        result = {
            "system_prompt": system_prompt,
            "context": context,
            "debug": debug_info,
        }

        return result

    async def _handle_out_of_scope(self, question: str) -> Dict[str, Any]:
        llm_answer = await self.llm_client.generate(
            OUT_OF_SCOPE_SYSTEM_PROMPT,
            user_prompt=question,
            context="",
        )
        answer_text = self.render_out_answer(
            question=question,
            evidence=[],
            citations=[],
            llm_answer=llm_answer,
        )
        return {
            "answer": answer_text,
            "question_type": QuestionType.OUT_OF_SCOPE.value,
            "sources": [],
            "ontology_facts": [],
            "debug": {
                "answer_style": "helpful_out_of_scope",
            },
        }

    def _graph_facts_block(self, graph_hits: List[Dict[str, Any]], article_id: Optional[str]) -> str:
        """Render Path-2 (knowledge-graph) triples grounded to `article_id` as a
        'GRAPH FACTS:' block, deduplicated and in verbalized subject/predicate/object form."""
        if not article_id or not graph_hits:
            return ""
        facts: List[str] = []
        seen: set[str] = set()
        for gh in graph_hits:
            if gh.get("article_id") != article_id:
                continue
            fact = f"{gh.get('subject','')} — {gh.get('predicate','')} — {gh.get('object','')}"
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
        if not facts:
            return ""
        return "GRAPH FACTS:\n" + "\n".join(facts)

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

    def _log_chunks(self, prefix: str, chunks: List[Dict[str, Any]], limit: int = 5) -> None:
        if not self.logger:
            return
        self.logger.debug("%s: total=%d", prefix, len(chunks))
        for idx, ch in enumerate(chunks[:limit], start=1):
            article = ch.get("article_id") or ch.get("article") or ""
            clause = ch.get("clause_id") or ""
            title = ""
            md = ch.get("metadata")
            if isinstance(md, dict):
                title = md.get("title") or md.get("heading") or ""
            text = ch.get("text") or ch.get("content") or ""
            score = ch.get("score")
            lex = ch.get("lexical_score")
            combined = ch.get("combined_score")
            excerpt = (text[:160] + "...") if len(text) > 160 else text
            self.logger.debug(
                "%s #%d article=%s clause=%s title=%r score=%s lex=%s combined=%s excerpt=%r",
                prefix,
                idx,
                article,
                clause,
                title,
                score,
                lex,
                combined,
                excerpt,
            )

    def _ensure_vector_store_loaded(self) -> None:
        """
        Ensure the vector/text index is populated from the canonical JSONL items file.
        If the DB is empty (e.g., fresh deploy), we build a lightweight index on startup.
        """
        try:
            existing = self.vector_store.count_chunks()
            if self.logger:
                self.logger.debug("[RAG-DEBUG] vector_store rows=%s", existing)
            if existing > 0:
                return
        except Exception as exc:
            if self.logger:
                self.logger.warning("[RAG-DEBUG] vector_store count failed: %s", exc)
            # If any issue checking count, attempt to rebuild
            pass

        content_path = Path(os.getenv("UIT_CONTENT_JSON", "graph/mongo_export_uit/KB_UIT.items.json"))
        max_chars = int(os.getenv("UIT_CHUNK_MAX_CHARS", "800"))

        if not content_path.exists():
            # Fail silently; retrieval will return empty, but we avoid crash
            if self.logger:
                self.logger.warning("[RAG-DEBUG] content file not found: %s", content_path)
            return

        # Choose embedder: only when local embedding is enabled
        embedder = None
        if not self.vector_store.disable_local_embedder:
            # Use existing embedder if available, otherwise create one
            if self.embedder is None:
                from retrieval.text_rag.embeddings import TextEmbedder

                embedder = TextEmbedder()
                self.embedder = embedder
            else:
                embedder = self.embedder

        # Build index in small batches
        total_chunks = 0
        batch: List[Dict[str, Any]] = []
        for chunk in iter_all_chunks(content_path, max_chars=max_chars):
            batch.append(chunk)
            total_chunks += 1
            if len(batch) >= 128:
                self.vector_store.index_chunks(batch, embedder)
                batch.clear()
        if batch:
            self.vector_store.index_chunks(batch, embedder)
        if self.logger:
            try:
                new_total = self.vector_store.count_chunks()
            except Exception:
                new_total = "unknown"
            self.logger.info(
                "[RAG-DEBUG] indexed chunks from %s: added=%s, total_now=%s",
                content_path,
                total_chunks,
                new_total,
            )

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
        Extract keywords from query by removing stopwords and highly generic academic words.
        Returns list of normalized keywords.
        """
        # Vietnamese stopwords (function words) + highly generic academic terms
        stopwords = {
            # Function words
            "la", "co", "thi", "em", "anh", "toi", "ban", "the", "mot", "cua", "va", "hoac", "nhung",
            "duoc", "se", "da", "dang", "ve", "cho", "voi", "trong", "tren", "duoi", "sao", "gi",
            "nao", "khi", "neu", "thi", "vay", "the", "con", "hay", "ma", "de", "nen", "khong",
            "khong", "chua", "chung", "nhieu", "it", "rat", "qua", "nhu", "nhu", "cung", "cac",
            # Highly generic academic terms (appear in almost every rule)
            "dieu", "kien", "sinh", "vien", "lop", "mon", "hoc", "truong",
        }
        
        normalized = self._normalize_vietnamese(query)
        tokens = normalized.split()
        keywords = [t for t in tokens if t not in stopwords and len(t) > 1]
        return keywords

    # Domain keyword groups (reusable across scoring and detection)
    DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "credits": [
            "tin chi", "hoc ky", "hoc ky chinh", "dang ky", "toi da", "toi thieu",
        ],
        "warning": [
            "canh bao", "hoc vu", "rot mon", "hoc lai", "thi lai", "diem trung binh", "dtbhk", "dtbc",
        ],
        "graduation": [
            "diem ren luyen", "tot nghiep", "xet tot nghiep",
        ],
        "thesis": [
            "khoa luan", "kltn", "bao ve", "nop", "bao ve khoa luan", "khoa luan tot nghiep",
        ],
    }

    def _detect_domains(self, query: str) -> List[str]:
        """Detect active domains based on domain keywords present in the query."""
        normalized = self._normalize_vietnamese(query)
        active_domains: List[str] = []
        for domain, kws in self.DOMAIN_KEYWORDS.items():
            for kw in kws:
                if kw in normalized:
                    active_domains.append(domain)
                    break
        return active_domains
    
    def _compute_hybrid_lexical_score(
        self, chunk_text: str, keywords: List[str]
    ) -> float:
        """
        Compute lexical overlap score for hybrid retrieval.
        Includes bonuses for common bigrams/domains.
        """
        normalized_content = self._normalize_vietnamese(chunk_text)
        score = 0.0
        
        # Basic keyword matching
        for kw in keywords:
            if kw in normalized_content:
                score += 1.0
        
        # Bigram bonuses (generic, domain-oriented)
        normalized_lower = normalized_content.lower()
        if "canh bao" in normalized_lower and "hoc vu" in normalized_lower:
            score += 2.0
        if ("khoa luan" in normalized_lower or "kltn" in normalized_lower) and (
            "tot nghiep" in normalized_lower or "nop" in normalized_lower or "bao ve" in normalized_lower
        ):
            score += 2.0
        if "tin chi" in normalized_lower and ("hoc ky" in normalized_lower or "toi da" in normalized_lower):
            score += 1.5
        if "diem ren luyen" in normalized_lower and "tot nghiep" in normalized_lower:
            score += 2.0
        
        return score

    def _compute_domain_score(self, chunk_text: str, active_domains: List[str]) -> float:
        """
        Compute domain match score based on DOMAIN_KEYWORDS presence in the chunk.
        """
        if not active_domains:
            return 0.0
        normalized = self._normalize_vietnamese(chunk_text)
        score = 0.0
        for domain in active_domains:
            kws = self.DOMAIN_KEYWORDS.get(domain, [])
            for kw in kws:
                if kw in normalized:
                    score += 1.0
        return score
    
    def _hybrid_retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval: vector search + lexical ranking.
        Generic implementation that works for any question.
        """
        # Step 1: Vector search (or text search if embedding disabled)
        if candidate_k is None:
            candidate_k = top_k * 2

        chunks = self.vector_store.search(query, self.embedder, top_k=candidate_k)  # Get more candidates
        
        if not chunks:
            return []
        
        # Step 2: Extract keywords from query
        keywords = self._extract_keywords_from_query(query)
        active_domains = self._detect_domains(query)
        
        # Step 3: Compute lexical scores and combine with embedding scores
        scored_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            lexical_score = self._compute_hybrid_lexical_score(chunk_text, keywords)
            embedding_score = chunk.get("score", 0.5)
            domain_score = self._compute_domain_score(chunk_text, active_domains)
            
            # Combine scores: lexical (0.75) + domain (1.0) + embedding
            combined_score = embedding_score + 0.75 * lexical_score + 1.0 * domain_score
            
            scored_chunks.append({
                **chunk,
                "lexical_score": lexical_score,
                "domain_score": domain_score,
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

    def is_study_domain(self, question: str) -> bool:
        """
        Heuristic check for whether the question is in the UIT study/training domain.
        Uses accent-insensitive matching on a generous keyword set.
        """
        normalized = self._normalize_vietnamese(question)
        return any(kw in normalized for kw in self.STUDY_KEYWORDS)

    async def is_study_domain_enhanced(
        self, question: str, rewritten_query: str = "", keywords: List[str] = None
    ) -> tuple[bool, str]:
        """
        Domain gate with LLM fallback (D0 spec).
        Layer A: heuristic keyword list (score-based).
        Layer B: if uncertain, call LLM classifier to output strict JSON.
        Returns: (is_study_domain: bool, reason: str)
        """
        # Layer A: heuristic keyword check
        normalized = self._normalize_vietnamese(question)
        heuristic_score = sum(1 for kw in self.STUDY_KEYWORDS if kw in normalized)
        
        # If strong signal (multiple keywords), return True
        if heuristic_score >= 2:
            return True, f"Heuristic: found {heuristic_score} study keywords"
        
        # If at least one keyword, likely study domain
        if heuristic_score >= 1:
            return True, f"Heuristic: found study keyword"
        
        # If no keywords but question is very short or ambiguous, use LLM fallback
        # For now, we'll use heuristic only (LLM fallback can be added if needed)
        # For uncertain cases with no keywords, default to False
        if heuristic_score == 0:
            return False, "Heuristic: no study keywords found"
        
        return False, "Heuristic: uncertain"

    def is_obviously_non_study(self, question: str) -> bool:
        """Detect clearly non-study topics like weather, canteen, movies, sports, love advice."""
        normalized = self._normalize_vietnamese(question)
        return any(kw in normalized for kw in self.NON_STUDY_KEYWORDS)

    def is_toc_like(self, text: str) -> bool:
        """
        Detect table-of-contents like lines (short, mostly punctuation, lots of dots).
        These often correspond to headings or index lines and should not drive EXACT routing.
        
        C1 spec:
        - very short (< 80 chars) AND matches trailing page-number pattern with many dots: r"\.{5,}\s*\d+\s*$"
        - OR contains common noise keywords: "Mục lục", "Sơ đồ tóm tắt"
        - OR dominated by dots/whitespace and contains no verbs/conditions (simple heuristic allowed)
        """
        t = (text or "").strip()
        if not t:
            return True

        # Check for common noise keywords
        if "Mục lục" in t or "Sơ đồ tóm tắt" in t:
            return True

        # Spec: very short (< 80 chars) AND matches trailing page-number pattern with many dots
        if len(t) < 80:
            # Pattern: 5+ dots followed by optional whitespace and digits at end
            if re.search(r"\.{5,}\s*\d+\s*$", t):
                return True
            # Also check for many dots in general
            if "..." in t or "….." in t or t.count(".") > 10:
                return True

        # Spec: many dots plus trailing page number (general case)
        if t.count(".") > 10 and any(ch.isdigit() for ch in t[-5:]):
            return True

        # Heuristic: dominated by dots/whitespace and contains no verbs/conditions
        # Simple check: very short lines with few letters are likely headings or noise
        if len(t) < 25:
            letters = sum(c.isalpha() for c in t)
            if letters < 5:
                return True

        # High ratio of punctuation to characters also suggests TOC
        non_alnum = sum(not c.isalnum() and not c.isspace() for c in t)
        if non_alnum / max(len(t), 1) > 0.5:
            return True

        return False

    # --- Answer rendering helpers ---

    def render_exact_answer(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        ontology_facts: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        llm_answer: str,
    ) -> str:
        """
        Render the final answer for EXACT_RULE.
        Currently we trust the LLM answer, which is already prompted to be grounded
        and citation-friendly, and return it directly.
        """
        return self._clean_contradictory_disclaimers(
            llm_answer,
            has_relevant_evidence=self._has_any_relevant_hit(question, evidence),
            has_only_toc_evidence=self._has_only_toc_relevant_hits(question, evidence),
        )

    def _build_near_evidence_bullets(self, evidence: List[Dict[str, Any]]) -> List[str]:
        """Create 1–2 grounded bullets summarizing what evidence actually says."""
        bullets: List[str] = []
        for ev in evidence[:2]:
            text = (ev.get("text") or "").strip()
            meta = ev.get("metadata") or {}
            article_id = ev.get("article_id") or meta.get("article_id")
            title = meta.get("title") or meta.get("heading")

            if not text:
                continue

            if self.is_toc_like(text):
                # Do not invent details; only mention that a section exists.
                label_parts = []
                if title:
                    label_parts.append(f"mục \"{title}\"")
                if article_id:
                    label_parts.append(f"Điều {article_id}")
                label = ", ".join(label_parts) if label_parts else "một mục trong quy chế"
                bullets.append(
                    f"- Mình thấy trong quy chế có {label}, nhưng đoạn trích hiện tại giống mục lục nên chưa có nội dung khoản chi tiết."
                )
            else:
                # Use a truncated snippet of the actual text as grounded summary.
                snippet = text
                if len(snippet) > 220:
                    snippet = snippet[:220].rstrip() + "..."
                bullets.append(f"- {snippet}")

        if not bullets:
            bullets.append(
                "- Hiện tại mình chưa tìm thấy đoạn trích cụ thể nào mô tả chi tiết tình huống này trong quy chế."
            )
        return bullets

    def choose_clarifying_question(self, question: str) -> str:
        """Choose exactly one clarifying question based on keywords in the question."""
        normalized = self._normalize_vietnamese(question)

        # nghỉ học / bảo lưu
        if any(kw in normalized for kw in ["nghi hoc", "thoi hoc", "tam dung", "bao luu"]):
            return "Bạn muốn thôi học hẳn hay tạm dừng/bảo lưu (tạm nghỉ rồi quay lại)?"

        # hủy / rút môn
        if "huy" in normalized or "rut" in normalized:
            if any(kw in normalized for kw in ["hoc phan", "mon"]):
                return "Bạn muốn hủy đăng ký trước hạn hay bỏ thi/không dự thi?"

        # tốt nghiệp
        if "tot nghiep" in normalized:
            return "Bạn đang thuộc hệ nào (đại trà/CLC/song ngành) và đã tích lũy khoảng bao nhiêu tín chỉ?"

        # Generic in-domain clarification
        return "Bạn đang hỏi theo học kỳ nào và tình huống cụ thể của bạn là gì (VD: đã đăng ký môn/chưa, đã thi/chưa)?"

    async def choose_clarifying_question_llm(
        self, question: str, evidence: List[Dict[str, Any]]
    ) -> str:
        """
        G2 spec: Use LLM to generate exactly ONE clarifying question.
        Input: original question + (optional) top evidence snippets (only if grounded_with_clarification).
        Output JSON: {"near_answer": "...", "clarifying_question": "...?"}
        """
        # Build evidence snippets for context
        evidence_text = ""
        if evidence:
            snippets = []
            for ev in evidence[:2]:  # Top 2 evidence snippets
                text = ev.get("text", "").strip()
                if text and not self.is_toc_like(text):
                    snippets.append(text[:200])  # Truncate to 200 chars
            if snippets:
                evidence_text = "\n\nBằng chứng tìm thấy:\n" + "\n".join(f"- {s}" for s in snippets)

        prompt = f"""Bạn nhận một câu hỏi của sinh viên về quy chế UIT và (có thể) một số đoạn trích bằng chứng.

Câu hỏi: {question}
{evidence_text}

Nhiệm vụ: Tạo một câu hỏi làm rõ (clarifying question) để giúp trả lời chính xác hơn.

Yêu cầu:
- Chỉ trả về JSON: {{"clarifying_question": "..."}}
- Câu hỏi phải có đúng một dấu chấm hỏi (?)
- Phải liên quan đến chủ đề
- KHÔNG được đề cập đến "Điều <uuid>" hoặc trích dẫn TOC text
- Tập trung vào thông tin còn thiếu (ví dụ: hệ đào tạo, học kỳ, tình huống cụ thể)

Ví dụ:
- "Bạn đang thuộc hệ nào (đại trà/CLC/song ngành) và đã tích lũy khoảng bao nhiêu tín chỉ?"
- "Bạn muốn thôi học hẳn hay tạm dừng/bảo lưu (tạm nghỉ rồi quay lại)?"
- "Bạn đang hỏi theo học kỳ nào và tình huống cụ thể của bạn là gì (VD: đã đăng ký môn/chưa, đã thi/chưa)?"

Chỉ trả về JSON, không có text khác."""

        try:
            result = await self.llm_client.generate_json(
                system_prompt="Bạn là trợ lý tạo câu hỏi làm rõ cho chatbot UIT.",
                user_prompt=prompt,
            )
            clar_q = result.get("clarifying_question", "")
            # Ensure it has exactly one question mark
            if "?" not in clar_q:
                clar_q = clar_q.rstrip(".") + "?"
            # Fallback to keyword-based if LLM fails
            if not clar_q or len(clar_q) < 10:
                return self.choose_clarifying_question(question)
            return clar_q
        except Exception as e:
            self.logger.warning("[PIPELINE] LLM clarifying question generation failed: %s", e)
            return self.choose_clarifying_question(question)

    def render_near_answer(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        ontology_facts: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        llm_answer: str,
        clarifying_question: str,
    ) -> str:
        """
        G2(a): Render NEAR_RULE answer with grounded_with_clarification style.
        - Provide up to 2 conditional bullets that address the user's main intent using evidence.
        - Ask exactly ONE clarifying question, tailored to missing slot.
        """
        bullets = self._build_near_evidence_bullets(evidence)

        parts: List[str] = []
        parts.append("**Mình tìm thấy trong quy chế/KB:**")
        parts.extend(bullets)
        parts.append("")
        parts.append("**Để trả lời chính xác hơn, bạn cho mình biết:**")
        parts.append(f"- {clarifying_question}")

        raw = "\n".join(parts)
        return self._clean_contradictory_disclaimers(
            raw,
            has_relevant_evidence=self._has_any_relevant_hit(question, evidence),
            has_only_toc_evidence=self._has_only_toc_relevant_hits(question, evidence),
        )

    async def render_near_no_evidence_answer(
        self,
        question: str,
        llm_answer: str,
    ) -> str:
        """
        G2(b): Render NEAR_RULE answer with in_domain_no_evidence style.
        - Start with: "Mình chưa tìm thấy điều khoản trực tiếp trong trích dẫn hiện có."
        - Provide safe general guidance WITHOUT numeric claims.
        - Ask exactly ONE clarifying question that helps retrieval.
        """
        # Generate clarifying question using LLM
        clar_q = await self.choose_clarifying_question_llm(question, [])

        parts: List[str] = []
        parts.append("Mình chưa tìm thấy điều khoản trực tiếp trong trích dẫn hiện có.")
        parts.append("")
        
        # Use LLM answer as safe general guidance (if available)
        if llm_answer and llm_answer.strip():
            # Clean up LLM answer to remove numeric claims if not in evidence
            guidance = llm_answer.strip()
            parts.append(guidance)
            parts.append("")
        
        parts.append("**Để tìm thông tin chính xác hơn, bạn cho mình biết:**")
        parts.append(f"- {clar_q}")

        return "\n".join(parts)

    def render_out_answer(
        self,
        question: str,
        evidence: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        llm_answer: str,
    ) -> str:
        """
        Render OUT_OF_SCOPE answer:
        - Helpful, best-effort response.
        - Clear disclaimer that it's outside the regulations KB.
        - Short redirection to in-domain topics.
        """
        # Use the LLM's best-effort guidance, but wrap with disclaimer + redirect.
        main_part = llm_answer.strip() if llm_answer else ""

        disclaimer = (
            "Lưu ý: câu hỏi này nằm ngoài phạm vi quy chế/KB đào tạo nên mình không có điều khoản để trích dẫn."
        )
        redirect = (
            "Nếu bạn muốn hỏi về học tập/quy chế (đăng ký học phần, điều kiện dự thi/tốt nghiệp, cảnh cáo học vụ…), mình hỗ trợ rất tốt."
        )

        parts: List[str] = []
        if main_part:
            parts.append(main_part)
            parts.append("")
        parts.append(disclaimer)
        parts.append(redirect)
        # OUT answers are allowed to mention lack of info, so no cleaning needed here.
        return "\n".join(parts)

    # --- Evidence / disclaimer helpers ---

    def _normalize_for_overlap(self, text: str) -> List[str]:
        """Normalize text for simple lexical overlap (Vietnamese accent-insensitive)."""
        norm = self._normalize_vietnamese(text)
        tokens = [t for t in norm.split() if len(t) > 1]
        return tokens

    def is_relevant_hit(self, question: str, hit_text: str) -> bool:
        """
        Simple heuristic: hit is relevant if it shares at least one non-trivial token
        with the question OR has high score (handled upstream).
        """
        if not hit_text:
            return False
        q_tokens = set(self._normalize_for_overlap(question))
        h_tokens = set(self._normalize_for_overlap(hit_text))
        if not q_tokens or not h_tokens:
            return False
        overlap = q_tokens & h_tokens
        return len(overlap) > 0

    def _relevant_hits(self, question: str, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relevant: List[Dict[str, Any]] = []
        for ev in evidence:
            text = ev.get("text") or ""
            if self.is_relevant_hit(question, text):
                relevant.append(ev)
        return relevant

    def _has_any_relevant_hit(self, question: str, evidence: List[Dict[str, Any]]) -> bool:
        return len(self._relevant_hits(question, evidence)) > 0

    def _has_only_toc_relevant_hits(self, question: str, evidence: List[Dict[str, Any]]) -> bool:
        relevant = self._relevant_hits(question, evidence)
        if not relevant:
            return False
        return all(self.is_toc_like(ev.get("text") or "") for ev in relevant)

    def _clean_contradictory_disclaimers(
        self,
        answer: str,
        has_relevant_evidence: bool,
        has_only_toc_evidence: bool,
    ) -> str:
        """
        If we have relevant non-TOC evidence, remove strong 'no info' disclaimers
        like 'không có thông tin trực tiếp' / 'không liên quan' / 'không có căn cứ'.
        """
        if not answer:
            return answer

        if not has_relevant_evidence or has_only_toc_evidence:
            # In these cases it's still acceptable to mention lack of detailed info.
            return answer

        lowered = answer.lower()
        bad_phrases = [
            "không có thông tin trực tiếp",
            "không có thông tin chi tiết",
            "không liên quan",
            "không có căn cứ",
        ]
        cleaned = answer
        for phrase in bad_phrases:
            if phrase in lowered:
                # Simple removal: replace phrase with empty string.
                cleaned = cleaned.replace(phrase, "")
        return cleaned

    def _load_kb_items(self) -> List[Dict[str, Any]]:
        """
        Load all KB items from UIT_CONTENT_JSON (JSON or JSONL) and cache them.
        Each item is expected to have at least: _id, title/heading, content.
        """
        if self._kb_items_cache is not None:
            return self._kb_items_cache

        content_path = Path(os.getenv("UIT_CONTENT_JSON", "graph/mongo_export_uit/KB_UIT.items.json"))

        items: List[Dict[str, Any]] = []
        if not content_path.exists():
            if self.logger:
                self.logger.warning("[RAG-DEBUG] KB items file not found for lexical fallback: %s", content_path)
            self._kb_items_cache = []
            return self._kb_items_cache

        try:
            if content_path.suffix.lower() in {".jsonl", ".jsonl.txt"}:
                import json

                with content_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            items.append(obj)
                        except Exception:
                            continue
            else:
                import json

                with content_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    maybe_items = data.get("items") or data.get("data")
                    if isinstance(maybe_items, list):
                        items = maybe_items
                    else:
                        items = [data]
        except Exception as exc:
            if self.logger:
                self.logger.warning("[RAG-DEBUG] failed to load KB items for lexical fallback: %s", exc)
            items = []

        self._kb_items_cache = items
        if self.logger:
            self.logger.info("[RAG-DEBUG] loaded %d KB items for lexical fallback", len(items))
        return self._kb_items_cache

    def _has_keyword_hit(self, question: str, chunks: List[Dict[str, Any]]) -> bool:
        """
        Check if any retrieved chunk contains at least one keyword/number from the question.
        Used to decide whether we need lexical fallback.
        """
        if not chunks:
            return False
        keywords, numbers = self._extract_keywords(question)
        if not keywords and not numbers:
            return False

        for ch in chunks:
            text = ch.get("text", "") or ch.get("content", "")
            if not text:
                continue
            score = self._compute_lexical_score(text, keywords, numbers)
            if score > 0:
                return True
        return False

    def _lexical_retrieve_from_kb(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Pure lexical retrieval over raw KB items (JSON/JSONL).
        Used as a fallback for EXACT_RULE when vector retrieval does not
        contain any keyword-matching chunk.
        """
        items = self._load_kb_items()
        if not items:
            return []

        keywords, numbers = self._extract_keywords(question)
        if not keywords and not numbers:
            return []

        scored: List[Dict[str, Any]] = []
        for it in items:
            title = it.get("title") or ""
            heading = it.get("heading") or ""
            content = it.get("content") or ""
            doc_id = it.get("doc_id") or ""
            doc_title = it.get("doc_title") or ""
            so_hieu = it.get("so_hieu") or ""
            combined_text = f"{title}\n{heading}\n{content}"

            lex_score = self._compute_lexical_score(combined_text, keywords, numbers)
            if lex_score <= 0:
                continue

            scored.append(
                {
                    "article_id": it.get("_id"),
                    "clause_id": None,
                    "text": content,
                    "metadata": {
                        "title": title,
                        "heading": heading,
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "so_hieu": so_hieu,
                    },
                    "score": lex_score,  # use lexical score as base
                }
            )

        if not scored:
            return []

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    
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

