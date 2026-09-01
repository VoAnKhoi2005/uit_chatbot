"""Dual-Path Graph Retrieval orchestrator.

Queries the chunk-level index (Path 1) and the knowledge graph (Path 2) in
parallel and merges their outputs into a single ranked context, so the LLM
gets evidence that is both textually precise and relationally complete.

Path 1 (text): ``ChunkVectorStore.search`` already fuses BM25 lexical scores
with dense cosine-similarity scores (min-max normalized, weighted linear
sum) - see ``retrieval.text_rag.vector_store``.

Path 2 (graph): seed anchor nodes via keyword + semantic search over the
knowledge graph, expand outward a bounded number of hops to recover
relationally-connected facts, then map the collected triples back to their
source text units so citations stay traceable to the regulation text.

Because chunk-level relevance (Path 1) and graph proximity to a seed node
(Path 2) are scored on fundamentally different, non-comparable bases, the
two ranked lists are fused with Reciprocal Rank Fusion (RRF) rather than a
raw weighted sum - this keeps whichever path happens to produce larger
score magnitudes from silently dominating the merged ranking.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from retrieval.src.retrieval.graph_retriever import GraphRetriever
from retrieval.text_rag.vector_store import ChunkVectorStore


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split())


class HybridOrchestrator:
    def __init__(
        self,
        text_store: ChunkVectorStore,
        graph_retriever: Optional[GraphRetriever],
        alpha: float = 0.5,
        rrf_k: int = 60,
    ):
        self.text_store = text_store
        self.graph_retriever = graph_retriever
        # Weight given to the lexical (BM25) signal within Path 1; passed
        # through to ChunkVectorStore.search's min-max-normalized fusion.
        self.alpha = alpha
        # RRF damping constant; higher values flatten the influence of rank.
        self.rrf_k = rrf_k

    # ------------------------------------------------------------------
    # Path 1: hybrid text retrieval (BM25 + dense, already fused in the store)
    # ------------------------------------------------------------------

    def retrieve_text(self, question: str, top_k: int = 8) -> List[dict]:
        return self.text_store.search(question, top_k=top_k, alpha=self.alpha)

    # ------------------------------------------------------------------
    # Path 2: knowledge-graph retrieval (seed -> expand -> map to text)
    # ------------------------------------------------------------------

    def retrieve_graph(self, question: str, top_k: int = 8, seed_top_k: int = 5, max_hops: int = 2) -> List[dict]:
        if self.graph_retriever is None:
            return []
        candidate_k = max(top_k * 3, 15)
        raw_triples = self.graph_retriever.retrieve(
            question, candidate_k=candidate_k, seed_top_k=seed_top_k, max_hops=max_hops
        )
        mapped = self._map_triples_to_text(raw_triples)
        mapped.sort(key=lambda h: h["score"], reverse=True)
        return mapped[:top_k]

    def _map_triples_to_text(self, triples: List[dict]) -> List[dict]:
        """Map collected triples back to their source text units, preserving traceability."""
        lookup = getattr(self.text_store, "get_chunks_by_so_hieu", None)
        results = []
        for t in triples:
            so_hieu = t.get("document_number")
            chunks = lookup(so_hieu) if (lookup and so_hieu) else []
            best_chunk = self._best_matching_chunk(chunks, t.get("subject", ""), t.get("object", ""))
            entry = {
                "subject": t.get("subject"),
                "predicate": t.get("predicate"),
                "object": t.get("object"),
                "score": t.get("score", 0.0),
                "so_hieu": so_hieu,
                "document_id": t.get("document_id"),
            }
            if best_chunk:
                entry.update(
                    {
                        "chunk_id": best_chunk.get("chunk_id"),
                        "article_id": best_chunk.get("article_id"),
                        "clause_id": best_chunk.get("clause_id"),
                        "text": best_chunk.get("text", ""),
                        "doc_id": best_chunk.get("doc_id"),
                        "doc_title": best_chunk.get("doc_title"),
                    }
                )
            else:
                entry.update(
                    {
                        "chunk_id": None,
                        "article_id": None,
                        "clause_id": None,
                        "text": "",
                        "doc_id": None,
                        "doc_title": None,
                    }
                )
            results.append(entry)
        return results

    @staticmethod
    def _best_matching_chunk(chunks: List[dict], subject: str, obj: str) -> Optional[dict]:
        if not chunks:
            return None
        s_norm, o_norm = _normalize(subject), _normalize(obj)
        best, best_score = None, -1
        for c in chunks:
            text_norm = _normalize(c.get("text", ""))
            score = (1 if s_norm and s_norm in text_norm else 0) + (1 if o_norm and o_norm in text_norm else 0)
            if score > best_score:
                best, best_score = c, score
        # Fall back to the document's first chunk when neither term is found verbatim,
        # so the triple still carries a doc-level citation rather than none at all.
        return best if best is not None else chunks[0]

    # ------------------------------------------------------------------
    # Fusion: Reciprocal Rank Fusion of the two ranked lists
    # ------------------------------------------------------------------

    @staticmethod
    def _item_key(item: dict, source: str) -> Tuple:
        if item.get("chunk_id"):
            return ("chunk", item["chunk_id"])
        if source == "graph":
            return ("triple", item.get("subject"), item.get("predicate"), item.get("object"))
        return ("text", item.get("article_id"), item.get("clause_id"), (item.get("text") or "")[:80])

    def reciprocal_rank_fusion(self, text_hits: List[dict], graph_hits: List[dict]) -> List[dict]:
        rrf_scores: Dict[Tuple, float] = defaultdict(float)
        sources_by_key: Dict[Tuple, set] = defaultdict(set)
        payload_by_key: Dict[Tuple, dict] = {}

        for rank, item in enumerate(text_hits, start=1):
            key = self._item_key(item, "text")
            rrf_scores[key] += 1.0 / (self.rrf_k + rank)
            sources_by_key[key].add("text")
            payload_by_key.setdefault(key, {**item, "source": "text"})

        for rank, item in enumerate(graph_hits, start=1):
            key = self._item_key(item, "graph")
            rrf_scores[key] += 1.0 / (self.rrf_k + rank)
            sources_by_key[key].add("graph")
            existing = payload_by_key.get(key)
            if existing is None:
                payload_by_key[key] = {**item, "source": "graph"}
            else:
                # Same underlying text unit reached via both paths: keep the text
                # excerpt and attach the graph triple as a supporting fact.
                existing.setdefault("graph_triples", []).append(
                    {"subject": item.get("subject"), "predicate": item.get("predicate"), "object": item.get("object")}
                )
                existing["source"] = "text+graph"

        fused = []
        for key, score in rrf_scores.items():
            payload = dict(payload_by_key[key])
            payload["score"] = score
            payload["rrf_score"] = score
            payload["sources"] = sorted(sources_by_key[key])
            fused.append(payload)
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Grounding + context assembly
    # ------------------------------------------------------------------

    def select_grounding(self, fused: List[dict]) -> Dict[str, Any]:
        score_by_article: Dict[str, float] = defaultdict(float)
        for ev in fused:
            aid = ev.get("article_id")
            if aid:
                score_by_article[aid] += ev.get("rrf_score", 0.0)
        if not score_by_article:
            return {"article_id": None, "dominance": 0.0, "candidates": []}
        sorted_items = sorted(score_by_article.items(), key=lambda x: x[1], reverse=True)
        total = sum(score_by_article.values())
        top_id, top_score = sorted_items[0]
        dominance = top_score / (total + 1e-6)
        return {
            "article_id": top_id,
            "dominance": dominance,
            "candidates": [{"article_id": aid, "score": s} for aid, s in sorted_items],
        }

    def build_context(self, fused: List[dict], article_id: Optional[str]) -> str:
        graph_facts: List[str] = []
        text_chunks: List[str] = []
        for ev in fused:
            if ev.get("article_id") != article_id:
                continue
            for t in ev.get("graph_triples", []) or []:
                graph_facts.append(f"{t.get('subject','')} — {t.get('predicate','')} — {t.get('object','')}")
            if ev.get("source") == "graph" and not ev.get("text"):
                graph_facts.append(f"{ev.get('subject','')} — {ev.get('predicate','')} — {ev.get('object','')}")
            if ev.get("text"):
                text_chunks.append(ev["text"])

        context = ""
        if graph_facts:
            context += "GRAPH FACTS:\n" + "\n".join(dict.fromkeys(graph_facts)) + "\n\n"
        if text_chunks:
            context += "TEXT EXCERPTS:\n" + "\n".join(dict.fromkeys(text_chunks))
        return context.strip()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, question: str, text_top_k: int = 8, graph_top_k: int = 8, debug: bool = False) -> Dict[str, Any]:
        text_hits = self.retrieve_text(question, top_k=text_top_k)
        graph_hits = self.retrieve_graph(question, top_k=graph_top_k)
        fused = self.reciprocal_rank_fusion(text_hits, graph_hits)
        grounding = self.select_grounding(fused)
        context = self.build_context(fused, grounding["article_id"])

        result: Dict[str, Any] = {
            "context": context,
            "grounding": grounding,
            "top_evidence_for_debug": fused[:10],
        }
        if debug:
            result["text_hits"] = text_hits[:5]
            result["graph_hits"] = graph_hits[:5]
            result["fused_evidence"] = fused[:10]
        return result
