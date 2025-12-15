from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

from backend.retrieval.src.retrieval.triplet_retriever import TripletRetriever
from backend.retrieval.text_rag.vector_store import ChunkVectorStore


@dataclass
class Evidence:
    source: str  # "text" or "graph"
    score: float  # normalized 0..1
    article_id: Optional[str]
    payload: dict
    text: str

def normalize_score(source: str, score: float) -> float:
    """
    Normalize score to [0,1] for fusion.
    Text: cosine similarity [-1,1] or [0,1].
    Graph: TODO - calibrate if needed.
    """
    if source == "text":
        # Assume cosine [-1,1] or [0,1]
        if score < 0:
            return (score + 1) / 2
        return min(max(score, 0.0), 1.0)
    elif source == "graph":
        # TODO: calibrate graph score if needed
        return min(max(score, 0.0), 1.0)
    return 0.0

class HybridOrchestrator:
    def __init__(self, text_store: ChunkVectorStore, triplet_retriever: TripletRetriever):
        self.text_store = text_store
        self.triplet_retriever = triplet_retriever

    def hybrid_retrieve(self, question: str, text_top_k=8, graph_top_k=8) -> Tuple[List[dict], List[dict]]:
        text_hits = self.text_store.search(question, top_k=text_top_k)
        graph_hits = self.triplet_retriever.search_triplets_from_question(question, top_k=graph_top_k)
        return text_hits, graph_hits

    def fuse_hits(self, text_hits: List[dict], graph_hits: List[dict]) -> List[Evidence]:
        evidence = []
        for hit in text_hits:
            evidence.append(Evidence(
                source="text",
                score=normalize_score("text", hit.get("score", 0)),
                article_id=hit.get("article_id"),
                payload=hit,
                text=hit.get("text", "")
            ))
        for hit in graph_hits:
            # Verbalize triplet
            s = hit.get("subject", "")
            p = hit.get("predicate", "")
            o = hit.get("object", "")
            triplet_text = f"{s} — {p} — {o}"
            evidence.append(Evidence(
                source="graph",
                score=normalize_score("graph", hit.get("score", 0)),
                article_id=hit.get("article_id"),
                payload=hit,
                text=triplet_text
            ))
        return evidence

    def rerank_evidence(self, evidence: List[Evidence], question: str) -> List[Evidence]:
        # Simple rerank: lexical overlap + type bonus
        def lexical_overlap(q, t):
            q_set = set(q.lower().split())
            t_set = set(t.lower().split())
            return len(q_set & t_set) / (len(q_set) + 1e-6)
        reranked = []
        for ev in evidence:
            overlap = lexical_overlap(question, ev.text)
            type_bonus = 0.1 if ev.source == "graph" else 0.0
            new_score = 0.7 * ev.score + 0.2 * overlap + 0.1 * type_bonus
            reranked.append(ev.__class__(**{**ev.__dict__, "score": new_score}))
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked

    def select_grounding(self, evidence: List[Evidence]) -> Dict[str, Any]:
        # Group by article_id, sum score
        from collections import defaultdict
        score_by_article = defaultdict(float)
        for ev in evidence:
            if ev.article_id:
                score_by_article[ev.article_id] += ev.score
        if not score_by_article:
            return {"article_id": None, "dominance": 0.0, "candidates": []}
        sorted_items = sorted(score_by_article.items(), key=lambda x: x[1], reverse=True)
        total = sum(score_by_article.values())
        top_id, top_score = sorted_items[0]
        dominance = top_score / (total + 1e-6)
        return {
            "article_id": top_id,
            "dominance": dominance,
            "candidates": [{"article_id": aid, "score": s} for aid, s in sorted_items]
        }

    def build_context(self, evidence: List[Evidence], article_id: Optional[str]) -> str:
        # GRAPH FACTS
        graph_facts = [ev.text for ev in evidence if ev.source == "graph" and ev.article_id == article_id]
        # TEXT EXCERPTS
        text_chunks = [ev.text for ev in evidence if ev.source == "text" and ev.article_id == article_id]
        context = ""
        if graph_facts:
            context += "GRAPH FACTS:\n" + "\n".join(graph_facts) + "\n\n"
        if text_chunks:
            context += "TEXT EXCERPTS:\n" + "\n".join(text_chunks)
        return context.strip()

    def run(self, question: str, text_top_k=8, graph_top_k=8, debug: bool = False) -> Dict[str, Any]:
        text_hits, graph_hits = self.hybrid_retrieve(question, text_top_k, graph_top_k)
        evidence = self.fuse_hits(text_hits, graph_hits)
        evidence = self.rerank_evidence(evidence, question)
        grounding = self.select_grounding(evidence)
        context = self.build_context(evidence, grounding["article_id"])
        result = {
            "context": context,
            "grounding": grounding,
            "text_hits": text_hits[:5],
            "graph_hits": graph_hits[:5],
            "top_evidence_for_debug": [ev.__dict__ for ev in evidence[:10]]
        }
        if not debug:
            # Only return context and grounding in normal mode
            return {"context": context, "grounding": grounding}
        return result