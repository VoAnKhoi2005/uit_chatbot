"""In-memory knowledge graph retriever (Path 2 of the dual-path retrieval pipeline).

Builds a lightweight subject-relation-object graph from the extracted regulation
triplets and answers queries with the two-stage strategy described in the
"Dual-Path Graph Retrieval" design:

  1. Seeding: an initial set of anchor nodes is found via the union of
     - a keyword search over node labels/aliases (exact/substring/token match), and
     - a semantic search comparing the query embedding against node and
       node-neighborhood embeddings.
  2. Expansion: starting from every seed node, the graph is traversed outward
     up to a bounded number of hops, collecting the triples encountered along
     the way so that information connected to the query - but living in a
     different clause/article - can still be recovered.

This class only reasons about the graph itself (nodes, edges, seeding,
expansion). Mapping the resulting triples back to their source text chunks is
the responsibility of the caller (see ``HybridOrchestrator``), since that step
needs access to the chunk store.
"""

from __future__ import annotations

import json
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Default location of the exported (subject, relation, object) triplets.
DEFAULT_TRIPLETS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "graph"
    / "mongo_export_uit"
    / "v3"
    / "KB_UIT.triplets.json"
)


def _normalize(text: str) -> str:
    """Lowercase + strip diacritics + collapse whitespace (accent-insensitive)."""
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(text.split())


def _cosine_sim_matrix(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    if mat is None or mat.size == 0:
        return np.zeros(0)
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return mat_norm @ vec_norm


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    document_number: Optional[str] = None
    document_id: Optional[str] = None

    def as_dict(self, score: float) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.relation,
            "object": self.object,
            "score": score,
            "document_number": self.document_number,
            "document_id": self.document_id,
        }


@dataclass
class _Node:
    name: str
    key: str
    neighbor_terms: Set[str] = field(default_factory=set)


class GraphRetriever:
    """Seed + bounded-hop expansion retrieval over the regulation knowledge graph."""

    def __init__(
        self,
        triplets_path: str | Path | None = None,
        embedder: Any = None,
        max_hops: int = 2,
    ) -> None:
        self.triplets_path = Path(triplets_path or DEFAULT_TRIPLETS_PATH)
        self.embedder = embedder
        self.default_max_hops = max_hops

        self.triples: List[Triple] = []
        self.nodes: Dict[str, _Node] = {}
        # node_key -> list of (triple_index, other_node_key)
        self.adjacency: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        self._node_keys: List[str] = []
        self._node_embeddings: Optional[np.ndarray] = None
        self._neighborhood_embeddings: Optional[np.ndarray] = None

        self._load()
        if self.embedder is not None:
            self._build_embeddings()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.triplets_path.exists():
            return
        with open(self.triplets_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            subject = (item.get("subject_name") or "").strip()
            relation = (item.get("relation_name") or "").strip()
            obj = (item.get("object_name") or "").strip()
            if not subject or not obj:
                continue
            triple = Triple(
                subject=subject,
                relation=relation,
                object=obj,
                document_number=item.get("document_number"),
                document_id=item.get("document_id"),
            )
            idx = len(self.triples)
            self.triples.append(triple)

            s_key = self._add_node(subject)
            o_key = self._add_node(obj)
            self.nodes[s_key].neighbor_terms.add(obj)
            self.nodes[o_key].neighbor_terms.add(subject)
            if relation:
                self.nodes[s_key].neighbor_terms.add(relation)
                self.nodes[o_key].neighbor_terms.add(relation)

            self.adjacency[s_key].append((idx, o_key))
            self.adjacency[o_key].append((idx, s_key))

    def _add_node(self, name: str) -> str:
        key = _normalize(name)
        if key not in self.nodes:
            self.nodes[key] = _Node(name=name, key=key)
        return key

    def _build_embeddings(self) -> None:
        self._node_keys = list(self.nodes.keys())
        if not self._node_keys:
            return
        names = [self.nodes[k].name for k in self._node_keys]
        neighborhoods = [
            self.nodes[k].name + " " + " ".join(sorted(self.nodes[k].neighbor_terms))
            for k in self._node_keys
        ]
        try:
            self._node_embeddings = np.asarray(self.embedder.embed(names))
            self._neighborhood_embeddings = np.asarray(self.embedder.embed(neighborhoods))
        except Exception:
            # Embedder unavailable/misconfigured: seeding falls back to keyword search only.
            self._node_embeddings = None
            self._neighborhood_embeddings = None

    # ------------------------------------------------------------------
    # Seeding stage
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Match query terms against node labels/aliases."""
        q_norm = _normalize(query)
        q_tokens = set(q_norm.split())
        if not q_tokens:
            return []
        scored: List[Tuple[str, float]] = []
        for key, node in self.nodes.items():
            if not key:
                continue
            if key == q_norm:
                scored.append((key, 1.0))
                continue
            if key in q_norm or q_norm in key:
                scored.append((key, 0.85))
                continue
            node_tokens = set(key.split())
            if not node_tokens:
                continue
            overlap = len(node_tokens & q_tokens) / len(node_tokens)
            if overlap > 0:
                scored.append((key, 0.6 * overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def semantic_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Compare the query embedding against node and neighborhood embeddings."""
        if self._node_embeddings is None or self.embedder is None:
            return []
        try:
            query_vec = np.asarray(self.embedder.embed([query]))[0]
        except Exception:
            return []
        node_sims = _cosine_sim_matrix(query_vec, self._node_embeddings)
        nbr_sims = _cosine_sim_matrix(query_vec, self._neighborhood_embeddings)
        if node_sims.size == 0:
            return []
        combined = np.maximum(node_sims, nbr_sims) if nbr_sims.size else node_sims
        order = np.argsort(-combined)[:top_k]
        return [(self._node_keys[i], float(combined[i])) for i in order if combined[i] > 0]

    def seed_nodes(self, query: str, top_k: int = 5) -> Dict[str, float]:
        """Union of the keyword and semantic anchor sets, keeping the best score per node."""
        seeds: Dict[str, float] = {}
        for key, score in self.keyword_search(query, top_k=top_k):
            seeds[key] = max(seeds.get(key, 0.0), score)
        for key, score in self.semantic_search(query, top_k=top_k):
            seeds[key] = max(seeds.get(key, 0.0), score)
        return seeds

    # ------------------------------------------------------------------
    # Expansion stage
    # ------------------------------------------------------------------

    def expand(
        self, seeds: Dict[str, float], max_hops: Optional[int] = None
    ) -> List[Tuple[Triple, float]]:
        """Bounded-hop BFS outward from every seed node, collecting triples."""
        max_hops = self.default_max_hops if max_hops is None else max_hops
        if not seeds:
            return []

        visited: Dict[str, int] = {}
        queue: deque = deque()
        for key in seeds:
            if key in self.nodes and key not in visited:
                visited[key] = 0
                queue.append((key, 0))

        triple_hop: Dict[int, int] = {}
        while queue:
            key, hop = queue.popleft()
            if hop >= max_hops:
                continue
            for triple_idx, other_key in self.adjacency.get(key, []):
                if triple_idx not in triple_hop or hop + 1 < triple_hop[triple_idx]:
                    triple_hop[triple_idx] = hop + 1
                if other_key not in visited:
                    visited[other_key] = hop + 1
                    queue.append((other_key, hop + 1))

        results: List[Tuple[Triple, float]] = []
        for triple_idx, hop in triple_hop.items():
            triple = self.triples[triple_idx]
            seed_bonus = max(
                seeds.get(_normalize(triple.subject), 0.0),
                seeds.get(_normalize(triple.object), 0.0),
            )
            proximity = 1.0 / (1 + hop)
            score = 0.6 * proximity + 0.4 * seed_bonus
            results.append((triple, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        candidate_k: int = 24,
        seed_top_k: int = 5,
        max_hops: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run seeding + expansion, returning raw triples (not yet mapped to text)."""
        seeds = self.seed_nodes(query, top_k=seed_top_k)
        expanded = self.expand(seeds, max_hops=max_hops)
        return [triple.as_dict(score) for triple, score in expanded[:candidate_k]]
