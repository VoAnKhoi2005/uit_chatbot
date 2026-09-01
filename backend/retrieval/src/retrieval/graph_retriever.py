"""In-memory knowledge graph retriever (Path 2 of the dual-path retrieval pipeline).

Builds a lightweight subject-relation-object graph from the extracted regulation
triplets and answers queries with the two-stage strategy described in the
"Dual-Path Graph Retrieval" design:

  1. Seeding: an initial set of anchor nodes is found via the union of
     - a keyword search over node labels/aliases (exact/substring/token match), and
     - a semantic search comparing the query embedding against each node's vector.
  2. Expansion: starting from every seed node, the graph is traversed outward
     up to a bounded number of hops, collecting the triples encountered along
     the way so that information connected to the query - but living in a
     different clause/article - can still be recovered. A branch is pruned
     (not traversed further) once it reaches a node that isn't semantically
     close enough to the query, so expansion doesn't drift into unrelated
     territory hop after hop.

A node's vector is not embedded on its own: it's the mean of its own name's
embedding and the embeddings of every triple it participates in - the same
mean-pooling principle sentence-transformer models use to turn token
embeddings into a sentence embedding, applied one level up to turn triple
embeddings into a node embedding. This keeps a node's representation aware of
its graph context while still being directly comparable to a query embedding.

All of those per-node/per-triple embeddings are looked up in (and, if
missing, written back to) ``FaissEmbeddingStore`` (see
``retrieval.src.db.faiss_store``), a small persisted FAISS-backed vector
store, so unchanged text is only ever sent to the embedder once across
process restarts - important once the embedder is a paid/remote API rather
than a local model. The resulting combined node vectors are searched via a
FAISS ``IndexFlatIP`` (exact cosine similarity) rather than a hand-rolled
numpy scan.

This class only reasons about the graph itself (nodes, edges, seeding,
expansion). Mapping the resulting triples back to their source text chunks is
the responsibility of the caller (see ``HybridOrchestrator``), since that step
needs access to the chunk store.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from retrieval.src.db.faiss_store import FaissEmbeddingStore, cosine_index, cosine_search

logger = logging.getLogger(__name__)

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


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    document_number: Optional[str] = None
    document_id: Optional[str] = None

    def verbalize(self) -> str:
        return f"{self.subject} — {self.relation} — {self.object}"

    def cache_id(self) -> str:
        """Stable identity for the embedding cache, independent of display formatting."""
        return "\x1f".join([self.subject, self.relation, self.object, self.document_number or ""])

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
        prune_threshold: Optional[float] = 0.1,
        embedding_cache: Any = None,
        use_embedding_cache: bool = True,
    ) -> None:
        self.triplets_path = Path(triplets_path or DEFAULT_TRIPLETS_PATH)
        self.embedder = embedder
        self.default_max_hops = max_hops
        # Minimum cosine similarity (query vs. node) required to keep expanding
        # *from* a newly-discovered node. None disables pruning entirely.
        self.default_prune_threshold = prune_threshold

        self.triples: List[Triple] = []
        self.nodes: Dict[str, _Node] = {}
        # node_key -> list of (triple_index, other_node_key)
        self.adjacency: Dict[str, List[Tuple[int, str]]] = defaultdict(list)

        self._node_keys: List[str] = []
        self._node_index: Dict[str, int] = {}
        # In-memory faiss.IndexFlatIP over the combined node vectors (rebuilt
        # from the persisted cache each construction - cheap, no embedder calls).
        self._node_faiss_index = None

        # Lazy: only opened once an embedder is actually available and about to
        # be used, so constructing a GraphRetriever without one (e.g. tests,
        # keyword-only mode) never touches faiss/disk at all.
        self._embedding_cache = embedding_cache
        self._use_embedding_cache = use_embedding_cache and embedding_cache is None

        self._load()
        self._node_keys = list(self.nodes.keys())
        self._node_index = {k: i for i, k in enumerate(self._node_keys)}
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
        """Compute each node's vector as the mean of its own name's embedding and
        every triple it participates in, fetching/storing individual pieces
        from ``FaissEmbeddingStore`` so unchanged text is embedded at most once,
        then index the resulting combined vectors for cosine search."""
        if not self._node_keys:
            return

        node_cache_keys = {k: f"node::{k}" for k in self._node_keys}
        triple_cache_keys = [f"triple::{t.cache_id()}" for t in self.triples]

        if self._embedding_cache is None and self._use_embedding_cache:
            try:
                self._embedding_cache = FaissEmbeddingStore(name="graph_embeddings")
            except Exception as e:
                logger.warning("GraphRetriever: embedding cache unavailable, continuing without it: %s", e)
                self._use_embedding_cache = False
        cache = self._embedding_cache
        cached: Dict[str, np.ndarray] = {}
        all_keys = list(node_cache_keys.values()) + triple_cache_keys
        if cache is not None:
            try:
                cached = cache.get_many(all_keys)
            except Exception as e:
                logger.warning("GraphRetriever: embedding cache read failed, ignoring cache: %s", e)
                cached = {}

        text_by_key: Dict[str, str] = {}
        for k in self._node_keys:
            ck = node_cache_keys[k]
            if ck not in cached:
                text_by_key[ck] = self.nodes[k].name
        for i, t in enumerate(self.triples):
            ck = triple_cache_keys[i]
            if ck not in cached:
                text_by_key[ck] = t.verbalize()

        if text_by_key:
            keys_to_embed = list(text_by_key.keys())
            texts_to_embed = [text_by_key[k] for k in keys_to_embed]
            try:
                # Single batched call for everything still missing from the cache -
                # this is the only point where the embedder actually gets invoked.
                fresh = np.asarray(self.embedder.embed(texts_to_embed), dtype=np.float32)
            except Exception as e:
                logger.warning("GraphRetriever: embedding computation failed, falling back to keyword-only seeding: %s", e)
                self._node_faiss_index = None
                return
            fresh_map = {k: fresh[i] for i, k in enumerate(keys_to_embed)}
            cached.update(fresh_map)
            if cache is not None:
                try:
                    cache.add_many(fresh_map)
                except Exception as e:
                    logger.warning("GraphRetriever: embedding cache write failed (continuing in-memory only): %s", e)

        if not cached:
            return
        dim = next(iter(cached.values())).shape[-1]

        triple_vecs = np.zeros((len(self.triples), dim), dtype=np.float32)
        for i, ck in enumerate(triple_cache_keys):
            triple_vecs[i] = cached[ck]

        node_vecs = np.zeros((len(self._node_keys), dim), dtype=np.float32)
        for idx, k in enumerate(self._node_keys):
            parts = [cached[node_cache_keys[k]]]
            for triple_idx, _other in self.adjacency.get(k, []):
                parts.append(triple_vecs[triple_idx])
            node_vecs[idx] = np.mean(parts, axis=0)

        try:
            self._node_faiss_index = cosine_index(node_vecs)
        except Exception as e:
            logger.warning("GraphRetriever: failed to build FAISS index, falling back to keyword-only seeding: %s", e)
            self._node_faiss_index = None

    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        if self.embedder is None:
            return None
        try:
            return np.asarray(self.embedder.embed([query]))[0]
        except Exception:
            return None

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

    def semantic_search(
        self, query: str, top_k: int = 5, query_vec: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:
        """Compare the query embedding against each node's combined vector via FAISS."""
        if self._node_faiss_index is None or self.embedder is None:
            return []
        if query_vec is None:
            query_vec = self._embed_query(query)
            if query_vec is None:
                return []
        hits = cosine_search(self._node_faiss_index, query_vec, top_k)
        return [(self._node_keys[i], score) for i, score in hits if score > 0]

    def seed_nodes(
        self, query: str, top_k: int = 5, query_vec: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Union of the keyword and semantic anchor sets, keeping the best score per node."""
        seeds: Dict[str, float] = {}
        for key, score in self.keyword_search(query, top_k=top_k):
            seeds[key] = max(seeds.get(key, 0.0), score)
        for key, score in self.semantic_search(query, top_k=top_k, query_vec=query_vec):
            seeds[key] = max(seeds.get(key, 0.0), score)
        return seeds

    # ------------------------------------------------------------------
    # Expansion stage
    # ------------------------------------------------------------------

    def expand(
        self,
        seeds: Dict[str, float],
        max_hops: Optional[int] = None,
        query_vec: Optional[np.ndarray] = None,
        prune_threshold: Optional[float] = None,
    ) -> List[Tuple[Triple, float]]:
        """Bounded-hop BFS outward from every seed node, collecting triples.

        A node discovered via an edge is always recorded (the triple that
        reached it is kept), but traversal only continues *from* it - into its
        own neighbors - when its combined vector is at least `prune_threshold`
        similar to the query. This stops a branch from drifting into
        unrelated territory hop after hop, without discarding the one edge
        that connected it to something relevant in the first place.
        """
        max_hops = self.default_max_hops if max_hops is None else max_hops
        if not seeds:
            return []

        threshold = self.default_prune_threshold if prune_threshold is None else prune_threshold
        node_sims: Optional[Dict[int, float]] = None
        if threshold is not None and query_vec is not None and self._node_faiss_index is not None:
            # One full-scan FAISS query for every node's similarity to the query,
            # looked up by row index during the BFS below.
            node_sims = dict(cosine_search(self._node_faiss_index, query_vec, self._node_faiss_index.ntotal))

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
                if other_key in visited:
                    continue
                visited[other_key] = hop + 1

                if node_sims is not None:
                    node_idx = self._node_index.get(other_key)
                    if node_idx is not None and node_sims.get(node_idx, -1.0) < threshold:
                        continue  # prune: keep the triple above, don't expand past this node

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
        prune_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run seeding + expansion, returning raw triples (not yet mapped to text)."""
        query_vec = self._embed_query(query)  # computed once, reused for seeding and pruning
        seeds = self.seed_nodes(query, top_k=seed_top_k, query_vec=query_vec)
        expanded = self.expand(seeds, max_hops=max_hops, query_vec=query_vec, prune_threshold=prune_threshold)
        return [triple.as_dict(score) for triple, score in expanded[:candidate_k]]
