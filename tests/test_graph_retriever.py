"""Tests for GraphRetriever's combined node embeddings, FAISS-backed embedding
cache, and expansion pruning. Skipped entirely if faiss-cpu isn't installed,
since GraphRetriever's embedder-driven paths (everything but keyword-only
search) depend on it."""

import hashlib
import json

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from backend.retrieval.src.db.faiss_store import FaissEmbeddingStore
from backend.retrieval.src.retrieval.graph_retriever import GraphRetriever


class DeterministicEmbedder:
    """Deterministic, hash-derived unit vectors - same text always maps to the
    same vector, distinct texts map to (very likely) distinct directions."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vec(self, text: str) -> np.ndarray:
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        v = rng.normal(size=self.dim)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.array([self._vec(t) for t in texts], dtype=np.float32)

    def calls_count(self) -> int:
        return sum(len(c) for c in self.calls)


def _write_triples(path, triples):
    path.write_text(json.dumps(triples, ensure_ascii=False), encoding="utf-8")


TOY_TRIPLES = [
    {"subject_name": "A", "relation_name": "rel", "object_name": "B", "document_number": "D1"},
    {"subject_name": "B", "relation_name": "rel", "object_name": "C", "document_number": "D1"},
]


def test_keyword_only_fallback_without_embedder(tmp_path):
    """No embedder at all: seeding/expansion still work via keyword search alone,
    and never touch faiss."""
    path = tmp_path / "triplets.json"
    _write_triples(path, TOY_TRIPLES)

    retriever = GraphRetriever(triplets_path=path)  # embedder=None
    assert retriever._node_faiss_index is None
    seeds = retriever.seed_nodes("A", top_k=5)
    assert "a" in seeds
    results = retriever.retrieve("A", max_hops=1)
    assert len(results) == 1  # only the A-B edge, 1 hop


def test_node_vector_is_mean_of_name_and_triple_embeddings(tmp_path):
    """A node's combined vector is the mean of its own name's embedding and the
    embeddings of every triple it participates in."""
    path = tmp_path / "triplets.json"
    _write_triples(path, TOY_TRIPLES)
    cache_dir = tmp_path / "faiss_idx"

    embedder = DeterministicEmbedder()
    cache = FaissEmbeddingStore(name="test", index_dir=cache_dir)
    retriever = GraphRetriever(triplets_path=path, embedder=embedder, embedding_cache=cache)

    expected = np.mean(
        [embedder._vec("B"), embedder._vec("A — rel — B"), embedder._vec("B — rel — C")], axis=0
    )
    hits = retriever.semantic_search("B", top_k=1, query_vec=expected)
    assert hits and hits[0][0] == "b"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-4)  # cosine(expected, stored) ~= 1


def test_embedding_cache_avoids_recomputation_across_restarts(tmp_path):
    """A second GraphRetriever construction over the same triples + cache dir
    must not call the embedder again - everything is served from the persisted
    FAISS-backed cache."""
    path = tmp_path / "triplets.json"
    _write_triples(path, TOY_TRIPLES)
    cache_dir = tmp_path / "faiss_idx"

    embedder1 = DeterministicEmbedder()
    GraphRetriever(
        triplets_path=path, embedder=embedder1, embedding_cache=FaissEmbeddingStore(name="test", index_dir=cache_dir)
    )
    assert embedder1.calls_count() == 5  # 3 node names + 2 triples, one batched call

    embedder2 = DeterministicEmbedder()
    GraphRetriever(
        triplets_path=path, embedder=embedder2, embedding_cache=FaissEmbeddingStore(name="test", index_dir=cache_dir)
    )
    assert embedder2.calls_count() == 0


def test_expand_prunes_branches_below_similarity_threshold(tmp_path):
    """A - B - C - D chain. B is query-relevant, C isn't: with pruning enabled,
    traversal stops at C and never reaches D; without it, D is still found."""
    path = tmp_path / "triplets.json"
    _write_triples(
        path,
        [
            {"subject_name": "A", "relation_name": "rel", "object_name": "B", "document_number": "D1"},
            {"subject_name": "B", "relation_name": "rel", "object_name": "C", "document_number": "D1"},
            {"subject_name": "C", "relation_name": "rel", "object_name": "D", "document_number": "D1"},
        ],
    )
    retriever = GraphRetriever(triplets_path=path)  # embedder=None: build the graph shape only

    # Manually install a controlled 2D index: A/B aligned with the query, C/D orthogonal.
    index = faiss.IndexFlatIP(2)
    node_vecs = np.zeros((len(retriever._node_keys), 2), dtype=np.float32)
    for key, i in retriever._node_index.items():
        node_vecs[i] = [1.0, 0.0] if key in ("a", "b") else [0.0, 1.0]
    index.add(node_vecs)
    retriever._node_faiss_index = index

    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    seeds = {"a": 1.0}

    pruned = retriever.expand(seeds, max_hops=5, query_vec=query_vec, prune_threshold=0.5)
    pruned_edges = sorted((t.subject, t.object) for t, _s in pruned)
    assert pruned_edges == [("A", "B"), ("B", "C")]  # C's own edge onward to D never traversed

    unpruned = retriever.expand(seeds, max_hops=5, query_vec=query_vec, prune_threshold=-1.0)
    unpruned_edges = sorted((t.subject, t.object) for t, _s in unpruned)
    assert unpruned_edges == [("A", "B"), ("B", "C"), ("C", "D")]
