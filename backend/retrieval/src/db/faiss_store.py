"""FAISS-backed persisted vector store for embeddings.

Used by ``GraphRetriever`` (and anything else that needs a small, on-disk,
searchable embedding index) instead of hand-rolled numpy cosine similarity or
a raw sqlite blob table. Two things live here:

- ``FaissEmbeddingStore``: a persisted key -> vector cache. Exists so a piece
  of text (a node's name, a triple's verbalized text, ...) only ever gets
  embedded once across process restarts - a later run with the same key is
  served straight from disk instead of calling the embedder again, which
  matters once the embedder is a paid/remote API rather than a local model.
- ``cosine_index``: build a plain, in-memory ``faiss.IndexFlatIP`` over a set
  of vectors for exact cosine-similarity search (nearest-neighbor lookup and
  full similarity scans alike), replacing ad-hoc numpy dot products.

Kept independent of ``retrieval.src.db.vector_db`` (the legacy
Mongo-exported concept/relation store) - this is a separate, dedicated
vector index, not another table bolted onto that file. ``faiss`` is imported
lazily inside each function/method here, so importing this module - or
anything that imports it without ever touching embeddings (e.g. keyword-only
/ lightweight deployments) - doesn't require it to be installed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path(__file__).resolve().parent / "faiss_indexes"


class FaissEmbeddingStore:
    """Persisted key -> vector cache backed by a FAISS ``IndexIDMap2``/``IndexFlatIP``.

    Not meant for large-scale nearest-neighbor search (see ``cosine_index``
    for that) - this is deliberately a plain get/put cache keyed by a stable
    string id, with FAISS doing the on-disk vector storage instead of a
    hand-rolled sqlite blob table.
    """

    def __init__(self, name: str, index_dir: str | Path | None = None):
        import faiss  # lazy: only needed once embeddings are actually used

        self._faiss = faiss
        self.name = name
        self.index_dir = Path(index_dir or DEFAULT_INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / f"{name}.faiss"
        self.meta_path = self.index_dir / f"{name}.meta.json"

        self.dim: int | None = None
        self.key_to_id: Dict[str, int] = {}
        self.id_to_key: Dict[int, str] = {}
        self._next_id = 0
        self._index = None  # faiss.IndexIDMap2, built once dim is known

        self._load()

    # -- persistence --------------------------------------------------

    def _load(self) -> None:
        if self.meta_path.exists():
            try:
                meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                self.dim = meta.get("dim")
                self.key_to_id = {k: int(v) for k, v in meta.get("key_to_id", {}).items()}
                self.id_to_key = {v: k for k, v in self.key_to_id.items()}
                self._next_id = meta.get("next_id", (max(self.id_to_key) + 1) if self.id_to_key else 0)
            except Exception as e:
                logger.warning("FaissEmbeddingStore(%s): failed to read metadata, starting fresh: %s", self.name, e)
                self.dim, self.key_to_id, self.id_to_key, self._next_id = None, {}, {}, 0

        if self.dim is not None and self.index_path.exists():
            try:
                self._index = self._faiss.read_index(str(self.index_path))
            except Exception as e:
                logger.warning("FaissEmbeddingStore(%s): failed to read index, rebuilding: %s", self.name, e)
                self._index = self._new_index(self.dim)
        elif self.dim is not None:
            self._index = self._new_index(self.dim)

    def _new_index(self, dim: int):
        return self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(dim))

    def save(self) -> None:
        if self._index is None:
            return
        try:
            self._faiss.write_index(self._index, str(self.index_path))
            self.meta_path.write_text(
                json.dumps({"dim": self.dim, "next_id": self._next_id, "key_to_id": self.key_to_id}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("FaissEmbeddingStore(%s): failed to persist (continuing in-memory only): %s", self.name, e)

    # -- read/write -----------------------------------------------------

    def get_many(self, keys: List[str]) -> Dict[str, np.ndarray]:
        if self._index is None:
            return {}
        found: Dict[str, np.ndarray] = {}
        for key in keys:
            fid = self.key_to_id.get(key)
            if fid is None:
                continue
            try:
                found[key] = np.asarray(self._index.reconstruct(fid), dtype=np.float32)
            except Exception:
                continue
        return found

    def add_many(self, items: Dict[str, np.ndarray]) -> None:
        """Insert/overwrite vectors for the given keys and persist immediately."""
        if not items:
            return
        dim = int(next(iter(items.values())).shape[-1])
        if self._index is None:
            self.dim = dim
            self._index = self._new_index(dim)

        ids: List[int] = []
        vectors: List[np.ndarray] = []
        stale_ids: List[int] = []
        for key, vec in items.items():
            existing = self.key_to_id.get(key)
            if existing is not None:
                stale_ids.append(existing)  # FAISS IDMap has no in-place update: drop, then re-add below.
            else:
                self.key_to_id[key] = self._next_id
                self.id_to_key[self._next_id] = key
                self._next_id += 1
            ids.append(self.key_to_id[key])
            vectors.append(np.asarray(vec, dtype=np.float32))

        if stale_ids:
            try:
                self._index.remove_ids(np.array(stale_ids, dtype=np.int64))
            except Exception as e:
                logger.warning("FaissEmbeddingStore(%s): failed to drop stale ids before re-add: %s", self.name, e)

        mat = np.vstack(vectors).astype(np.float32)
        self._faiss.normalize_L2(mat)  # store unit vectors so inner product == cosine similarity
        self._index.add_with_ids(mat, np.array(ids, dtype=np.int64))
        self.save()

    def close(self) -> None:
        pass


def cosine_index(vectors: np.ndarray):
    """Build a plain in-memory ``faiss.IndexFlatIP`` over L2-normalized `vectors`
    for exact cosine-similarity search. Row order is preserved: row *i* of
    `vectors` is result index *i* from `.search(...)`."""
    import faiss

    mat = np.ascontiguousarray(vectors, dtype=np.float32).copy()
    faiss.normalize_L2(mat)
    index = faiss.IndexFlatIP(mat.shape[1])
    index.add(mat)
    return index


def cosine_search(index, query_vec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    """Query a `cosine_index(...)` index, returning up to `top_k` (row_index, score) pairs."""
    import faiss

    if index is None or index.ntotal == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1).copy()
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, min(top_k, index.ntotal))
    return [(int(i), float(s)) for s, i in zip(scores[0], idxs[0]) if i != -1]
