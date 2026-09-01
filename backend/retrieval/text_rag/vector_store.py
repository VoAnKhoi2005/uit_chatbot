"""A lightweight SQLite-backed vector store for regulation text chunks.

Path 1 of the dual-path retrieval pipeline lives here: ``search()`` fuses a
BM25 lexical score (good at exact regulation terminology, article numbers,
and other keywords students reuse verbatim) with a dense cosine similarity
score (captures semantic relatedness when the query is phrased differently
from the source text). Each score is min-max normalized within its own
result list before being combined via a weighted linear sum, so neither
signal dominates just because of its raw scale.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .chunker import TextChunk
from .config import UIT_DISABLE_LOCAL_EMBEDDER

# Only import heavy dependencies when local embedding is enabled
if not UIT_DISABLE_LOCAL_EMBEDDER:
    import numpy as np
    from .embeddings import TextEmbedder


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Accent-insensitive tokenization shared by indexing and querying."""
    text = (text or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return _TOKEN_RE.findall(text)


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize a result list's scores to [0, 1] independently of any other list."""
    if not scores:
        return {}
    vmin = min(scores.values())
    vmax = max(scores.values())
    if vmax - vmin < 1e-9:
        return {k: (1.0 if vmax > 0 else 0.0) for k in scores}
    return {k: (v - vmin) / (vmax - vmin) for k, v in scores.items()}


class _BM25Index:
    """Minimal, dependency-free Okapi BM25 index over tokenized chunk texts."""

    def __init__(self, chunk_ids: List[str], tokenized_docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.chunk_ids = chunk_ids
        self.k1 = k1
        self.b = b
        self.doc_len = [len(d) for d in tokenized_docs]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if tokenized_docs else 0.0

        # inverted index: term -> list of (doc_idx, term_freq)
        self.inverted: Dict[str, List[tuple]] = defaultdict(list)
        df: Dict[str, int] = defaultdict(int)
        for doc_idx, doc in enumerate(tokenized_docs):
            freq: Dict[str, int] = defaultdict(int)
            for term in doc:
                freq[term] += 1
            for term, f in freq.items():
                self.inverted[term].append((doc_idx, f))
                df[term] += 1

        n = len(tokenized_docs)
        self.idf: Dict[str, float] = {
            term: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for term, f in df.items()
        }

    def get_scores(self, query_tokens: List[str]) -> Dict[str, float]:
        scores: Dict[int, float] = defaultdict(float)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if not idf:
                continue
            for doc_idx, f in self.inverted.get(term, []):
                dl = self.doc_len[doc_idx]
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[doc_idx] += idf * (f * (self.k1 + 1)) / denom
        return {self.chunk_ids[doc_idx]: score for doc_idx, score in scores.items()}


class ChunkVectorStore:
    def __init__(self, db_path: str | Path, disable_local_embedder: bool | None = None) -> None:
        self.db_path = Path(db_path)
        print(self.db_path)
        self.conn = sqlite3.connect(self.db_path)
        # Allow override of global config via constructor parameter
        self.disable_local_embedder = (
            disable_local_embedder if disable_local_embedder is not None else UIT_DISABLE_LOCAL_EMBEDDER
        )
        self._setup()

        # Caches invalidated (rebuilt) whenever count_chunks() changes.
        self._rows_cache: Optional[List[dict]] = None
        self._rows_cache_count: int = -1
        self._bm25_index: Optional[_BM25Index] = None
        self._bm25_index_count: int = -1
        self._so_hieu_index: Optional[Dict[str, List[dict]]] = None
        self._so_hieu_index_count: int = -1

    def count_chunks(self) -> int:
        """Return number of indexed chunks (rows) currently stored."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunk_vectors")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def clear(self) -> None:
        """Delete every indexed chunk. `index_chunks` only ever upserts by
        chunk_id, so a source item that's removed/renamed (e.g. a duplicate
        dropped at ingestion) leaves its old row behind forever unless the
        table is cleared first - a rebuild (build_index.py) should always
        start from empty, not accumulate stale rows across runs."""
        self.conn.execute("DELETE FROM chunk_vectors")
        self.conn.commit()
        self._rows_cache = None
        self._rows_cache_count = -1
        self._bm25_index = None
        self._bm25_index_count = -1
        self._so_hieu_index = None
        self._so_hieu_index_count = -1

    def _setup(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                chunk_id TEXT PRIMARY KEY,
                article_id TEXT,
                clause_id TEXT,
                text TEXT,
                metadata_json TEXT,
                embedding BLOB
            )
            """
        )
        # Create index for text search when embedding is disabled
        if self.disable_local_embedder:
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunk_text ON chunk_vectors(text)
                """
            )
        self.conn.commit()

    def index_chunks(self, chunks: List[TextChunk], embedder=None) -> None:
        """Index chunks with embeddings. Requires embedder when local embedding is enabled."""
        if not chunks:
            return

        if self.disable_local_embedder:
            # Lightweight mode: store chunks without embeddings
            rows = []
            for chunk in chunks:
                rows.append(
                    (
                        chunk["chunk_id"],
                        chunk["article_id"],
                        chunk["clause_id"],
                        chunk["text"],
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                        None,  # No embedding in lightweight mode
                    )
                )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_vectors
                (chunk_id, article_id, clause_id, text, metadata_json, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        else:
            # Full vector mode: requires embedder
            if embedder is None:
                from .embeddings import TextEmbedder

                embedder = TextEmbedder()
            import numpy as np

            embeddings = embedder.embed([c["text"] for c in chunks])
            rows = []
            for chunk, emb in zip(chunks, embeddings, strict=False):
                rows.append(
                    (
                        chunk["chunk_id"],
                        chunk["article_id"],
                        chunk["clause_id"],
                        chunk["text"],
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
                        emb.astype(np.float32).tobytes(),
                    )
                )
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO chunk_vectors
                (chunk_id, article_id, clause_id, text, metadata_json, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Shared row cache (rebuilt whenever the chunk count changes)
    # ------------------------------------------------------------------

    def _all_rows(self) -> List[dict]:
        count = self.count_chunks()
        if self._rows_cache is not None and self._rows_cache_count == count:
            return self._rows_cache
        cursor = self.conn.execute(
            "SELECT chunk_id, article_id, clause_id, text, metadata_json FROM chunk_vectors"
        )
        rows = []
        for chunk_id, article_id, clause_id, text, metadata_json in cursor.fetchall():
            metadata = json.loads(metadata_json) if metadata_json else {}
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "clause_id": clause_id,
                    "text": text,
                    "metadata": metadata,
                    "doc_id": metadata.get("doc_id"),
                    "doc_title": metadata.get("doc_title"),
                    "so_hieu": metadata.get("so_hieu"),
                }
            )
        self._rows_cache = rows
        self._rows_cache_count = count
        return rows

    def get_chunks_by_so_hieu(self, so_hieu: str) -> List[dict]:
        """Look up chunks belonging to a document by its regulation number (so_hieu).

        Used to map knowledge-graph triples (Path 2) back to their source text
        units, since the triplet export carries the document number rather
        than a chunk-level id.
        """
        count = self.count_chunks()
        if self._so_hieu_index is None or self._so_hieu_index_count != count:
            index: Dict[str, List[dict]] = defaultdict(list)
            for row in self._all_rows():
                if row.get("so_hieu"):
                    index[row["so_hieu"]].append(row)
            self._so_hieu_index = index
            self._so_hieu_index_count = count
        return self._so_hieu_index.get(so_hieu, [])

    def _ensure_bm25(self) -> _BM25Index:
        count = self.count_chunks()
        if self._bm25_index is not None and self._bm25_index_count == count:
            return self._bm25_index
        rows = self._all_rows()
        chunk_ids = [r["chunk_id"] for r in rows]
        tokenized = [_tokenize(r["text"]) for r in rows]
        self._bm25_index = _BM25Index(chunk_ids, tokenized)
        self._bm25_index_count = count
        return self._bm25_index

    def _row_result(self, row: dict, score: float) -> dict:
        return {
            "chunk_id": row["chunk_id"],
            "article_id": row["article_id"],
            "clause_id": row["clause_id"],
            "text": row["text"],
            "metadata": row["metadata"],
            "score": score,
            "doc_id": row.get("doc_id"),
            "doc_title": row.get("doc_title"),
            "so_hieu": row.get("so_hieu"),
        }

    def search_bm25(self, query: str, top_k: int = 5) -> List[dict]:
        """Lexical relevance via BM25 - strong on exact regulation terminology,
        article numbers, and other domain keywords reused verbatim from the source."""
        bm25 = self._ensure_bm25()
        scores = bm25.get_scores(_tokenize(query))
        if not scores:
            return []
        rows_by_id = {r["chunk_id"]: r for r in self._all_rows()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [self._row_result(rows_by_id[cid], score) for cid, score in ranked if cid in rows_by_id]

    def _search_text_only(self, query: str, top_k: int = 5) -> List[dict]:
        """Lightweight, embedder-free search (used when local embedding is disabled)."""
        return self.search_bm25(query, top_k=top_k)

    def search_dense(self, query: str, embedder, top_k: int = 5) -> List[dict]:
        """Semantic relevance via dense cosine similarity over chunk embeddings -
        catches relatedness even when the query is phrased differently from the source."""
        import numpy as np

        query_vec = embedder.embed([query])[0]
        cursor = self.conn.execute(
            "SELECT chunk_id, article_id, clause_id, text, metadata_json, embedding FROM chunk_vectors WHERE embedding IS NOT NULL"
        )
        candidates = []
        for chunk_id, article_id, clause_id, text, metadata_json, emb_blob in cursor.fetchall():
            if emb_blob is None:
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            score = float(
                np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-8)
            )
            metadata = json.loads(metadata_json) if metadata_json else {}
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "clause_id": clause_id,
                    "text": text,
                    "metadata": metadata,
                    "score": score,
                    "doc_id": metadata.get("doc_id"),
                    "doc_title": metadata.get("doc_title"),
                    "so_hieu": metadata.get("so_hieu"),
                }
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

    # Backward-compatible alias for the previous private name.
    _search_vector = search_dense

    def search(
        self, query: str, embedder=None, top_k: int = 5, alpha: float = 0.5, candidate_k: Optional[int] = None
    ) -> List[dict]:
        """
        Path 1 (hybrid text retrieval): fuse BM25 lexical scores with dense
        cosine-similarity scores. Each list is min-max normalized on its own
        scale before being combined via a weighted linear sum

            combined = alpha * norm(bm25) + (1 - alpha) * norm(dense)

        so that a spike in one signal's raw scores can't dominate just
        because of its scale. When local embedding is disabled (lightweight
        deployment), the dense signal is skipped and BM25 alone drives ranking.
        """
        candidate_k = candidate_k or max(top_k * 4, 20)

        bm25_hits = self.search_bm25(query, top_k=candidate_k)
        dense_hits: List[dict] = []
        if not self.disable_local_embedder:
            if embedder is None:
                from .embeddings import TextEmbedder

                embedder = TextEmbedder()
            dense_hits = self.search_dense(query, embedder, top_k=candidate_k)

        if not dense_hits:
            # Lightweight mode, or no embedder available: BM25-only ranking.
            return bm25_hits[:top_k]

        bm25_scores = {h["chunk_id"]: h["score"] for h in bm25_hits}
        dense_scores = {h["chunk_id"]: h["score"] for h in dense_hits}
        bm25_norm = _min_max_normalize(bm25_scores)
        dense_norm = _min_max_normalize(dense_scores)

        rows_by_id = {h["chunk_id"]: h for h in bm25_hits}
        rows_by_id.update({h["chunk_id"]: h for h in dense_hits})

        fused = []
        for chunk_id in set(bm25_norm) | set(dense_norm):
            b = bm25_norm.get(chunk_id, 0.0)
            d = dense_norm.get(chunk_id, 0.0)
            combined = alpha * b + (1 - alpha) * d
            row = dict(rows_by_id[chunk_id])
            row["score"] = combined
            row["bm25_score"] = b
            row["dense_score"] = d
            fused.append(row)
        fused.sort(key=lambda r: r["score"], reverse=True)
        return fused[:top_k]

