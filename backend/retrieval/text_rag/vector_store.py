"""A lightweight SQLite-backed vector store for regulation text chunks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .chunker import TextChunk
from .config import UIT_DISABLE_LOCAL_EMBEDDER

# Only import heavy dependencies when local embedding is enabled
if not UIT_DISABLE_LOCAL_EMBEDDER:
    import numpy as np
    from .embeddings import TextEmbedder


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

    def count_chunks(self) -> int:
        """Return number of indexed chunks (rows) currently stored."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunk_vectors")
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

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

    def _search_text_only(self, query: str, top_k: int = 5) -> List[dict]:
        """Lightweight text-based search using SQL LIKE."""
        query_terms = query.lower().split()
        # Build a pattern that matches any of the query terms
        patterns = [f"%{term}%" for term in query_terms]

        # Use OR conditions for multiple terms
        where_clause = " OR ".join(["text LIKE ?" for _ in patterns])

        cursor = self.conn.execute(
            f"""
            SELECT chunk_id, article_id, clause_id, text, metadata_json
            FROM chunk_vectors
            WHERE {where_clause}
            ORDER BY 
                CASE 
                    WHEN text LIKE ? THEN 1
                    ELSE 2
                END,
                LENGTH(text) ASC
            LIMIT ?
            """,
            (*patterns, patterns[0] if patterns else "%", top_k),
        )

        results = []
        for chunk_id, article_id, clause_id, text, metadata_json in cursor.fetchall():
            # Simple relevance score: count matching terms
            text_lower = text.lower()
            match_count = sum(1 for term in query_terms if term in text_lower)
            score = match_count / max(len(query_terms), 1)
            
            metadata = json.loads(metadata_json) if metadata_json else {}
            results.append(
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

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _search_vector(self, query: str, embedder, top_k: int = 5) -> List[dict]:
        """Vector-based search using embeddings."""
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

    def search(
        self, query: str, embedder=None, top_k: int = 5
    ) -> List[dict]:
        """
        Search for chunks. Uses text-only search if local embedding is disabled,
        otherwise uses vector search with embedder.
        """
        if self.disable_local_embedder:
            return self._search_text_only(query, top_k)
        else:
            if embedder is None:
                from .embeddings import TextEmbedder

                embedder = TextEmbedder()
            return self._search_vector(query, embedder, top_k)

