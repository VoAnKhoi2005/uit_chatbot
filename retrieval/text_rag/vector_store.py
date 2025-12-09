"""A lightweight SQLite-backed vector store for regulation text chunks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List

import numpy as np

from .chunker import TextChunk
from .embeddings import TextEmbedder


class ChunkVectorStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self._setup()

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
        self.conn.commit()

    def index_chunks(self, chunks: List[TextChunk], embedder: TextEmbedder) -> None:
        if not chunks:
            return
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

    def search(
        self, query: str, embedder: TextEmbedder, top_k: int = 5
    ) -> List[dict]:
        query_vec = embedder.embed([query])[0]
        cursor = self.conn.execute(
            "SELECT chunk_id, article_id, clause_id, text, metadata_json, embedding FROM chunk_vectors"
        )
        candidates = []
        for chunk_id, article_id, clause_id, text, metadata_json, emb_blob in cursor.fetchall():
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            score = float(np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-8))
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "article_id": article_id,
                    "clause_id": clause_id,
                    "text": text,
                    "metadata": json.loads(metadata_json) if metadata_json else {},
                    "score": score,
                }
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_k]

