"""Build a vector index from UIT regulation content."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from retrieval.text_rag.chunker import iter_all_chunks
from retrieval.text_rag.embeddings import TextEmbedder
from retrieval.text_rag.vector_store import ChunkVectorStore


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    content_path = Path(os.getenv("UIT_CONTENT_JSON", r"E:\Github\uit_chatbot\backend\graph\mongo_export_uit\KB_UIT.items.json"))
    db_path = Path(os.getenv("UIT_VECTOR_DB", "vector_store.db"))
    print("DB Path:", db_path)
    max_chars = int(os.getenv("UIT_CHUNK_MAX_CHARS", "800"))

    # build_index always uses full embedding (offline indexing)
    embedder = TextEmbedder()
    # Explicitly disable lightweight mode for indexing
    store = ChunkVectorStore(db_path, disable_local_embedder=False)

    batch = []
    seen_docs: set[tuple[str, str | None]] = set()
    total_chunks = 0

    for chunk in iter_all_chunks(content_path, max_chars=max_chars):
        key = (chunk["article_id"], chunk["clause_id"])
        seen_docs.add(key)
        batch.append(chunk)
        total_chunks += 1
        if len(batch) >= 128:
            store.index_chunks(batch, embedder)
            batch.clear()
    if batch:
        store.index_chunks(batch, embedder)

    logging.info(
        "Indexed %s document units into %s chunks at %s",
        len(seen_docs),
        total_chunks,
        db_path,
    )


if __name__ == "__main__":
    main()

