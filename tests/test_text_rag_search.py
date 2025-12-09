import numpy as np

from retrieval.text_rag.vector_store import ChunkVectorStore


class DummyEmbedder:
    def embed(self, texts):
        # Deterministic embedding based on text length.
        return np.array([[float(len(t))] for t in texts], dtype=np.float32)


def test_vector_store_index_and_search(tmp_path):
    store = ChunkVectorStore(tmp_path / "chunks.db")
    embedder = DummyEmbedder()

    chunks = [
        {
            "chunk_id": "c1",
            "article_id": "A-1",
            "clause_id": None,
            "text": "hello world",
            "metadata": {},
        },
        {
            "chunk_id": "c2",
            "article_id": "A-2",
            "clause_id": None,
            "text": "short",
            "metadata": {},
        },
    ]
    store.index_chunks(chunks, embedder)
    results = store.search("hello", embedder, top_k=1)
    assert results
    assert results[0]["chunk_id"] == "c1"

