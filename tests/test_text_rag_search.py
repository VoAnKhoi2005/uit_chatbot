import numpy as np

from backend.retrieval.text_rag.vector_store import ChunkVectorStore


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


def test_search_bm25_only_when_local_embedder_disabled(tmp_path):
    """Path 1 without dense embeddings: BM25 alone should still rank and score chunks."""
    store = ChunkVectorStore(tmp_path / "chunks_lite.db", disable_local_embedder=True)
    chunks = [
        {"chunk_id": "c1", "article_id": "A-1", "clause_id": None, "text": "Sinh viên phải đăng ký tín chỉ.", "metadata": {}},
        {"chunk_id": "c2", "article_id": "A-2", "clause_id": None, "text": "Quy định về học phí và lệ phí.", "metadata": {}},
    ]
    store.index_chunks(chunks)  # no embedder needed in lightweight mode
    results = store.search("đăng ký tín chỉ", top_k=2)
    assert results
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] > 0


def test_search_hybrid_fuses_bm25_and_dense(tmp_path):
    """Path 1 fusion: an item strong on both BM25 and dense should outrank one strong on only one."""
    store = ChunkVectorStore(tmp_path / "chunks_hybrid.db")

    class KeywordAndSemanticEmbedder:
        # c1 gets a high dense score for the query; c2 gets a high BM25 score only.
        vectors = {"đăng ký tín chỉ online": 1.0, "Sinh viên phải đăng ký tín chỉ.": 0.95, "Không liên quan gì cả.": -0.9}

        def embed(self, texts):
            return np.array([[self.vectors.get(t, 0.0)] for t in texts], dtype=np.float32)

    embedder = KeywordAndSemanticEmbedder()
    chunks = [
        {"chunk_id": "c1", "article_id": "A-1", "clause_id": None, "text": "Sinh viên phải đăng ký tín chỉ.", "metadata": {}},
        {"chunk_id": "c2", "article_id": "A-2", "clause_id": None, "text": "Không liên quan gì cả.", "metadata": {}},
    ]
    store.index_chunks(chunks, embedder)
    results = store.search("đăng ký tín chỉ online", embedder, top_k=2)
    assert results[0]["chunk_id"] == "c1"
    assert "bm25_score" in results[0] and "dense_score" in results[0]

