import pytest
from retrieval.text_rag.vector_store import ChunkVectorStore
from retrieval.src.retrieval.triplet_retriever import TripletRetriever
from retrieval.src.retrieval.hybrid_orchestrator import HybridOrchestrator, Evidence

class DummyTextStore:
    def search(self, question, top_k=8):
        # Simulate 2 chunks for article A, 1 for B
        return [
            {"chunk_id": "c1", "article_id": "A", "text": "Điều A: Quy định 1", "score": 0.9},
            {"chunk_id": "c2", "article_id": "A", "text": "Điều A: Quy định 2", "score": 0.8},
            {"chunk_id": "c3", "article_id": "B", "text": "Điều B: Quy định", "score": 0.7},
        ]

class DummyTripletRetriever:
    def search_triplets_from_question(self, question, top_k=8):
        # Simulate 2 triplets for A, 1 for C
        return [
            {"subject": "SV", "predicate": "phải", "object": "nộp học phí", "article_id": "A", "score": 0.95},
            {"subject": "SV", "predicate": "được", "object": "miễn học phí", "article_id": "A", "score": 0.7},
            {"subject": "SV", "predicate": "bị", "object": "cảnh cáo", "article_id": "C", "score": 0.6},
        ]

def test_in_scope_exact():
    orchestrator = HybridOrchestrator(DummyTextStore(), DummyTripletRetriever())
    result = orchestrator.run("SV phải nộp học phí", debug=True)
    assert result["grounding"]["article_id"] == "A"
    assert result["grounding"]["dominance"] > 0.65
    assert "GRAPH FACTS" in result["context"]
    assert "TEXT EXCERPTS" in result["context"]

def test_near_case():
    class NearTextStore(DummyTextStore):
        def search(self, question, top_k=8):
            return [
                {"chunk_id": "c1", "article_id": "A", "text": "Điều A", "score": 0.6},
                {"chunk_id": "c2", "article_id": "B", "text": "Điều B", "score": 0.6},
            ]
    class NearTripletRetriever(DummyTripletRetriever):
        def search_triplets_from_question(self, question, top_k=8):
            return [
                {"subject": "SV", "predicate": "phải", "object": "nộp học phí", "article_id": "B", "score": 0.6},
                {"subject": "SV", "predicate": "được", "object": "miễn học phí", "article_id": "A", "score": 0.6},
            ]
    orchestrator = HybridOrchestrator(NearTextStore(), NearTripletRetriever())
    result = orchestrator.run("SV phải nộp học phí", debug=True)
    assert result["grounding"]["dominance"] < 0.65
    assert result["grounding"]["article_id"] in ["A", "B"]

def test_out_case():
    class OutTextStore(DummyTextStore):
        def search(self, question, top_k=8):
            return []
    class OutTripletRetriever(DummyTripletRetriever):
        def search_triplets_from_question(self, question, top_k=8):
            return []
    orchestrator = HybridOrchestrator(OutTextStore(), OutTripletRetriever())
    result = orchestrator.run("không liên quan", debug=True)
    assert result["grounding"]["article_id"] is None
    assert result["grounding"]["dominance"] == 0.0
