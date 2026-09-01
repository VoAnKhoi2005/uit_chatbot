from backend.retrieval.src.retrieval.graph_retriever import GraphRetriever
from backend.retrieval.src.retrieval.hybrid_orchestrator import HybridOrchestrator


class DummyTextStore:
    """Stands in for Path 1 (ChunkVectorStore.search already fuses BM25 + dense)."""

    def __init__(self, hits):
        self._hits = hits
        self.chunks_by_so_hieu = {}

    def search(self, question, top_k=8, embedder=None, alpha=0.5, candidate_k=None):
        return self._hits[:top_k]

    def get_chunks_by_so_hieu(self, so_hieu):
        return self.chunks_by_so_hieu.get(so_hieu, [])


class DummyGraphRetriever:
    """Stands in for Path 2 (GraphRetriever.retrieve: seed + expand, raw triples)."""

    def __init__(self, triples):
        self._triples = triples

    def retrieve(self, question, candidate_k=24, seed_top_k=5, max_hops=2):
        return self._triples[:candidate_k]


def test_in_scope_exact():
    text_store = DummyTextStore(
        [
            {"chunk_id": "c1", "article_id": "A", "clause_id": None, "text": "Điều A: Quy định 1", "score": 0.9},
            {"chunk_id": "c2", "article_id": "A", "clause_id": None, "text": "Điều A: Quy định 2", "score": 0.8},
            {"chunk_id": "c3", "article_id": "B", "clause_id": None, "text": "Điều B: Quy định", "score": 0.7},
        ]
    )
    # Graph triples map onto article A's chunk via so_hieu + subject/object containment.
    text_store.chunks_by_so_hieu["DOC-1"] = [text_store._hits[0]]
    graph_retriever = DummyGraphRetriever(
        [
            {"subject": "SV", "predicate": "phải", "object": "nộp học phí", "score": 0.95, "document_number": "DOC-1"},
            {"subject": "SV", "predicate": "được", "object": "miễn học phí", "score": 0.7, "document_number": "DOC-1"},
        ]
    )
    orchestrator = HybridOrchestrator(text_store, graph_retriever)
    result = orchestrator.run("SV phải nộp học phí", debug=True)

    assert result["grounding"]["article_id"] == "A"
    assert result["grounding"]["dominance"] > 0.5
    assert "GRAPH FACTS" in result["context"]
    assert "TEXT EXCERPTS" in result["context"]


def test_near_case():
    text_store = DummyTextStore(
        [
            {"chunk_id": "c1", "article_id": "A", "clause_id": None, "text": "Điều A", "score": 0.6},
            {"chunk_id": "c2", "article_id": "B", "clause_id": None, "text": "Điều B", "score": 0.6},
        ]
    )
    graph_retriever = DummyGraphRetriever(
        [
            {"subject": "SV", "predicate": "phải", "object": "nộp học phí", "score": 0.6, "document_number": None},
        ]
    )
    orchestrator = HybridOrchestrator(text_store, graph_retriever)
    result = orchestrator.run("SV phải nộp học phí", debug=True)

    assert result["grounding"]["dominance"] < 0.9
    assert result["grounding"]["article_id"] in ["A", "B"]


def test_out_case():
    text_store = DummyTextStore([])
    graph_retriever = DummyGraphRetriever([])
    orchestrator = HybridOrchestrator(text_store, graph_retriever)
    result = orchestrator.run("không liên quan", debug=True)

    assert result["grounding"]["article_id"] is None
    assert result["grounding"]["dominance"] == 0.0


def test_graph_retriever_none_degrades_to_text_only():
    text_store = DummyTextStore(
        [{"chunk_id": "c1", "article_id": "A", "clause_id": None, "text": "Điều A", "score": 0.9}]
    )
    orchestrator = HybridOrchestrator(text_store, graph_retriever=None)
    result = orchestrator.run("bất kỳ câu hỏi nào", debug=True)

    assert result["graph_hits"] == []
    assert result["grounding"]["article_id"] == "A"


def test_reciprocal_rank_fusion_boosts_items_found_by_both_paths():
    text_store = DummyTextStore(
        [
            {"chunk_id": "c1", "article_id": "A", "clause_id": None, "text": "Điều A nội dung", "score": 0.6},
            {"chunk_id": "c2", "article_id": "B", "clause_id": None, "text": "Điều B nội dung", "score": 0.9},
        ]
    )
    text_store.chunks_by_so_hieu["DOC-1"] = [text_store._hits[0]]  # graph triple maps onto c1 (article A)
    graph_retriever = DummyGraphRetriever(
        [{"subject": "Điều A", "predicate": "quy định", "object": "nội dung", "score": 0.5, "document_number": "DOC-1"}]
    )
    orchestrator = HybridOrchestrator(text_store, graph_retriever)
    fused = orchestrator.reciprocal_rank_fusion(
        orchestrator.retrieve_text("q", top_k=8), orchestrator.retrieve_graph("q", top_k=8)
    )
    by_article = {ev["article_id"]: ev for ev in fused}
    # c1/article A is reachable via both paths (chunk_id match) so it should outrank
    # article B despite B's higher standalone text score.
    assert by_article["A"]["rrf_score"] > by_article["B"]["rrf_score"]
    assert by_article["A"]["source"] == "text+graph"


def test_graph_retriever_seed_and_expand_on_toy_graph(tmp_path):
    triples = [
        {"subject_name": "quy chế", "relation_name": "áp dụng đối với", "object_name": "sinh viên", "document_number": "790/QĐ-ĐHCNTT"},
        {"subject_name": "sinh viên", "relation_name": "phải", "object_name": "nộp học phí", "document_number": "790/QĐ-ĐHCNTT"},
    ]
    path = tmp_path / "triplets.json"
    import json

    path.write_text(json.dumps(triples, ensure_ascii=False), encoding="utf-8")

    retriever = GraphRetriever(triplets_path=path)
    assert len(retriever.nodes) == 3
    seeds = retriever.seed_nodes("sinh viên", top_k=5)
    assert any("sinh vien" in k for k in seeds)  # accent-insensitive key match

    # 1-hop expansion from "sinh viên" should reach both adjacent triples.
    results_1hop = retriever.retrieve("sinh viên", max_hops=1)
    assert len(results_1hop) == 2

    # 0-hop expansion should reach no edges at all.
    results_0hop = retriever.retrieve("sinh viên", max_hops=0)
    assert results_0hop == []
