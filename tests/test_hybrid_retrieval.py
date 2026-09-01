"""
Test để đảm bảo hybrid retrieval, query rewriting, và multi-turn query building hoạt động đúng.

Example usage:
    python -m pytest tests/test_hybrid_retrieval.py -v
    hoặc
    python tests/test_hybrid_retrieval.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.llm.orchestrator import ChatPipeline


def test_vietnamese_normalization():
    """Test Vietnamese text normalization."""
    pipeline = ChatPipeline()
    
    # Test accent removal
    assert pipeline._normalize_vietnamese("Cảnh báo học vụ") == "canh bao hoc vu"
    assert pipeline._normalize_vietnamese("Khóa luận tốt nghiệp") == "khoa luan tot nghiep"
    assert pipeline._normalize_vietnamese("Điểm rèn luyện") == "diem ren luyen"
    
    print("✅ Vietnamese normalization test passed!")


def test_keyword_extraction():
    """Test keyword extraction from query."""
    pipeline = ChatPipeline()
    
    # Test stopword removal
    keywords = pipeline._extract_keywords_from_query("Em rớt 3 môn thì có bị sao không ạ?")
    assert "em" not in keywords, "Stopwords should be removed"
    assert "thi" not in keywords, "Stopwords should be removed"
    assert "co" in keywords or "bi" in keywords or "sao" in keywords, "Important words should remain"
    
    keywords2 = pipeline._extract_keywords_from_query("Cảnh báo học vụ là gì?")
    assert "canh" in keywords2 or "bao" in keywords2, "Domain keywords should be extracted"
    
    print("✅ Keyword extraction test passed!")


def test_hybrid_lexical_scoring():
    """Test hybrid lexical scoring with bigram bonuses."""
    pipeline = ChatPipeline()
    
    keywords = ["canh", "bao", "hoc", "vu"]
    
    # Test bigram bonus
    text1 = "Sinh viên bị cảnh báo học vụ nếu vi phạm quy định."
    score1 = pipeline._compute_hybrid_lexical_score(text1, keywords)
    
    text2 = "Sinh viên được phép đăng ký học những môn học được mở trong học kỳ."
    score2 = pipeline._compute_hybrid_lexical_score(text2, keywords)
    
    assert score1 > score2, f"Text with 'cảnh báo học vụ' should score higher. Got: {score1} vs {score2}"
    
    # Test KLTN bigram
    keywords3 = ["khoa", "luan", "tot", "nghiep", "nop"]
    text3 = "Cách nộp khóa luận tốt nghiệp theo quy định."
    score3 = pipeline._compute_hybrid_lexical_score(text3, keywords3)
    
    text4 = "Điều kiện về điểm trung bình để tốt nghiệp."
    score4 = pipeline._compute_hybrid_lexical_score(text4, keywords3)
    
    assert score3 > score4, f"Text with 'khóa luận tốt nghiệp' should score higher. Got: {score3} vs {score4}"
    
    print("✅ Hybrid lexical scoring test passed!")


async def test_hybrid_retrieval_ranking():
    """
    Test that hybrid retrieval correctly ranks chunks.
    For a question about "cảnh báo học vụ", a snippet about warning should outrank one about KLTN.
    """
    mock_vector_store = MagicMock()
    
    # Simulate retrieved chunks with different topics
    mock_chunks = [
        {
            "article_id": "Điều 20",
            "clause_id": "Khoản 1",
            "text": "Cách nộp khóa luận tốt nghiệp: Sinh viên nộp khóa luận tại Phòng Đào tạo...",
            "metadata": {"title": "Điều 20"},
            "score": 0.85,  # Higher embedding score (wrong topic)
        },
        {
            "article_id": "Điều 16",
            "clause_id": "Khoản 1",
            "text": "Sinh viên bị cảnh báo học vụ nếu vi phạm một trong những trường hợp sau đây: Có ĐTBHK dưới 3,0...",
            "metadata": {"title": "Điều 16"},
            "score": 0.75,  # Lower embedding score (correct topic)
        }
    ]
    
    mock_vector_store.search = MagicMock(return_value=mock_chunks)
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    # Test hybrid retrieval with "cảnh báo học vụ" question
    query = "Điều kiện để bị cảnh báo học vụ là gì?"
    ranked_chunks = pipeline._hybrid_retrieve(query, top_k=2)
    
    # The chunk about "cảnh báo học vụ" should be ranked first
    assert len(ranked_chunks) > 0, "Should return at least one chunk"
    top_chunk = ranked_chunks[0]
    assert "cảnh báo" in top_chunk["text"].lower() or "học vụ" in top_chunk["text"].lower(), \
        f"Top chunk should be about 'cảnh báo học vụ'. Got: {top_chunk['text'][:50]}"
    
    print("✅ Hybrid retrieval ranking test passed!")
    print(f"Top chunk: {top_chunk['text'][:80]}...")


async def test_query_rewriting():
    """Test query rewriting for NEAR_RULE questions."""
    mock_llm_client = MagicMock()
    
    async def mock_rewrite(system_prompt, user_prompt, context=""):
        if "rớt 3 môn" in user_prompt:
            return "Quy định về cảnh báo học vụ và xử lý khi sinh viên rớt nhiều môn, điểm trung bình học kỳ thấp."
        return "Default rewritten query"
    
    mock_llm_client.generate = AsyncMock(side_effect=mock_rewrite)
    
    mock_vector_store = MagicMock()
    mock_vector_store.search = MagicMock(return_value=[])
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    # Test query rewriting
    original_question = "Em rớt 3 môn thì có bị sao không ạ?"
    rewritten = await pipeline._rewrite_query_for_regulations(original_question)
    
    assert "cảnh báo" in rewritten.lower() or "học vụ" in rewritten.lower() or "điểm trung bình" in rewritten.lower(), \
        f"Rewritten query should contain regulation keywords. Got: {rewritten}"
    assert "rớt 3 môn" not in rewritten or "có bị sao" not in rewritten, \
        f"Rewritten query should not contain informal phrases. Got: {rewritten}"
    
    print("✅ Query rewriting test passed!")
    print(f"Original: {original_question}")
    print(f"Rewritten: {rewritten}")


async def test_multi_turn_query_building():
    """Test multi-turn query building with discourse markers."""
    mock_llm_client = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.search = MagicMock(return_value=[])
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    # Test with discourse marker
    conversation_history = [
        {"role": "user", "content": "Em rớt 3 môn thì có bị sao không ạ?"},
        {"role": "bot", "content": "Quy chế thường xét theo điểm trung bình..."}
    ]
    
    current_question = "Vậy nếu em bị cảnh báo thì có ảnh hưởng gì không?"
    retrieval_query = await pipeline._build_retrieval_query_async(
        current_question,
        conversation_history
    )

    # Should combine previous question with current
    assert "rớt 3 môn" in retrieval_query.lower() or "cảnh báo" in retrieval_query.lower(), \
        f"Multi-turn query should include context. Got: {retrieval_query}"

    # Test without discourse marker
    current_question2 = "Điều kiện để bị cảnh báo học vụ là gì?"
    retrieval_query2 = await pipeline._build_retrieval_query_async(
        current_question2,
        conversation_history
    )

    # Should NOT combine (no discourse marker): the previous turn's question
    # must not be pulled into this one, regardless of how the (always-on)
    # regulation-oriented rewrite phrases the retrieval query itself.
    assert "rớt 3 môn" not in retrieval_query2.lower(), \
        f"Query without discourse marker should not pull in the previous turn. Got: {retrieval_query2}"
    
    print("✅ Multi-turn query building test passed!")
    print(f"With discourse marker: {retrieval_query[:100]}...")
    print(f"Without discourse marker: {retrieval_query2[:100]}...")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("HYBRID RETRIEVAL TESTS")
    print("=" * 60)
    
    print("\n1. Testing Vietnamese normalization...")
    test_vietnamese_normalization()
    
    print("\n2. Testing keyword extraction...")
    test_keyword_extraction()
    
    print("\n3. Testing hybrid lexical scoring...")
    test_hybrid_lexical_scoring()
    
    print("\n4. Testing hybrid retrieval ranking...")
    await test_hybrid_retrieval_ranking()
    
    print("\n5. Testing query rewriting...")
    await test_query_rewriting()
    
    print("\n6. Testing multi-turn query building...")
    await test_multi_turn_query_building()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

