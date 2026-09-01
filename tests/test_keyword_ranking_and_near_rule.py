"""
Test để đảm bảo keyword-aware ranking cho EXACT_RULE và NEAR_RULE answers hoạt động đúng.

Example usage:
    python -m pytest tests/test_keyword_ranking_and_near_rule.py -v
    hoặc
    python tests/test_keyword_ranking_and_near_rule.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.llm.orchestrator import ChatPipeline


# Demo questions
Q1 = "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?"
Q2 = "Em rớt 3 môn thì có bị sao không ạ?"


async def test_q1_keyword_ranking():
    """
    TASK 1 - EXACT_RULE keyword-aware ranking
    Q1: "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?"
    
    Test that the system picks the rule with credit limits (14-24, 30) instead of a generic rule.
    """
    mock_llm_client = MagicMock()

    mock_vector_store = MagicMock()

    # Simulate retrieved chunks: Rule A (wrong, generic) and Rule B (correct, has credit limits)
    mock_chunks = [
        {
            "article_id": "Điều 13",
            "clause_id": "Khoản 1",
            "text": "Sinh viên được phép đăng ký học những môn học được mở trong học kỳ theo quy định của nhà trường.",
            "metadata": {"title": "Điều 13"},
            "score": 0.85,  # Higher similarity score (wrong rule)
        },
        {
            "article_id": "Điều 14",
            "clause_id": "Khoản 1a",
            "text": (
                "Số tín chỉ đăng ký học n trong mỗi học kỳ chính (bao gồm học lại, cải thiện và học mới) "
                "thỏa điều kiện 14 ≤ n ≤ 24. Sinh viên có ĐTBC ≥ 8,0 đến thời điểm đăng ký, "
                "được phép đăng ký tối đa 30 tín chỉ."
            ),
            "metadata": {"title": "Điều 14"},
            "score": 0.75,  # Lower similarity score (correct rule)
        }
    ]
    
    mock_vector_store.search = MagicMock(return_value=mock_chunks)
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    result = await pipeline.answer_question(Q1)
    
    # Assertions
    assert result["question_type"] == "EXACT_RULE"
    
    # LLM KHÔNG được gọi
    mock_llm_client.generate.assert_not_called()
    
    # Answer phải chứa số liệu cụ thể từ Rule B (correct rule)
    answer = result["answer"]
    assert "14" in answer or "24" in answer or "30" in answer, \
        f"Answer phải chứa số liệu cụ thể (14, 24, 30). Got: {answer}"
    
    # Answer KHÔNG được chứa text từ Rule A (wrong rule)
    assert "được mở trong học kỳ" not in answer, \
        f"Answer không được chứa text từ rule sai. Got: {answer}"
    
    # Answer phải quote rule text với credit limits
    assert ("14 ≤ n ≤ 24" in answer or "tối đa 30 tín chỉ" in answer or "ĐTBC ≥ 8,0" in answer), \
        f"Answer phải quote rule text với credit limits. Got: {answer}"
    
    # Answer phải tham chiếu Điều 14 (correct article)
    assert "Điều 14" in answer, \
        f"Answer phải tham chiếu Điều 14. Got: {answer}"
    
    print(f"✅ Q1 Keyword Ranking Test passed!")
    print(f"Answer: {answer[:150]}...")
    return result


async def test_q2_near_rule_helpful():
    """
    TASK 2 - NEAR_RULE helpful answer
    Q2: "Em rớt 3 môn thì có bị sao không ạ?"
    
    Test that the system gives a helpful answer mapping informal language to regulations,
    instead of saying "không có thông tin".
    """
    mock_llm_client = MagicMock()

    async def mock_generate(system_prompt, user_prompt, context=""):
        # LLM should map "rớt 3 môn" to academic warning regulations
        # and answer based on available rules in context
        if "rớt" in user_prompt.lower() or "cảnh báo" in context.lower():
            return (
                "Quy chế thường xét theo điểm trung bình học kỳ (ĐTBHK) và số tín chỉ tích lũy "
                "hơn là 'rớt bao nhiêu môn' cụ thể. Nếu em rớt nhiều môn dẫn đến ĐTBHK xuống dưới 3,0 "
                "hoặc điểm trung bình của liên tiếp 2 học kỳ gần nhất đều dưới 4,0, em có thể bị cảnh báo học vụ. "
                "Thời hạn cảnh báo kéo dài trong một học kỳ chính tiếp theo."
            )
        return "Default answer"
    
    mock_llm_client.generate = AsyncMock(side_effect=mock_generate)
    mock_llm_client.generate_json = AsyncMock(return_value={"label": "IN_SCOPE", "reason": "..."})

    mock_vector_store = MagicMock()
    # Simulate some academic warning rules in context
    mock_chunks = [
        {
            "article_id": "Điều 16",
            "clause_id": "Khoản 1",
            "text": (
                "Sinh viên bị cảnh báo học vụ nếu vi phạm một trong những trường hợp sau đây: "
                "Không hoàn thành nghĩa vụ học phí đúng quy định. "
                "Có ĐTBHK dưới 3,0 hoặc điểm trung bình của liên tiếp 2 học kỳ gần nhất đều dưới 4,0."
            ),
            "metadata": {"title": "Điều 16"},
            "score": 0.80
        }
    ]
    mock_vector_store.search = MagicMock(return_value=mock_chunks)
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    result = await pipeline.answer_question(Q2)
    
    assert result["question_type"] == "NEAR_RULE", \
        f"Expected NEAR_RULE, got {result['question_type']}"
    
    # LLM should be called for NEAR_RULE
    assert mock_llm_client.generate.called, "LLM should be called for NEAR_RULE"
    
    # Answer should NOT say "không có thông tin" or "không đề cập"
    answer = result["answer"]
    assert "không có thông tin" not in answer.lower(), \
        f"Answer should not say 'không có thông tin'. Got: {answer}"
    assert "không đề cập" not in answer.lower(), \
        f"Answer should not say 'không đề cập'. Got: {answer}"
    assert "cần thêm thông tin" not in answer.lower(), \
        f"Answer should not ask for more info. Got: {answer}"
    
    # Answer should map informal language to regulations
    assert ("cảnh báo" in answer.lower() or "đtbhk" in answer.lower() or "điểm trung bình" in answer.lower()), \
        f"Answer should map 'rớt môn' to academic warning/GPA regulations. Got: {answer}"
    
    # Answer should reference the available rules
    assert ("3,0" in answer or "4,0" in answer or "dưới" in answer.lower()), \
        f"Answer should reference GPA thresholds from rules. Got: {answer}"
    
    print(f"✅ Q2 NEAR_RULE Helpful Answer Test passed!")
    print(f"Answer: {answer[:150]}...")
    return result


async def test_keyword_extraction():
    """Test keyword extraction function."""
    pipeline = ChatPipeline()
    
    # Test credit question
    keywords, numbers = pipeline._extract_keywords(Q1)
    assert "tín chỉ" in keywords or "học kỳ" in keywords or "đăng ký" in keywords, \
        f"Should extract credit keywords. Got: {keywords}"
    # Numbers might be extracted but not guaranteed for this question
    
    # Test warning question
    keywords2, numbers2 = pipeline._extract_keywords("Em bị cảnh báo học vụ thì sao?")
    assert "cảnh báo" in keywords2 or "học vụ" in keywords2, \
        f"Should extract warning keywords. Got: {keywords2}"
    
    print(f"✅ Keyword Extraction Test passed!")
    print(f"Q1 keywords: {keywords}, numbers: {numbers}")
    print(f"Warning keywords: {keywords2}, numbers: {numbers2}")


async def test_lexical_scoring():
    """Test lexical scoring function."""
    pipeline = ChatPipeline()
    
    keywords = ["tín chỉ", "học kỳ", "tối đa"]
    numbers = ["14", "24", "30"]
    
    # Rule with keywords and numbers (should score high)
    rule_good = "Số tín chỉ đăng ký học n trong mỗi học kỳ chính thỏa điều kiện 14 ≤ n ≤ 24. Tối đa 30 tín chỉ."
    score_good = pipeline._compute_lexical_score(rule_good, keywords, numbers)
    
    # Rule without keywords (should score low)
    rule_bad = "Sinh viên được phép đăng ký học những môn học được mở trong học kỳ."
    score_bad = pipeline._compute_lexical_score(rule_bad, keywords, numbers)
    
    assert score_good > score_bad, \
        f"Rule with keywords should score higher. Good: {score_good}, Bad: {score_bad}"
    
    print(f"✅ Lexical Scoring Test passed!")
    print(f"Good rule score: {score_good}, Bad rule score: {score_bad}")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("KEYWORD RANKING & NEAR_RULE TESTS")
    print("=" * 60)
    
    print("\n1. Testing keyword extraction...")
    await test_keyword_extraction()
    
    print("\n2. Testing lexical scoring...")
    await test_lexical_scoring()
    
    print("\n3. Testing Q1 keyword ranking (EXACT_RULE)...")
    await test_q1_keyword_ranking()
    
    print("\n4. Testing Q2 NEAR_RULE helpful answer...")
    await test_q2_near_rule_helpful()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

