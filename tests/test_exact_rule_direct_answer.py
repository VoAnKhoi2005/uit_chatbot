"""
Test để minh họa fix cho EXACT_RULE: bypass LLM, trả lời trực tiếp từ retrieved rules.

Example usage:
    python -m pytest tests/test_exact_rule_direct_answer.py -v
    hoặc
    python tests/test_exact_rule_direct_answer.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from backend.llm.orchestrator import ChatPipeline


async def test_exact_rule_direct_answer():
    """
    Test: Khi question_type = EXACT_RULE và có retrieved rules,
    chatbot phải trả lời TRỰC TIẾP từ rules, KHÔNG gọi LLM.
    """
    # Mock LLM client (should NOT be called for EXACT_RULE)
    mock_llm_client = MagicMock()

    # Mock vector store để trả về chunks có thông tin về tín chỉ
    mock_vector_store = MagicMock()
    mock_chunks = [
        {
            "article_id": "Điều 14",
            "clause_id": "Khoản 1a",
            "text": (
                "Số tín chỉ đăng ký học n trong mỗi học kỳ chính (bao gồm học lại, cải thiện và học mới) "
                "thỏa điều kiện 14 ≤ n ≤ 24. Sinh viên có ĐTBC ≥ 8,0 đến thời điểm đăng ký, "
                "được phép đăng ký tối đa 30 tín chỉ."
            ),
            "metadata": {
                "title": "Điều 14",
                "section": "Đăng ký học tập"
            },
            "score": 0.95
        }
    ]
    mock_vector_store.search = MagicMock(return_value=mock_chunks)
    
    # Mock ontology
    mock_ontology_graph = MagicMock()
    
    # Tạo pipeline với mocks
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    # Test question
    question = "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?"
    result = await pipeline.answer_question(question)
    
    # Assertions
    assert result["question_type"] == "EXACT_RULE"
    assert len(result["sources"]) > 0, "Phải có sources"
    
    # QUAN TRỌNG: LLM KHÔNG được gọi cho EXACT_RULE
    mock_llm_client.generate.assert_not_called(), "LLM không được gọi cho EXACT_RULE!"
    
    # Answer phải chứa thông tin từ rule
    answer = result["answer"]
    assert "14" in answer or "24" in answer or "30" in answer, f"Answer phải chứa số liệu cụ thể. Got: {answer}"
    assert "Điều 14" in answer or "Khoản 1a" in answer, f"Answer phải tham chiếu đến quy định. Got: {answer}"
    
    # Answer KHÔNG được chứa fallback messages
    assert "Thông tin không đủ" not in answer, f"Answer không được chứa 'Thông tin không đủ'. Got: {answer}"
    assert "Không có thông tin" not in answer, f"Answer không được chứa 'Không có thông tin'. Got: {answer}"
    assert "Tôi không tìm thấy" not in answer, f"Answer không được chứa 'Tôi không tìm thấy'. Got: {answer}"
    
    print(f"✅ Test passed!")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Sources: {len(result['sources'])} chunks")
    print(f"LLM called: {mock_llm_client.generate.called} (should be False)")
    return result


async def test_exact_rule_no_sources():
    """
    Test: Khi question_type = EXACT_RULE nhưng KHÔNG có retrieved rules,
    trả về fallback message.
    """
    mock_llm_client = MagicMock()

    mock_vector_store = MagicMock()
    mock_vector_store.search = MagicMock(return_value=[])  # No chunks

    mock_ontology_graph = MagicMock()

    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )

    question = "Câu hỏi không có trong dữ liệu"
    result = await pipeline.answer_question(question)
    
    assert result["question_type"] == "EXACT_RULE"
    assert len(result["sources"]) == 0, "Không có sources"
    
    # LLM vẫn không được gọi (fallback được trả về trực tiếp)
    mock_llm_client.generate.assert_not_called()
    
    # Answer phải là fallback message
    answer = result["answer"]
    assert "Không tìm thấy" in answer or "quy định phù hợp" in answer, \
        f"Answer phải là fallback message. Got: {answer}"
    
    print(f"✅ Test passed (no sources case)!")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    return result


async def main():
    """Run tests manually."""
    print("=" * 60)
    print("TEST 1: EXACT_RULE với sources")
    print("=" * 60)
    result1 = await test_exact_rule_direct_answer()
    
    print("\n" + "=" * 60)
    print("TEST 2: EXACT_RULE không có sources")
    print("=" * 60)
    result2 = await test_exact_rule_no_sources()
    
    print("\n" + "=" * 60)
    print("KẾT QUẢ MONG ĐỢI SAU KHI FIX:")
    print("=" * 60)
    print(f"Question: Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?")
    print(f"Answer: {result1['answer']}")
    print(f"\n✅ Answer phải:")
    print("   - Chứa số liệu cụ thể: 14-24 tín chỉ, 30 tín chỉ")
    print("   - Tham chiếu: 'Theo Điều 14, Khoản 1a...'")
    print("   - KHÔNG chứa: 'Thông tin không đủ' hoặc 'Không có thông tin'")
    print("   - KHÔNG gọi LLM (bypass hoàn toàn)")


if __name__ == "__main__":
    asyncio.run(main())

