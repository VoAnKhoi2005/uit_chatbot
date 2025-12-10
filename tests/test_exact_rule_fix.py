"""
Test để minh họa fix cho EXACT_RULE: đảm bảo chatbot trả lời dựa trên sources thay vì "Thông tin không đủ".

Example usage:
    python -m pytest tests/test_exact_rule_fix.py -v
    hoặc
    python tests/test_exact_rule_fix.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.llm.orchestrator import ChatPipeline
from backend.llm.question_types import QuestionType


async def test_exact_rule_with_sources():
    """
    Test: Khi question_type = EXACT_RULE và có retrieved sources,
    chatbot phải trả lời dựa trên sources thay vì "Thông tin không đủ".
    """
    # Mock LLM client
    mock_llm_client = MagicMock()
    
    # Mock classify_question để trả về EXACT_RULE
    async def mock_classify(question, client):
        return QuestionType.EXACT_RULE
    
    # Mock generate để trả về answer dựa trên context
    async def mock_generate(system_prompt, user_prompt, context=""):
        # Kiểm tra xem có dùng EXACT_RULE_ANSWER_SYSTEM_PROMPT không
        if "EXACT_RULE" in system_prompt or "BẮT BUỘC phải trả lời" in system_prompt:
            # Nếu có context với thông tin về tín chỉ, trả lời dựa trên đó
            if "14 ≤ n ≤ 24" in context or "30 tín chỉ" in context:
                return (
                    "Theo Điều 14, Khoản 1a: Sinh viên được đăng ký từ 14 đến 24 tín chỉ "
                    "trong mỗi học kỳ chính. Sinh viên có ĐTBC ≥ 8,0 được phép đăng ký tối đa 30 tín chỉ."
                )
        # Fallback (không nên xảy ra với EXACT_RULE + sources)
        return "Thông tin không đủ để trả lời câu hỏi này."
    
    mock_llm_client.generate = AsyncMock(side_effect=mock_generate)
    mock_llm_client.generate_json = AsyncMock(return_value={"label": "EXACT_RULE", "reason": "..."})
    
    # Mock vector store để trả về chunks có thông tin
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
    
    # Patch classify_question
    with patch("backend.llm.orchestrator.classify_question", side_effect=mock_classify):
        # Test question
        question = "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?"
        result = await pipeline.answer_question(question)
    
    # Assertions
    assert result["question_type"] == "EXACT_RULE"
    assert len(result["sources"]) > 0, "Phải có sources"
    
    # QUAN TRỌNG: Answer không được chứa "Thông tin không đủ" hoặc "Không có thông tin"
    answer = result["answer"]
    assert "Thông tin không đủ" not in answer, f"Answer không được chứa 'Thông tin không đủ'. Got: {answer}"
    assert "Không có thông tin" not in answer, f"Answer không được chứa 'Không có thông tin'. Got: {answer}"
    
    # Answer phải chứa thông tin cụ thể từ sources
    assert "14" in answer or "24" in answer or "30" in answer, f"Answer phải chứa số liệu cụ thể. Got: {answer}"
    assert "Điều 14" in answer or "tín chỉ" in answer, f"Answer phải tham chiếu đến quy định. Got: {answer}"
    
    print(f"✅ Test passed!")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f"Sources: {len(result['sources'])} chunks")
    return result


async def main():
    """Run test manually."""
    result = await test_exact_rule_with_sources()
    print("\n" + "=" * 60)
    print("KẾT QUẢ MONG ĐỢI SAU KHI FIX:")
    print("=" * 60)
    print(f"Question: Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?")
    print(f"Answer: {result['answer']}")
    print(f"Question Type: {result['question_type']}")
    print(f"Sources: {len(result['sources'])} chunks")
    print("\n✅ Answer phải chứa:")
    print("   - Số liệu cụ thể: 14-24 tín chỉ, 30 tín chỉ")
    print("   - Tham chiếu: 'Theo Điều 14, Khoản 1a...'")
    print("   - KHÔNG chứa: 'Thông tin không đủ' hoặc 'Không có thông tin'")


if __name__ == "__main__":
    asyncio.run(main())

