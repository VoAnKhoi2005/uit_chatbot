"""
Test để đảm bảo demo script hoạt động đúng theo yêu cầu.

Example usage:
    python -m pytest tests/test_demo_script.py -v
    hoặc
    python tests/test_demo_script.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from backend.llm.orchestrator import ChatPipeline
from backend.llm.question_types import QuestionType


# Demo questions
Q1 = "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?"
Q2 = "Em rớt 3 môn thì có bị sao không ạ?"
Q3 = "Theo thầy em nên học lại hay rút môn thì tốt hơn?"
Q4 = "Điều kiện để bị cảnh báo học vụ là gì?"
Q5 = "Vậy nếu em bị cảnh báo thì có ảnh hưởng gì không?"


async def test_q1_exact_rule():
    """
    Segment 2 – EXACT_RULE + Sources (Must-have #1)
    Q1: "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?"
    """
    mock_llm_client = MagicMock()
    
    async def mock_classify(question, client):
        # Should classify as EXACT_RULE
        return QuestionType.EXACT_RULE
    
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
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    with patch("backend.llm.orchestrator.classify_question", side_effect=mock_classify):
        result = await pipeline.answer_question(Q1)
    
    # Assertions
    assert result["question_type"] == "EXACT_RULE", f"Expected EXACT_RULE, got {result['question_type']}"
    assert len(result["sources"]) > 0, "Phải có sources"
    
    # LLM KHÔNG được gọi
    mock_llm_client.generate.assert_not_called()
    
    # Answer phải chứa số liệu cụ thể
    answer = result["answer"]
    assert "14" in answer or "24" in answer or "30" in answer, \
        f"Answer phải chứa số liệu cụ thể. Got: {answer}"
    assert "Điều 14" in answer, f"Answer phải tham chiếu Điều 14. Got: {answer}"
    
    # Answer KHÔNG được chứa trailing sentence
    assert "Bạn nên tham khảo" not in answer, \
        f"Answer không được chứa trailing sentence. Got: {answer}"
    
    # Answer phải quote full rule text
    assert "14 ≤ n ≤ 24" in answer or "tối đa 30 tín chỉ" in answer, \
        f"Answer phải quote rule text. Got: {answer}"
    
    print(f"✅ Q1 Test passed!")
    print(f"Answer: {answer[:100]}...")
    return result


async def test_q2_near_rule():
    """
    Segment 3 – Everyday language → NEAR_RULE (Must-have #2)
    Q2: "Em rớt 3 môn thì có bị sao không ạ?"
    """
    mock_llm_client = MagicMock()
    
    async def mock_classify(question, client):
        # Should classify as NEAR_RULE (informal but about regulations)
        return QuestionType.NEAR_RULE
    
    async def mock_generate(system_prompt, user_prompt, context=""):
        # LLM should answer based on context
        return "Theo quy định, sinh viên có thể bị cảnh báo học vụ nếu kết quả học tập không đạt..."
    
    mock_llm_client.generate = AsyncMock(side_effect=mock_generate)
    mock_llm_client.generate_json = AsyncMock(return_value={"label": "NEAR_RULE", "reason": "..."})
    
    mock_vector_store = MagicMock()
    mock_vector_store.search = MagicMock(return_value=[])  # May or may not have sources
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    with patch("backend.llm.orchestrator.classify_question", side_effect=mock_classify):
        result = await pipeline.answer_question(Q2)
    
    assert result["question_type"] == "NEAR_RULE", \
        f"Expected NEAR_RULE, got {result['question_type']}"
    
    # LLM should be called for NEAR_RULE
    assert mock_llm_client.generate.called, "LLM should be called for NEAR_RULE"
    
    print(f"✅ Q2 Test passed!")
    print(f"Question Type: {result['question_type']}")
    return result


async def test_q3_out_of_scope():
    """
    Segment 4 – OUT_OF_SCOPE advice (Must-have #3)
    Q3: "Theo thầy em nên học lại hay rút môn thì tốt hơn?"
    """
    mock_llm_client = MagicMock()
    
    async def mock_classify(question, client):
        return QuestionType.OUT_OF_SCOPE
    
    async def mock_generate(system_prompt, user_prompt, context=""):
        return "Câu hỏi này cần lời khuyên cá nhân. Bạn nên liên hệ cố vấn học tập hoặc Phòng Đào tạo..."
    
    mock_llm_client.generate = AsyncMock(side_effect=mock_generate)
    mock_llm_client.generate_json = AsyncMock(return_value={"label": "OUT_OF_SCOPE", "reason": "..."})
    
    mock_vector_store = MagicMock()
    mock_vector_store.search = MagicMock(return_value=[])
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    with patch("backend.llm.orchestrator.classify_question", side_effect=mock_classify):
        result = await pipeline.answer_question(Q3)
    
    assert result["question_type"] == "OUT_OF_SCOPE", \
        f"Expected OUT_OF_SCOPE, got {result['question_type']}"
    assert len(result["sources"]) == 0, "Sources should be empty for OUT_OF_SCOPE"
    
    answer = result["answer"]
    assert "cố vấn" in answer.lower() or "phòng đào tạo" in answer.lower() or "ctsv" in answer.lower(), \
        f"Answer should suggest contacting advisor. Got: {answer}"
    
    print(f"✅ Q3 Test passed!")
    print(f"Answer: {answer[:100]}...")
    return result


async def test_q4_q5_multiturn():
    """
    Segment 5 – Multi-turn conversation (Nice-to-have #1)
    Q4: "Điều kiện để bị cảnh báo học vụ là gì?"
    Q5: "Vậy nếu em bị cảnh báo thì có ảnh hưởng gì không?"
    """
    mock_llm_client = MagicMock()
    
    async def mock_classify_q4(question, client):
        return QuestionType.EXACT_RULE
    
    async def mock_classify_q5(question, client):
        # Q5 should be NEAR_RULE (informal but about regulations)
        return QuestionType.NEAR_RULE
    
    async def mock_generate(system_prompt, user_prompt, context=""):
        if "cảnh báo" in user_prompt.lower() and "ảnh hưởng" in user_prompt.lower():
            # Q5: should reference previous context about warning
            return "Nếu bị cảnh báo học vụ, sinh viên sẽ bị hạn chế trong một học kỳ chính tiếp theo..."
        return "Default answer"
    
    mock_llm_client.generate = AsyncMock(side_effect=mock_generate)
    mock_llm_client.generate_json = AsyncMock(return_value={"label": "NEAR_RULE", "reason": "..."})
    
    mock_vector_store = MagicMock()
    mock_chunks_q4 = [
        {
            "article_id": "Điều 16",
            "clause_id": "Khoản 1",
            "text": (
                "Sinh viên bị cảnh báo học vụ nếu vi phạm một trong những trường hợp sau đây: "
                "Không hoàn thành nghĩa vụ học phí đúng quy định. "
                "Có ĐTBHK dưới 3,0 hoặc điểm trung bình của liên tiếp 2 học kỳ gần nhất đều dưới 4,0."
            ),
            "metadata": {"title": "Điều 16"},
            "score": 0.95
        }
    ]
    mock_vector_store.search = MagicMock(side_effect=[mock_chunks_q4, []])  # Q4 has chunks, Q5 may not
    mock_ontology_graph = MagicMock()
    
    pipeline = ChatPipeline(
        llm_client=mock_llm_client,
        vector_store=mock_vector_store,
        ontology_graph=mock_ontology_graph,
    )
    
    # Q4
    with patch("backend.llm.orchestrator.classify_question", side_effect=mock_classify_q4):
        result_q4 = await pipeline.answer_question(Q4)
    
    assert result_q4["question_type"] == "EXACT_RULE"
    assert "3,0" in result_q4["answer"] or "4,0" in result_q4["answer"] or "cảnh báo" in result_q4["answer"].lower()
    
    # Q5 with conversation history
    conversation_history = [
        {"role": "user", "content": Q4},
        {"role": "bot", "content": result_q4["answer"]}
    ]
    
    with patch("backend.llm.orchestrator.classify_question", side_effect=mock_classify_q5):
        result_q5 = await pipeline.answer_question(Q5, conversation_history=conversation_history)
    
    assert result_q5["question_type"] == "NEAR_RULE" or result_q5["question_type"] == "EXACT_RULE"
    # Answer should reference "cảnh báo" from previous context
    assert "cảnh báo" in result_q5["answer"].lower() or "học vụ" in result_q5["answer"].lower(), \
        f"Q5 answer should reference previous context. Got: {result_q5['answer']}"
    
    print(f"✅ Q4-Q5 Multi-turn Test passed!")
    print(f"Q4 Answer: {result_q4['answer'][:80]}...")
    print(f"Q5 Answer: {result_q5['answer'][:80]}...")
    return result_q4, result_q5


async def main():
    """Run all demo script tests."""
    print("=" * 60)
    print("DEMO SCRIPT TESTS")
    print("=" * 60)
    
    print("\n1. Testing Q1 (EXACT_RULE)...")
    await test_q1_exact_rule()
    
    print("\n2. Testing Q2 (NEAR_RULE)...")
    await test_q2_near_rule()
    
    print("\n3. Testing Q3 (OUT_OF_SCOPE)...")
    await test_q3_out_of_scope()
    
    print("\n4. Testing Q4-Q5 (Multi-turn)...")
    await test_q4_q5_multiturn()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

