import pytest

from backend.llm.question_classifier import classify_question
from backend.llm.question_types import QuestionType


class DummyLLM:
    async def generate_json(self, system_prompt: str, user_prompt: str):
        return {"label": "EXACT_RULE", "reason": "test"}


@pytest.mark.asyncio
async def test_classify_question_returns_enum():
    q = "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ?"
    result = await classify_question(q, DummyLLM())
    assert result == QuestionType.EXACT_RULE

