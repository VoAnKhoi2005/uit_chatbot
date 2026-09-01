import pytest

from backend.llm.scope_gate import check_in_scope


class DummyLLM:
    def __init__(self, label: str):
        self.label = label

    async def generate_json(self, system_prompt: str, user_prompt: str):
        return {"label": self.label, "reason": "test"}


@pytest.mark.asyncio
async def test_check_in_scope_true_for_in_scope_label():
    q = "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ?"
    result = await check_in_scope(q, DummyLLM("IN_SCOPE"))
    assert result is True


@pytest.mark.asyncio
async def test_check_in_scope_false_for_out_of_scope_label():
    q = "Theo thầy em nên học lại hay rút môn thì tốt hơn?"
    result = await check_in_scope(q, DummyLLM("OUT_OF_SCOPE"))
    assert result is False


@pytest.mark.asyncio
async def test_check_in_scope_defaults_true_on_unexpected_label():
    """Defaults to in-scope (try to answer) rather than silently refusing,
    if the LLM's JSON output is missing/unusable."""
    q = "Bất kỳ câu hỏi nào"
    result = await check_in_scope(q, DummyLLM(""))
    assert result is True
