from __future__ import annotations

from .client import LLMClient
from .prompts import SCOPE_GATE_SYSTEM_PROMPT


async def check_in_scope(question: str, llm_client: LLMClient) -> bool:
    """Simple binary LLM gate: is this question in the UIT study/training
    regulation domain, or out of scope?

    Replaces the old 3-way EXACT_RULE/NEAR_RULE/OUT_OF_SCOPE classifier -
    routing between EXACT_RULE and NEAR_RULE is decided later, from
    retrieval evidence quality, not by an upfront LLM label. This is only
    the domain/out-of-scope decision. Defaults to in-scope if the LLM call
    or its JSON output is unusable, so a transient LLM failure degrades to
    "try to answer" rather than silently refusing every question.
    """
    result = await llm_client.generate_json(SCOPE_GATE_SYSTEM_PROMPT, question)
    label = str(result.get("label") or "").strip().upper()
    if label == "OUT_OF_SCOPE":
        return False
    return True
