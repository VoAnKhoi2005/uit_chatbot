from __future__ import annotations

import json

from .client import LLMClient
from .prompts import QUESTION_CLASSIFIER_SYSTEM_PROMPT
from .question_types import QuestionType


async def classify_question(question: str, llm_client: LLMClient) -> QuestionType:
    user_prompt = f'Hãy phân loại câu hỏi sau và trả về JSON: "{question}"'
    result = await llm_client.generate_json(QUESTION_CLASSIFIER_SYSTEM_PROMPT, user_prompt)
    label = str(result.get("label") or "").strip().upper()
    try:
        return QuestionType(label)
    except Exception:
        # fallback to NEAR_RULE if parsing fails or label unknown
        return QuestionType.NEAR_RULE

