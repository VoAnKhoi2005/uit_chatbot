from __future__ import annotations

import os
from typing import Any, Dict

from openai import AsyncOpenAI
from pydantic import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    class Config:
        env_prefix = ""
        env_file = ".env"


class LLMClient:
    """Minimal async wrapper over OpenAI-compatible chat models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required")
        self.model = os.getenv("OPENAI_MODEL", self.settings.openai_model)
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{user_prompt}\n\nContext:\n{context}".strip()},
        ]
        resp = await self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""

    async def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raw = await self.generate(system_prompt, user_prompt)
        try:
            import json

            return json.loads(raw)
        except Exception:
            return {"label": None, "reason": raw}

