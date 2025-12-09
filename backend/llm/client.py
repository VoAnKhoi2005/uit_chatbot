from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

# Note: With Pydantic v2, BaseSettings lives in the separate pydantic-settings package.
from pydantic_settings import BaseSettings

from groq_client import call_groq_llm


class Settings(BaseSettings):
    groq_api_key: str | None = None
    groq_model: str = "llama3-8b-8192"

    class Config:
        env_prefix = ""
        env_file = ".env"
        extra = "ignore"


class LLMClient:
    """Async wrapper over Groq via shared groq_client helper."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        api_key = self.settings.groq_api_key or os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is required")
        # groq_client reads env directly; we still keep model for clarity
        self.model = os.getenv("GROQ_MODEL", self.settings.groq_model)

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        # Build user prompt with context if provided
        full_user_prompt = user_prompt
        if context:
            full_user_prompt = f"{user_prompt}\n\nContext:\n{context}"
        
        # Pass system and user prompts separately to Groq API
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, call_groq_llm, system_prompt, full_user_prompt
        )

    async def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raw = await self.generate(system_prompt, user_prompt)
        try:
            import json

            return json.loads(raw)
        except Exception:
            return {"label": None, "reason": raw}

