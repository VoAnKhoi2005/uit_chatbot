from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from pydantic_settings import BaseSettings
from openai import OpenAI


class GPTSettings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    class Config:
        env_prefix = ""
        env_file = ".env"
        extra = "ignore"


class GPTLLMClient:
    """Async wrapper for OpenAI GPT API."""

    def __init__(self, settings: GPTSettings | None = None) -> None:
        self.settings = settings or GPTSettings()
        api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is required")
        
        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", self.settings.openai_model)

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        """
        Generate response using OpenAI GPT API.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            context: Additional context to append to user prompt
        
        Returns:
            Generated response text
        """
        full_user_prompt = user_prompt
        if context:
            full_user_prompt = f"{user_prompt}\n\nContext:\n{context}"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": full_user_prompt})
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._sync_generate, messages
        )
    
    def _sync_generate(self, messages: list) -> str:
        """Synchronous call to OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content

    async def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generate JSON response using OpenAI GPT API."""
        raw = await self.generate(system_prompt, user_prompt)
        try:
            import json
            return json.loads(raw)
        except Exception:
            return {"label": None, "reason": raw}
