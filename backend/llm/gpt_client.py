from __future__ import annotations

import asyncio
import logging
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
        self.logger = logging.getLogger("uitchatbot.gpt_client")

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
        
        self.logger.debug("[GPT] Calling OpenAI API with model=%s, messages=%d", self.model, len(messages))
        self.logger.debug("[GPT] User prompt length: %d chars, context length: %d chars", 
                         len(user_prompt), len(context))
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._sync_generate, messages
        )
        
        self.logger.debug("[GPT] Response received, length=%d chars", len(result) if result else 0)
        return result
    
    def _sync_generate(self, messages: list) -> str:
        """Synchronous call to OpenAI API."""
        self.logger.info("[GPT] Making sync API call to model=%s", self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # temperature=0.2,
        )
        content = response.choices[0].message.content
        self.logger.info("[GPT] API call completed, tokens used: prompt=%d, completion=%d, total=%d",
                        response.usage.prompt_tokens if response.usage else 0,
                        response.usage.completion_tokens if response.usage else 0,
                        response.usage.total_tokens if response.usage else 0)
        return content

    async def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generate JSON response using OpenAI GPT API."""
        self.logger.debug("[GPT] Generating JSON response")
        raw = await self.generate(system_prompt, user_prompt)
        try:
            import json
            result = json.loads(raw)
            self.logger.debug("[GPT] Successfully parsed JSON response")
            return result
        except Exception as e:
            self.logger.warning("[GPT] Failed to parse JSON response: %s", e)
            return {"label": None, "reason": raw}
