from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from openai import OpenAI
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_max_tokens: int = 2048
    llm_disable_reasoning: bool = True

    class Config:
        env_prefix = ""
        env_file = ".env"
        extra = "ignore"


class LLMClient:
    """Generic OpenAI-protocol chat client.

    Works against any endpoint that speaks the OpenAI chat-completions API -
    OpenAI itself, Groq, OpenRouter, a local vLLM/Ollama server, etc. - by
    pointing `OpenAI(base_url=...)` at whatever URL is configured. Nothing
    here is provider-specific: there is exactly one client for the whole
    system, configured entirely by LLM_BASE_URL / LLM_API_KEY / LLM_MODEL.

    `OPENAI_API_KEY`/`OPENAI_MODEL` are read as fallbacks only so an existing
    .env doesn't need every key renamed at once; the LLM_* names win when set.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.logger = logging.getLogger("uit_chatbot.llm_client")

        base_url = self.settings.llm_base_url or os.getenv("LLM_BASE_URL")
        if not base_url:
            raise EnvironmentError(
                "LLM_BASE_URL is required - set it to the OpenAI-protocol endpoint "
                "to call (e.g. https://api.openai.com/v1, https://api.groq.com/openai/v1, "
                "or a local server's /v1 URL)."
            )

        api_key = (
            self.settings.llm_api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise EnvironmentError("LLM_API_KEY (or OPENAI_API_KEY) is required")

        model = self.settings.llm_model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")
        if not model:
            raise EnvironmentError("LLM_MODEL (or OPENAI_MODEL) is required")

        self.base_url = base_url
        self.model = model
        self.max_tokens = self.settings.llm_max_tokens
        self.disable_reasoning = self.settings.llm_disable_reasoning
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    async def generate(self, system_prompt: str, user_prompt: str, context: str = "") -> str:
        full_user_prompt = user_prompt
        if context:
            full_user_prompt = f"{user_prompt}\n\nContext:\n{context}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": full_user_prompt})

        self.logger.debug("[LLM] Calling %s (model=%s), messages=%d", self.base_url, self.model, len(messages))

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_generate, messages)

    def _sync_generate(self, messages: list) -> str:
        # Without max_tokens, some OpenRouter models (reasoning/"flash"
        # variants that emit hidden reasoning tokens counted against the
        # output budget) fall back to a provider default too small to finish
        # - the visible answer gets cut off mid-sentence with no error, which
        # silently produced incomplete answers (and tanked eval scores that
        # compare against a complete golden answer). Disabling reasoning
        # removes that hidden consumption entirely for a customer-facing
        # answer that doesn't need visible chain-of-thought; LLM_DISABLE_REASONING=false
        # restores it, LLM_MAX_TOKENS overrides the token budget (default 2048).
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if self.disable_reasoning:
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        usage = response.usage
        finish_reason = response.choices[0].finish_reason
        self.logger.debug(
            "[LLM] Response received (prompt=%s, completion=%s, total=%s tokens, finish_reason=%s)",
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
            getattr(usage, "total_tokens", None),
            finish_reason,
        )
        if finish_reason == "length":
            self.logger.warning(
                "[LLM] Response truncated by max_tokens=%s - answer may be incomplete", self.max_tokens
            )
        return content

    async def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raw = await self.generate(system_prompt, user_prompt)
        try:
            import json

            return json.loads(raw)
        except Exception:
            return {"label": None, "reason": raw}
