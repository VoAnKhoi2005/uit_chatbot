"""OpenAI-protocol embeddings client for regulation text chunks and graph nodes/triples.

Calls a server-side embedding model (default: Qwen3 Embedding 8B, e.g. via
OpenRouter) instead of loading a model locally - no torch/sentence-transformers
needed for retrieval. Shares the same OpenAI-protocol config as the chat
LLMClient (backend/llm/client.py): EMBEDDING_BASE_URL/EMBEDDING_API_KEY fall
back to LLM_BASE_URL/LLM_API_KEY, since a provider like OpenRouter serves both
chat and embeddings from the same account/endpoint.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
from openai import OpenAI

# OpenRouter's embeddings endpoint rejects a request whose `input` list has
# more than 1024 items (HTTP 422 "array_above_max_length"), regardless of how
# short each string is. GraphRetriever's batch (node names + triple texts)
# can exceed that in one shot, so embed() splits into chunks of this size.
MAX_ITEMS_PER_REQUEST = 1024


class TextEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL")
        if not base_url:
            raise EnvironmentError("EMBEDDING_BASE_URL (or LLM_BASE_URL) is required")

        api_key = (
            os.getenv("EMBEDDING_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise EnvironmentError("EMBEDDING_API_KEY (or LLM_API_KEY/OPENAI_API_KEY) is required")

        self.model = model_name or os.getenv("EMBEDDING_MODEL") or os.getenv("UIT_RAG_MODEL", "qwen/qwen3-embedding-8b")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype="float32")
        vectors = np.concatenate(
            [self._embed_batch(texts[i : i + MAX_ITEMS_PER_REQUEST]) for i in range(0, len(texts), MAX_ITEMS_PER_REQUEST)],
            axis=0,
        )
        # Normalize to unit vectors, matching the previous local model's
        # normalize_embeddings=True - callers (FAISS cosine_index, BM25+dense
        # fusion, ...) assume unit-norm embeddings.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        return (vectors / norms).astype("float32")

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in response.data], dtype="float32")
