"""SentenceTransformers wrapper for regulation text chunks."""

from __future__ import annotations

import os
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    def __init__(self, model_name: str | None = None) -> None:
        chosen = model_name or os.getenv("UIT_RAG_MODEL", "keepitreal/vietnamese-sbert")
        self.model = SentenceTransformer(chosen)

    def embed(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype("float32")

