"""Configuration for RAG retrieval system."""

from __future__ import annotations

import os

# Flag to disable heavy local embedding (for lightweight deployment on Render free tier)
# When True: use text-only search, do NOT import sentence-transformers or torch
# When False: use full vector search with SentenceTransformer
UIT_DISABLE_LOCAL_EMBEDDER = os.getenv("UIT_DISABLE_LOCAL_EMBEDDER", "false").lower() in {
    "1",
    "true",
    "yes",
}

