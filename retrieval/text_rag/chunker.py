"""Simple text chunker for UIT regulation documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, TypedDict

from .load_from_jsonl import RawRegulationDoc, iter_raw_docs


class TextChunk(TypedDict):
    chunk_id: str
    article_id: str
    clause_id: str | None
    text: str
    metadata: dict


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_document(doc: RawRegulationDoc, max_chars: int = 800) -> List[TextChunk]:
    text = doc["text"].strip()
    if len(text) <= max_chars:
        return [
            {
                "chunk_id": f"{doc['article_id']}_{doc['clause_id'] or 'none'}_0",
                "article_id": doc["article_id"],
                "clause_id": doc["clause_id"],
                "text": text,
                "metadata": {
                    "title": doc.get("title"),
                    "section": doc.get("section"),
                    "doc_id": doc.get("doc_id"),
                    "doc_title": doc.get("doc_title"),
                    "so_hieu": doc.get("so_hieu"),
                },
            }
        ]

    sentences = _split_sentences(text) or [text]
    chunks: list[TextChunk] = []
    buffer = ""
    idx = 0
    for sent in sentences:
        if not buffer:
            buffer = sent
            continue
        candidate = f"{buffer} {sent}".strip()
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        chunks.append(
            {
                "chunk_id": f"{doc['article_id']}_{doc['clause_id'] or 'none'}_{idx}",
                "article_id": doc["article_id"],
                "clause_id": doc["clause_id"],
                "text": buffer,
                "metadata": {
                    "title": doc.get("title"),
                    "section": doc.get("section"),
                    "doc_id": doc.get("doc_id"),
                    "doc_title": doc.get("doc_title"),
                    "so_hieu": doc.get("so_hieu"),
                },
            }
        )
        idx += 1
        buffer = sent

    if buffer:
        chunks.append(
            {
                "chunk_id": f"{doc['article_id']}_{doc['clause_id'] or 'none'}_{idx}",
                "article_id": doc["article_id"],
                "clause_id": doc["clause_id"],
                "text": buffer,
                "metadata": {
                    "title": doc.get("title"),
                    "section": doc.get("section"),
                    "doc_id": doc.get("doc_id"),
                    "doc_title": doc.get("doc_title"),
                    "so_hieu": doc.get("so_hieu"),
                },
            }
        )
    return chunks


def iter_all_chunks(jsonl_path: str | Path, max_chars: int = 800) -> Iterator[TextChunk]:
    for doc in iter_raw_docs(jsonl_path):
        for chunk in chunk_document(doc, max_chars=max_chars):
            yield chunk

