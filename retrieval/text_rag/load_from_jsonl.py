"""Loader for UIT regulation content exported as JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TypedDict


class RawRegulationDoc(TypedDict):
    article_id: str
    clause_id: str | None
    title: str | None
    section: str | None
    text: str


def _find_article_id(item_id: str, index: dict[str, dict]) -> str | None:
    current = index.get(item_id)
    while current:
        level = str(current.get("level", "")).lower()
        if level == "dieu":
            return str(current["_id"])
        parent_id = current.get("parent_id")
        if parent_id is None:
            return None
        current = index.get(str(parent_id))
    return None


def iter_raw_docs(path: str | Path) -> Iterator[RawRegulationDoc]:
    """
    Yield article/clause documents from the UIT items export.

    Alignment: we keep `_id` strings from `KB_UIT.items.json` as article_id /
    clause_id so they can be reused to construct ontology URIs
    (UIT:Article_<article_id>, UIT:Clause_<clause_id>).
    """
    data_path = Path(path)
    items = json.loads(data_path.read_text(encoding="utf-8"))
    index = {str(item["_id"]): item for item in items}

    for item in items:
        level = str(item.get("level", "")).lower()
        if level not in {"dieu", "khoan"}:
            continue

        text = item.get("content") or ""
        if not text.strip():
            continue

        if level == "dieu":
            article_id = str(item["_id"])
            clause_id = None
        else:
            clause_id = str(item["_id"])
            article_id = _find_article_id(clause_id, index) or ""

        yield {
            "article_id": article_id,
            "clause_id": clause_id,
            "title": item.get("title"),
            "section": item.get("heading"),
            "text": text,
        }

