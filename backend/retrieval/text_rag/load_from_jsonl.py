"""Loader for UIT regulation content exported as JSON."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator, TypedDict


class RawRegulationDoc(TypedDict):
    article_id: str
    clause_id: str | None
    title: str | None
    section: str | None
    text: str
    doc_id: str
    doc_title: str | None
    so_hieu: str


def _dedup_fingerprint(text: str) -> str:
    """Accent-insensitive, whitespace-*removed* (not just collapsed) key.

    The export contains literal duplicate articles under the same doc_id,
    where one copy has a text-extraction defect: a stray space injected
    before certain accented Vietnamese letters (e.g. "ch ỉ" instead of
    "chỉ"). Collapsing whitespace to single spaces would still leave the
    defective copy with extra space-separated fragments the clean copy
    doesn't have, so this strips whitespace out entirely before comparing -
    the defect then disappears from both, and the two collapse to the same
    fingerprint.
    """
    normalized = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", "", normalized)


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


def _select_duplicates_to_drop(items: list[dict]) -> set[str]:
    """Item `_id`s to skip: near-duplicate content (see `_dedup_fingerprint`)
    within the same `doc_id`, keeping the shortest text in each group - the
    defect strictly adds characters (extra spaces) for the same content, so
    the shortest copy is the clean one."""
    groups: dict[tuple[object, str], list[dict]] = {}
    for item in items:
        level = str(item.get("level", "")).lower()
        text = item.get("content") or ""
        if level not in {"dieu", "khoan"} or not text.strip():
            continue
        key = (item.get("doc_id"), _dedup_fingerprint(text))
        groups.setdefault(key, []).append(item)

    drop: set[str] = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        keep = min(group, key=lambda it: len(it.get("content") or ""))
        for item in group:
            if item is not keep:
                drop.add(str(item["_id"]))
    return drop


def iter_raw_docs(path: str | Path) -> Iterator[RawRegulationDoc]:
    """
    Yield article/clause documents from the UIT items export.

    Alignment: we keep `_id` strings from `KB_UIT.items.json` as article_id /
    clause_id so they can be reused to construct ontology URIs
    (UIT:Article_<article_id>, UIT:Clause_<clause_id>).

    Skips near-duplicate articles/clauses within the same document (see
    `_select_duplicates_to_drop`) - the export has been observed to contain
    the same article twice, once with a text-extraction defect, which
    otherwise wastes retrieval's top-k on redundant near-identical chunks.
    """
    data_path = Path(path)
    items = json.loads(data_path.read_text(encoding="utf-8"))
    index = {str(item["_id"]): item for item in items}
    drop = _select_duplicates_to_drop(items)

    for item in items:
        if str(item["_id"]) in drop:
            continue

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
            "doc_id": item.get("doc_id"),
            "doc_title": item.get("doc_title"),
            "so_hieu": item.get("so_hieu"),
        }

