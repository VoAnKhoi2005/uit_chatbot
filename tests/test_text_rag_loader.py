import json
from pathlib import Path

from backend.retrieval.text_rag.chunker import chunk_document
from backend.retrieval.text_rag.load_from_jsonl import iter_raw_docs


def test_iter_raw_docs_and_chunking(tmp_path: Path) -> None:
    items = [
        {
            "_id": "DOC-1",
            "doc_id": "DOC-1",
            "parent_id": None,
            "level": "chuong",
            "title": "Chương 1",
            "heading": "",
            "content": "",
            "ordinal": 1,
            "path": "1",
        },
        {
            "_id": "A-1",
            "doc_id": "DOC-1",
            "parent_id": "DOC-1",
            "level": "dieu",
            "title": "Điều 1",
            "heading": "Điều khoản",
            "content": "Nội dung điều 1. Đây là câu thứ hai để kiểm tra tách câu.",
            "ordinal": 1001,
            "path": "1/1",
        },
    ]
    path = tmp_path / "items.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    docs = list(iter_raw_docs(path))
    assert len(docs) == 1
    doc = docs[0]
    assert doc["article_id"] == "A-1"
    chunks = chunk_document(doc, max_chars=40)
    assert len(chunks) >= 1
    assert chunks[0]["article_id"] == "A-1"

