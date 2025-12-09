import json
from pathlib import Path

from ontology.from_jsonl import build_graph
from ontology.loader import get_article_by_id, get_clauses_for_article


def test_build_graph_from_sample(tmp_path: Path) -> None:
    triplets = [
        {
            "_id": {"$oid": "1"},
            "subject_id": {"$oid": "s1"},
            "relation_id": {"$oid": "r1"},
            "object_id": {"$oid": "o1"},
            "subject_name": "quyết định",
            "relation_name": "có",
            "object_name": "hiệu lực",
            "document_id": "DOC-1",
            "document_number": None,
        }
    ]
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
            "content": "Điều 1 nội dung.",
            "ordinal": 1001,
            "path": "1/1",
        },
        {
            "_id": "C-1",
            "doc_id": "DOC-1",
            "parent_id": "A-1",
            "level": "khoan",
            "title": "Khoản 1",
            "heading": "",
            "content": "Khoản 1 nội dung.",
            "ordinal": 1002,
            "path": "1/1/1",
        },
    ]

    triplets_path = tmp_path / "triplets.json"
    items_path = tmp_path / "items.json"
    ttl_path = tmp_path / "out.ttl"
    triplets_path.write_text(json.dumps(triplets, ensure_ascii=False), encoding="utf-8")
    items_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    graph = build_graph(triplets_path=triplets_path, items_path=items_path, out_path=ttl_path)
    assert len(graph) > 0
    assert ttl_path.exists()

    article_rows = get_article_by_id(graph, "A-1")
    assert article_rows

    clause_rows = get_clauses_for_article(graph, "A-1")
    assert clause_rows

