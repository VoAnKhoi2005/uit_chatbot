"""
Converter from UIT JSON exports to RDF/OWL (rdflib Graph).

Data sources:
- Graph/triplets: `graph/mongo_export_uit/KB_UIT.triplets.json`
  Fields: subject_id, relation_id, object_id, subject_name, relation_name,
  object_name, document_id, document_number.
- Content/hierarchy: `graph/mongo_export_uit/KB_UIT.items.json`
  Fields: _id, doc_id, parent_id, level, title, heading, content, ordinal, path.

Run as a script:
    python -m ontology.from_jsonl
Environment:
- Paths default to the repo locations above; override via env vars
  `UIT_TRIPLETS_PATH`, `UIT_ITEMS_PATH`, `UIT_TTL_PATH`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

from rdflib import Graph, Literal, RDF, RDFS, URIRef

from . import schema as SC

logger = logging.getLogger(__name__)

DEFAULT_TRIPLETS_PATH = Path("backend/graph/mongo_export_uit/KB_UIT.triplets.json")
DEFAULT_ITEMS_PATH = Path("backend/graph/mongo_export_uit/KB_UIT.items.json")
DEFAULT_TTL_PATH = Path("backend/ontology/uit_regulations.ttl")


def _extract_id(raw: Any) -> Optional[str]:
    """Extract string id from Mongo-ish values (dict with $oid or plain string)."""
    if raw is None:
        return None
    if isinstance(raw, dict) and "$oid" in raw:
        return str(raw["$oid"])
    return str(raw)


def _sanitize_local_name(value: str) -> str:
    """Create a URI-friendly local name."""
    return "".join(ch if ch.isalnum() else "_" for ch in value)


def _iter_json_objects(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Streaming iterator over JSON array or JSONL file.

    Avoids loading the whole file into memory by decoding incrementally.
    """
    decoder = json.JSONDecoder()
    buffer = ""
    with path.open("r", encoding="utf-8") as handle:
        for chunk in iter(lambda: handle.read(65536), ""):
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] in "[,":
                    buffer = buffer[1:]
                    continue
                if buffer[0] == "]":
                    buffer = buffer[1:]
                    return
                try:
                    obj, idx = decoder.raw_decode(buffer)
                except ValueError:
                    break
                yield obj
                buffer = buffer[idx:]


def _relation_predicate(relation_name: str) -> URIRef:
    key = relation_name.strip().lower()
    for candidate, uri in SC.RELATION_PROPERTY_MAP.items():
        if key == candidate:
            return uri
    return SC.relatedTo


def _ensure_doc_node(graph: Graph, doc_id: str) -> URIRef:
    doc_uri = SC.UIT[f"Document_{_sanitize_local_name(doc_id)}"]
    graph.add((doc_uri, RDF.type, SC.Document))
    graph.add((doc_uri, SC.docId, Literal(doc_id)))
    return doc_uri


def _add_triplets(graph: Graph, path: Path) -> None:
    logger.info("Loading triplets from %s", path)
    for triple in _iter_json_objects(path):
        subject_id = _extract_id(triple.get("subject_id"))
        object_id = _extract_id(triple.get("object_id"))
        relation_name = str(triple.get("relation_name", "") or "")
        subject_name = triple.get("subject_name")
        object_name = triple.get("object_name")
        document_id = triple.get("document_id")

        if not subject_id or not object_id:
            continue

        sub_uri = SC.UIT[f"Entity_{_sanitize_local_name(subject_id)}"]
        obj_uri = SC.UIT[f"Entity_{_sanitize_local_name(object_id)}"]
        graph.add((sub_uri, RDF.type, SC.Entity))
        graph.add((obj_uri, RDF.type, SC.Entity))

        if subject_name:
            graph.add((sub_uri, RDFS.label, Literal(subject_name)))
            graph.add((sub_uri, SC.subjectName, Literal(subject_name)))
        if object_name:
            graph.add((obj_uri, RDFS.label, Literal(object_name)))
            graph.add((obj_uri, SC.objectName, Literal(object_name)))

        predicate_uri = _relation_predicate(relation_name)
        graph.add((sub_uri, predicate_uri, obj_uri))
        if relation_name:
            graph.add((sub_uri, SC.relationName, Literal(relation_name)))

        doc_id_str = document_id if isinstance(document_id, str) else _extract_id(document_id)
        if doc_id_str:
            doc_uri = _ensure_doc_node(graph, doc_id_str)
            graph.add((sub_uri, SC.inDocument, doc_uri))
            graph.add((obj_uri, SC.inDocument, doc_uri))


def _item_class(level: str) -> URIRef:
    level_lower = level.lower()
    if level_lower == "dieu":
        return SC.Article
    if level_lower == "khoan":
        return SC.Clause
    if level_lower == "chuong":
        return SC.Chapter
    if level_lower == "muc":
        return SC.Section
    return SC.Entity


# python
def _add_items(graph: Graph, path: Path) -> None:
    logger.info("Loading items from %s", path)
    with path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)

    uri_by_id: dict[str, URIRef] = {}
    level_by_id: dict[str, str] = {}
    for item in items:
        item_id = str(item["_id"])
        level = item.get("level", "")
        uri = SC.UIT[f"{level}_{_sanitize_local_name(item_id)}"]
        uri_by_id[item_id] = uri
        level_by_id[item_id] = level.lower()
        graph.add((uri, RDF.type, _item_class(level)))

        doc_id = item.get("doc_id")
        doc_title = item.get("doc_title")
        so_hieu = item.get("so_hieu")
        if doc_id:
            doc_uri = _ensure_doc_node(graph, doc_id)
            graph.add((uri, SC.inDocument, doc_uri))
            graph.add((uri, SC.docId, Literal(doc_id)))
            if doc_title:
                graph.add((uri, SC.docTitle, Literal(doc_title)))
            if so_hieu:
                graph.add((uri, SC.soHieu, Literal(so_hieu)))

        if level.lower() == "dieu":
            graph.add((uri, SC.articleId, Literal(item_id)))
        if level.lower() == "khoan":
            graph.add((uri, SC.clauseId, Literal(item_id)))

        if title := item.get("title"):
            graph.add((uri, SC.title, Literal(title)))
            graph.add((uri, RDFS.label, Literal(title)))
        if heading := item.get("heading"):
            graph.add((uri, SC.heading, Literal(heading)))
        if content := item.get("content"):
            graph.add((uri, SC.fullText, Literal(content)))
        if level:
            graph.add((uri, SC.level, Literal(level)))
        if path_str := item.get("path"):
            graph.add((uri, SC.path, Literal(path_str)))
        if (ord_val := item.get("ordinal")) is not None:
            graph.add((uri, SC.ordinal, Literal(ord_val)))

    for item in items:
        item_id = str(item["_id"])
        parent_id = item.get("parent_id")
        if not parent_id:
            continue
        child_uri = uri_by_id.get(item_id)
        parent_uri = uri_by_id.get(str(parent_id))
        if not child_uri or not parent_uri:
            continue
        graph.add((child_uri, SC.hasParent, parent_uri))

        parent_level = level_by_id.get(str(parent_id), "")
        if parent_level == "dieu":
            graph.add((parent_uri, SC.hasClause, child_uri))
        if parent_level in {"chuong", "muc"}:
            graph.add((parent_uri, SC.hasArticle, child_uri))


def build_graph(
    triplets_path: Path = DEFAULT_TRIPLETS_PATH,
    items_path: Optional[Path] = DEFAULT_ITEMS_PATH,
    out_path: Path = DEFAULT_TTL_PATH,
) -> Graph:
    """
    Build the RDF graph from the exported JSON and optionally serialize it.

    The returned graph is also serialized to `out_path` if provided.
    """
    graph = Graph()
    graph.bind("uit", SC.UIT)
    graph.bind("rdfs", RDFS)

    _add_triplets(graph, triplets_path)
    if items_path and items_path.exists():
        _add_items(graph, items_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(destination=str(out_path), format="turtle")
    logger.info("Ontology written to %s with %s triples", out_path, len(graph))
    return graph


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    triplets_path = Path(os.getenv("UIT_TRIPLETS_PATH", DEFAULT_TRIPLETS_PATH))
    items_path_env = os.getenv("UIT_ITEMS_PATH")
    items_path = Path(items_path_env) if items_path_env else DEFAULT_ITEMS_PATH
    ttl_path = Path(os.getenv("UIT_TTL_PATH", DEFAULT_TTL_PATH))
    build_graph(triplets_path=triplets_path, items_path=items_path, out_path=ttl_path)


if __name__ == "__main__":
    main()

