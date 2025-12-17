from typing import List, Dict, Any

from retrieval.src.registry.metadata_registry import MetadataRegistry


def build_citations(evidence_used: List[dict], registry: MetadataRegistry) -> List[dict]:
    """
    Build user-facing citations from evidence and registry.
    Deduplicate by (doc_id, article_id, clause_id).
    """
    seen = set()
    citations = []
    for ev in evidence_used:
        ctype = ev.get("source") or ev.get("type")
        doc_id = None
        doc_title = None
        so_hieu = None
        article_id = None
        clause_id = None
        display = None
        if ctype == "text":
            chunk_id = ev.get("payload", {}).get("chunk_id") or ev.get("chunk_id")
            meta = registry.get_chunk(chunk_id) if chunk_id else None
            if meta:
                doc_id = meta.get("doc_id")
                doc_title = meta.get("doc_title")
                so_hieu = meta.get("so_hieu")
                article_id = meta.get("article_id")
                clause_id = meta.get("clause_id")
        elif ctype == "graph":
            triple_id = ev.get("payload", {}).get("triple_id") or ev.get("triple_id")
            meta = registry.get_triple(triple_id) if triple_id else None
            if not meta:
                article_id = ev.get("article_id")
                meta = registry.get_citation_by_article(article_id) if article_id else None
            if meta:
                doc_id = meta.get("doc_id")
                doc_title = meta.get("doc_title")
                so_hieu = meta.get("so_hieu")
                article_id = meta.get("article_id")
                clause_id = meta.get("clause_id")
        # Fallbacks
        if not article_id:
            article_id = ev.get("article_id")
        if not clause_id:
            clause_id = ev.get("clause_id")
        key = (doc_id, article_id, clause_id)
        if key in seen:
            continue
        seen.add(key)
        # Build display string
        if doc_id and doc_title and so_hieu and article_id:
            display = f"QĐ {so_hieu} – {doc_title}, Điều {article_id}" + (f", Khoản {clause_id}" if clause_id else "")
        elif article_id:
            display = f"Điều {article_id}" + (f", Khoản {clause_id}" if clause_id else "")
        else:
            display = "(Nguồn không xác định)"
        citations.append({
            "type": ctype,
            "doc_id": doc_id,
            "doc_title": doc_title,
            "so_hieu": so_hieu,
            "article_id": article_id,
            "clause_id": clause_id,
            "display": display,
        })
    return citations
