from typing import List, Dict, Any

def hit_at_k(expected_ids, retrieved_ids, k=5):
    if not expected_ids:
        return None
    retrieved = set(retrieved_ids[:k])
    return bool(set(expected_ids) & retrieved)

def refusal_accuracy(expected_in_scope, predicted_intent, answer):
    if expected_in_scope is False:
        return predicted_intent == "OUT_OF_SCOPE" or "không thể trả lời" in answer.lower()
    return None

def faithfulness_proxy(answer, expected_keywords, expected_article_ids):
    if not expected_keywords and not expected_article_ids:
        return None
    for kw in (expected_keywords or []):
        if kw.lower() in answer.lower():
            return True
    for aid in (expected_article_ids or []):
        if aid in answer:
            return True
    return False

def citation_precision_proxy(citations, registry, expected_article_ids):
    if not expected_article_ids:
        return None
    for aid in expected_article_ids:
        if any(c.get("article_id") == aid for c in citations):
            meta = registry.get_citation_by_article(aid)
            if meta:
                return True
    return False
