import argparse
import json
from retrieval.src.eval.datasets import load_eval_set
from retrieval.src.eval.metrics import hit_at_k, refusal_accuracy, faithfulness_proxy, citation_precision_proxy
from retrieval.src.registry.metadata_registry import MetadataRegistry
from backend.llm.orchestrator import ChatPipeline
from pathlib import Path
import logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_json', type=str, default='reports/eval_results.json')
    parser.add_argument('--output_md', type=str, default='reports/eval_summary.md')
    parser.add_argument('--registry_db', type=str, default='metadata_registry.db')
    args = parser.parse_args()

    Path("reports").mkdir(exist_ok=True)
    eval_set = load_eval_set(args.dataset)
    registry = MetadataRegistry(args.registry_db)
    pipeline = ChatPipeline()
    results = []
    for ex in eval_set:
        qid = ex["id"]
        question = ex["question"]
        expected_in_scope = ex.get("expected_in_scope", True)
        expected_article_ids = ex.get("expected_article_ids", [])
        expected_keywords = ex.get("expected_keywords", [])
        # Call pipeline
        out = pipeline.answer_question(question, debug=True)
        answer = out["answer"]
        grounding = out.get("grounding", {})
        predicted_article_id = grounding.get("article_id")
        predicted_intent = out.get("question_type")
        text_hits = out.get("text_hits", [])
        graph_hits = out.get("graph_hits", [])
        citations = []
        if predicted_article_id:
            meta = registry.get_citation_by_article(predicted_article_id)
            if meta:
                citations.append({"article_id": predicted_article_id, **meta})
        # Metrics
        res = {
            "id": qid,
            "question": question,
            "answer": answer,
            "predicted_article_id": predicted_article_id,
            "predicted_intent": predicted_intent,
            "text_hits": text_hits,
            "graph_hits": graph_hits,
            "citations": citations,
            "hit@5": hit_at_k(expected_article_ids, [h.get("article_id") for h in text_hits+graph_hits], k=5),
            "refusal_acc": refusal_accuracy(expected_in_scope, predicted_intent, answer),
            "faithfulness": faithfulness_proxy(answer, expected_keywords, expected_article_ids),
            "citation_precision": citation_precision_proxy(citations, registry, expected_article_ids),
        }
        results.append(res)
        print(f"[{qid}] hit@5={res['hit@5']} refusal={res['refusal_acc']} faithfulness={res['faithfulness']} citation={res['citation_precision']}")
    # Save json
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # Save md summary
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("| id | hit@5 | refusal | faithfulness | citation |\n|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['id']} | {r['hit@5']} | {r['refusal_acc']} | {r['faithfulness']} | {r['citation_precision']} |\n")
        pass_rate = sum(1 for r in results if r['hit@5']) / len(results)
        f.write(f"\n**Pass rate (hit@5): {pass_rate:.2%}**\n")
    print(f"Eval done. Pass rate (hit@5): {pass_rate:.2%}")

if __name__ == "__main__":
    main()
