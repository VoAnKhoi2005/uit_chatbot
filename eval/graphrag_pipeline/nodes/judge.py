"""Node: judge — score every collected answer with RAGAS.

Replaces SchemaGraph's hand-rolled claims+clarity judge (judges/, prompts/)
with the RAGAS library (https://docs.ragas.io). Two calls to `ragas.evaluate`
per run, not one: `context_recall` and `answer_correctness` need a golden
answer and the rest don't, so an item without one only skips those two
metrics rather than being dropped or scored against a blank reference.

Judge LLM/embeddings are OpenAI-API-compatible and configurable via env vars
(JUDGE_MODEL/JUDGE_BASE_URL/JUDGE_API_KEY - defaults reuse this repo's own
GROQ_API_KEY, since Groq's endpoint is OpenAI-compatible; JUDGE_EMBED_MODEL
defaults to the same sentence-transformers model retrieval already uses, via
HuggingfaceEmbeddings, so no second paid embedding API is required).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from graphrag_pipeline.config import RunPaths, ensure_run_config, load_pipeline_env, resolve_testset
from graphrag_pipeline.contracts import (
    ContractError,
    RAGAS_METRICS,
    RagasByType,
    RagasScoreRow,
    RagasSummary,
    ResponseRow,
    append_jsonl,
    dump_json,
    load_jsonl,
)
from graphrag_pipeline.nodes._base import EXIT_ITEM_FAILURES, EXIT_OK, NodeRun, add_common_args, run_cli
from graphrag_pipeline.testset import load_testset

REFERENCE_METRICS = {"context_recall", "answer_correctness"}
REFERENCE_FREE_METRICS = tuple(m for m in RAGAS_METRICS if m not in REFERENCE_METRICS)


def _build_llm_and_embeddings():
    """Build ragas' LLM/embeddings wrappers. Isolated so a ragas version bump
    that changes these constructors only needs a change here."""
    from langchain_openai import ChatOpenAI
    from ragas.embeddings import HuggingfaceEmbeddings
    from ragas.llms import LangchainLLMWrapper

    model = os.getenv("JUDGE_MODEL", "llama-3.3-70b-versatile")
    base_url = os.getenv("JUDGE_BASE_URL", "https://api.groq.com/openai/v1")
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ContractError("environment", "JUDGE_API_KEY/GROQ_API_KEY/OPENAI_API_KEY is required for the judge")

    llm = LangchainLLMWrapper(ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0))

    embed_model = os.getenv("JUDGE_EMBED_MODEL", "keepitreal/vietnamese-sbert")
    embeddings = HuggingfaceEmbeddings(model_name=embed_model)

    return llm, embeddings, model, embed_model


def _to_ragas_row(row: ResponseRow) -> dict[str, Any]:
    return {
        "user_input": row.question,
        "response": row.answer or "",
        "retrieved_contexts": [s.content for s in row.sources if s.content],
        "reference": row.golden_answer or "",
    }


def _run_ragas(rows: list[ResponseRow], metrics: tuple[str, ...], llm, embeddings) -> dict[str, dict[str, float | None]]:
    """Returns {qa_id: {metric_name: score}}. Empty dict input -> empty output,
    since ragas.evaluate rejects an empty dataset."""
    if not rows:
        return {}

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

    metric_objs = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
        "answer_correctness": AnswerCorrectness(),
    }
    selected = [metric_objs[m] for m in metrics]

    dataset = EvaluationDataset.from_list([_to_ragas_row(r) for r in rows])
    result = evaluate(dataset=dataset, metrics=selected, llm=llm, embeddings=embeddings, show_progress=True)
    df = result.to_pandas()

    out: dict[str, dict[str, float | None]] = {}
    for row, (_, scored) in zip(rows, df.iterrows()):
        out[row.qa_id] = {m: (float(scored[m]) if m in scored and scored[m] == scored[m] else None) for m in metrics}
    return out


def judge_all(rows: list[ResponseRow], testset: dict[str, dict]) -> list[RagasScoreRow]:
    collect_errors = [r for r in rows if r.status != "ok"]
    scoreable = [r for r in rows if r.status == "ok"]
    with_golden = [r for r in scoreable if (r.golden_answer or "").strip()]
    without_golden = [r for r in scoreable if not (r.golden_answer or "").strip()]

    if with_golden or without_golden:
        llm, embeddings, model, embed_model = _build_llm_and_embeddings()
    else:
        llm = embeddings = model = embed_model = None

    scores_by_id: dict[str, dict[str, float | None]] = {}
    if with_golden:
        scores_by_id.update(_run_ragas(with_golden, RAGAS_METRICS, llm, embeddings))
    if without_golden:
        scores_by_id.update(_run_ragas(without_golden, REFERENCE_FREE_METRICS, llm, embeddings))

    results: list[RagasScoreRow] = []
    for r in scoreable:
        results.append(
            RagasScoreRow(
                qa_id=r.qa_id,
                type=r.type or (testset.get(r.qa_id) or {}).get("type"),
                question=r.question,
                answer=r.answer,
                golden_answer=r.golden_answer,
                n_contexts=len(r.sources),
                scores=scores_by_id.get(r.qa_id, {}),
                judged_status="ok",
            )
        )
    for r in collect_errors:
        results.append(
            RagasScoreRow(
                qa_id=r.qa_id,
                type=r.type or (testset.get(r.qa_id) or {}).get("type"),
                question=r.question,
                answer=r.answer,
                golden_answer=r.golden_answer,
                n_contexts=0,
                scores={},
                judged_status="collect_error",
                reason=r.error,
            )
        )
    return results, (model, embed_model)


def summarise(results: list[RagasScoreRow], models: tuple[str | None, str | None]) -> RagasSummary:
    ok = [r for r in results if r.judged_status == "ok"]
    by_type: dict[str, list[RagasScoreRow]] = {}
    for r in ok:
        by_type.setdefault(r.type or "untyped", []).append(r)

    def mean_scores(items: list[RagasScoreRow]) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for metric in RAGAS_METRICS:
            values = [r.scores[metric] for r in items if r.scores.get(metric) is not None]
            out[metric] = (sum(values) / len(values)) if values else None
        return out

    return RagasSummary(
        n=len(results),
        n_ok=len(ok),
        n_skipped=0,
        n_collect_error=len([r for r in results if r.judged_status == "collect_error"]),
        mean_scores=mean_scores(ok),
        by_type={
            t: RagasByType(n=len(items), n_ok=len(items), mean_scores=mean_scores(items))
            for t, items in by_type.items()
        },
        model=models[0],
        embedding_model=models[1],
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="graphrag_pipeline.nodes.judge")
    add_common_args(parser)
    parser.add_argument("--testset", default=None)
    parser.add_argument("--responses", default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    load_pipeline_env()

    if args.run:
        paths = RunPaths(Path(args.run))
        ensure_run_config(args.run, testset=args.testset)
        testset_path = resolve_testset(args.run, args.testset)
        responses_path = Path(args.responses) if args.responses else paths.responses
        out_dir = Path(args.out) if args.out else paths.node_dir("judge")
    else:
        if not (args.testset and args.responses and args.out):
            raise ContractError("arguments", "without --run you must pass --testset, --responses and --out")
        testset_path = Path(args.testset)
        responses_path = Path(args.responses)
        out_dir = Path(args.out)

    testset = load_testset(testset_path)
    rows = load_jsonl(responses_path, ResponseRow)
    if not rows:
        raise ContractError(responses_path, "no responses to judge")

    with NodeRun("judge", out_dir, config={"testset": str(testset_path)}) as run:
        run.record_input("testset", testset_path)
        run.record_input("responses", responses_path)

        results, models = judge_all(rows, testset)
        results_path = out_dir / "results.jsonl"
        if results_path.exists() and args.force:
            results_path.unlink()
        for r in results:
            append_jsonl(results_path, r)

        summary = summarise(results, models)
        dump_json(out_dir / "summary.json", summary)

        run.count("total", len(results))
        run.count("ok", summary.n_ok)
        run.count("collect_error", summary.n_collect_error)
        run.set_status("ok" if summary.n_collect_error == 0 else "partial")

    return EXIT_OK if summary.n_collect_error == 0 else EXIT_ITEM_FAILURES


if __name__ == "__main__":
    sys.exit(run_cli(main, sys.argv[1:]))
