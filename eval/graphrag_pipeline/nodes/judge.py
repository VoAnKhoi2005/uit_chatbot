"""Node: judge — score every collected answer with RAGAS.

Replaces SchemaGraph's hand-rolled claims+clarity judge (judges/, prompts/)
with the RAGAS library (https://docs.ragas.io), limited to `answer_correctness`
("accuracy" against the golden answer) rather than the full metric set - each
extra metric is another judge LLM call per question, and this is the one
metric that needs a reference answer to mean anything anyway. An item
without a `golden_answer` is judged_status="skipped" (not scored, not an
error) - accuracy can't be computed without something to compare against.

Judge LLM/embeddings are OpenAI-protocol and configurable via env vars
(JUDGE_MODEL/JUDGE_BASE_URL/JUDGE_API_KEY - unset, they fall back to the same
LLM_BASE_URL/LLM_API_KEY/LLM_MODEL the chat backend's own generic
backend/llm/client.py::LLMClient uses, so there's one place to point at an
OpenAI-protocol endpoint, not two; JUDGE_EMBED_MODEL similarly falls back to
EMBEDDING_MODEL/backend/retrieval/text_rag/embeddings.py's default, so no
second paid embedding API is required).
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


def _build_llm_and_embeddings():
    """Build ragas' LLM/embeddings wrappers. Isolated so a ragas version bump
    that changes these constructors only needs a change here."""
    from langchain_openai import ChatOpenAI
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper

    # JUDGE_* overrides let the judge use a different model/endpoint than chat
    # itself (e.g. a stronger judge model); unset, it reuses the same generic
    # OpenAI-protocol config backend/llm/client.py::LLMClient uses.
    model = os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL")
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not base_url:
        raise ContractError("environment", "JUDGE_BASE_URL or LLM_BASE_URL is required for the judge")
    if not model:
        raise ContractError("environment", "JUDGE_MODEL, LLM_MODEL, or OPENAI_MODEL is required for the judge")
    if not api_key:
        raise ContractError("environment", "JUDGE_API_KEY, LLM_API_KEY, or OPENAI_API_KEY is required for the judge")

    # max_tokens matters here, not just cosmetically: without it, some
    # OpenRouter models (e.g. reasoning/"flash" variants that emit hidden
    # reasoning tokens counted against the output budget) fall back to a
    # provider default too small to finish answer_correctness' structured
    # JSON output, which ragas surfaces as LLMDidNotFinishException("...
    # increase max_tokens...") rather than a clean truncation. JUDGE_MAX_TOKENS
    # overrides the default of 4096.
    #
    # Confirmed by a raw probe against this deployment: a trivial
    # decompose-into-statements call (the same shape answer_correctness makes
    # internally) burned 300 of its 319 completion tokens on hidden
    # `reasoning` content before answering, and disabling reasoning via
    # OpenRouter's `reasoning.enabled=False` dropped that to 0 with
    # finish_reason=stop - it's what was blowing through max_tokens (and,
    # earlier, the timeout) on the real structured-JSON prompts, not a token
    # limit that was merely a bit too low. A judge task needs accurate
    # classification, not visible chain-of-thought, so this is a legitimate
    # place to turn it off; JUDGE_DISABLE_REASONING=false restores it.
    max_tokens = int(os.getenv("JUDGE_MAX_TOKENS", "4096"))
    llm_kwargs: dict[str, Any] = dict(
        model=model, base_url=base_url, api_key=api_key, temperature=0, max_tokens=max_tokens
    )
    disable_reasoning = os.getenv("JUDGE_DISABLE_REASONING", "true").lower() not in {"false", "0", "no"}
    if disable_reasoning:
        llm_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    llm = LangchainLLMWrapper(ChatOpenAI(**llm_kwargs))

    # Same OpenAI-protocol embeddings backend/retrieval/text_rag/embeddings.py
    # uses (default: Qwen3 Embedding 8B) - ragas' own HuggingFaceEmbeddings
    # wrapper is version-mismatched against this ragas release (missing
    # embed_query), and this avoids a second, local embedding model besides.
    #
    # Must be AsyncOpenAI, not OpenAI: ragas' OpenAIEmbeddings auto-detects
    # sync vs async from the client type (`_check_client_async`) and
    # answer_correctness's similarity step always calls the async path
    # (`aembed_text`) - a sync client raises
    # "Cannot use aembed_text() with a synchronous client" instead of embedding.
    embed_base_url = os.getenv("EMBEDDING_BASE_URL") or base_url
    embed_api_key = os.getenv("EMBEDDING_API_KEY") or api_key
    embed_model = os.getenv("JUDGE_EMBED_MODEL") or os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-8b")
    embeddings = OpenAIEmbeddings(
        client=AsyncOpenAI(api_key=embed_api_key, base_url=embed_base_url), model=embed_model
    )

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
    from ragas.run_config import RunConfig

    metric_objs = {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
        "answer_correctness": AnswerCorrectness(),
    }
    selected = [metric_objs[m] for m in metrics]

    # ragas' own default timeout (180s) is too short for answer_correctness,
    # which chains several LLM calls per item (decompose answer, decompose
    # golden, classify each statement, embed for similarity) - a run that
    # times out doesn't raise, it just silently scores that item None/NaN
    # (see the per-Job "Exception raised: TimeoutError()" ragas prints to
    # stderr), which looks exactly like "the answer scored badly" unless you
    # go looking. JUDGE_TIMEOUT_S overrides the default of 900s.
    #
    # max_workers also matters: ragas' default of 16 fires every item's
    # sub-calls (decompose/classify/embed, several per item) concurrently,
    # which for a handful of testset items can mean 15-20+ simultaneous
    # requests against one OpenRouter model - easily enough to get queued
    # past the timeout on a rate-limited endpoint even though each call
    # individually would finish quickly. JUDGE_MAX_WORKERS overrides the
    # default of 4 to keep contention low for small eval runs.
    run_config = RunConfig(
        timeout=int(os.getenv("JUDGE_TIMEOUT_S", "900")),
        max_workers=int(os.getenv("JUDGE_MAX_WORKERS", "4")),
    )

    dataset = EvaluationDataset.from_list([_to_ragas_row(r) for r in rows])
    result = evaluate(
        dataset=dataset, metrics=selected, llm=llm, embeddings=embeddings, run_config=run_config, show_progress=True
    )
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

    model = embed_model = None
    scores_by_id: dict[str, dict[str, float | None]] = {}
    if with_golden:
        llm, embeddings, model, embed_model = _build_llm_and_embeddings()
        scores_by_id = _run_ragas(with_golden, RAGAS_METRICS, llm, embeddings)

    results: list[RagasScoreRow] = []
    for r in with_golden:
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
    for r in without_golden:
        results.append(
            RagasScoreRow(
                qa_id=r.qa_id,
                type=r.type or (testset.get(r.qa_id) or {}).get("type"),
                question=r.question,
                answer=r.answer,
                golden_answer=r.golden_answer,
                n_contexts=len(r.sources),
                scores={},
                judged_status="skipped",
                reason="no golden_answer - answer_correctness needs a reference to compare against",
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

    # require_golden=True: answer_correctness needs a reference to mean
    # anything, so a testset item missing golden_answer fails the whole
    # judge run here - before any judge LLM call is paid for - rather than
    # silently judging some items and not others.
    testset = load_testset(testset_path, require_golden=True)
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
