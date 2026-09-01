"""Node: collect — ask the UIT chatbot every test-set question, keep what really happened.

Adapted from SchemaGraph's graphrag_pipeline/nodes/collect.py. Calls the
backend's own `POST /chat` over HTTP (see adapters/uit_chatbot_client.py) -
this node never imports anything from `backend/`, so it only needs
eval/graphrag_pipeline/requirements.txt installed, and the backend just needs
to be reachable (e.g. `docker compose up backend`, EVAL_BACKEND_URL pointed
at it). No RAG-trace-file polling either way: the retrieval evidence
(`debug.text_hits`/`debug.graph_hits`) comes back in the same response as
the answer.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from graphrag_pipeline.adapters.uit_chatbot_client import UitChatbotClient
from graphrag_pipeline.config import RunPaths, ensure_run_config, load_pipeline_env, resolve_testset, utcnow_iso
from graphrag_pipeline.contracts import ContractError, ResponseRow, append_jsonl
from graphrag_pipeline.nodes._base import EXIT_ITEM_FAILURES, EXIT_OK, NodeRun, add_common_args, run_cli
from graphrag_pipeline.testset import load_testset


async def collect_one(item: dict, client: UitChatbotClient, ts: str) -> ResponseRow:
    question = item["question"]
    completion = await client.complete(question, conversation_history=item.get("context_turns"))

    if completion.error is not None or completion.answer is None:
        return ResponseRow(
            qa_id=item["qa_id"],
            type=item.get("type"),
            question=question,
            sent_text=question,
            answer=completion.answer,
            golden_answer=item.get("golden_answer"),
            sources=[],
            status="error",
            error=completion.error or "no answer returned",
            latency_ms=completion.latency_ms,
            ts=ts,
        )

    return ResponseRow(
        qa_id=item["qa_id"],
        type=item.get("type"),
        question=question,
        sent_text=question,
        answer=completion.answer,
        golden_answer=item.get("golden_answer"),
        sources=completion.sources,
        question_type=completion.question_type,
        latency_ms=completion.latency_ms,
        status="ok",
        error=None,
        ts=ts,
    )


async def _run_collect(items: list[dict], workers: int, responses_path: Path) -> int:
    client = UitChatbotClient()
    semaphore = asyncio.Semaphore(max(1, workers))
    written = 0
    write_lock = asyncio.Lock()

    async def worker(item: dict) -> None:
        nonlocal written
        async with semaphore:
            row = await collect_one(item, client, utcnow_iso())
        async with write_lock:
            append_jsonl(responses_path, row)
            written += 1
            print(f"[{written}/{len(items)}] {row.qa_id} status={row.status}")

    try:
        await asyncio.gather(*(worker(item) for item in items))
    finally:
        await client.aclose()
    return written


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="graphrag_pipeline.nodes.collect")
    add_common_args(parser)
    parser.add_argument("--testset", default=None)
    parser.add_argument("--skip-types", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--workers", type=int, default=1, help="questions asked concurrently (1 = sequential)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if args.run:
        paths = RunPaths(Path(args.run))
        ensure_run_config(args.run, testset=args.testset)
        testset_path = resolve_testset(args.run, args.testset)
        out_dir = Path(args.out) if args.out else paths.node_dir("collect")
    else:
        if not (args.testset and args.out):
            raise ContractError("arguments", "without --run you must pass --testset and --out")
        testset_path = Path(args.testset)
        out_dir = Path(args.out)

    items = list(load_testset(testset_path).values())
    if args.skip_types:
        items = [it for it in items if it.get("type") not in args.skip_types]
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise ContractError(testset_path, "no items left to collect after filtering")

    responses_path = out_dir / "responses.jsonl"
    if args.force and responses_path.exists():
        responses_path.unlink()

    already_done = set()
    if responses_path.exists():
        for line in responses_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                import json

                already_done.add(json.loads(line)["qa_id"])
    pending = [it for it in items if it["qa_id"] not in already_done]

    with NodeRun("collect", out_dir, config={"testset": str(testset_path), "workers": args.workers}) as run:
        run.record_input("testset", testset_path)
        run.count("total", len(items))
        run.count("already_done", len(already_done))
        run.count("pending", len(pending))

        if pending:
            load_pipeline_env()
            n_written = asyncio.run(_run_collect(pending, args.workers, responses_path))
        else:
            n_written = 0

        n_error = 0
        if responses_path.exists():
            import json

            for line in responses_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line).get("status") == "error":
                    n_error += 1
        run.count("collected_this_run", n_written)
        run.count("errors_total", n_error)
        run.set_status("ok" if n_error == 0 else "partial")

    return EXIT_OK if not pending or n_written == len(pending) else EXIT_ITEM_FAILURES


if __name__ == "__main__":
    sys.exit(run_cli(main, sys.argv[1:]))
