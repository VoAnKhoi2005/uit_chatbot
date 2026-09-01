# UIT chatbot evaluation pipeline

Adapted from SchemaGraph's `evaluate/graphrag_pipeline` (a separate project,
`/mnt/DATA/Github/SchemaGraph/evaluate/graphrag_pipeline` on this machine).
Kept: the `--run DIR` node/manifest/contract conventions, resumable
`collect`, and the artifact-validation approach. Dropped: `ingest` (no
remote workspace here - the corpus is already indexed via
`retrieval.text_rag.build_index` / `retrieval.src.retrieval.build_graph_index`,
see the main [README.md](../../README.md)'s "Data rebuild" section) and the
hand-rolled claims+clarity LLM judge, replaced by
[RAGAS](https://docs.ragas.io).

Three file-contract nodes:

```text
collect -> judge -> report
```

`collect` calls the backend's own `POST /chat` over HTTP
(`adapters/uit_chatbot_client.py`) - it never imports anything from
`backend/`, so this venv only ever needs this package's own requirements,
not the backend's (fastapi, faiss-cpu, rdflib, ...). The backend just needs
to be reachable (`docker compose up backend` from the repo root, or any
other way it's running) - `EVAL_BACKEND_URL` points at it, default
`http://localhost:10000`. The backend's `/chat` always returns
`debug.text_hits`/`debug.graph_hits` in the same response as the answer, so
there's no separate trace call to poll.

## Install and run

```bash
pip install -r requirements.txt
cp .env.example .env   # EVAL_BACKEND_URL, plus JUDGE_* (or leave those blank to reuse the backend's LLM_*/EMBEDDING_* config)
```

Run the whole pipeline. `graphrag_pipeline` is a package (this directory has
its `__init__.py`), so `python -m graphrag_pipeline.run` needs to be run from
**`eval/`, one level up** - not from inside `eval/graphrag_pipeline/` itself:

```bash
cd ..   # eval/, if you were inside eval/graphrag_pipeline/
python -m graphrag_pipeline.run --run graphrag_pipeline/runs/2026-09-01-baseline \
  --testset testset_example.json --workers 4
```

Or one node at a time, e.g. to re-judge without re-collecting:

```bash
python -m graphrag_pipeline.run --run graphrag_pipeline/runs/2026-09-01-baseline --from judge
```

`--limit N` (collect only) caps how many questions are asked - useful for a
smoke run before spending a full pass. `--force` discards existing collect
responses / judge results for the range being run.

## Test set format

A plain JSON list. `qa_id`, `question`, and **`golden_answer` are all
required** - the judge (`answer_correctness`) needs a reference to compare
against, and a testset item missing one fails the whole judge run before any
LLM call is paid for, rather than silently scoring some items and not others:

```json
[
  {
    "qa_id": "q1",
    "type": "credit_limit",
    "question": "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong 1 học kỳ chính?",
    "golden_answer": "14 đến 24 tín chỉ, tối đa 30 nếu ĐTBC >= 8.0."
  }
]
```

- `type` is free-form and only used to group the report - there's no fixed
  taxonomy to satisfy.
- `context_turns` (optional, `[{"role": "user"|"bot", "content": "..."}]`) is
  sent as-is as the request's `conversation_history`, for multi-turn test items.

`eval/testset_example.json` has a couple of worked examples.

## Score

Judged on [`answer_correctness`](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/)
only - how close the answer is to the golden answer - rather than the full
five-metric RAGAS set (`faithfulness`, `answer_relevancy`,
`context_precision`, `context_recall` too). Each extra metric is another
judge LLM call per question; `answer_correctness` is the one that actually
needs `golden_answer` to mean anything, so it's the one kept. Add the others
back in `contracts.py::RAGAS_METRICS` (and `nodes/judge.py::_run_ragas`'s
`metric_objs` map already has all five available) if the speed tradeoff is
worth it for a given run.

Judge LLM and embedding model are configurable via env vars (`JUDGE_MODEL`,
`JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_EMBED_MODEL`) - see `.env.example`.
Left unset, they fall back to the same `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`
(and `EMBEDDING_*`) the chat backend's own generic client and embedder use
(any OpenAI-protocol endpoint; default embedding model Qwen3 Embedding 8B) -
so a run needs no separate judge-specific config unless you deliberately want
the judge on a different model/endpoint than chat/retrieval.

## Artifacts

```text
runs/<run-id>/
  run.json                    # testset path + sha256, for resume validation
  collect/responses.jsonl     # one ResponseRow per question (resume source)
  collect/manifest.json
  judge/results.jsonl         # one RagasScoreRow per question (resume source)
  judge/summary.json          # RagasSummary: means overall + by type
  judge/manifest.json
  report/index.html           # self-contained, single file
  report/manifest.json
```

Exit codes: `0` clean, `1` some items errored (rest still ran, still worth
reading), `2` invalid input/config (nothing ran).
