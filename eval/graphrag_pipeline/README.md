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

`collect` calls `ChatPipeline` in-process (`adapters/uit_chatbot_client.py`)
instead of a remote chat API - no server needs to be running, and retrieval
evidence (`debug.text_hits`/`debug.graph_hits`) comes back in the same call
as the answer, so there's no separate trace file to poll.

## Install and run

This imports the backend directly (`ChatPipeline`, `TextEmbedder`, ...), so
both requirement sets are needed:

```bash
pip install -r ../../backend/requirements.txt
pip install -r requirements.txt
cp .env.example .env   # fill in JUDGE_*, or leave blank to reuse the backend's LLM_* config
```

Run the whole pipeline (from this directory):

```bash
python -m graphrag_pipeline.run --run runs/2026-09-01-baseline \
  --testset ../testset_example.json --workers 4
```

Or one node at a time, e.g. to re-judge without re-collecting:

```bash
python -m graphrag_pipeline.run --run runs/2026-09-01-baseline --from judge
```

`--limit N` (collect only) caps how many questions are asked - useful for a
smoke run before spending a full pass. `--force` discards existing collect
responses / judge results for the range being run.

## Test set format

A plain JSON list. Only `qa_id` and `question` are required:

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
- `golden_answer` unlocks two extra RAGAS metrics (see below); items without
  one still get scored on the other three.
- `context_turns` (optional, `[{"role": "user"|"bot", "content": "..."}]`) is
  passed straight through to `ChatPipeline.answer_question`'s own
  `conversation_history`, for multi-turn test items.

`eval/testset_example.json` has a couple of worked examples.

## Score

Five [RAGAS metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/):

| Metric | Needs `golden_answer` | What it measures |
|---|:---:|---|
| `faithfulness` | no | is the answer grounded in the retrieved context, or does it add unsupported claims |
| `answer_relevancy` | no | does the answer actually address the question |
| `context_precision` | no | are the retrieved chunks/graph facts relevant, ranked well |
| `context_recall` | yes | did retrieval surface what the golden answer needed |
| `answer_correctness` | yes | how close the answer is to the golden answer |

An item without a `golden_answer` is still judged - just on the first three.
The judge splits collected rows into "has golden" / "doesn't" and runs RAGAS
once per group, rather than either failing the whole run over one missing
golden answer or silently scoring against a blank reference.

Judge LLM and embedding model are configurable via env vars (`JUDGE_MODEL`,
`JUDGE_BASE_URL`, `JUDGE_API_KEY`, `JUDGE_EMBED_MODEL`) - see `.env.example`.
Left unset, they fall back to the same `LLM_BASE_URL`/`LLM_API_KEY`/`LLM_MODEL`
the chat backend's own generic client uses (any OpenAI-protocol endpoint), and
embeddings fall back to the same `sentence-transformers` model retrieval
already uses locally - so a run needs only one paid API (the judge LLM calls)
unless you deliberately point the judge somewhere different.

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
