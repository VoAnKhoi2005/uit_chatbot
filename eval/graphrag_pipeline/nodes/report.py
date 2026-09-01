"""Node: report — one self-contained HTML page from the RAGAS judge results.

Simpler than SchemaGraph's claims/clarity dashboard (templates/report.html,
dropped): one table of per-question scores plus per-metric/per-type means,
no JS framework, no external assets - a single static file safe to open
straight from disk.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

from graphrag_pipeline.config import RunPaths
from graphrag_pipeline.contracts import (
    RAGAS_METRICS,
    ContractError,
    RagasScoreRow,
    RagasSummary,
    load_json,
    load_jsonl,
)
from graphrag_pipeline.nodes._base import EXIT_OK, NodeRun, add_common_args, run_cli

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
    "answer_correctness": "Answer correctness",
}


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def _esc(text: str | None) -> str:
    return html.escape(text or "", quote=True)


def render_html(results: list[RagasScoreRow], summary: RagasSummary) -> str:
    metric_cols = "".join(f"<th>{METRIC_LABELS[m]}</th>" for m in RAGAS_METRICS)
    summary_row = "".join(f"<td>{_fmt(summary.mean_scores.get(m))}</td>" for m in RAGAS_METRICS)

    by_type_rows = []
    for t, stats in sorted(summary.by_type.items()):
        cells = "".join(f"<td>{_fmt(stats.mean_scores.get(m))}</td>" for m in RAGAS_METRICS)
        by_type_rows.append(f"<tr><td>{_esc(t)}</td><td>{stats.n}</td>{cells}</tr>")

    detail_rows = []
    for r in sorted(results, key=lambda r: r.qa_id):
        status_class = "ok" if r.judged_status == "ok" else "err"
        cells = "".join(f"<td>{_fmt(r.scores.get(m))}</td>" for m in RAGAS_METRICS)
        detail_rows.append(
            f"<tr class='{status_class}'>"
            f"<td>{_esc(r.qa_id)}</td><td>{_esc(r.type)}</td>"
            f"<td class='q'>{_esc(r.question)}</td>"
            f"<td class='a'>{_esc((r.answer or '')[:400])}</td>"
            f"<td>{r.n_contexts}</td>"
            f"{cells}"
            f"<td>{_esc(r.judged_status)}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>UIT chatbot - RAGAS evaluation report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ margin-bottom: 0.2rem; }}
  .meta {{ color: #666; margin-bottom: 1.5rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f4f4f4; position: sticky; top: 0; }}
  tr.err {{ background: #fff3f3; }}
  td.q, td.a {{ max-width: 320px; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.2rem; min-width: 140px; }}
  .card .value {{ font-size: 1.6rem; font-weight: 600; }}
  .card .label {{ color: #666; font-size: 0.85rem; }}
</style>
</head>
<body>
<h1>UIT chatbot - RAGAS evaluation report</h1>
<div class="meta">
  n={summary.n} · ok={summary.n_ok} · collect_error={summary.n_collect_error} ·
  judge model={_esc(summary.model)} · embedding model={_esc(summary.embedding_model)}
</div>

<div class="cards">
{"".join(f'<div class="card"><div class="value">{_fmt(summary.mean_scores.get(m))}</div><div class="label">{METRIC_LABELS[m]}</div></div>' for m in RAGAS_METRICS)}
</div>

<h2>By type</h2>
<table>
<tr><th>Type</th><th>n</th>{metric_cols}</tr>
<tr><td><b>overall</b></td><td>{summary.n_ok}</td>{summary_row}</tr>
{"".join(by_type_rows)}
</table>

<h2>Per-question</h2>
<table>
<tr><th>qa_id</th><th>type</th><th>question</th><th>answer</th><th>#contexts</th>{metric_cols}<th>status</th></tr>
{"".join(detail_rows)}
</table>
</body>
</html>
"""


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="graphrag_pipeline.nodes.report")
    add_common_args(parser, force=False)
    parser.add_argument("--results", default=None)
    parser.add_argument("--summary", default=None)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if args.run:
        paths = RunPaths(Path(args.run))
        results_path = Path(args.results) if args.results else paths.judge_results
        summary_path = Path(args.summary) if args.summary else paths.judge_summary
        out_dir = Path(args.out) if args.out else paths.node_dir("report")
    else:
        if not (args.results and args.summary and args.out):
            raise ContractError("arguments", "without --run you must pass --results, --summary and --out")
        results_path = Path(args.results)
        summary_path = Path(args.summary)
        out_dir = Path(args.out)

    results = load_jsonl(results_path, RagasScoreRow)
    summary = load_json(summary_path, RagasSummary)

    with NodeRun("report", out_dir, config={}) as run:
        run.record_input("results", results_path)
        run.record_input("summary", summary_path)
        html_out = render_html(results, summary)
        (out_dir / "index.html").write_text(html_out, encoding="utf-8")
        run.count("rows", len(results))
        run.set_status("ok")

    print(f"report written to {out_dir / 'index.html'}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(run_cli(main, sys.argv[1:]))
