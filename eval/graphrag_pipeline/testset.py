"""Reading the QA test set.

Adapted from SchemaGraph's graphrag_pipeline/testset.py. Our items are
simpler - no fixed 8-type taxonomy (that gated which claims/clarity rubric
prompt to use; RAGAS's metrics don't need one). A minimal item is
`{"qa_id": ..., "question": ...}`; `golden_answer` unlocks the
context_recall/answer_correctness RAGAS metrics, `type` is free-form and only
used to group the report, `context_turns` is passed straight through to
ChatPipeline.answer_question's own conversation_history.

Example testset item:
    {
      "qa_id": "q1",
      "type": "credit_limit",
      "question": "Sinh viên được đăng ký tối đa bao nhiêu tín chỉ?",
      "golden_answer": "14 đến 24 tín chỉ, tối đa 30 nếu ĐTBC >= 8.0.",
      "context_turns": [{"role": "user", "content": "..."}, {"role": "bot", "content": "..."}]
    }
"""
from __future__ import annotations

import json
from pathlib import Path

from graphrag_pipeline.contracts import ContractError


def load_testset(path: str | Path, *, require_golden: bool = False) -> dict[str, dict]:
    """Return `{qa_id: item}`.

    `require_golden` is off by default: a smoke run against a handful of
    questions is legitimate without golden answers, it just means the judge
    skips context_recall/answer_correctness for those items (see nodes/judge.py).
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(path, "testset file not found") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(path, f"invalid JSON: {exc.msg}", line=exc.lineno) from exc
    if not isinstance(data, list):
        raise ContractError(path, f"expected a JSON list of items, got {type(data).__name__}")

    items: dict[str, dict] = {}
    offenders: list[str] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ContractError(path, f"item {index} is not an object")
        qa_id = item.get("qa_id")
        if not qa_id:
            raise ContractError(path, f"item {index} has no qa_id", field="qa_id")
        if qa_id in items:
            raise ContractError(path, f"duplicate qa_id {qa_id!r}", field="qa_id")
        if not item.get("question"):
            raise ContractError(path, f"item {qa_id!r} has no question", field="question")
        items[qa_id] = item

        if require_golden:
            golden = item.get("golden_answer")
            if golden is None or not str(golden).strip():
                offenders.append(f"{qa_id}: golden_answer is blank")

    if offenders:
        raise ContractError(
            path,
            "test set items are not usable - " + "; ".join(offenders),
            field="golden_answer",
        )
    return items
