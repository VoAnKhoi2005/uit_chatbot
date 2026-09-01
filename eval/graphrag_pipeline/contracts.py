"""Artifact schemas for the UIT chatbot evaluation pipeline.

Adapted from SchemaGraph's graphrag_pipeline (see repo
/mnt/DATA/Github/SchemaGraph/evaluate/graphrag_pipeline). Kept: the
`_Artifact`/`ContractError`/io-helper foundation, `ResponseRow`, `RunConfig`,
`NodeManifest` - all generic. Dropped: the ingest/workspace models (we have
no remote workspace - the corpus is already indexed via
retrieval.text_rag.build_index / retrieval.src.retrieval.build_graph_index)
and the claims+clarity judge models (`Claim`, `ClarityScores`, `JudgeRow`,
...), replaced by RAGAS-shaped `RagasScoreRow`/`RagasSummary` below.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA_VERSION = 1

T = TypeVar("T", bound=BaseModel)


class ContractError(Exception):
    """A file could not be read as the artifact it was supposed to be."""

    def __init__(
        self,
        path: str | Path,
        message: str,
        line: int | None = None,
        field: str | None = None,
    ) -> None:
        self.path = str(path)
        self.message = message
        self.line = line
        self.field = field
        location = self.path
        if line is not None:
            location += f":{line}"
        if field:
            location += f" (field {field!r})"
        super().__init__(f"{location}: {message}")


class _Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(default=SCHEMA_VERSION)


class Source(BaseModel):
    # extra="allow": Path 1/Path 2 hits carry route-specific diagnostic keys
    # (bm25_score, dense_score, subject/predicate/object, ...) that vary by
    # which retrieval path produced them; forbidding extras would break
    # collect whenever the pipeline adds a field.
    model_config = ConfigDict(extra="allow")

    chunk_id: str | None = None
    article_id: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    content: str = ""
    score: float | None = None
    rank: int | None = None


class ResponseRow(_Artifact):
    qa_id: str
    type: str | None = None
    question: str
    sent_text: str
    answer: str | None = None
    golden_answer: str | None = None
    sources: list[Source] = Field(default_factory=list)
    question_type: str | None = None  # EXACT_RULE / NEAR_RULE / OUT_OF_SCOPE
    latency_ms: int | None = None
    status: Literal["ok", "error"]
    error: str | None = None
    ts: str


# ── RAGAS judge artifacts ───────────────────────────────────────────────────

# https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ - the
# subset relevant to a retrieval+generation pipeline without a fixed golden
# context list. Limited to answer_correctness ("accuracy" against the golden
# answer) rather than all five - each extra metric is another judge LLM call
# per question, and this is the one that actually needs a reference answer
# to mean anything. Add metrics back here (faithfulness, answer_relevancy,
# context_precision don't need golden_answer; context_recall does) if the
# speed tradeoff is worth it for a given run.
RAGAS_METRICS = ("answer_correctness",)


class RagasScoreRow(_Artifact):
    """One question's RAGAS scores, merged with its ResponseRow for context."""

    qa_id: str
    type: str | None = None
    question: str
    answer: str | None = None
    golden_answer: str | None = None
    n_contexts: int = 0
    scores: dict[str, float | None] = Field(default_factory=dict)
    judged_status: Literal["ok", "skipped", "collect_error"]
    reason: str | None = None


class RagasByType(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int
    n_ok: int
    mean_scores: dict[str, float | None] = Field(default_factory=dict)


class RagasSummary(_Artifact):
    n: int
    n_ok: int
    n_skipped: int
    n_collect_error: int
    mean_scores: dict[str, float | None] = Field(default_factory=dict)
    by_type: dict[str, RagasByType] = Field(default_factory=dict)
    model: str | None = None
    embedding_model: str | None = None


class InputRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str


class NodeManifest(_Artifact):
    node: str
    status: Literal["ok", "partial", "failed"]
    started_at: str
    finished_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, InputRef] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)


class RunConfig(_Artifact):
    run_id: str
    created_at: str
    system: Literal["uit_chatbot"] = "uit_chatbot"
    testset_path: str | None = None
    testset_sha256: str | None = None
    notes: str | None = None


# ── io helpers ────────────────────────────────────────────────────────────────

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _first_error_field(exc: ValidationError) -> str | None:
    errors = exc.errors()
    if not errors:
        return None
    return ".".join(str(part) for part in errors[0]["loc"]) or None


def _first_error_message(exc: ValidationError) -> str:
    errors = exc.errors()
    return errors[0]["msg"] if errors else str(exc)


def _check_schema_version(path: str | Path, data: Any, line: int | None) -> None:
    if not isinstance(data, dict):
        raise ContractError(path, f"expected a JSON object, got {type(data).__name__}", line=line)
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ContractError(
            path,
            f"unsupported schema_version {version!r} (this build reads {SCHEMA_VERSION})",
            line=line,
            field="schema_version",
        )


def _validate(path: str | Path, model: type[T], data: Any, line: int | None) -> T:
    _check_schema_version(path, data, line)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ContractError(
            path,
            _first_error_message(exc),
            line=line,
            field=_first_error_field(exc),
        ) from exc


def load_json(path: str | Path, model: type[T]) -> T:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(path, "file not found") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError(path, f"invalid JSON: {exc.msg}", line=exc.lineno) from exc
    return _validate(path, model, data, line=None)


def load_jsonl(path: str | Path, model: type[T]) -> list[T]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(path, "file not found") from exc
    rows: list[T] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(path, f"invalid JSON: {exc.msg}", line=lineno) from exc
        rows.append(_validate(path, model, data, line=lineno))
    return rows


def dump_json(path: str | Path, obj: BaseModel) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(obj.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, obj: BaseModel) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj.model_dump(mode="json"), ensure_ascii=False) + "\n")
        handle.flush()
