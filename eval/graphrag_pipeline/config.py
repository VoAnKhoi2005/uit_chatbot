"""Run-directory conventions and `run.json` handling.

Adapted from SchemaGraph's graphrag_pipeline/config.py: same `--run DIR`
shorthand, trimmed to the collect -> judge -> report node set (no `ingest` -
our corpus is already indexed, see eval/README.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from graphrag_pipeline.contracts import ContractError, RunConfig, dump_json, load_json, sha256_file

NODE_NAMES = ("collect", "judge", "report")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path

    def node_dir(self, name: str) -> Path:
        return self.run_dir / name

    @property
    def run_json(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def responses(self) -> Path:
        return self.run_dir / "collect" / "responses.jsonl"

    @property
    def collect_manifest(self) -> Path:
        return self.run_dir / "collect" / "manifest.json"

    @property
    def judge_results(self) -> Path:
        """Validated per-question RAGAS scores - the resume source for judge."""
        return self.run_dir / "judge" / "results.jsonl"

    @property
    def judge_summary(self) -> Path:
        return self.run_dir / "judge" / "summary.json"

    @property
    def judge_manifest(self) -> Path:
        return self.run_dir / "judge" / "manifest.json"

    @property
    def report_html(self) -> Path:
        return self.run_dir / "report" / "index.html"

    @property
    def report_manifest(self) -> Path:
        return self.run_dir / "report" / "manifest.json"


def load_pipeline_env(package_dir: Path | None = None) -> None:
    """Load env vars for the pipeline, without overriding values already in the shell.

    Two files, repo-root first: `collect` constructs ChatPipeline directly, so
    it needs the backend's own config (LLM_BASE_URL, UIT_TTL_PATH, ...) from
    the repo-root `.env` - the same one Docker Compose loads - not just this
    package's own `.env` (JUDGE_*-specific overrides only).
    """
    base = package_dir or Path(__file__).parent
    repo_root_env = base.parent.parent / ".env"
    load_dotenv(repo_root_env, override=False)
    load_dotenv(base / ".env", override=False)


def ensure_run_config(
    run_dir: str | Path,
    *,
    run_id: str | None = None,
    testset: str | Path | None = None,
    notes: str | None = None,
    now: str | None = None,
) -> RunConfig:
    """Read `run.json`, creating it and filling in fields on first knowledge.

    Setting an already-set field to a different value is an error, not an
    update: a run must not quietly become a description of something else.
    """
    paths = RunPaths(Path(run_dir))
    testset_path = Path(testset) if testset is not None else None
    if testset_path is not None and not testset_path.exists():
        raise ContractError(testset_path, "testset file not found")

    if not paths.run_json.exists():
        config = RunConfig(
            run_id=run_id or paths.run_dir.name,
            created_at=now or utcnow_iso(),
            testset_path=str(testset_path) if testset_path else None,
            testset_sha256=sha256_file(testset_path) if testset_path else None,
            notes=notes,
        )
        dump_json(paths.run_json, config)
        return config

    existing = load_json(paths.run_json, RunConfig)
    updated = existing.model_copy()
    changed = False

    if testset_path is not None:
        digest = sha256_file(testset_path)
        if existing.testset_path is None:
            updated.testset_path = str(testset_path)
            updated.testset_sha256 = digest
            changed = True
        elif existing.testset_path != str(testset_path):
            raise ContractError(
                paths.run_json,
                f"run was created with testset {existing.testset_path!r}, "
                f"but this invocation passed {str(testset_path)!r}",
                field="testset_path",
            )
        elif existing.testset_sha256 != digest:
            raise ContractError(
                paths.run_json,
                f"testset {str(testset_path)!r} changed on disk since this run recorded it "
                f"(sha256 mismatch)",
                field="testset_sha256",
            )

    if changed:
        dump_json(paths.run_json, updated)
    return updated


def resolve_testset(run_dir: str | Path, testset_flag: str | Path | None) -> Path:
    """`--testset` wins; otherwise take the path recorded in `run.json`."""
    if testset_flag is not None:
        return Path(testset_flag)
    paths = RunPaths(Path(run_dir))
    if not paths.run_json.exists():
        raise ContractError(paths.run_json, "no run.json and no --testset given")
    recorded = load_json(paths.run_json, RunConfig).testset_path
    if recorded is None:
        raise ContractError(paths.run_json, "run.json has no testset_path yet - pass --testset")
    return Path(recorded)


def env_str(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise ContractError("environment", f"required environment variable {name} is not set")
    return value
