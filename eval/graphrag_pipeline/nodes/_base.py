"""Shared CLI conventions for every node.

Nodes know nothing about each other or about the orchestrator. What they do
share is this: common flags, the exit-code contract, and a manifest that is
written whether the node succeeds or dies — a failed run still has to be
traceable.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

from graphrag_pipeline.config import utcnow_iso
from graphrag_pipeline.contracts import (
    ContractError,
    InputRef,
    NodeManifest,
    dump_json,
    sha256_file,
)

EXIT_OK = 0
EXIT_ITEM_FAILURES = 1
EXIT_INVALID = 2


def add_common_args(parser: argparse.ArgumentParser, *, force: bool = True) -> None:
    """Flags every node shares.

    `force` is opt-out because `ingest` has nothing to recompute: it creates a
    new workspace and uploads the whole corpus every time, so there is never a
    previous result to ignore. A flag that is accepted and does nothing is worse
    than one that does not exist.
    """
    parser.add_argument("--run", default=None, help="run directory; derives default input/output paths")
    parser.add_argument("--out", default=None, help="output directory for this node")
    if force:
        parser.add_argument("--force", action="store_true", help="recompute, ignoring existing output")


class NodeRun:
    """Context manager that owns one node's manifest."""

    def __init__(self, node: str, out_dir: str | Path, config: dict[str, Any]) -> None:
        self.node = node
        self.out_dir = Path(out_dir)
        self.config = config
        self.inputs: dict[str, InputRef] = {}
        self.counts: dict[str, int] = {}
        self._status: str | None = None
        self._started_at = utcnow_iso()

    def record_input(self, name: str, path: str | Path) -> None:
        self.inputs[name] = InputRef(path=str(path), sha256=sha256_file(path))

    def count(self, key: str, value: int) -> None:
        self.counts[key] = value

    def set_status(self, status: str) -> None:
        self._status = status

    def __enter__(self) -> "NodeRun":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        status = self._status or ("ok" if exc_type is None else "failed")
        if exc_type is not None:
            status = "failed"
        manifest = NodeManifest(
            node=self.node,
            status=status,
            started_at=self._started_at,
            finished_at=utcnow_iso(),
            config=self.config,
            inputs=self.inputs,
            counts=self.counts,
        )
        dump_json(self.out_dir / "manifest.json", manifest)
        return False


def run_cli(fn: Callable[[list[str]], int], argv: list[str]) -> int:
    """Translate exceptions into the pipeline's exit-code contract."""
    try:
        return fn(argv)
    except ContractError as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return EXIT_INVALID
    except KeyboardInterrupt:
        # States what happened rather than promising a resume. `collect` and
        # `judge` write each item as it finishes, so rerunning them does
        # continue from here; `ingest` creates a new workspace every run and
        # has nothing to continue from, so a blanket "rerun to resume" was
        # true for two nodes out of three.
        print("interrupted — finished items are on disk", file=sys.stderr)
        return EXIT_ITEM_FAILURES
    except Exception:  # noqa: BLE001 - top-level CLI boundary
        traceback.print_exc()
        return EXIT_ITEM_FAILURES
