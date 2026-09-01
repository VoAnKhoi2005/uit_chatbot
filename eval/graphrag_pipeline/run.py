"""Thin orchestrator: run a contiguous range of nodes over one run directory.

Adapted from SchemaGraph's graphrag_pipeline/run.py, trimmed to the
collect -> judge -> report node set (no `ingest`).
"""
from __future__ import annotations

import argparse
import sys

from graphrag_pipeline.contracts import ContractError
from graphrag_pipeline.nodes import collect as collect_node
from graphrag_pipeline.nodes import judge as judge_node
from graphrag_pipeline.nodes import report as report_node
from graphrag_pipeline.nodes._base import EXIT_INVALID, EXIT_OK, run_cli

NODE_ORDER = ("collect", "judge", "report")

NODE_MAINS = {
    "collect": collect_node.main,
    "judge": judge_node.main,
    "report": report_node.main,
}


def select_nodes(from_node: str | None, to_node: str | None) -> list[str]:
    order = list(NODE_ORDER)
    for name in (from_node, to_node):
        if name is not None and name not in order:
            raise ContractError("arguments", f"unknown node {name!r} (known: {', '.join(order)})")
    start = order.index(from_node) if from_node else 0
    end = order.index(to_node) if to_node else len(order) - 1
    if start > end:
        raise ContractError("arguments", f"--from {from_node!r} comes after --to {to_node!r}")
    return order[start : end + 1]


def node_argv(name: str, args: argparse.Namespace) -> list[str]:
    argv = ["--run", args.run]
    if args.force:
        argv.append("--force")
    if name == "collect":
        if args.testset:
            argv += ["--testset", args.testset]
        if args.skip_types:
            argv += ["--skip-types", *args.skip_types]
        if args.limit:
            argv += ["--limit", str(args.limit)]
        argv += ["--workers", str(args.workers)]
    elif name == "judge":
        if args.testset:
            argv += ["--testset", args.testset]
    return argv


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="graphrag_pipeline.run")
    parser.add_argument("--run", required=True, help="run directory")
    parser.add_argument("--from", dest="from_node", default=None)
    parser.add_argument("--to", dest="to_node", default=None)
    parser.add_argument("--testset", default=None)
    parser.add_argument("--skip-types", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0, help="collect: stop after N questions; 0 = no limit")
    parser.add_argument("--workers", type=int, default=1, help="collect: questions asked concurrently")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    worst = EXIT_OK
    for name in select_nodes(args.from_node, args.to_node):
        print(f"\n=== {name} ===")
        code = NODE_MAINS[name](node_argv(name, args))
        if code == EXIT_INVALID:
            print(f"{name} rejected its input — stopping", file=sys.stderr)
            return EXIT_INVALID
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    sys.exit(run_cli(main, sys.argv[1:]))
