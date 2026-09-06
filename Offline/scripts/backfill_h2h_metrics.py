#!/usr/bin/env python3
"""Backfill answer metrics for completed non-HiveMem H2HMEM runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.io_utils import write_json_atomic
from benchmarks.memgallery_harness.runner.metrics import (
    merge_llm_judge_metrics,
    summarize_results,
)


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = OFFLINE_ROOT / "outputs" / "H2HMEM"


def load_completed_metrics(result_dir: Path, *, top_k: int = 7) -> dict[str, Any]:
    results_path = result_dir / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(results, list) or not results:
        raise ValueError(f"Expected a non-empty result list: {results_path}")
    metrics = summarize_results(results, k=top_k)

    judge_path = result_dir / "llm_judge_metrics.json"
    if judge_path.is_file():
        judge = json.loads(judge_path.read_text(encoding="utf-8"))
        expected = len(results)
        judge_complete = (
            judge.get("count") == expected
            and judge.get("valid_count") == expected
            and int(judge.get("judge_errors") or 0) == 0
            and not judge.get("provisional", True)
        )
        if not judge_complete:
            raise ValueError(f"Judge artifact is incomplete: {judge_path}")
        metrics = merge_llm_judge_metrics(metrics, judge)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--top-k", type=int, default=7)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    output_root = Path(args.output_root)
    methods = args.method or sorted(
        path.name
        for path in output_root.iterdir()
        if path.is_dir() and path.name != "HiveMem"
    )
    for method in methods:
        if method == "HiveMem":
            raise ValueError("HiveMem is intentionally excluded from this backfill")
        result_dir = output_root / method
        metrics = load_completed_metrics(result_dir, top_k=args.top_k)
        write_json_atomic(result_dir / "metrics.json", metrics)
        print(
            f"{method}: count={metrics['count']} f1={metrics['f1']:.12g} "
            f"em={metrics['em']:.12g} judge={metrics.get('llm_judge')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
