#!/usr/bin/env python3
"""Merge completed non-HiveMem MB-call measurements into formal metrics files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from benchmarks.io_utils import write_json_atomic
from benchmarks.memgallery_harness.runner.metrics import (
    calculate_calls_qa,
    combine_call_metrics,
)

try:
    from scripts.run_full_baseline_matrix import BASELINES, BENCHMARKS, OUTPUT_ROOT
except ModuleNotFoundError:  # Direct execution adds scripts/, not its parent.
    from run_full_baseline_matrix import BASELINES, BENCHMARKS, OUTPUT_ROOT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MB_CALL_ROOT = ROOT / "outputs" / "mb_call_rerun_20260904"
EXPECTED_SAMPLE_COUNTS = {
    "Mem-Gallery": 20,
    "WorldMemArena": 38,
    "H2HMEM": 25,
}


def _load_dict(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _load_list(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected a JSON object list: {path}")
    return payload


def normalize_mb_metric(metrics: dict[str, Any], *, expected: int) -> dict[str, Any]:
    """Validate a measurement artifact and adapt it to ``calls.memory_bank``."""
    if not metrics.get("available"):
        raise ValueError("MB-call metrics are not marked available")
    if metrics.get("num_samples") != expected:
        raise ValueError(
            f"MB-call sample count is {metrics.get('num_samples')!r}; expected {expected}"
        )
    if metrics.get("completed_samples") != expected or metrics.get("failed_samples") != 0:
        raise ValueError("MB-call measurement has incomplete or failed samples")
    total = int(metrics.get("total_calls"))
    failed = int(metrics.get("failed_calls"))
    successful = int(metrics.get("successful_calls"))
    if total < 0 or failed < 0 or successful < 0 or failed + successful != total:
        raise ValueError("MB-call measurement has inconsistent call totals")
    mean = total / expected
    return {
        "total_calls": total,
        "failed_calls": failed,
        "successful_calls": successful,
        "num_samples": expected,
        "mean_per_sample": mean,
        "formula": f"{total} / {expected} = {mean:.12g}",
        "aggregation": "build_calls_divided_by_samples",
        "available": True,
    }


def calculate_formal_qa_metric(
    benchmark: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate QA calls while preserving H2HMEM's composite sample identity."""
    if benchmark == "Mem-Gallery":
        return calculate_calls_qa(results, sample_id_field="dataset")
    if benchmark == "WorldMemArena":
        return calculate_calls_qa(results, sample_id_field="sample_id")
    if benchmark != "H2HMEM":
        raise ValueError(f"unsupported benchmark: {benchmark}")

    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(results, start=1):
        variant = str(row.get("variant") or "").strip()
        conversation = str(row.get("conversation_id") or "").strip()
        if not variant or not conversation:
            raise ValueError(f"H2HMEM result row {index} lacks variant/conversation_id")
        item = dict(row)
        item["_metric_sample_id"] = f"{variant}/{conversation}"
        if item.get("answer_failed_attempts") is None:
            if str(item.get("error") or "").strip():
                raise ValueError(
                    f"H2HMEM result row {index} has an error but no failed-attempt count"
                )
            item["answer_failed_attempts"] = 0
        normalized.append(item)
    return calculate_calls_qa(normalized, sample_id_field="_metric_sample_id")


def merge_job(
    benchmark: str,
    method: str,
    *,
    output_root: Path,
    mb_call_root: Path,
) -> bool:
    """Merge one complete measurement; return False when an input is not ready."""
    if method == "HiveMem" or method not in BASELINES:
        raise ValueError(f"only registered non-HiveMem baselines are supported: {method}")
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {benchmark}")

    result_dir = output_root / benchmark / method
    metrics_path = result_dir / "metrics.json"
    results_path = result_dir / "results.json"
    mb_path = mb_call_root / benchmark / method / "metrics.json"
    if not metrics_path.is_file() or not results_path.is_file() or not mb_path.is_file():
        return False

    expected = EXPECTED_SAMPLE_COUNTS[benchmark]
    try:
        memory_bank = normalize_mb_metric(_load_dict(mb_path), expected=expected)
    except ValueError:
        return False
    qa = calculate_formal_qa_metric(benchmark, _load_list(results_path))
    if not qa.get("available") or qa.get("num_samples") != expected:
        raise ValueError(
            f"formal QA calls are incomplete for {benchmark}/{method}: {qa.get('reason')}"
        )

    formal_metrics = _load_dict(metrics_path)
    formal_metrics["calls"] = combine_call_metrics(memory_bank, qa)
    write_json_atomic(metrics_path, formal_metrics)
    return True


def merge_available(*, output_root: Path, mb_call_root: Path) -> tuple[int, list[str]]:
    merged = 0
    missing: list[str] = []
    for benchmark in BENCHMARKS:
        for method in BASELINES:
            name = f"{benchmark}/{method}"
            if merge_job(
                benchmark,
                method,
                output_root=output_root,
                mb_call_root=mb_call_root,
            ):
                merged += 1
            else:
                missing.append(name)
    return merged, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--mb-call-root", default=str(DEFAULT_MB_CALL_ROOT))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=1800)
    args = parser.parse_args()
    if args.poll_seconds < 1:
        parser.error("--poll-seconds must be positive")

    while True:
        merged, missing = merge_available(
            output_root=Path(args.output_root),
            mb_call_root=Path(args.mb_call_root),
        )
        print(
            f"integrated_calls={merged}/21 missing={','.join(missing) or 'none'}",
            flush=True,
        )
        if not args.watch or not missing:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
