#!/usr/bin/env python3
"""Verify formal, answer-metric, judge, and MB-call artifacts for 21 baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.run_full_baseline_matrix import (
        BASELINES,
        BENCHMARKS,
        EXPECTED_RESULT_COUNTS,
        Job,
        OUTPUT_ROOT,
        validate_job_outputs,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not its parent.
    from run_full_baseline_matrix import (
        BASELINES,
        BENCHMARKS,
        EXPECTED_RESULT_COUNTS,
        Job,
        OUTPUT_ROOT,
        validate_job_outputs,
    )


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


def _score(value: Any, label: str) -> float:
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{label} is outside [0, 1]: {value!r}")
    return score


def validate_answer_metrics(path: Path, expected: int) -> None:
    metrics = _load_dict(path)
    if metrics.get("count") != expected:
        raise ValueError(
            f"answer metric count is {metrics.get('count')!r}; expected {expected}"
        )
    _score(metrics.get("f1"), "f1")
    _score(metrics.get("em"), "em")


def validate_judge_metrics(path: Path, expected: int) -> None:
    metrics = _load_dict(path)
    mismatched = {
        "count": metrics.get("count"),
        "valid_count": metrics.get("valid_count"),
        "judge_errors": metrics.get("judge_errors"),
        "provisional": metrics.get("provisional"),
    }
    if (
        mismatched["count"] != expected
        or mismatched["valid_count"] != expected
        or int(mismatched["judge_errors"] or 0) != 0
        or bool(mismatched["provisional"])
    ):
        raise ValueError(f"incomplete judge metrics: {mismatched}")
    _score(metrics.get("accuracy"), "judge accuracy")


def validate_mb_call_metrics(path: Path, expected: int) -> None:
    metrics = _load_dict(path)
    if not metrics.get("available"):
        raise ValueError("MB-call metrics are not marked available")
    if metrics.get("num_samples") != expected:
        raise ValueError(
            f"MB-call sample count is {metrics.get('num_samples')!r}; expected {expected}"
        )
    if metrics.get("completed_samples") != expected or metrics.get("failed_samples") != 0:
        raise ValueError(
            "MB-call sample completion mismatch: "
            f"completed={metrics.get('completed_samples')!r} "
            f"failed={metrics.get('failed_samples')!r}"
        )
    total = int(metrics.get("total_calls"))
    failed = int(metrics.get("failed_calls"))
    successful = int(metrics.get("successful_calls"))
    if total < 0 or failed < 0 or successful < 0 or failed + successful != total:
        raise ValueError(
            f"invalid MB-call totals: total={total} failed={failed} successful={successful}"
        )
    mean = float(metrics.get("mean_per_sample"))
    if not math.isclose(mean, total / expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"MB-call mean {mean!r} does not equal {total}/{expected}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--mb-call-root", default=str(DEFAULT_MB_CALL_ROOT))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    mb_call_root = Path(args.mb_call_root)
    checks: dict[str, Callable[[Job], None]] = {
        "formal": lambda job: validate_job_outputs(
            job, output_root / job.benchmark / job.method
        ),
        "metrics": lambda job: validate_answer_metrics(
            output_root / job.benchmark / job.method / "metrics.json",
            EXPECTED_RESULT_COUNTS[job.benchmark],
        ),
        "judge": lambda job: validate_judge_metrics(
            output_root / job.benchmark / job.method / "llm_judge_metrics.json",
            EXPECTED_RESULT_COUNTS[job.benchmark],
        ),
        "mb_calls": lambda job: validate_mb_call_metrics(
            mb_call_root / job.benchmark / job.method / "metrics.json",
            EXPECTED_SAMPLE_COUNTS[job.benchmark],
        ),
    }
    jobs = [
        Job(
            f"{method.lower().replace('-', '_')}_{benchmark.lower().replace('-', '_')}",
            benchmark,
            method,
        )
        for benchmark in BENCHMARKS
        for method in BASELINES
    ]
    passed = {name: [] for name in checks}
    failures: list[tuple[str, str, str]] = []
    for job in jobs:
        for name, check in checks.items():
            try:
                check(job)
            except Exception as exc:
                failures.append((name, job.name, f"{type(exc).__name__}: {exc}"))
            else:
                passed[name].append(job.name)

    print(" ".join(f"{name}={len(values)}/21" for name, values in passed.items()))
    for name, job, error in failures:
        print("INCOMPLETE", name, job, error)
    if failures and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
