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
NATIVE_MEMORY_PATTERNS = {
    "OmniSimpleMem": ("index/mau_store/*.jsonl",),
    "M2A": ("raw.db",),
    "MIRIX": ("sqlite.db",),
    "MMA": ("sqlite.db",),
    "MemVerse": (
        "memory_chunks/core_memory.json",
        "memory_chunks/episodic_memory.json",
        "memory_chunks/semantic_memory.json",
        "graph/core/vdb_entities.json",
        "graph/episodic/vdb_entities.json",
        "graph/semantic/vdb_entities.json",
    ),
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


def validate_integrated_call_metrics(
    formal_path: Path,
    mb_call_path: Path,
    expected: int,
) -> None:
    formal = _load_dict(formal_path)
    measured = _load_dict(mb_call_path)
    calls = formal.get("calls")
    if not isinstance(calls, dict):
        raise ValueError("formal metrics do not contain integrated call metrics")
    memory_bank = calls.get("memory_bank") or {}
    qa = calls.get("qa") or {}
    total = calls.get("total") or {}
    for key in ("total_calls", "failed_calls", "successful_calls", "num_samples"):
        if memory_bank.get(key) != measured.get(key):
            raise ValueError(
                f"integrated MB-call {key}={memory_bank.get(key)!r}; "
                f"measured={measured.get(key)!r}"
            )
    if not memory_bank.get("available") or memory_bank.get("num_samples") != expected:
        raise ValueError("integrated MB-call metric is incomplete")
    if not qa.get("available") or qa.get("num_samples") != expected:
        raise ValueError("integrated QA-call metric is incomplete")
    if not total.get("available") or total.get("num_samples") != expected:
        raise ValueError("integrated total-call metric is incomplete")
    expected_total = int(memory_bank["total_calls"]) + int(qa["total_calls"])
    if total.get("total_calls") != expected_total:
        raise ValueError(
            f"integrated total calls are {total.get('total_calls')!r}; expected {expected_total}"
        )


def validate_native_memory_artifacts(
    result_dir: Path,
    method: str,
    expected: int,
) -> None:
    """Prove that each sample has a native store or serialized native snapshot."""
    patterns = NATIVE_MEMORY_PATTERNS.get(method)
    if patterns is None:
        # AUGUSTUSMemory and M3-Agent are in-memory implementations. Their
        # unified snapshot is the durable serialization of the native store.
        snapshot = result_dir / "memory" / "memory_snapshot.jsonl"
        if not snapshot.is_file() or snapshot.stat().st_size == 0:
            raise ValueError(f"missing serialized native memory: {snapshot}")
        return

    state_root = result_dir / "memory" / "datasets"
    for pattern in patterns:
        artifacts = [
            path
            for path in state_root.rglob(pattern)
            if path.is_file() and path.stat().st_size > 0
        ]
        if len(artifacts) < expected:
            raise ValueError(
                f"native memory pattern {pattern!r} has {len(artifacts)} "
                f"non-empty artifacts; expected at least {expected}"
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
        "integrated_calls": lambda job: validate_integrated_call_metrics(
            output_root / job.benchmark / job.method / "metrics.json",
            mb_call_root / job.benchmark / job.method / "metrics.json",
            EXPECTED_SAMPLE_COUNTS[job.benchmark],
        ),
        "native_memory": lambda job: validate_native_memory_artifacts(
            output_root / job.benchmark / job.method,
            job.method,
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
