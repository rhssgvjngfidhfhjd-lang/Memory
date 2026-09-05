#!/usr/bin/env python3
"""Verify formal, answer-metric, judge, and MB-call artifacts for 21 baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

try:
    from scripts.run_full_baseline_matrix import (
        BASELINES,
        BENCHMARKS,
        EMBEDDING_ENDPOINT,
        EMBEDDING_MODEL,
        EXPECTED_RESULT_COUNTS,
        INFERENCE_ENDPOINTS,
        Job,
        MODEL,
        OUTPUT_ROOT,
        validate_job_outputs,
    )
except ModuleNotFoundError:  # Direct execution adds scripts/, not its parent.
    from run_full_baseline_matrix import (
        BASELINES,
        BENCHMARKS,
        EMBEDDING_ENDPOINT,
        EMBEDDING_MODEL,
        EXPECTED_RESULT_COUNTS,
        INFERENCE_ENDPOINTS,
        Job,
        MODEL,
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
JUDGE_MODEL = "openai/gpt-4o-mini"
JUDGE_BASE_URL = "https://openrouter.ai/api/v1"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_judge_metrics(path: Path, expected: int, *, deep: bool = False) -> None:
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
    if not deep:
        return
    if metrics.get("model") != JUDGE_MODEL:
        raise ValueError(f"unexpected Judge model: {metrics.get('model')!r}")

    result_dir = path.parent
    source_path = result_dir / "results.json"
    raw_path = result_dir / "llm_judge_results.json"
    checkpoint_path = result_dir / "llm_judge_checkpoint.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    checkpoint = _load_dict(checkpoint_path)
    if not isinstance(source, list) or len(source) != expected:
        raise ValueError("Judge source result count mismatch")
    if not isinstance(raw, list) or len(raw) != expected:
        raise ValueError("raw Judge result count mismatch")
    signature = checkpoint.get("signature") or {}
    judge_config = signature.get("judge") or {}
    if (
        checkpoint.get("completed") != expected
        or checkpoint.get("expected") != expected
        or signature.get("results_sha256") != _sha256_file(source_path)
        or judge_config.get("model") != JUDGE_MODEL
        or judge_config.get("base_url") != JUDGE_BASE_URL
        or float(judge_config.get("temperature")) != 0.0
    ):
        raise ValueError("Judge checkpoint does not match source results or configuration")

    indexed = {int(row.get("index") or 0): row for row in raw}
    if set(indexed) != set(range(1, expected + 1)):
        raise ValueError("raw Judge indices are missing or duplicated")
    score_sum = 0.0
    correct = 0
    for index, source_row in enumerate(source, start=1):
        row = indexed[index]
        score = _score(row.get("score"), f"Judge row {index} score")
        score_sum += score
        correct += int(score == 1.0)
        judge = row.get("judge") or {}
        if (
            row.get("label") == "judge_error"
            or judge.get("status") != "complete"
            or judge.get("model") != JUDGE_MODEL
            or str(row.get("question") or "") != str(source_row.get("question") or "")
            or str(row.get("category") or "") != str(source_row.get("category") or "")
            or str(row.get("prediction") or "")
            != str(source_row.get("system_answer") or "")
        ):
            raise ValueError(f"raw Judge row {index} does not match its source result")
    average = score_sum / expected
    if (
        int(metrics.get("correct")) != correct
        or not math.isclose(float(metrics.get("score_sum")), score_sum, abs_tol=1e-12)
        or not math.isclose(float(metrics.get("average_score")), average, abs_tol=1e-12)
        or not math.isclose(float(metrics.get("accuracy")), average, abs_tol=1e-12)
    ):
        raise ValueError("Judge aggregate does not match raw per-question scores")


def validate_mb_call_metrics(
    path: Path,
    expected: int,
    *,
    benchmark: str = "",
    method: str = "",
) -> None:
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
    if not benchmark and not method:
        return
    if metrics.get("benchmark") != benchmark or metrics.get("baseline") != method:
        raise ValueError("MB-call artifact benchmark/baseline identity mismatch")

    config_path = path.parent / "run_config.json"
    config = _load_dict(config_path)
    expected_config = {
        "benchmark": benchmark,
        "baseline": method,
        "executor_model": MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_base_url": EMBEDDING_ENDPOINT,
        "embedding_dim": 2048,
        "request_timeout": 180,
        "retries": 2,
        "max_samples": 0,
        "max_chunks": 0,
    }
    mismatched = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatched:
        raise ValueError(f"MB-call run configuration mismatch: {mismatched}")
    if str(config.get("executor_base_url") or "") not in INFERENCE_ENDPOINTS:
        raise ValueError("MB-call run used an unexpected executor endpoint")
    if benchmark == "WorldMemArena" and Path(str(config.get("data_dir") or "")).name != "lifelong":
        raise ValueError("MB-call WMA input is not the lifelong split")

    samples = metrics.get("samples")
    if not isinstance(samples, list) or len(samples) != expected:
        raise ValueError("MB-call metrics do not contain every sample record")
    sample_ids = [str(row.get("sample_id") or "") for row in samples]
    configured_ids = [str(value) for value in config.get("samples") or []]
    if (
        "" in sample_ids
        or len(set(sample_ids)) != expected
        or len(configured_ids) != expected
        or set(configured_ids) != set(sample_ids)
    ):
        raise ValueError("MB-call sample identities are missing, duplicated, or mismatched")

    sample_total = 0
    sample_failed = 0
    for row in samples:
        if (
            row.get("status") != "completed"
            or row.get("benchmark") != benchmark
            or row.get("baseline") != method
        ):
            raise ValueError(f"incomplete or misidentified MB-call sample: {row.get('sample_id')}")
        row_total = int(row.get("total_calls"))
        row_failed = int(row.get("failed_calls"))
        if row_total < 0 or row_failed < 0 or row_failed > row_total:
            raise ValueError(f"invalid per-sample call totals: {row.get('sample_id')}")
        sample_total += row_total
        sample_failed += row_failed
        recorded_trace = Path(str(row.get("trace_path") or ""))
        trace_path = (
            recorded_trace
            if recorded_trace.is_file()
            else path.parent / "traces" / recorded_trace.name
        )
        if row_total == 0 and not trace_path.exists():
            continue
        if not trace_path.is_file():
            raise ValueError(f"missing raw MB-call trace: {trace_path}")
        trace_count = 0
        trace_failed = 0
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            trace = json.loads(line)
            trace_count += 1
            trace_failed += int(bool(trace.get("failed")))
            if str(trace.get("sample_id") or "") != str(row.get("sample_id")):
                raise ValueError(f"raw trace sample identity mismatch: {trace_path}")
        if trace_count != row_total or trace_failed != row_failed:
            raise ValueError(
                f"raw trace totals differ for {row.get('sample_id')}: "
                f"trace={trace_count}/{trace_failed} metric={row_total}/{row_failed}"
            )
    if sample_total != total or sample_failed != failed:
        raise ValueError(
            f"per-sample MB-call sums differ: samples={sample_total}/{sample_failed} "
            f"aggregate={total}/{failed}"
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


def validate_unified_metrics(result_dir: Path, expected: int) -> None:
    metrics = _load_dict(result_dir / "metrics.json")
    judge = _load_dict(result_dir / "llm_judge_metrics.json")
    summary = _load_dict(result_dir / "summary.json")
    if metrics.get("count") != expected:
        raise ValueError("canonical metrics have the wrong result count")
    if not math.isclose(
        float(metrics.get("llm_judge")),
        float(judge.get("accuracy")),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("canonical metrics do not match the Judge artifact")
    for key in ("f1", "em", "llm_judge", "count", "calls"):
        if summary.get(key) != metrics.get(key):
            raise ValueError(f"summary and canonical metrics differ at {key!r}")


def validate_wma_prefix_safety(result_dir: Path, expected: int) -> None:
    """Ensure every retrieved WMA source was visible at its QA checkpoint."""
    trace_path = result_dir / "retrieval_trace.jsonl"
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(traces) != expected:
        raise ValueError(f"WMA prefix audit found {len(traces)} traces; expected {expected}")
    violations: list[str] = []
    for row_index, row in enumerate(traces, start=1):
        visible = {str(value) for value in row.get("visible_sessions") or []}
        covered = {str(value) for value in row.get("covered_sessions") or []}
        if not visible or not covered.issubset(visible):
            violations.append(f"row {row_index}: invalid visible/covered sessions")
            continue
        for hit in row.get("top_k") or []:
            session_id = str(hit.get("session_id") or "")
            if session_id and session_id not in visible:
                violations.append(
                    f"row {row_index}: retrieved future session {session_id}"
                )
            for source_id in hit.get("source_dialogue_ids") or []:
                match = re.match(r"^(S\d+):", str(source_id))
                if match and match.group(1) not in visible:
                    violations.append(
                        f"row {row_index}: retrieved future source {source_id}"
                    )
        if len(violations) >= 10:
            break
    if violations:
        raise ValueError("; ".join(violations))


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
            deep=True,
        ),
        "mb_calls": lambda job: validate_mb_call_metrics(
            mb_call_root / job.benchmark / job.method / "metrics.json",
            EXPECTED_SAMPLE_COUNTS[job.benchmark],
            benchmark=job.benchmark,
            method=job.method,
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
        "unified_metrics": lambda job: validate_unified_metrics(
            output_root / job.benchmark / job.method,
            EXPECTED_RESULT_COUNTS[job.benchmark],
        ),
        "prefix_safety": lambda job: (
            validate_wma_prefix_safety(
                output_root / job.benchmark / job.method,
                EXPECTED_RESULT_COUNTS[job.benchmark],
            )
            if job.benchmark == "WorldMemArena"
            else None
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
