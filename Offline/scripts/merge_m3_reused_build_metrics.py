#!/usr/bin/env python3
"""Merge an M3 source run's build accounting into a graph-reuse QA run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmarks.memgallery_harness.runner.metrics import (  # noqa: E402
    _combine_inference_aggregates,
    _cost_from_inference_aggregate,
    _inference_aggregate,
    _latency_from_inference_aggregate,
)


BENCHMARKS = ("H2HMEM", "Mem-Gallery", "WorldMemArena")
METHOD = "M3-Agent-caption"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_aggregate(source_dir: Path, num_samples: int) -> tuple[dict[str, Any], dict[str, Any]]:
    trace_paths = sorted((source_dir / "call_traces").glob("*.jsonl"))
    if len(trace_paths) != num_samples:
        raise ValueError(
            f"Expected {num_samples} source trace files, found {len(trace_paths)}: "
            f"{source_dir / 'call_traces'}"
        )
    rows: list[dict[str, Any]] = []
    for path in trace_paths:
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    build_rows = [row for row in rows if row.get("phase") == "memory_build"]
    if not build_rows:
        raise ValueError(f"No memory_build calls found under {source_dir}")
    failed = [row for row in build_rows if row.get("failed")]
    missing_usage = [
        row
        for row in build_rows
        if row.get("prompt_tokens") is None or row.get("completion_tokens") is None
    ]
    if failed or missing_usage:
        raise ValueError(
            f"Incomplete source build accounting: failed={len(failed)}, "
            f"missing_usage={len(missing_usage)}"
        )
    aggregate = _inference_aggregate(
        num_samples=num_samples,
        source=f"reused_memory_build:{source_dir / 'call_traces'}",
        input_tokens=sum(int(row["prompt_tokens"]) for row in build_rows),
        output_tokens=sum(int(row["completion_tokens"]) for row in build_rows),
        calls=len(build_rows),
        image_count=sum(int(row.get("image_count") or 0) for row in build_rows),
    )
    provenance = {
        "source_call_trace_dir": str((source_dir / "call_traces").resolve()),
        "source_trace_files": {
            path.name: sha256_file(path) for path in trace_paths
        },
        "source_trace_file_count": len(trace_paths),
        "selected_phase": "memory_build",
        "build_calls": len(build_rows),
        "failed_build_calls": len(failed),
    }
    return aggregate, provenance


def call_summary(
    *, calls: int, failed: int, num_samples: int, aggregation: str
) -> dict[str, Any]:
    mean = calls / num_samples
    return {
        "total_calls": calls,
        "failed_calls": failed,
        "successful_calls": calls - failed,
        "num_samples": num_samples,
        "mean_per_sample": mean,
        "formula": f"{calls} / {num_samples} = {mean:.12g}",
        "aggregation": aggregation,
        "available": True,
    }


def merge_benchmark(
    output_root: Path, benchmark: str, source_run_id: str, target_run_id: str
) -> dict[str, Any]:
    source_dir = output_root / benchmark / METHOD / source_run_id
    target_dir = output_root / benchmark / METHOD / target_run_id
    metrics_path = target_dir / "metrics.json"
    efficiency_path = target_dir / "efficiency_metrics.json"
    manifest_path = target_dir / "run_manifest.json"
    for path in (source_dir, target_dir, metrics_path, efficiency_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metrics = load_json(metrics_path)
    efficiency = load_json(efficiency_path)
    manifest = load_json(manifest_path)
    num_samples = int(
        efficiency.get("components", {})
        .get("memory_build", {})
        .get("num_samples", 0)
    )
    if num_samples < 1:
        raise ValueError(f"Invalid sample count in {efficiency_path}")

    reused_states = list((target_dir / "memory" / "datasets").rglob("reused_state.json"))
    if len(reused_states) != num_samples:
        raise ValueError(
            f"Expected {num_samples} reused_state records, found {len(reused_states)}"
        )
    expected_source_root = (source_dir / "memory" / "datasets").resolve()
    for path in reused_states:
        state = load_json(path)
        source_state = Path(str(state.get("source_state_dir") or "")).resolve()
        if expected_source_root not in source_state.parents:
            raise ValueError(
                f"Target state does not reference source run {source_run_id}: {path}"
            )

    memory_build, source_provenance = build_aggregate(source_dir, num_samples)
    components = efficiency["components"]
    retrieval = dict(components["retrieval"])
    answer = dict(components["answer"])
    qa = _combine_inference_aggregates(
        retrieval, answer, source="retrieval_plus_answer"
    )
    total = _combine_inference_aggregates(
        memory_build, qa, source="reused_memory_build_plus_retrieval_plus_answer"
    )
    profile = efficiency["profile"]
    efficiency["components"]["memory_build"] = memory_build
    efficiency["cost_mb"] = _cost_from_inference_aggregate(
        memory_build, profile, aggregation="sum_mb_cost_divided_by_samples"
    )
    efficiency["latency_mb"] = _latency_from_inference_aggregate(
        memory_build, profile, aggregation="sum_mb_latency_divided_by_samples"
    )
    efficiency["cost_total"] = _cost_from_inference_aggregate(
        total, profile, aggregation="sum_total_cost_divided_by_samples"
    )
    efficiency["latency_total"] = _latency_from_inference_aggregate(
        total, profile, aggregation="sum_total_latency_divided_by_samples"
    )

    qa_calls = dict(metrics["calls"]["qa"])
    build_calls = int(memory_build["calls"])
    metrics["calls"]["memory_bank"] = call_summary(
        calls=build_calls,
        failed=0,
        num_samples=num_samples,
        aggregation="source_memory_build_calls_divided_by_samples",
    )
    metrics["calls"]["total"] = call_summary(
        calls=build_calls + int(qa_calls["total_calls"]),
        failed=int(qa_calls["failed_calls"]),
        num_samples=num_samples,
        aggregation="source_build_plus_target_qa_calls_divided_by_samples",
    )
    for key in (
        "cost_mb",
        "cost_qa",
        "cost_total",
        "latency_mb",
        "latency_qa",
        "latency_total",
    ):
        metrics[key] = dict(efficiency[key])

    provenance = {
        "schema_version": 1,
        "method": METHOD,
        "benchmark": benchmark,
        "source_run_id": source_run_id,
        "target_run_id": target_run_id,
        "merge_policy": {
            "memory_build": "source_run_memory_build_call_trace",
            "retrieval_and_answer": "target_reuse_run",
            "judge_excluded": True,
            "embedding_excluded": True,
            "database_excluded": True,
            "wall_clock_excluded": True,
        },
        "source": source_provenance,
        "memory_build_aggregate": memory_build,
    }
    efficiency["reused_memory_build_provenance"] = provenance
    metrics["reused_memory_build_provenance"] = provenance
    manifest["reused_memory_build_metrics"] = provenance

    for path in (metrics_path, efficiency_path):
        backup = path.with_name(path.stem + ".reuse_qa_only.json")
        if not backup.exists():
            shutil.copy2(path, backup)
    write_json_atomic(efficiency_path, efficiency)
    write_json_atomic(metrics_path, metrics)
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(target_dir / "reused_memory_build_metrics.json", provenance)
    return {
        "benchmark": benchmark,
        "samples": num_samples,
        "build_calls": build_calls,
        "cost_mb": efficiency["cost_mb"]["mean_per_sample_usd"],
        "latency_mb": efficiency["latency_mb"]["mean_per_sample_seconds"],
        "cost_total": efficiency["cost_total"]["mean_per_sample_usd"],
        "latency_total": efficiency["latency_total"]["mean_per_sample_seconds"],
        "calls_mb_qa": metrics["calls"]["total"]["mean_per_sample"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--target-run-id", required=True)
    args = parser.parse_args()
    if args.source_run_id == args.target_run_id:
        parser.error("source and target Run-ID must differ")
    summaries = [
        merge_benchmark(
            args.output_root.expanduser().resolve(),
            benchmark,
            args.source_run_id,
            args.target_run_id,
        )
        for benchmark in BENCHMARKS
    ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
