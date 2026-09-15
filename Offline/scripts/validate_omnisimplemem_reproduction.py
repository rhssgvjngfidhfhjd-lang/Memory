#!/usr/bin/env python3
"""Validate an OmniSimpleMem reproduction run and emit an audit report."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_COMMIT = "836ce9718f3e9cb7f93c9d7c842b47f62e177a66"
EXPECTED_TREE = "685109637c4c8b9a2469e695ad3dbed40762c0f2"
EXPECTED_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
EXPECTED_ANSWER_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
EXPECTED_TOP_K = 7


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def validate(result_dir: Path) -> dict[str, Any]:
    result_dir = result_dir.resolve()
    checks: dict[str, bool] = {}
    manifest = _json(result_dir / "run_manifest.json")
    results = _json(result_dir / "results.json")
    traces = _jsonl(result_dir / "retrieval_trace.jsonl")
    snapshots = _jsonl(result_dir / "memory" / "memory_snapshot.jsonl")
    calls = _jsonl(result_dir / "call_trace.jsonl")

    source_root = Path(manifest["baseline_runtime"]["source_root"])
    repository = source_root.parent
    actual_commit = _git("rev-parse", "HEAD", cwd=repository)
    actual_tree = _git("rev-parse", "HEAD:OmniSimpleMem", cwd=repository)
    dirty = _git("status", "--porcelain", cwd=repository)
    checks["official_commit_exact"] = actual_commit == EXPECTED_COMMIT
    checks["official_tree_exact"] = actual_tree == EXPECTED_TREE
    checks["official_checkout_clean"] = not dirty
    checks["manifest_official_core"] = (
        manifest["baseline_runtime"].get("upstream_commit") == EXPECTED_COMMIT
        and manifest["baseline_runtime"].get("upstream_tree") == EXPECTED_TREE
    )
    checks["fixed_top_k_config"] = manifest.get("top_k") == EXPECTED_TOP_K
    checks["embedding_model"] = (
        manifest.get("embedding_model") == EXPECTED_EMBEDDING_MODEL
    )
    checks["executor_and_answer_model"] = (
        manifest.get("executor_model") == EXPECTED_ANSWER_MODEL
        and manifest.get("answer_model") == EXPECTED_ANSWER_MODEL
    )
    checks["benchmark_native_inputs"] = (
        manifest.get("chunk_input", {}).get("shared_fixed_chunks") is False
    )
    conformance = manifest.get("omni_conformance") or {}
    checks["benchmark_prompt_answer_path"] = (
        conformance.get("answer_path") == "benchmark QA prompt"
        and conformance.get("orchestrator_answer_used") is False
    )
    checks["no_force_and_graph_enabled"] = (
        conformance.get("force_ingest") is False
        and conformance.get("adapter_silent_fallback") is False
        and conformance.get("graph_entity_flow_enabled") is True
    )
    checks["modality_hard_filter_disabled"] = (
        conformance.get("modality_hard_filter_disabled") is True
    )

    checks["qa_count_matches"] = (
        len(results) == len(traces) == int(manifest.get("questions") or 0) > 0
    )
    checks["qa_success"] = (
        int(manifest.get("answer_errors") or 0) == 0
        and all(not row.get("error") for row in results)
        and all(str(row.get("answer_raw_response") or "").startswith("<answer>") for row in results)
    )
    checks["prompt_hash_present"] = bool(manifest.get("prompt_sha256"))
    checks["retrieval_count_seven"] = all(
        len(row.get("top_k") or []) == EXPECTED_TOP_K for row in traces
    )
    method_traces = [row.get("retrieval_method_trace") or {} for row in traces]
    checks["official_query_trace"] = all(
        row.get("via") == "OmniMemoryOrchestrator.query"
        and row.get("effective_top_k") == EXPECTED_TOP_K
        and row.get("returned") == EXPECTED_TOP_K
        and row.get("query_embedding_calls") == 1
        for row in method_traces
    )
    checks["official_dynamic_top_k_disabled"] = all(
        row.get("effective_top_k") == EXPECTED_TOP_K for row in method_traces
    )
    checks["modality_filter_policy_recorded"] = all(
        row.get("modality_hard_filter_disabled") is True
        and row.get("effective_modality_filter") is None
        for row in method_traces
    )

    modality_counts = Counter(row.get("backend_type") for row in snapshots)
    checks["text_and_visual_mau_present"] = (
        modality_counts["omni_mau:text"] > 0
        and modality_counts["omni_mau:visual"] > 0
    )
    checks["all_embeddings_2048"] = all(
        row.get("metadata", {}).get("embedding_dim") == 2048 for row in snapshots
    )
    visual_rows = [row for row in snapshots if row.get("backend_type") == "omni_mau:visual"]
    checks["visual_raw_and_provenance_exist"] = all(
        Path(row["metadata"]["native"]["raw_pointer"]).is_file()
        and row.get("image_paths")
        and all(Path(path).is_file() for path in row["image_paths"])
        and row.get("source_dialogue_ids")
        for row in visual_rows
    )
    checks["native_schema_recorded"] = all(
        {
            "id",
            "timestamp",
            "modality_type",
            "summary",
            "raw_pointer",
            "details",
            "status",
            "storage_tier",
            "metadata",
            "links",
            "region_pointers",
        }.issubset(row.get("metadata", {}).get("native", {}))
        for row in snapshots
    )
    checks["provider_calls_recorded"] = bool(calls) and {
        "memory_build",
        "retrieval",
        "qa",
    }.issubset({str(row.get("phase")) for row in calls})
    checks["provider_calls_successful"] = bool(calls) and all(
        bool(row.get("success")) for row in calls
    )
    checks["required_artifacts_present"] = all(
        (result_dir / name).is_file()
        for name in (
            "results.json",
            "retrieval_trace.jsonl",
            "memory/memory_snapshot.jsonl",
            "run_manifest.json",
            "metrics.json",
            "pipeline_qa.jsonl",
            "call_trace.jsonl",
            "efficiency_metrics.json",
        )
    )

    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "result_dir": str(result_dir),
        "checks": checks,
        "summary": {
            "qa_count": len(results),
            "snapshot_count": len(snapshots),
            "modality_counts": dict(modality_counts),
            "retrieval_counts": [len(row.get("top_k") or []) for row in traces],
            "provider_call_counts": dict(
                Counter(str(row.get("phase")) for row in calls)
            ),
            "prompt_sha256": manifest.get("prompt_sha256"),
            "official_commit": actual_commit,
            "official_tree": actual_tree,
        },
        "declared_deviations": {
            "official_benchmark_entry_used": conformance.get(
                "official_benchmark_entry_used"
            ),
            "protocol_bridge": conformance.get("protocol_bridge"),
            "upstream_add_multimodal_used": conformance.get(
                "upstream_add_multimodal_used"
            ),
            "upstream_add_multimodal_reason": conformance.get(
                "upstream_add_multimodal_reason"
            ),
        },
    }
    output = result_dir / "acceptance_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if report["status"] != "pass":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"OmniSimpleMem acceptance failed: {failed}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    report = validate(args.result_dir)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
