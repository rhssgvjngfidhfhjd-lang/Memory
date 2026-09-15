#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = OFFLINE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarks.baseline_runtime.adapters.m3_agent import (  # noqa: E402
    M3_PROMPT_SHA256,
)
from benchmarks.memgallery_harness.runner.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    prompt_sha256,
)


FEATURE_TAG = re.compile(r"<(?:face|voice|character)_[^<>]+>", re.IGNORECASE)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate(
    result_dir: Path,
    *,
    dataset: str,
    expected_clips: int,
    expected_qas: int,
) -> dict[str, Any]:
    state_dir = result_dir / "state" / dataset
    required = {
        "run_manifest": result_dir / "run_manifest.json",
        "results": result_dir / "results.json",
        "retrieval_trace": result_dir / "retrieval_trace.jsonl",
        "memory_snapshot": result_dir / "memory" / "memory_snapshot.jsonl",
        "memory_graph": state_dir / "memory_graph.pkl",
        "clip_manifest": state_dir / "clip_manifest.jsonl",
        "execution_trace": state_dir / "m3_execution_trace.jsonl",
        "conformance": state_dir / "m3_conformance.json",
        "memory_generation": state_dir / "memory_generation",
    }
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    missing = [name for name, path in required.items() if not path.exists()]
    check("required_artifacts", not missing, {"missing": missing})
    if missing:
        return _report(result_dir, dataset, checks, {})

    manifest = _json(required["run_manifest"])
    conformance = _json(required["conformance"])
    results = _json(required["results"])
    retrieval_traces = _jsonl(required["retrieval_trace"])
    clips = _jsonl(required["clip_manifest"])
    execution = _jsonl(required["execution_trace"])
    snapshots = _jsonl(required["memory_snapshot"])
    generation_paths = sorted(required["memory_generation"].glob("clip_*.json"))
    generations = [_json(path) for path in generation_paths]

    check(
        "dataset_cardinality",
        len(clips) == len(generations) == expected_clips,
        {"clips": len(clips), "generations": len(generations), "expected": expected_clips},
    )
    check(
        "qa_cardinality",
        len(results) == len(retrieval_traces) == expected_qas,
        {"results": len(results), "traces": len(retrieval_traces), "expected": expected_qas},
    )
    check(
        "qa_success",
        all(not str(row.get("error") or "") for row in results),
        {"errors": sum(bool(str(row.get("error") or "")) for row in results)},
    )

    internal = manifest.get("m3_conformance", {}).get("internal_prompt_sha256", {})
    internal_mismatches = {
        name: row
        for name, row in internal.items()
        if row.get("expected") != row.get("actual")
    }
    check(
        "official_internal_prompt_hashes",
        set(internal) == set(M3_PROMPT_SHA256) and not internal_mismatches,
        {"hashes": internal, "mismatches": internal_mismatches},
    )
    actual_qa_hash = manifest.get("m3_conformance", {}).get("answer_prompt_sha256")
    check(
        "benchmark_qa_prompt_hash",
        actual_qa_hash == prompt_sha256(),
        {"expected": prompt_sha256(), "actual": actual_qa_hash},
    )
    check(
        "approved_source_patch_recorded",
        bool(conformance.get("official_core_patched"))
        and all(
            row.get("approved_by_user")
            for row in conformance.get("approved_source_patches", [])
        ),
        conformance.get("approved_source_patches", []),
    )

    prompt_failures = [
        row.get("clip_id")
        for row in generations
        if row.get("prompt_name") != "prompt_generate_memory_with_ids_sft"
        or row.get("prompt_sha256")
        != M3_PROMPT_SHA256["prompt_generate_memory_with_ids_sft"]
    ]
    schema_failures = [
        row.get("clip_id")
        for row in generations
        if not isinstance(row.get("episodic_memory"), list)
        or not isinstance(row.get("semantic_memory"), list)
    ]
    check("memorization_prompt_per_clip", not prompt_failures, prompt_failures)
    check("memory_schema_per_clip", not schema_failures, schema_failures)

    accepted_feature_tags: list[dict[str, Any]] = []
    rejected_count = 0
    empty_counts = {"clip": 0, "episodic": 0, "semantic": 0}
    for row in generations:
        episodic = row.get("episodic_memory") or []
        semantic = row.get("semantic_memory") or []
        empty_counts["clip"] += int(not episodic and not semantic)
        empty_counts["episodic"] += int(not episodic)
        empty_counts["semantic"] += int(not semantic)
        rejected_count += len(row.get("rejected_unsupported_feature_memories") or [])
        for memory_type, values in (("episodic", episodic), ("semantic", semantic)):
            for text in values:
                if FEATURE_TAG.search(str(text)):
                    accepted_feature_tags.append(
                        {
                            "clip_id": row.get("clip_id"),
                            "type": memory_type,
                            "text": text,
                        }
                    )
    check(
        "no_unsupported_feature_ids_in_graph_input",
        not accepted_feature_tags,
        {"violations": accepted_feature_tags[:20], "rejected_count": rejected_count},
    )

    node_counts = conformance.get("node_counts", {})
    check(
        "episodic_semantic_graph",
        int(node_counts.get("episodic", 0)) > 0
        and int(node_counts.get("semantic", 0)) > 0,
        node_counts,
    )
    check(
        "no_face_voice_nodes",
        int(node_counts.get("img", 0)) == 0 and int(node_counts.get("voice", 0)) == 0,
        node_counts,
    )
    check(
        "graph_persisted",
        required["memory_graph"].stat().st_size > 0,
        {"bytes": required["memory_graph"].stat().st_size},
    )

    clip_dialogue_ids = {str(row.get("dialogue_id") or "") for row in clips}
    invalid_provenance: list[dict[str, Any]] = []
    invalid_images: list[dict[str, Any]] = []
    retrieval_counts: list[int] = []
    round_counts: list[int] = []
    search_top_k: Counter[int] = Counter()
    top_up_count = 0
    final_prompt_failures: list[str] = []
    for trace in retrieval_traces:
        query_id = str(trace.get("query_id") or "")
        items = trace.get("top_k") or []
        retrieval_counts.append(len(items))
        method = trace.get("retrieval_method_trace") or {}
        rounds = method.get("rounds") or []
        round_counts.append(len(rounds))
        top_up_count += int(bool(method.get("handoff_top_up_used")))
        for round_row in rounds:
            if round_row.get("action") == "Search":
                search_top_k[int(round_row.get("native_search_top_k") or 0)] += 1
        messages = trace.get("answer_prompt_messages") or []
        if (
            len(messages) != 2
            or messages[0].get("role") != "system"
            or messages[0].get("content") != SYSTEM_PROMPT
            or messages[1].get("role") != "user"
        ):
            final_prompt_failures.append(query_id)
        for item in items:
            source_ids = [str(value) for value in item.get("source_dialogue_ids") or []]
            if not source_ids or any(value not in clip_dialogue_ids for value in source_ids):
                invalid_provenance.append(
                    {"query_id": query_id, "memory_id": item.get("memory_id")}
                )
            image_ids = item.get("image_ids") or []
            image_paths = item.get("image_paths") or []
            if len(image_ids) != len(image_paths) or any(
                not Path(path).is_file() for path in image_paths
            ):
                invalid_images.append(
                    {"query_id": query_id, "memory_id": item.get("memory_id")}
                )

    check(
        "control_round_limit",
        bool(round_counts) and min(round_counts) >= 1 and max(round_counts) <= 5,
        Counter(round_counts),
    )
    check(
        "native_search_top_k_2",
        bool(search_top_k) and set(search_top_k) == {2},
        dict(search_top_k),
    )
    check(
        "handoff_at_most_7_without_top_up",
        bool(retrieval_counts)
        and min(retrieval_counts) >= 1
        and max(retrieval_counts) <= 7
        and top_up_count == 0,
        {"counts": dict(Counter(retrieval_counts)), "top_up_count": top_up_count},
    )
    check("retrieval_provenance", not invalid_provenance, invalid_provenance[:20])
    check("retrieval_images", not invalid_images, invalid_images[:20])
    check("final_answer_message_shape", not final_prompt_failures, final_prompt_failures)

    memory_events = [
        row
        for row in execution
        if row.get("component") == "M3Memorization"
        and row.get("action") == "generate_and_insert"
    ]
    control_events = [
        row
        for row in execution
        if row.get("component") == "M3Control" and row.get("action") == "retrieve"
    ]
    check(
        "execution_trace_coverage",
        len(memory_events) == expected_clips and len(control_events) == expected_qas,
        {"memory_events": len(memory_events), "control_events": len(control_events)},
    )
    check(
        "snapshot_provenance",
        bool(snapshots)
        and all(row.get("source_dialogue_ids") for row in snapshots),
        {"snapshot_count": len(snapshots)},
    )

    summary = {
        "models": {
            "executor": manifest.get("executor_model"),
            "answer": manifest.get("answer_model"),
            "embedding": manifest.get("embedding_model"),
        },
        "clip_count": len(clips),
        "qa_count": len(results),
        "node_counts": node_counts,
        "empty_memory_counts": empty_counts,
        "rejected_unsupported_feature_memories": rejected_count,
        "retrieval_count_distribution": dict(Counter(retrieval_counts)),
        "round_count_distribution": dict(Counter(round_counts)),
        "native_search_top_k_distribution": dict(search_top_k),
    }
    return _report(result_dir, dataset, checks, summary)


def _report(
    result_dir: Path,
    dataset: str,
    checks: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    failures = [row["name"] for row in checks if not row["passed"]]
    return {
        "schema_version": 1,
        "baseline": "M3-Agent-caption",
        "benchmark": "Mem-Gallery",
        "dataset": dataset,
        "result_dir": str(result_dir.resolve()),
        "status": "pass" if not failures else "fail",
        "failed_checks": failures,
        "checks": checks,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-clips", type=int, required=True)
    parser.add_argument("--expected-qas", type=int, required=True)
    args = parser.parse_args()
    report = validate(
        args.result_dir.resolve(),
        dataset=args.dataset,
        expected_clips=args.expected_clips,
        expected_qas=args.expected_qas,
    )
    output = args.result_dir.resolve() / "m3_acceptance_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
