#!/usr/bin/env python3
"""Validate persisted MMA artifacts against the strict 0912 contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = OFFLINE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmarks.baseline_runtime.adapters.mma_original import (  # noqa: E402
    UPSTREAM_COMMIT,
    UPSTREAM_TREE,
)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    run_dir: str
    passed: bool = True
    checks: list[Check] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name, passed, detail))
        self.passed = self.passed and passed

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["failed_checks"] = [row.name for row in self.checks if not row.passed]
        return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"missing artifact: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise ValueError(f"missing artifact: {path}") from None
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        rows.append(value)
    return rows


def _runtime(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("baseline_runtime") or manifest.get("baseline") or {}
    return value if isinstance(value, dict) else {}


def _configured_top_k(manifest: dict[str, Any]) -> int:
    value = manifest.get("top_k")
    if value is None:
        value = (manifest.get("configuration") or {}).get("top_k")
    return int(value if value is not None else -1)


def _prompt_hash(manifest: dict[str, Any]) -> str:
    value = manifest.get("prompt_sha256")
    if value is None:
        value = (manifest.get("configuration") or {}).get("prompt_sha256")
    return str(value or "")


def validate(run_dir: Path, expected_qa: int | None = None) -> Report:
    run_dir = run_dir.resolve()
    report = Report(run_dir=str(run_dir))
    manifest = _read_json(run_dir / "run_manifest.json")
    results = _read_json(run_dir / "results.json")
    traces = _read_jsonl(run_dir / "retrieval_trace.jsonl")
    pipeline = _read_jsonl(run_dir / "pipeline_qa.jsonl")
    snapshot = _read_jsonl(run_dir / "memory" / "memory_snapshot.jsonl")
    if not isinstance(results, list):
        raise ValueError("results.json must contain a list")

    runtime = _runtime(manifest)
    report.add(
        "official_runtime",
        runtime.get("adapter") == "mma_original"
        and runtime.get("upstream_commit") == UPSTREAM_COMMIT
        and runtime.get("upstream_tree") == UPSTREAM_TREE,
        f"adapter={runtime.get('adapter')} commit={runtime.get('upstream_commit')} "
        f"tree={runtime.get('upstream_tree')}",
    )
    report.add(
        "global_top7_configuration",
        _configured_top_k(manifest) == 7,
        f"top_k={_configured_top_k(manifest)}",
    )

    conformance = manifest.get("mma_conformance") or {}
    prompt_rows = conformance.get("internal_prompt_sha256") or {}
    prompt_hashes_match = bool(prompt_rows) and all(
        isinstance(value, dict)
        and value.get("expected")
        and value.get("expected") == value.get("actual")
        for value in prompt_rows.values()
    )
    report.add(
        "original_agent_prompt_hashes",
        prompt_hashes_match,
        f"matched={sum(1 for value in prompt_rows.values() if isinstance(value, dict) and value.get('expected') == value.get('actual'))}/{len(prompt_rows)}",
    )
    benchmark_prompt_hash = _prompt_hash(manifest)
    report.add(
        "benchmark_qa_prompt_hash",
        bool(benchmark_prompt_hash)
        and conformance.get("answer_prompt_sha256") == benchmark_prompt_hash,
        f"manifest={benchmark_prompt_hash} conformance={conformance.get('answer_prompt_sha256')}",
    )
    forbidden = conformance.get("forbidden_paths") or {}
    report.add(
        "forbidden_paths_disabled",
        bool(forbidden) and not any(bool(value) for value in forbidden.values()),
        json.dumps(forbidden, sort_keys=True),
    )

    count_ok = len(results) == len(traces) == len(pipeline)
    if expected_qa is not None:
        count_ok = count_ok and len(results) == expected_qa
    report.add(
        "qa_artifact_counts",
        count_ok,
        f"results={len(results)} traces={len(traces)} pipeline={len(pipeline)} expected={expected_qa}",
    )
    report.add(
        "answers_complete",
        bool(results) and all(not str(row.get("error") or "") for row in results),
        f"errors={sum(bool(str(row.get('error') or '')) for row in results)}",
    )

    memory_ids = [str(row.get("memory_id") or "") for row in snapshot]
    structured_snapshot = bool(snapshot) and len(memory_ids) == len(set(memory_ids)) and all(
        memory_id
        and str(row.get("backend_type") or "").startswith("mma_")
        and bool(row.get("source_dialogue_ids"))
        and isinstance(json.loads(str(row.get("text") or "")), dict)
        and bool(json.loads(str(row.get("text") or "")).get("memory_type"))
        for memory_id, row in zip(memory_ids, snapshot)
    )
    report.add(
        "structured_native_memory_snapshot",
        structured_snapshot,
        f"memories={len(snapshot)} unique_ids={len(set(memory_ids))}",
    )

    image_rows = 0
    image_provenance_ok = True
    for row in snapshot:
        image_ids = list(row.get("image_ids") or [])
        image_paths = list(row.get("image_paths") or [])
        if image_ids or image_paths:
            image_rows += 1
            image_provenance_ok = image_provenance_ok and bool(image_ids) and bool(image_paths)
            image_provenance_ok = image_provenance_ok and all(
                Path(str(path)).is_file() for path in image_paths
            )
    report.add(
        "image_provenance",
        image_provenance_ok,
        f"memory_rows_with_images={image_rows}",
    )

    snapshot_ids = set(memory_ids)
    retrieval_ok = True
    final_prompt_ok = True
    native_chat_ok = True
    retrieved_total = 0
    for trace in traces:
        items = trace.get("top_k")
        if not isinstance(items, list) or len(items) > 7:
            retrieval_ok = False
            continue
        ids = [str(item.get("memory_id") or "") for item in items if isinstance(item, dict)]
        retrieved_total += len(ids)
        retrieval_ok = retrieval_ok and len(ids) == len(set(ids))
        retrieval_ok = retrieval_ok and all(memory_id in snapshot_ids for memory_id in ids)
        retrieval_ok = retrieval_ok and all(
            bool(item.get("source_dialogue_ids"))
            and isinstance(json.loads(str(item.get("content") or "")), dict)
            for item in items
            if isinstance(item, dict)
        )
        retrieval_method = trace.get("retrieval_method_trace") or {}
        final_prompt_ok = final_prompt_ok and retrieval_method.get("qa_prompt_applied") is False
    for result in results:
        native = result.get("native_answer_trace") or {}
        native_chat_ok = native_chat_ok and (
            native.get("via") == "mma_original_chat_agent"
            and native.get("qa_prompt_applied_stage") == "final_answer_only"
            and native.get("original_chat_system_prompt") is True
            and native.get("memory_tool_scope") == "original_unmodified_chat_tools"
            and len(native.get("retrieved_memory_ids") or []) <= 7
        )
    report.add(
        "structured_top7_retrieval",
        bool(traces) and retrieval_ok,
        f"queries={len(traces)} retrieved_items={retrieved_total}",
    )
    report.add(
        "qa_prompt_final_only",
        bool(traces) and final_prompt_ok,
        "retrieval traces declare qa_prompt_applied=false",
    )
    report.add(
        "original_chat_agent_answers",
        bool(results) and native_chat_ok,
        "native answer trace requires original Chat Agent and unmodified tools",
    )

    state_root = Path(str(manifest.get("baseline_state_dir") or ""))
    sqlite_files = list(state_root.rglob("sqlite.db")) if state_root.is_dir() else []
    report.add(
        "native_sqlite_artifacts",
        bool(sqlite_files),
        f"state_root={state_root} sqlite_files={len(sqlite_files)}",
    )
    source_root = Path(str(runtime.get("source_root") or ""))
    source_repo = source_root.parent
    clean = False
    detail = f"repo={source_repo}"
    if (source_repo / ".git").is_dir():
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_repo, text=True
        ).strip()
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=source_repo, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_repo, text=True
        ).strip()
        clean = head == UPSTREAM_COMMIT and tree == UPSTREAM_TREE and not dirty
        detail = f"commit={head} tree={tree} dirty={bool(dirty)}"
    report.add("official_checkout_clean", clean, detail)
    report.statistics = {
        "qa": len(results),
        "memories": len(snapshot),
        "memory_rows_with_images": image_rows,
        "retrieved_items": retrieved_total,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-qa", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.run_dir, args.expected_qa)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
