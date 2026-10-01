from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "baseline_call_contracts.json"
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_call_contract(
    baseline: str, path: Path = DEFAULT_CONTRACT_PATH
) -> dict[str, Any]:
    payload = _json(path)
    try:
        return dict(payload["baselines"][baseline])
    except KeyError as exc:
        raise ValueError(f"No call-flow contract for baseline {baseline}") from exc


def audit_baseline_call_flow(
    run_dir: Path,
    *,
    baseline: str,
    benchmark: str,
    expected_qa: int | None = None,
    expected_top_k: int | None = None,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
) -> dict[str, Any]:
    """Join results, retrieval evidence, and HTTP traces into one gate report."""
    run_dir = Path(run_dir)
    contract = load_call_contract(baseline, contract_path)
    errors: list[str] = []
    required = {
        name: run_dir / name
        for name in (
            "run_manifest.json",
            "results.json",
            "retrieval_trace.jsonl",
            "call_trace.jsonl",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return {
            "passed": False,
            "baseline": baseline,
            "benchmark": benchmark,
            "run_dir": str(run_dir),
            "errors": [f"missing required artifact: {name}" for name in missing],
        }

    manifest = _json(required["run_manifest.json"])
    results = _json(required["results.json"])
    retrieval_rows = _jsonl(required["retrieval_trace.jsonl"])
    calls = _jsonl(required["call_trace.jsonl"])
    if not isinstance(results, list):
        errors.append("results.json is not a list")
        results = []

    count = expected_qa
    if count is None:
        count = int(contract.get("expected_qa_counts", {}).get(benchmark, 0)) or None
    if count is not None:
        if len(results) != count:
            errors.append(f"result count {len(results)} != expected {count}")
        if len(retrieval_rows) != count:
            errors.append(
                f"retrieval trace count {len(retrieval_rows)} != expected {count}"
            )

    top_k = int(expected_top_k or contract.get("top_k") or 0)
    manifest_top_k = manifest.get("top_k")
    if top_k and manifest_top_k is not None and int(manifest_top_k) != top_k:
        errors.append(f"manifest top_k {manifest_top_k} != expected {top_k}")

    result_ids: set[str] = set()
    for index, row in enumerate(results):
        query_id = str(row.get("query_id") or "")
        if not query_id:
            errors.append(f"result[{index}] has no query_id")
            continue
        if query_id in result_ids:
            errors.append(f"duplicate result query_id: {query_id}")
        result_ids.add(query_id)
        if contract.get("require_answer_success") and (
            row.get("error") or not str(row.get("system_answer") or "").strip()
        ):
            errors.append(f"logical answer failure: {query_id}")
        if contract.get("require_single_attempt") and (
            int(row.get("answer_attempts") or 0) != 1
            or int(row.get("answer_failed_attempts") or 0) != 0
        ):
            errors.append(f"answer was not single-attempt: {query_id}")
        native = dict(row.get("native_answer_trace") or {})
        if baseline == "MIRIX" and (
            native.get("single_agent_lifecycle") is not True
            or int(native.get("native_agent_lifecycle_count") or 0) != 1
            or native.get("provisional_native_answer_ignored") is not False
        ):
            errors.append(f"invalid MIRIX native lifecycle trace: {query_id}")

    retrieval_ids: set[str] = set()
    for index, row in enumerate(retrieval_rows):
        query_id = str(row.get("query_id") or "")
        if query_id:
            retrieval_ids.add(query_id)
        evidence = row.get("top_k")
        if not isinstance(evidence, list) or (top_k and len(evidence) > top_k):
            errors.append(f"invalid Top-{top_k} evidence at retrieval row {index}")
        method = dict(row.get("retrieval_method_trace") or {})
        if baseline == "MIRIX" and (
            method.get("stage") != "answer_complete"
            or method.get("automatic_prefetch_counts_toward_top_k") is not True
            or (top_k and int(method.get("requested_top_k") or 0) != top_k)
        ):
            errors.append(f"invalid MIRIX retrieval trace: {query_id or index}")
    if retrieval_ids != result_ids:
        errors.append("results and retrieval traces have different query_id sets")

    qa_counts: dict[str, int] = {}
    retrieval_call_count = 0
    unowned_qa_calls = 0
    build_fault_count = 0
    for row in calls:
        phase = str(row.get("phase") or "")
        if phase == "build_fault":
            build_fault_count += 1
        if phase == "retrieval":
            retrieval_call_count += 1
        if phase != "qa":
            continue
        query_id = str(row.get("query_id") or "")
        if not query_id:
            unowned_qa_calls += 1
            continue
        qa_counts[query_id] = qa_counts.get(query_id, 0) + 1
    if not contract.get("api_calls_allowed_in_retrieval", True) and retrieval_call_count:
        errors.append(f"retrieval phase made {retrieval_call_count} API calls")
    if unowned_qa_calls:
        errors.append(f"{unowned_qa_calls} QA API calls have no query_id")
    missing_calls = sorted(result_ids - qa_counts.keys())
    unknown_calls = sorted(qa_counts.keys() - result_ids)
    if missing_calls:
        errors.append(f"{len(missing_calls)} answers have no owned QA API call")
    if unknown_calls:
        errors.append(f"{len(unknown_calls)} QA call owners are absent from results")
    if contract.get("require_zero_build_faults") and build_fault_count:
        errors.append(f"build phase recorded {build_fault_count} skipped fault(s)")

    return {
        "passed": not errors,
        "baseline": baseline,
        "benchmark": benchmark,
        "run_dir": str(run_dir),
        "contract": contract,
        "counts": {
            "results": len(results),
            "retrieval_traces": len(retrieval_rows),
            "api_calls": len(calls),
            "qa_calls": sum(qa_counts.values()),
            "retrieval_calls": retrieval_call_count,
            "unowned_qa_calls": unowned_qa_calls,
            "build_faults": build_fault_count,
        },
        "errors": errors,
    }
