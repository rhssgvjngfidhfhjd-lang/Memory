from __future__ import annotations

from collections import defaultdict
from typing import Any

from benchmarks.memgallery_harness.runner.metrics import exact_match, f1_score


def _row_metrics(row: dict[str, Any], k: int) -> dict[str, float]:
    gold_sessions = {
        str(value)
        for value in row.get(
            "gold_visible_sessions", row.get("gold_sessions", [])
        )
    }
    retrieved = [str(value) for value in row.get("retrieved_sessions", [])[:k]]
    hit = float(bool(gold_sessions.intersection(retrieved))) if gold_sessions else 0.0
    return {
        "f1": f1_score(row.get("system_answer", ""), row.get("original_answer", "")),
        "exact_match": exact_match(
            row.get("system_answer", ""), row.get("original_answer", "")
        ),
        "retrieval_hit": hit,
        "error": float(bool(row.get("error"))),
    }


def _aggregate(rows: list[dict[str, float]], k: int) -> dict[str, float | int]:
    count = len(rows)
    return {
        "count": count,
        "f1": sum(row["f1"] for row in rows) / count if count else 0.0,
        "exact_match": sum(row["exact_match"] for row in rows) / count if count else 0.0,
        f"retrieval_hitrate@{k}": (
            sum(row["retrieval_hit"] for row in rows) / count if count else 0.0
        ),
        "errors": int(sum(row["error"] for row in rows)),
    }


def summarize_results(results: list[dict[str, Any]], k: int = 5) -> dict[str, Any]:
    scored = [_row_metrics(row, k) for row in results]
    by_category: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, float]]] = defaultdict(list)
    for raw, row in zip(results, scored):
        by_category[str(raw.get("category", ""))].append(row)
        by_difficulty[str(raw.get("difficulty", ""))].append(row)
    result: dict[str, Any] = _aggregate(scored, k)
    result["future_gold_evidence_questions"] = sum(
        bool(row.get("gold_future_evidence_ids")) for row in results
    )
    result["unmapped_gold_evidence_questions"] = sum(
        bool(row.get("gold_unmapped_evidence_ids")) for row in results
    )
    result["by_category"] = {
        key: _aggregate(value, k) for key, value in sorted(by_category.items())
    }
    result["by_difficulty"] = {
        key: _aggregate(value, k) for key, value in sorted(by_difficulty.items())
    }
    return result
