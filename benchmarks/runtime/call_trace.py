"""Read and aggregate persisted LLM call traces for benchmark metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRACE_VERSION = 3


def load_call_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def summarize_call_rows(
    rows: list[dict[str, Any]],
    *,
    phase: str,
    num_samples: int,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("phase") == phase]
    total = len(selected)
    failed = sum(bool(row.get("failed")) for row in selected)
    mean = total / num_samples if num_samples else None
    return {
        "total_calls": total,
        "failed_calls": failed,
        "successful_calls": total - failed,
        "num_samples": num_samples,
        "mean_per_sample": mean,
        "formula": f"{total} / {num_samples} = {mean:.12g}" if num_samples else None,
        "aggregation": f"{phase}_calls_divided_by_samples",
        "available": True,
    }
