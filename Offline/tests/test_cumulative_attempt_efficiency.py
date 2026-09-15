from __future__ import annotations

import json
from pathlib import Path

from benchmarks.memgallery_harness.runner.metrics import write_efficiency_metrics


def test_cumulative_retry_usage_covers_failed_attempt_placeholder(tmp_path: Path):
    rows = [
        {
            "phase": "memory_build",
            "sample_id": "sample",
            "query_id": "build-1",
            "call_id": "build-1",
            "success": True,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "image_count": 0,
        },
        {
            "phase": "qa",
            "sample_id": "sample",
            "query_id": "question-1",
            "call_id": "qa:question-1:1",
            "attempt": 1,
            "success": False,
            "image_count": 1,
        },
        {
            "phase": "qa",
            "sample_id": "sample",
            "query_id": "question-1",
            "call_id": "qa:question-1:2",
            "attempt": 2,
            "success": True,
            "prompt_tokens": 30,
            "completion_tokens": 4,
            "total_tokens": 34,
            "image_count": 1,
            "usage_scope": "cumulative_query_attempts",
        },
    ]
    (tmp_path / "call_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    metrics = write_efficiency_metrics(
        tmp_path,
        [],
        sample_id_field="dataset",
        sample_ids=["sample"],
        model="Qwen/Qwen3-VL-4B-Instruct",
        config_path=Path(__file__).parents[1] / "configs" / "model_efficiency.json",
    )

    assert metrics["cost_qa"]["available"] is True
    assert metrics["components"]["answer"]["calls"] == 2
    assert metrics["components"]["answer"]["input_tokens"] == 30
    assert metrics["components"]["answer"]["output_tokens"] == 4
    assert metrics["components"]["answer"]["image_count"] == 2
