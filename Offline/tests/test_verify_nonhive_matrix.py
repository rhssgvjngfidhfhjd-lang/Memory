import json

import pytest

from scripts.verify_nonhive_matrix import (
    validate_answer_metrics,
    validate_integrated_call_metrics,
    validate_judge_metrics,
    validate_mb_call_metrics,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_answer_and_judge_metric_validation(tmp_path):
    answer = tmp_path / "metrics.json"
    _write(answer, {"count": 3, "f1": 0.25, "em": 0.5})
    validate_answer_metrics(answer, 3)

    judge = tmp_path / "judge.json"
    _write(
        judge,
        {
            "count": 3,
            "valid_count": 3,
            "judge_errors": 0,
            "provisional": False,
            "accuracy": 2 / 3,
        },
    )
    validate_judge_metrics(judge, 3)

    _write(answer, {"count": 2, "f1": 0.25, "em": 0.5})
    with pytest.raises(ValueError, match="expected 3"):
        validate_answer_metrics(answer, 3)


def test_mb_call_metric_validation(tmp_path):
    path = tmp_path / "metrics.json"
    payload = {
        "available": True,
        "num_samples": 2,
        "completed_samples": 2,
        "failed_samples": 0,
        "total_calls": 5,
        "failed_calls": 1,
        "successful_calls": 4,
        "mean_per_sample": 2.5,
    }
    _write(path, payload)
    validate_mb_call_metrics(path, 2)

    payload["successful_calls"] = 5
    _write(path, payload)
    with pytest.raises(ValueError, match="invalid MB-call totals"):
        validate_mb_call_metrics(path, 2)


def test_integrated_call_metric_validation(tmp_path):
    measured = tmp_path / "measured.json"
    formal = tmp_path / "formal.json"
    _write(
        measured,
        {
            "total_calls": 5,
            "failed_calls": 1,
            "successful_calls": 4,
            "num_samples": 2,
        },
    )
    _write(
        formal,
        {
            "calls": {
                "memory_bank": {
                    "available": True,
                    "total_calls": 5,
                    "failed_calls": 1,
                    "successful_calls": 4,
                    "num_samples": 2,
                },
                "qa": {"available": True, "total_calls": 3, "num_samples": 2},
                "total": {"available": True, "total_calls": 8, "num_samples": 2},
            }
        },
    )
    validate_integrated_call_metrics(formal, measured, 2)
    payload = json.loads(formal.read_text())
    payload["calls"]["memory_bank"]["total_calls"] = 6
    _write(formal, payload)
    with pytest.raises(ValueError, match="measured=5"):
        validate_integrated_call_metrics(formal, measured, 2)
