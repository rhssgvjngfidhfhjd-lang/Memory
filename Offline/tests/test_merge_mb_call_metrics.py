import json

import pytest

from scripts.merge_mb_call_metrics import (
    calculate_formal_qa_metric,
    merge_job,
    normalize_mb_metric,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _mb_payload(samples=2, calls=5, failed=1):
    return {
        "available": True,
        "num_samples": samples,
        "completed_samples": samples,
        "failed_samples": 0,
        "total_calls": calls,
        "failed_calls": failed,
        "successful_calls": calls - failed,
    }


def test_normalize_mb_metric_rejects_incomplete_artifact():
    payload = _mb_payload()
    assert normalize_mb_metric(payload, expected=2)["mean_per_sample"] == 2.5
    payload["failed_samples"] = 1
    with pytest.raises(ValueError, match="incomplete"):
        normalize_mb_metric(payload, expected=2)


def test_h2_qa_uses_variant_and_conversation_as_sample_id():
    rows = [
        {
            "variant": "dyadic",
            "conversation_id": "dialogue1",
            "answer_attempts": 1,
            "error": "",
        },
        {
            "variant": "multiparty",
            "conversation_id": "dialogue1",
            "answer_attempts": 2,
            "answer_failed_attempts": 1,
            "error": "",
        },
    ]
    metric = calculate_formal_qa_metric("H2HMEM", rows)
    assert metric["available"]
    assert metric["num_samples"] == 2
    assert metric["total_calls"] == 3
    assert metric["failed_calls"] == 1


def test_merge_job_preserves_scores_and_writes_combined_calls(tmp_path, monkeypatch):
    import scripts.merge_mb_call_metrics as module

    monkeypatch.setattr(module, "EXPECTED_SAMPLE_COUNTS", {"Mem-Gallery": 2})
    output_root = tmp_path / "outputs"
    mb_root = tmp_path / "mb"
    result_dir = output_root / "Mem-Gallery" / "MIRIX"
    _write(result_dir / "metrics.json", {"f1": 0.25, "em": 0.5, "count": 2})
    _write(
        result_dir / "results.json",
        [
            {
                "dataset": "one",
                "answer_attempts": 1,
                "answer_failed_attempts": 0,
            },
            {
                "dataset": "two",
                "answer_attempts": 2,
                "answer_failed_attempts": 1,
            },
        ],
    )
    _write(mb_root / "Mem-Gallery" / "MIRIX" / "metrics.json", _mb_payload())

    assert merge_job(
        "Mem-Gallery",
        "MIRIX",
        output_root=output_root,
        mb_call_root=mb_root,
    )
    merged = json.loads((result_dir / "metrics.json").read_text())
    assert merged["f1"] == 0.25
    assert merged["calls"]["memory_bank"]["total_calls"] == 5
    assert merged["calls"]["qa"]["total_calls"] == 3
    assert merged["calls"]["total"]["total_calls"] == 8


def test_merge_job_never_accepts_hivemem(tmp_path):
    with pytest.raises(ValueError, match="non-HiveMem"):
        merge_job(
            "Mem-Gallery",
            "HiveMem",
            output_root=tmp_path,
            mb_call_root=tmp_path,
        )
