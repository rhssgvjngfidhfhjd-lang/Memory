import json
import threading
from concurrent.futures import ThreadPoolExecutor

from scripts import measure_baseline_mb_calls as measure


def _kwargs(tmp_path):
    return {
        "baseline": "MemVerse",
        "job_root": tmp_path,
        "executor_base_url": "http://127.0.0.1:8015/v1",
        "executor_model": "Qwen/Qwen3-VL-4B-Instruct",
        "embedding_base_url": "http://127.0.0.1:8001/v1",
        "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
        "embedding_dim": 2048,
        "request_timeout": 180,
        "retries": 2,
        "max_chunks": 0,
        "resume": True,
    }


def test_sample_lock_prevents_duplicate_rebuild(monkeypatch, tmp_path):
    spec = measure.SampleSpec(
        "H2HMEM", "dyadic/dialogue1", ("dyadic", "dialogue1"), tmp_path, tmp_path
    )
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def fake_claimed(current_spec, **kwargs):
        calls.append(current_spec.sample_id)
        entered.set()
        assert release.wait(timeout=5)
        payload = {"sample_id": current_spec.sample_id, "status": "completed"}
        artifact = kwargs["job_root"] / "samples" / measure._artifact_name(
            current_spec.sample_id
        )
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(measure, "_run_sample_claimed", fake_claimed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(measure._run_sample, spec, **_kwargs(tmp_path))
        assert entered.wait(timeout=5)
        second = measure._run_sample(spec, **_kwargs(tmp_path))
        assert second["status"] == "running_elsewhere"
        release.set()
        assert first.result(timeout=5)["status"] == "completed"

    assert measure._run_sample(spec, **_kwargs(tmp_path))["status"] == "completed"
    assert calls == [spec.sample_id]


def test_ordered_rows_prefers_durable_sample_artifact(tmp_path):
    spec = measure.SampleSpec(
        "H2HMEM", "dyadic/dialogue1", ("dyadic", "dialogue1"), tmp_path, tmp_path
    )
    artifact = tmp_path / "samples" / measure._artifact_name(spec.sample_id)
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps({"sample_id": spec.sample_id, "status": "completed"}),
        encoding="utf-8",
    )
    rows = measure._ordered_rows(
        [spec],
        tmp_path,
        {spec.sample_id: {"sample_id": spec.sample_id, "status": "running_elsewhere"}},
    )
    assert rows == [{"sample_id": spec.sample_id, "status": "completed"}]
