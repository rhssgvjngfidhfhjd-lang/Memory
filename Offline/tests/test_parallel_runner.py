from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.parallel_runner import (
    load_sample_artifact,
    parallel_map_ordered,
    save_sample_artifact,
    signature_digest,
    validated_paired_resume_signatures,
    validated_qa_only_resume_signatures,
)


def test_parallel_map_preserves_input_order() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def worker(value: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01 * (4 - value))
        with lock:
            active -= 1
        return value * 10

    assert parallel_map_ordered([1, 2, 3], worker, max_workers=3) == [10, 20, 30]
    assert peak >= 2


def test_parallel_map_reports_deferred_sample_failure(capsys) -> None:
    def worker(value: int) -> int:
        if value == 2:
            raise ValueError("broken sample")
        return value

    try:
        parallel_map_ordered([1, 2, 3], worker, max_workers=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("parallel_map_ordered should report aggregate failure")

    captured = capsys.readouterr()
    assert "[sample-error]" in captured.err
    assert '"sample_id": "2"' in captured.err
    assert "broken sample" in captured.err


def test_sample_artifact_is_signature_guarded_and_atomic() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        signature = signature_digest({"model": "demo", "top_k": 5})
        save_sample_artifact(
            root,
            "dyadic/dialogue1",
            signature=signature,
            artifact={"jobs": [{"query_id": "q1"}], "snapshots": []},
        )
        loaded = load_sample_artifact(
            root,
            "dyadic/dialogue1",
            signature=signature,
        )
        assert loaded == {"jobs": [{"query_id": "q1"}], "snapshots": []}
        assert load_sample_artifact(
            root,
            "dyadic/dialogue1",
            signature="different",
        ) is None


def test_output_layout_standardizes_pipeline_and_sample_checkpoints() -> None:
    layout = BaselineOutputLayout(Path("outputs/H2HMEM/MemVerse"))
    assert layout.pipeline_qa == Path("outputs/H2HMEM/MemVerse/pipeline_qa.jsonl")
    assert layout.sample_checkpoint_dir == Path(
        "outputs/H2HMEM/MemVerse/.checkpoint/samples"
    )


def test_historical_resume_signature_requires_matching_state_and_qa_pair() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        state = root / "state.json"
        qa = root / "qa.json"
        state.write_text('{"signature":"old"}', encoding="utf-8")
        qa.write_text('{"signature":"old"}', encoding="utf-8")
        assert validated_paired_resume_signatures(
            [("sample", state, qa)]
        ) == ("old",)

        qa.write_text('{"signature":"other"}', encoding="utf-8")
        try:
            validated_paired_resume_signatures([("sample", state, qa)])
        except RuntimeError as exc:
            assert "signature mismatch" in str(exc)
        else:
            raise AssertionError("mismatched durable checkpoints must be rejected")


def test_qa_only_resume_uses_audited_provenance_not_build_signature() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        provenance = root / ".mma_reuse_provenance.json"
        qa = root / "qa.json"
        provenance.write_text(
            '{"mode":"qa_only_isolated_memory_reuse"}', encoding="utf-8"
        )
        qa.write_text(
            '{"version":1,"sample_id":"sample","signature":"qa-run"}',
            encoding="utf-8",
        )
        assert validated_qa_only_resume_signatures(
            [("sample", provenance, qa)]
        ) == ("qa-run",)

        provenance.write_text('{"mode":"other"}', encoding="utf-8")
        try:
            validated_qa_only_resume_signatures([("sample", provenance, qa)])
        except RuntimeError as exc:
            assert "provenance mode" in str(exc)
        else:
            raise AssertionError("unaudited memory reuse must be rejected")
