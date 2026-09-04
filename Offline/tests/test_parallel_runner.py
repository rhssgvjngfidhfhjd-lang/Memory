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
