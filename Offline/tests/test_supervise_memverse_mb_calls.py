import json

from scripts.supervise_memverse_mb_calls import (
    GPU5_H2_HELPER,
    GPU5_HELPERS,
    GPU5_WMA_HELPER,
    TASKS,
    metrics_complete,
    task_command,
)


def test_metrics_complete_requires_full_available_matrix(tmp_path):
    path = tmp_path / "metrics.json"
    payload = {
        "available": True,
        "num_samples": 20,
        "completed_samples": 20,
        "failed_samples": 0,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert metrics_complete(path, 20)
    payload["failed_samples"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not metrics_complete(path, 20)


def test_task_command_is_memverse_only_and_resumable(tmp_path):
    command = task_command(
        TASKS[0],
        output_root=tmp_path,
        embedding_base_url="http://127.0.0.1:8001/v1",
        sample_concurrency=2,
    )
    assert command[command.index("--baseline") + 1] == "MemVerse"
    assert "HiveMem" not in command
    assert "--resume" in command


def test_gpu5_helpers_prioritize_wma_then_h2_and_remain_memverse_only(tmp_path):
    assert GPU5_HELPERS == (GPU5_WMA_HELPER, GPU5_H2_HELPER)
    assert [task.benchmark for task in GPU5_HELPERS] == ["WorldMemArena", "H2HMEM"]
    for helper in GPU5_HELPERS:
        command = task_command(
            helper,
            output_root=tmp_path,
            embedding_base_url="http://127.0.0.1:8001/v1",
            sample_concurrency=2,
        )
        assert command[command.index("--benchmark") + 1] == helper.benchmark
        assert command[command.index("--baseline") + 1] == "MemVerse"
        assert command[command.index("--executor-base-url") + 1].endswith(":8015/v1")
        assert "HiveMem" not in command
