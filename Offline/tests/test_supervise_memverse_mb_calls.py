import json

from scripts.supervise_memverse_mb_calls import (
    GPU5_H2_HELPER,
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


def test_gpu5_helper_only_targets_remaining_h2_memverse(tmp_path):
    command = task_command(
        GPU5_H2_HELPER,
        output_root=tmp_path,
        embedding_base_url="http://127.0.0.1:8001/v1",
        sample_concurrency=2,
    )
    assert command[command.index("--benchmark") + 1] == "H2HMEM"
    assert command[command.index("--baseline") + 1] == "MemVerse"
    assert command[command.index("--executor-base-url") + 1].endswith(":8015/v1")
    assert "HiveMem" not in command
