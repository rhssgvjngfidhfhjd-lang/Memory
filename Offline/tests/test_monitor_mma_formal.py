from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sqlite3
import os


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "monitor_mma_formal.py"
SPEC = importlib.util.spec_from_file_location("monitor_mma_formal", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
monitor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(monitor)


def test_command_match_requires_exact_resumable_mma_job(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    job = {"result_dir": str(result_dir)}
    valid = [
        "python",
        "-m",
        "benchmarks.wma_harness.eval_wma",
        "--baseline",
        "MMA",
        "--result-dir",
        str(result_dir),
        "--resume",
    ]
    assert monitor.command_matches_job(valid, job)
    assert not monitor.command_matches_job(
        [part for part in valid if part != "--resume"], job
    )
    assert not monitor.command_matches_job(
        ["bash", "-c", " ".join(valid)], job
    )


def test_validate_resume_checkpoints_checks_snapshot_integrity(tmp_path: Path) -> None:
    result_dir = tmp_path / "results"
    sample = result_dir / "memory" / "datasets" / "sample"
    resume_dir = sample / ".resume"
    resume_dir.mkdir(parents=True)
    snapshot = resume_dir / "sqlite.session-000001.db"
    connection = sqlite3.connect(snapshot)
    try:
        connection.execute("CREATE TABLE memory (id INTEGER PRIMARY KEY)")
        connection.commit()
    finally:
        connection.close()
    (sample / ".offline_mma_resume.json").write_text(
        json.dumps(
            {
                "version": 1,
                "sample_id": "sample",
                "signature": "signature",
                "sqlite_snapshot": snapshot.name,
            }
        ),
        encoding="utf-8",
    )

    valid, message = monitor.validate_resume_checkpoints(result_dir)
    assert valid
    assert "validated 1" in message

    snapshot.write_bytes(b"not sqlite")
    valid, message = monitor.validate_resume_checkpoints(result_dir)
    assert not valid
    assert "SQLite" in message


def test_final_artifacts_require_complete_set(tmp_path: Path) -> None:
    for name in monitor.FINAL_ARTIFACTS[:-1]:
        (tmp_path / name).write_text("", encoding="utf-8")
    assert not monitor.final_artifacts_present(tmp_path)
    (tmp_path / monitor.FINAL_ARTIFACTS[-1]).write_text("", encoding="utf-8")
    assert monitor.final_artifacts_present(tmp_path)


def test_launch_environment_prepends_offline_source(
    tmp_path: Path, monkeypatch
) -> None:
    offline_root = tmp_path / "Offline"
    source_root = offline_root / "src"
    source_root.mkdir(parents=True)
    old_source = tmp_path / "old-src"
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join((str(old_source), str(source_root.resolve()))),
    )

    environment = monitor.launch_environment(offline_root)

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(source_root.resolve()),
        str(old_source),
    ]


def test_inherit_required_resume_environment_from_live_job(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(monitor, "process_command", lambda pid: ["python"])
    monkeypatch.setattr(
        monitor,
        "process_environment_value",
        lambda pid, key: "inherited-secret" if pid == 42 else "",
    )

    inherited = monitor.inherit_required_resume_environment(
        {"mma": {"method": "MMA", "child_pid": 42}}
    )

    assert inherited == [("OPENAI_API_KEY", 42)]
    assert os.environ["OPENAI_API_KEY"] == "inherited-secret"
