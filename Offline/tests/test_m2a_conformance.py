from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = OFFLINE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from validate_m2a_original_run import (  # noqa: E402
    DEFAULT_UPSTREAM_COMMIT,
    DEFAULT_UPSTREAM_URL,
    validate_run,
)


PROMPT_HASH = "a" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _build_fixture(root: Path) -> Path:
    run_dir = root / "run"
    state_dir = run_dir / "memory" / "datasets" / "demo"
    state_dir.mkdir(parents=True)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "baseline": "M2A",
            "baseline_state_dir": str(run_dir / "memory" / "datasets"),
            "questions": 2,
            "prompt_sha256": PROMPT_HASH,
            "baseline_runtime": {"compatibility_mode": "native_original_agent"},
            "m2a_conformance": {
                "upstream_url": DEFAULT_UPSTREAM_URL,
                "upstream_commit": DEFAULT_UPSTREAM_COMMIT,
                "deviations": [
                    {"id": "qwen_backbone", "detail": "model replaced by Qwen"},
                    {"id": "top7_memory_budget", "detail": "final memory budget is Top-7"},
                    {"id": "benchmark_answer_prompt", "detail": "harness QA prompt answers"},
                ],
                "internal_prompt_sha256": {
                    "chat_agent": "b" * 64,
                    "memory_manager_query": "c" * 64,
                    "memory_manager_update": "d" * 64,
                },
                "answer_prompt_sha256": PROMPT_HASH,
            },
        },
    )

    raw = sqlite3.connect(state_dir / "raw.db")
    raw.execute(
        "CREATE TABLE messages ("
        "msg_id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
        "role TEXT NOT NULL, text TEXT NOT NULL, image_path TEXT)"
    )
    raw.executemany(
        "INSERT INTO messages(timestamp, role, text, image_path) VALUES (?, ?, ?, ?)",
        [
            ("2026-01-01T10:00:00", "Alice", "first", None),
            ("2026-01-01T10:01:00", "Bob", "second", "/images/one.png"),
            ("2026-01-01T10:02:00", "Alice", "third", None),
        ],
    )
    raw.commit()
    raw.close()

    semantic = sqlite3.connect(state_dir / "semantic.db")
    semantic.execute(
        "CREATE TABLE memory (id INTEGER PRIMARY KEY, text TEXT, image_caption TEXT, "
        "image_path TEXT, evidence_ids TEXT, pad_vec TEXT)"
    )
    semantic.execute(
        "CREATE TABLE text_embs (id INTEGER PRIMARY KEY, text TEXT, "
        "text_dense TEXT, text_sparse TEXT)"
    )
    semantic.execute(
        "CREATE TABLE image_embs (id INTEGER PRIMARY KEY, image_dense TEXT)"
    )
    for memory_id in range(1, 8):
        image_path = "/images/one.png" if memory_id == 1 else ""
        semantic.execute(
            "INSERT INTO memory VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, f"memory {memory_id}", "caption" if image_path else "", image_path, "[[1, 3]]", "[]"),
        )
        semantic.execute(
            "INSERT INTO text_embs VALUES (?, ?, ?, ?)",
            (memory_id, f"memory {memory_id}", "[]", "{}"),
        )
    semantic.execute("INSERT INTO image_embs VALUES (1, '[]')")
    semantic.commit()
    semantic.close()

    results = [
        {
            "query_id": f"q{index}",
            "manifest_question_id": f"demo:Q{index}",
            "sample_id": "demo",
            "system_answer": "answer",
            "error": "",
        }
        for index in range(1, 3)
    ]
    _write_json(run_dir / "results.json", results)
    final_ids = [f"m2a:{index}" for index in range(1, 8)]
    traces = []
    for index in range(1, 3):
        traces.append(
            {
                "query_id": f"q{index}",
                "manifest_question_id": f"demo:Q{index}",
                "top_k": [{"memory_id": memory_id} for memory_id in final_ids],
                "m2a_trace": {
                    "chat_agent_query_calls": 1,
                    "memory_manager_query_calls": 1,
                    "semantic_searches": [
                        {
                            "requested_top_k": 10,
                            "effective_top_k": 7,
                            "candidate_count": 7,
                            "returned_memory_ids": final_ids,
                        }
                    ],
                    "raw_fetches": [{"raw_ids": [1, 2, 3]}],
                    "final_memory_ids": final_ids,
                    "final_memory_count": 7,
                },
            }
        )
    _write_jsonl(run_dir / "retrieval_trace.jsonl", traces)
    _write_jsonl(
        state_dir / "m2a_execution_trace.jsonl",
        [
            {"event": "call", "component": "ChatAgent", "action": "ingest", "raw_ids": [1], "turn_id": "t1"},
            {"event": "call", "component": "MemoryManager", "action": "semantic_delta", "memory_ids": [1, 2, 3]},
            {"event": "call", "component": "ChatAgent", "action": "ingest", "raw_ids": [2], "turn_id": "t2"},
            {"event": "call", "component": "MemoryManager", "action": "semantic_delta", "memory_ids": [4, 5]},
            {"event": "call", "component": "ChatAgent", "action": "ingest", "raw_ids": [3], "turn_id": "t3"},
            {"event": "call", "component": "MemoryManager", "action": "semantic_delta", "memory_ids": [6, 7]},
            {"event": "call", "component": "MemoryManager", "action": "query", "query_id": "q1"},
            {"event": "call", "component": "MemoryManager", "action": "semantic_search", "query_id": "q1"},
            {"event": "call", "component": "M2AAdapter", "action": "final_handoff", "query_id": "q1"},
            {"event": "call", "component": "MemoryManager", "action": "query", "query_id": "q2"},
            {"event": "call", "component": "MemoryManager", "action": "semantic_search", "query_id": "q2"},
            {"event": "call", "component": "M2AAdapter", "action": "final_handoff", "query_id": "q2"},
        ],
    )
    return run_dir


class M2AConformanceValidatorTest(unittest.TestCase):
    def test_valid_fixture_passes_all_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _build_fixture(Path(directory))
            report = validate_run(
                run_dir,
                expected_sample="demo",
                expected_utterances=3,
                expected_images=1,
                expected_questions=2,
            )

        self.assertTrue(report.passed, report.to_dict())
        self.assertEqual(report.statistics["semantic_memories"], 7)
        self.assertEqual(report.statistics["questions_with_exactly_seven_memories"], 2)

    def test_out_of_range_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _build_fixture(Path(directory))
            db_path = run_dir / "memory" / "datasets" / "demo" / "semantic.db"
            connection = sqlite3.connect(db_path)
            connection.execute("UPDATE memory SET evidence_ids='[[1, 99]]' WHERE id=1")
            connection.commit()
            connection.close()

            report = validate_run(run_dir, expected_sample="demo")

        failed = {check.name for check in report.checks if not check.passed}
        self.assertIn("semantic_evidence_ranges", failed)

    def test_more_than_seven_final_memories_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _build_fixture(Path(directory))
            trace_path = run_dir / "retrieval_trace.jsonl"
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            for row in rows:
                row["top_k"].append({"memory_id": "m2a:8"})
                row["m2a_trace"]["final_memory_ids"].append("m2a:8")
                row["m2a_trace"]["final_memory_count"] = 8
                row["m2a_trace"]["semantic_searches"][0]["returned_memory_ids"].append("m2a:8")
                row["m2a_trace"]["semantic_searches"][0]["candidate_count"] = 8
            _write_jsonl(trace_path, rows)

            report = validate_run(run_dir, expected_sample="demo")

        failed = {check.name for check in report.checks if not check.passed}
        self.assertIn("m2a_retrieval_budget_and_trace", failed)

    def test_simplified_or_fallback_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _build_fixture(Path(directory))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["baseline_runtime"]["compatibility_mode"] = "semantic_store_direct"
            _write_json(manifest_path, manifest)
            trace_path = run_dir / "memory" / "datasets" / "demo" / "m2a_execution_trace.jsonl"
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            rows.append({"event": "fallback", "component": "semantic_store", "action": "direct_insert"})
            _write_jsonl(trace_path, rows)

            report = validate_run(run_dir, expected_sample="demo")

        failed = {check.name for check in report.checks if not check.passed}
        self.assertIn("no_simplified_compatibility_mode", failed)
        self.assertIn("original_agent_manager_chain", failed)


if __name__ == "__main__":
    unittest.main()
