from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.baseline_runtime.call_contract import audit_baseline_call_flow


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class BaselineCallContractTest(unittest.TestCase):
    def _run(self, mutate=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        result = {
            "query_id": "q1",
            "system_answer": "<answer>x</answer>",
            "error": "",
            "answer_attempts": 1,
            "answer_failed_attempts": 0,
            "native_answer_trace": {
                "single_agent_lifecycle": True,
                "native_agent_lifecycle_count": 1,
                "provisional_native_answer_ignored": False,
            },
        }
        retrieval = {
            "query_id": "q1",
            "top_k": [],
            "retrieval_method_trace": {
                "stage": "answer_complete",
                "requested_top_k": 7,
                "automatic_prefetch_counts_toward_top_k": True,
            },
        }
        calls = [
            {
                "phase": "qa",
                "query_id": "q1",
                "operation": "native_answer",
                "success": True,
            }
        ]
        state = {"result": result, "retrieval": retrieval, "calls": calls}
        if mutate:
            mutate(state)
        _write_json(root / "run_manifest.json", {"top_k": 7})
        _write_json(root / "results.json", [state["result"]])
        _write_jsonl(root / "retrieval_trace.jsonl", [state["retrieval"]])
        _write_jsonl(root / "call_trace.jsonl", state["calls"])
        return audit_baseline_call_flow(
            root,
            baseline="MIRIX",
            benchmark="Mem-Gallery",
            expected_qa=1,
            expected_top_k=7,
        )

    def test_accepts_single_native_lifecycle(self):
        self.assertTrue(self._run()["passed"])

    def test_rejects_retrieval_api_call(self):
        def mutate(state):
            state["calls"].append({"phase": "retrieval", "query_id": "q1"})

        report = self._run(mutate)
        self.assertFalse(report["passed"])
        self.assertIn("retrieval phase made 1 API calls", report["errors"])

    def test_rejects_logical_failure_even_when_http_succeeded(self):
        def mutate(state):
            state["result"]["error"] = "empty final answer"
            state["result"]["system_answer"] = ""

        report = self._run(mutate)
        self.assertFalse(report["passed"])
        self.assertIn("logical answer failure: q1", report["errors"])

    def test_rejects_unowned_qa_call(self):
        def mutate(state):
            state["calls"][0].pop("query_id")

        report = self._run(mutate)
        self.assertFalse(report["passed"])
        self.assertIn("1 QA API calls have no query_id", report["errors"])

    def test_rejects_skipped_build_fault(self):
        def mutate(state):
            state["calls"].append(
                {
                    "phase": "build_fault",
                    "event": "skipped_build_point",
                    "failed": True,
                }
            )

        report = self._run(mutate)
        self.assertFalse(report["passed"])
        self.assertEqual(report["counts"]["build_faults"], 1)
        self.assertIn(
            "build phase recorded 1 skipped fault(s)", report["errors"]
        )


if __name__ == "__main__":
    unittest.main()
