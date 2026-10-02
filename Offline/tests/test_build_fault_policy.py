from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.baseline_runtime.build_fault_policy import (
    ConsecutiveBuildFaultPolicy,
    is_non_skippable_build_failure,
)
from benchmarks.baseline_runtime.call_trace import CallRecorder
from embedding.chunk_builder import Chunk


class BuildFaultPolicyTests(unittest.TestCase):
    def test_success_resets_consecutive_failures_and_audits_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "calls.jsonl"
            recorder = CallRecorder(
                trace_path=trace_path,
                baseline="M2A",
                benchmark="WorldMemArena",
                sample_id="demo",
                reset=True,
            )
            policy = ConsecutiveBuildFaultPolicy(
                baseline="M2A",
                benchmark="WorldMemArena",
                enabled=True,
                maximum=10,
                recorder=recorder,
            )
            chunk = Chunk(
                chunk_id="S1:R1",
                text="bad",
                metadata={"session_id": "S1", "dialogue_id": "D1"},
            )
            for _ in range(9):
                policy.handle(
                    RuntimeError("malformed tool arguments"),
                    chunk=chunk,
                    point_kind="ingest",
                    session_id="S1",
                )
            policy.success()
            policy.handle(
                RuntimeError("truncated completion"),
                chunk=chunk,
                point_kind="ingest",
                session_id="S1",
            )

            self.assertEqual(policy.consecutive, 1)
            self.assertEqual(len(policy.failures), 10)
            self.assertEqual(
                policy.failures[-1]["consecutive_failed_build_points"], 1
            )
            self.assertIn('"phase": "build_fault"', trace_path.read_text())

    def test_tenth_consecutive_failure_stops(self) -> None:
        policy = ConsecutiveBuildFaultPolicy(
            baseline="M2A",
            benchmark="H2HMEM",
            enabled=True,
            maximum=10,
        )
        for _ in range(9):
            policy.handle(
                RuntimeError("bad point"),
                chunk=None,
                point_kind="ingest",
                session_id="session1",
            )
        with self.assertRaisesRegex(RuntimeError, "10 consecutive failed M2A"):
            policy.handle(
                RuntimeError("bad point"),
                chunk=None,
                point_kind="ingest",
                session_id="session1",
            )

    def test_global_failure_is_never_skipped(self) -> None:
        policy = ConsecutiveBuildFaultPolicy(
            baseline="M2A",
            benchmark="H2HMEM",
            enabled=True,
        )
        with self.assertRaisesRegex(RuntimeError, "No space left on device"):
            policy.handle(
                RuntimeError("No space left on device"),
                chunk=None,
                point_kind="ingest",
                session_id="session1",
            )
        self.assertEqual(policy.failures, [])

    def test_fail_open_audits_and_skips_normally_global_failure(self) -> None:
        policy = ConsecutiveBuildFaultPolicy(
            baseline="M2A",
            benchmark="H2HMEM",
            enabled=True,
            fail_open=True,
        )
        policy.handle(
            RuntimeError("No space left on device"),
            chunk=None,
            point_kind="ingest",
            session_id="session1",
        )
        self.assertEqual(policy.consecutive, 1)
        self.assertEqual(len(policy.failures), 1)

    def test_database_lifecycle_failures_are_never_skipped(self) -> None:
        for name in (
            "DetachedInstanceError",
            "PendingRollbackError",
            "IntegrityError",
            "OperationalError",
            "StatementError",
        ):
            with self.subTest(name=name):
                self.assertTrue(
                    is_non_skippable_build_failure(
                        RuntimeError(f"wrapped MMA native failure: {name}: broken")
                    )
                )

    def test_native_json_parse_failure_is_an_isolated_build_point(self) -> None:
        policy = ConsecutiveBuildFaultPolicy(
            baseline="MMA",
            benchmark="WorldMemArena",
            enabled=True,
        )
        policy.handle(
            RuntimeError(
                "MMANativeAgentFailure: resource_memory_agent response JSON: "
                "Unterminated string"
            ),
            chunk=None,
            point_kind="ingest",
            session_id="S07",
        )
        self.assertEqual(policy.consecutive, 1)
        self.assertEqual(len(policy.failures), 1)


if __name__ == "__main__":
    unittest.main()
