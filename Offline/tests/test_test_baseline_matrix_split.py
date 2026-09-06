from __future__ import annotations

from pathlib import Path
import unittest

from scripts.run_test_baseline_matrix import (
    EXPECTED_COUNTS,
    Job,
    JOB_ORDER,
    PROTOCOL,
    ROOT,
    command_for,
    load_json,
    load_selection,
    validate_run_manifest_selection,
)


class TestBaselineMatrixSplitRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.selection = load_selection(
            ROOT / "configs" / "multimodal_split_manifest.json"
        )
        cls.config = load_json(ROOT / "configs" / "defaults.json")

    def test_formal_test_question_counts_are_exact(self):
        actual = {
            benchmark: len(
                self.selection.question_ids_for_benchmark(benchmark)
            )
            for benchmark in EXPECTED_COUNTS
        }
        self.assertEqual(
            actual,
            {
                "Mem-Gallery": 275,
                "H2HMEM": 360,
                "WorldMemArena": 440,
            },
        )

    def test_protocol_fixes_top_k(self):
        self.assertEqual(PROTOCOL["top_k"], 7)
        self.assertEqual(PROTOCOL["efficiency_config"], "model_efficiency.json")

    def test_matrix_contains_all_seven_non_hivemem_baselines(self):
        methods = {method for method, _ in JOB_ORDER}
        benchmarks = {benchmark for _, benchmark in JOB_ORDER}
        self.assertEqual(
            methods,
            {
                "AUGUSTUSMemory",
                "OmniSimpleMem",
                "M2A",
                "MIRIX",
                "MMA",
                "MemVerse",
                "M3-Agent-caption",
            },
        )
        self.assertEqual(benchmarks, set(EXPECTED_COUNTS))
        self.assertEqual(len(JOB_ORDER), 21)

    def test_every_harness_command_uses_question_level_manifest(self):
        data_dirs = {
            benchmark: Path("/tmp") / benchmark
            for benchmark in EXPECTED_COUNTS
        }
        for benchmark in EXPECTED_COUNTS:
            with self.subTest(benchmark=benchmark):
                command = command_for(
                    Job("M2A", benchmark),
                    Path("/tmp/result"),
                    "http://127.0.0.1:8013/v1",
                    "http://127.0.0.1:8001/v1",
                    self.config,
                    data_dirs,
                    self.selection,
                )
                self.assertEqual(command.count("--split-manifest"), 1)
                manifest_index = command.index("--split-manifest")
                self.assertEqual(
                    Path(command[manifest_index + 1]),
                    self.selection.manifest_path.resolve(),
                )
                self.assertEqual(command.count("--split"), 1)
                split_index = command.index("--split")
                self.assertEqual(command[split_index + 1], "test")
                self.assertEqual(command.count("--efficiency-config"), 1)
                efficiency_index = command.index("--efficiency-config")
                self.assertEqual(
                    command[efficiency_index + 1],
                    self.config["efficiency_config"],
                )

    def test_legacy_conversation_only_run_is_rejected(self):
        job = Job("M2A", "Mem-Gallery")
        with self.assertRaisesRegex(RuntimeError, "strict question-level"):
            validate_run_manifest_selection(
                job,
                {
                    "selection_mode": "legacy",
                    "split": "test",
                    "questions": 301,
                },
                self.selection,
            )

    def test_strict_run_manifest_is_accepted(self):
        job = Job("M2A", "H2HMEM")
        question_ids = self.selection.question_ids_for_benchmark(job.benchmark)
        validate_run_manifest_selection(
            job,
            {
                "selection_mode": "strict_manifest",
                "split": "test",
                "split_manifest": str(self.selection.manifest_path),
                "split_manifest_sha256": self.selection.manifest_sha256,
                "questions": len(question_ids),
                "ordered_question_ids": list(question_ids),
            },
            self.selection,
        )


if __name__ == "__main__":
    unittest.main()
