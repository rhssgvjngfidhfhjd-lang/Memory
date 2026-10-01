from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_test_baseline_matrix import (
    EXPECTED_COUNTS,
    Job,
    JOB_ORDER,
    SMOKE_JOB_ORDER,
    PROTOCOL,
    ROOT,
    check_services,
    command_for,
    formal_job_attempts,
    load_json,
    load_selection,
    validate_run_manifest_selection,
)
from benchmarks.h2hmem_harness.prompts import prompt_sha256 as h2hmem_prompt_sha256


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

    def test_budget_defaults_disable_formal_job_restart(self):
        config = load_json(ROOT.parent / "Nvida_api" / "defaults_gpt-5-mini.json")
        self.assertEqual(formal_job_attempts(config), 1)
        self.assertEqual(config["sample_concurrency"], 1)
        self.assertEqual(formal_job_attempts({}), 2)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            formal_job_attempts({"formal_job_attempts": 0})

    def test_matrix_contains_all_ten_non_hivemem_baselines(self):
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
                "NaiveRAG",
                "MuRAG",
                "UniversalRAG",
            },
        )
        self.assertEqual(benchmarks, set(EXPECTED_COUNTS))
        self.assertEqual(len(JOB_ORDER), 30)

    def test_m2a_smoke_covers_all_three_benchmarks(self):
        self.assertEqual(
            {
                benchmark
                for method, benchmark in SMOKE_JOB_ORDER
                if method == "M2A"
            },
            set(EXPECTED_COUNTS),
        )

    def test_mirix_smoke_covers_all_three_benchmarks(self):
        self.assertEqual(
            {
                benchmark
                for method, benchmark in SMOKE_JOB_ORDER
                if method == "MIRIX"
            },
            set(EXPECTED_COUNTS),
        )

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

    def test_baselines_use_separate_memory_build_output_limits(self):
        data_dirs = {
            benchmark: Path("/tmp") / benchmark
            for benchmark in EXPECTED_COUNTS
        }
        for method, expected in (
            ("M3-Agent-caption", "1024"),
            ("MIRIX", "2048"),
            ("M2A", "4096"),
        ):
            with self.subTest(method=method):
                command = command_for(
                    Job(method, "Mem-Gallery"),
                    Path("/tmp/result"),
                    "http://127.0.0.1:8015/v1",
                    "http://127.0.0.1:8001/v1",
                    self.config,
                    data_dirs,
                    self.selection,
                )
                self.assertEqual(command.count("--executor-max-tokens"), 1)
                option_index = command.index("--executor-max-tokens")
                self.assertEqual(command[option_index + 1], expected)
                if method == "MIRIX":
                    hard_index = command.index("--executor-hard-max-tokens")
                    self.assertEqual(command[hard_index + 1], "4096")

    def test_smoke_uses_configured_sample_concurrency(self):
        data_dirs = {
            benchmark: Path("/tmp") / benchmark
            for benchmark in EXPECTED_COUNTS
        }
        command = command_for(
            Job("MIRIX", "H2HMEM"),
            Path("/tmp/result"),
            "http://127.0.0.1:8014/v1",
            "http://127.0.0.1:8001/v1",
            self.config,
            data_dirs,
            self.selection,
            smoke=True,
        )
        index = command.index("--sample-concurrency")
        self.assertEqual(command[index + 1], "4")

    def test_mirix_wma_uses_lower_formal_sample_concurrency(self):
        data_dirs = {
            benchmark: Path("/tmp") / benchmark
            for benchmark in EXPECTED_COUNTS
        }
        command = command_for(
            Job("MIRIX", "WorldMemArena"),
            Path("/tmp/result"),
            "http://127.0.0.1:8015/v1",
            "http://127.0.0.1:8001/v1",
            self.config,
            data_dirs,
            self.selection,
        )
        index = command.index("--sample-concurrency")
        self.assertEqual(command[index + 1], "2")

    def test_default_mirix_formal_run_does_not_skip_build_faults(self):
        self.assertFalse(self.config["mirix_skip_failed_build_points"])

    def test_service_preflight_uses_configured_embedding_model(self):
        config = dict(self.config)
        config.update(
            {
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "embedding_dim": 3,
            }
        )
        calls = []

        def fake_request(url, payload=None, **_kwargs):
            calls.append((url, payload))
            if url.endswith("/models"):
                return {"data": [{"id": config["answer_model"]}]}
            if url.endswith("/embeddings"):
                return {"data": [{"embedding": [0.0, 0.0, 0.0]}]}
            return {"choices": [{"message": {"content": "OK"}}]}

        with tempfile.TemporaryDirectory() as directory:
            key_file = Path(directory) / "key"
            key_file.write_text("sk-or-v1-test", encoding="utf-8")
            config["judge_key_file"] = str(key_file)
            with patch(
                "scripts.run_test_baseline_matrix.request_json",
                side_effect=fake_request,
            ):
                check_services(
                    ["http://127.0.0.1:8014/v1"],
                    "http://127.0.0.1:8002/v1",
                    config,
                )

        embedding_payloads = [
            payload for url, payload in calls if url.endswith("/embeddings")
        ]
        self.assertEqual(
            embedding_payloads,
            [{"model": "Qwen/Qwen3-Embedding-0.6B", "input": ["test-only matrix preflight"]}],
        )

    def test_m2a_build_fault_policy_reaches_all_three_harnesses(self):
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
                self.assertEqual(
                    command.count("--m2a-skip-failed-build-points"), 1
                )
                option_index = command.index(
                    "--m2a-max-consecutive-failed-build-points"
                )
                self.assertEqual(command[option_index + 1], "10")

    def test_m2a_build_fault_policy_explicitly_disables_skipping(self):
        data_dirs = {
            benchmark: Path("/tmp") / benchmark
            for benchmark in EXPECTED_COUNTS
        }
        config = {
            **self.config,
            "m2a_skip_failed_build_points": False,
        }
        command = command_for(
            Job("M2A", "Mem-Gallery"),
            Path("/tmp/result"),
            "https://openrouter.ai/api/v1",
            "http://127.0.0.1:8002/v1",
            config,
            data_dirs,
            self.selection,
        )
        self.assertIn("--no-m2a-skip-failed-build-points", command)
        self.assertNotIn("--m2a-skip-failed-build-points", command)

    def test_mirix_build_fault_policy_reaches_memgallery_and_h2hmem(self):
        data_dirs = {
            benchmark: Path("/tmp") / benchmark
            for benchmark in EXPECTED_COUNTS
        }
        config = {
            **self.config,
            "mirix_skip_failed_build_points": True,
            "mirix_max_consecutive_failed_build_points": 10,
        }
        for benchmark in EXPECTED_COUNTS:
            with self.subTest(benchmark=benchmark):
                command = command_for(
                    Job("MIRIX", benchmark),
                    Path("/tmp/result"),
                    "http://127.0.0.1:8013/v1",
                    "http://127.0.0.1:8001/v1",
                    config,
                    data_dirs,
                    self.selection,
                )
                self.assertEqual(
                    command.count("--mirix-skip-failed-build-points"), 1
                )
                option_index = command.index(
                    "--mirix-max-consecutive-failed-build-points"
                )
                self.assertEqual(command[option_index + 1], "10")

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
                "prompt_sha256": h2hmem_prompt_sha256(),
            },
            self.selection,
        )


if __name__ == "__main__":
    unittest.main()
