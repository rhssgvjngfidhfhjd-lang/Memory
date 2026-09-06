from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.benchmark_manifest import BenchmarkExpectation, load_benchmark_manifest


OFFLINE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = OFFLINE_ROOT / "scripts" / "run_full_baseline_matrix.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("_test_full_matrix", SCRIPT_PATH)
if SCRIPT_SPEC is None or SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
full_matrix = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = full_matrix
SCRIPT_SPEC.loader.exec_module(full_matrix)


class BenchmarkManifestTest(unittest.TestCase):
    def test_loads_repository_manifest(self) -> None:
        expectations = load_benchmark_manifest(
            OFFLINE_ROOT / "configs" / "full_benchmark_manifest.json",
            required_benchmarks=full_matrix.BENCHMARKS,
        )

        self.assertEqual(set(expectations), set(full_matrix.BENCHMARKS))
        h2hmem = expectations["H2HMEM"]
        self.assertEqual(sum(h2hmem.variants.values()), h2hmem.question_count)

    def test_rejects_variant_total_that_disagrees_with_question_count(self) -> None:
        payload = {
            "schema_version": 1,
            "benchmarks": {
                "Example": {
                    "judge_name": "example",
                    "question_count": 3,
                    "variants": {"first": 1, "second": 1},
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match question_count"):
                load_benchmark_manifest(path)

    def test_requires_benchmark_specific_completeness_fields(self) -> None:
        payload = {
            "schema_version": 1,
            "benchmarks": {
                "WorldMemArena": {
                    "judge_name": "worldmemarena",
                    "question_count": 1,
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires sample_count"):
                load_benchmark_manifest(path)

    def test_output_validation_uses_configured_counts(self) -> None:
        expectation = BenchmarkExpectation(
            judge_name="memgallery",
            question_count=1,
            dataset_count=1,
            require_all_datasets=True,
        )
        job = full_matrix.Job("test_memgallery", "Mem-Gallery", "HiveMem")
        configuration = {
            "answer_model": full_matrix.MODEL,
            "answer_temperature": 0.0,
            "executor_model": full_matrix.MODEL,
            "executor_temperature": 0.0,
            "executor_visual_input": "image",
            "embedding_model": full_matrix.EMBEDDING_MODEL,
            "embedding_base_url": full_matrix.EMBEDDING_ENDPOINT,
            "embedding_dim": 2048,
            "top_k": full_matrix.TOP_K,
            "request_timeout": 180,
            "retries": 2,
            "answer_base_url": next(iter(full_matrix.INFERENCE_ENDPOINTS)),
        }
        configuration["executor_base_url"] = configuration["answer_base_url"]
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory)
            memory_dir = result_dir / "memory"
            memory_dir.mkdir()
            (result_dir / "results.json").write_text(
                json.dumps([{"dataset": "demo", "question": "q", "category": "c"}]),
                encoding="utf-8",
            )
            (result_dir / "retrieval_trace.jsonl").write_text(
                json.dumps(
                    {
                        "dataset": "demo",
                        "question": "q",
                        "category": "c",
                        "qa_index": 0,
                        "top_k": [
                            {
                                "memory_id": "m1",
                                "content": "memory",
                                "rank": 1,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (memory_dir / "memory_snapshot.jsonl").write_text(
                json.dumps(
                    {"memory_id": "m1", "text": "memory", "backend_type": "test"}
                )
                + "\n",
                encoding="utf-8",
            )
            (result_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "baseline": "HiveMem",
                        "configuration": configuration,
                        "all_datasets": True,
                        "questions": 1,
                    }
                ),
                encoding="utf-8",
            )

            full_matrix.validate_job_outputs(
                job,
                result_dir,
                {"Mem-Gallery": expectation},
            )


if __name__ == "__main__":
    unittest.main()
