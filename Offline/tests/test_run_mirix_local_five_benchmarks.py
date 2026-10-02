from __future__ import annotations

import unittest

from scripts.run_mirix_local_five_benchmarks import (
    _manifest_runtime_values,
    _recovered_truncation_count,
)


class MirixFiveBenchmarkManifestTest(unittest.TestCase):
    def test_normalizes_flat_multimodal_manifest(self):
        manifest = {
            "baseline": "MIRIX",
            "top_k": 5,
            "embedding_model": "embedding-0.6b",
            "embedding_dim": 2048,
            "answer_model": "qwen3-vl-4b",
            "executor_model": "qwen3-vl-4b",
            "executor_max_tokens": 8192,
        }
        self.assertEqual(_manifest_runtime_values(manifest), manifest)

    def test_normalizes_nested_matrix_manifest(self):
        expected = {
            "baseline": "MIRIX",
            "top_k": 5,
            "embedding_model": "embedding-0.6b",
            "embedding_dim": 2048,
            "answer_model": "qwen3-vl-4b",
            "executor_model": "qwen3-vl-4b",
            "executor_max_tokens": 8192,
        }
        manifest = {
            "baseline": {"name": "MIRIX", "adapter": "mirix_family"},
            "top_k": 5,
            "configuration": {
                key: value
                for key, value in expected.items()
                if key not in {"baseline", "top_k"}
            },
        }
        self.assertEqual(_manifest_runtime_values(manifest), expected)

    def test_accepts_recovered_memory_build_truncation(self):
        calls = [
            {
                "request_id": 22,
                "sample_id": "sample-1",
                "phase": "memory_build",
                "success": True,
                "failed": False,
                "truncated": True,
            },
            {
                "request_id": 23,
                "sample_id": "sample-1",
                "phase": "memory_build",
                "success": True,
                "failed": False,
                "truncated": False,
            },
        ]
        self.assertEqual(_recovered_truncation_count(calls), 1)

    def test_accepts_intervening_retries_and_concurrent_trace_rows(self):
        calls = [
            {
                "request_id": 7,
                "sample_id": "sample-1",
                "phase": "memory_build",
                "success": True,
                "failed": False,
                "truncated": True,
            },
            {
                "request_id": 8,
                "sample_id": "sample-2",
                "phase": "memory_build",
                "success": True,
                "failed": False,
                "truncated": False,
            },
            {
                "request_id": 9,
                "sample_id": "sample-1",
                "phase": "memory_build",
                "success": False,
                "failed": True,
                "truncated": False,
            },
            {
                "request_id": 10,
                "sample_id": "sample-1",
                "phase": "memory_build",
                "success": True,
                "failed": False,
                "truncated": False,
            },
        ]
        self.assertEqual(_recovered_truncation_count(calls), 1)

    def test_rejects_terminal_or_cross_sample_truncation(self):
        terminal = [
            {
                "request_id": 7,
                "sample_id": "sample-1",
                "phase": "memory_build",
                "success": True,
                "truncated": True,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "unrecovered"):
            _recovered_truncation_count(terminal)

        wrong_successor = terminal + [
            {
                "request_id": 8,
                "sample_id": "sample-2",
                "phase": "memory_build",
                "success": True,
                "truncated": False,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "unrecovered"):
            _recovered_truncation_count(wrong_successor)


if __name__ == "__main__":
    unittest.main()
