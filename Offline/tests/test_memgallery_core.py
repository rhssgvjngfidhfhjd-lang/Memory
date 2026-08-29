import argparse
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hive_mem.executor import ExecutionResult, MemoryExecutor
from hive_mem.mau import MAUBank
from hive_mem.builder import MAUBuilder, build_signatures_compatible
from hive_mem.builder import MemoryEvent
from benchmarks.memgallery_harness.runner.metrics import (
    add_memory_metrics,
    add_retrieval_memory_tokens,
    calculate_cost_mb,
    calculate_cost_qa,
    calculate_calls_mb,
    calculate_calls_qa,
    combine_call_metrics,
    calculate_memory_metrics,
    calculate_retrieval_memory_tokens,
    merge_llm_judge_metrics,
    f1_score,
    normalize_answer,
    provenance_hit,
    write_retrieval_memory_token,
)
from benchmarks.memgallery_harness.runner.answer_client import (
    VLMAnswerClient,
    build_retrieved_memory_context,
)
from hive_mem.llm_client import GenerationResponse, LLMClient, _build_user_content
from hive_mem.build_memories import apply_config_defaults, completed_dataset_stats
from hive_mem.output_layout import DatasetLayout
from embedding.build_query_embeddings import _prepare_text_query, _resolve_devices
from benchmarks.memgallery_harness.runner.prompts import (
    CATEGORY_PROMPTS,
    PROMPT_DIR,
    SYSTEM_PROMPT,
    prompt_manifest,
)
from embedding.qwen3_text_embedding import (
    DEFAULT_QUERY_INSTRUCTION,
    Qwen3TextEmbeddingService,
    Qwen3TextMemoryEmbedder,
    create_embedding_service,
    create_memory_embedder,
)
from embedding.qwen3vl_embedding import Qwen3VLEmbeddingService, QwenMemoryEmbedder


class FakeEmbedder:
    def embed_texts(self, texts, mode="context"):
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        result = np.asarray([[float(len(text)), 1.0] for text in values], dtype=np.float32)
        return result[0] if single else result


class OfficialMetricAndAnswerRetryTest(unittest.TestCase):
    def test_f1_matches_memgallery_decimal_id_and_stemming_rules(self):
        self.assertEqual(
            normalize_answer("The cats and 3.14 IMG_001"),
            "cats 3.14 img_001",
        )
        self.assertEqual(f1_score("The cats and 3.14 IMG_001", "cat 3.14 img_001"), 1.0)

    def test_empty_answer_is_retried(self):
        class EmptyThenValidClient(VLMAnswerClient):
            def __init__(self):
                super().__init__(retries=1)
                self.calls = 0

            def _post_json(self, url, payload):
                self.calls += 1
                content = "" if self.calls == 1 else "valid answer"
                return {"choices": [{"message": {"content": content}}]}

        client = EmptyThenValidClient()
        with patch("benchmarks.memgallery_harness.runner.answer_client.time.sleep"):
            answer = client.answer(
                system_prompt="system",
                memory_items=[],
                question_prompt="question",
            )
        self.assertEqual(answer, "valid answer")
        self.assertEqual(client.calls, 2)

    def test_answer_response_records_exact_provider_usage_and_attempts(self):
        class EmptyThenValidClient(VLMAnswerClient):
            def __init__(self):
                super().__init__(retries=1)
                self.calls = 0

            def _post_json(self, url, payload):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "choices": [{"message": {"content": ""}}],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 1,
                            "total_tokens": 6,
                        },
                    }
                return {
                    "choices": [{"message": {"content": "valid answer"}}],
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 7,
                        "total_tokens": 32,
                    },
                }

        client = EmptyThenValidClient()
        with patch("benchmarks.memgallery_harness.runner.answer_client.time.sleep"):
            response = client.answer_with_usage(
                system_prompt="system",
                memory_items=[],
                question_prompt="question",
            )
        self.assertEqual(response.text, "valid answer")
        self.assertEqual(response.attempts, 2)
        self.assertEqual(response.failed_attempts, 1)
        self.assertEqual(
            response.usage,
            {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
        )

    def test_retry_without_usage_marks_aggregate_usage_unavailable(self):
        class MissingRetryUsageClient(VLMAnswerClient):
            def __init__(self):
                super().__init__(retries=1)
                self.calls = 0

            def _post_json(self, url, payload):
                self.calls += 1
                if self.calls == 1:
                    return {"choices": [{"message": {"content": ""}}]}
                return {
                    "choices": [{"message": {"content": "valid answer"}}],
                    "usage": {
                        "prompt_tokens": 25,
                        "completion_tokens": 7,
                        "total_tokens": 32,
                    },
                }

        client = MissingRetryUsageClient()
        with patch("benchmarks.memgallery_harness.runner.answer_client.time.sleep"):
            response = client.answer_with_usage(
                system_prompt="system",
                memory_items=[],
                question_prompt="question",
            )
        self.assertEqual(response.attempts, 2)
        self.assertEqual(response.failed_attempts, 1)
        self.assertIsNone(response.usage)


class RuntimeConfigurationTest(unittest.TestCase):
    def test_memgallery_prompts_resolve_from_workspace_and_are_nonempty(self):
        expected = Path(__file__).resolve().parents[2] / "Mem-Gallery" / "benchmark" / "prompt"
        self.assertEqual(PROMPT_DIR, expected.resolve())
        self.assertTrue(SYSTEM_PROMPT)
        self.assertEqual(set(CATEGORY_PROMPTS), {"AR", "CD", "VS"})
        self.assertTrue(all(CATEGORY_PROMPTS.values()))
        manifest = prompt_manifest()
        self.assertEqual(set(manifest["prompt_sha256"]), {
            "sys_prompt.txt", "ar_prompt.txt", "cd_prompt.txt", "vs_prompt.txt"
        })

    def test_config_overlay_can_be_limited_to_shared_wma_keys(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-dir", default="wma-data")
        parser.add_argument("--answer-base-url", default="http://unconfigured.invalid/v1")
        apply_config_defaults(parser, allowed_keys={"answer_base_url"})
        args = parser.parse_args([])
        self.assertEqual(args.data_dir, "wma-data")
        self.assertEqual(args.answer_base_url, "http://127.0.0.1:18000/v1")

    def test_query_embedding_devices_auto_detect_and_validate(self):
        self.assertEqual(
            _resolve_devices("auto", cuda_device_count=1),
            ["cuda:0"],
        )
        self.assertEqual(
            _resolve_devices("auto", cuda_device_count=0),
            ["cpu"],
        )
        self.assertEqual(
            _resolve_devices("0", cuda_device_count=1),
            ["cuda:0"],
        )
        with self.assertRaisesRegex(ValueError, "only 1 CUDA"):
            _resolve_devices("0,1", cuda_device_count=1)


class Qwen3TextEmbeddingTest(unittest.TestCase):
    def test_completed_dataset_detection_checks_event_count_and_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = DatasetLayout(Path(directory) / "dataset")
            layout.reports_dir.mkdir(parents=True)
            layout.vectors_dir.mkdir(parents=True)
            (layout.root / "memories.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            layout.build_stats.write_text(
                json.dumps(
                    {
                        "input_events": 3,
                        "final_memories": 2,
                        "executor_visual_input": "image",
                    }
                ),
                encoding="utf-8",
            )
            np.save(layout.text_vectors, np.zeros((2, 1024), dtype=np.float32))
            checkpoint = Path(directory) / "checkpoint"
            stats = completed_dataset_stats(
                layout, checkpoint, expected_events=3, expected_dim=1024
            )
            self.assertTrue(stats["skipped_complete"])
            self.assertIsNone(
                completed_dataset_stats(
                    layout, checkpoint, expected_events=3, expected_dim=2048
                )
            )
            self.assertIsNone(
                completed_dataset_stats(
                    layout,
                    checkpoint,
                    expected_events=3,
                    expected_dim=1024,
                    expected_executor_visual_input="caption",
                )
            )

    def test_factory_keeps_vl_default_and_selects_text_model_explicitly(self):
        vl_service = create_embedding_service(
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            device="cpu",
            expected_dim=2048,
        )
        text_service = create_embedding_service(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device="cpu",
            expected_dim=1024,
        )
        vl_memory = create_memory_embedder(
            model_name="Qwen/Qwen3-VL-Embedding-2B",
            device="cpu",
            expected_dim=2048,
        )
        text_memory = create_memory_embedder(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            device="cpu",
            expected_dim=1024,
        )
        self.assertIsInstance(vl_service, Qwen3VLEmbeddingService)
        self.assertIsInstance(text_service, Qwen3TextEmbeddingService)
        self.assertIsInstance(vl_memory, QwenMemoryEmbedder)
        self.assertIsInstance(text_memory, Qwen3TextMemoryEmbedder)
        self.assertTrue(vl_service.supports_images)
        self.assertFalse(text_service.supports_images)

    def test_query_instruction_and_image_caption_are_text(self):
        service = Qwen3TextEmbeddingService(device="cpu")
        query = _prepare_text_query(
            {
                "question": "What color is the bag?",
                "query_image": {"path": "/tmp/bag.png", "caption": "A blue bag."},
            }
        )
        self.assertEqual(query, "What color is the bag?\nImage caption: A blue bag.")
        self.assertEqual(
            service._format_query(query),
            f"Instruct: {DEFAULT_QUERY_INSTRUCTION}\nQuery: {query}",
        )
        with self.assertRaisesRegex(ValueError, "text-only"):
            service.embed_query("question", ["/tmp/bag.png"])

    def test_last_token_pool_handles_left_and_right_padding(self):
        import torch

        hidden = torch.tensor(
            [
                [[1.0], [2.0], [3.0]],
                [[4.0], [5.0], [6.0]],
            ]
        )
        left_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        right_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        np.testing.assert_allclose(
            Qwen3TextEmbeddingService._last_token_pool(hidden, left_mask).numpy(),
            [[3.0], [6.0]],
        )
        np.testing.assert_allclose(
            Qwen3TextEmbeddingService._last_token_pool(hidden, right_mask).numpy(),
            [[2.0], [6.0]],
        )

    def test_normalization_is_done_in_float32(self):
        import torch
        import torch.nn.functional as functional

        vector = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.bfloat16)
        normalized = functional.normalize(vector.float(), p=2, dim=1)
        self.assertEqual(normalized.dtype, torch.float32)
        np.testing.assert_allclose(
            torch.linalg.vector_norm(normalized, dim=1).numpy(),
            [1.0],
            atol=1e-6,
        )


class ProvenanceMemoryBankTest(unittest.TestCase):
    def test_save_uses_mau_shaped_json_and_keeps_legacy_aliases(self):
        bank = MAUBank()
        bank.add_memory(
            "Lena likes Maltese dogs",
            np.asarray([1.0, 0.0]),
            metadata={
                "session_id": "D1",
                "image_paths": ["/tmp/dog.jpg"],
                "image_captions": ["A small white dog."],
            },
        )
        item = bank.memories[0]
        self.assertEqual(item.content, item.summary)
        self.assertEqual(item.memory_id, item.id)
        self.assertTrue(item.id.startswith("mau_"))

        with tempfile.TemporaryDirectory() as directory:
            bank.save(directory)
            self.assertTrue((Path(directory) / "vectors" / "text.npy").exists())
            with (Path(directory) / "memories.jsonl").open() as handle:
                row = json.loads(handle.readline())
            loaded = MAUBank.load(directory)

        self.assertNotIn("memory_id", row)
        self.assertNotIn("content", row)
        self.assertEqual(row["id"], item.id)
        self.assertEqual(row["summary"], "Lena likes Maltese dogs")
        self.assertEqual(row["modality_type"], "multimodal")
        self.assertNotIn("embedding", row)
        self.assertNotIn("details", row)
        self.assertNotIn("timestamp", row)
        self.assertEqual(row["links"], {"prev": None, "next": None, "related": []})
        self.assertEqual(loaded.memories[0].content, "Lena likes Maltese dogs")
        self.assertEqual(loaded.memories[0].memory_id, item.id)

    def test_load_accepts_legacy_agentmem_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_row = {
                "memory_id": "mem_old",
                "content": "legacy fact",
                "metadata": {"source_dialogue_ids": ["D1:1"]},
            }
            (root / "memories.jsonl").write_text(json.dumps(old_row) + "\n")
            np.save(root / "vectors.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
            loaded = MAUBank.load(root)

        self.assertEqual(loaded.memories[0].id, "mem_old")
        self.assertEqual(loaded.memories[0].summary, "legacy fact")
        self.assertEqual(loaded.memories[0].content, "legacy fact")

    def test_archived_status_and_links_survive_save_load(self):
        bank = MAUBank()
        bank.add_memory("old", np.asarray([1.0, 0.0]), metadata={"session_id": "D1"})
        bank.add_memory("new", np.asarray([0.0, 1.0]), metadata={"session_id": "D2"})
        old, new = bank.memories[0], bank.memories[1]
        old.status = "ARCHIVED"
        new.links["related"].append({"target": old.id, "type": "SUPERSEDES"})

        with tempfile.TemporaryDirectory() as directory:
            bank.save(directory)
            loaded = MAUBank.load(directory)
        self.assertEqual(loaded.memories[0].status, "ARCHIVED")
        self.assertEqual(loaded.memories[1].links["related"], [{"target": old.id, "type": "SUPERSEDES"}])

    def test_insert_inherits_event_metadata(self):
        bank = MAUBank()
        executor = MemoryExecutor(llm_client=None, embedder=FakeEmbedder())
        executor.apply_to_memory_bank(
            [ExecutionResult(success=True, memory_content="new fact")],
            bank,
            event_metadata={"source_dialogue_ids": ["D2:3"]},
        )
        self.assertEqual(bank.memories[0].metadata["source_dialogue_ids"], ["D2:3"])

    def test_hit_uses_top_k_memory_groups(self):
        groups = [["D1:1", "D1:2"], ["D2:1"]]
        self.assertEqual(provenance_hit(groups, ["D1:2"], k=1), 1.0)
        self.assertEqual(provenance_hit(groups, ["D2:1"], k=1), 0.0)

    def test_merge_llm_judge_metrics_adds_overall_and_category_scores(self):
        metrics = {
            "count": 2,
            "f1": 0.5,
            "exact_match": 0.25,
            "llm_judge": None,
            "retrieval_hitrate@5": 1.0,
            "by_category": {
                "AR": {
                    "count": 2,
                    "f1": 0.5,
                    "exact_match": 0.25,
                    "llm_judge": None,
                    "retrieval_hitrate@5": 1.0,
                }
            },
        }
        judge_metrics = {
            "accuracy": 0.75,
            "by_category": {"AR": {"accuracy": 0.5}},
        }
        merged = merge_llm_judge_metrics(metrics, judge_metrics)
        self.assertEqual(merged["llm_judge"], 0.75)
        self.assertEqual(merged["em"], 0.25)
        self.assertEqual(merged["by_category"]["AR"]["llm_judge"], 0.5)
        self.assertEqual(list(merged)[:3], ["f1", "em", "llm_judge"])
        self.assertEqual(
            list(merged["by_category"]["AR"])[:3],
            ["f1", "em", "llm_judge"],
        )
        self.assertNotIn("exact_match", merged)

    def test_memory_metrics_use_recorded_usage_and_active_summary_characters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "d"
            (dataset / "traces").mkdir(parents=True)
            (dataset / "memories.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"summary": "active fact", "status": "ACTIVE"}),
                        json.dumps({"summary": "old fact", "status": "ARCHIVED"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (dataset / "traces" / "build.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "llm_usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 4,
                                    "total_tokens": 14,
                                }
                            }
                        ),
                        json.dumps(
                            {
                                "llm_usage": {
                                    "prompt_tokens": 20,
                                    "completion_tokens": 6,
                                    "total_tokens": 26,
                                }
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            memory_metrics = calculate_memory_metrics(root)

        self.assertEqual(
            memory_metrics["memory_build_tokens"],
            {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
        )
        self.assertEqual(memory_metrics["summary_characters"], len("active fact"))

    def test_cost_mb_uses_only_evaluated_samples_and_sample_wise_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usage_by_sample = {
                "sample_a": (10, 4),
                "sample_b": (20, 6),
                "not_evaluated": (1000, 1000),
            }
            for sample_id, (prompt_tokens, completion_tokens) in usage_by_sample.items():
                trace_dir = root / "datasets" / sample_id / "traces"
                trace_dir.mkdir(parents=True)
                (trace_dir / "build.jsonl").write_text(
                    json.dumps(
                        {
                            "llm_usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            }
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )

            cost = calculate_cost_mb(
                root,
                ["sample_a", "sample_b"],
                input_price=0.1,
                output_price=0.6,
            )

        self.assertTrue(cost["available"])
        self.assertEqual(cost["input_tokens"], 30)
        self.assertEqual(cost["output_tokens"], 10)
        self.assertEqual(cost["num_samples"], 2)
        self.assertAlmostEqual(cost["cost_sum"], 9.0)
        self.assertAlmostEqual(cost["mean_per_sample"], 4.5)
        self.assertEqual(cost["formula"], "(30 * 0.1 + 10 * 0.6) / 2 = 4.5")

    def test_cost_mb_is_unavailable_without_prices_or_exact_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_dir = root / "datasets" / "sample_a" / "traces"
            trace_dir.mkdir(parents=True)
            (trace_dir / "build.jsonl").write_text("{}\n", encoding="utf-8")

            missing_prices = calculate_cost_mb(
                root,
                ["sample_a"],
                input_price=None,
                output_price=None,
            )
            missing_usage = calculate_cost_mb(
                root,
                ["sample_a"],
                input_price=0.1,
                output_price=0.6,
            )

        self.assertFalse(missing_prices["available"])
        self.assertIn("configs/defaults.json", missing_prices["reason"])
        self.assertFalse(missing_usage["available"])
        self.assertIn("missing exact provider usage", missing_usage["reason"])

    def test_cost_qa_uses_sample_wise_not_qa_wise_denominator(self):
        results = [
            {
                "dataset": "sample_a",
                "answer_token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
                "answer_attempts": 1,
                "error": "",
            },
            {
                "dataset": "sample_a",
                "answer_token_usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 3,
                    "total_tokens": 23,
                },
                "answer_attempts": 2,
                "error": "",
            },
            {
                "dataset": "sample_b",
                "answer_token_usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 5,
                    "total_tokens": 35,
                },
                "answer_attempts": 1,
                "error": "",
            },
        ]

        cost = calculate_cost_qa(
            results,
            sample_id_field="dataset",
            input_price=0.1,
            output_price=0.6,
        )

        self.assertTrue(cost["available"])
        self.assertEqual(cost["input_tokens"], 60)
        self.assertEqual(cost["output_tokens"], 10)
        self.assertEqual(cost["num_queries"], 3)
        self.assertEqual(cost["num_samples"], 2)
        self.assertEqual(cost["total_attempts"], 4)
        self.assertEqual(cost["retried_queries"], 1)
        self.assertAlmostEqual(cost["cost_sum"], 12.0)
        self.assertAlmostEqual(cost["mean_per_sample"], 6.0)
        self.assertEqual(cost["formula"], "(60 * 0.1 + 10 * 0.6) / 2 = 6")

    def test_cost_qa_is_strict_about_prices_usage_and_answer_failures(self):
        valid = {
            "sample_id": "sample_a",
            "answer_token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
            "answer_attempts": 1,
            "error": "",
        }
        missing_prices = calculate_cost_qa(
            [valid],
            sample_id_field="sample_id",
            input_price=None,
            output_price=None,
        )
        missing_usage = calculate_cost_qa(
            [{**valid, "answer_token_usage": None}],
            sample_id_field="sample_id",
            input_price=0.1,
            output_price=0.6,
        )
        failed = calculate_cost_qa(
            [{**valid, "error": "timeout"}],
            sample_id_field="sample_id",
            input_price=0.1,
            output_price=0.6,
        )

        self.assertFalse(missing_prices["available"])
        self.assertIn("configs/defaults.json", missing_prices["reason"])
        self.assertFalse(missing_usage["available"])
        self.assertIn("missing exact provider usage", missing_usage["reason"])
        self.assertFalse(failed["available"])
        self.assertIn("answer failures", failed["reason"])

    def test_calls_are_split_and_combined_with_sample_wise_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_by_sample = {
                "sample_a": [(1, 0), (3, 2)],
                "sample_b": [(1, 0)],
            }
            for sample_id, counts in rows_by_sample.items():
                trace_dir = root / "datasets" / sample_id / "traces"
                trace_dir.mkdir(parents=True)
                (trace_dir / "build.jsonl").write_text(
                    "".join(
                        json.dumps(
                            {
                                "llm_attempts": attempts,
                                "llm_failed_attempts": failed,
                            }
                        )
                        + "\n"
                        for attempts, failed in counts
                    ),
                    encoding="utf-8",
                )
            results = [
                {
                    "dataset": "sample_a",
                    "answer_attempts": 1,
                    "answer_failed_attempts": 0,
                },
                {
                    "dataset": "sample_a",
                    "answer_attempts": 2,
                    "answer_failed_attempts": 1,
                },
                {
                    "dataset": "sample_b",
                    "answer_attempts": 1,
                    "answer_failed_attempts": 0,
                },
            ]
            calls = combine_call_metrics(
                calculate_calls_mb(root, ["sample_a", "sample_b"]),
                calculate_calls_qa(results, sample_id_field="dataset"),
            )

        self.assertEqual(calls["memory_bank"]["total_calls"], 5)
        self.assertEqual(calls["memory_bank"]["failed_calls"], 2)
        self.assertEqual(calls["memory_bank"]["formula"], "5 / 2 = 2.5")
        self.assertEqual(calls["qa"]["total_calls"], 4)
        self.assertEqual(calls["qa"]["failed_calls"], 1)
        self.assertEqual(calls["qa"]["formula"], "4 / 2 = 2")
        self.assertEqual(calls["total"]["total_calls"], 9)
        self.assertEqual(calls["total"]["failed_calls"], 3)
        self.assertEqual(calls["total"]["successful_calls"], 6)
        self.assertEqual(calls["total"]["formula"], "(5 + 4) / 2 = 4.5")

    def test_calls_are_unavailable_for_historical_rows_without_attempt_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_dir = root / "datasets" / "sample_a" / "traces"
            trace_dir.mkdir(parents=True)
            (trace_dir / "build.jsonl").write_text("{}\n", encoding="utf-8")
            memory_bank = calculate_calls_mb(root, ["sample_a"])
        qa = calculate_calls_qa(
            [{"dataset": "sample_a"}],
            sample_id_field="dataset",
        )
        calls = combine_call_metrics(memory_bank, qa)

        self.assertFalse(memory_bank["available"])
        self.assertFalse(qa["available"])
        self.assertFalse(calls["total"]["available"])
        self.assertIn("exact call counts", calls["total"]["reason"])

    def test_memory_metrics_follow_f1_and_judge_in_summary(self):
        combined = add_memory_metrics(
            {
                "f1": 0.5,
                "em": 0.25,
                "llm_judge": 0.75,
                "count": 2,
                "by_category": {
                    "AR": {
                        "count": 2,
                        "f1": 0.4,
                        "em": 0.2,
                        "llm_judge": 0.6,
                    }
                },
            },
            {
                "memory_build_tokens": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "summary_characters": 100,
            },
        )
        self.assertEqual(
            list(combined)[:5],
            ["f1", "em", "llm_judge", "memory_build_tokens", "summary_characters"],
        )
        self.assertEqual(
            list(combined["by_category"]["AR"])[:3],
            ["f1", "em", "llm_judge"],
        )

    def test_inserts_are_never_deduplicated(self):
        # 1 chunk -> 1 MAU design (2026-08-06): identical text from two chunks
        # stays as two memories, each keeping its own provenance.
        bank = MAUBank()
        executor = MemoryExecutor(llm_client=None, embedder=FakeEmbedder())
        executor.apply_to_memory_bank(
            [ExecutionResult(success=True, memory_content="Alice likes coffee")],
            bank,
            event_metadata={"source_dialogue_ids": ["D1:1"], "source_chunk_ids": ["x:D1:1"]},
        )
        executor.apply_to_memory_bank(
            [ExecutionResult(success=True, memory_content="Alice likes coffee")],
            bank,
            event_metadata={"source_dialogue_ids": ["D1:5"], "source_chunk_ids": ["x:D1:5"]},
        )
        self.assertEqual(len(bank), 2)
        self.assertEqual(bank.memories[0].metadata["source_dialogue_ids"], ["D1:1"])
        self.assertEqual(bank.memories[1].metadata["source_dialogue_ids"], ["D1:5"])


class RetrievalMemoryTokenTest(unittest.TestCase):
    class WhitespaceTokenizer:
        def encode(self, text, add_special_tokens=False):
            return str(text).split()

    def test_context_formatter_matches_answer_prompt_memory_section(self):
        items = [
            {
                "text": "memory D1:IMG_001",
                "image": {"path": "/tmp/image.jpg", "img_id": "D1:IMG_001"},
                "metadata": {
                    "session_id": "D1",
                    "dialogue_id": "D1:1",
                    "image_id": "D1:IMG_001",
                },
            }
        ]
        context, image_paths = build_retrieved_memory_context(items, "VS")
        client = VLMAnswerClient()
        full_text, full_image_paths = client._build_text_and_image_paths(
            items,
            "question",
            None,
            "VS",
        )
        self.assertTrue(full_text.startswith(context + "\n\n\n\nquestion"))
        self.assertEqual(image_paths, ["/tmp/image.jpg"])
        self.assertEqual(full_image_paths, image_paths)

    def test_historical_trace_counts_empty_graph_and_image_memory(self):
        rows = [
            {
                "category": "MR",
                "top_k": [
                    {
                        "via": "vector",
                        "content": "fact D1:IMG_001",
                        "source_dialogue_ids": ["D1:1"],
                        "image_ids": [],
                        "image_paths": [],
                    },
                    {
                        "via": "graph",
                        "content": "related fact",
                        "source_dialogue_ids": ["D1:2"],
                        "image_ids": [],
                        "image_paths": [],
                    },
                ],
            },
            {
                "category": "VS",
                "top_k": [
                    {
                        "via": "vector",
                        "content": "visual fact",
                        "source_dialogue_ids": ["D2:1"],
                        "image_ids": ["D2:IMG_001"],
                        "image_paths": ["/tmp/image.jpg"],
                    }
                ],
            },
            {"category": "FR", "top_k": []},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text(
                json.dumps({"graph_mode": "append"}),
                encoding="utf-8",
            )
            (root / "retrieval_trace.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            metrics = calculate_retrieval_memory_tokens(
                root,
                tokenizer=self.WhitespaceTokenizer(),
            )
            written = write_retrieval_memory_token(
                root,
                tokenizer=self.WhitespaceTokenizer(),
            )
            saved = json.loads((root / "retrieval_memory_token.json").read_text())

        self.assertEqual(metrics["total_tokens"], 38)
        self.assertEqual(metrics["query_count"], 3)
        self.assertEqual(metrics["average_tokens_per_query"], 12.67)
        self.assertEqual(written, metrics)
        self.assertEqual(saved, metrics)

    def test_retrieval_tokens_are_inserted_with_memory_metrics(self):
        combined = add_retrieval_memory_tokens(
            {
                "f1": 0.5,
                "em": 0.25,
                "llm_judge": 0.75,
                "memory_build_tokens": {"total_tokens": 10},
                "summary_characters": 100,
                "count": 2,
            },
            {
                "total_tokens": 20,
                "query_count": 2,
                "average_tokens_per_query": 10.0,
            },
        )
        self.assertEqual(
            list(combined)[:6],
            [
                "f1",
                "em",
                "llm_judge",
                "memory_build_tokens",
                "summary_characters",
                "retrieval_memory_tokens",
            ],
        )


class CrashingLLMClient:
    def __init__(self, crash_at):
        self.n = 0
        self.crash_at = crash_at

    def generate(self, prompt):
        self.n += 1
        if self.n - 1 == self.crash_at:
            raise RuntimeError("simulated crash")
        return f"MEMORY_ITEM: fact number {self.n}"


def _event_index_from_prompt(prompt):
    chunk = prompt.split("### Current Chunk\n", 1)[1].splitlines()[0]
    return int(chunk.rsplit(" ", 1)[1])


class OutOfOrderLLMClient:
    def __init__(self, requests):
        self.barrier = threading.Barrier(requests)
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.completion_order = []

    def generate(self, prompt):
        event_index = _event_index_from_prompt(prompt)
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.barrier.wait(timeout=2)
        time.sleep((3 - event_index) * 0.01)
        with self.lock:
            self.completion_order.append(event_index)
            self.active -= 1
        return f"MEMORY_ITEM: fact {event_index}"


class IndexedFailureLLMClient:
    def generate(self, prompt):
        event_index = _event_index_from_prompt(prompt)
        if event_index == 1:
            raise RuntimeError("simulated concurrent crash")
        return f"MEMORY_ITEM: fact {event_index}"


class BuilderResumeTest(unittest.TestCase):
    def test_resume_allows_only_loopback_endpoint_port_change(self):
        stored = {
            "dataset": "demo",
            "executor_model": "model",
            "executor_base_url": "http://127.0.0.1:28001/v1",
            "embedding_base_url": "http://127.0.0.1:8001/v1",
        }
        current = {
            **stored,
            "executor_base_url": "http://localhost:18000/v1",
        }
        self.assertTrue(build_signatures_compatible(stored, current))
        self.assertFalse(
            build_signatures_compatible(
                stored, {**current, "executor_model": "different-model"}
            )
        )

    @staticmethod
    def events(count=5):
        return [
            MemoryEvent(
                text=f"event {i}",
                dataset="d",
                dialogue_id=f"D1:{i}",
                session_id="D1",
                round_id=i,
                source_chunk_id=f"c{i}",
            )
            for i in range(count)
        ]

    def test_concurrent_executor_commits_memories_and_traces_in_event_order(self):
        events = self.events(3)
        client = OutOfOrderLLMClient(requests=3)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            stats = MAUBuilder(client, FakeEmbedder()).build(
                events,
                out,
                resume=False,
                executor_concurrency=3,
            )
            bank = MAUBank.load(out)
            with (out / "traces" / "build.jsonl").open() as handle:
                traces = [json.loads(line) for line in handle]

        self.assertEqual(client.max_active, 3)
        self.assertEqual(client.completion_order, [2, 1, 0])
        self.assertEqual(
            [item.content for item in bank.memories],
            ["fact 0", "fact 1", "fact 2"],
        )
        self.assertEqual(
            [item.metadata["source_chunk_ids"] for item in bank.memories],
            [["c0"], ["c1"], ["c2"]],
        )
        self.assertEqual([trace["event_index"] for trace in traces], [0, 1, 2])
        self.assertEqual(stats["executor_concurrency"], 3)

    def test_concurrent_failure_only_checkpoints_contiguous_commits(self):
        events = self.events(3)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            builder = MAUBuilder(IndexedFailureLLMClient(), FakeEmbedder())
            with self.assertRaisesRegex(RuntimeError, "concurrent crash"):
                builder.build(
                    events,
                    out,
                    resume=True,
                    checkpoint_every=1,
                    executor_concurrency=3,
                )
            state = json.loads(
                (out / ".checkpoint" / "builder_state.json").read_text()
            )
            with (out / "traces" / "build.jsonl").open() as handle:
                indices = [json.loads(line)["event_index"] for line in handle]

        self.assertEqual(state["next_event_index"], 1)
        self.assertEqual(indices, [0])

    def test_executor_concurrency_must_be_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "at least 1"):
                MAUBuilder(CrashingLLMClient(crash_at=999), FakeEmbedder()).build(
                    self.events(1),
                    Path(directory) / "out",
                    executor_concurrency=0,
                )

    def test_resume_after_mid_run_crash_does_not_duplicate_trace_entries(self):
        events = self.events()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            builder = MAUBuilder(CrashingLLMClient(crash_at=4), FakeEmbedder())
            with self.assertRaises(RuntimeError):
                builder.build(events, out, resume=True, checkpoint_every=3)

            builder_resumed = MAUBuilder(CrashingLLMClient(crash_at=999), FakeEmbedder())
            builder_resumed.build(events, out, resume=True, checkpoint_every=3)

            with (out / "traces" / "build.jsonl").open() as handle:
                indices = [json.loads(line)["event_index"] for line in handle]
        self.assertEqual(indices, [0, 1, 2, 3, 4])

    def test_resume_rejects_a_different_executor_visual_input(self):
        events = [
            MemoryEvent(
                text=f"event {i}",
                dataset="d",
                dialogue_id=f"D1:{i}",
                session_id="D1",
                round_id=i,
                source_chunk_id=f"c{i}",
            )
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            builder = MAUBuilder(CrashingLLMClient(crash_at=1), FakeEmbedder())
            with self.assertRaises(RuntimeError):
                builder.build(
                    events,
                    out,
                    resume=True,
                    checkpoint_every=1,
                    executor_visual_input="caption",
                )

            resumed = MAUBuilder(CrashingLLMClient(crash_at=999), FakeEmbedder())
            with self.assertRaisesRegex(ValueError, "visual input mismatch"):
                resumed.build(
                    events,
                    out,
                    resume=True,
                    checkpoint_every=1,
                    executor_visual_input="image",
                )


class ScriptedLLMClient:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        self.last_prompt = prompt
        return self.response


class MultimodalScriptedLLMClient(ScriptedLLMClient):
    def generate(self, prompt, image_paths=None):
        self.last_prompt = prompt
        self.last_image_paths = list(image_paths or [])
        return self.response


class UsageLLMClient:
    def generate_with_usage(self, prompt):
        return GenerationResponse(
            text="MEMORY_ITEM: measured fact",
            usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        )


class OperationTogglesTest(unittest.TestCase):
    def test_insert_only_prompt_omits_disabled_blocks_and_mandates_insert(self):
        executor = MemoryExecutor(llm_client=None, embedder=FakeEmbedder())
        prompt = executor._build_prompt("chunk")
        self.assertIn("MEMORY_ITEM:", prompt)
        self.assertNotIn("ACTION:", prompt)
        self.assertIn("### Memory Item Rules", prompt)
        self.assertIn("### Entities Rules", prompt)
        self.assertNotIn("UPDATE", prompt)
        self.assertNotIn("NOOP", prompt)
        self.assertIn("MUST output at least one memory item", prompt)

    def test_image_visual_input_replaces_current_and_previous_captions(self):
        llm_client = MultimodalScriptedLLMClient("MEMORY_ITEM: visual fact")
        executor = MemoryExecutor(llm_client=llm_client, embedder=FakeEmbedder())
        chunk = (
            "session: D2\n"
            "user: What is shown?\n"
            "assistant: Please inspect it.\n"
            "image_id: D2:IMG_001\n"
            "image_caption: A caption that must not be sent.\n"
            "previous_round_summary: D1:1; user asked; image_caption old caption"
        )

        executor.execute(
            chunk,
            image_paths=["/tmp/original.jpg"],
            visual_input="image",
        )

        self.assertEqual(llm_client.last_image_paths, ["/tmp/original.jpg"])
        self.assertIn("user: What is shown?", llm_client.last_prompt)
        self.assertIn("image_id: D2:IMG_001", llm_client.last_prompt)
        self.assertIn("### Attached Image", llm_client.last_prompt)
        self.assertNotIn("caption that must not be sent", llm_client.last_prompt)
        self.assertNotIn("old caption", llm_client.last_prompt)

    def test_caption_visual_input_keeps_caption_and_sends_no_image(self):
        llm_client = MultimodalScriptedLLMClient("MEMORY_ITEM: caption fact")
        executor = MemoryExecutor(llm_client=llm_client, embedder=FakeEmbedder())

        executor.execute(
            "user: What is shown?\nimage_caption: A red bicycle.",
            image_paths=["/tmp/original.jpg"],
            visual_input="caption",
        )

        self.assertEqual(llm_client.last_image_paths, [])
        self.assertIn("image_caption: A red bicycle.", llm_client.last_prompt)
        self.assertNotIn("### Attached Image", llm_client.last_prompt)

    def test_image_caption_visual_input_keeps_caption_and_sends_image(self):
        llm_client = MultimodalScriptedLLMClient("MEMORY_ITEM: combined fact")
        executor = MemoryExecutor(llm_client=llm_client, embedder=FakeEmbedder())

        executor.execute(
            "user: What is shown?\nimage_caption: A red bicycle.",
            image_paths=["/tmp/original.jpg"],
            visual_input="image_caption",
        )

        self.assertEqual(llm_client.last_image_paths, ["/tmp/original.jpg"])
        self.assertIn("image_caption: A red bicycle.", llm_client.last_prompt)
        self.assertIn("### Attached Image", llm_client.last_prompt)

    def test_llm_user_content_embeds_original_image_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            image_path.write_bytes(b"original-image-bytes")
            content = _build_user_content("prompt", [str(image_path)])

        self.assertEqual(content[0], {"type": "text", "text": "prompt"})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(_build_user_content("prompt", []), "prompt")

    def test_llm_client_sends_multimodal_content_and_keeps_usage(self):
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="MEMORY_ITEM: visual fact")
                        )
                    ],
                    usage={
                        "prompt_tokens": 9,
                        "completion_tokens": 3,
                        "total_tokens": 12,
                    },
                )

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        client = LLMClient("model", "http://localhost/v1", "key", retry_sleep=0)
        client._next_client = lambda: fake_client
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.jpg"
            image_path.write_bytes(b"raw-jpeg-bytes")
            response = client.generate_with_usage(
                "prompt",
                image_paths=[str(image_path)],
            )

        content = captured["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "prompt"})
        self.assertTrue(
            content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )
        self.assertEqual(response.text, "MEMORY_ITEM: visual fact")
        self.assertEqual(response.usage["total_tokens"], 12)
        self.assertEqual(response.attempts, 1)
        self.assertEqual(response.failed_attempts, 0)

    def test_llm_client_records_failed_retry_and_does_not_claim_exact_usage(self):
        class Completions:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary failure")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                    usage={
                        "prompt_tokens": 9,
                        "completion_tokens": 3,
                        "total_tokens": 12,
                    },
                )

        completions = Completions()
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        client = LLMClient(
            "model",
            "http://localhost/v1",
            "key",
            max_retries=2,
            retry_sleep=0,
        )
        client._next_client = lambda: fake_client

        response = client.generate_with_usage("prompt")

        self.assertEqual(completions.calls, 2)
        self.assertEqual(response.attempts, 2)
        self.assertEqual(response.failed_attempts, 1)
        self.assertEqual(response.usage, {})

    def test_llm_client_disables_openai_sdk_hidden_retries(self):
        client = LLMClient("model", "http://localhost/v1", "key")
        with patch("openai.OpenAI") as openai_client:
            client._next_client()

        self.assertEqual(openai_client.call_args.kwargs["max_retries"], 0)

    def test_default_config_uses_raw_image_for_executor(self):
        config_path = Path(__file__).resolve().parents[1] / "configs" / "defaults.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["executor_visual_input"], "image")
        self.assertEqual(config["executor_concurrency"], 16)

    def test_response_without_memory_item_is_rejected(self):
        llm_client = ScriptedLLMClient("Here is a summary of the chunk, but not in the required format.")
        executor = MemoryExecutor(llm_client=llm_client, embedder=FakeEmbedder())
        _, actions = executor.execute(chunk_text="chunk")
        self.assertEqual(len(actions), 1)
        self.assertFalse(actions[0].success)
        self.assertIn("No MEMORY_ITEM block", actions[0].reasoning)

    def test_insert_only_builder_falls_back_to_raw_chunk_on_unusable_response(self):
        events = [
            MemoryEvent(
                text="event zero",
                dataset="d",
                dialogue_id="D1:0",
                session_id="D1",
                round_id=0,
                source_chunk_id="c0",
            )
        ]
        builder = MAUBuilder(
            ScriptedLLMClient("gibberish with no action"),
            FakeEmbedder(),
        )
        with tempfile.TemporaryDirectory() as directory:
            stats = builder.build(events, Path(directory) / "out", resume=False)
            bank = MAUBank.load(Path(directory) / "out")
        self.assertEqual(stats["fallback_inserts_this_run"], 1)
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank.memories[0].content, "event zero")
        self.assertEqual(bank.memories[0].metadata["source_dialogue_ids"], ["D1:0"])
        self.assertEqual(bank.memories[0].metadata["source"], "fallback_insert")

    def test_builder_records_generation_usage_in_trace(self):
        event = MemoryEvent(
            text="event zero",
            dataset="d",
            dialogue_id="D1:0",
            session_id="D1",
            round_id=0,
            source_chunk_id="c0",
        )
        builder = MAUBuilder(UsageLLMClient(), FakeEmbedder())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "out"
            builder.build([event], root, resume=False)
            trace = json.loads((root / "traces" / "build.jsonl").read_text())
        self.assertEqual(
            trace["llm_usage"],
            {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        )
        self.assertEqual(trace["llm_attempts"], 1)
        self.assertEqual(trace["llm_failed_attempts"], 0)


class MemoryEdgesTest(unittest.TestCase):
    @staticmethod
    def _bank_with_sessions():
        from hive_mem.build_memory_edges import build_temporal_chain

        bank = MAUBank()
        rows = [
            ("bought allergy medicine", "D3", 1, "2024-06-18"),
            ("medicine was recommended by Alice", "D3", 2, "2024-06-18"),
            ("had an allergic reaction", "D4", 1, "2024-06-25"),
            ("switched to medicine B", "D7", 1, "2024-07-28"),
        ]
        # Insert out of order to prove sorting is metadata-driven.
        for summary, session, round_id, date in [rows[2], rows[0], rows[3], rows[1]]:
            bank.add_memory(
                summary,
                np.asarray([float(len(summary)), 1.0]),
                metadata={"session_id": session, "round_id": round_id, "date": date},
            )
        build_temporal_chain(bank)
        return bank

    def test_temporal_chain_orders_by_date_session_round(self):
        bank = self._bank_with_sessions()
        by_summary = {item.summary: item for item in bank.memories}
        chain = []
        cursor = next(item for item in bank.memories if item.links["prev"] is None)
        while cursor is not None:
            chain.append(cursor.summary)
            next_id = cursor.links["next"]
            cursor = next((i for i in bank.memories if i.id == next_id), None)
        self.assertEqual(
            chain,
            [
                "bought allergy medicine",
                "medicine was recommended by Alice",
                "had an allergic reaction",
                "switched to medicine B",
            ],
        )
        self.assertIsNone(by_summary["switched to medicine B"].links["next"])

    def test_temporal_chain_skips_archived_memories(self):
        from hive_mem.build_memory_edges import build_temporal_chain

        bank = self._bank_with_sessions()
        bank.memories[1].status = "ARCHIVED"
        build_temporal_chain(bank)
        archived = bank.memories[1]
        self.assertIsNone(archived.links["prev"])
        self.assertIsNone(archived.links["next"])
        chained_ids = {
            item.links["prev"] for item in bank.memories if item.links["prev"]
        } | {item.links["next"] for item in bank.memories if item.links["next"]}
        self.assertNotIn(archived.id, chained_ids)

    def test_candidate_pairs_respect_session_window(self):
        from hive_mem.build_memory_edges import generate_candidate_pairs

        bank = self._bank_with_sessions()
        pairs = generate_candidate_pairs(bank, session_window=1)
        summaries = {
            (bank.memories[a].summary, bank.memories[b].summary) for a, b in pairs
        }
        self.assertIn(
            ("bought allergy medicine", "medicine was recommended by Alice"), summaries
        )
        self.assertIn(
            ("medicine was recommended by Alice", "had an allergic reaction"), summaries
        )
        # D3 (session 3) and D7 (session 7) are farther than the window.
        self.assertNotIn(
            ("bought allergy medicine", "switched to medicine B"), summaries
        )

    def test_event_relation_classification_writes_confident_edges_only(self):
        from hive_mem.build_memory_edges import classify_event_relations

        bank = self._bank_with_sessions()
        pairs = [(1, 0), (0, 2)]  # (earlier, later) bank indices

        class RelationBackend:
            def __init__(self):
                self.responses = iter(
                    [
                        '{"relation": "SAME_EPISODE", "confidence": 0.4}',
                        '{"relation": "CAUSES", "confidence": 0.9}',
                    ]
                )

            def generate(self, prompt):
                return next(self.responses)

        stats = classify_event_relations(
            bank, pairs, RelationBackend(), min_confidence=0.7
        )
        self.assertEqual(stats["CAUSES"], 1)
        self.assertEqual(stats["NONE"], 1)
        edges = bank.memories[0].links["related"]
        self.assertEqual(
            edges,
            [
                {
                    "target": bank.memories[2].id,
                    "type": "CAUSES",
                    "confidence": 0.9,
                }
            ],
        )
        self.assertEqual(bank.memories[1].links["related"], [])


class EntityExtractionTest(unittest.TestCase):
    def test_parse_and_filter_entities(self):
        from hive_mem.entity_schema import normalize_entities as filter_entities, parse_entities_payload as parse_entities

        raw = """Here you go:
        [{"name": "Alice", "type": "person", "aliases": ["her friend Alice", "Alice"],
          "attributes": {"relation": "friend", "breed": "Maltese", "bogus_key": "x"}},
         {"name": "it", "type": "OBJECT"},
         {"name": "X", "type": "OBJECT"},
         {"name": "Lumi", "type": "ANIMAL",
          "attributes": {"species": "dog", "trait": ["intelligent", "quick learner", ""]}},
         {"name": "Alice", "type": "PERSON"},
         {"name": "happiness", "type": "EMOTION"}]"""
        entities = filter_entities(parse_entities(raw))
        # PERSON keeps 'relation', drops 'breed' (ANIMAL-only) and unknown keys.
        self.assertEqual(
            entities,
            [
                {
                    "name": "Alice",
                    "type": "PERSON",
                    "aliases": ["her friend Alice"],
                    "attributes": {"relation": "friend"},
                },
                {
                    "name": "Lumi",
                    "type": "ANIMAL",
                    "attributes": {
                        "species": "dog",
                        "trait": ["intelligent", "quick learner"],
                    },
                },
            ],
        )

    def test_executor_insert_block_carries_entities(self):
        executor = MemoryExecutor(llm_client=None, embedder=FakeEmbedder())
        block = (
            "MEMORY_ITEM: Lena's Maltese dog Lumi is a quick learner.\n"
            'ENTITIES: [{"name": "Lumi", "type": "ANIMAL", '
            '"attributes": {"breed": "Maltese", "owner": "Lena", "trait": "quick learner"}}]'
        )
        result = executor._parse_single_action(block)
        self.assertTrue(result.success)
        self.assertEqual(result.memory_content, "Lena's Maltese dog Lumi is a quick learner.")
        self.assertEqual(
            result.entities,
            [{"name": "Lumi", "type": "ANIMAL",
              "attributes": {"breed": "Maltese", "owner": "Lena", "trait": "quick learner"}}],
        )
        bank = MAUBank()
        executor.apply_to_memory_bank([result], bank, event_metadata={})
        self.assertEqual(bank.memories[0].entities[0]["name"], "Lumi")

    def test_executor_insert_with_bad_entities_keeps_memory(self):
        executor = MemoryExecutor(llm_client=None, embedder=FakeEmbedder())
        block = "MEMORY_ITEM: a fact\nENTITIES: [not valid json"
        result = executor._parse_single_action(block)
        self.assertTrue(result.success)
        self.assertEqual(result.memory_content, "a fact")
        self.assertEqual(result.entities, [])

    def test_executor_prompt_mentions_entities_when_insert_enabled(self):
        executor = MemoryExecutor(llm_client=None, embedder=FakeEmbedder())
        prompt = executor._build_prompt("chunk")
        self.assertIn("ENTITIES", prompt)
        self.assertIn("ANIMAL: species, breed", prompt)

    def test_parse_entities_rejects_garbage(self):
        from hive_mem.entity_schema import parse_entities_payload as parse_entities

        self.assertIsNone(parse_entities("no json here"))

    def test_entities_survive_save_load_roundtrip(self):
        bank = MAUBank()
        bank.add_memory("fact", np.asarray([1.0, 0.0]), metadata={"session_id": "D1"})
        bank.memories[0].entities = [{"name": "Alice", "type": "PERSON"}]
        with tempfile.TemporaryDirectory() as directory:
            bank.save(directory)
            with (Path(directory) / "memories.jsonl").open() as handle:
                row = json.loads(handle.readline())
            loaded = MAUBank.load(directory)
        self.assertEqual(row["entities"], [{"name": "Alice", "type": "PERSON"}])
        self.assertEqual(loaded.memories[0].entities, [{"name": "Alice", "type": "PERSON"}])


class EntityEdgeDerivationTest(unittest.TestCase):
    @staticmethod
    def _bank(rows):
        bank = MAUBank()
        for summary, session, entities in rows:
            bank.add_memory(
                summary,
                np.asarray([1.0, 0.0]),
                metadata={"session_id": session, "round_id": 1},
            )
            bank.memories[-1].entities = entities
        return bank

    def test_rare_shared_entity_links_across_sessions(self):
        from hive_mem.build_memory_edges import derive_entity_pairs

        bank = self._bank(
            [
                ("bought medicine", "D1", [{"name": "allergy medicine", "type": "OBJECT"}]),
                ("weather chat", "D2", [{"name": "Beijing", "type": "PLACE"}]),
                ("reaction to medicine", "D9", [{"name": "allergy medicine", "type": "OBJECT"}]),
            ]
        )
        pairs = derive_entity_pairs(bank, df_max=0.7, df_stop=0.9)
        self.assertEqual(pairs, [(0, 2)])

    def test_stoplisted_entity_never_links(self):
        from hive_mem.build_memory_edges import derive_entity_pairs

        entity = [{"name": "Lena", "type": "PERSON"}]
        bank = self._bank(
            [(f"m{i}", f"D{i + 1}", list(entity)) for i in range(4)]
        )
        # df = 1.0 > df_stop, so no pairs even though the entity is shared.
        self.assertEqual(derive_entity_pairs(bank, df_max=0.3, df_stop=0.5), [])

    def test_min_shared_entities_qualify_without_rare_entity(self):
        from hive_mem.build_memory_edges import derive_entity_pairs

        shared = [
            {"name": "Alice", "type": "PERSON"},
            {"name": "camping trip", "type": "EVENT"},
        ]
        bank = self._bank(
            [
                ("planning", "D1", list(shared)),
                ("during trip", "D5", list(shared)),
                ("other", "D3", [{"name": "library", "type": "PLACE"}]),
            ]
        )
        pairs = derive_entity_pairs(bank, df_max=0.1, df_stop=0.9, min_shared=2)
        self.assertEqual(pairs, [(0, 1)])

    def test_degree_cap_limits_pairs_per_memory(self):
        from hive_mem.build_memory_edges import derive_entity_pairs

        entity = [{"name": "shelter", "type": "PLACE"}]
        bank = self._bank(
            [(f"m{i}", f"D{i + 1}", list(entity)) for i in range(5)]
        )
        pairs = derive_entity_pairs(bank, df_max=1.0, df_stop=1.0, degree_cap=1)
        degree = {}
        for a, b in pairs:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        self.assertTrue(all(count <= 1 for count in degree.values()))

    def test_conflict_candidates_report_value_changes(self):
        from hive_mem.build_memory_edges import find_conflict_candidates

        bank = self._bank(
            [
                (
                    "sister likes Standard Poodles",
                    "D1",
                    [{"name": "Lena's sister", "type": "PERSON",
                      "attributes": {"preference": "Standard Poodle"}}],
                ),
                (
                    "actually she likes Schnauzers",
                    "D6",
                    [{"name": "Lena's sister", "type": "PERSON",
                      "attributes": {"preference": "Standard Schnauzer"}}],
                ),
            ]
        )
        conflicts = find_conflict_candidates(bank)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["earlier_value"], "Standard Poodle")
        self.assertEqual(conflicts[0]["later_value"], "Standard Schnauzer")

    def test_attribute_pairs_join_different_entities_with_shared_value(self):
        from hive_mem.build_memory_edges import derive_attribute_pairs

        bank = self._bank(
            [
                ("Lumi is intelligent", "D1",
                 [{"name": "Lumi", "type": "ANIMAL",
                   "attributes": {"trait": "intelligent", "species": "dog"}}]),
                ("weather chat", "D2", []),
                ("Coco is intelligent too", "D9",
                 [{"name": "Coco", "type": "ANIMAL",
                   "attributes": {"trait": "Intelligent", "species": "dog"}}]),
                ("a smart fridge", "D3",
                 [{"name": "smart fridge", "type": "OBJECT",
                   "attributes": {"use": "intelligent"}}]),
            ]
        )
        pairs = derive_attribute_pairs(bank, df_max=0.6, df_stop=0.9)
        # Lumi & Coco join via (ANIMAL, trait, intelligent) — case-insensitive;
        # the OBJECT with value "intelligent" does NOT join (type-scoped buckets).
        self.assertIn((0, 2), pairs)
        self.assertTrue(all({a, b} != {0, 3} and {a, b} != {2, 3} for a, b in pairs))

    def test_entity_pairs_feed_event_relation_candidates(self):
        from hive_mem.build_memory_edges import (
            derive_entity_pairs,
            generate_candidate_pairs,
        )

        bank = self._bank(
            [
                ("bought medicine", "D1", [{"name": "allergy medicine", "type": "OBJECT"}]),
                ("weather chat", "D2", []),
                ("reaction to medicine", "D9", [{"name": "allergy medicine", "type": "OBJECT"}]),
            ]
        )
        entity_pairs = derive_entity_pairs(bank, df_max=0.7, df_stop=0.9)
        candidates = generate_candidate_pairs(
            bank, session_window=1, entity_pairs=entity_pairs
        )
        # (0, 2) spans sessions D1 -> D9: only reachable via the entity pair.
        self.assertIn((0, 2), candidates)


class GraphExpandedRetrievalTest(unittest.TestCase):
    @staticmethod
    def _build_index_dir(directory, graph=True, **options):
        from hive_mem.retriever import GraphExpandedIndex
        from hive_mem.retriever import SimpleMemoryIndex

        bank = MAUBank()
        rows = [
            ("seed memory", [1.0, 0.0], "D1"),
            ("linked but dissimilar", [0.1, 1.0], "D9"),
            ("mildly similar filler", [0.5, 0.5], "D2"),
            ("other filler", [0.45, 0.55], "D3"),
        ]
        for summary, vector, session in rows:
            bank.add_memory(
                summary,
                np.asarray(vector, dtype=np.float32),
                metadata={"session_id": session, "round_id": 1},
            )
        # Typed edge connecting the strong seed to the dissimilar memory.
        bank.memories[0].links["related"].append(
            {"target": bank.memories[1].id, "type": "CAUSES", "confidence": 0.9}
        )
        bank.save(directory)
        if graph:
            return GraphExpandedIndex(directory, **options)
        return SimpleMemoryIndex(directory)

    def test_graph_expansion_pulls_in_linked_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self._build_index_dir(directory, graph=False)
            base_hits = baseline.search([1.0, 0.0], top_k=2)
            self.assertEqual(
                [hit.item.summary for hit in base_hits],
                ["seed memory", "mildly similar filler"],
            )

            index = self._build_index_dir(
                directory, expansion_bonus=0.7, expand_entity=False
            )
            hits = index.search([1.0, 0.0], top_k=2)
            self.assertEqual(hits[0].item.summary, "seed memory")
            self.assertEqual(hits[0].via, "vector")
            self.assertEqual(hits[1].item.summary, "linked but dissimilar")
            self.assertEqual(hits[1].via, "graph")

    def test_zero_bonus_and_no_expansion_matches_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            index = self._build_index_dir(
                directory,
                expansion_bonus=0.0,
                expand_temporal=False,
                expand_related=False,
                expand_entity=False,
            )
            hits = index.search([1.0, 0.0], top_k=2)
            self.assertEqual(
                [hit.item.summary for hit in hits],
                ["seed memory", "mildly similar filler"],
            )
            self.assertTrue(all(hit.via == "vector" for hit in hits))

    def test_entity_edges_expand_across_sessions(self):
        from hive_mem.retriever import GraphExpandedIndex

        with tempfile.TemporaryDirectory() as directory:
            bank = MAUBank()
            for summary, vector, session in [
                ("seed about medicine", [1.0, 0.0], "D1"),
                ("far session medicine memory", [0.0, 1.0], "D9"),
                ("noise", [0.4, 0.6], "D2"),
            ]:
                bank.add_memory(
                    summary,
                    np.asarray(vector, dtype=np.float32),
                    metadata={"session_id": session, "round_id": 1},
                )
            for position in (0, 1):
                bank.memories[position].entities = [
                    {"name": "allergy medicine", "type": "OBJECT"}
                ]
            bank.save(directory)
            index = GraphExpandedIndex(
                directory,
                expansion_bonus=1.0,
                expand_temporal=False,
                expand_related=False,
                df_max=0.9,
                df_stop=0.95,
            )
            hits = index.search([1.0, 0.0], top_k=2)
            self.assertEqual(hits[1].item.summary, "far session medicine memory")
            self.assertEqual(hits[1].via, "graph")

    def test_append_mode_keeps_vector_topk_and_appends_neighbours(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = self._build_index_dir(directory, graph=False)
            base_hits = baseline.search([1.0, 0.0], top_k=2)
            index = self._build_index_dir(
                directory,
                mode="append",
                append_k=1,
                expansion_bonus=0.5,
                expand_entity=False,
            )
            hits = index.search([1.0, 0.0], top_k=2)
            # First top_k identical to the pure vector ranking.
            self.assertEqual(
                [h.item.summary for h in hits[:2]],
                [h.item.summary for h in base_hits],
            )
            self.assertTrue(all(h.via == "vector" for h in hits[:2]))
            # Appended neighbour follows with via=graph and rank 3.
            self.assertEqual(len(hits), 3)
            self.assertEqual(hits[2].via, "graph")
            self.assertEqual(hits[2].rank, 3)
            self.assertEqual(hits[2].item.summary, "linked but dissimilar")

    def test_category_gating_uses_plain_vector_for_other_categories(self):
        from benchmarks.baseline_runtime.adapters.hivemem import HiveMemAdapter
        from benchmarks.baseline_runtime.protocol import RetrievalRequest, result_trace_rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "datasets" / "toy"
            self._build_index_dir(dataset_dir, graph=False)  # writes the bank
            adapter = HiveMemAdapter(
                baseline="HiveMem",
                source_root=Path(),
                config={
                    "index_root": str(root),
                    "top_k": 2,
                    "graph_options": {
                        "expansion_bonus": 0.7,
                        "expand_entity": False,
                        "categories": {"TR"},
                    },
                },
            )
            adapter.reset("toy", Path())
            request = lambda category: RetrievalRequest(
                query_id="q",
                text="q",
                category=category,
                top_k=2,
                query_vector=[1.0, 0.0],
            )
            self.assertIn(
                "graph",
                {row["via"] for row in result_trace_rows(adapter.retrieve(request("TR")))},
            )
            self.assertEqual(
                {row["via"] for row in result_trace_rows(adapter.retrieve(request("VS")))},
                {"vector"},
            )

    def test_archived_memories_never_enter_expansion(self):
        from hive_mem.retriever import GraphExpandedIndex

        with tempfile.TemporaryDirectory() as directory:
            bank = MAUBank()
            bank.add_memory("old fact", np.asarray([0.9, 0.1]), metadata={"session_id": "D1"})
            bank.add_memory("seed", np.asarray([1.0, 0.0]), metadata={"session_id": "D2"})
            bank.memories[0].status = "ARCHIVED"
            bank.memories[1].links["related"].append(
                {"target": bank.memories[0].id, "type": "CAUSES", "confidence": 0.9}
            )
            bank.save(directory)
            index = GraphExpandedIndex(directory, expansion_bonus=1.0)
            hits = index.search([1.0, 0.0], top_k=3)
            summaries = [hit.item.summary for hit in hits]
            self.assertNotIn("old fact", summaries)


class MemoryEventMetadataTest(unittest.TestCase):
    def test_metadata_includes_singular_keys_for_answer_prompt_formatter(self):
        event = MemoryEvent(
            text="hello",
            dataset="d",
            dialogue_id="D1:3",
            session_id="D1",
            round_id=3,
            source_chunk_id="c3",
            image_ids=["D1:IMG_001"],
        )
        metadata = event.metadata
        self.assertEqual(metadata["dialogue_id"], "D1:3")
        self.assertEqual(metadata["image_id"], "D1:IMG_001")


class AnswerPromptImageIdLeakTest(unittest.TestCase):
    def setUp(self):
        self.client = VLMAnswerClient()
        self.memory = {
            "text": "A robot painting at an easel (image_id: D6:IMG_001).",
            "image": None,
            "metadata": {
                "session_id": "D6",
                "dialogue_id": "D6:1",
                "image_id": "D6:IMG_001",
            },
        }

    def test_unattached_memory_id_is_redacted_for_visual_question(self):
        text, paths = self.client._build_text_and_image_paths(
            [self.memory], "question", {"path": "/tmp/query.png"}, "VS"
        )
        self.assertNotIn("D6:IMG_001", text)
        self.assertIn("[IMAGE_ID_REDACTED]", text)
        self.assertEqual(paths, ["/tmp/query.png"])

    def test_attached_memory_image_keeps_its_candidate_id(self):
        memory = {**self.memory, "image": {"path": "/tmp/memory.png", "img_id": "D6:IMG_001"}}
        text, paths = self.client._build_text_and_image_paths(
            [memory], "question", {"path": "/tmp/query.png"}, "VS"
        )
        self.assertIn("IMG:D6:IMG_001", text)
        self.assertIn("Attached memory image 1: D6:IMG_001", text)
        self.assertEqual(paths, ["/tmp/memory.png", "/tmp/query.png"])

    def test_ttl_question_attaches_memory_image_and_keeps_candidate_id(self):
        memory = {**self.memory, "image": {"path": "/tmp/memory.png", "img_id": "D6:IMG_001"}}
        text, paths = self.client._build_text_and_image_paths([memory], "question", None, "TTL")
        self.assertIn("IMG:D6:IMG_001", text)
        self.assertIn("Attached memory image 1: D6:IMG_001", text)
        self.assertEqual(paths, ["/tmp/memory.png"])

    def test_nonvisual_question_does_not_expose_unattached_candidate_id(self):
        memory = {**self.memory, "image": {"path": "/tmp/memory.png", "img_id": "D6:IMG_001"}}
        text, paths = self.client._build_text_and_image_paths([memory], "question", None, "FR")
        self.assertNotIn("D6:IMG_001", text)
        self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
