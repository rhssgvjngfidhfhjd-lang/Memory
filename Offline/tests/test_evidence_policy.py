from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests
import torch

from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient
from benchmarks.memgallery_harness.runner.metrics import calculate_usage_cost
from benchmarks.memgallery_harness.retrieval.query_embedding_cache import (
    QueryEmbeddingCache,
    make_query_id,
)
from evidence_policy.evidence import (
    DialogueStore,
    EvidenceChainBuilder,
    EvidenceStrategy,
    EvidenceType,
    MAUEvidenceAction,
    choose_baseline_actions,
    make_policy_observation,
)
from evidence_policy.policy import EvidenceSelectionPolicy
from evidence_policy.ppo import PPOBuffer, PPOTrainer, SlidingCostNormalizer
from evidence_policy.retrieval import (
    question_retrieval_seed,
    resolve_graph_options,
    resolve_retrieval_settings,
    retrieval_signature,
    retrieve_hits,
    validate_graph_config,
)
from evidence_policy.rollout import (
    EvidenceEpisode,
    EvidenceSelectionEnv,
    RolloutCache,
)
from evidence_policy.vp_store import VPArtifactIndex
from hive_mem.mau import MAU
from hive_mem.retriever import MemoryHit, SimpleMemoryIndex
from scripts.evidence_policy import (
    CostRewardRuntime,
    RealtimeWandbLogger,
    build_h2hmem_policy_messages,
    build_wma_policy_messages,
    initial_validation_signature,
    parse_h2hmem_policy_answer,
    parse_wma_policy_answer,
    prepare_initial_validation,
    reconcile_ppo_metrics_for_resume,
    resume_configs_match,
    rollout_record,
    rollout_with_endpoint_recovery,
    validation_checkpoints,
)


EMBEDDING_DIM = 8


class CostRewardTest(unittest.TestCase):
    def test_exact_usage_cost_formula(self):
        self.assertAlmostEqual(
            calculate_usage_cost(
                {
                    "prompt_tokens": 200,
                    "completion_tokens": 40,
                    "total_tokens": 240,
                },
                input_price=0.05,
                output_price=0.25,
            ),
            0.00002,
        )

    def test_sliding_normalizer_warmup_snapshot_floor_and_round_trip(self):
        normalizer = SlidingCostNormalizer(
            window_size=4,
            min_window_size=2,
            lower_quantile=0.05,
            upper_quantile=0.95,
            initial_range_floor_ratio=0.25,
        )
        warmup = normalizer.snapshot()
        self.assertFalse(warmup["active"])
        self.assertEqual(normalizer.normalize(10.0, snapshot=warmup), 0.0)

        normalizer.extend([1.0, 4.0], [0.2, 0.8])
        fixed = normalizer.snapshot()
        self.assertTrue(fixed["active"])
        self.assertEqual(normalizer.normalize(1.0, snapshot=fixed), 0.0)
        self.assertEqual(normalizer.normalize(4.0, snapshot=fixed), 1.0)
        expected = (1.5 - fixed["effective_transformed_cost_min"]) / fixed[
            "effective_transformed_cost_range"
        ]
        self.assertAlmostEqual(fixed["task_reward_std"], 0.3)
        self.assertAlmostEqual(fixed["normalized_cost_std"], 0.5)
        self.assertAlmostEqual(
            fixed["cost_scale_alpha"],
            fixed["task_reward_std"]
            / (fixed["normalized_cost_std"] + normalizer.std_epsilon),
        )
        normalizer.extend([400.0, 900.0], [0.1, 0.9])
        self.assertAlmostEqual(
            normalizer.normalize(2.25, snapshot=fixed), expected
        )

        initial_range = (
            normalizer.initial_transformed_cost_max
            - normalizer.initial_transformed_cost_min
        )
        normalizer.extend([0.0, 0.0, 0.0, 0.0], [0.4, 0.4, 0.4, 0.4])
        collapsed = normalizer.snapshot()
        self.assertGreaterEqual(
            collapsed["effective_transformed_cost_range"], 0.25 * initial_range
        )
        self.assertEqual(list(normalizer.transformed_costs), [0.0] * 4)
        self.assertEqual(list(normalizer.quality_rewards), [0.4] * 4)
        restored = SlidingCostNormalizer(
            window_size=4,
            min_window_size=2,
            lower_quantile=0.05,
            upper_quantile=0.95,
            initial_range_floor_ratio=0.25,
        )
        restored.load_state_dict(normalizer.state_dict())
        self.assertEqual(restored.state_dict(), normalizer.state_dict())

    def test_cost_runtime_uses_frozen_window_and_alpha_zero_is_f1_only(self):
        def config(alpha):
            return {
                "model": {"name": "Qwen/Qwen3-VL-4B-Instruct"},
                "efficiency_config": str(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "model_efficiency.json"
                ),
                "reward": {
                    "cost_enabled": True,
                    "cost_tradeoff_lambda": alpha,
                    "cost_transform": "sqrt_incremental",
                    "window_size": 4,
                    "min_window_size": 2,
                    "lower_quantile": 0.05,
                    "upper_quantile": 0.95,
                    "initial_range_floor_ratio": 0.25,
                    "range_epsilon": 1e-12,
                    "std_epsilon": 1e-8,
                },
            }

        def build_messages(items, *, allow_empty_evidence=False):
            if not items and not allow_empty_evidence:
                raise ValueError("empty evidence is disabled")
            evidence = (
                f"Conversation memory:\n[Evidence 1]\n{items[0]['text']}\n\n"
                if items
                else ""
            )
            return [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"{evidence}Question: question"},
            ]

        episode = SimpleNamespace(
            query_id="q1",
            answer_messages_builder=build_messages,
            query_image=None,
        )

        def rollout():
            return SimpleNamespace(
                reward=0.8,
                quality_reward=0.8,
                error="",
                answer_usage={
                    "prompt_tokens": 300,
                    "completion_tokens": 20,
                    "total_tokens": 320,
                },
                cost_weight=None,
                cost_error="",
            )

        counter = MagicMock()
        counter.count.return_value = 100
        runtime = CostRewardRuntime(config(0.1), token_counter=counter)
        first = rollout()
        warmup_snapshot = runtime.snapshot()
        self.assertTrue(runtime.apply(first, episode, snapshot=warmup_snapshot))
        self.assertNotIn(
            "Conversation memory:", counter.count.call_args.args[0][1]["content"]
        )
        self.assertEqual(first.reward, first.quality_reward)
        self.assertIsNone(first.cost_min)
        self.assertIsNone(first.cost_max)
        runtime.commit(
            [first.incremental_cost, 0.000035],
            [first.quality_reward, 0.4],
        )

        second = rollout()
        second.answer_usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "total_tokens": 1020,
        }
        active_snapshot = runtime.snapshot()
        self.assertTrue(runtime.apply(second, episode, snapshot=active_snapshot))
        self.assertAlmostEqual(second.normalized_cost, 1.0)
        expected_scale = active_snapshot["task_reward_std"] / (
            active_snapshot["normalized_cost_std"] + 1e-8
        )
        self.assertAlmostEqual(second.cost_scale_alpha, expected_scale)
        self.assertAlmostEqual(
            second.reward,
            second.quality_reward - 0.1 * expected_scale,
        )
        self.assertAlmostEqual(
            second.transformed_cost,
            np.sqrt(second.incremental_cost),
        )
        self.assertAlmostEqual(
            second.cost_min,
            second.base_cost
            + active_snapshot["effective_transformed_cost_min"] ** 2,
        )
        self.assertGreater(second.cost_max, second.cost_min)

        no_penalty = CostRewardRuntime(config(0.0), token_counter=counter)
        no_penalty.commit(
            [first.incremental_cost, 0.000035],
            [first.quality_reward, 0.4],
        )
        third = rollout()
        third.answer_usage = second.answer_usage
        self.assertTrue(no_penalty.apply(third, episode, snapshot=no_penalty.snapshot()))
        self.assertEqual(third.reward, third.quality_reward)

        missing = rollout()
        missing.answer_usage = None
        self.assertFalse(runtime.apply(missing, episode, snapshot=active_snapshot))
        self.assertIn("missing exact", missing.cost_error)


class ValidationScheduleTest(unittest.TestCase):
    def test_answer_client_reduces_image_resolution_only_after_context_overflow(self):
        client = VLMAnswerClient(base_url="http://127.0.0.1:1/v1")
        response = {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }
        with patch.object(
            client,
            "_build_openai_content",
            return_value=[{"type": "text", "text": "prompt"}],
        ) as build_content, patch.object(
            client,
            "_post_json",
            side_effect=[
                requests.HTTPError("decoder prompt exceeds maximum model length"),
                response,
            ],
        ):
            answer, usage = client._answer_openai_compatible(
                system_prompt="system",
                memory_items=[],
                question_prompt="question",
            )

        self.assertEqual(answer, "answer")
        self.assertEqual(usage["total_tokens"], 11)
        self.assertEqual(
            [call.kwargs["max_image_side"] for call in build_content.call_args_list],
            [1344, 896],
        )

    def test_query_cache_falls_back_across_workspace_path_moves(self):
        vector = np.arange(EMBEDDING_DIM, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            np.save(cache_dir / "vectors.npy", vector.reshape(1, -1))
            old_image = {"path": "/old/workspace/image.jpg", "caption": "same image"}
            metadata = {
                "query_id": make_query_id(
                    dataset_name="dataset",
                    qa_index=1,
                    category="VS",
                    question="Where is it?",
                    query_image=old_image,
                ),
                "dataset": "dataset",
                "qa_index": 1,
                "category": "VS",
                "question": "Where is it?",
            }
            (cache_dir / "metadata.jsonl").write_text(
                json.dumps(metadata) + "\n", encoding="utf-8"
            )
            (cache_dir / "manifest.json").write_text(
                json.dumps({"count": 1, "dim": EMBEDDING_DIM}), encoding="utf-8"
            )

            cache = QueryEmbeddingCache(cache_dir, expected_dim=EMBEDDING_DIM)
            actual = cache.get(
                dataset_name="dataset",
                qa_index=1,
                category="VS",
                question="Where is it?",
                query_image={"path": "/new/workspace/image.jpg", "caption": "same image"},
            )

            self.assertEqual(actual, vector.tolist())

    def test_query_cache_load_is_thread_safe(self):
        vectors = np.arange(4 * EMBEDDING_DIM, dtype=np.float32).reshape(4, -1)
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            np.save(cache_dir / "vectors.npy", vectors)
            rows = [
                {
                    "query_id": f"query-{index}",
                    "dataset": "dataset",
                    "qa_index": index,
                    "category": "FR",
                    "question": f"Question {index}",
                }
                for index in range(len(vectors))
            ]
            (cache_dir / "metadata.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (cache_dir / "manifest.json").write_text(
                json.dumps({"count": len(vectors), "dim": EMBEDDING_DIM}),
                encoding="utf-8",
            )
            cache = QueryEmbeddingCache(cache_dir, expected_dim=EMBEDDING_DIM)
            barrier = threading.Barrier(8)
            real_load = np.load

            def slow_load(*args, **kwargs):
                time.sleep(0.05)
                return real_load(*args, **kwargs)

            def lookup(_: int):
                barrier.wait()
                return cache.get_by_id("query-2")

            with patch(
                "benchmarks.memgallery_harness.retrieval.query_embedding_cache.np.load",
                side_effect=slow_load,
            ) as load_vectors, ThreadPoolExecutor(max_workers=8) as pool:
                actual = list(pool.map(lookup, range(8)))

            self.assertEqual(load_vectors.call_count, 1)
            self.assertEqual(actual, [vectors[2].tolist()] * 8)

    def test_wandb_init_failure_is_persisted_without_raising(self):
        fake_wandb = MagicMock()
        fake_wandb.util.generate_id.return_value = "test-run-id"
        fake_wandb.init.side_effect = RuntimeError("network unavailable")
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "sys.modules", {"wandb": fake_wandb}
        ):
            logger = RealtimeWandbLogger(
                enabled=True,
                output_dir=Path(directory),
                config={"benchmark": "wma"},
                project="test-project",
                entity="test-entity",
                name="test-run",
            )

            self.assertIsNone(logger.run)
            errors = [
                json.loads(line)
                for line in (Path(directory) / "run_control" / "wandb_errors.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(errors[0]["stage"], "init")
            self.assertIn("network unavailable", errors[0]["message"])

    def test_resume_allows_only_output_directory_to_change(self):
        stored = {"seed": 42, "output_dir": "/old", "ppo": {"epochs": 6}}
        current = {"seed": 42, "output_dir": "/new", "ppo": {"epochs": 6}}

        self.assertTrue(resume_configs_match(stored, current))
        current["seed"] = 43
        self.assertFalse(resume_configs_match(stored, current))

    def test_resume_discards_uncommitted_and_duplicate_ppo_metrics(self):
        rows = [
            {"update_step": 1, "reward_mean": 0.1},
            {"update_step": 2, "reward_mean": 0.2},
            {"update_step": 3, "reward_mean": 0.3},
            {"update_step": 2, "reward_mean": 0.25},
            {"update_step": 4, "reward_mean": 0.4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ppo_metrics.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            result = reconcile_ppo_metrics_for_resume(
                path,
                checkpoint_update_step=2,
            )

            kept = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["update_step"] for row in kept], [1, 2])
            self.assertEqual(kept[-1]["reward_mean"], 0.25)
            self.assertEqual(result["original_rows"], 5)
            self.assertEqual(result["kept_rows"], 2)
            self.assertEqual(result["removed_rows"], 3)
            self.assertTrue(Path(result["backup"]).is_file())

    def test_transient_endpoint_error_is_retried_without_zero_reward(self):
        failed = MagicMock(error="Connection refused", reward=0.0)
        successful = MagicMock(error="", reward=0.75)
        env = MagicMock()
        env.rollout.side_effect = [failed, successful]
        episode = MagicMock(query_id="q1")

        with patch("scripts.evidence_policy.time.sleep") as sleep:
            result = rollout_with_endpoint_recovery(
                env,
                episode,
                EvidenceStrategy.SUMMARY,
                policy=None,
                deterministic=True,
                attempts=2,
                delay_seconds=0.01,
            )

        self.assertIs(result, successful)
        self.assertEqual(env.rollout.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_half_epoch_aligns_to_completed_rollout_batch(self):
        self.assertEqual(
            validation_checkpoints(
                1026, interval_fraction=0.5, rollout_batch_size=32
            ),
            {512: "half"},
        )

    def test_epoch_only_schedule_has_no_midpoint(self):
        self.assertEqual(
            validation_checkpoints(
                1026, interval_fraction=1.0, rollout_batch_size=32
            ),
            {},
        )

    def test_initial_validation_is_persisted_and_reused(self):
        config = {"seed": 42, "ppo": {"validation_limit": 20}}
        event = {
            "phase": "initial",
            "update_step": 0,
            "train_question_count": 0,
            "metrics": {"count": 20, "mean_reward": 0.5},
            "rollouts": "initial_rollouts.jsonl",
        }
        trainer = MagicMock()
        trainer.update_steps = 0
        torch.manual_seed(1234)
        rng_state = torch.random.get_rng_state().clone()

        def stochastic_validation(*args, **kwargs):
            torch.rand(8)
            return dict(event)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch(
                "scripts.evidence_policy.run_training_validation",
                side_effect=stochastic_validation,
            ) as run_validation:
                first = prepare_initial_validation(
                    config,
                    MagicMock(),
                    MagicMock(),
                    {},
                    MagicMock(),
                    trainer,
                    output_dir=output,
                    device=torch.device("cpu"),
                    enabled=True,
                )
            metrics_path = output / "validation" / "initial_metrics.json"
            self.assertTrue(metrics_path.is_file())
            self.assertEqual(first["run_signature"], initial_validation_signature(config, "cpu"))
            self.assertEqual(first["update_step"], 0)
            run_validation.assert_called_once()
            self.assertFalse(run_validation.call_args.kwargs["deterministic"])
            self.assertEqual(first["sampling_mode"], "independent_bernoulli")
            self.assertEqual(first["initial_action_probability"], 0.5)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_state))
            trainer.save_checkpoint.assert_called_once()

            with patch(
                "scripts.evidence_policy.run_training_validation",
                side_effect=AssertionError("baseline must be reused"),
            ):
                second = prepare_initial_validation(
                    config,
                    MagicMock(),
                    MagicMock(),
                    {},
                    MagicMock(),
                    trainer,
                    output_dir=output,
                    device=torch.device("cpu"),
                    enabled=True,
                )
            self.assertEqual(second, first)


class GraphRetrievalConfigTest(unittest.TestCase):
    def test_graph_defaults_are_five_plus_two_append(self):
        options = resolve_graph_options({"top_k": 5})

        self.assertIsNotNone(options)
        self.assertEqual(options["mode"], "append")
        self.assertEqual(options["append_k"], 2)
        self.assertEqual(options["seed_k"], 0)
        validate_graph_config({"top_k": 5})

    def test_graph_five_plus_two_contract_is_validated(self):
        with self.assertRaisesRegex(ValueError, "top_k=5"):
            validate_graph_config({"top_k": 4})
        with self.assertRaisesRegex(ValueError, "append_k=2"):
            resolve_graph_options({"top_k": 5, "graph_options": {"append_k": 1}})

    def test_ablation_modes_resolve_without_changing_legacy_defaults(self):
        self.assertEqual(
            resolve_retrieval_settings(
                {"retrieval_mode": "vector", "top_k": 7, "seed": 42}
            )["vector_k"],
            7,
        )
        random_settings = resolve_retrieval_settings(
            {
                "retrieval_mode": "random_append",
                "top_k": 5,
                "random_append_k": 2,
                "retrieval_seed": 43,
            }
        )
        self.assertEqual(random_settings["append_k"], 2)
        self.assertIsNone(random_settings["graph_options"])

    def test_random_append_is_question_deterministic_unique_and_auditable(self):
        items = [
            MAU(
                id=f"m{index}",
                summary=f"memory {index}",
                embedding=np.full(EMBEDDING_DIM, index + 1, dtype=np.float32),
                metadata={"session_id": "visible"},
            )
            for index in range(10)
        ]
        index = SimpleMemoryIndex.__new__(SimpleMemoryIndex)
        index.bank = SimpleNamespace(memories=items)
        index.search = MagicMock(
            return_value=[
                MemoryHit(item=item, score=1.0 - rank / 100, rank=rank)
                for rank, item in enumerate(items[:5], start=1)
            ]
        )
        index._scores = MagicMock(return_value=np.linspace(1.0, 0.1, 10))
        settings = resolve_retrieval_settings(
            {
                "retrieval_mode": "random_append",
                "top_k": 5,
                "random_append_k": 2,
                "retrieval_seed": 42,
            }
        )

        first, first_metadata = retrieve_hits(
            index,
            np.ones(EMBEDDING_DIM),
            settings,
            benchmark="wma",
            manifest_question_id="sample:QA00:Q001",
            allowed_session_ids={"visible"},
        )
        second, second_metadata = retrieve_hits(
            index,
            np.ones(EMBEDDING_DIM),
            settings,
            benchmark="wma",
            manifest_question_id="sample:QA00:Q001",
            allowed_session_ids={"visible"},
        )

        self.assertEqual(
            [hit.item.id for hit in first], [hit.item.id for hit in second]
        )
        self.assertEqual([hit.rank for hit in first], list(range(1, 8)))
        self.assertEqual([hit.via for hit in first], ["vector"] * 5 + ["random"] * 2)
        self.assertEqual(len(set(first_metadata["retrieval_final_ids"])), 7)
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["append_k_actual"], 2)
        score_args = index._scores.call_args.args
        np.testing.assert_array_equal(score_args[0], np.ones(EMBEDDING_DIM))
        self.assertEqual(score_args[1:], ("", {"visible"}))

    def test_question_seed_and_retrieval_signatures_are_isolated(self):
        self.assertEqual(
            question_retrieval_seed(42, "wma", "q1"),
            question_retrieval_seed(42, "wma", "q1"),
        )
        self.assertNotEqual(
            question_retrieval_seed(42, "wma", "q1"),
            question_retrieval_seed(43, "wma", "q1"),
        )
        vector = retrieval_signature(
            ".", None, retrieval_mode="vector", vector_k=5
        )
        top7 = retrieval_signature(
            ".", None, retrieval_mode="vector", vector_k=7
        )
        random42 = retrieval_signature(
            ".",
            None,
            retrieval_mode="random_append",
            vector_k=5,
            append_k=2,
            retrieval_seed=42,
        )
        random43 = retrieval_signature(
            ".",
            None,
            retrieval_mode="random_append",
            vector_k=5,
            append_k=2,
            retrieval_seed=43,
        )
        self.assertEqual(len({vector, top7, random42, random43}), 4)


def make_hit(
    memory_id: str,
    *,
    summary: str = "summary fact",
    dialogue_id: str = "D1:1",
    image_path: str = "",
    caption: str = "",
) -> MemoryHit:
    metadata = {
        "session_id": "D1",
        "dialogue_id": dialogue_id,
        "source_dialogue_ids": [dialogue_id],
        "image_id": "D1:IMG_001" if image_path else "",
        "image_ids": ["D1:IMG_001"] if image_path else [],
        "image_paths": [image_path] if image_path else [],
        "image_captions": [caption] if caption else [],
    }
    item = MAU(
        id=memory_id,
        summary=summary,
        embedding=np.linspace(0.1, 0.8, EMBEDDING_DIM, dtype=np.float32),
        metadata=metadata,
    )
    return MemoryHit(item=item, score=0.9, rank=1)


def write_dialogue_dataset(root: Path, dataset: str = "toy") -> None:
    dialog_dir = root / "dialog"
    dialog_dir.mkdir(parents=True)
    payload = {
        "multi_session_dialogues": [
            {
                "dialogues": [
                    {
                        "round": "D1:1",
                        "user": "What should I bake?",
                        "assistant": "Bake a tart.",
                    }
                ]
            }
        ]
    }
    (dialog_dir / f"{dataset}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def write_vp_run(root: Path, source_image: Path) -> Path:
    run = root / "vp_run"
    crop = run / "items" / "img_test" / "vp_0001.jpg"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"vp crop")
    (run / "exports").mkdir()
    (run / "run.json").write_text(
        json.dumps({"schema_version": "1.0", "run_id": "test"}), encoding="utf-8"
    )
    record = {
        "schema_version": "1.0",
        "run_id": "test",
        "image_id": "img_test",
        "source": {
            "dataset": "Mem-Gallery",
            "relative_path": source_image.name,
            "sha256": "",
        },
        "status": "success",
        "primitives": [
            {
                "vp_id": "img_test_vp_0001",
                "label": "subject",
                "bbox_norm": [0, 0, 500, 500],
                "bbox_px": [0, 0, 5, 5],
                "crop_path": "items/img_test/vp_0001.jpg",
            }
        ],
    }
    (run / "exports" / "images.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    return run


class EvidenceChainTest(unittest.TestCase):
    def test_builds_selected_dialogue_and_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"not decoded by the chain builder")
            hit = make_hit(
                "m1", image_path=str(image_path), caption="A fruit tart."
            )
            action = MAUEvidenceAction(
                "m1", frozenset({EvidenceType.DIALOGUE, EvidenceType.IMAGE})
            )
            items = EvidenceChainBuilder(DialogueStore(root)).build(
                "toy", "VS", [hit], [action]
            )

        self.assertIn("User: What should I bake?", items[0]["text"])
        self.assertEqual(items[0]["images"][0]["path"], str(image_path))
        self.assertNotIn("A fruit tart.", items[0]["text"])

    def test_caption_action_adds_caption_without_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            hit = make_hit("m1", image_path="old/image.jpg", caption="A tart.")
            action = MAUEvidenceAction(
                "m1", frozenset({EvidenceType.SUMMARY, EvidenceType.CAPTION})
            )
            items = EvidenceChainBuilder(DialogueStore(root)).build(
                "toy", "FR", [hit], [action]
            )

        self.assertIn("Summary:\nsummary fact", items[0]["text"])
        self.assertIn("Image captions:\n- A tart.", items[0]["text"])
        self.assertEqual(items[0]["images"], [])

    def test_zero_mask_drops_mau_and_image_plus_vp_attaches_both(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            image_path = root / "image.jpg"
            image_path.write_bytes(b"original")
            index = VPArtifactIndex(write_vp_run(root, image_path))
            builder = EvidenceChainBuilder(DialogueStore(root), vp_index=index)
            hit = make_hit("m1", image_path=str(image_path), caption="A tart.")
            self.assertEqual(builder.build("toy", "VS", [hit], [MAUEvidenceAction("m1")]), [])
            action = MAUEvidenceAction(
                "m1", frozenset({EvidenceType.IMAGE, EvidenceType.VP})
            )
            items = builder.build("toy", "VS", [hit], [action])

        self.assertEqual([row["kind"] for row in items[0]["images"]], ["image", "vp"])
        self.assertEqual(items[0]["text"], "")

    def test_baseline_actions_respect_visual_constraints(self):
        visual = make_hit("visual", image_path="image.jpg", caption="caption")
        text_only = make_hit("text")
        actions = choose_baseline_actions(
            [visual, text_only], "FR", EvidenceStrategy.FULL
        )
        self.assertEqual(
            actions[0].selected,
            frozenset({EvidenceType.SUMMARY, EvidenceType.DIALOGUE, EvidenceType.CAPTION}),
        )
        self.assertEqual(
            actions[1].selected,
            frozenset({EvidenceType.SUMMARY, EvidenceType.DIALOGUE}),
        )
        with self.assertRaisesRegex(ValueError, "selected unavailable evidence"):
            EvidenceChainBuilder(DialogueStore("unused")).build(
                "toy",
                "FR",
                [visual],
                [
                    MAUEvidenceAction(
                        "visual",
                        frozenset({EvidenceType.SUMMARY, EvidenceType.IMAGE}),
                    )
                ],
            )


class EvidencePolicyTest(unittest.TestCase):
    def setUp(self):
        self.hits = (make_hit("m1"), make_hit("m2"))
        self.observation = make_policy_observation(
            np.ones(EMBEDDING_DIM, dtype=np.float32), self.hits, "FR"
        )

    def test_policy_sampling_and_deterministic_actions_are_valid(self):
        policy = EvidenceSelectionPolicy(
            embedding_dim=EMBEDDING_DIM, hidden_dim=16, hidden_layers=1
        )
        sampled = policy.sample(self.observation)
        deterministic_a = policy.select_deterministic(self.observation)
        deterministic_b = policy.select_deterministic(self.observation)

        self.assertEqual(len(sampled.actions), 2)
        self.assertEqual(deterministic_a.actions, deterministic_b.actions)
        self.assertTrue(torch.isfinite(sampled.joint_log_prob))
        self.assertTrue(torch.isfinite(sampled.value))
        self.assertTrue(
            all(
                action.selected.issubset({EvidenceType.SUMMARY, EvidenceType.DIALOGUE})
                for action in sampled.actions
            )
        )

    def test_policy_starts_with_independent_half_probability_per_bit(self):
        policy = EvidenceSelectionPolicy(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=16,
            hidden_layers=1,
            initial_action_probability=0.5,
        )

        with torch.no_grad():
            logits, _ = policy._forward(self.observation)

        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))
        self.assertTrue(
            torch.equal(logits.sigmoid(), torch.full_like(logits, 0.5))
        )

    def test_initial_action_probability_sets_all_actor_logits(self):
        policy = EvidenceSelectionPolicy(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=16,
            hidden_layers=1,
            initial_action_probability=0.25,
        )

        with torch.no_grad():
            logits, _ = policy._forward(self.observation)

        self.assertTrue(
            torch.allclose(logits.sigmoid(), torch.full_like(logits, 0.25))
        )

    def test_ppo_update_changes_parameters(self):
        policy = EvidenceSelectionPolicy(
            embedding_dim=EMBEDDING_DIM, hidden_dim=16, hidden_layers=1
        )
        trainer = PPOTrainer(policy, update_epochs=1, minibatch_size=1)
        with torch.no_grad():
            step = policy.sample(self.observation)
        buffer = PPOBuffer()
        buffer.add(
            self.observation,
            step.actions,
            old_log_prob=float(step.joint_log_prob),
            old_value=float(step.value),
            reward=1.0,
        )
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        metrics = trainer.update(buffer)

        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            {
                "ppo_kl",
                "pg_loss",
                "pg_clipfrac",
                "lr",
                "grad_norm",
                "entropy_loss",
                "value_loss",
                "predicted_value_mean",
                "target_return_mean",
                "absolute_value_error",
                "explained_variance",
                "reward_mean",
                "reward_min",
                "reward_max",
                "batch_size",
            }.issubset(metrics)
        )
        self.assertEqual(metrics["batch_size"], 1.0)
        self.assertTrue(
            any(
                not torch.equal(old, new.detach())
                for old, new in zip(before, policy.parameters())
            )
        )

    def test_checkpoint_round_trip(self):
        self._checkpoint_round_trip(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_checkpoint_resume_restores_rng_state(self):
        self._checkpoint_round_trip(torch.device("cuda:0"))

    def _checkpoint_round_trip(self, device: torch.device) -> None:
        policy = EvidenceSelectionPolicy(
            embedding_dim=EMBEDDING_DIM, hidden_dim=16, hidden_layers=1
        ).to(device)
        trainer = PPOTrainer(policy, update_epochs=1, minibatch_size=1)
        observation = self.observation.to(device)
        expected = policy.select_deterministic(observation).actions
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            trainer.save_checkpoint(path, config={"test": True}, epoch=3)
            state = trainer.load_checkpoint(path)
        actual = policy.select_deterministic(observation).actions
        self.assertEqual(state["epoch"], 3)
        self.assertEqual(actual, expected)


class RolloutTest(unittest.TestCase):
    class FakeClient:
        model = "fake-vlm"
        base_url = "http://fake/v1"
        num_predict = 16
        think = False
        backend = "openai"

        def __init__(self):
            self.calls = 0

        def answer(self, **kwargs):
            self.calls += 1
            return "fruit tart"

    def test_rollout_cache_avoids_duplicate_vlm_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            client = self.FakeClient()
            env = EvidenceSelectionEnv(
                client,
                EvidenceChainBuilder(DialogueStore(root)),
                cache=RolloutCache(root / "cache.jsonl"),
            )
            episode = EvidenceEpisode(
                query_id="q1",
                dataset="toy",
                category="FR",
                question_prompt="What was baked?",
                system_prompt="Answer briefly.",
                ground_truth="fruit tart",
                query_embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
                memory_hits=(make_hit("m1"),),
            )
            first = env.rollout(episode, EvidenceStrategy.SUMMARY)
            second = env.rollout(episode, EvidenceStrategy.SUMMARY)

        self.assertEqual(client.calls, 1)
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(second.reward, 1.0)
        self.assertEqual(first.answer_attempts, 1)
        self.assertEqual(first.answer_failed_attempts, 0)
        self.assertEqual(second.answer_attempts, 1)
        self.assertEqual(second.answer_failed_attempts, 0)

    def test_retrieval_signature_separates_rollout_cache_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            client = self.FakeClient()
            env = EvidenceSelectionEnv(
                client,
                EvidenceChainBuilder(DialogueStore(root)),
                cache=RolloutCache(root / "cache.jsonl"),
            )
            episode = EvidenceEpisode(
                query_id="q1",
                dataset="toy",
                category="FR",
                question_prompt="What was baked?",
                system_prompt="Answer briefly.",
                ground_truth="fruit tart",
                query_embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
                memory_hits=(make_hit("m1"),),
                retrieval_signature="graph-v1",
            )

            first = env.rollout(episode, EvidenceStrategy.SUMMARY)
            second = env.rollout(
                replace(episode, retrieval_signature="graph-v2"),
                EvidenceStrategy.SUMMARY,
            )

        self.assertEqual(client.calls, 2)
        self.assertFalse(first.cached)
        self.assertFalse(second.cached)

    def test_prompt_signature_separates_rollout_cache_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            client = self.FakeClient()
            env = EvidenceSelectionEnv(
                client,
                EvidenceChainBuilder(DialogueStore(root)),
                cache=RolloutCache(root / "cache.jsonl"),
            )
            episode = EvidenceEpisode(
                query_id="q1",
                dataset="toy",
                category="FR",
                question_prompt="What was baked?",
                system_prompt="Answer briefly.",
                ground_truth="fruit tart",
                query_embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
                memory_hits=(make_hit("m1"),),
                prompt_signature="prompt-v1",
            )

            first = env.rollout(episode, EvidenceStrategy.SUMMARY)
            second = env.rollout(
                replace(episode, prompt_signature="prompt-v2"),
                EvidenceStrategy.SUMMARY,
            )

        self.assertEqual(client.calls, 2)
        self.assertFalse(first.cached)
        self.assertFalse(second.cached)

    def test_ppo_all_zero_uses_empty_evidence_prompt(self):
        class CapturingClient(self.FakeClient):
            def answer(self, **kwargs):
                self.calls += 1
                self.request = kwargs
                return "fruit tart"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            client = CapturingClient()
            env = EvidenceSelectionEnv(
                client, EvidenceChainBuilder(DialogueStore(root))
            )
            episode = EvidenceEpisode(
                query_id="q1",
                dataset="toy",
                category="Unimodal Precise Recall",
                question_prompt="What was baked?",
                system_prompt="",
                ground_truth="fruit tart",
                query_embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
                memory_hits=(make_hit("m1"),),
                answer_messages_builder=partial(
                    build_h2hmem_policy_messages,
                    question="What was baked?",
                    category="Unimodal Precise Recall",
                ),
                prepend_memory_context=False,
                metadata={
                    "prompt_version": "answer-prompts-custom-20260909-v1",
                    "prompt_sha256": "base-sha",
                    "ppo_empty_prompt_version": "ppo-empty-evidence-20260911-v1",
                    "ppo_empty_prompt_sha256": "empty-sha",
                },
            )
            policy = EvidenceSelectionPolicy(
                embedding_dim=EMBEDDING_DIM, hidden_dim=16, hidden_layers=1
            )
            with torch.no_grad():
                policy.evidence_head.bias.fill_(-100.0)
            rollout = env.rollout(
                episode,
                EvidenceStrategy.PPO,
                policy=policy,
                deterministic=True,
            )

        self.assertFalse(rollout.error)
        self.assertTrue(all(action.bitmask == "00000" for action in rollout.actions))
        self.assertNotIn("Conversation memory:", client.request["question_prompt"])
        self.assertIn(
            "No conversation-memory evidence was selected",
            client.request["system_prompt"],
        )
        record = rollout_record(rollout, episode)
        self.assertEqual(record["prompt_variant"], "ppo_empty_evidence")
        self.assertEqual(record["prompt_version"], "ppo-empty-evidence-20260911-v1")
        self.assertEqual(record["prompt_sha256"], "empty-sha")
        self.assertEqual(record["base_prompt_sha256"], "base-sha")

    def test_h2hmem_custom_messages_and_tag_parser_are_used_by_rollout(self):
        class CapturingClient(self.FakeClient):
            def answer_with_usage(self, **kwargs):
                self.calls += 1
                self.request = kwargs
                return SimpleNamespace(
                    text="<answer>fruit tart</answer>",
                    usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                    attempts=1,
                    failed_attempts=0,
                    image_count=0,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            client = CapturingClient()
            env = EvidenceSelectionEnv(
                client, EvidenceChainBuilder(DialogueStore(root))
            )
            episode = EvidenceEpisode(
                query_id="q1",
                dataset="toy",
                category="Unimodal Precise Recall",
                question_prompt="What was baked?",
                system_prompt="",
                ground_truth="fruit tart",
                query_embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
                memory_hits=(make_hit("m1"),),
                answer_messages_builder=partial(
                    build_h2hmem_policy_messages,
                    question="What was baked?",
                    category="Unimodal Precise Recall",
                ),
                answer_parser=parse_h2hmem_policy_answer,
                prepend_memory_context=False,
                prompt_signature="h2-custom",
            )

            rollout = env.rollout(episode, EvidenceStrategy.SUMMARY)

        self.assertEqual(rollout.answer, "fruit tart")
        self.assertEqual(rollout.reward, 1.0)
        self.assertEqual(rollout.raw_answer, "<answer>fruit tart</answer>")
        self.assertIn("memory testing system", client.request["system_prompt"])
        self.assertFalse(client.request["prepend_memory_context"])
        self.assertEqual(client.request["question_prompt"].count("summary fact"), 1)

    def test_wma_custom_messages_and_tag_parser_are_used_by_rollout(self):
        class CapturingClient(self.FakeClient):
            def answer_with_usage(self, **kwargs):
                self.calls += 1
                self.request = kwargs
                return SimpleNamespace(
                    text="<answer>fruit tart</answer>",
                    usage={"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                    attempts=1,
                    failed_attempts=0,
                    image_count=0,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            client = CapturingClient()
            env = EvidenceSelectionEnv(
                client, EvidenceChainBuilder(DialogueStore(root))
            )
            episode = EvidenceEpisode(
                query_id="q1",
                dataset="toy",
                category="TR",
                question_prompt="What was baked?",
                system_prompt="official WMA system prompt",
                ground_truth="fruit tart",
                query_embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
                memory_hits=(make_hit("m1"),),
                answer_messages_builder=partial(
                    build_wma_policy_messages,
                    question="What was baked?",
                    category="TR",
                ),
                answer_parser=parse_wma_policy_answer,
                prepend_memory_context=False,
                prompt_signature="wma-custom",
            )

            rollout = env.rollout(episode, EvidenceStrategy.SUMMARY)

        self.assertEqual(rollout.answer, "fruit tart")
        self.assertEqual(rollout.reward, 1.0)
        self.assertEqual(rollout.raw_answer, "<answer>fruit tart</answer>")
        self.assertFalse(client.request["prepend_memory_context"])
        self.assertEqual(client.request["question_prompt"].count("summary fact"), 1)

    def test_rollout_record_preserves_vector_and_graph_provenance(self):
        client = self.FakeClient()
        env = EvidenceSelectionEnv(
            client,
            EvidenceChainBuilder(DialogueStore(".")),
        )
        vector_hit = make_hit("vector")
        graph_base = make_hit("graph")
        graph_hit = MemoryHit(
            item=graph_base.item,
            score=0.5,
            rank=2,
            via="graph",
        )
        episode = EvidenceEpisode(
            query_id="q1",
            dataset="toy",
            category="FR",
            question_prompt="What was baked?",
            system_prompt="Answer briefly.",
            ground_truth="fruit tart",
            query_embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
            memory_hits=(vector_hit, graph_hit),
            retrieval_signature="retrieval-signature",
        )

        rollout = env.rollout(episode, EvidenceStrategy.SUMMARY)
        row = rollout_record(rollout, episode)

        self.assertEqual(row["retrieval_signature"], "retrieval-signature")
        self.assertEqual(
            [hit["via"] for hit in row["retrieval_top_k"]],
            ["vector", "graph"],
        )

    def test_openai_payload_uses_vllm_thinking_switch(self):
        class CapturingClient(VLMAnswerClient):
            def _post_json(self, url, payload):
                self.payload = payload
                return {"choices": [{"message": {"content": "ok"}}]}

        client = CapturingClient(think=False)
        answer = client.answer(
            system_prompt="system",
            memory_items=[],
            question_prompt="question",
        )
        self.assertEqual(answer, "ok")
        self.assertEqual(
            client.payload["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertNotIn("think", client.payload)
        self.assertNotIn("extra_body", client.payload)


if __name__ == "__main__":
    unittest.main()
