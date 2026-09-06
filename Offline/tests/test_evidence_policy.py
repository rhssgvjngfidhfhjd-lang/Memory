from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient
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
from evidence_policy.ppo import PPOBuffer, PPOTrainer
from evidence_policy.retrieval import resolve_graph_options, validate_graph_config
from evidence_policy.rollout import (
    EvidenceEpisode,
    EvidenceSelectionEnv,
    RolloutCache,
)
from evidence_policy.vp_store import VPArtifactIndex
from hive_mem.mau import MAU
from hive_mem.retriever import MemoryHit
from scripts.evidence_policy import (
    initial_validation_signature,
    prepare_initial_validation,
    reconcile_ppo_metrics_for_resume,
    resume_configs_match,
    rollout_record,
    rollout_with_endpoint_recovery,
    validation_checkpoints,
)


EMBEDDING_DIM = 8


class ValidationScheduleTest(unittest.TestCase):
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
