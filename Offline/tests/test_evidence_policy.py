from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient
from evidence_policy.evidence import (
    DialogueStore,
    EvidenceChainBuilder,
    EvidenceStrategy,
    EvidenceTextAction,
    EvidenceVisualAction,
    MAUEvidenceAction,
    choose_baseline_actions,
    make_policy_observation,
)
from evidence_policy.policy import EvidenceSelectionPolicy
from evidence_policy.ppo import PPOBuffer, PPOTrainer
from evidence_policy.rollout import (
    EvidenceEpisode,
    EvidenceSelectionEnv,
    RolloutCache,
)
from hive_mem.mau import MAU
from hive_mem.retriever import MemoryHit
from scripts.evidence_policy import validation_checkpoints


EMBEDDING_DIM = 8


class ValidationScheduleTest(unittest.TestCase):
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
                "m1", EvidenceTextAction.DIALOGUE, EvidenceVisualAction.IMAGE
            )
            items = EvidenceChainBuilder(DialogueStore(root)).build(
                "toy", "VS", [hit], [action]
            )

        self.assertIn("User: What should I bake?", items[0]["text"])
        self.assertEqual(items[0]["image"]["path"], str(image_path))
        self.assertNotIn("A fruit tart.", items[0]["text"])

    def test_caption_action_adds_caption_without_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dialogue_dataset(root)
            hit = make_hit("m1", image_path="old/image.jpg", caption="A tart.")
            action = MAUEvidenceAction(
                "m1", EvidenceTextAction.SUMMARY, EvidenceVisualAction.CAPTION
            )
            items = EvidenceChainBuilder(DialogueStore(root)).build(
                "toy", "FR", [hit], [action]
            )

        self.assertEqual(items[0]["text"], "summary fact\nImage caption: A tart.")
        self.assertIsNone(items[0]["image"])

    def test_baseline_actions_respect_visual_constraints(self):
        visual = make_hit("visual", image_path="image.jpg", caption="caption")
        text_only = make_hit("text")
        actions = choose_baseline_actions(
            [visual, text_only], "FR", EvidenceStrategy.FULL
        )
        self.assertIs(actions[0].visual, EvidenceVisualAction.CAPTION)
        self.assertIsNone(actions[1].visual)
        with self.assertRaisesRegex(ValueError, "only valid for VS/VR"):
            EvidenceChainBuilder(DialogueStore("unused")).build(
                "toy",
                "FR",
                [visual],
                [
                    MAUEvidenceAction(
                        "visual",
                        EvidenceTextAction.SUMMARY,
                        EvidenceVisualAction.IMAGE,
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
        self.assertTrue(all(action.visual is None for action in sampled.actions))

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
