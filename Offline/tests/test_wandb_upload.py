from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.upload_evidence_policy_wandb import load_run_data


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class WandbUploadTest(unittest.TestCase):
    def test_loads_validation_and_ppo_metrics_by_update_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "seed": 42,
                        "ppo": {"learning_rate": 3e-4},
                        "split": {"train": [], "validation": [], "test": []},
                    }
                ),
                encoding="utf-8",
            )
            write_jsonl(
                root / "train.log",
                [
                    {
                        "epoch": 0,
                        "update_step": 2,
                        "train_question_count": 32,
                        "train_episodes": 2,
                        "updates": {},
                        "validation": {
                            "count": 1,
                            "f1": 1.0,
                            "exact_match": 1.0,
                            "retrieval_hitrate@5": 1.0,
                            "by_category": {"FR": {"f1": 1.0}},
                            "mean_reward": 1.0,
                            "errors": 0,
                        },
                        "validations": [
                            {
                                "phase": "half",
                                "update_step": 1,
                                "metrics": {
                                    "count": 1,
                                    "f1": 0.5,
                                    "exact_match": 0.0,
                                    "retrieval_hitrate@5": 1.0,
                                    "by_category": {"FR": {"f1": 0.5}},
                                    "mean_reward": 0.5,
                                    "errors": 0,
                                },
                            },
                            {
                                "phase": "end",
                                "update_step": 2,
                                "metrics": {
                                    "count": 1,
                                    "f1": 1.0,
                                    "exact_match": 1.0,
                                    "retrieval_hitrate@5": 1.0,
                                    "by_category": {"FR": {"f1": 1.0}},
                                    "mean_reward": 1.0,
                                    "errors": 0,
                                },
                            },
                        ],
                    }
                ],
            )
            ppo_row = {
                "epoch": 0,
                "question_count": 32,
                "update_step": 1,
                "ppo_kl": 0.01,
                "pg_loss": -0.1,
                "pg_clipfrac": 0.2,
                "lr": 3e-4,
                "grad_norm": 0.5,
                "entropy_loss": 0.7,
                "value_loss": 0.2,
                "predicted_value_mean": 0.4,
                "target_return_mean": 0.5,
                "absolute_value_error": 0.1,
                "explained_variance": 0.3,
                "reward_mean": 0.5,
                "reward_min": 0.0,
                "reward_max": 1.0,
                "batch_size": 32.0,
            }
            write_jsonl(root / "ppo_metrics.jsonl", [ppo_row])
            write_jsonl(
                root / "validation" / "epoch_000_rollouts.jsonl",
                [
                    {
                        "category": "FR",
                        "answer": "fruit tart",
                        "original_answer": "fruit tart",
                        "reward": 1.0,
                        "retrieved_source_groups": [["D1:1"]],
                        "clue": ["D1:1"],
                        "error": "",
                    }
                ],
            )

            data = load_run_data(root)

        self.assertEqual(len(data.update_rows), 1)
        self.assertEqual(data.update_rows[0]["update_step"], 1)
        self.assertEqual(data.update_rows[0]["question_count"], 32)
        self.assertEqual(len(data.validation_rows), 2)
        self.assertEqual(
            [row["update_step"] for row in data.validation_rows], [1, 2]
        )
        self.assertEqual(
            [row["phase"] for row in data.validation_rows], ["half", "end"]
        )
        self.assertEqual(data.validation_rows[0]["f1"], 0.5)
        self.assertEqual(data.validation_rows[1]["exact_match"], 1.0)
        self.assertEqual(data.validation_rows[1]["retrieval_hitrate_at_5"], 1.0)
        self.assertEqual(
            data.validation_rows[1]["by_category"], {"FR": {"f1": 1.0}}
        )
        self.assertEqual(data.warnings, ())

    def test_legacy_run_does_not_fabricate_missing_actor_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "train.log",
                [
                    {
                        "epoch": 0,
                        "train_episodes": 1,
                        "updates": {
                            "policy_loss": -0.1,
                            "value_loss": 0.2,
                            "entropy": 0.7,
                            "grad_norm": 0.5,
                        },
                        "validation": {
                            "count": 1,
                            "f1": 0.5,
                            "exact_match": 0.0,
                            "retrieval_hitrate@5": 1.0,
                            "by_category": {},
                            "mean_reward": 0.5,
                            "errors": 0,
                        },
                    }
                ],
            )
            write_jsonl(
                root / "train" / "epoch_000_rollouts.jsonl",
                [{"reward": 0.5, "value": 0.25}],
            )

            data = load_run_data(root)

        self.assertNotIn("ppo_kl", data.update_rows[0])
        self.assertNotIn("pg_clipfrac", data.update_rows[0])
        self.assertTrue(any("not fabricated" in warning for warning in data.warnings))


if __name__ == "__main__":
    unittest.main()
