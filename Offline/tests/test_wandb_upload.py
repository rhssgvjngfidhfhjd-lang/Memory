from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.upload_evidence_policy_wandb import (
    ALL_EVIDENCE_MASKS,
    build_evidence_level_ratio_line_chart,
    build_test_summary,
    evidence_level_distribution,
    load_run_data,
    mask_distribution,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class WandbUploadTest(unittest.TestCase):
    def test_builds_llm_judge_and_calls_summary(self) -> None:
        summary = build_test_summary(
            {
                "count": 275,
                "f1": 0.6,
                "llm_judge": 0.67,
                "calls": {
                    "memory_bank": {
                        "available": True,
                        "total_calls": 670,
                        "failed_calls": 2,
                        "successful_calls": 668,
                        "num_samples": 4,
                        "mean_per_sample": 167.5,
                    },
                    "qa": {
                        "available": True,
                        "total_calls": 275,
                        "failed_calls": 0,
                        "successful_calls": 275,
                        "num_samples": 4,
                        "mean_per_sample": 68.75,
                    },
                    "total": {
                        "available": True,
                        "total_calls": 945,
                        "failed_calls": 2,
                        "successful_calls": 943,
                        "num_samples": 4,
                        "mean_per_sample": 236.25,
                    },
                },
            }
        )

        self.assertEqual(summary["test/llm_judge"], 0.67)
        self.assertEqual(summary["test/calls/memory_bank/total_calls"], 670)
        self.assertEqual(summary["test/calls/qa/mean_per_sample"], 68.75)
        self.assertEqual(summary["test/calls/total/mean_per_sample"], 236.25)

    def test_builds_validation_evidence_level_ratio_lines_by_update_step(self) -> None:
        class FakePlot:
            @staticmethod
            def line_series(**kwargs):
                return kwargs

        class FakeWandb:
            plot = FakePlot()

        chart = build_evidence_level_ratio_line_chart(
            FakeWandb(),
            [
                {
                    "update_step": 0,
                    "evidence_actions": {
                        "mask:00000": 1,
                        "mask:11000": 1,
                    },
                },
                {
                    "update_step": 29,
                    "evidence_actions": {"mask:00011": 2},
                },
            ],
            title="Evidence levels",
        )

        self.assertIsNotNone(chart)
        self.assertEqual(chart["xs"], [0, 29])
        self.assertEqual(
            chart["keys"], ["summary", "dialogue", "caption", "image", "vp"]
        )
        self.assertEqual(chart["ys"][0], [0.5, 0.0])
        self.assertEqual(chart["ys"][1], [0.5, 0.0])
        self.assertEqual(chart["ys"][2], [0.0, 0.0])
        self.assertEqual(chart["ys"][3], [0.0, 1.0])
        self.assertEqual(chart["ys"][4], [0.0, 1.0])

    def test_loads_step_zero_and_train_mask_ratios(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "train.log",
                [
                    {
                        "epoch": 0,
                        "update_step": 2,
                        "validations": [
                            {
                                "phase": "end",
                                "update_step": 2,
                                "metrics": {
                                    "mean_reward": 0.75,
                                    "evidence_actions": {"mask:11000": 4},
                                },
                            }
                        ],
                    }
                ],
            )
            write_jsonl(
                root / "ppo_metrics.jsonl",
                [{"epoch": 0, "update_step": 2, "ppo_kl": 0.01}],
            )
            initial = {
                "epoch": 0,
                "phase": "initial",
                "update_step": 0,
                "train_question_count": 0,
                "metrics": {
                    "mean_reward": 0.25,
                    "evidence_actions": {
                        "mask:00000": 3,
                        "mask:11000": 1,
                    },
                },
            }
            (root / "validation").mkdir()
            (root / "validation" / "initial_metrics.json").write_text(
                json.dumps(initial), encoding="utf-8"
            )
            write_jsonl(
                root / "train" / "epoch_000_rollouts.jsonl",
                [
                    {
                        "actions": [
                            {"mask": "00000"},
                            {"mask": "00011"},
                            {"mask": "00011"},
                        ]
                    }
                ],
            )

            data = load_run_data(root)

        self.assertEqual([row["update_step"] for row in data.validation_rows], [0, 2])
        self.assertEqual(data.validation_rows[0]["phase"], "initial")
        counts, ratios, total = mask_distribution(
            data.validation_rows[0]["evidence_actions"]
        )
        self.assertEqual(total, 4)
        self.assertEqual(counts["00000"], 3)
        self.assertEqual(ratios["00000"], 0.75)
        self.assertEqual(len(ratios), 32)
        self.assertAlmostEqual(sum(ratios.values()), 1.0)
        train_counts, train_ratios, train_total = mask_distribution(
            data.train_action_rows[0]["evidence_actions"]
        )
        self.assertEqual(train_total, 3)
        self.assertEqual(train_counts["00011"], 2)
        self.assertAlmostEqual(train_ratios["00011"], 2 / 3)
        self.assertEqual(set(train_ratios), set(ALL_EVIDENCE_MASKS))

    def test_evidence_level_ratios_are_derived_from_independent_mask_bits(self) -> None:
        counts, ratios, total = evidence_level_distribution(
            {
                "mask:00000": 2,
                "mask:01000": 3,
                "mask:00011": 5,
            }
        )

        self.assertEqual(total, 10)
        self.assertEqual(
            counts,
            {
                "summary": 0,
                "dialogue": 3,
                "caption": 0,
                "image": 5,
                "vp": 5,
            },
        )
        self.assertEqual(ratios["dialogue"], 0.3)
        self.assertEqual(ratios["image"], 0.5)
        self.assertEqual(ratios["vp"], 0.5)

    def test_recovers_epoch_summaries_from_checkpoints_without_train_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "ppo_metrics.jsonl",
                [{"epoch": 0, "update_step": 3, "ppo_kl": 0.01}],
            )
            checkpoint = root / "checkpoints" / "epoch_000.pt"
            checkpoint.parent.mkdir(parents=True)
            validation = {
                "mean_reward": 0.75,
                "f1": 0.5,
                "exact_match": 0.25,
                "retrieval_hitrate@5": 1.0,
                "errors": 0,
                "by_category": {},
            }
            torch.save(
                {
                    "epoch": 0,
                    "update_steps": 3,
                    "config": {"seed": 42, "ppo": {"learning_rate": 3e-4}},
                    "extra": {
                        "train_question_count": 32,
                        "validation": validation,
                        "validations": [
                            {
                                "phase": "end",
                                "update_step": 3,
                                "metrics": validation,
                            }
                        ],
                    },
                },
                checkpoint,
            )

            data = load_run_data(root)

        self.assertEqual(data.config["seed"], 42)
        self.assertEqual(len(data.epoch_rows), 1)
        self.assertEqual(data.validation_rows[0]["update_step"], 3)
        self.assertEqual(data.validation_rows[0]["reward"], 0.75)
        self.assertTrue(any("recovered from checkpoints" in row for row in data.warnings))

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
