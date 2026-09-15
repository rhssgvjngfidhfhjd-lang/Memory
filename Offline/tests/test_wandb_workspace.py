from __future__ import annotations

import unittest
from unittest.mock import patch

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

from scripts.configure_evidence_policy_wandb_workspace import (
    TRAIN_COST_METRICS,
    TRAIN_METRICS,
    VALIDATION_ACTION_MASKS,
    VALIDATION_COST_METRICS,
    build_sections,
    configure_workspace,
)


class WandbWorkspaceTest(unittest.TestCase):
    def test_workspace_has_planned_sections_and_panels(self) -> None:
        sections = build_sections(ws, wr)
        panel_counts = {section.name: len(section.panels) for section in sections}

        self.assertEqual(
            [section.name for section in sections],
            [
                "Quality-Cost Frontier",
                "Training",
                "Critic",
                "Validation",
                "Test",
            ],
        )
        self.assertEqual(
            panel_counts,
            {
                "Quality-Cost Frontier": 1,
                "Training": 28,
                "Critic": 8,
                "Validation": 61,
                "Test": 2,
            },
        )
        self.assertTrue(sections[0].is_open)
        self.assertTrue(sections[1].is_open)
        self.assertFalse(sections[2].is_open)
        self.assertFalse(sections[3].is_open)
        self.assertFalse(sections[4].is_open)

    def test_frontier_is_the_first_visible_section(self) -> None:
        frontier = build_sections(ws, wr)[0]

        self.assertEqual(frontier.name, "Quality-Cost Frontier")
        self.assertTrue(frontier.is_open)
        self.assertEqual(
            frontier.panels[0].title,
            "Test F1 vs Raw QA Cost",
        )

    def test_required_training_and_critic_metrics_are_independent_panels(self) -> None:
        sections = build_sections(ws, wr)
        critic = next(section for section in sections if section.name == "Critic")
        training = next(section for section in sections if section.name == "Training")

        reward_mean = next(
            panel
            for panel in critic.panels
            if getattr(panel, "title", None) == "critic/rewards/mean"
        )
        entropy = next(
            panel
            for panel in training.panels
            if getattr(panel, "title", None) == "train/entropy_loss"
        )

        self.assertEqual(reward_mean.y, ["critic/rewards/mean"])
        self.assertEqual(entropy.y, ["train/entropy_loss"])
        self.assertTrue(reward_mean.smoothing_show_original)
        self.assertEqual(entropy.smoothing_type, "none")

    def test_training_contains_fixed_policy_and_cost_panels(self) -> None:
        training = next(
            section for section in build_sections(ws, wr)
            if section.name == "Training"
        )
        titles = {getattr(panel, "title", None) for panel in training.panels}
        self.assertTrue({f"train/{name}" for name in TRAIN_METRICS} <= titles)
        self.assertTrue(
            {f"train/cost/{name}" for name in TRAIN_COST_METRICS} <= titles
        )

    def test_validation_has_requested_metrics_and_evidence_panels(self) -> None:
        sections = build_sections(ws, wr)
        validation = next(
            section for section in sections if section.name == "Validation"
        )
        titles = [
            getattr(panel, "title", None)
            or panel.chart_strings.get("title")
            for panel in validation.panels
        ]

        self.assertEqual(
            titles,
            [
                "Validation Category F1",
                "Validation Reward",
                "Validation F1",
                "Validation Exact Match",
                "Validation Retrieval Hit Rate@5",
                "Validation Errors",
                *[f"val/cost/{name}" for name in VALIDATION_COST_METRICS],
                "Evidence Combination Ratio",
                "Evidence Level Selection Ratio",
                *[f"val/action_ratio/{mask}" for mask in VALIDATION_ACTION_MASKS],
            ],
        )

    def test_test_section_contains_only_final_distributions(self) -> None:
        test = next(
            section for section in build_sections(ws, wr)
            if section.name == "Test"
        )
        self.assertEqual(
            [panel.chart_strings["title"] for panel in test.panels],
            ["Final Combination Distribution", "Final Evidence Level Ratio"],
        )

    def test_each_validation_action_mask_has_an_independent_ratio_panel(self) -> None:
        sections = build_sections(ws, wr)
        validation = next(
            section for section in sections if section.name == "Validation"
        )
        panels = {
            getattr(panel, "title", ""): panel
            for panel in validation.panels
            if getattr(panel, "title", "").startswith("val/action_ratio/")
        }

        self.assertEqual(
            set(panels),
            {f"val/action_ratio/{mask}" for mask in VALIDATION_ACTION_MASKS},
        )
        for mask in VALIDATION_ACTION_MASKS:
            panel = panels[f"val/action_ratio/{mask}"]
            self.assertEqual(panel.x, "val/update_step")
            self.assertEqual(panel.y, [f"val/action_ratio/{mask}"])
            self.assertEqual(panel.smoothing_type, "none")

    def test_validation_reward_uses_explicit_update_step(self) -> None:
        sections = build_sections(ws, wr)
        validation = next(
            section for section in sections if section.name == "Validation"
        )
        reward = next(
            panel
            for panel in validation.panels
            if getattr(panel, "title", None) == "Validation Reward"
        )

        self.assertEqual(reward.x, "val/update_step")
        self.assertEqual(reward.y, ["val/reward"])

    def test_per_run_workspace_filters_and_pins_target_run(self) -> None:
        saved = []

        class FakeWorkspace:
            url = "https://wandb.ai/example/project?nw=run-view"

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

            def save(self):
                saved.append(self)

        with patch.object(ws, "Workspace", FakeWorkspace):
            url = configure_workspace(
                entity="example",
                project="project",
                name="target Dashboard",
                run_name="target",
                run_id="target-id",
            )

        self.assertEqual(url, FakeWorkspace.url)
        self.assertEqual(len(saved), 1)
        settings = saved[0].runset_settings
        self.assertEqual(settings.query, "^target$")
        self.assertTrue(settings.regex_query)
        self.assertEqual(settings.pinned_runs, ["target-id"])


if __name__ == "__main__":
    unittest.main()
