from __future__ import annotations

import unittest
from unittest.mock import patch

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

from scripts.configure_evidence_policy_wandb_workspace import (
    build_sections,
    configure_workspace,
)


class WandbWorkspaceTest(unittest.TestCase):
    def test_workspace_has_planned_sections_and_panels(self) -> None:
        sections = build_sections(ws, wr)
        panel_counts = {section.name: len(section.panels) for section in sections}

        self.assertEqual(
            [section.name for section in sections],
            ["Validation", "Critic", "Actor"],
        )
        self.assertEqual(
            panel_counts,
            {
                "Validation": 9,
                "Critic": 8,
                "Actor": 7,
            },
        )
        self.assertTrue(sections[0].is_open)
        self.assertFalse(sections[1].is_open)
        self.assertFalse(sections[2].is_open)

    def test_required_actor_and_critic_metrics_are_independent_panels(self) -> None:
        sections = build_sections(ws, wr)
        critic = next(section for section in sections if section.name == "Critic")
        actor = next(section for section in sections if section.name == "Actor")

        reward_mean = next(
            panel
            for panel in critic.panels
            if getattr(panel, "title", None) == "critic/rewards/mean"
        )
        entropy = next(
            panel
            for panel in actor.panels
            if getattr(panel, "title", None) == "actor/entropy_loss"
        )

        self.assertEqual(reward_mean.y, ["critic/rewards/mean"])
        self.assertEqual(entropy.y, ["actor/entropy_loss"])
        self.assertTrue(reward_mean.smoothing_show_original)
        self.assertTrue(entropy.smoothing_show_original)

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
                "Evidence Combination Ratio",
                "Final Combination Distribution",
                "Final Evidence Level Ratio",
            ],
        )

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
