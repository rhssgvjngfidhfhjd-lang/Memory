#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from typing import Any


DEFAULT_ENTITY = (
    "rhssgvjngfidhfhjd-nanyang-technological-university-singapore"
)
DEFAULT_PROJECT = "hivemem-evidence-policy"
DEFAULT_WORKSPACE_NAME = "Evidence Policy PPO Dashboard"
DEFAULT_WORKSPACE_URL = (
    "https://wandb.ai/"
    "rhssgvjngfidhfhjd-nanyang-technological-university-singapore/"
    "hivemem-evidence-policy?nw=aw3roht3dqk"
)
VALIDATION_ACTION_MASKS = tuple(f"{value:05b}" for value in range(32))


def custom_table_chart(
    wr: Any,
    *,
    table_key: str,
    panel_def_id: str,
    field_settings: dict[str, str],
    title: str,
    string_settings: dict[str, str] | None = None,
) -> Any:
    return wr.CustomChart(
        query={"summaryTable": {"tableKey": table_key}},
        chart_name=panel_def_id,
        chart_fields=field_settings,
        chart_strings={"title": title, **(string_settings or {})},
    )


def build_sections(ws: Any, wr: Any) -> list[Any]:
    validation = ws.Section(
        name="Validation",
        is_open=True,
        panels=[
            custom_table_chart(
                wr,
                table_key="val/category_f1_table",
                panel_def_id="wandb/lineseries/v0",
                field_settings={
                    "lineKey": "lineKey",
                    "lineVal": "lineVal",
                    "step": "step",
                },
                title="Validation Category F1",
                string_settings={"xname": "PPO update step"},
            ),
            wr.LinePlot(
                title="Validation Reward",
                x="val/update_step",
                y=["val/reward"],
                title_x="PPO update step",
                title_y="Reward",
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="Validation F1",
                x="val/update_step",
                y=["val/f1"],
                title_x="PPO update step",
                title_y="F1",
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="Validation Exact Match",
                x="val/update_step",
                y=["val/exact_match"],
                title_x="PPO update step",
                title_y="Exact match",
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="Validation Retrieval Hit Rate@5",
                x="val/update_step",
                y=["val/retrieval_hitrate_at_5"],
                title_x="PPO update step",
                title_y="Hit rate@5",
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="Validation Errors",
                x="val/update_step",
                y=["val/errors"],
                title_x="PPO update step",
                title_y="Errors",
                smoothing_type="none",
            ),
            custom_table_chart(
                wr,
                table_key="val/action_mask_ratio_table",
                panel_def_id="wandb/lineseries/v0",
                field_settings={
                    "lineKey": "lineKey",
                    "lineVal": "lineVal",
                    "step": "step",
                },
                title="Evidence Combination Ratio",
                string_settings={"xname": "PPO update step"},
            ),
            custom_table_chart(
                wr,
                table_key="val/evidence_level_ratio_table",
                panel_def_id="wandb/lineseries/v0",
                field_settings={
                    "lineKey": "lineKey",
                    "lineVal": "lineVal",
                    "step": "step",
                },
                title="Evidence Level Selection Ratio",
                string_settings={"xname": "PPO update step"},
            ),
            *[
                wr.LinePlot(
                    title=f"val/action_ratio/{mask}",
                    x="val/update_step",
                    y=[f"val/action_ratio/{mask}"],
                    title_x="PPO update step",
                    title_y="Selection ratio",
                    smoothing_type="none",
                )
                for mask in VALIDATION_ACTION_MASKS
            ],
            custom_table_chart(
                wr,
                table_key="test/action_mask_ratio_table",
                panel_def_id="wandb/bar/v0",
                field_settings={"label": "combination", "value": "ratio"},
                title="Final Combination Distribution",
            ),
            custom_table_chart(
                wr,
                table_key="test/evidence_level_ratio_table",
                panel_def_id="wandb/bar/v0",
                field_settings={"label": "evidence_level", "value": "ratio"},
                title="Final Evidence Level Ratio",
            ),
        ],
    )

    actor = ws.Section(
        name="Actor",
        is_open=False,
        panels=[
            wr.LinePlot(
                title="actor/update_step",
                x="Step",
                y=["actor/update_step"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="actor/ppo_kl",
                x="actor/update_step",
                y=["actor/ppo_kl"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="actor/pg_loss",
                x="actor/update_step",
                y=["actor/pg_loss"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="actor/pg_clipfrac",
                x="actor/update_step",
                y=["actor/pg_clipfrac"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="actor/lr",
                x="actor/update_step",
                y=["actor/lr"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="actor/grad_norm",
                x="actor/update_step",
                y=["actor/grad_norm"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="actor/entropy_loss",
                x="actor/update_step",
                y=["actor/entropy_loss"],
                title_x="actor/update_step",
                smoothing_factor=0.6,
                smoothing_type="exponential",
                smoothing_show_original=True,
            ),
        ],
    )

    critic = ws.Section(
        name="Critic",
        is_open=False,
        panels=[
            custom_table_chart(
                wr,
                table_key="critic/predicted_value_vs_reward_table",
                panel_def_id="wandb/lineseries/v0",
                field_settings={
                    "lineKey": "lineKey",
                    "lineVal": "lineVal",
                    "step": "step",
                },
                title="Predicted Value vs Reward",
                string_settings={"xname": "PPO update step"},
            ),
            wr.LinePlot(
                title="critic/update_step",
                x="Step",
                y=["critic/update_step"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="critic/value_loss",
                x="critic/update_step",
                y=["critic/value_loss"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="critic/absolute_value_error",
                x="critic/update_step",
                y=["critic/absolute_value_error"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="critic/explained_variance",
                x="critic/update_step",
                y=["critic/explained_variance"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="critic/rewards/mean",
                x="critic/update_step",
                y=["critic/rewards/mean"],
                title_x="Step",
                smoothing_factor=0.6,
                smoothing_type="exponential",
                smoothing_show_original=True,
            ),
            wr.LinePlot(
                title="critic/rewards/min",
                x="critic/update_step",
                y=["critic/rewards/min"],
                smoothing_type="none",
            ),
            wr.LinePlot(
                title="critic/rewards/max",
                x="critic/update_step",
                y=["critic/rewards/max"],
                smoothing_type="none",
            ),
        ],
    )
    return [validation, critic, actor]


def configure_workspace(
    *,
    entity: str,
    project: str,
    name: str,
    workspace_url: str = "",
    run_name: str = "",
    run_id: str = "",
) -> str:
    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError as error:
        raise RuntimeError(
            "wandb-workspaces is required; install Offline/requirements.txt"
        ) from error

    sections = build_sections(ws, wr)
    runset_settings = ws.RunsetSettings(
        query=f"^{re.escape(run_name)}$" if run_name else "",
        regex_query=bool(run_name),
        pinned_runs=[run_id] if run_id else [],
    )
    if workspace_url:
        workspace = ws.Workspace.from_url(workspace_url)
        workspace.name = name
        workspace.sections = sections
        workspace.runset_settings = runset_settings
    else:
        workspace = ws.Workspace(
            entity=entity,
            project=project,
            name=name,
            sections=sections,
            runset_settings=runset_settings,
            auto_generate_panels=False,
        )
    workspace.save()
    return workspace.url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or update the Evidence Policy W&B saved workspace"
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--name", default=DEFAULT_WORKSPACE_NAME)
    parser.add_argument(
        "--workspace-url",
        default=DEFAULT_WORKSPACE_URL,
        help="Existing saved-view URL to update instead of creating a duplicate",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws

        sections = build_sections(ws, wr)
        print(
            json.dumps(
                {
                    "name": args.name,
                    "sections": [
                        {"name": section.name, "panels": len(section.panels)}
                        for section in sections
                    ],
                },
                indent=2,
            )
        )
        return

    url = configure_workspace(
        entity=args.entity,
        project=args.project,
        name=args.name,
        workspace_url=args.workspace_url,
        run_name=args.run_name,
        run_id=args.run_id,
    )
    print(json.dumps({"url": url}, indent=2))


if __name__ == "__main__":
    main()
