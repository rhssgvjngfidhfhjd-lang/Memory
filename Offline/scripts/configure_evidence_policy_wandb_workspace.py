#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from typing import Any


DEFAULT_ENTITY = (
    "rhssgvjngfidhfhjd-nanyang-technological-university-singapore"
)
DEFAULT_PROJECT = "hivemem-evidence-policy-v2"
DEFAULT_WORKSPACE_NAME = "Evidence Policy PPO Dashboard"
DEFAULT_WORKSPACE_URL = ""
VALIDATION_ACTION_MASKS = tuple(f"{value:05b}" for value in range(32))
TRAIN_METRICS = (
    "ppo_kl",
    "pg_loss",
    "pg_clipfrac",
    "lr",
    "grad_norm",
    "entropy_loss",
)
TRAIN_COST_METRICS = (
    "quality_reward_mean",
    "final_reward_mean",
    "raw_cost_mean",
    "base_cost_mean",
    "cost_min_mean",
    "cost_max_mean",
    "incremental_cost_mean",
    "transformed_cost_mean",
    "normalized_cost_mean",
    "cost_penalty_mean",
    "cost_effective_transformed_min",
    "cost_effective_transformed_max",
    "cost_effective_transformed_range",
    "task_reward_std",
    "normalized_cost_std",
    "cost_scale_alpha",
    "effective_cost_weight",
    "cost_low_clip_rate",
    "cost_high_clip_rate",
    "cost_saturation_rate",
    "all_zero_rollout_rate",
)
VALIDATION_COST_METRICS = (
    "quality_reward_mean",
    "final_reward_mean",
    "raw_cost_mean",
    "base_cost_mean",
    "cost_min_mean",
    "cost_max_mean",
    "incremental_cost_mean",
    "transformed_cost_mean",
    "normalized_cost_mean",
    "cost_penalty_mean",
    "cost_effective_transformed_min",
    "cost_effective_transformed_max",
    "cost_effective_transformed_range",
    "task_reward_std",
    "normalized_cost_std",
    "cost_scale_alpha",
    "effective_cost_weight",
    "cost_low_clip_rate",
    "cost_high_clip_rate",
    "cost_saturation_rate",
    "all_zero_rollout_rate",
)
EVIDENCE_LEVEL_CHART_SPEC = (
    f"{DEFAULT_ENTITY}/hivemem-evidence-level-ratio-small-multiples-v2"
)


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
    frontier = ws.Section(
        name="Quality-Cost Frontier",
        is_open=True,
        panels=[
            wr.ScatterPlot(
                title="Test F1 vs Raw QA Cost",
                x=wr.SummaryMetric("test/cost/raw_cost_mean"),
                y=wr.SummaryMetric("test/f1"),
                z=wr.Config("cost_tradeoff_lambda"),
                legend_template="${runName}: lambda=${config:cost_tradeoff_lambda}",
            )
        ],
    )
    training = ws.Section(
        name="Training",
        is_open=True,
        panels=[
            *[
                wr.LinePlot(
                    title=f"train/{metric}",
                    x="train/update_step",
                    y=[f"train/{metric}"],
                    smoothing_type="none",
                )
                for metric in TRAIN_METRICS
            ],
            *[
                wr.LinePlot(
                    title=f"train/cost/{metric}",
                    x="train/update_step",
                    y=[f"train/cost/{metric}"],
                    smoothing_type="none",
                )
                for metric in TRAIN_COST_METRICS
            ],
            custom_table_chart(
                wr,
                table_key="train/action_mask_ratio_table",
                panel_def_id="wandb/lineseries/v0",
                field_settings={
                    "lineKey": "lineKey",
                    "lineVal": "lineVal",
                    "step": "step",
                },
                title="Training Evidence Combination Ratio",
                string_settings={"xname": "PPO update step"},
            ),
        ],
    )

    validation = ws.Section(
        name="Validation",
        is_open=False,
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
            *[
                wr.LinePlot(
                    title=f"val/cost/{metric}",
                    x="val/update_step",
                    y=[f"val/cost/{metric}"],
                    smoothing_type="none",
                )
                for metric in VALIDATION_COST_METRICS
            ],
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
                panel_def_id=EVIDENCE_LEVEL_CHART_SPEC,
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
        ],
    )

    test = ws.Section(
        name="Test",
        is_open=False,
        panels=[
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
    return [frontier, training, critic, validation, test]


def configure_workspace(
    *,
    entity: str,
    project: str,
    name: str,
    workspace_url: str = "",
    run_name: str = "",
    run_id: str = "",
    run_name_regex: str = "",
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
        query=run_name_regex or (f"^{re.escape(run_name)}$" if run_name else ""),
        regex_query=bool(run_name_regex or run_name),
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
    parser.add_argument("--run-name-regex", default="")
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
        run_name_regex=args.run_name_regex,
    )
    print(json.dumps({"url": url}, indent=2))


if __name__ == "__main__":
    main()
