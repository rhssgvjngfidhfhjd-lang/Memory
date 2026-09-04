#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scripts.configure_evidence_policy_wandb_workspace import (
        configure_workspace,
    )
except ModuleNotFoundError:
    from configure_evidence_policy_wandb_workspace import configure_workspace


ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)


ACTOR_FIELDS = (
    "ppo_kl",
    "pg_loss",
    "pg_clipfrac",
    "lr",
    "grad_norm",
    "entropy_loss",
)
CRITIC_FIELDS = (
    "value_loss",
    "absolute_value_error",
    "explained_variance",
    "reward_mean",
    "reward_min",
    "reward_max",
)
EVIDENCE_ORDER = ("summary", "dialogue", "caption", "image", "vp")
ALL_EVIDENCE_MASKS = tuple(f"{value:05b}" for value in range(32))


@dataclass(frozen=True)
class RunData:
    config: dict[str, Any]
    epoch_rows: tuple[dict[str, Any], ...]
    update_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]
    train_action_rows: tuple[dict[str, Any], ...]
    test_metrics: dict[str, Any]
    warnings: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload Evidence Policy validation, actor, and critic charts to W&B"
    )
    parser.add_argument("--run-dir", required=True, help="Evidence Policy output directory")
    parser.add_argument("--project", default="hivemem-evidence-policy")
    parser.add_argument("--entity", default="")
    parser.add_argument("--name", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument(
        "--workspace-url",
        default="",
        help="Existing per-run saved-view URL to update",
    )
    parser.add_argument("--skip-workspace", action="store_true")
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Only upload derived evidence charts; do not append scalar history",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    data = load_run_data(run_dir)
    if args.dry_run:
        print(json.dumps(data_summary(data), ensure_ascii=False, indent=2))
        return
    run_url = upload_to_wandb(
        data,
        run_dir=run_dir,
        project=args.project,
        entity=args.entity or None,
        name=args.name or run_dir.name,
        run_id=args.run_id or None,
        tags=args.tag,
        charts_only=args.charts_only,
    )
    dashboard_url = ""
    if not args.skip_workspace:
        dashboard_file = run_dir / "run_control" / "wandb_dashboard.json"
        workspace_url = args.workspace_url or load_dashboard_url(dashboard_file)
        dashboard_url = configure_workspace(
            entity=args.entity
            or "rhssgvjngfidhfhjd-nanyang-technological-university-singapore",
            project=args.project,
            name=f"{args.name or run_dir.name} Dashboard",
            workspace_url=workspace_url,
            run_name=args.name or run_dir.name,
            run_id=args.run_id or "",
        )
        dashboard_file.parent.mkdir(parents=True, exist_ok=True)
        dashboard_file.write_text(
            json.dumps({"url": dashboard_url}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "url": dashboard_url or run_url,
                "dashboard_url": dashboard_url,
                "run_url": run_url,
                **data_summary(data),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_dashboard_url(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(payload.get("url", "")) if isinstance(payload, dict) else ""


def load_run_data(run_dir: Path) -> RunData:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    train_log = run_dir / "train.log"
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    warnings: list[str] = []
    if train_log.exists():
        epoch_rows = tuple(read_jsonl(train_log))
    else:
        checkpoint_config, epoch_rows = load_checkpoint_epoch_rows(run_dir)
        if not config:
            config = checkpoint_config
        warnings.append(
            "train.log is missing; epoch summaries were recovered from checkpoints"
        )
    if not epoch_rows:
        raise ValueError(
            f"No epoch summaries found in {train_log} or {run_dir / 'checkpoints'}"
        )

    ppo_metrics_path = run_dir / "ppo_metrics.jsonl"
    if ppo_metrics_path.exists():
        update_rows = tuple(read_jsonl(ppo_metrics_path))
    else:
        update_rows = tuple(build_legacy_update_rows(run_dir, config, epoch_rows))
        warnings.append(
            "ppo_metrics.jsonl is missing; legacy epoch-level actor/critic metrics were used"
        )

    missing_actor = sorted(
        field
        for field in ACTOR_FIELDS
        if not any(field in row for row in update_rows)
    )
    if missing_actor:
        warnings.append(
            "Actor metrics unavailable and not fabricated: " + ", ".join(missing_actor)
        )

    initial_path = run_dir / "validation" / "initial_metrics.json"
    initial_validation = (
        json.loads(initial_path.read_text(encoding="utf-8"))
        if initial_path.exists()
        else None
    )
    validation_rows, validation_warnings = build_validation_rows(
        epoch_rows, update_rows, initial_validation=initial_validation
    )
    warnings.extend(validation_warnings)
    train_action_rows = build_train_action_rows(run_dir, epoch_rows)

    test_path = run_dir / "eval" / "test_ppo" / "metrics.json"
    test_metrics = (
        json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
    )
    return RunData(
        config=config,
        epoch_rows=epoch_rows,
        update_rows=update_rows,
        validation_rows=validation_rows,
        train_action_rows=train_action_rows,
        test_metrics=test_metrics,
        warnings=tuple(warnings),
    )


def load_checkpoint_epoch_rows(
    run_dir: Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Recover upload metadata when training stdout was not redirected."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to recover epoch summaries from checkpoints"
        ) from exc

    config: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    checkpoint_dir = run_dir / "checkpoints"
    for checkpoint in sorted(checkpoint_dir.glob("epoch_*.pt")):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            continue
        checkpoint_config = payload.get("config")
        if isinstance(checkpoint_config, dict):
            config = checkpoint_config
        extra = payload.get("extra")
        if not isinstance(extra, dict):
            extra = {}
        epoch = int(payload.get("epoch", len(rows)))
        validation = extra.get("validation", {})
        validations = extra.get("validations", [])
        rows.append(
            {
                "epoch": epoch,
                "update_step": int(payload.get("update_steps", epoch + 1)),
                "train_question_count": int(
                    extra.get("train_question_count", 0)
                ),
                "validation": validation if isinstance(validation, dict) else {},
                "validations": validations if isinstance(validations, list) else [],
            }
        )
    return config, tuple(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        rows.append(row)
    return rows


def build_validation_rows(
    epoch_rows: Iterable[dict[str, Any]],
    update_rows: Iterable[dict[str, Any]],
    *,
    initial_validation: dict[str, Any] | None = None,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    updates_by_epoch: dict[int, int] = {}
    for row in update_rows:
        if "epoch" in row and "update_step" in row:
            epoch = int(row["epoch"])
            updates_by_epoch[epoch] = max(
                updates_by_epoch.get(epoch, 0), int(row["update_step"])
            )

    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    if initial_validation is not None:
        rows.append(validation_event_row(initial_validation, default_epoch=0))
    for epoch_row in epoch_rows:
        epoch = int(epoch_row.get("epoch", len(rows)))
        validation_events = epoch_row.get("validations")
        if not isinstance(validation_events, list) or not validation_events:
            validation_events = [
                {
                    "phase": "end",
                    "update_step": epoch_row.get(
                        "update_step", updates_by_epoch.get(epoch)
                    ),
                    "metrics": epoch_row.get("validation", {}),
                }
            ]
        for event in validation_events:
            if event.get("update_step") is None:
                event = dict(event)
                event["update_step"] = epoch + 1
                warnings.append(
                    f"Epoch {epoch} validation has no PPO update step; epoch index was used"
                )
            rows.append(validation_event_row(event, default_epoch=epoch))
    unique: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["update_step"]), str(row["phase"]))
        previous = unique.get(key)
        if previous is not None and previous != row:
            raise ValueError(f"Conflicting validation events for step/phase {key}")
        unique[key] = row
    ordered = sorted(
        unique.values(), key=lambda row: (int(row["update_step"]), str(row["phase"]))
    )
    return tuple(ordered), tuple(warnings)


def validation_event_row(
    event: dict[str, Any], *, default_epoch: int
) -> dict[str, Any]:
    validation = event.get("metrics", {})
    if not isinstance(validation, dict):
        validation = {}
    return {
        "update_step": int(event.get("update_step", 0)),
        "epoch": int(event.get("epoch", default_epoch)),
        "phase": str(event.get("phase", "end")),
        "reward": float(validation.get("mean_reward", 0.0)),
        "f1": float(validation.get("f1", 0.0)),
        "exact_match": float(
            validation.get("exact_match", validation.get("em", 0.0))
        ),
        "retrieval_hitrate_at_5": float(validation.get("retrieval_hitrate@5", 0.0)),
        "errors": float(validation.get("errors", 0)),
        "by_category": validation.get("by_category", {}),
        "evidence_actions": validation.get("evidence_actions", {}),
    }


def build_train_action_rows(
    run_dir: Path, epoch_rows: Iterable[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    updates = {
        int(row.get("epoch", index)): int(row.get("update_step", index + 1))
        for index, row in enumerate(epoch_rows)
    }
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "train").glob("epoch_*_rollouts.jsonl")):
        try:
            epoch = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        counts = {mask: 0 for mask in ALL_EVIDENCE_MASKS}
        for rollout in read_jsonl(path):
            for action in rollout.get("actions", []) or []:
                mask = str(action.get("mask", ""))
                if mask in counts:
                    counts[mask] += 1
        rows.append(
            {
                "epoch": epoch,
                "update_step": updates.get(epoch, epoch + 1),
                "evidence_actions": {
                    f"mask:{mask}": count for mask, count in counts.items()
                },
            }
        )
    return tuple(rows)


def mask_distribution(
    evidence_actions: Any,
) -> tuple[dict[str, int], dict[str, float], int]:
    values = evidence_actions if isinstance(evidence_actions, dict) else {}
    counts = {
        mask: max(0, int(values.get(f"mask:{mask}", 0)))
        for mask in ALL_EVIDENCE_MASKS
    }
    total = sum(counts.values())
    ratios = {
        mask: (count / total if total else 0.0) for mask, count in counts.items()
    }
    return counts, ratios, total


def mask_label(mask: str) -> str:
    selected = [name for bit, name in zip(mask, EVIDENCE_ORDER) if bit == "1"]
    return f"{mask} ({'+'.join(selected) if selected else 'none'})"


def evidence_level_distribution(
    evidence_actions: Any,
) -> tuple[dict[str, int], dict[str, float], int]:
    counts, _, total = mask_distribution(evidence_actions)
    selected_counts = {
        evidence: sum(
            count
            for mask, count in counts.items()
            if mask[index] == "1"
        )
        for index, evidence in enumerate(EVIDENCE_ORDER)
    }
    ratios = {
        evidence: (count / total if total else 0.0)
        for evidence, count in selected_counts.items()
    }
    return selected_counts, ratios, total


def build_legacy_update_rows(
    run_dir: Path,
    config: dict[str, Any],
    epoch_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    learning_rate = float(config.get("ppo", {}).get("learning_rate", 0.0))
    for epoch_row in epoch_rows:
        epoch = int(epoch_row.get("epoch", len(rows)))
        updates = epoch_row.get("updates", {})
        trace_path = run_dir / "train" / f"epoch_{epoch:03d}_rollouts.jsonl"
        rollouts = read_jsonl(trace_path) if trace_path.exists() else []
        rewards = np.asarray(
            [float(row.get("reward", 0.0)) for row in rollouts], dtype=np.float32
        )
        values = np.asarray(
            [float(row.get("value", 0.0)) for row in rollouts], dtype=np.float32
        )
        diagnostics = critic_diagnostics(values, rewards)
        row: dict[str, Any] = {
            "epoch": epoch,
            "update_step": int(epoch_row.get("update_step", epoch + 1)),
            "question_count": int(
                epoch_row.get("train_question_count", 0)
            ),
            "pg_loss": updates.get("policy_loss"),
            "lr": learning_rate,
            "grad_norm": updates.get("grad_norm"),
            "entropy_loss": updates.get("entropy"),
            "value_loss": updates.get("value_loss"),
            **diagnostics,
        }
        rows.append({key: value for key, value in row.items() if value is not None})
    return rows


def critic_diagnostics(values: np.ndarray, rewards: np.ndarray) -> dict[str, float]:
    if not len(rewards):
        return {
            "predicted_value_mean": 0.0,
            "target_return_mean": 0.0,
            "absolute_value_error": 0.0,
            "explained_variance": 0.0,
            "reward_mean": 0.0,
            "reward_min": 0.0,
            "reward_max": 0.0,
        }
    errors = rewards - values
    reward_variance = float(np.var(rewards))
    return {
        "predicted_value_mean": float(np.mean(values)),
        "target_return_mean": float(np.mean(rewards)),
        "absolute_value_error": float(np.mean(np.abs(errors))),
        "explained_variance": (
            1.0 - float(np.var(errors)) / reward_variance
            if reward_variance > 1e-8
            else 0.0
        ),
        "reward_mean": float(np.mean(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
    }


def upload_to_wandb(
    data: RunData,
    *,
    run_dir: Path,
    project: str,
    entity: str | None,
    name: str,
    run_id: str | None,
    tags: Iterable[str],
    charts_only: bool = False,
) -> str:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed; install it in the active environment first"
        ) from exc

    init_kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "name": name,
        "job_type": "evidence-policy-metrics",
        "tags": list(tags),
        "config": safe_wandb_config(data.config, run_dir),
    }
    if run_id:
        init_kwargs.update({"id": run_id, "resume": "allow"})
    run = wandb.init(**init_kwargs)
    if not charts_only:
        run.define_metric("val/update_step")
        run.define_metric("val/*", step_metric="val/update_step")
        run.define_metric("val/action_ratio/*", step_metric="val/update_step")
        run.define_metric("train/update_step")
        run.define_metric("train/action_ratio/*", step_metric="train/update_step")
        run.define_metric("actor/update_step")
        run.define_metric("actor/*", step_metric="actor/update_step")
        run.define_metric("critic/update_step")
        run.define_metric("critic/*", step_metric="critic/update_step")

        for row in data.validation_rows:
            _, ratios, _ = mask_distribution(row.get("evidence_actions"))
            payload = {
                "val/update_step": row["update_step"],
                "val/reward": row["reward"],
                "val/f1": row["f1"],
                "val/exact_match": row["exact_match"],
                "val/retrieval_hitrate_at_5": row["retrieval_hitrate_at_5"],
                "val/errors": row["errors"],
            }
            payload.update(
                {f"val/action_ratio/{mask}": ratio for mask, ratio in ratios.items()}
            )
            run.log(payload)

        for row in data.train_action_rows:
            _, ratios, _ = mask_distribution(row.get("evidence_actions"))
            run.log(
                {
                    "train/update_step": row["update_step"],
                    "train/epoch": row["epoch"],
                    **{
                        f"train/action_ratio/{mask}": ratio
                        for mask, ratio in ratios.items()
                    },
                }
            )

    predicted_values: list[float] = []
    target_values: list[float] = []
    critic_steps: list[int] = []
    for row in (() if charts_only else data.update_rows):
        update_step = int(row["update_step"])
        payload: dict[str, Any] = {
            "actor/update_step": update_step,
            "critic/update_step": update_step,
        }
        for field in ACTOR_FIELDS:
            add_finite(payload, f"actor/{field}", row.get(field))
        for field in CRITIC_FIELDS:
            wandb_name = {
                "reward_mean": "rewards/mean",
                "reward_min": "rewards/min",
                "reward_max": "rewards/max",
            }.get(field, field)
            add_finite(payload, f"critic/{wandb_name}", row.get(field))
        run.log(payload)
        predicted = row.get("predicted_value_mean")
        target = row.get("target_return_mean")
        if is_finite(predicted) and is_finite(target):
            critic_steps.append(update_step)
            predicted_values.append(float(predicted))
            target_values.append(float(target))

    charts: dict[str, Any] = {}
    category_chart = None if charts_only else build_category_f1_chart(wandb, data)
    if category_chart is not None:
        charts["val/category_f1"] = category_chart
    if not charts_only and critic_steps:
        charts["critic/predicted_value_vs_reward"] = wandb.plot.line_series(
            xs=critic_steps,
            ys=[predicted_values, target_values],
            keys=["predicted value", "reward target"],
            title="Predicted Value vs Reward",
            xname="PPO update step",
        )
    validation_mask_chart = build_mask_ratio_line_chart(
        wandb,
        data.validation_rows,
        title="Validation Evidence Combination Selection Ratio",
    )
    if validation_mask_chart is not None:
        charts["val/action_mask_ratio"] = validation_mask_chart
    train_mask_chart = build_mask_ratio_line_chart(
        wandb,
        data.train_action_rows,
        title="Training Evidence Combination Selection Ratio",
    )
    if train_mask_chart is not None:
        charts["train/action_mask_ratio"] = train_mask_chart
    ratio_table = build_mask_ratio_table(wandb, data)
    if ratio_table is not None:
        charts["evidence/action_mask_ratio_table"] = ratio_table
    test_counts, test_ratios, test_total = mask_distribution(
        data.test_metrics.get("evidence_actions")
    )
    if test_total:
        test_table = wandb.Table(columns=["mask", "combination", "count", "ratio"])
        for mask in ALL_EVIDENCE_MASKS:
            run.summary[f"test/action_ratio/{mask}"] = test_ratios[mask]
            if test_counts[mask]:
                test_table.add_data(
                    mask, mask_label(mask), test_counts[mask], test_ratios[mask]
                )
        charts["test/action_mask_ratio"] = wandb.plot.bar(
            test_table,
            "combination",
            "ratio",
            title="Test Evidence Combination Selection Ratio",
        )
        level_counts, level_ratios, _ = evidence_level_distribution(
            data.test_metrics.get("evidence_actions")
        )
        level_table = wandb.Table(
            columns=["evidence_level", "selected_count", "total", "ratio"]
        )
        for evidence in EVIDENCE_ORDER:
            level_table.add_data(
                evidence,
                level_counts[evidence],
                test_total,
                level_ratios[evidence],
            )
            run.summary[f"test/evidence_level_ratio/{evidence}"] = level_ratios[
                evidence
            ]
        charts["test/evidence_level_ratio"] = wandb.plot.bar(
            level_table,
            "evidence_level",
            "ratio",
            title="Final Evidence Level Selection Ratio",
        )
    if charts:
        run.log(charts)

    for key in ("count", "f1", "exact_match", "em", "mean_reward", "errors"):
        if key in data.test_metrics:
            run.summary[f"test/{key}"] = data.test_metrics[key]
    if "retrieval_hitrate@5" in data.test_metrics:
        run.summary["test/retrieval_hitrate_at_5"] = data.test_metrics[
            "retrieval_hitrate@5"
        ]
    if data.warnings:
        run.summary["upload/warnings"] = list(data.warnings)
    url = run.url
    run.finish()
    return url


def build_category_f1_chart(wandb: Any, data: RunData) -> Any | None:
    categories = sorted(
        {
            category
            for row in data.validation_rows
            for category in row.get("by_category", {})
        }
    )
    if not categories:
        return None
    steps = [int(row["update_step"]) for row in data.validation_rows]
    return wandb.plot.line_series(
        xs=steps,
        ys=[
            [
                float(
                    row.get("by_category", {})
                    .get(category, {})
                    .get("f1", math.nan)
                )
                for row in data.validation_rows
            ]
            for category in categories
        ],
        keys=categories,
        title="Validation Category F1",
        xname="PPO update step",
    )


def build_mask_ratio_line_chart(
    wandb: Any,
    rows: Iterable[dict[str, Any]],
    *,
    title: str,
) -> Any | None:
    materialized = list(rows)
    if not materialized:
        return None
    distributions = [mask_distribution(row.get("evidence_actions")) for row in materialized]
    active_masks = [
        mask
        for mask in ALL_EVIDENCE_MASKS
        if any(counts[mask] for counts, _, _ in distributions)
    ]
    if not active_masks:
        return None
    return wandb.plot.line_series(
        xs=[int(row["update_step"]) for row in materialized],
        ys=[
            [ratios[mask] for _, ratios, _ in distributions]
            for mask in active_masks
        ],
        keys=[mask_label(mask) for mask in active_masks],
        title=title,
        xname="PPO update step",
    )


def build_mask_ratio_table(wandb: Any, data: RunData) -> Any | None:
    table = wandb.Table(
        columns=[
            "split",
            "update_step",
            "epoch",
            "phase",
            "mask",
            "combination",
            "count",
            "ratio",
        ]
    )
    added = False
    for split, rows in (
        ("train", data.train_action_rows),
        ("validation", data.validation_rows),
    ):
        for row in rows:
            counts, ratios, total = mask_distribution(row.get("evidence_actions"))
            if not total:
                continue
            for mask in ALL_EVIDENCE_MASKS:
                table.add_data(
                    split,
                    int(row["update_step"]),
                    int(row.get("epoch", 0)),
                    str(row.get("phase", "epoch")),
                    mask,
                    mask_label(mask),
                    counts[mask],
                    ratios[mask],
                )
                added = True
    test_counts, test_ratios, test_total = mask_distribution(
        data.test_metrics.get("evidence_actions")
    )
    if test_total:
        for mask in ALL_EVIDENCE_MASKS:
            table.add_data(
                "test",
                -1,
                -1,
                "final",
                mask,
                mask_label(mask),
                test_counts[mask],
                test_ratios[mask],
            )
            added = True
    return table if added else None


def safe_wandb_config(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    model = config.get("model", {})
    split = config.get("split", {})
    return {
        "source_run_dir": run_dir.name,
        "seed": config.get("seed"),
        "top_k": config.get("top_k"),
        "model": model.get("name"),
        "policy": config.get("policy", {}),
        "ppo": config.get("ppo", {}),
        "split_sizes": {
            name: len(split.get(name, []))
            for name in ("train", "validation", "test")
        },
    }


def data_summary(data: RunData) -> dict[str, Any]:
    return {
        "epochs": len(data.epoch_rows),
        "actor_critic_points": len(data.update_rows),
        "validation_points": len(data.validation_rows),
        "train_action_points": len(data.train_action_rows),
        "warnings": list(data.warnings),
    }


def add_finite(payload: dict[str, Any], key: str, value: Any) -> None:
    if is_finite(value):
        payload[key] = float(value)


def is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
