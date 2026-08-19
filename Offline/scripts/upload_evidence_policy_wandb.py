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


@dataclass(frozen=True)
class RunData:
    config: dict[str, Any]
    epoch_rows: tuple[dict[str, Any], ...]
    update_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    data = load_run_data(run_dir)
    if args.dry_run:
        print(json.dumps(data_summary(data), ensure_ascii=False, indent=2))
        return
    url = upload_to_wandb(
        data,
        run_dir=run_dir,
        project=args.project,
        entity=args.entity or None,
        name=args.name or run_dir.name,
        run_id=args.run_id or None,
        tags=args.tag,
    )
    print(json.dumps({"url": url, **data_summary(data)}, ensure_ascii=False, indent=2))


def load_run_data(run_dir: Path) -> RunData:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    train_log = run_dir / "train.log"
    if not train_log.exists():
        raise FileNotFoundError(f"Training log not found: {train_log}")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    epoch_rows = tuple(read_jsonl(train_log))
    if not epoch_rows:
        raise ValueError(f"No epoch summaries found in {train_log}")

    warnings: list[str] = []
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

    validation_rows, validation_warnings = build_validation_rows(
        epoch_rows, update_rows
    )
    warnings.extend(validation_warnings)

    test_path = run_dir / "eval" / "test_ppo" / "metrics.json"
    test_metrics = (
        json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
    )
    return RunData(
        config=config,
        epoch_rows=epoch_rows,
        update_rows=update_rows,
        validation_rows=validation_rows,
        test_metrics=test_metrics,
        warnings=tuple(warnings),
    )


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
            validation = event.get("metrics", {})
            update_step = event.get("update_step")
            if update_step is None:
                update_step = epoch + 1
                warnings.append(
                    f"Epoch {epoch} validation has no PPO update step; epoch index was used"
                )
            rows.append(
                {
                    "update_step": int(update_step),
                    "epoch": epoch,
                    "phase": str(event.get("phase", "end")),
                    "reward": float(validation.get("mean_reward", 0.0)),
                    "f1": float(validation.get("f1", 0.0)),
                    "exact_match": float(validation.get("exact_match", 0.0)),
                    "retrieval_hitrate_at_5": float(
                        validation.get("retrieval_hitrate@5", 0.0)
                    ),
                    "errors": float(validation.get("errors", 0)),
                    "by_category": validation.get("by_category", {}),
                }
            )
    return tuple(rows), tuple(warnings)


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
    run.define_metric("val/update_step")
    run.define_metric("val/*", step_metric="val/update_step")
    run.define_metric("actor/update_step")
    run.define_metric("actor/*", step_metric="actor/update_step")
    run.define_metric("critic/update_step")
    run.define_metric("critic/*", step_metric="critic/update_step")

    for row in data.validation_rows:
        run.log(
            {
                "val/update_step": row["update_step"],
                "val/reward": row["reward"],
                "val/f1": row["f1"],
                "val/exact_match": row["exact_match"],
                "val/retrieval_hitrate_at_5": row["retrieval_hitrate_at_5"],
                "val/errors": row["errors"],
            }
        )

    predicted_values: list[float] = []
    target_values: list[float] = []
    critic_steps: list[int] = []
    for row in data.update_rows:
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
    category_chart = build_category_f1_chart(wandb, data)
    if category_chart is not None:
        charts["val/category_f1"] = category_chart
    if critic_steps:
        charts["critic/predicted_value_vs_reward"] = wandb.plot.line_series(
            xs=critic_steps,
            ys=[predicted_values, target_values],
            keys=["predicted value", "reward target"],
            title="Predicted Value vs Reward",
            xname="PPO update step",
        )
    if charts:
        run.log(charts)

    for key in ("count", "f1", "exact_match", "mean_reward", "errors"):
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
