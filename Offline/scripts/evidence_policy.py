#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import secrets
import sys
import time
from collections import Counter
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)

from benchmarks.memgallery_harness.retrieval.query_embedding_cache import (  # noqa: E402
    QueryEmbeddingCache,
    make_query_id,
)
from benchmarks.memgallery_harness.runner.answer_client import (  # noqa: E402
    VLMAnswerClient,
    build_retrieved_memory_evidence,
    query_image_prompt_metadata,
)
from benchmarks.memgallery_harness.runner.metrics import (  # noqa: E402
    ChatPromptTokenCounter,
    add_efficiency_metrics,
    calculate_calls_mb,
    calculate_calls_qa,
    calculate_usage_cost,
    combine_call_metrics,
    load_model_efficiency_profile,
    summarize_results,
    write_efficiency_metrics,
    write_runtime_call_metrics,
)
from benchmarks.memgallery_harness.runner.prompts import (  # noqa: E402
    PPO_EMPTY_PROMPT_VERSION as MEMGALLERY_PPO_EMPTY_PROMPT_VERSION,
    PROMPT_SOURCE as MEMGALLERY_PROMPT_SOURCE,
    PROMPT_VERSION as MEMGALLERY_PROMPT_VERSION,
    build_answer_messages as build_memgallery_answer_messages,
    parse_answer_response as parse_memgallery_answer,
    ppo_empty_prompt_sha256 as memgallery_ppo_empty_prompt_sha256,
    prompt_sha256 as memgallery_prompt_sha256,
    resolve_question_image,
)
from benchmarks.question_filter import (  # noqa: E402
    is_excluded_category,
    parse_excluded_categories,
)
from evidence_policy.evidence import (  # noqa: E402
    EVIDENCE_ORDER,
    DialogueStore,
    EvidenceChainBuilder,
    EvidenceStrategy,
    H2HMemDialogueStore,
    WMADialogueStore,
)
from evidence_policy.episode_sources import iter_source_questions  # noqa: E402
from evidence_policy.policy import EvidenceSelectionPolicy  # noqa: E402
from evidence_policy.ppo import (  # noqa: E402
    PPOBuffer,
    PPOTrainer,
    SlidingCostNormalizer,
    WANDB_SCHEMA_VERSION,
    build_wandb_update_payload,
    build_wandb_validation_payload,
    define_wandb_metrics,
    load_policy_checkpoint,
    save_json,
)
from evidence_policy.retrieval import (  # noqa: E402
    build_graph_index,
    build_wma_prefix_graph_index,
    retrieve_hits,
    resolve_retrieval_settings,
    retrieval_signature,
    retrieval_trace,
    validate_graph_config,
)
from evidence_policy.rollout import (  # noqa: E402
    EvidenceEpisode,
    EvidenceRollout,
    EvidenceSelectionEnv,
    RolloutCache,
)
from evidence_policy.split_manifest import SplitManifestIndex  # noqa: E402
from evidence_policy.vp_store import VPArtifactIndex  # noqa: E402
from hive_mem.retriever import SimpleMemoryIndex  # noqa: E402


ROLLOUT_RETRY_ATTEMPTS = 360
ROLLOUT_RETRY_DELAY_SECONDS = 5.0
TRANSIENT_ENDPOINT_ERROR_MARKERS = (
    "connection refused",
    "failed to establish a new connection",
    "max retries exceeded",
    "connection reset",
    "connection aborted",
    "read timed out",
    "connect timeout",
    "remote end closed connection",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
)
class RealtimeWandbLogger:
    """Best-effort W&B logging with local state as the source of truth."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: Path,
        config: dict[str, Any],
        project: str,
        entity: str,
        name: str,
    ) -> None:
        self.output_dir = output_dir
        self.errors_path = output_dir / "run_control" / "wandb_errors.jsonl"
        self.control_path = output_dir / "run_control" / "wandb.json"
        self.run: Any | None = None
        self.error_count = 0
        if not enabled:
            return
        try:
            import wandb

            control = (
                json.loads(self.control_path.read_text(encoding="utf-8"))
                if self.control_path.is_file()
                else {}
            )
            if control.get("run_id"):
                run_id = str(control["run_id"])
            else:
                run_id = secrets.token_hex(4)
            save_json(
                self.control_path,
                {
                    "project": project,
                    "entity": entity,
                    "name": name,
                    "run_id": run_id,
                    "status": "initializing",
                },
            )
            self.run = wandb.init(
                project=project,
                entity=entity or None,
                name=name,
                id=run_id,
                resume="allow",
                job_type="ppo-training",
                tags=[
                    "hivemem",
                    "ppo",
                    WANDB_SCHEMA_VERSION,
                    str(config.get("benchmark") or config.get("data_source") or ""),
                ],
                config={**config, "wandb_schema_version": WANDB_SCHEMA_VERSION},
                settings=wandb.Settings(init_timeout=15),
            )
            define_wandb_metrics(self.run)
            save_json(
                self.control_path,
                {
                    "project": project,
                    "entity": entity,
                    "name": name,
                    "run_id": run_id,
                    "status": "active",
                    "url": str(getattr(self.run, "url", "") or ""),
                },
            )
        except Exception as exc:
            self.run = None
            self._record_error("init", exc)

    def log_update(self, row: dict[str, Any]) -> None:
        if self.run is None:
            return
        self._log("update", build_wandb_update_payload(row))

    def log_validation(self, event: dict[str, Any], *, epoch: int) -> None:
        if self.run is None:
            return
        self._log(
            "validation", build_wandb_validation_payload(event, epoch=epoch)
        )

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
            control = json.loads(self.control_path.read_text(encoding="utf-8"))
            control["status"] = "finished"
            save_json(self.control_path, control)
        except Exception as exc:
            self._record_error("finish", exc)

    def _log(self, stage: str, payload: dict[str, Any]) -> None:
        try:
            self.run.log(payload)
        except Exception as exc:
            self._record_error(stage, exc)

    def _record_error(self, stage: str, error: Exception) -> None:
        self.error_count += 1
        append_jsonl(
            self.errors_path,
            {
                "timestamp_ns": time.time_ns(),
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
        if self.error_count == 1 or self.error_count % 50 == 0:
            print(
                f"W&B {stage} failed ({type(error).__name__}): {error}; "
                "training continues with local logs",
                file=sys.stderr,
                flush=True,
            )

ALL_ZERO_REWARD = -1.0


def actions_are_all_zero(actions: Sequence[Any]) -> bool:
    return not actions or all(action.bitmask == "00000" for action in actions)


class CostRewardRuntime:
    """Compute shaped rewards while owning one benchmark's normalizer state."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        token_counter: ChatPromptTokenCounter | None = None,
    ) -> None:
        reward = config.get("reward") or {}
        self.enabled = bool(reward.get("cost_enabled", False))
        if self.enabled and "cost_tradeoff_lambda" not in reward:
            raise ValueError(
                "reward.cost_tradeoff_lambda is required when cost reward is enabled"
            )
        self.cost_tradeoff_lambda = float(
            reward.get("cost_tradeoff_lambda", 0.0)
        )
        if self.cost_tradeoff_lambda < 0:
            raise ValueError("reward.cost_tradeoff_lambda must be non-negative")
        self.normalizer = SlidingCostNormalizer(
            window_size=int(reward.get("window_size", 512)),
            min_window_size=int(reward.get("min_window_size", 128)),
            lower_quantile=float(reward.get("lower_quantile", 0.05)),
            upper_quantile=float(
                reward.get("upper_quantile", reward.get("quantile", 0.95))
            ),
            initial_range_floor_ratio=float(
                reward.get(
                    "initial_range_floor_ratio",
                    reward.get("initial_floor_ratio", 0.25),
                )
            ),
            range_epsilon=float(reward.get("range_epsilon", 1e-12)),
            std_epsilon=float(reward.get("std_epsilon", 1e-8)),
            cost_transform=str(
                reward.get("cost_transform", "sqrt_incremental")
            ),
        )
        self.input_price = 0.0
        self.output_price = 0.0
        self.token_counter = token_counter
        self._base_prompt_tokens_by_query: dict[str, int] = {}
        if self.enabled:
            efficiency_config = Path(
                config.get("efficiency_config")
                or ROOT / "configs" / "model_efficiency.json"
            )
            profile = load_model_efficiency_profile(
                efficiency_config, str(config["model"]["name"])
            )
            self.input_price = float(profile["pricing"]["input_per_million_usd"])
            self.output_price = float(profile["pricing"]["output_per_million_usd"])
            if self.token_counter is None:
                self.token_counter = ChatPromptTokenCounter(
                    str(reward.get("tokenizer_name") or config["model"]["name"])
                )

    def snapshot(self) -> dict[str, Any]:
        return self.normalizer.snapshot()

    def apply(
        self,
        rollout: EvidenceRollout,
        episode: EvidenceEpisode,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> bool:
        quality = float(
            rollout.reward if rollout.quality_reward is None else rollout.quality_reward
        )
        rollout.quality_reward = quality
        rollout.cost_weight = (
            self.cost_tradeoff_lambda if self.enabled else 0.0
        )
        if not self.enabled:
            rollout.reward = (
                ALL_ZERO_REWARD
                if actions_are_all_zero(rollout.actions)
                else quality
            )
            return not bool(rollout.error)
        if rollout.error:
            rollout.cost_error = "answer rollout failed"
            return False
        if rollout.answer_usage is None:
            rollout.cost_error = "missing exact answer usage"
            return False
        try:
            raw_cost = calculate_usage_cost(
                rollout.answer_usage,
                input_price=self.input_price,
                output_price=self.output_price,
            )
        except (TypeError, ValueError) as exc:
            rollout.cost_error = str(exc)
            return False

        base_tokens = self._base_prompt_tokens(episode)
        base_cost = calculate_usage_cost(
            {
                "prompt_tokens": base_tokens,
                "completion_tokens": 0,
                "total_tokens": base_tokens,
            },
            input_price=self.input_price,
            output_price=self.output_price,
        )
        incremental = max(raw_cost - base_cost, 0.0)
        state = snapshot or self.normalizer.snapshot()
        normalized = self.normalizer.normalize(incremental, snapshot=state)
        transformed = self.normalizer.transform(incremental)
        effective_min = state.get("effective_transformed_cost_min")
        effective_max = state.get("effective_transformed_cost_max")
        scale_alpha = state.get("cost_scale_alpha")
        effective_weight = (
            self.cost_tradeoff_lambda * float(scale_alpha)
            if scale_alpha is not None
            else 0.0
        )
        rollout.raw_cost = raw_cost
        rollout.base_cost = base_cost
        rollout.cost_min = (
            base_cost + self.normalizer.inverse_transform(float(effective_min))
            if effective_min is not None
            else None
        )
        rollout.incremental_cost = incremental
        rollout.transformed_cost = transformed
        rollout.cost_max = (
            base_cost + self.normalizer.inverse_transform(float(effective_max))
            if effective_max is not None
            else None
        )
        rollout.normalized_cost = normalized
        rollout.cost_scale_alpha = (
            float(scale_alpha) if scale_alpha is not None else None
        )
        rollout.task_reward_std = (
            float(state["task_reward_std"])
            if state.get("task_reward_std") is not None
            else None
        )
        rollout.normalized_cost_std = (
            float(state["normalized_cost_std"])
            if state.get("normalized_cost_std") is not None
            else None
        )
        rollout.effective_cost_weight = (
            effective_weight if scale_alpha is not None else None
        )
        rollout.cost_window_count = int(state["window_count"])
        rollout.cost_normalizer_active = bool(state["active"])
        rollout.reward = (
            ALL_ZERO_REWARD
            if actions_are_all_zero(rollout.actions)
            else quality - effective_weight * normalized
        )
        rollout.cost_error = ""
        return True

    def commit(
        self,
        incremental_costs: Sequence[float],
        quality_rewards: Sequence[float],
    ) -> None:
        if self.enabled:
            self.normalizer.extend(incremental_costs, quality_rewards)

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "cost_tradeoff_lambda": self.cost_tradeoff_lambda,
            "input_price_per_million_usd": self.input_price,
            "output_price_per_million_usd": self.output_price,
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state: Any) -> None:
        if not isinstance(state, dict):
            raise ValueError("Checkpoint is missing cost reward state")
        if bool(state.get("enabled")) != self.enabled:
            raise ValueError("Checkpoint cost reward enabled state does not match config")
        if (
            float(state.get("cost_tradeoff_lambda", -1.0))
            != self.cost_tradeoff_lambda
        ):
            raise ValueError(
                "Checkpoint cost reward lambda does not match config"
            )
        self.normalizer.load_state_dict(state.get("normalizer") or {})

    def _base_prompt_tokens(self, episode: EvidenceEpisode) -> int:
        if episode.query_id in self._base_prompt_tokens_by_query:
            return self._base_prompt_tokens_by_query[episode.query_id]
        if episode.answer_messages_builder is None or self.token_counter is None:
            raise ValueError(
                "BaseCost requires an answer messages builder and token counter"
            )
        messages = episode.answer_messages_builder(
            [], allow_empty_evidence=True
        )
        image_paths = (
            [str(episode.query_image["path"])]
            if episode.query_image and episode.query_image.get("path")
            else []
        )
        count = self.token_counter.count(messages, image_paths=image_paths)
        self._base_prompt_tokens_by_query[episode.query_id] = count
        return count

def summarize_cost_rewards(rollouts: Sequence[EvidenceRollout]) -> dict[str, float]:
    if not rollouts:
        return {}
    valid = [
        rollout
        for rollout in rollouts
        if not rollout.cost_error and rollout.raw_cost is not None
    ]
    result = {
        "all_zero_rollout_rate": float(
            np.mean(
                [
                    actions_are_all_zero(rollout.actions)
                    for rollout in rollouts
                ]
            )
        )
    }
    if not valid:
        return result

    def mean(field: str) -> float:
        return float(np.mean([float(getattr(row, field)) for row in valid]))

    result.update(
        {
            "quality_reward_mean": mean("quality_reward"),
            "final_reward_mean": mean("reward"),
            "raw_cost_mean": mean("raw_cost"),
            "base_cost_mean": mean("base_cost"),
            "incremental_cost_mean": mean("incremental_cost"),
            "transformed_cost_mean": mean("transformed_cost"),
            "normalized_cost_mean": mean("normalized_cost"),
            "cost_penalty_mean": float(
                np.mean(
                    [
                        float(row.effective_cost_weight or 0.0)
                        * float(row.normalized_cost or 0.0)
                        for row in valid
                    ]
                )
            ),
            "cost_window_count": float(valid[0].cost_window_count or 0),
            "cost_normalizer_active": float(
                bool(valid[0].cost_normalizer_active)
            ),
        }
    )
    if (
        valid[0].cost_max is not None
        and valid[0].cost_min is not None
        and valid[0].base_cost is not None
    ):
        result["cost_min_mean"] = mean("cost_min")
        result["cost_max_mean"] = mean("cost_max")
        result["task_reward_std"] = float(valid[0].task_reward_std)
        result["normalized_cost_std"] = float(valid[0].normalized_cost_std)
        result["cost_scale_alpha"] = float(valid[0].cost_scale_alpha)
        result["effective_cost_weight"] = float(
            valid[0].effective_cost_weight
        )
        result["cost_low_clip_rate"] = float(
            np.mean([float(row.normalized_cost) <= 0.0 for row in valid])
        )
        result["cost_high_clip_rate"] = float(
            np.mean([float(row.normalized_cost) >= 1.0 for row in valid])
        )
        result["cost_saturation_rate"] = result["cost_high_clip_rate"]
    return result


def update_ppo_batch(
    trainer: PPOTrainer,
    buffer: PPOBuffer,
    cost_runtime: CostRewardRuntime,
    batch_rollouts: Sequence[EvidenceRollout],
) -> dict[str, float]:
    metrics = trainer.update(buffer)
    metrics.update(summarize_cost_rewards(batch_rollouts))
    snapshot = cost_runtime.snapshot()
    for source, target in (
        ("percentile_transformed_cost_min", "cost_percentile_transformed_min"),
        ("percentile_transformed_cost_max", "cost_percentile_transformed_max"),
        ("effective_transformed_cost_min", "cost_effective_transformed_min"),
        ("effective_transformed_cost_max", "cost_effective_transformed_max"),
        ("effective_transformed_cost_range", "cost_effective_transformed_range"),
        ("task_reward_std", "task_reward_std"),
        ("normalized_cost_std", "normalized_cost_std"),
        ("cost_scale_alpha", "cost_scale_alpha"),
    ):
        value = snapshot.get(source)
        if value is not None:
            metrics[target] = float(value)
    valid_cost_rollouts = [
        rollout
        for rollout in batch_rollouts
        if rollout.incremental_cost is not None
        and rollout.quality_reward is not None
    ]
    cost_runtime.commit(
        [float(rollout.incremental_cost) for rollout in valid_cost_rollouts],
        [float(rollout.quality_reward) for rollout in valid_cost_rollouts],
    )
    metrics["cost_window_count_after"] = float(
        cost_runtime.snapshot()["window_count"]
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO evidence selection for benchmark MAUs")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "evidence_policy.json")
    )
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Conversation-level train/val/test manifest; overrides config split lists",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Override the configured output directory for an isolated run",
    )
    parser.add_argument(
        "--model-base-url",
        default="",
        help="Override the configured VLM endpoint for this run",
    )
    parser.add_argument(
        "--memory-bank",
        default="",
        help="Override the configured memory bank for an isolated graph ablation",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=("vector", "random_append", "graph_append"),
        default="",
        help="Override the configured retrieval mode for a controlled ablation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Override vector top-k for a controlled ablation",
    )
    parser.add_argument(
        "--append-k",
        type=int,
        default=None,
        help="Override graph/random append count for a controlled ablation",
    )
    parser.add_argument(
        "--degree-cap",
        type=int,
        default=None,
        help="Record the graph degree cap used by an isolated graph ablation",
    )
    parser.add_argument(
        "--retrieval-seed",
        type=int,
        default=None,
        help="Global random-append seed; per-question seeds are derived by SHA256",
    )
    parser.add_argument(
        "--ppo-force-visual-evidence",
        action="store_true",
        help="During deterministic PPO evaluation, add every available image and VP",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("prepare-split", help="Create balanced benchmark splits")
    split_parser.add_argument("--trials", type=int, default=20000)

    subparsers.add_parser("audit-vp", help="Audit memory-image coverage in the VP run")

    train_parser = subparsers.add_parser("train", help="Train the PPO policy")
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--epochs", type=int, default=0)
    train_parser.add_argument("--max-train-episodes", type=int, default=0)
    train_parser.add_argument("--validation-limit", type=int, default=0)
    train_parser.add_argument("--resume", default="")
    train_parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    train_parser.add_argument("--wandb-project", default="hivemem-evidence-policy-v2")
    train_parser.add_argument("--wandb-entity", default="")
    train_parser.add_argument("--wandb-name", default="")

    eval_parser = subparsers.add_parser("eval", help="Evaluate one evidence strategy")
    eval_parser.add_argument(
        "--strategy", choices=[item.value for item in EvidenceStrategy], required=True
    )
    eval_parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    eval_parser.add_argument("--checkpoint", default="")
    eval_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    eval_parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        config["output_dir"] = str(
            output_dir if output_dir.is_absolute() else (ROOT / output_dir).resolve()
        )
    if args.model_base_url:
        config["model"]["base_url"] = str(args.model_base_url).rstrip("/")
    if args.memory_bank:
        config["memory_bank"] = str(Path(args.memory_bank).expanduser().resolve())
    if args.retrieval_mode:
        config["retrieval_mode"] = args.retrieval_mode
    if args.top_k:
        config["top_k"] = int(args.top_k)
    if args.append_k is not None:
        retrieval_mode = str(
            args.retrieval_mode or config.get("retrieval_mode") or "graph_append"
        )
        if retrieval_mode == "graph_append":
            graph_options = config.get("graph_options")
            if graph_options is False:
                raise ValueError("--append-k cannot enable disabled graph_options")
            config["graph_options"] = dict(graph_options or {})
            config["graph_options"]["append_k"] = int(args.append_k)
        elif retrieval_mode == "random_append":
            config["random_append_k"] = int(args.append_k)
        else:
            raise ValueError("--append-k requires graph_append or random_append mode")
    if args.degree_cap is not None:
        if args.degree_cap < 0:
            raise ValueError("--degree-cap cannot be negative")
        graph_options = config.get("graph_options")
        if graph_options is False:
            raise ValueError("--degree-cap cannot override disabled graph_options")
        config["graph_options"] = dict(graph_options or {})
        config["graph_options"]["degree_cap"] = int(args.degree_cap)
    if args.retrieval_seed is not None:
        config["retrieval_seed"] = int(args.retrieval_seed)
    if args.ppo_force_visual_evidence:
        config.setdefault("evidence", {})["ppo_force_visual_evidence"] = True
    config["qa_latency_denominator"] = str(
        config.get("qa_latency_denominator", "queries")
    )
    if args.split_manifest:
        config["split_manifest"] = str(Path(args.split_manifest).expanduser().resolve())
    if args.command == "train" and args.validation_limit:
        config["ppo"]["validation_limit"] = int(args.validation_limit)
    if config.get("split_manifest"):
        split_index = SplitManifestIndex(config["split_manifest"])
        config["split_manifest"] = str(split_index.path)
        config["split_manifest_file_sha256"] = split_index.file_sha256
    if args.command == "prepare-split":
        prepare_split(config, config_path, trials=args.trials)
    elif args.command == "audit-vp":
        audit_vp(config)
    elif args.command == "train":
        train(config, args)
    else:
        evaluate_command(config, args)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "data_dir",
        "memory_bank",
        "query_cache",
        "output_dir",
        "workspace_root",
    ):
        if not config.get(key):
            continue
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else (ROOT / value).resolve())
    if config.get("profiles_file"):
        value = Path(config["profiles_file"])
        config["profiles_file"] = str(
            value if value.is_absolute() else (ROOT / value).resolve()
        )
    if config.get("efficiency_config"):
        value = Path(config["efficiency_config"])
        config["efficiency_config"] = str(
            value if value.is_absolute() else (ROOT / value).resolve()
        )
    if config.get("split_manifest"):
        value = Path(config["split_manifest"])
        config["split_manifest"] = str(
            value if value.is_absolute() else (path.parent / value).resolve()
        )
    evidence = config.setdefault("evidence", {})
    if evidence.get("vp_run_dir"):
        value = Path(evidence["vp_run_dir"])
        evidence["vp_run_dir"] = str(
            value if value.is_absolute() else (ROOT / value).resolve()
        )
    return config


def audit_vp(config: dict[str, Any]) -> None:
    evidence = config.get("evidence") or {}
    index = VPArtifactIndex(
        evidence["vp_run_dir"],
        max_vps_per_image=int(evidence.get("max_vps_per_image", 0)),
    )
    paths = memory_image_paths(config["memory_bank"])
    report = {
        "vp_run_id": index.run_id,
        "vp_signature": index.signature,
        **index.audit(paths),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def memory_image_paths(memory_bank: str | Path) -> list[str]:
    paths: list[str] = []
    for memories_path in (Path(memory_bank) / "datasets").glob("*/memories.jsonl"):
        with memories_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                paths.extend(
                    str(value)
                    for value in (row.get("metadata") or {}).get("image_paths", [])
                    if str(value)
                )
    return paths


def prepare_split(config: dict[str, Any], config_path: Path, *, trials: int) -> None:
    if config.get("split_manifest"):
        index = SplitManifestIndex(config["split_manifest"])
        source = evidence_data_source(config)
        if source not in index.data_sources:
            raise ValueError(
                f"Configured data_source {source!r} is absent from {index.path}"
            )
        report = index.summary()["data_sources"][source]
        print(
            json.dumps(
                {
                    "mode": "manifest",
                    "data_source": source,
                    "manifest": str(index.path),
                    "manifest_file_sha256": index.file_sha256,
                    "report": report,
                },
                indent=2,
            )
        )
        return
    data_dir = Path(config["data_dir"])
    benchmark = str(config.get("benchmark", "memgallery")).lower()
    stats = dataset_statistics(
        data_dir,
        benchmark=benchmark,
        excluded_categories=parse_excluded_categories(
            config.get("excluded_categories", ["AR"] if benchmark == "memgallery" else [])
        ),
    )
    names = sorted(stats)
    if benchmark == "memgallery" and len(names) != 20:
        raise ValueError(f"Expected 20 Mem-Gallery datasets, found {len(names)}")
    configured_sizes = config.get("split_sizes")
    if configured_sizes:
        sizes = tuple(int(value) for value in configured_sizes)
    elif benchmark == "memgallery":
        sizes = (12, 4, 4)
    else:
        train_size = int(len(names) * 0.6)
        validation_size = int(len(names) * 0.2)
        sizes = (train_size, validation_size, len(names) - train_size - validation_size)
    if len(sizes) != 3 or sum(sizes) != len(names):
        raise ValueError(f"Split sizes {sizes} do not cover {len(names)} datasets")
    rng = random.Random(int(config["seed"]))
    best: tuple[float, list[str]] | None = None
    for _ in range(max(1, int(trials))):
        candidate = rng.sample(names, len(names))
        score = split_score(candidate, stats, sizes)
        if best is None or score < best[0]:
            best = (score, candidate)
    assert best is not None
    ordered = best[1]
    split = {
        "train": sorted(ordered[: sizes[0]]),
        "validation": sorted(ordered[sizes[0] : sizes[0] + sizes[1]]),
        "test": sorted(ordered[-sizes[2] :]),
    }
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config["split"] = split
    save_json(config_path, raw_config)
    report = {name: aggregate_split(rows, stats) for name, rows in split.items()}
    print(json.dumps({"score": best[0], "split": split, "report": report}, indent=2))


def dataset_statistics(
    data_dir: Path,
    *,
    benchmark: str = "memgallery",
    excluded_categories: frozenset[str] = frozenset(),
) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = {}
    if benchmark == "wma":
        from embedding.chunk_builder import iter_wma_sample_files

        paths = iter_wma_sample_files(data_dir)
    else:
        paths = sorted((data_dir / "dialog").glob("*.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if benchmark == "wma":
            categories = Counter(
                str(row.get("question_type_abbrev", ""))
                for checkpoint in payload.get("qa_checkpoints", []) or []
                for row in checkpoint.get("questions", []) or []
                if not is_excluded_category(
                    row.get("question_type_abbrev", ""), excluded_categories
                )
            )
            name = str(payload["sample_id"])
        else:
            categories = Counter(
                str(row.get("point", ""))
                for row in payload.get("human-annotated QAs", [])
                if not is_excluded_category(row.get("point", ""), excluded_categories)
            )
            name = path.stem
        categories["__total__"] = sum(categories.values())
        stats[name] = categories
    return stats


def split_score(
    ordered_names: Sequence[str],
    stats: dict[str, Counter[str]],
    sizes: tuple[int, int, int],
) -> float:
    total = aggregate_split(ordered_names, stats)
    categories = sorted(total)
    score = 0.0
    start = 0
    for size in sizes:
        rows = ordered_names[start : start + size]
        actual = aggregate_split(rows, stats)
        fraction = size / len(ordered_names)
        for category in categories:
            target = total[category] * fraction
            score += ((actual[category] - target) / (target + 1.0)) ** 2
        start += size
    return score


def aggregate_split(
    names: Iterable[str], stats: dict[str, Counter[str]]
) -> Counter[str]:
    result: Counter[str] = Counter()
    for name in names:
        result.update(stats[name])
    return result


def train(config: dict[str, Any], args: argparse.Namespace) -> None:
    validate_runtime(config, require_split=True)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    policy = build_policy(config, device)
    trainer = build_trainer(config, policy)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)
    cost_runtime = CostRewardRuntime(config)
    wandb_logger = RealtimeWandbLogger(
        enabled=bool(args.wandb),
        output_dir=output_dir,
        config=config,
        project=str(args.wandb_project),
        entity=str(args.wandb_entity),
        name=str(args.wandb_name or output_dir.name),
    )
    ppo_metrics_path = output_dir / "ppo_metrics.jsonl"
    start_epoch = 0
    train_question_count = 0
    if args.resume:
        state = trainer.load_checkpoint(args.resume)
        if not resume_configs_match(state.get("config"), config):
            raise ValueError(
                "Checkpoint configuration does not match the current evidence-policy "
                "config (only output_dir, model.base_url, and the training-only "
                "ppo.skip_invalid_response flag may differ for recovery)"
            )
        start_epoch = int(state["epoch"]) + 1
        train_question_count = int(
            state["extra"].get(
                "train_question_count",
                state["extra"].get("train_question_step", 0),
            )
        )
        cost_runtime.load_state_dict(state["extra"].get("cost_reward"))
        reconciliation = reconcile_ppo_metrics_for_resume(
            ppo_metrics_path,
            checkpoint_update_step=trainer.update_steps,
        )
        if reconciliation["removed_rows"]:
            print(json.dumps({"resume_metrics_reconciliation": reconciliation}))
    client, env = build_environment(config)
    client.assert_model_available()
    query_cache = QueryEmbeddingCache(
        config["query_cache"], expected_dim=int(config["policy"]["embedding_dim"])
    )
    profiles = load_profiles(config)
    epochs = int(args.epochs or config["ppo"]["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    initial_validation = prepare_initial_validation(
        config,
        env,
        query_cache,
        profiles,
        policy,
        trainer,
        cost_runtime,
        output_dir=output_dir,
        device=device,
        enabled=(
            start_epoch == 0
            and bool(config["ppo"].get("validation_at_start", False))
        ),
    )
    if initial_validation is not None:
        wandb_logger.log_validation(initial_validation, epoch=0)
    if start_epoch == 0 and ppo_metrics_path.exists():
        ppo_metrics_path.unlink()
    for epoch in range(start_epoch, epochs):
        buffer = PPOBuffer()
        batch_rollouts: list[EvidenceRollout] = []
        cost_snapshot = cost_runtime.snapshot()
        episode_iter = iter_episodes(config, "train", query_cache, profiles)
        episodes = list(
            islice(episode_iter, args.max_train_episodes)
            if args.max_train_episodes
            else episode_iter
        )
        random.shuffle(episodes)
        validation_points = validation_checkpoints(
            len(episodes),
            interval_fraction=float(
                config["ppo"].get("validation_interval_fraction", 1.0)
            ),
            rollout_batch_size=int(config["ppo"]["rollout_batch_size"]),
        )
        updates: list[dict[str, float]] = []
        rewards: list[float] = []
        train_rollouts: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = (
            [initial_validation]
            if epoch == 0 and initial_validation is not None
            else []
        )
        failed_rollouts = 0
        skipped_invalid_responses = 0
        for episode_index, episode in enumerate(episodes, start=1):
            train_question_count += 1
            with torch.no_grad():
                rollout = rollout_with_endpoint_recovery(
                    env,
                    episode,
                    EvidenceStrategy.PPO,
                    policy=policy,
                    deterministic=False,
                    allow_skip_invalid_response=bool(
                        config["ppo"].get("skip_invalid_response", False)
                    ),
                )
            step = rollout.policy_step
            assert step is not None
            cost_eligible = cost_runtime.apply(
                rollout, episode, snapshot=cost_snapshot
            )
            train_rollouts.append(rollout_record(rollout, episode))
            skipped_invalid_responses += int(rollout.skipped_invalid_response)
            if rollout.error or not cost_eligible:
                failed_rollouts += 1
            else:
                buffer.add(
                    rollout.observation,
                    rollout.actions,
                    old_log_prob=float(step.joint_log_prob.cpu()),
                    old_value=float(step.value.cpu()),
                    reward=rollout.reward,
                )
                rewards.append(rollout.reward)
                batch_rollouts.append(rollout)
                if len(buffer) >= int(config["ppo"]["rollout_batch_size"]):
                    metrics = update_ppo_batch(
                        trainer, buffer, cost_runtime, batch_rollouts
                    )
                    updates.append(metrics)
                    update_row = {
                        "epoch": epoch,
                        "question_count": train_question_count,
                        "update_step": trainer.update_steps,
                        **metrics,
                    }
                    append_jsonl(
                        ppo_metrics_path,
                        update_row,
                    )
                    wandb_logger.log_update(update_row)
                    buffer.clear()
                    batch_rollouts.clear()
                    cost_snapshot = cost_runtime.snapshot()
            validation_phase = validation_points.get(episode_index)
            if validation_phase is not None:
                validation_event = run_training_validation(
                    config,
                    env,
                    query_cache,
                    profiles,
                    policy,
                    output_dir=output_dir,
                    epoch=epoch,
                    phase=validation_phase,
                    update_step=trainer.update_steps,
                    train_question_count=train_question_count,
                    cost_runtime=cost_runtime,
                )
                validations.append(validation_event)
                wandb_logger.log_validation(validation_event, epoch=epoch)
        if len(buffer):
            metrics = update_ppo_batch(
                trainer, buffer, cost_runtime, batch_rollouts
            )
            updates.append(metrics)
            update_row = {
                "epoch": epoch,
                "question_count": train_question_count,
                "update_step": trainer.update_steps,
                **metrics,
            }
            append_jsonl(
                ppo_metrics_path,
                update_row,
            )
            wandb_logger.log_update(update_row)
            buffer.clear()
            batch_rollouts.clear()
        end_validation = run_training_validation(
            config,
            env,
            query_cache,
            profiles,
            policy,
            output_dir=output_dir,
            epoch=epoch,
            phase="end",
            update_step=trainer.update_steps,
            train_question_count=train_question_count,
            cost_runtime=cost_runtime,
        )
        validations.append(end_validation)
        wandb_logger.log_validation(end_validation, epoch=epoch)
        checkpoint = output_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
        train_trace = output_dir / "train" / f"epoch_{epoch:03d}_rollouts.jsonl"
        write_jsonl(train_trace, train_rollouts)
        trainer.save_checkpoint(
            checkpoint,
            config=config,
            epoch=epoch,
            extra={
                "validation": end_validation["metrics"],
                "validations": validations,
                "train_question_count": train_question_count,
                "cost_reward": cost_runtime.state_dict(),
            },
        )
        summary = {
            "epoch": epoch,
            "update_step": trainer.update_steps,
            "train_question_count": train_question_count,
            "train_episodes": len(episodes),
            "successful_rollouts": len(rewards),
            "failed_rollouts": failed_rollouts,
            "skipped_invalid_responses": skipped_invalid_responses,
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "updates": mean_dicts(updates),
            "validation": end_validation["metrics"],
            "validations": validations,
            "checkpoint": str(checkpoint),
            "rollouts": str(train_trace),
            "validation_rollouts": end_validation["rollouts"],
        }
        print(json.dumps(summary, ensure_ascii=False))
    wandb_logger.finish()


def validation_checkpoints(
    total_episodes: int,
    *,
    interval_fraction: float,
    rollout_batch_size: int,
) -> dict[int, str]:
    if not 0.0 < interval_fraction <= 1.0:
        raise ValueError("validation_interval_fraction must be in (0, 1]")
    if rollout_batch_size <= 0:
        raise ValueError("rollout_batch_size must be positive")
    checkpoints: dict[int, str] = {}
    multiple = 1
    while multiple * interval_fraction < 1.0 - 1e-9:
        fraction = multiple * interval_fraction
        target = int(total_episodes * fraction)
        aligned = target - target % rollout_batch_size
        if 0 < aligned < total_episodes:
            phase = "half" if abs(fraction - 0.5) < 1e-9 else (
                f"fraction_{fraction:g}".replace(".", "_")
            )
            checkpoints.setdefault(aligned, phase)
        multiple += 1
    return checkpoints


def run_training_validation(
    config: dict[str, Any],
    env: EvidenceSelectionEnv,
    query_cache: QueryEmbeddingCache,
    profiles: dict[str, str],
    policy: EvidenceSelectionPolicy,
    *,
    output_dir: Path,
    epoch: int,
    phase: str,
    update_step: int,
    train_question_count: int,
    cost_runtime: CostRewardRuntime | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    was_training = policy.training
    policy.eval()
    try:
        validation = evaluate(
            config,
            "validation",
            EvidenceStrategy.PPO,
            env,
            query_cache,
            profiles,
            policy=policy,
            deterministic=deterministic,
            limit=int(config["ppo"].get("validation_limit", 0)),
            cost_runtime=cost_runtime,
        )
    finally:
        if was_training:
            policy.train()
    if phase == "initial":
        filename = "initial_rollouts.jsonl"
    elif phase == "end":
        filename = f"epoch_{epoch:03d}_rollouts.jsonl"
    else:
        filename = f"epoch_{epoch:03d}_{phase}_rollouts.jsonl"
    trace = output_dir / "validation" / filename
    write_jsonl(trace, validation["rollouts"])
    event = {
        "phase": phase,
        "update_step": int(update_step),
        "train_question_count": int(train_question_count),
        "metrics": validation["metrics"],
        "rollouts": str(trace),
    }
    if phase != "initial":
        save_json(trace.with_name(filename.replace("rollouts.jsonl", "metrics.json")), event)
    return event


def prepare_initial_validation(
    config: dict[str, Any],
    env: EvidenceSelectionEnv,
    query_cache: QueryEmbeddingCache,
    profiles: dict[str, str],
    policy: EvidenceSelectionPolicy,
    trainer: PPOTrainer,
    cost_runtime: CostRewardRuntime | None = None,
    *,
    output_dir: Path,
    device: torch.device,
    enabled: bool,
) -> dict[str, Any] | None:
    """Run or recover the real pre-update validation point.

    The event is persisted before the first training rollout so retries cannot
    lose or silently change the step-zero baseline.
    """
    if not enabled:
        return None
    if cost_runtime is None:
        cost_runtime = CostRewardRuntime(config)
    metrics_path = output_dir / "validation" / "initial_metrics.json"
    checkpoint_path = output_dir / "checkpoints" / "initial.pt"
    signature = initial_validation_signature(config, device)
    if metrics_path.is_file():
        event = json.loads(metrics_path.read_text(encoding="utf-8"))
        if str(event.get("run_signature", "")) != signature:
            raise ValueError(
                "Stored initial validation does not match the current config/device: "
                f"{metrics_path}"
            )
        if (
            str(event.get("phase", "")) != "initial"
            or int(event.get("update_step", -1)) != 0
            or int(event.get("train_question_count", -1)) != 0
        ):
            raise ValueError(f"Invalid initial validation event: {metrics_path}")
        if not checkpoint_path.is_file():
            if trainer.update_steps != 0:
                raise ValueError("Cannot recover an initial checkpoint after PPO updates")
            trainer.save_checkpoint(
                checkpoint_path,
                config=config,
                epoch=-1,
                extra={
                    "initial_validation": event,
                    "device": str(device),
                    "cost_reward": cost_runtime.state_dict(),
                },
            )
        return event

    if trainer.update_steps != 0:
        raise ValueError("Initial validation requires an untrained PPO policy")
    validation_seed = int(config["seed"]) + 10_000_019
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(validation_seed)
        event = run_training_validation(
            config,
            env,
            query_cache,
            profiles,
            policy,
            output_dir=output_dir,
            epoch=0,
            phase="initial",
            update_step=0,
            train_question_count=0,
            cost_runtime=cost_runtime,
            deterministic=False,
        )
    event.update(
        {
            "seed": int(config["seed"]),
            "sampling_seed": validation_seed,
            "sampling_mode": "independent_bernoulli",
            "initial_action_probability": float(
                config.get("policy", {}).get("initial_action_probability", 0.5)
            ),
            "device": str(device),
            "run_signature": signature,
        }
    )
    save_json(metrics_path, event)
    trainer.save_checkpoint(
        checkpoint_path,
        config=config,
        epoch=-1,
        extra={
            "initial_validation": event,
            "device": str(device),
            "cost_reward": cost_runtime.state_dict(),
        },
    )
    return event


def initial_validation_signature(
    config: dict[str, Any], device: torch.device | str
) -> str:
    payload = {
        "config": config,
        "device": str(device),
        "initial_validation_version": "independent_bernoulli_v1",
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def evaluate_command(config: dict[str, Any], args: argparse.Namespace) -> None:
    validate_runtime(config, require_split=True)
    seed_everything(int(config["seed"]))
    strategy = EvidenceStrategy(args.strategy)
    policy = None
    checkpoint_state: dict[str, Any] | None = None
    if strategy is EvidenceStrategy.PPO:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --strategy ppo")
        policy = build_policy(config, torch.device(args.device))
        checkpoint_state = load_policy_checkpoint(
            policy, args.checkpoint, device=args.device
        )
        policy.eval()
    cost_runtime = CostRewardRuntime(config)
    checkpoint_cost_state = (
        (checkpoint_state.get("extra") or {}).get("cost_reward")
        if checkpoint_state is not None
        else None
    )
    if checkpoint_cost_state is not None:
        cost_runtime.load_state_dict(checkpoint_cost_state)
    client, env = build_environment(config)
    client.assert_model_available()
    query_cache = QueryEmbeddingCache(
        config["query_cache"], expected_dim=int(config["policy"]["embedding_dim"])
    )
    result = evaluate(
        config,
        args.split,
        strategy,
        env,
        query_cache,
        load_profiles(config),
        policy=policy,
        deterministic=True,
        limit=args.limit,
        cost_runtime=cost_runtime,
    )
    output = Path(config["output_dir"]) / "eval" / f"{args.split}_{strategy.value}"
    output.mkdir(parents=True, exist_ok=True)
    save_json(Path(config["output_dir"]) / "config.json", config)
    sample_ids = sorted(
        {
            str(row.get("dataset") or "").strip()
            for row in result["rollouts"]
            if str(row.get("dataset") or "").strip()
        }
    )
    efficiency_config = Path(
        config.get("efficiency_config")
        or ROOT / "configs" / "model_efficiency.json"
    )
    runtime_calls = write_runtime_call_metrics(
        [],
        output,
        result["rollouts"],
        sample_id_field="dataset",
        sample_ids=sample_ids,
    )
    # Runtime tracing owns the per-call audit files, but QA-only evaluation
    # has no build trace paths to pass to it.  Keep the canonical MB and QA
    # totals calculated by evaluate(), and attach retrieval calls separately.
    canonical_calls = result["metrics"]["calls"]
    canonical_calls["retrieval"] = runtime_calls["retrieval"]
    result["metrics"]["calls"] = canonical_calls
    save_json(output / "call_metrics.json", canonical_calls)
    efficiency = write_efficiency_metrics(
        output,
        result["rollouts"],
        sample_id_field="dataset",
        sample_ids=sample_ids,
        model=str(config["model"]["name"]),
        config_path=efficiency_config,
        hivemem_index_root=config["memory_bank"],
        qa_latency_denominator=str(
            config.get("qa_latency_denominator", "queries")
        ),
    )
    result["metrics"] = add_efficiency_metrics(result["metrics"], efficiency)
    save_json(output / "metrics.json", result["metrics"])
    save_json(output / "summary.json", result["metrics"])
    with (output / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["rollouts"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json(output / "results.json", result["rollouts"])
    with (output / "retrieval_trace.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["rollouts"]:
            trace_row = {
                key: row.get(key)
                for key in (
                    "query_id",
                    "dataset",
                    "category",
                    "manifest_question_id",
                    "retrieval_mode",
                    "vector_k",
                    "append_k_requested",
                    "append_k_actual",
                    "retrieval_seed",
                    "retrieval_global_seed",
                    "retrieval_vector_ids",
                    "retrieval_append_ids",
                    "retrieval_final_ids",
                    "append_shortfall_reason",
                    "retrieval_signature",
                    "visible_sessions",
                    "prefix_graph_signature",
                )
                if key in row
            }
            trace_row["hits"] = row.get("retrieval_top_k", [])
            handle.write(json.dumps(trace_row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), **result["metrics"]}, ensure_ascii=False))


def evaluate(
    config: dict[str, Any],
    split: str,
    strategy: EvidenceStrategy,
    env: EvidenceSelectionEnv,
    query_cache: QueryEmbeddingCache,
    profiles: dict[str, str],
    *,
    policy: EvidenceSelectionPolicy | None,
    deterministic: bool,
    limit: int = 0,
    cost_runtime: CostRewardRuntime | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    rollout_objects: list[EvidenceRollout] = []
    cost_snapshot = cost_runtime.snapshot() if cost_runtime is not None else None
    for index, episode in enumerate(iter_episodes(config, split, query_cache, profiles)):
        if limit and index >= limit:
            break
        with torch.no_grad():
            rollout = rollout_with_endpoint_recovery(
                env,
                episode,
                strategy,
                policy=policy,
                deterministic=deterministic,
            )
        if cost_runtime is not None:
            cost_runtime.apply(rollout, episode, snapshot=cost_snapshot)
        source_groups = [
            list(hit.item.metadata.get("source_dialogue_ids", []))
            for hit in episode.memory_hits
        ]
        records.append(
            {
                "dataset": episode.dataset,
                "category": episode.category,
                "system_answer": rollout.answer,
                "original_answer": episode.ground_truth,
                "retrieved_source_groups": source_groups,
                "clue": list(episode.clue),
                "gold_sessions": list(episode.clue),
                "retrieved_sessions": [
                    str(hit.item.metadata.get("session_id", ""))
                    for hit in episode.memory_hits
                ],
                "difficulty": episode.metadata.get("difficulty", ""),
                "error": rollout.error,
                "answer_attempts": rollout.answer_attempts,
                "answer_failed_attempts": rollout.answer_failed_attempts,
            }
        )
        rollouts.append(rollout_record(rollout, episode, source_groups=source_groups))
        rollout_objects.append(rollout)
    benchmark = str(config.get("benchmark", "memgallery")).lower()
    if benchmark == "wma":
        from benchmarks.wma_harness.runner.metrics import summarize_results as summarize_wma_results

        metrics = summarize_wma_results(records, k=int(config["top_k"]))
    elif benchmark == "h2hmem":
        from benchmarks.wma_harness.runner.metrics import summarize_results as summarize_h2h_results

        metrics = summarize_h2h_results(records, k=int(config["top_k"]))
    else:
        metrics = summarize_results(records, k=int(config["top_k"]))
    metrics["mean_reward"] = (
        float(np.mean([row["reward"] for row in rollouts])) if rollouts else 0.0
    )
    metrics["evidence_actions"] = summarize_evidence_actions(rollouts)
    metrics["cached_rollouts"] = sum(bool(row["cached"]) for row in rollouts)
    metrics["errors"] = sum(bool(row["error"]) for row in rollouts)
    if cost_runtime is not None:
        metrics.update(summarize_cost_rewards(rollout_objects))
    evaluated_sample_ids = sorted(
        {str(row.get("dataset") or "").strip() for row in records} - {""}
    )
    metrics["calls"] = combine_call_metrics(
        calculate_calls_mb(config.get("memory_bank"), evaluated_sample_ids),
        calculate_calls_qa(records, sample_id_field="dataset"),
    )
    return {"metrics": metrics, "rollouts": rollouts}


def resume_configs_match(stored: Any, current: dict[str, Any]) -> bool:
    if not isinstance(stored, dict):
        return False
    stored_copy = json.loads(json.dumps(stored))
    current_copy = json.loads(json.dumps(current))
    for config in (stored_copy, current_copy):
        config.pop("output_dir", None)
        model = config.get("model")
        if isinstance(model, dict):
            model.pop("base_url", None)
        ppo = config.get("ppo")
        if isinstance(ppo, dict):
            ppo.pop("skip_invalid_response", None)
    return stored_copy == current_copy


def reconcile_ppo_metrics_for_resume(
    path: Path,
    *,
    checkpoint_update_step: int,
) -> dict[str, Any]:
    """Discard metric rows produced after the checkpoint being resumed.

    A process can be interrupted after writing PPO updates but before saving the
    next epoch checkpoint.  Those updates are not represented by the checkpoint
    and must not remain in the resumed run's W&B history.  Keeping the last row
    for every committed step also repairs duplicates left by an earlier resume.
    """
    checkpoint_update_step = int(checkpoint_update_step)
    result: dict[str, Any] = {
        "checkpoint_update_step": checkpoint_update_step,
        "original_rows": 0,
        "kept_rows": 0,
        "removed_rows": 0,
        "backup": "",
    }
    if not path.exists():
        return result

    payload = path.read_bytes()
    retained_by_step: dict[int, str] = {}
    original_rows = 0
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        original_rows += 1
        try:
            row = json.loads(line)
            update_step = int(row["update_step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid PPO metrics row at {path}:{line_number}"
            ) from exc
        if update_step <= checkpoint_update_step:
            retained_by_step[update_step] = line

    retained_steps = sorted(retained_by_step)
    expected_steps = list(range(1, checkpoint_update_step + 1))
    if retained_steps != expected_steps:
        raise ValueError(
            "PPO metrics do not cover the checkpoint update range: "
            f"expected 1..{checkpoint_update_step}, got "
            f"{retained_steps[0] if retained_steps else 'none'}.."
            f"{retained_steps[-1] if retained_steps else 'none'}"
        )

    kept_lines = [retained_by_step[step] for step in retained_steps]
    removed_rows = original_rows - len(kept_lines)
    result.update(
        {
            "original_rows": original_rows,
            "kept_rows": len(kept_lines),
            "removed_rows": removed_rows,
        }
    )
    if removed_rows <= 0:
        return result

    backup = path.with_name(f"{path.name}.pre_resume_{time.time_ns()}.bak")
    backup.write_bytes(payload)
    temporary = path.with_name(f"{path.name}.resume.tmp")
    temporary.write_text(
        "\n".join(kept_lines) + ("\n" if kept_lines else ""),
        encoding="utf-8",
    )
    temporary.replace(path)
    result["backup"] = str(backup)
    return result


def is_transient_endpoint_error(error: str) -> bool:
    normalized = str(error).lower()
    return any(marker in normalized for marker in TRANSIENT_ENDPOINT_ERROR_MARKERS)


def rollout_with_endpoint_recovery(
    env: EvidenceSelectionEnv,
    episode: EvidenceEpisode,
    strategy: EvidenceStrategy,
    *,
    policy: EvidenceSelectionPolicy | None,
    deterministic: bool,
    attempts: int = ROLLOUT_RETRY_ATTEMPTS,
    delay_seconds: float = ROLLOUT_RETRY_DELAY_SECONDS,
    allow_skip_invalid_response: bool = False,
) -> EvidenceRollout:
    """Pause on endpoint outages instead of turning them into zero rewards."""
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    for attempt in range(1, attempts + 1):
        rollout = env.rollout(
            episode,
            strategy,
            policy=policy,
            deterministic=deterministic,
        )
        if not rollout.error:
            return rollout
        if not is_transient_endpoint_error(rollout.error):
            if allow_skip_invalid_response and (
                "Response must contain only one <answer>...</answer> block"
                in rollout.error
            ):
                rollout.skipped_invalid_response = True
                rollout.skipped_error = rollout.error
                rollout.error = ""
                rollout.reward = ALL_ZERO_REWARD
                rollout.quality_reward = ALL_ZERO_REWARD
                print(
                    f"Skipping invalid training response for {episode.query_id}; "
                    "reward=-1",
                    file=sys.stderr,
                    flush=True,
                )
                return rollout
            action_masks = [action.bitmask for action in rollout.actions]
            raise RuntimeError(
                f"Rollout failed for {episode.query_id} with actions "
                f"{action_masks}: {rollout.error}"
            )
        if attempt == attempts:
            break
        print(
            f"Endpoint unavailable for {episode.query_id}; "
            f"pausing {delay_seconds:g}s before retry {attempt + 1}/{attempts}",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay_seconds)
    raise RuntimeError(
        f"Endpoint remained unavailable after {attempts} attempts for "
        f"{episode.query_id}: {rollout.error}"
    )


def iter_episodes(
    config: dict[str, Any],
    split: str,
    query_cache: QueryEmbeddingCache,
    profiles: dict[str, str],
) -> Iterator[EvidenceEpisode]:
    benchmark = str(config.get("benchmark", "memgallery")).lower()
    if benchmark == "wma":
        yield from iter_wma_episodes(config, split, query_cache)
        return
    if benchmark == "h2hmem":
        yield from iter_h2hmem_episodes(config, split, query_cache)
        return
    data_dir = Path(config["data_dir"])
    excluded_categories = parse_excluded_categories(
        config.get("excluded_categories", ["AR"])
    )
    split_index = configured_split_manifest(config)
    data_source = evidence_data_source(config)
    dataset_names = (
        split_index.source_ids(split, data_source)
        if split_index is not None
        else tuple(config["split"][split])
    )
    retrieval_settings = resolve_retrieval_settings(config)
    graph_options = retrieval_settings["graph_options"]
    prompt_digest = memgallery_prompt_sha256()
    for dataset_name in dataset_names:
        path = data_dir / "dialog" / f"{dataset_name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset_dir = Path(config["memory_bank"]) / "datasets" / dataset_name
        index = (
            build_graph_index(dataset_dir, graph_options)
            if graph_options is not None
            else SimpleMemoryIndex(dataset_dir)
        )
        index_signature = retrieval_signature(
            dataset_dir,
            graph_options,
            retrieval_mode=retrieval_settings["mode"],
            vector_k=retrieval_settings["vector_k"],
            append_k=retrieval_settings["append_k"],
            retrieval_seed=(
                retrieval_settings["retrieval_seed"]
                if retrieval_settings["mode"] == "random_append"
                else None
            ),
        )
        for qa_index, qa in enumerate(payload.get("human-annotated QAs", []), start=1):
            manifest_question_id = f"{dataset_name}_q{qa_index - 1:04d}"
            if split_index is not None and not split_index.contains_question(
                split, data_source, manifest_question_id
            ):
                continue
            category = str(qa.get("point", ""))
            if is_excluded_category(category, excluded_categories):
                continue
            question = str(qa.get("question", ""))
            query_image = resolve_question_image(data_dir, qa)
            query_id = make_query_id(
                dataset_name=dataset_name,
                qa_index=qa_index,
                category=category,
                question=question,
                query_image=query_image,
            )
            query_vector = query_cache.get(
                dataset_name=dataset_name,
                qa_index=qa_index,
                category=category,
                question=question,
                query_image=query_image,
            )
            if query_vector is None:
                raise KeyError(f"Missing cached query embedding: {query_id}")
            hits, retrieval_metadata = retrieve_hits(
                index,
                query_vector,
                retrieval_settings,
                benchmark="memgallery",
                manifest_question_id=manifest_question_id,
                category=category,
            )
            raw_clue = qa.get("clue", [])
            clue = raw_clue if isinstance(raw_clue, list) else []
            yield EvidenceEpisode(
                query_id=query_id,
                dataset=dataset_name,
                category=category,
                question_prompt=question,
                system_prompt="",
                ground_truth=str(qa.get("answer", "")),
                query_embedding=query_vector,
                memory_hits=tuple(hits),
                query_image=query_image,
                clue=tuple(str(item) for item in clue),
                retrieval_signature=index_signature,
                answer_messages_builder=partial(
                    build_memgallery_policy_messages,
                    question=question,
                    category=category,
                    query_image=query_image,
                ),
                answer_parser=parse_memgallery_answer,
                prepend_memory_context=False,
                prompt_signature=prompt_digest,
                metadata={
                    "manifest_question_id": manifest_question_id,
                    "prompt_version": MEMGALLERY_PROMPT_VERSION,
                    "prompt_source": MEMGALLERY_PROMPT_SOURCE,
                    "prompt_sha256": prompt_digest,
                    "ppo_empty_prompt_version": MEMGALLERY_PPO_EMPTY_PROMPT_VERSION,
                    "ppo_empty_prompt_sha256": memgallery_ppo_empty_prompt_sha256(),
                    **retrieval_metadata,
                    "graph_append_k": (
                        int(graph_options["append_k"]) if graph_options else 0
                    ),
                },
            )


def build_memgallery_policy_messages(
    memory_items: Sequence[dict[str, Any]],
    *,
    question: str,
    category: str,
    query_image: dict[str, Any] | None,
    allow_empty_evidence: bool = False,
) -> list[dict[str, str]]:
    evidence, _ = build_retrieved_memory_evidence(list(memory_items), category)
    return build_memgallery_answer_messages(
        question=question,
        question_type=category,
        memory_evidence=evidence,
        query_images=query_image_prompt_metadata(query_image),
        allow_empty_evidence=allow_empty_evidence,
    )


def build_h2hmem_policy_messages(
    memory_items: Sequence[dict[str, Any]],
    *,
    question: str,
    category: str,
    query_image: dict[str, Any] | None = None,
    allow_empty_evidence: bool = False,
) -> list[dict[str, str]]:
    from benchmarks.h2hmem_harness.prompts import build_answer_messages

    evidence, _ = build_retrieved_memory_evidence(
        list(memory_items), category="VR"
    )
    return build_answer_messages(
        question=question,
        question_type=category,
        memory_evidence=evidence,
        query_images=query_image_prompt_metadata(query_image),
        allow_empty_evidence=allow_empty_evidence,
    )


def parse_h2hmem_policy_answer(raw: str) -> str:
    from benchmarks.h2hmem_harness.prompts import parse_answer_response

    return parse_answer_response(raw)


def build_wma_policy_messages(
    memory_items: Sequence[dict[str, Any]],
    *,
    question: str,
    category: str,
    allow_empty_evidence: bool = False,
) -> list[dict[str, str]]:
    from benchmarks.wma_harness.runner.answer_client import (
        build_retrieved_memory_evidence as build_wma_evidence,
    )
    from benchmarks.wma_harness.runner.prompts import build_answer_messages

    evidence, _ = build_wma_evidence(list(memory_items), category)
    return build_answer_messages(
        question=question,
        question_type=category,
        memory_evidence=evidence,
        allow_empty_evidence=allow_empty_evidence,
    )


def parse_wma_policy_answer(raw: str) -> str:
    from benchmarks.wma_harness.runner.prompts import parse_answer_response

    return parse_answer_response(raw)


def iter_h2hmem_episodes(
    config: dict[str, Any],
    split: str,
    query_cache: QueryEmbeddingCache,
) -> Iterator[EvidenceEpisode]:
    from benchmarks.h2hmem_harness.eval_h2hmem import _question_image
    from benchmarks.h2hmem_harness.prompts import (
        PPO_EMPTY_PROMPT_VERSION,
        PROMPT_SOURCE,
        PROMPT_VERSION,
        ppo_empty_prompt_sha256,
        prompt_sha256,
    )

    split_index = configured_split_manifest(config)
    if split_index is None:
        raise ValueError("H2HMem PPO requires split_manifest")
    data_sources = evidence_data_sources(config)
    workspace_root = Path(
        config.get("workspace_root") or Path(config["data_dir"]).resolve().parents[1]
    )
    visual_categories = {
        str(value).upper() for value in config.get("visual_categories", [])
    }
    prompt_digest = prompt_sha256()
    retrieval_settings = resolve_retrieval_settings(config)
    graph_options = retrieval_settings["graph_options"]
    indexes: dict[str, Any] = {}
    index_signatures: dict[str, str] = {}
    for row in iter_source_questions(
        split_index,
        workspace_root,
        split=split,
        data_sources=data_sources,
    ):
        variant = str(row.metadata["variant"])
        dataset_name = f"{variant}_{row.source_id}"
        query_vector = query_cache.get_by_id(row.question_id)
        if query_vector is None:
            raise KeyError(f"Missing cached query embedding: {row.question_id}")
        if dataset_name not in indexes:
            dataset_dir = Path(config["memory_bank"]) / "datasets" / dataset_name
            indexes[dataset_name] = (
                build_graph_index(
                    dataset_dir,
                    graph_options,
                    visual_categories=visual_categories,
                )
                if graph_options is not None
                else SimpleMemoryIndex(
                    dataset_dir,
                    visual_categories=visual_categories,
                )
            )
            index_signatures[dataset_name] = retrieval_signature(
                dataset_dir,
                graph_options,
                retrieval_mode=retrieval_settings["mode"],
                vector_k=retrieval_settings["vector_k"],
                append_k=retrieval_settings["append_k"],
                retrieval_seed=(
                    retrieval_settings["retrieval_seed"]
                    if retrieval_settings["mode"] == "random_append"
                    else None
                ),
            )
        index = indexes[dataset_name]
        hits, retrieval_metadata = retrieve_hits(
            index,
            query_vector,
            retrieval_settings,
            benchmark="h2hmem",
            manifest_question_id=row.question_id,
            category=row.category,
        )
        raw_image = str(row.metadata.get("question_image", ""))
        query_image = (
            _question_image(Path(row.source_path), raw_image) if raw_image else None
        )
        yield EvidenceEpisode(
            query_id=row.question_id,
            dataset=dataset_name,
            category=row.category,
            question_prompt=row.question,
            system_prompt="",
            ground_truth=row.answer,
            query_embedding=query_vector,
            memory_hits=tuple(hits),
            query_image=query_image,
            clue=tuple(str(value) for value in row.metadata.get("answer_session", [])),
            retrieval_signature=index_signatures[dataset_name],
            answer_messages_builder=partial(
                build_h2hmem_policy_messages,
                question=row.question,
                category=row.category,
                query_image=query_image,
            ),
            answer_parser=parse_h2hmem_policy_answer,
            prepend_memory_context=False,
            prompt_signature=prompt_digest,
            metadata={
                "manifest_question_id": row.question_id,
                "variant": variant,
                "conversation_id": row.source_id,
                "session_id": row.metadata.get("session_id", ""),
                "difficulty": row.metadata.get("difficulty", ""),
                "prompt_version": PROMPT_VERSION,
                "prompt_source": PROMPT_SOURCE,
                "prompt_sha256": prompt_digest,
                "ppo_empty_prompt_version": PPO_EMPTY_PROMPT_VERSION,
                "ppo_empty_prompt_sha256": ppo_empty_prompt_sha256(),
                **retrieval_metadata,
                "graph_append_k": (
                    int(graph_options["append_k"]) if graph_options else 0
                ),
                # MemGallery's answer renderer uses compact visual category names.
                # H2HMem category labels are descriptive, so force only the renderer
                # into visual mode while retaining the original label for metrics.
                "answer_category": "VR",
            },
        )


def iter_wma_episodes(
    config: dict[str, Any],
    split: str,
    query_cache: QueryEmbeddingCache,
) -> Iterator[EvidenceEpisode]:
    from benchmarks.wma_harness.retrieval.query_embedding_cache import (
        build_gold_evidence_map,
        make_query_id as make_wma_query_id,
        session_ids,
        visible_sessions_for_checkpoint,
    )
    from benchmarks.wma_harness.runner.prompts import (
        PPO_EMPTY_PROMPT_VERSION,
        PROMPT_SOURCE,
        PROMPT_VERSION,
        ppo_empty_prompt_sha256,
        prompt_sha256,
    )
    from embedding.chunk_builder import iter_wma_sample_files

    data_dir = Path(config["data_dir"])
    paths = {path.stem: path for path in iter_wma_sample_files(data_dir)}
    visual_categories = {
        str(value).upper()
        for value in config.get("visual_categories", ["VFR", "VS", "VU", "CMR", ""])
    }
    excluded_categories = parse_excluded_categories(
        config.get("excluded_categories", [])
    )
    prompt_digest = prompt_sha256()
    split_index = configured_split_manifest(config)
    data_source = evidence_data_source(config)
    sample_ids = (
        split_index.source_ids(split, data_source)
        if split_index is not None
        else tuple(config["split"][split])
    )
    retrieval_settings = resolve_retrieval_settings(config)
    graph_options = retrieval_settings["graph_options"]
    prefix_cache_root = Path(config["output_dir"]) / "retrieval_indexes" / "wma_prefix"
    for sample_id in sample_ids:
        payload = json.loads(paths[sample_id].read_text(encoding="utf-8"))
        ordered_sessions = session_ids(payload)
        gold_points = build_gold_evidence_map(payload)
        point_sessions = {
            evidence_id: row["session_id"]
            for evidence_id, row in gold_points.items()
        }
        source_dataset_dir = Path(config["memory_bank"]) / "datasets" / sample_id
        vector_index = (
            SimpleMemoryIndex(source_dataset_dir, visual_categories=visual_categories)
            if graph_options is None
            else None
        )
        for checkpoint in payload.get("qa_checkpoints", []) or []:
            checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
            covered_sessions = [
                str(value) for value in checkpoint.get("covered_sessions", [])
            ]
            visible_sessions = visible_sessions_for_checkpoint(
                ordered_sessions, covered_sessions
            )
            visible_session_set = set(visible_sessions)
            prefix_signature = hashlib.sha256(
                json.dumps(
                    {"visible_sessions": visible_sessions},
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if graph_options is not None:
                index, prefix_signature = build_wma_prefix_graph_index(
                    source_dataset_dir,
                    prefix_cache_root,
                    sample_id=sample_id,
                    checkpoint_id=checkpoint_id,
                    visible_session_ids=visible_sessions,
                    options=graph_options,
                    visual_categories=visual_categories,
                )
            else:
                index = vector_index
            if index is None:
                raise RuntimeError(f"Failed to initialize retrieval index for {sample_id}")
            index_signature = retrieval_signature(
                source_dataset_dir, graph_options,
                prefix_signature=prefix_signature,
                retrieval_mode=retrieval_settings["mode"],
                vector_k=retrieval_settings["vector_k"],
                append_k=retrieval_settings["append_k"],
                retrieval_seed=(
                    retrieval_settings["retrieval_seed"]
                    if retrieval_settings["mode"] == "random_append"
                    else None
                ),
            )
            for qa_index, qa in enumerate(checkpoint.get("questions", []) or [], start=1):
                manifest_question_id = f"{sample_id}:{checkpoint_id}:Q{qa_index:03d}"
                if split_index is not None and not split_index.contains_question(
                    split, data_source, manifest_question_id
                ):
                    continue
                category = str(qa.get("question_type_abbrev", ""))
                if is_excluded_category(category, excluded_categories):
                    continue
                question = str(qa.get("question", ""))
                query_id = make_wma_query_id(
                    sample_id=sample_id,
                    checkpoint_id=checkpoint_id,
                    qa_index=qa_index,
                    category=category,
                    question=question,
                )
                query_vector = query_cache.get_by_id(query_id)
                if query_vector is None:
                    raise KeyError(f"Missing cached query embedding: {query_id}")
                hits, retrieval_metadata = retrieve_hits(
                    index,
                    query_vector,
                    retrieval_settings,
                    benchmark="wma",
                    manifest_question_id=manifest_question_id,
                    category=category,
                    allowed_session_ids=visible_session_set,
                )
                evidence_ids = [
                    str(row.get("memory_id") or row.get("image_id") or "")
                    for row in qa.get("evidence", []) or []
                    if isinstance(row, dict)
                ]
                yield EvidenceEpisode(
                    query_id=query_id,
                    dataset=sample_id,
                    category=category,
                    question_prompt=question,
                    system_prompt="",
                    ground_truth=str(qa.get("answer", "")),
                    query_embedding=query_vector,
                    memory_hits=tuple(hits),
                    retrieval_signature=index_signature,
                    answer_messages_builder=partial(
                        build_wma_policy_messages,
                        question=question,
                        category=category,
                    ),
                    answer_parser=parse_wma_policy_answer,
                    prepend_memory_context=False,
                    prompt_signature=prompt_digest,
                    clue=tuple(
                        dict.fromkeys(
                            point_sessions[value]
                            for value in evidence_ids
                            if value in point_sessions
                            and point_sessions[value] in visible_session_set
                        )
                    ),
                    metadata={
                        "manifest_question_id": manifest_question_id,
                        "checkpoint_id": checkpoint_id,
                        "question": question,
                        "question_type": qa.get("question_type", ""),
                        "difficulty": qa.get("difficulty", ""),
                        "prompt_version": PROMPT_VERSION,
                        "prompt_source": PROMPT_SOURCE,
                        "prompt_sha256": prompt_digest,
                        "ppo_empty_prompt_version": PPO_EMPTY_PROMPT_VERSION,
                        "ppo_empty_prompt_sha256": ppo_empty_prompt_sha256(),
                        "evidence": qa.get("evidence", []),
                        "covered_sessions": covered_sessions,
                        "visible_sessions": visible_sessions,
                        **retrieval_metadata,
                        "graph_append_k": (
                            int(graph_options["append_k"]) if graph_options else 0
                        ),
                        "prefix_graph_signature": prefix_signature,
                        "gold_future_evidence_ids": [
                            value
                            for value in evidence_ids
                            if value in point_sessions
                            and point_sessions[value] not in visible_session_set
                        ],
                        "gold_unmapped_evidence_ids": [
                            value for value in evidence_ids if value not in point_sessions
                        ],
                    },
                )


def evidence_data_source(config: dict[str, Any]) -> str:
    sources = evidence_data_sources(config)
    if len(sources) != 1:
        raise ValueError(
            "This operation requires one data source, got " + ", ".join(sources)
        )
    return sources[0]


def evidence_data_sources(config: dict[str, Any]) -> tuple[str, ...]:
    configured = config.get("data_sources")
    if configured is not None:
        sources = tuple(str(value).strip() for value in configured if str(value).strip())
        if not sources:
            raise ValueError("data_sources cannot be empty")
        if len(set(sources)) != len(sources):
            raise ValueError("data_sources cannot contain duplicates")
        return sources
    explicit = str(config.get("data_source", "")).strip()
    if explicit:
        return (explicit,)
    benchmark = str(config.get("benchmark", "memgallery")).strip().lower()
    if benchmark == "wma":
        return ("worldmemarena_lifelong",)
    if benchmark == "memgallery":
        return ("mem_gallery",)
    if benchmark == "h2hmem":
        return ("h2hmem_dyadic", "h2hmem_multiparty")
    raise ValueError(
        f"Cannot infer manifest data_source for benchmark {benchmark!r}; "
        "set data_source in the evidence-policy config"
    )


def configured_split_manifest(
    config: dict[str, Any],
) -> SplitManifestIndex | None:
    path = str(config.get("split_manifest", "")).strip()
    return SplitManifestIndex(path) if path else None


def build_policy(config: dict[str, Any], device: torch.device) -> EvidenceSelectionPolicy:
    return EvidenceSelectionPolicy(**config["policy"]).to(device)


def build_trainer(
    config: dict[str, Any], policy: EvidenceSelectionPolicy
) -> PPOTrainer:
    keys = {
        "learning_rate",
        "clip_ratio",
        "value_coefficient",
        "entropy_coefficient",
        "max_grad_norm",
        "update_epochs",
        "minibatch_size",
        "gamma",
        "gae_lambda",
    }
    return PPOTrainer(policy, **{key: config["ppo"][key] for key in keys})


def build_environment(
    config: dict[str, Any],
) -> tuple[VLMAnswerClient, EvidenceSelectionEnv]:
    model = config["model"]
    api_key = str(model.get("api_key", "")).strip()
    api_key_env = str(model.get("api_key_env", "")).strip()
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env!r} is empty")
    benchmark = str(config.get("benchmark", "memgallery")).lower()
    client_class = VLMAnswerClient
    if benchmark == "wma":
        from benchmarks.wma_harness.runner.answer_client import VLMAnswerClient as WMAAnswerClient

        client_class = WMAAnswerClient
    client = client_class(
        model=model["name"],
        base_url=model["base_url"],
        api_key=api_key,
        num_predict=int(model["max_tokens"]),
        timeout=int(model["timeout"]),
        retries=int(model["retries"]),
        think=bool(model["think"]),
        reasoning_effort=str(model.get("reasoning_effort", "")),
    )
    cache = RolloutCache(Path(config["output_dir"]) / "rollout_cache.jsonl")
    visual_categories = {
        str(value).upper() for value in config.get("visual_categories", ["VS", "VR"])
    }
    if benchmark == "wma":
        store = WMADialogueStore(config["data_dir"])
    elif benchmark == "h2hmem":
        store = H2HMemDialogueStore(config["data_dir"])
    else:
        store = DialogueStore(config["data_dir"])
    evidence = config.get("evidence") or {}
    vp_index = (
        VPArtifactIndex(
            evidence["vp_run_dir"],
            max_vps_per_image=int(evidence.get("max_vps_per_image", 0)),
        )
        if evidence.get("vp_run_dir")
        else None
    )
    builder = EvidenceChainBuilder(
        store, vp_index=vp_index, visual_categories=visual_categories
    )
    return client, EvidenceSelectionEnv(
        client,
        builder,
        cache=cache,
        rng=random.Random(int(config["seed"])),
        visual_categories=visual_categories,
        ppo_force_visual_evidence=bool(
            evidence.get("ppo_force_visual_evidence", False)
        ),
        disabled_evidence_types=evidence.get("disabled_types", ()),
    )


def validate_runtime(config: dict[str, Any], *, require_split: bool) -> None:
    validate_graph_config(config)
    for key in ("data_dir", "memory_bank"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {config[key]}")
    evidence = config.get("evidence") or {}
    configured_order = evidence.get("order")
    expected_order = [kind.value for kind in EVIDENCE_ORDER]
    if int(evidence.get("schema_version", 0)) != 2 or configured_order != expected_order:
        raise ValueError(
            f"Evidence schema must be version 2 with order {expected_order}, "
            f"got version={evidence.get('schema_version')}, order={configured_order!r}"
        )
    disabled_types = evidence.get("disabled_types", [])
    if not isinstance(disabled_types, list):
        raise ValueError("evidence.disabled_types must be a list")
    if len(disabled_types) != len(set(disabled_types)):
        raise ValueError("evidence.disabled_types must not contain duplicates")
    unknown_disabled = sorted(set(disabled_types) - set(expected_order))
    if unknown_disabled:
        raise ValueError(
            f"Unknown evidence.disabled_types values: {unknown_disabled}; "
            f"expected a subset of {expected_order}"
        )
    if evidence.get("vp_run_dir"):
        vp_index = VPArtifactIndex(
            evidence["vp_run_dir"],
            max_vps_per_image=int(evidence.get("max_vps_per_image", 0)),
        )
        if bool(evidence.get("strict_vp_coverage", False)):
            coverage = vp_index.audit(memory_image_paths(config["memory_bank"]))
            if coverage["missing_records"] or coverage["missing_crop_files"]:
                raise ValueError(f"Incomplete VP coverage for memory bank: {coverage}")
    query_cache = Path(config["query_cache"])
    missing = [name for name in ("vectors.npy", "metadata.jsonl") if not (query_cache / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing query cache files in {query_cache}: {', '.join(missing)}. "
            "Generate the 2048-dimensional query cache before train/eval."
        )
    manifest_path = query_cache / "manifest.json"
    build_manifest_path = Path(config["memory_bank"]) / "build_manifest.json"
    if manifest_path.exists() and build_manifest_path.exists():
        query_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
        expected_model = str(build_manifest.get("embedding_model", ""))
        actual_model = str(query_manifest.get("model_name", ""))
        if expected_model and actual_model and actual_model != expected_model:
            raise ValueError(
                f"Query cache model {actual_model!r} does not match memory bank "
                f"embedding model {expected_model!r}"
            )
        expected_dim = int(build_manifest.get("embedding_dim", config["policy"]["embedding_dim"]))
        actual_dim = int(query_manifest.get("dim", expected_dim))
        if actual_dim != expected_dim or expected_dim != int(config["policy"]["embedding_dim"]):
            raise ValueError(
                f"Embedding dimension mismatch: query={actual_dim}, bank={expected_dim}, "
                f"policy={config['policy']['embedding_dim']}"
            )
    vectors = np.load(query_cache / "vectors.npy", mmap_mode="r")
    if vectors.ndim != 2 or vectors.shape[1] != int(config["policy"]["embedding_dim"]):
        raise ValueError(
            f"Query cache vectors must have shape (*, {config['policy']['embedding_dim']}), "
            f"got {vectors.shape}"
        )
    if require_split:
        split_index = configured_split_manifest(config)
        if split_index is not None:
            sources = evidence_data_sources(config)
            missing_sources = [
                source for source in sources if source not in split_index.data_sources
            ]
            if missing_sources:
                raise ValueError(
                    f"Configured data sources {missing_sources!r} are absent from "
                    f"{split_index.path}"
                )
            for source in sources:
                empty = [
                    name
                    for name in ("train", "val", "test")
                    if not split_index.conversations(name, data_source=source)
                ]
                if empty:
                    raise ValueError(
                        f"Manifest has empty splits for {source}: {', '.join(empty)}"
                    )
            return
        split = config.get("split", {})
        groups = [split.get(name, []) for name in ("train", "validation", "test")]
        benchmark = str(config.get("benchmark", "memgallery")).lower()
        expected_sizes = (
            list(config.get("split_sizes", []))
            if config.get("split_sizes")
            else [12, 4, 4] if benchmark == "memgallery" else []
        )
        if expected_sizes and [len(group) for group in groups] != expected_sizes:
            raise ValueError(f"Run prepare-split first; expected split sizes {expected_sizes}")
        if any(not group for group in groups):
            raise ValueError("Run prepare-split first; train/validation/test must be non-empty")
        flattened = [name for group in groups for name in group]
        if len(set(flattened)) != len(flattened):
            raise ValueError("Benchmark splits overlap")


def load_profiles(config: dict[str, Any]) -> dict[str, str]:
    if not config.get("profiles_file"):
        return {}
    path = Path(config["profiles_file"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing profiles_file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_dicts(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def summarize_evidence_actions(rollouts: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for rollout in rollouts:
        for action in rollout.get("actions", []):
            mask = str(action.get("mask", "00000"))
            counts[f"mask:{mask}"] += 1
            if mask == "00000":
                counts["all-zero"] += 1
            for kind in action.get("selected", []):
                counts[str(kind)] += 1
    return dict(sorted(counts.items()))


def rollout_record(
    rollout: EvidenceRollout,
    episode: EvidenceEpisode,
    *,
    source_groups: list[list[str]] | None = None,
) -> dict[str, Any]:
    if source_groups is None:
        source_groups = [
            list(hit.item.metadata.get("source_dialogue_ids", []))
            for hit in episode.memory_hits
        ]
    row = rollout.to_dict()
    row.update(
        {
            "original_answer": episode.ground_truth,
            "retrieved_source_groups": source_groups,
            "retrieval_top_k": retrieval_trace(episode.memory_hits),
            "retrieval_signature": episode.retrieval_signature,
            "clue": list(episode.clue),
            **episode.metadata,
        }
    )
    empty_prompt_version = row.pop("ppo_empty_prompt_version", "")
    empty_prompt_sha256 = row.pop("ppo_empty_prompt_sha256", "")
    is_all_zero = actions_are_all_zero(rollout.actions)
    if is_all_zero and empty_prompt_version and empty_prompt_sha256:
        row.update(
            {
                "prompt_variant": "ppo_empty_evidence",
                "base_prompt_version": row.get("prompt_version", ""),
                "base_prompt_sha256": row.get("prompt_sha256", ""),
                "prompt_version": empty_prompt_version,
                "prompt_sha256": empty_prompt_sha256,
            }
        )
    else:
        row["prompt_variant"] = "standard"
    return row


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
