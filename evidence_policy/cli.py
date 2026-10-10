#!/usr/bin/env python3
from __future__ import annotations

from src.utils import PROJECT_ROOT as _HIVE_PROJECT_ROOT
from src.utils import CONFIG_ROOT, api_key_for, load_policy_config, project_path

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

ROOT = _HIVE_PROJECT_ROOT

from benchmarks.common.query_cache import (  # noqa: E402
    QueryEmbeddingCache,
    make_query_id,
)
from benchmarks.common.answer_client import (  # noqa: E402
    VLMAnswerClient,
    build_retrieved_memory_evidence,
    query_image_prompt_metadata,
)
from benchmarks.common.metrics import (  # noqa: E402
    add_efficiency_metrics,
    calculate_calls_mb,
    calculate_calls_qa,
    combine_call_metrics,
    summarize_results,
    write_efficiency_metrics,
    write_runtime_call_metrics,
)
from benchmarks.common.prompts import (  # noqa: E402
    PPO_EMPTY_PROMPT_VERSION as MEMGALLERY_PPO_EMPTY_PROMPT_VERSION,
    MEMGALLERY_PROMPT_SOURCE,
    MEMGALLERY_PROMPT_VERSION,
    build_memgallery_answer_messages,
    parse_answer_response as parse_memgallery_answer,
    memgallery_ppo_empty_prompt_sha256,
    memgallery_prompt_sha256,
    resolve_question_image,
)
from benchmarks.common.utils import (  # noqa: E402
    is_excluded_category,
    parse_excluded_categories,
)
from evidence_policy.evidence import (  # noqa: E402
    EVIDENCE_ORDER,
    DialogueStore,
    EvidenceComposer,
    EvidenceStrategy,
    H2HMemDialogueStore,
    WMADialogueStore,
    SourceQuestion,
    SplitManifestIndex,
    GVVArtifactIndex,
    build_graph_index,
    build_wma_prefix_graph_index,
    iter_source_questions,
    retrieve_hits,
    resolve_retrieval_settings,
    retrieval_signature,
    retrieval_trace,
    validate_graph_config,
)
from evidence_policy.ppo import (  # noqa: E402
    EvidenceSelectionPolicy,
    PPOBuffer,
    PPOTrainer,
    load_policy_checkpoint,
    save_json,
)
from evidence_policy.rollout import (  # noqa: E402
    ALL_ZERO_REWARD,
    CostRewardRuntime,
    actions_are_all_zero,
    summarize_cost_rewards,
    EvidenceEpisode,
    EvidenceRollout,
    EvidenceSelectionEnv,
    RolloutCache,
)


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
    parser = argparse.ArgumentParser(description="PPO evidence selection for benchmark memory episodes")
    parser.add_argument(
        "--config",
        default=os.getenv("HIVE_POLICY_CONFIG") or str(CONFIG_ROOT / "experiments.json"),
    )
    parser.add_argument(
        "--benchmark", choices=("memgallery", "h2hmem", "wma"),
        default=os.getenv("HIVE_POLICY_BENCHMARK", "memgallery"),
    )
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Override the fixed conversation-level train/val/test manifest",
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
        help="Override the configured memory bank",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Set the embedding dimension used by the policy and cached vectors",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Override the number of vector seeds for graph retrieval",
    )
    parser.add_argument(
        "--append-k",
        type=int,
        default=None,
        help="Override the number of appended affinity-graph memories",
    )
    parser.add_argument(
        "--degree-cap",
        type=int,
        default=None,
        help="Set the graph degree cap used by the memory bank",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("audit-gvv", help="Audit memory-image coverage in the GVV run")

    train_parser = subparsers.add_parser("train", help="Train the PPO policy")
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--epochs", type=int, default=0)
    train_parser.add_argument("--max-train-episodes", type=int, default=0)
    train_parser.add_argument("--validation-limit", type=int, default=0)
    train_parser.add_argument("--resume", default="")

    eval_parser = subparsers.add_parser("eval", help="Evaluate one evidence strategy")
    eval_parser.add_argument(
        "--strategy", choices=[item.value for item in EvidenceStrategy], required=True
    )
    eval_parser.add_argument("--split", choices=("train", "val", "validation", "test"), default="test")
    eval_parser.add_argument("--checkpoint", default="")
    eval_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    eval_parser.add_argument("--limit", type=int, default=0)

    for command in (train_parser, eval_parser):
        command.add_argument(
            "--benchmark", choices=("memgallery", "h2hmem", "wma"),
            default=argparse.SUPPRESS,
        )
        command.add_argument(
            "--embedding-dim",
            type=int,
            default=argparse.SUPPRESS,
            help="Set the embedding dimension used by the policy and cached vectors",
        )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path, benchmark=args.benchmark)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        config["output_dir"] = str(
            output_dir if output_dir.is_absolute() else (ROOT / output_dir).resolve()
        )
    if args.model_base_url:
        config["model"]["base_url"] = str(args.model_base_url).rstrip("/")
    if args.memory_bank:
        config["memory_bank"] = str(Path(args.memory_bank).expanduser().resolve())
    if args.embedding_dim is not None:
        config.setdefault("policy", {})["embedding_dim"] = args.embedding_dim
    if args.top_k:
        config["top_k"] = int(args.top_k)
    if args.append_k is not None:
        config["graph_options"] = dict(config.get("graph_options") or {})
        config["graph_options"]["append_k"] = int(args.append_k)
    if args.degree_cap is not None:
        if args.degree_cap < 0:
            raise ValueError("--degree-cap cannot be negative")
        config["graph_options"] = dict(config.get("graph_options") or {})
        config["graph_options"]["degree_cap"] = int(args.degree_cap)
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
    if args.command == "audit-gvv":
        audit_gvv(config)
    elif args.command == "train":
        train(config, args)
    else:
        evaluate_command(config, args)


def load_config(path: Path, *, benchmark: str = "memgallery") -> dict[str, Any]:
    config = load_policy_config(path, benchmark=benchmark)
    for env, key in (("HIVE_MEMORY_BANK", "memory_bank"), ("HIVE_QUERY_CACHE", "query_cache")):
        if env in os.environ:
            config[key] = str(project_path(os.environ[env]))
    if os.getenv("HIVE_OUTPUT_ROOT"):
        config["output_dir"] = str(
            project_path(os.environ["HIVE_OUTPUT_ROOT"]) / "evidence_policy" / benchmark
        )
    for env, section, key in (
        ("HIVE_ANSWER_MODEL", "model", "name"),
        ("HIVE_ANSWER_BASE_URL", "model", "base_url"),
        ("HIVE_TOKENIZER", "reward", "tokenizer_name"),
        ("HIVE_GVV_RUN_DIR", "evidence", "gvv_run_dir"),
    ):
        if env in os.environ:
            value = os.environ[env]
            if key == "base_url":
                value = value.strip()
                if not value:
                    continue
            config.setdefault(section, {})[key] = (
                str(project_path(value)) if key == "gvv_run_dir" else value
            )
    return config


def audit_gvv(config: dict[str, Any]) -> None:
    evidence = config.get("evidence") or {}
    index = GVVArtifactIndex(
        evidence["gvv_run_dir"],
        max_views_per_image=int(evidence.get("max_views_per_image", 0)),
    )
    paths = memory_image_paths(config["memory_bank"])
    report = {
        "gvv_run_id": index.run_id,
        "gvv_signature": index.signature,
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


def train(config: dict[str, Any], args: argparse.Namespace) -> None:
    require_model_base_url(config)
    require_embedding_dim(config)
    validate_runtime(config, require_split=True)
    seed_everything(int(config["seed"]))
    device = torch.device(args.device)
    policy = build_policy(config, device)
    trainer = build_trainer(config, policy)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)
    cost_runtime = CostRewardRuntime(config)
    ppo_metrics_path = output_dir / "ppo_metrics.jsonl"
    start_epoch = 0
    train_question_count = 0
    if args.resume:
        state = trainer.load_checkpoint(args.resume)
        if not resume_configs_match(state.get("config"), config):
            raise ValueError(
                "Checkpoint configuration does not match the current evidence-policy "
                "config (only output_dir, model.base_url, and the training-only "
                "ppo.skip_invalid_response flag may differ for recovery, together with "
                "omission of retired reward.quality_metric='f1' and "
                "reward.cost_scope='answer_llm_per_qa' fields)"
            )
        start_epoch = int(state["epoch"]) + 1
        train_question_count = int(
            state["extra"].get(
                "train_question_count",
                state["extra"].get("train_question_step", 0),
            )
        )
        cost_runtime.load_state_dict(
            state["extra"].get("cost_reward"), checkpoint_config=state.get("config")
        )
        reconciliation = reconcile_ppo_metrics_for_resume(
            ppo_metrics_path,
            checkpoint_update_step=trainer.update_steps,
        )
        if reconciliation["removed_rows"]:
            print(json.dumps({"resume_metrics_reconciliation": reconciliation}))
    client, env = build_environment(config)
    client.assert_model_available()
    query_cache = QueryEmbeddingCache(
        config["query_cache"], expected_dim=int(config["policy"]["embedding_dim"]),
        expected_model=config.get("embedding_model", ""),
        expected_revision=config.get("embedding_revision", ""),
    )
    epochs = int(args.epochs or config["ppo"]["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    initial_validation = prepare_initial_validation(
        config,
        env,
        query_cache,
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
    if start_epoch == 0 and ppo_metrics_path.exists():
        ppo_metrics_path.unlink()
    for epoch in range(start_epoch, epochs):
        buffer = PPOBuffer()
        batch_rollouts: list[EvidenceRollout] = []
        cost_snapshot = cost_runtime.snapshot()
        episode_iter = iter_episodes(config, "train", query_cache)
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
                    buffer.clear()
                    batch_rollouts.clear()
                    cost_snapshot = cost_runtime.snapshot()
            validation_phase = validation_points.get(episode_index)
            if validation_phase is not None:
                validation_event = run_training_validation(
                    config,
                    env,
                    query_cache,
                    policy,
                    output_dir=output_dir,
                    epoch=epoch,
                    phase=validation_phase,
                    update_step=trainer.update_steps,
                    train_question_count=train_question_count,
                    cost_runtime=cost_runtime,
                )
                validations.append(validation_event)
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
            buffer.clear()
            batch_rollouts.clear()
        end_validation = run_training_validation(
            config,
            env,
            query_cache,
            policy,
            output_dir=output_dir,
            epoch=epoch,
            phase="end",
            update_step=trainer.update_steps,
            train_question_count=train_question_count,
            cost_runtime=cost_runtime,
        )
        validations.append(end_validation)
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
    lose or silently change the initial validation result.
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
    require_model_base_url(config)
    require_embedding_dim(config)
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
        cost_runtime.load_state_dict(
            checkpoint_cost_state, checkpoint_config=checkpoint_state.get("config")
        )
    client, env = build_environment(config)
    client.assert_model_available()
    query_cache = QueryEmbeddingCache(
        config["query_cache"], expected_dim=int(config["policy"]["embedding_dim"]),
        expected_model=config.get("embedding_model", ""),
        expected_revision=config.get("embedding_revision", ""),
    )
    result = evaluate(
        config,
        args.split,
        strategy,
        env,
        query_cache,
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
        or CONFIG_ROOT / "defaults.json"
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
    for index, episode in enumerate(iter_episodes(config, split, query_cache)):
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
        from benchmarks.wma_harness.metrics import summarize_results as summarize_wma_results

        metrics = summarize_wma_results(records, k=int(config["top_k"]))
    elif benchmark == "h2hmem":
        from benchmarks.wma_harness.metrics import summarize_results as summarize_h2h_results

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
    """Compare algorithm settings, accepting equivalent retired reward metadata."""
    if not isinstance(stored, dict):
        return False
    stored_copy = json.loads(json.dumps(stored))
    current_copy = json.loads(json.dumps(current))
    current_revision = str(current_copy.get("embedding_revision") or "").strip()
    stored_revision = str(stored_copy.get("embedding_revision") or "").strip()
    if not current_revision and not stored_revision:
        # Unpinned legacy checkpoints can use the model identity checked by the
        # query cache. A recorded model difference must still reject resume.
        if "embedding_model" not in stored_copy:
            current_copy.pop("embedding_model", None)
        stored_copy.pop("embedding_revision", None)
        current_copy.pop("embedding_revision", None)
    for config in (stored_copy, current_copy):
        config.pop("output_dir", None)
        model = config.get("model")
        if isinstance(model, dict):
            model.pop("base_url", None)
        ppo = config.get("ppo")
        if isinstance(ppo, dict):
            ppo.pop("skip_invalid_response", None)
        policy = config.get("policy")
        if isinstance(policy, dict) and "embedding_dim" in policy:
            try:
                require_embedding_dim(config)
            except ValueError:
                return False
        reward = config.get("reward")
        if isinstance(reward, dict):
            for key, fixed_value in (
                ("quality_metric", "f1"),
                ("cost_scope", "answer_llm_per_qa"),
            ):
                if key in reward:
                    if reward[key] != fixed_value:
                        return False
                    reward.pop(key)
    return stored_copy == current_copy


def reconcile_ppo_metrics_for_resume(
    path: Path,
    *,
    checkpoint_update_step: int,
) -> dict[str, Any]:
    """Discard metric rows produced after the checkpoint being resumed.

    A process can be interrupted after writing PPO updates but before saving the
    next epoch checkpoint.  Those updates are not represented by the checkpoint
    and must not remain in the resumed run's local history. Keeping the last row
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


def validated_source_questions(
    config: dict[str, Any], split: str
) -> list[SourceQuestion]:
    """Check the complete fixed split before exclusions, retrieval, or limits."""
    benchmark = str(config.get("benchmark", "memgallery")).lower()
    return list(
        iter_source_questions(
            configured_split_manifest(config),
            config.get("workspace_root") or None,
            split=split,
            data_sources=evidence_data_sources(config),
            dataset_roots={benchmark: config["data_dir"]},
        )
    )


def iter_episodes(
    config: dict[str, Any],
    split: str,
    query_cache: QueryEmbeddingCache,
) -> Iterator[EvidenceEpisode]:
    benchmark = str(config.get("benchmark", "memgallery")).lower()
    if benchmark == "wma":
        yield from iter_wma_episodes(config, split, query_cache)
        return
    if benchmark == "h2hmem":
        yield from iter_h2hmem_episodes(config, split, query_cache)
        return
    validated_source_questions(config, split)
    data_dir = Path(config["data_dir"])
    excluded_categories = parse_excluded_categories(
        config.get("excluded_categories", ["AR"])
    )
    split_index = configured_split_manifest(config)
    data_source = evidence_data_source(config)
    dataset_names = split_index.source_ids(split, data_source)
    retrieval_settings = resolve_retrieval_settings(config)
    graph_options = retrieval_settings["graph_options"]
    prompt_digest = memgallery_prompt_sha256()
    for dataset_name in dataset_names:
        path = data_dir / "dialog" / f"{dataset_name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset_dir = Path(config["memory_bank"]) / "datasets" / dataset_name
        index = build_graph_index(dataset_dir, graph_options)
        index_signature = retrieval_signature(
            dataset_dir,
            graph_options,
            vector_k=retrieval_settings["vector_k"],
            append_k=retrieval_settings["append_k"],
        )
        for qa_index, qa in enumerate(payload.get("human-annotated QAs", []), start=1):
            manifest_question_id = f"{dataset_name}_q{qa_index - 1:04d}"
            if not split_index.contains_question(
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
                    "graph_append_k": int(graph_options["append_k"]),
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
    from benchmarks.common.prompts import build_h2hmem_answer_messages as build_answer_messages

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
    from benchmarks.common.prompts import parse_answer_response

    return parse_answer_response(raw)


def build_wma_policy_messages(
    memory_items: Sequence[dict[str, Any]],
    *,
    question: str,
    category: str,
    allow_empty_evidence: bool = False,
) -> list[dict[str, str]]:
    from benchmarks.wma_harness.eval_wma import (
        build_retrieved_memory_evidence as build_wma_evidence,
    )
    from benchmarks.common.prompts import build_wma_answer_messages as build_answer_messages

    evidence, _ = build_wma_evidence(list(memory_items), category)
    return build_answer_messages(
        question=question,
        question_type=category,
        memory_evidence=evidence,
        allow_empty_evidence=allow_empty_evidence,
    )


def parse_wma_policy_answer(raw: str) -> str:
    from benchmarks.common.prompts import parse_answer_response

    return parse_answer_response(raw)


def iter_h2hmem_episodes(
    config: dict[str, Any],
    split: str,
    query_cache: QueryEmbeddingCache,
) -> Iterator[EvidenceEpisode]:
    from benchmarks.h2hmem_harness.eval_h2hmem import _question_image
    from benchmarks.common.prompts import (
        PPO_EMPTY_PROMPT_VERSION,
        H2HMEM_PROMPT_SOURCE as PROMPT_SOURCE,
        H2HMEM_PROMPT_VERSION as PROMPT_VERSION,
        h2hmem_ppo_empty_prompt_sha256 as ppo_empty_prompt_sha256,
        h2hmem_prompt_sha256 as prompt_sha256,
    )

    source_questions = validated_source_questions(config, split)
    visual_categories = {
        str(value).upper() for value in config.get("visual_categories", [])
    }
    prompt_digest = prompt_sha256()
    retrieval_settings = resolve_retrieval_settings(config)
    graph_options = retrieval_settings["graph_options"]
    indexes: dict[str, Any] = {}
    index_signatures: dict[str, str] = {}
    for row in source_questions:
        variant = str(row.metadata["variant"])
        dataset_name = f"{variant}_{row.source_id}"
        query_vector = query_cache.get_by_id(row.question_id)
        if query_vector is None:
            raise KeyError(f"Missing cached query embedding: {row.question_id}")
        if dataset_name not in indexes:
            dataset_dir = Path(config["memory_bank"]) / "datasets" / dataset_name
            indexes[dataset_name] = build_graph_index(
                dataset_dir,
                graph_options,
                visual_categories=visual_categories,
            )
            index_signatures[dataset_name] = retrieval_signature(
                dataset_dir,
                graph_options,
                vector_k=retrieval_settings["vector_k"],
                append_k=retrieval_settings["append_k"],
            )
        index = indexes[dataset_name]
        hits, retrieval_metadata = retrieve_hits(
            index,
            query_vector,
            retrieval_settings,
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
                "graph_append_k": int(graph_options["append_k"]),
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
    from benchmarks.wma_harness.questions import (
        build_gold_evidence_map,
        make_query_id as make_wma_query_id,
        session_ids,
        visible_sessions_for_checkpoint,
    )
    from benchmarks.common.prompts import (
        PPO_EMPTY_PROMPT_VERSION,
        WMA_PROMPT_SOURCE as PROMPT_SOURCE,
        WMA_PROMPT_VERSION as PROMPT_VERSION,
        wma_ppo_empty_prompt_sha256 as ppo_empty_prompt_sha256,
        wma_prompt_sha256 as prompt_sha256,
    )
    source_questions = validated_source_questions(config, split)
    paths = {row.source_id: Path(row.source_path) for row in source_questions}
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
    sample_ids = tuple(
        sample_id for sample_id in split_index.source_ids(split, data_source)
        if sample_id in paths
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
        for checkpoint in payload.get("qa_checkpoints", []) or []:
            checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
            covered_sessions = [
                str(value) for value in checkpoint.get("covered_sessions", [])
            ]
            visible_sessions = visible_sessions_for_checkpoint(
                ordered_sessions, covered_sessions
            )
            visible_session_set = set(visible_sessions)
            index, prefix_signature = build_wma_prefix_graph_index(
                source_dataset_dir,
                prefix_cache_root,
                sample_id=sample_id,
                checkpoint_id=checkpoint_id,
                visible_session_ids=visible_sessions,
                options=graph_options,
                visual_categories=visual_categories,
            )
            index_signature = retrieval_signature(
                source_dataset_dir, graph_options,
                prefix_signature=prefix_signature,
                vector_k=retrieval_settings["vector_k"],
                append_k=retrieval_settings["append_k"],
            )
            for qa_index, qa in enumerate(checkpoint.get("questions", []) or [], start=1):
                manifest_question_id = f"{sample_id}:{checkpoint_id}:Q{qa_index:03d}"
                if not split_index.contains_question(
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
                        "graph_append_k": int(graph_options["append_k"]),
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


def configured_split_manifest(config: dict[str, Any]) -> SplitManifestIndex:
    path = str(config.get("split_manifest", "")).strip()
    if not path:
        raise ValueError("Evidence-policy training and evaluation require split_manifest")
    return SplitManifestIndex(path)


def require_embedding_dim(config: dict[str, Any]) -> int:
    """Validate and normalize the explicit dimension for policy and cached vectors."""
    policy = config.get("policy") or {}
    dimension = policy.get("embedding_dim") if isinstance(policy, dict) else None
    if isinstance(dimension, str):
        try:
            dimension = int(dimension.strip())
        except ValueError:
            dimension = None
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
        raise ValueError(
            "A positive integer embedding dimension is required for training and "
            "evaluation. Set HIVE_EMBEDDING_DIM, pass --embedding-dim, or set "
            "policy.embedding_dim in the policy config."
        )
    policy["embedding_dim"] = dimension
    return dimension


def build_policy(config: dict[str, Any], device: torch.device) -> EvidenceSelectionPolicy:
    require_embedding_dim(config)
    return EvidenceSelectionPolicy(**config["policy"]).to(device)


def build_trainer(
    config: dict[str, Any], policy: EvidenceSelectionPolicy
) -> PPOTrainer:
    dimension = require_embedding_dim(config)
    if policy.embedding_dim != dimension:
        raise ValueError(
            f"Policy embedding dimension {policy.embedding_dim} does not match "
            f"configured embedding dimension {dimension}"
        )
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


def require_model_base_url(config: dict[str, Any]) -> str:
    """Require an explicit answer endpoint only for operations that call it."""
    model = config.get("model") or {}
    raw_url = model.get("base_url") if isinstance(model, dict) else None
    base_url = raw_url.strip().rstrip("/") if isinstance(raw_url, str) else ""
    if not base_url:
        raise ValueError(
            "An answer model URL is required for training and evaluation. "
            "Set HIVE_ANSWER_BASE_URL, pass --model-base-url, or set model.base_url "
            "in the policy config."
        )
    return base_url


def require_query_cache(config: dict[str, Any]) -> Path:
    """Require a cache path after model-based defaults and explicit overrides."""
    value = config.get("query_cache")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(
            "A query embedding cache is required for training and evaluation. "
            "Set HIVE_QUERY_CACHE or query_cache in the policy config, or set "
            "HIVE_EMBEDDING_MODEL to resolve the standard cache path."
        )
    return Path(value)


def build_environment(
    config: dict[str, Any],
) -> tuple[VLMAnswerClient, EvidenceSelectionEnv]:
    base_url = require_model_base_url(config)
    model = config["model"]
    api_key = str(model.get("api_key", "")).strip()
    api_key_env = str(model.get("api_key_env", "")).strip()
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
    api_key = api_key_for("answer", api_key)
    benchmark = str(config.get("benchmark", "memgallery")).lower()
    client_class = VLMAnswerClient
    if benchmark == "wma":
        from benchmarks.wma_harness.eval_wma import VLMAnswerClient as WMAAnswerClient

        client_class = WMAAnswerClient
    client = client_class(
        model=model["name"],
        base_url=base_url,
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
    gvv_index = (
        GVVArtifactIndex(
            evidence["gvv_run_dir"],
            max_views_per_image=int(evidence.get("max_views_per_image", 0)),
        )
        if evidence.get("gvv_run_dir")
        else None
    )
    builder = EvidenceComposer(
        store, gvv_index=gvv_index, visual_categories=visual_categories
    )
    return client, EvidenceSelectionEnv(
        client,
        builder,
        cache=cache,
        visual_categories=visual_categories,
    )


def validate_runtime(config: dict[str, Any], *, require_split: bool) -> None:
    dimension = require_embedding_dim(config)
    query_cache = require_query_cache(config)
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
    if evidence.get("gvv_run_dir"):
        gvv_index = GVVArtifactIndex(
            evidence["gvv_run_dir"],
            max_views_per_image=int(evidence.get("max_views_per_image", 0)),
        )
        if bool(evidence.get("strict_gvv_coverage", False)):
            coverage = gvv_index.audit(memory_image_paths(config["memory_bank"]))
            if coverage["missing_records"] or coverage["missing_crop_files"]:
                raise ValueError(f"Incomplete GVV coverage for memory bank: {coverage}")
    missing = [name for name in ("vectors.npy", "metadata.jsonl") if not (query_cache / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing query cache files in {query_cache}: {', '.join(missing)}. "
            f"Generate the {dimension}-dimensional query cache before train/eval."
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
