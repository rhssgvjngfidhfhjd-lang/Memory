#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
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
from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient  # noqa: E402
from benchmarks.memgallery_harness.runner.metrics import (  # noqa: E402
    calculate_calls_mb,
    calculate_calls_qa,
    combine_call_metrics,
    summarize_results,
)
from benchmarks.memgallery_harness.runner.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    format_question_prompt,
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
from evidence_policy.ppo import PPOBuffer, PPOTrainer, load_policy_checkpoint, save_json  # noqa: E402
from evidence_policy.retrieval import (  # noqa: E402
    build_graph_index,
    build_wma_prefix_graph_index,
    resolve_graph_options,
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
            config.get("excluded_categories", ["MB"] if benchmark == "wma" else ["AR"])
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
    ppo_metrics_path = output_dir / "ppo_metrics.jsonl"
    start_epoch = 0
    train_question_count = 0
    if args.resume:
        state = trainer.load_checkpoint(args.resume)
        if not resume_configs_match(state.get("config"), config):
            raise ValueError(
                "Checkpoint configuration does not match the current evidence-policy "
                "config (only output_dir may differ for a clean recovery run)"
            )
        start_epoch = int(state["epoch"]) + 1
        train_question_count = int(
            state["extra"].get(
                "train_question_count",
                state["extra"].get("train_question_step", 0),
            )
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
        for episode_index, episode in enumerate(episodes, start=1):
            train_question_count += 1
            with torch.no_grad():
                rollout = rollout_with_endpoint_recovery(
                    env,
                    episode,
                    EvidenceStrategy.PPO,
                    policy=policy,
                    deterministic=False,
                )
            step = rollout.policy_step
            assert step is not None
            train_rollouts.append(rollout_record(rollout, episode))
            if rollout.error:
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
                if len(buffer) >= int(config["ppo"]["rollout_batch_size"]):
                    metrics = trainer.update(buffer)
                    updates.append(metrics)
                    append_jsonl(
                        ppo_metrics_path,
                        {
                            "epoch": epoch,
                            "question_count": train_question_count,
                            "update_step": trainer.update_steps,
                            **metrics,
                        },
                    )
                    buffer.clear()
            validation_phase = validation_points.get(episode_index)
            if validation_phase is not None:
                validations.append(
                    run_training_validation(
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
                    )
                )
        if len(buffer):
            metrics = trainer.update(buffer)
            updates.append(metrics)
            append_jsonl(
                ppo_metrics_path,
                {
                    "epoch": epoch,
                    "question_count": train_question_count,
                    "update_step": trainer.update_steps,
                    **metrics,
                },
            )
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
            },
        )
        summary = {
            "epoch": epoch,
            "update_step": trainer.update_steps,
            "train_question_count": train_question_count,
            "train_episodes": len(episodes),
            "successful_rollouts": len(rewards),
            "failed_rollouts": failed_rollouts,
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
    profiles: dict[str, str],
    policy: EvidenceSelectionPolicy,
    *,
    output_dir: Path,
    epoch: int,
    phase: str,
    update_step: int,
    train_question_count: int,
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
    return {
        "phase": phase,
        "update_step": int(update_step),
        "train_question_count": int(train_question_count),
        "metrics": validation["metrics"],
        "rollouts": str(trace),
    }


def prepare_initial_validation(
    config: dict[str, Any],
    env: EvidenceSelectionEnv,
    query_cache: QueryEmbeddingCache,
    profiles: dict[str, str],
    policy: EvidenceSelectionPolicy,
    trainer: PPOTrainer,
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
                extra={"initial_validation": event, "device": str(device)},
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
        extra={"initial_validation": event, "device": str(device)},
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
    if strategy is EvidenceStrategy.PPO:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for --strategy ppo")
        policy = build_policy(config, torch.device(args.device))
        load_policy_checkpoint(policy, args.checkpoint, device=args.device)
        policy.eval()
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
    )
    output = Path(config["output_dir"]) / "eval" / f"{args.split}_{strategy.value}"
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "metrics.json", result["metrics"])
    with (output / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["rollouts"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
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
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
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
    stored_copy = dict(stored)
    current_copy = dict(current)
    stored_copy.pop("output_dir", None)
    current_copy.pop("output_dir", None)
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
            raise RuntimeError(
                f"Rollout failed for {episode.query_id}: {rollout.error}"
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
    graph_options = resolve_graph_options(config)
    for dataset_name in dataset_names:
        path = data_dir / "dialog" / f"{dataset_name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile = payload.get("character_profile") or {}
        speaker = f"user ({profile.get('name')})" if profile.get("name") else "user"
        system_prompt = SYSTEM_PROMPT
        if profiles.get(dataset_name):
            system_prompt += (
                "\n\nUser profile (background about the person the memories are about):\n"
                + profiles[dataset_name]
            )
        dataset_dir = Path(config["memory_bank"]) / "datasets" / dataset_name
        index = (
            build_graph_index(dataset_dir, graph_options)
            if graph_options is not None
            else SimpleMemoryIndex(dataset_dir)
        )
        index_signature = retrieval_signature(dataset_dir, graph_options)
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
            query_vector = query_cache.get_by_id(query_id)
            if query_vector is None:
                raise KeyError(f"Missing cached query embedding: {query_id}")
            hits = index.search(query_vector, top_k=int(config["top_k"]), category=category)
            raw_clue = qa.get("clue", [])
            clue = raw_clue if isinstance(raw_clue, list) else []
            yield EvidenceEpisode(
                query_id=query_id,
                dataset=dataset_name,
                category=category,
                question_prompt=format_question_prompt(question, category, speaker, "assistant"),
                system_prompt=system_prompt,
                ground_truth=str(qa.get("answer", "")),
                query_embedding=query_vector,
                memory_hits=tuple(hits),
                query_image=query_image,
                clue=tuple(str(item) for item in clue),
                retrieval_signature=index_signature,
                metadata={
                    "manifest_question_id": manifest_question_id,
                    "retrieval_mode": "graph_append" if graph_options else "vector",
                    "vector_k": int(config["top_k"]),
                    "graph_append_k": int(graph_options["append_k"]) if graph_options else 0,
                },
            )


def iter_h2hmem_episodes(
    config: dict[str, Any],
    split: str,
    query_cache: QueryEmbeddingCache,
) -> Iterator[EvidenceEpisode]:
    from benchmarks.h2hmem_harness.eval_h2hmem import (
        SYSTEM_PROMPT as H2HMEM_SYSTEM_PROMPT,
        _question_image,
        _question_prompt,
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
    graph_options = resolve_graph_options(config)
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
                dataset_dir, graph_options
            )
        index = indexes[dataset_name]
        hits = index.search(
            query_vector,
            top_k=int(config["top_k"]),
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
            question_prompt=_question_prompt(row.question, row.category),
            system_prompt=H2HMEM_SYSTEM_PROMPT,
            ground_truth=row.answer,
            query_embedding=query_vector,
            memory_hits=tuple(hits),
            query_image=query_image,
            clue=tuple(str(value) for value in row.metadata.get("answer_session", [])),
            retrieval_signature=index_signatures[dataset_name],
            metadata={
                "manifest_question_id": row.question_id,
                "variant": variant,
                "conversation_id": row.source_id,
                "session_id": row.metadata.get("session_id", ""),
                "difficulty": row.metadata.get("difficulty", ""),
                "retrieval_mode": "graph_append" if graph_options else "vector",
                "vector_k": int(config["top_k"]),
                "graph_append_k": int(graph_options["append_k"]) if graph_options else 0,
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
    from benchmarks.wma_harness.runner.prompts import SYSTEM_PROMPT as WMA_SYSTEM_PROMPT
    from benchmarks.wma_harness.runner.prompts import format_question_prompt as format_wma_prompt
    from embedding.chunk_builder import iter_wma_sample_files

    data_dir = Path(config["data_dir"])
    paths = {path.stem: path for path in iter_wma_sample_files(data_dir)}
    visual_categories = {
        str(value).upper()
        for value in config.get("visual_categories", ["VFR", "VS", "VU", "CMR", ""])
    }
    excluded_categories = parse_excluded_categories(
        config.get("excluded_categories", ["MB"])
    )
    split_index = configured_split_manifest(config)
    data_source = evidence_data_source(config)
    sample_ids = (
        split_index.source_ids(split, data_source)
        if split_index is not None
        else tuple(config["split"][split])
    )
    graph_options = resolve_graph_options(config)
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
            prefix_signature = ""
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
                source_dataset_dir,
                graph_options,
                prefix_signature=prefix_signature,
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
                hits = index.search(
                    query_vector,
                    top_k=int(config["top_k"]),
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
                    question_prompt=format_wma_prompt(question, category),
                    system_prompt=WMA_SYSTEM_PROMPT,
                    ground_truth=str(qa.get("answer", "")),
                    query_embedding=query_vector,
                    memory_hits=tuple(hits),
                    retrieval_signature=index_signature,
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
                        "evidence": qa.get("evidence", []),
                        "covered_sessions": covered_sessions,
                        "visible_sessions": visible_sessions,
                        "retrieval_mode": "graph_append" if graph_options else "vector",
                        "vector_k": int(config["top_k"]),
                        "graph_append_k": int(graph_options["append_k"]) if graph_options else 0,
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
    benchmark = str(config.get("benchmark", "memgallery")).lower()
    client_class = VLMAnswerClient
    if benchmark == "wma":
        from benchmarks.wma_harness.runner.answer_client import VLMAnswerClient as WMAAnswerClient

        client_class = WMAAnswerClient
    client = client_class(
        model=model["name"],
        base_url=model["base_url"],
        api_key=model["api_key"],
        num_predict=int(model["max_tokens"]),
        timeout=int(model["timeout"]),
        retries=int(model["retries"]),
        think=bool(model["think"]),
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
