#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
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
from benchmarks.memgallery_harness.runner.metrics import summarize_results  # noqa: E402
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
    DialogueStore,
    EvidenceChainBuilder,
    EvidenceStrategy,
    WMADialogueStore,
)
from evidence_policy.policy import EvidenceSelectionPolicy  # noqa: E402
from evidence_policy.ppo import PPOBuffer, PPOTrainer, load_policy_checkpoint, save_json  # noqa: E402
from evidence_policy.rollout import (  # noqa: E402
    EvidenceEpisode,
    EvidenceRollout,
    EvidenceSelectionEnv,
    RolloutCache,
)
from evidence_policy.split_manifest import SplitManifestIndex  # noqa: E402
from hive_mem.retriever import SimpleMemoryIndex  # noqa: E402


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
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("prepare-split", help="Create balanced benchmark splits")
    split_parser.add_argument("--trials", type=int, default=20000)

    train_parser = subparsers.add_parser("train", help="Train the PPO policy")
    train_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train_parser.add_argument("--epochs", type=int, default=0)
    train_parser.add_argument("--max-train-episodes", type=int, default=0)
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
    if args.split_manifest:
        config["split_manifest"] = str(Path(args.split_manifest).expanduser().resolve())
    if config.get("split_manifest"):
        split_index = SplitManifestIndex(config["split_manifest"])
        config["split_manifest"] = str(split_index.path)
        config["split_manifest_file_sha256"] = split_index.file_sha256
    if args.command == "prepare-split":
        prepare_split(config, config_path, trials=args.trials)
    elif args.command == "train":
        train(config, args)
    else:
        evaluate_command(config, args)


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("data_dir", "memory_bank", "query_cache", "output_dir"):
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
            value if value.is_absolute() else (ROOT / value).resolve()
        )
    return config


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
    start_epoch = 0
    train_question_count = 0
    if args.resume:
        state = trainer.load_checkpoint(args.resume)
        if state.get("config") != config:
            raise ValueError(
                "Checkpoint configuration does not match the current evidence-policy config"
            )
        start_epoch = int(state["epoch"]) + 1
        train_question_count = int(
            state["extra"].get(
                "train_question_count",
                state["extra"].get("train_question_step", 0),
            )
        )
    client, env = build_environment(config)
    client.assert_model_available()
    query_cache = QueryEmbeddingCache(
        config["query_cache"], expected_dim=int(config["policy"]["embedding_dim"])
    )
    profiles = load_profiles(config)
    epochs = int(args.epochs or config["ppo"]["epochs"])
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    output_dir = Path(config["output_dir"])
    ppo_metrics_path = output_dir / "ppo_metrics.jsonl"
    if start_epoch == 0 and ppo_metrics_path.exists():
        ppo_metrics_path.unlink()
    for epoch in range(start_epoch, epochs):
        buffer = PPOBuffer()
        episodes = list(iter_episodes(config, "train", query_cache, profiles))
        random.shuffle(episodes)
        if args.max_train_episodes:
            episodes = episodes[: args.max_train_episodes]
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
        validations: list[dict[str, Any]] = []
        failed_rollouts = 0
        for episode_index, episode in enumerate(episodes, start=1):
            train_question_count += 1
            with torch.no_grad():
                rollout = env.rollout(episode, EvidenceStrategy.PPO, policy=policy)
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
            deterministic=True,
            limit=int(config["ppo"].get("validation_limit", 0)),
        )
    finally:
        if was_training:
            policy.train()
    filename = (
        f"epoch_{epoch:03d}_rollouts.jsonl"
        if phase == "end"
        else f"epoch_{epoch:03d}_{phase}_rollouts.jsonl"
    )
    trace = output_dir / "validation" / filename
    write_jsonl(trace, validation["rollouts"])
    return {
        "phase": phase,
        "update_step": int(update_step),
        "train_question_count": int(train_question_count),
        "metrics": validation["metrics"],
        "rollouts": str(trace),
    }


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
            rollout = env.rollout(
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
            }
        )
        rollouts.append(rollout_record(rollout, episode, source_groups=source_groups))
    if str(config.get("benchmark", "memgallery")).lower() == "wma":
        from benchmarks.wma_harness.runner.metrics import summarize_results as summarize_wma_results

        metrics = summarize_wma_results(records, k=int(config["top_k"]))
    else:
        metrics = summarize_results(records, k=int(config["top_k"]))
    metrics["mean_reward"] = (
        float(np.mean([row["reward"] for row in rollouts])) if rollouts else 0.0
    )
    metrics["evidence_actions"] = summarize_evidence_actions(rollouts)
    metrics["cached_rollouts"] = sum(bool(row["cached"]) for row in rollouts)
    metrics["errors"] = sum(bool(row["error"]) for row in rollouts)
    return {"metrics": metrics, "rollouts": rollouts}


def iter_episodes(
    config: dict[str, Any],
    split: str,
    query_cache: QueryEmbeddingCache,
    profiles: dict[str, str],
) -> Iterator[EvidenceEpisode]:
    if str(config.get("benchmark", "memgallery")).lower() == "wma":
        yield from iter_wma_episodes(config, split, query_cache)
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
        index = SimpleMemoryIndex(Path(config["memory_bank"]) / "datasets" / dataset_name)
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
                metadata={"manifest_question_id": manifest_question_id},
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
    for sample_id in sample_ids:
        payload = json.loads(paths[sample_id].read_text(encoding="utf-8"))
        ordered_sessions = session_ids(payload)
        gold_points = build_gold_evidence_map(payload)
        point_sessions = {
            evidence_id: row["session_id"]
            for evidence_id, row in gold_points.items()
        }
        index = SimpleMemoryIndex(
            Path(config["memory_bank"]) / "datasets" / sample_id,
            visual_categories=visual_categories,
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
    explicit = str(config.get("data_source", "")).strip()
    if explicit:
        return explicit
    benchmark = str(config.get("benchmark", "memgallery")).strip().lower()
    if benchmark == "wma":
        return "worldmemarena_lifelong"
    if benchmark == "memgallery":
        return "mem_gallery"
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
    store = (
        WMADialogueStore(config["data_dir"])
        if benchmark == "wma"
        else DialogueStore(config["data_dir"])
    )
    builder = EvidenceChainBuilder(store, visual_categories=visual_categories)
    return client, EvidenceSelectionEnv(
        client,
        builder,
        cache=cache,
        rng=random.Random(int(config["seed"])),
        visual_categories=visual_categories,
    )


def validate_runtime(config: dict[str, Any], *, require_split: bool) -> None:
    for key in ("data_dir", "memory_bank"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"Missing {key}: {config[key]}")
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
            source = evidence_data_source(config)
            if source not in split_index.data_sources:
                raise ValueError(
                    f"Configured data_source {source!r} is absent from {split_index.path}"
                )
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
            counts[str(action["text"])] += 1
            visual = action.get("visual")
            if visual:
                counts[str(visual)] += 1
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
