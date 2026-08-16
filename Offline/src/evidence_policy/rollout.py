from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient
from benchmarks.memgallery_harness.runner.metrics import f1_score
from hive_mem.retriever import MemoryHit

from .evidence import (
    EvidenceChainBuilder,
    EvidenceStrategy,
    MAUEvidenceAction,
    PolicyObservation,
    PolicyStep,
    action_signature,
    choose_baseline_actions,
    make_policy_observation,
)
from .policy import EvidenceSelectionPolicy


EVIDENCE_CACHE_VERSION = 2


class RewardFunction(Protocol):
    def __call__(self, prediction: str, ground_truth: str) -> float: ...


class F1Reward:
    def __call__(self, prediction: str, ground_truth: str) -> float:
        return f1_score(prediction, ground_truth)


@dataclass(frozen=True)
class EvidenceEpisode:
    query_id: str
    dataset: str
    category: str
    question_prompt: str
    system_prompt: str
    ground_truth: str
    query_embedding: Sequence[float]
    memory_hits: tuple[MemoryHit, ...]
    query_image: dict[str, Any] | None = None
    clue: tuple[str, ...] = ()


@dataclass
class EvidenceRollout:
    query_id: str
    dataset: str
    category: str
    observation: PolicyObservation
    actions: tuple[MAUEvidenceAction, ...]
    answer: str
    reward: float
    error: str
    cached: bool
    policy_step: PolicyStep | None = None

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "query_id": self.query_id,
            "dataset": self.dataset,
            "category": self.category,
            "actions": [action.to_dict() for action in self.actions],
            "answer": self.answer,
            "reward": self.reward,
            "error": self.error,
            "cached": self.cached,
        }
        if self.policy_step is not None:
            row["joint_log_prob"] = float(self.policy_step.joint_log_prob.detach().cpu())
            row["value"] = float(self.policy_step.value.detach().cpu())
            row["entropy"] = float(self.policy_step.entropy.detach().cpu())
        return row


class RolloutCache:
    """Append-only JSONL cache; the last valid row for a key wins."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._rows: dict[str, dict[str, Any]] | None = None

    def get(self, key: str) -> dict[str, Any] | None:
        self._load()
        assert self._rows is not None
        row = self._rows.get(key)
        return dict(row) if row is not None else None

    def put(self, key: str, row: dict[str, Any]) -> None:
        self._load()
        assert self._rows is not None
        payload = {"cache_key": key, **row}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._rows[key] = payload

    def _load(self) -> None:
        if self._rows is not None:
            return
        self._rows = {}
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("cache_key", ""))
                if key:
                    self._rows[key] = row


class EvidenceSelectionEnv:
    def __init__(
        self,
        client: VLMAnswerClient,
        chain_builder: EvidenceChainBuilder,
        *,
        reward_function: RewardFunction | None = None,
        cache: RolloutCache | None = None,
        rng: random.Random | None = None,
    ):
        self.client = client
        self.chain_builder = chain_builder
        self.reward_function = reward_function or F1Reward()
        self.cache = cache
        self.rng = rng or random.Random()

    def rollout(
        self,
        episode: EvidenceEpisode,
        strategy: EvidenceStrategy,
        *,
        policy: EvidenceSelectionPolicy | None = None,
        deterministic: bool = False,
    ) -> EvidenceRollout:
        observation = make_policy_observation(
            episode.query_embedding, episode.memory_hits, episode.category
        )
        policy_step = None
        if strategy is EvidenceStrategy.PPO:
            if policy is None:
                raise ValueError("PPO strategy requires an EvidenceSelectionPolicy")
            policy_device = next(policy.parameters()).device
            observation = observation.to(policy_device)
            policy_step = (
                policy.select_deterministic(observation)
                if deterministic
                else policy.sample(observation)
            )
            actions = policy_step.actions
        else:
            actions = choose_baseline_actions(
                episode.memory_hits, episode.category, strategy, self.rng
            )
        cache_key = self._cache_key(episode, actions)
        cached = self.cache.get(cache_key) if self.cache is not None else None
        if cached is not None:
            answer = str(cached.get("answer", ""))
            reward = float(self.reward_function(answer, episode.ground_truth))
            error = str(cached.get("error", ""))
            was_cached = True
        else:
            items = self.chain_builder.build(
                episode.dataset, episode.category, episode.memory_hits, actions
            )
            try:
                answer = self.client.answer(
                    system_prompt=episode.system_prompt,
                    memory_items=items,
                    question_prompt=episode.question_prompt,
                    query_image=episode.query_image,
                    category=episode.category,
                )
                error = ""
            except Exception as exc:
                answer = ""
                error = str(exc)
            reward = float(self.reward_function(answer, episode.ground_truth))
            was_cached = False
            if self.cache is not None and not error:
                self.cache.put(
                    cache_key,
                    {
                        "query_id": episode.query_id,
                        "actions": [action.to_dict() for action in actions],
                        "answer": answer,
                        "reward": reward,
                        "error": error,
                    },
                )
        return EvidenceRollout(
            query_id=episode.query_id,
            dataset=episode.dataset,
            category=episode.category,
            observation=observation,
            actions=tuple(actions),
            answer=answer,
            reward=reward,
            error=error,
            cached=was_cached,
            policy_step=policy_step,
        )

    def _cache_key(
        self,
        episode: EvidenceEpisode,
        actions: Sequence[MAUEvidenceAction],
    ) -> str:
        config = {
            "cache_version": EVIDENCE_CACHE_VERSION,
            "model": self.client.model,
            "base_url": self.client.base_url,
            "num_predict": self.client.num_predict,
            "think": self.client.think,
            "backend": self.client.backend,
            "system_prompt": episode.system_prompt,
            "question_prompt": episode.question_prompt,
            "category": episode.category,
            "memory_evidence": [
                {
                    "id": hit.item.id,
                    "summary": hit.item.summary,
                    "source_dialogue_ids": hit.item.metadata.get("source_dialogue_ids", []),
                    "image_paths": hit.item.metadata.get("image_paths", []),
                    "image_captions": hit.item.metadata.get("image_captions", []),
                }
                for hit in episode.memory_hits
            ],
        }
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        raw = f"{episode.query_id}\n{action_signature(actions)}\n{config_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
