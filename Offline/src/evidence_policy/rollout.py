from __future__ import annotations

import hashlib
import inspect
import json
import random
from dataclasses import dataclass, field
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


EVIDENCE_CACHE_VERSION = 5


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
    retrieval_signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


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
    answer_attempts: int | None = None
    answer_failed_attempts: int | None = None
    answer_usage: dict[str, int] | None = None
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
            "answer_attempts": self.answer_attempts,
            "answer_failed_attempts": self.answer_failed_attempts,
            "answer_usage": self.answer_usage,
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
        visual_categories: set[str] | frozenset[str] | None = None,
    ):
        self.client = client
        self.chain_builder = chain_builder
        self.reward_function = reward_function or F1Reward()
        self.cache = cache
        self.rng = rng or random.Random()
        self.visual_categories = visual_categories

    def rollout(
        self,
        episode: EvidenceEpisode,
        strategy: EvidenceStrategy,
        *,
        policy: EvidenceSelectionPolicy | None = None,
        deterministic: bool = False,
    ) -> EvidenceRollout:
        availability = self.chain_builder.availability(
            episode.dataset, episode.category, episode.memory_hits
        )
        observation = make_policy_observation(
            episode.query_embedding,
            episode.memory_hits,
            episode.category,
            visual_categories=self.visual_categories,
            evidence_availability_mask=availability,
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
                episode.memory_hits,
                episode.category,
                strategy,
                self.rng,
                visual_categories=self.visual_categories,
                evidence_availability_mask=availability,
            )
        items = self.chain_builder.build(
            episode.dataset, episode.category, episode.memory_hits, actions
        )
        cache_key = self._cache_key(episode, actions, items)
        cached = self.cache.get(cache_key) if self.cache is not None else None
        if cached is not None:
            answer = str(cached.get("answer", ""))
            reward = float(self.reward_function(answer, episode.ground_truth))
            error = str(cached.get("error", ""))
            answer_attempts = _optional_int(cached.get("answer_attempts"))
            answer_failed_attempts = _optional_int(
                cached.get("answer_failed_attempts")
            )
            answer_usage = _optional_usage(cached.get("answer_usage"))
            was_cached = True
        else:
            answer_attempts: int | None = None
            answer_failed_attempts: int | None = None
            answer_usage: dict[str, int] | None = None
            try:
                answer_category = str(
                    episode.metadata.get("answer_category", episode.category)
                )
                request = {
                    "system_prompt": episode.system_prompt,
                    "memory_items": items,
                    "question_prompt": episode.question_prompt,
                    "query_image": episode.query_image,
                    "category": answer_category,
                }
                # ``hasattr`` is not reliable for dynamic proxy clients such as
                # ``MagicMock`` because they synthesize arbitrary attributes.
                # Inspect the object statically so minimal/third-party clients
                # continue to use the plain ``answer`` compatibility path.
                if (
                    inspect.getattr_static(
                        self.client, "answer_with_usage", None
                    )
                    is not None
                ):
                    response = self.client.answer_with_usage(**request)
                    answer = response.text
                    answer_attempts = int(response.attempts)
                    answer_failed_attempts = int(response.failed_attempts)
                    answer_usage = _optional_usage(response.usage)
                else:
                    # Compatibility for minimal test or third-party clients.
                    answer = self.client.answer(**request)
                    answer_attempts = 1
                    answer_failed_attempts = 0
                error = ""
            except Exception as exc:
                answer = ""
                error = str(exc)
                answer_attempts = int(getattr(self.client, "retries", 0)) + 1
                answer_failed_attempts = answer_attempts
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
                        "answer_attempts": answer_attempts,
                        "answer_failed_attempts": answer_failed_attempts,
                        "answer_usage": answer_usage,
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
            answer_attempts=answer_attempts,
            answer_failed_attempts=answer_failed_attempts,
            answer_usage=answer_usage,
            policy_step=policy_step,
        )

    def _cache_key(
        self,
        episode: EvidenceEpisode,
        actions: Sequence[MAUEvidenceAction],
        memory_items: Sequence[dict[str, Any]],
    ) -> str:
        config = {
            "cache_version": EVIDENCE_CACHE_VERSION,
            "model": self.client.model,
            "base_url": self.client.base_url,
            "num_predict": self.client.num_predict,
            "temperature": getattr(self.client, "temperature", 0.0),
            "think": self.client.think,
            "backend": self.client.backend,
            "system_prompt": episode.system_prompt,
            "question_prompt": episode.question_prompt,
            "category": episode.category,
            "answer_category": episode.metadata.get(
                "answer_category", episode.category
            ),
            "query_image": episode.query_image,
            "rendered_memory_items": list(memory_items),
            "retrieval_signature": episode.retrieval_signature,
            "vp_run_id": (
                self.chain_builder.vp_index.run_id
                if self.chain_builder.vp_index is not None
                else ""
            ),
            "vp_signature": (
                self.chain_builder.vp_index.signature
                if self.chain_builder.vp_index is not None
                else ""
            ),
        }
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        raw = f"{episode.query_id}\n{action_signature(actions)}\n{config_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): int(count) for key, count in value.items()}
