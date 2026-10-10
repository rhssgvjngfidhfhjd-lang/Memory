from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from benchmarks.common.answer_client import VLMAnswerClient
from benchmarks.common.metrics import (
    ChatPromptTokenCounter,
    calculate_usage_cost,
    f1_score,
    load_model_efficiency_profile,
)
from src.utils import CONFIG_ROOT

import numpy as np
from src.retriever import MemoryHit

from .evidence import (
    EvidenceComposer,
    EvidenceStrategy,
    MemoryEvidenceAction,
    PolicyObservation,
    PolicyStep,
    action_signature,
    choose_full_evidence_actions,
    make_policy_observation,
)
from .ppo import EvidenceSelectionPolicy, SlidingCostNormalizer


EVIDENCE_CACHE_VERSION = 9
ALL_ZERO_REWARD = -1.0


AnswerPromptBuilder = Callable[[Sequence[dict[str, Any]]], str]
AnswerMessagesBuilder = Callable[..., list[dict[str, str]]]
AnswerParser = Callable[[str], str]


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
    answer_prompt_builder: AnswerPromptBuilder | None = field(
        default=None, repr=False, compare=False
    )
    answer_messages_builder: AnswerMessagesBuilder | None = field(
        default=None, repr=False, compare=False
    )
    answer_parser: AnswerParser | None = field(default=None, repr=False, compare=False)
    prepend_memory_context: bool = True
    prompt_signature: str = ""


@dataclass
class EvidenceRollout:
    query_id: str
    dataset: str
    category: str
    observation: PolicyObservation
    actions: tuple[MemoryEvidenceAction, ...]
    answer: str
    raw_answer: str
    reward: float
    error: str
    cached: bool
    answer_attempts: int | None = None
    answer_failed_attempts: int | None = None
    answer_usage: dict[str, int] | None = None
    answer_image_count: int | None = None
    answer_total_image_count: int | None = None
    policy_step: PolicyStep | None = None
    quality_reward: float | None = None
    raw_cost: float | None = None
    base_cost: float | None = None
    cost_min: float | None = None
    incremental_cost: float | None = None
    transformed_cost: float | None = None
    cost_max: float | None = None
    normalized_cost: float | None = None
    cost_weight: float | None = None
    cost_scale_alpha: float | None = None
    task_reward_std: float | None = None
    normalized_cost_std: float | None = None
    effective_cost_weight: float | None = None
    cost_window_count: int | None = None
    cost_normalizer_active: bool | None = None
    cost_error: str = ""
    skipped_invalid_response: bool = False
    skipped_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "query_id": self.query_id,
            "dataset": self.dataset,
            "category": self.category,
            "actions": [action.to_dict() for action in self.actions],
            "answer": self.answer,
            "answer_raw_response": self.raw_answer,
            "reward": self.reward,
            "error": self.error,
            "cached": self.cached,
            "answer_attempts": self.answer_attempts,
            "answer_failed_attempts": self.answer_failed_attempts,
            "answer_usage": self.answer_usage,
            "answer_image_count": self.answer_image_count,
            "answer_total_image_count": self.answer_total_image_count,
            "quality_reward": (
                self.reward if self.quality_reward is None else self.quality_reward
            ),
            "raw_cost": self.raw_cost,
            "base_cost": self.base_cost,
            "cost_min": self.cost_min,
            "incremental_cost": self.incremental_cost,
            "transformed_cost": self.transformed_cost,
            "cost_max": self.cost_max,
            "normalized_cost": self.normalized_cost,
            "cost_weight": self.cost_weight,
            "cost_scale_alpha": self.cost_scale_alpha,
            "task_reward_std": self.task_reward_std,
            "normalized_cost_std": self.normalized_cost_std,
            "effective_cost_weight": self.effective_cost_weight,
            "cost_window_count": self.cost_window_count,
            "cost_normalizer_active": self.cost_normalizer_active,
            "cost_error": self.cost_error,
            "skipped_invalid_response": self.skipped_invalid_response,
            "skipped_error": self.skipped_error,
            "evidence_availability_mask": [
                [bool(value) for value in values]
                for values in self.observation.evidence_availability_mask.detach()
                .cpu()
                .tolist()
            ],
        }
        if self.policy_step is not None:
            row["joint_log_prob"] = float(self.policy_step.joint_log_prob.detach().cpu())
            row["value"] = float(self.policy_step.value.detach().cpu())
            row["entropy"] = float(self.policy_step.entropy.detach().cpu())
        return row


def actions_are_all_zero(actions: Sequence[Any]) -> bool:
    return not actions or all(action.bitmask == "00000" for action in actions)


def _cost_profile_signature(
    config: dict[str, Any], input_price: float, output_price: float
) -> str:
    reward = config.get("reward") or {}
    model_name = str((config.get("model") or {}).get("name", ""))
    payload = {
        "protocol_version": 1,
        "cost_scope": "answer_usage_minus_empty_evidence_prompt",
        "model": model_name,
        "tokenizer_name": str(reward.get("tokenizer_name") or model_name),
        "input_per_million_usd": input_price,
        "output_per_million_usd": output_price,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


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
        self.cost_profile_signature = ""
        self.token_counter = token_counter
        self._base_prompt_tokens_by_query: dict[str, int] = {}
        if self.enabled:
            efficiency_config = Path(
                config.get("efficiency_config")
                or CONFIG_ROOT / "defaults.json"
            )
            profile = load_model_efficiency_profile(
                efficiency_config, str(config["model"]["name"])
            )
            self.input_price = float(profile["pricing"]["input_per_million_usd"])
            self.output_price = float(profile["pricing"]["output_per_million_usd"])
            # Bind saved windows to the effective cost units and tokenization
            # protocol, rather than to an efficiency file's mutable path.
            self.cost_profile_signature = _cost_profile_signature(
                config, self.input_price, self.output_price
            )
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
            "cost_profile_signature": self.cost_profile_signature,
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(
        self, state: Any, *, checkpoint_config: dict[str, Any] | None = None
    ) -> None:
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
        for field, current_price in (
            ("input_price_per_million_usd", self.input_price),
            ("output_price_per_million_usd", self.output_price),
        ):
            if float(state.get(field, -1.0)) != current_price:
                raise ValueError(
                    f"Checkpoint cost reward {field} does not match the effective "
                    "model pricing; restore the original efficiency profile"
                )
        stored_signature = state.get("cost_profile_signature")
        if stored_signature is None and self.enabled and checkpoint_config is not None:
            stored_signature = _cost_profile_signature(
                checkpoint_config, self.input_price, self.output_price
            )
        if stored_signature is not None and stored_signature != self.cost_profile_signature:
            raise ValueError("Checkpoint cost reward profile or tokenization protocol does not match config")
        # Older checkpoints record prices but have no protocol signature.
        # Recover their protocol identity from the saved config when available.
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
        evidence_composer: EvidenceComposer,
        *,
        reward_function: RewardFunction | None = None,
        cache: RolloutCache | None = None,
        visual_categories: set[str] | frozenset[str] | None = None,
    ):
        self.client = client
        self.evidence_composer = evidence_composer
        self.reward_function = reward_function or F1Reward()
        self.cache = cache
        self.visual_categories = visual_categories

    def rollout(
        self,
        episode: EvidenceEpisode,
        strategy: EvidenceStrategy,
        *,
        policy: EvidenceSelectionPolicy | None = None,
        deterministic: bool = False,
    ) -> EvidenceRollout:
        availability = self.evidence_composer.availability(
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
        elif strategy is EvidenceStrategy.FULL:
            actions = choose_full_evidence_actions(
                episode.memory_hits,
                episode.category,
                visual_categories=self.visual_categories,
                evidence_availability_mask=availability,
            )
        else:
            raise ValueError(f"Unsupported evidence strategy: {strategy!r}")
        items = self.evidence_composer.build(
            episode.dataset, episode.category, episode.memory_hits, actions
        )
        answer_category = str(
            episode.metadata.get("answer_category", episode.category)
        )
        try:
            if episode.answer_messages_builder is not None:
                builder_kwargs = (
                    {"allow_empty_evidence": True}
                    if strategy is EvidenceStrategy.PPO
                    else {}
                )
                request = {
                    "messages": episode.answer_messages_builder(
                        items, **builder_kwargs
                    ),
                    "memory_items": items,
                    "query_image": episode.query_image,
                    "category": answer_category,
                }
            else:
                question_prompt = (
                    episode.answer_prompt_builder(items)
                    if episode.answer_prompt_builder is not None
                    else episode.question_prompt
                )
                request = {
                    "system_prompt": episode.system_prompt,
                    "memory_items": items,
                    "question_prompt": question_prompt,
                    "query_image": episode.query_image,
                    "category": answer_category,
                    "prepend_memory_context": episode.prepend_memory_context,
                }
        except Exception as exc:
            reward = float(self.reward_function("", episode.ground_truth))
            return EvidenceRollout(
                query_id=episode.query_id,
                dataset=episode.dataset,
                category=episode.category,
                observation=observation,
                actions=tuple(actions),
                answer="",
                raw_answer="",
                reward=reward,
                error=f"answer request construction failed: {exc}",
                cached=False,
                answer_attempts=0,
                answer_failed_attempts=0,
                answer_usage=None,
                answer_image_count=0,
                answer_total_image_count=0,
                policy_step=policy_step,
                quality_reward=reward,
            )
        cache_key = self._cache_key(episode, actions, items, request=request)
        cached = self.cache.get(cache_key) if self.cache is not None else None
        if cached is not None:
            raw_answer = str(cached.get("raw_answer", cached.get("answer", "")))
            answer = self._parse_answer(episode, raw_answer)
            reward = float(self.reward_function(answer, episode.ground_truth))
            error = str(cached.get("error", ""))
            answer_attempts = _optional_int(cached.get("answer_attempts"))
            answer_failed_attempts = _optional_int(
                cached.get("answer_failed_attempts")
            )
            answer_usage = _optional_usage(cached.get("answer_usage"))
            answer_image_count = _optional_int(cached.get("answer_image_count"))
            answer_total_image_count = _optional_int(cached.get("answer_total_image_count"))
            if (
                answer_image_count is None
                and inspect.getattr_static(
                    self.client, "count_answer_images", None
                )
                is not None
            ):
                answer_image_count = int(
                    self.client.count_answer_images(
                        items,
                        query_image=episode.query_image,
                        category=answer_category,
                    )
                )
            if answer_total_image_count is None and answer_image_count is not None:
                answer_total_image_count = answer_image_count * int(answer_attempts or 0)
            was_cached = True
        else:
            raw_answer = ""
            response = None
            answer_attempts: int | None = None
            answer_failed_attempts: int | None = None
            answer_usage: dict[str, int] | None = None
            answer_image_count: int | None = None
            answer_total_image_count: int | None = None
            try:
                # ``hasattr`` is not reliable for dynamic proxy clients such as
                # ``MagicMock`` because they synthesize arbitrary attributes.
                # Inspect the object statically so minimal/third-party clients
                # continue to use the plain ``answer`` compatibility path.
                if episode.answer_messages_builder is not None and (
                    inspect.getattr_static(
                        self.client, "answer_messages_with_usage", None
                    )
                    is not None
                ):
                    response = self.client.answer_messages_with_usage(**request)
                    raw_answer = str(getattr(response, "raw_text", None) or response.text)
                    answer = self._parse_answer(episode, response.text)
                    answer_attempts = int(response.attempts)
                    answer_failed_attempts = int(response.failed_attempts)
                    answer_usage = _optional_usage(response.usage)
                    answer_image_count = int(response.image_count)
                    answer_total_image_count = _response_total_image_count(response)
                elif (
                    inspect.getattr_static(
                        self.client, "answer_with_usage", None
                    )
                    is not None
                ):
                    compatibility_request = request
                    if episode.answer_messages_builder is not None:
                        messages = request["messages"]
                        compatibility_request = {
                            "system_prompt": next(
                                (
                                    message["content"]
                                    for message in messages
                                    if message["role"] == "system"
                                ),
                                "",
                            ),
                            "memory_items": items,
                            "question_prompt": next(
                                message["content"]
                                for message in reversed(messages)
                                if message["role"] == "user"
                            ),
                            "query_image": episode.query_image,
                            "category": answer_category,
                            "prepend_memory_context": False,
                        }
                    response = self.client.answer_with_usage(**compatibility_request)
                    raw_answer = response.text
                    answer = self._parse_answer(episode, raw_answer)
                    answer_attempts = int(response.attempts)
                    answer_failed_attempts = int(response.failed_attempts)
                    answer_usage = _optional_usage(response.usage)
                    answer_image_count = int(response.image_count)
                    answer_total_image_count = _response_total_image_count(response)
                else:
                    # Compatibility for minimal test or third-party clients.
                    plain_request = request
                    if episode.answer_messages_builder is not None:
                        messages = request["messages"]
                        plain_request = {
                            "system_prompt": next(
                                (
                                    message["content"]
                                    for message in messages
                                    if message["role"] == "system"
                                ),
                                "",
                            ),
                            "memory_items": items,
                            "question_prompt": next(
                                message["content"]
                                for message in reversed(messages)
                                if message["role"] == "user"
                            ),
                            "query_image": episode.query_image,
                            "category": answer_category,
                            "prepend_memory_context": False,
                        }
                    raw_answer = self.client.answer(**plain_request)
                    answer = self._parse_answer(episode, raw_answer)
                    answer_attempts = 1
                    answer_failed_attempts = 0
                error = ""
            except Exception as exc:
                answer = ""
                error = str(exc)
                if response is not None:
                    answer_attempts = int(response.attempts)
                    answer_failed_attempts = min(
                        answer_attempts, int(response.failed_attempts) + 1
                    )
                    answer_usage = _optional_usage(response.usage)
                    answer_image_count = int(response.image_count)
                    answer_total_image_count = _response_total_image_count(response)
                else:
                    answer_attempts = int(
                        getattr(exc, "attempts", int(getattr(self.client, "retries", 0)) + 1)
                    )
                    answer_failed_attempts = int(getattr(exc, "failed_attempts", answer_attempts))
                    answer_usage = _optional_usage(getattr(exc, "usage", None))
                    answer_image_count = _optional_int(getattr(exc, "image_count", None))
                    answer_total_image_count = _optional_int(getattr(exc, "total_image_count", None))
                    raw_answer = str(getattr(exc, "raw_text", None) or raw_answer)
            reward = float(self.reward_function(answer, episode.ground_truth))
            was_cached = False
            if self.cache is not None and not error:
                self.cache.put(
                    cache_key,
                    {
                        "query_id": episode.query_id,
                        "actions": [action.to_dict() for action in actions],
                        "answer": answer,
                        "raw_answer": raw_answer,
                        "reward": reward,
                        "error": error,
                        "answer_attempts": answer_attempts,
                        "answer_failed_attempts": answer_failed_attempts,
                        "answer_usage": answer_usage,
                        "answer_image_count": answer_image_count,
                        "answer_total_image_count": answer_total_image_count,
                    },
                )
        return EvidenceRollout(
            query_id=episode.query_id,
            dataset=episode.dataset,
            category=episode.category,
            observation=observation,
            actions=tuple(actions),
            answer=answer,
            raw_answer=raw_answer,
            reward=reward,
            error=error,
            cached=was_cached,
            answer_attempts=answer_attempts,
            answer_failed_attempts=answer_failed_attempts,
            answer_usage=answer_usage,
            answer_image_count=answer_image_count,
            answer_total_image_count=answer_total_image_count,
            policy_step=policy_step,
            quality_reward=reward,
        )

    def _cache_key(
        self,
        episode: EvidenceEpisode,
        actions: Sequence[MemoryEvidenceAction],
        memory_items: Sequence[dict[str, Any]],
        *,
        request: dict[str, Any],
    ) -> str:
        config = {
            "cache_version": EVIDENCE_CACHE_VERSION,
            "model": self.client.model,
            "base_url": self.client.base_url,
            "num_predict": self.client.num_predict,
            "temperature": getattr(self.client, "temperature", 0.0),
            "think": self.client.think,
            "reasoning_effort": getattr(self.client, "reasoning_effort", ""),
            "retries": getattr(self.client, "retries", 0),
            "backend": self.client.backend,
            "messages": request.get("messages"),
            "system_prompt": request.get("system_prompt", ""),
            "question_prompt": request.get("question_prompt", ""),
            "prepend_memory_context": request.get("prepend_memory_context", False),
            "prompt_signature": episode.prompt_signature,
            "category": episode.category,
            "answer_category": episode.metadata.get(
                "answer_category", episode.category
            ),
            "query_image": episode.query_image,
            "rendered_memory_items": list(memory_items),
            "answer_image_fingerprints": self._answer_image_fingerprints(
                memory_items, request
            ),
            "retrieval_signature": episode.retrieval_signature,
            "gvv_run_id": (
                self.evidence_composer.gvv_index.run_id
                if self.evidence_composer.gvv_index is not None
                else ""
            ),
            "gvv_signature": (
                self.evidence_composer.gvv_index.signature
                if self.evidence_composer.gvv_index is not None
                else ""
            ),
        }
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16]
        raw = f"{episode.query_id}\n{action_signature(actions)}\n{config_hash}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _answer_image_fingerprints(
        self,
        memory_items: Sequence[dict[str, Any]],
        request: dict[str, Any],
    ) -> list[dict[str, str]]:
        # Use the transport's image selection so category filtering, H2 answer
        # rendering, WMA images, and GVV crops match the actual request.
        if inspect.getattr_static(self.client, "_build_text_and_image_paths", None) is not None:
            _, paths = self.client._build_text_and_image_paths(
                list(memory_items),
                str(request.get("question_prompt", "")),
                request.get("query_image"),
                str(request.get("category", "")),
                prepend_memory_context=bool(request.get("prepend_memory_context", False)),
            )
        else:
            # Minimal clients receive only the already selected/rendered items.
            paths = []
            for item in memory_items:
                images = item.get("images")
                if not isinstance(images, list):
                    legacy = item.get("image")
                    images = [legacy] if isinstance(legacy, dict) else []
                paths.extend(
                    str(image["path"])
                    for image in images
                    if isinstance(image, dict) and image.get("path")
                )
            query_image = request.get("query_image")
            if query_image and query_image.get("path"):
                paths.append(str(query_image["path"]))
        fingerprints = []
        digests: dict[str, str] = {}
        for raw_path in paths:
            path = str(raw_path)
            if path not in digests:
                digest = hashlib.sha256()
                try:
                    with Path(path).open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                except OSError as exc:
                    raise OSError(
                        f"Cannot read attached answer image {path!r} for rollout caching: {exc}"
                    ) from exc
                digests[path] = digest.hexdigest()
            fingerprints.append({"path": path, "sha256": digests[path]})
        return fingerprints

    @staticmethod
    def _parse_answer(episode: EvidenceEpisode, raw_answer: str) -> str:
        if episode.answer_parser is None:
            return str(raw_answer)
        return str(episode.answer_parser(str(raw_answer)))


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _response_total_image_count(response: Any) -> int:
    count = getattr(response, "total_image_count", None)
    return int(count) if count is not None else int(response.image_count) * int(response.attempts)


def _optional_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    return {str(key): int(count) for key, count in value.items()}
