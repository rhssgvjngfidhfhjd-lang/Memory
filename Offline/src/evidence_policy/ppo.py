from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .evidence import (
    EVIDENCE_ORDER,
    EVIDENCE_SCHEMA_VERSION,
    MAUEvidenceAction,
    PolicyObservation,
)
from .policy import EvidenceSelectionPolicy


WANDB_SCHEMA_VERSION = "evidence-policy-v2"
WANDB_TOP_LEVEL_GROUPS = frozenset({"train", "critic", "val", "test"})
WANDB_TRAIN_FIELDS = (
    "ppo_kl",
    "pg_loss",
    "pg_clipfrac",
    "lr",
    "grad_norm",
    "entropy_loss",
)
WANDB_CRITIC_FIELDS = (
    "value_loss",
    "absolute_value_error",
    "explained_variance",
    "predicted_value_mean",
    "target_return_mean",
    "reward_mean",
    "reward_min",
    "reward_max",
)
WANDB_COST_FIELDS = (
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
    "cost_percentile_transformed_min",
    "cost_percentile_transformed_max",
    "cost_effective_transformed_min",
    "cost_effective_transformed_max",
    "cost_effective_transformed_range",
    "task_reward_std",
    "normalized_cost_std",
    "cost_scale_alpha",
    "effective_cost_weight",
    "cost_window_count",
    "cost_window_count_after",
    "cost_low_clip_rate",
    "cost_high_clip_rate",
    "cost_saturation_rate",
    "all_zero_rollout_rate",
)
WANDB_VALIDATION_FIELDS = (
    ("count", "count"),
    ("f1", "f1"),
    ("exact_match", "exact_match"),
    ("retrieval_hitrate@5", "retrieval_hitrate_at_5"),
    ("mean_reward", "reward"),
    ("errors", "errors"),
    ("cached_rollouts", "cached_rollouts"),
)
WANDB_METRIC_DEFINITIONS = (
    ("train/update_step", None),
    ("train/*", "train/update_step"),
    ("critic/update_step", None),
    ("critic/*", "critic/update_step"),
    ("val/update_step", None),
    ("val/*", "val/update_step"),
)


def define_wandb_metrics(run: Any) -> None:
    """Define the one canonical PPO metric schema on a W&B run."""
    for name, step_metric in WANDB_METRIC_DEFINITIONS:
        if step_metric is None:
            run.define_metric(name)
        else:
            run.define_metric(name, step_metric=step_metric)


def build_wandb_update_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Map one PPO update to the canonical train/critic namespaces."""
    step = int(row["update_step"])
    payload: dict[str, Any] = {
        "train/update_step": step,
        "critic/update_step": step,
        "train/epoch": int(row["epoch"]),
        "train/question_count": int(row.get("question_count", 0)),
    }
    for field in WANDB_TRAIN_FIELDS:
        _add_finite(payload, f"train/{field}", row.get(field))
    for field in WANDB_CRITIC_FIELDS:
        name = {
            "reward_mean": "rewards/mean",
            "reward_min": "rewards/min",
            "reward_max": "rewards/max",
        }.get(field, field)
        _add_finite(payload, f"critic/{name}", row.get(field))
    for field in WANDB_COST_FIELDS:
        _add_finite(payload, f"train/cost/{field}", row.get(field))
    assert_wandb_schema(payload)
    return payload


def build_wandb_validation_payload(
    event: dict[str, Any], *, epoch: int | None = None
) -> dict[str, Any]:
    """Map a live or normalized validation event to the canonical schema."""
    nested = event.get("metrics")
    metrics = nested if isinstance(nested, dict) else event
    payload: dict[str, Any] = {
        "val/update_step": int(event.get("update_step", 0)),
        "val/epoch": int(event.get("epoch", epoch if epoch is not None else 0)),
        "val/phase": str(event.get("phase", "")),
        "val/train_question_count": int(event.get("train_question_count", 0)),
    }
    for source, target in WANDB_VALIDATION_FIELDS:
        value = metrics.get(source)
        if source == "exact_match" and value is None:
            value = metrics.get("em")
        if source == "retrieval_hitrate@5" and value is None:
            value = metrics.get("retrieval_hitrate_at_5")
        if source == "mean_reward" and value is None:
            value = metrics.get("reward")
        _add_finite(payload, f"val/{target}", value)
    for field in WANDB_COST_FIELDS:
        _add_finite(payload, f"val/cost/{field}", metrics.get(field))
    evidence_actions = metrics.get("evidence_actions") or {}
    for evidence, count in evidence_actions.items():
        _add_finite(payload, f"val/evidence_actions/{evidence}", count)
    for mask, ratio in _action_mask_ratios(evidence_actions).items():
        _add_finite(payload, f"val/action_ratio/{mask}", ratio)
    assert_wandb_schema(payload)
    return payload


def build_wandb_test_summary(test_metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten final test metrics under the canonical test namespace."""
    summary: dict[str, Any] = {}
    for key in (
        "count",
        "f1",
        "exact_match",
        "em",
        "mean_reward",
        "llm_judge",
        "errors",
    ):
        if key in test_metrics:
            summary[f"test/{key}"] = test_metrics[key]
    if "retrieval_hitrate@5" in test_metrics:
        summary["test/retrieval_hitrate_at_5"] = test_metrics[
            "retrieval_hitrate@5"
        ]
    for field in WANDB_COST_FIELDS:
        if field in test_metrics:
            summary[f"test/cost/{field}"] = test_metrics[field]
    for section in ("cost_mb", "cost_qa", "cost_total"):
        _copy_summary_section(
            summary,
            test_metrics,
            section,
            (
                "available",
                "input_tokens",
                "output_tokens",
                "cost_sum_usd",
                "num_samples",
                "mean_per_sample_usd",
            ),
        )
    for section in ("latency_mb", "latency_qa", "latency_total"):
        _copy_summary_section(
            summary,
            test_metrics,
            section,
            (
                "available",
                "calls",
                "input_tokens",
                "output_tokens",
                "image_count",
                "latency_sum_seconds",
                "num_samples",
                "mean_per_sample_seconds",
            ),
        )
    calls = test_metrics.get("calls")
    if isinstance(calls, dict):
        for section in ("memory_bank", "qa", "total"):
            values = calls.get(section)
            if not isinstance(values, dict):
                continue
            for field in (
                "available",
                "total_calls",
                "failed_calls",
                "successful_calls",
                "num_samples",
                "mean_per_sample",
            ):
                if field in values:
                    summary[f"test/calls/{section}/{field}"] = values[field]
    assert_wandb_schema(summary)
    return summary


def assert_wandb_schema(payload: dict[str, Any]) -> None:
    invalid = sorted(
        key
        for key in payload
        if key.split("/", 1)[0] not in WANDB_TOP_LEVEL_GROUPS
    )
    if invalid:
        raise ValueError(
            f"W&B keys violate {WANDB_SCHEMA_VERSION}: {', '.join(invalid)}"
        )


def _action_mask_ratios(evidence_actions: Any) -> dict[str, float]:
    values = evidence_actions if isinstance(evidence_actions, dict) else {}
    counts = {
        mask: max(0, int(values.get(f"mask:{mask}", 0)))
        for mask in (f"{value:05b}" for value in range(32))
    }
    total = sum(counts.values())
    return {
        mask: (count / total if total else 0.0)
        for mask, count in counts.items()
    }


def _copy_summary_section(
    target: dict[str, Any],
    source: dict[str, Any],
    section: str,
    fields: Sequence[str],
) -> None:
    values = source.get(section)
    if not isinstance(values, dict):
        return
    for field in fields:
        if values.get(field) is not None:
            target[f"test/{section}/{field}"] = values[field]


def _add_finite(payload: dict[str, Any], key: str, value: Any) -> None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(normalized):
        payload[key] = normalized


@dataclass(frozen=True)
class PPOTransition:
    observation: PolicyObservation
    actions: tuple[MAUEvidenceAction, ...]
    old_log_prob: float
    old_value: float
    reward: float
    done: bool = True


class PPOBuffer:
    def __init__(self) -> None:
        self.transitions: list[PPOTransition] = []

    def add(
        self,
        observation: PolicyObservation,
        actions: Sequence[MAUEvidenceAction],
        *,
        old_log_prob: float,
        old_value: float,
        reward: float,
        done: bool = True,
    ) -> None:
        observation.validate()
        self.transitions.append(
            PPOTransition(
                observation=_observation_to_cpu(observation),
                actions=tuple(actions),
                old_log_prob=float(old_log_prob),
                old_value=float(old_value),
                reward=float(reward),
                done=bool(done),
            )
        )

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)


class SlidingCostNormalizer:
    """Robustly normalize sqrt incremental cost over paired recent rollouts."""

    STATE_VERSION = 2

    def __init__(
        self,
        *,
        window_size: int = 512,
        min_window_size: int = 128,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.95,
        initial_range_floor_ratio: float = 0.25,
        range_epsilon: float = 1e-12,
        std_epsilon: float = 1e-8,
        cost_transform: str = "sqrt_incremental",
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_window_size <= 0 or min_window_size > window_size:
            raise ValueError("min_window_size must be in [1, window_size]")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError(
                "lower_quantile and upper_quantile must satisfy "
                "0 <= lower < upper <= 1"
            )
        if not 0.0 <= initial_range_floor_ratio <= 1.0:
            raise ValueError("initial_range_floor_ratio must be in [0, 1]")
        if range_epsilon <= 0:
            raise ValueError("range_epsilon must be positive")
        if std_epsilon <= 0:
            raise ValueError("std_epsilon must be positive")
        if cost_transform != "sqrt_incremental":
            raise ValueError("cost_transform must be 'sqrt_incremental'")
        self.window_size = int(window_size)
        self.min_window_size = int(min_window_size)
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.initial_range_floor_ratio = float(initial_range_floor_ratio)
        self.range_epsilon = float(range_epsilon)
        self.std_epsilon = float(std_epsilon)
        self.cost_transform = cost_transform
        self.transformed_costs: deque[float] = deque(maxlen=self.window_size)
        self.quality_rewards: deque[float] = deque(maxlen=self.window_size)
        self.initial_transformed_cost_min: float | None = None
        self.initial_transformed_cost_max: float | None = None

    def snapshot(self) -> dict[str, Any]:
        count = len(self.transformed_costs)
        active = count >= self.min_window_size
        percentile_min = None
        percentile_max = None
        effective_min = None
        effective_max = None
        effective_range = None
        if active:
            if (
                self.initial_transformed_cost_min is None
                or self.initial_transformed_cost_max is None
            ):
                raise RuntimeError("Active cost normalizer is missing its initial bounds")
            values = np.asarray(self.transformed_costs, dtype=np.float64)
            percentile_min = float(np.quantile(values, self.lower_quantile))
            percentile_max = float(np.quantile(values, self.upper_quantile))
            initial_range = (
                self.initial_transformed_cost_max
                - self.initial_transformed_cost_min
            )
            effective_min = percentile_min
            effective_range = max(
                percentile_max - percentile_min,
                self.initial_range_floor_ratio * initial_range,
                self.range_epsilon,
            )
            effective_max = effective_min + effective_range
        task_reward_std = None
        normalized_cost_std = None
        cost_scale_alpha = None
        if active:
            normalized_values = np.clip(
                (np.asarray(self.transformed_costs, dtype=np.float64) - effective_min)
                / effective_range,
                0.0,
                1.0,
            )
            task_reward_std = float(
                np.std(np.asarray(self.quality_rewards, dtype=np.float64))
            )
            normalized_cost_std = float(np.std(normalized_values))
            cost_scale_alpha = task_reward_std / (
                normalized_cost_std + self.std_epsilon
            )
        return {
            "active": active,
            "window_count": count,
            "window_size": self.window_size,
            "cost_transform": self.cost_transform,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "percentile_transformed_cost_min": percentile_min,
            "percentile_transformed_cost_max": percentile_max,
            "initial_transformed_cost_min": self.initial_transformed_cost_min,
            "initial_transformed_cost_max": self.initial_transformed_cost_max,
            "initial_transformed_cost_range": (
                None
                if self.initial_transformed_cost_min is None
                or self.initial_transformed_cost_max is None
                else self.initial_transformed_cost_max
                - self.initial_transformed_cost_min
            ),
            "effective_transformed_cost_min": effective_min,
            "effective_transformed_cost_max": effective_max,
            "effective_transformed_cost_range": effective_range,
            "task_reward_std": task_reward_std,
            "normalized_cost_std": normalized_cost_std,
            "cost_scale_alpha": cost_scale_alpha,
        }

    def normalize(
        self,
        incremental_cost: float,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> float:
        value = self.transform(incremental_cost)
        state = snapshot or self.snapshot()
        if not bool(state.get("active")):
            return 0.0
        lower = float(state["effective_transformed_cost_min"])
        denominator = float(state["effective_transformed_cost_range"])
        return float(np.clip((value - lower) / denominator, 0.0, 1.0))

    def extend(
        self,
        incremental_costs: Sequence[float],
        quality_rewards: Sequence[float],
    ) -> None:
        costs = [self.transform(value) for value in incremental_costs]
        qualities = [self._validated_quality(value) for value in quality_rewards]
        if len(costs) != len(qualities):
            raise ValueError(
                "incremental_costs and quality_rewards must have equal length"
            )
        for transformed_cost, quality_reward in zip(costs, qualities):
            self.transformed_costs.append(transformed_cost)
            self.quality_rewards.append(quality_reward)
            if (
                self.initial_transformed_cost_min is None
                and len(self.transformed_costs) >= self.min_window_size
            ):
                values = np.asarray(self.transformed_costs, dtype=np.float64)
                self.initial_transformed_cost_min = float(
                    np.quantile(values, self.lower_quantile)
                )
                self.initial_transformed_cost_max = float(
                    np.quantile(values, self.upper_quantile)
                )

    def transform(self, incremental_cost: float) -> float:
        return math.sqrt(self._validated_cost(incremental_cost))

    def inverse_transform(self, transformed_cost: float) -> float:
        value = self._validated_cost(transformed_cost)
        return value * value

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.STATE_VERSION,
            "window_size": self.window_size,
            "min_window_size": self.min_window_size,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "initial_range_floor_ratio": self.initial_range_floor_ratio,
            "range_epsilon": self.range_epsilon,
            "std_epsilon": self.std_epsilon,
            "cost_transform": self.cost_transform,
            "transformed_costs": list(self.transformed_costs),
            "quality_rewards": list(self.quality_rewards),
            "initial_transformed_cost_min": self.initial_transformed_cost_min,
            "initial_transformed_cost_max": self.initial_transformed_cost_max,
            "snapshot": self.snapshot(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "state_version": self.STATE_VERSION,
            "window_size": self.window_size,
            "min_window_size": self.min_window_size,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "initial_range_floor_ratio": self.initial_range_floor_ratio,
            "range_epsilon": self.range_epsilon,
            "std_epsilon": self.std_epsilon,
            "cost_transform": self.cost_transform,
        }
        for key, value in expected.items():
            if key not in state or state[key] != value:
                raise ValueError(
                    f"Cost normalizer checkpoint mismatch for {key}: "
                    f"expected {value!r}, got {state.get(key)!r}"
                )
        costs = [
            self._validated_cost(value)
            for value in state.get("transformed_costs", [])
        ]
        qualities = [
            self._validated_quality(value)
            for value in state.get("quality_rewards", [])
        ]
        if len(costs) != len(qualities):
            raise ValueError("Cost normalizer checkpoint has unpaired window values")
        if len(costs) > self.window_size:
            raise ValueError("Cost normalizer checkpoint exceeds window_size")
        self.transformed_costs = deque(costs, maxlen=self.window_size)
        self.quality_rewards = deque(qualities, maxlen=self.window_size)
        initial_min = state.get("initial_transformed_cost_min")
        self.initial_transformed_cost_min = (
            None if initial_min is None else self._validated_cost(initial_min)
        )
        initial = state.get("initial_transformed_cost_max")
        self.initial_transformed_cost_max = (
            None if initial is None else self._validated_cost(initial)
        )
        if len(self.transformed_costs) >= self.min_window_size and (
            self.initial_transformed_cost_min is None
            or self.initial_transformed_cost_max is None
        ):
            raise ValueError("Cost normalizer checkpoint is missing its initial bounds")
        if (
            self.initial_transformed_cost_min is not None
            and self.initial_transformed_cost_max is not None
            and self.initial_transformed_cost_max < self.initial_transformed_cost_min
        ):
            raise ValueError("Cost normalizer checkpoint has reversed initial bounds")

    @staticmethod
    def _validated_cost(value: float) -> float:
        normalized = float(value)
        if not np.isfinite(normalized) or normalized < 0:
            raise ValueError("incremental cost must be finite and non-negative")
        return normalized

    @staticmethod
    def _validated_quality(value: float) -> float:
        quality = float(value)
        if not np.isfinite(quality):
            raise ValueError("quality reward must be finite")
        return quality


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    next_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not (len(rewards) == len(values) == len(dones)):
        raise ValueError("rewards, values, and dones must have the same length")
    advantages = np.zeros(len(rewards), dtype=np.float32)
    last_advantage = 0.0
    following_value = float(next_value)
    for index in range(len(rewards) - 1, -1, -1):
        nonterminal = 0.0 if dones[index] else 1.0
        delta = float(rewards[index]) + gamma * following_value * nonterminal - float(
            values[index]
        )
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
        following_value = float(values[index])
    returns = advantages + np.asarray(values, dtype=np.float32)
    return advantages, returns


def clipped_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    clip_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = torch.exp(new_log_prob - old_log_prob)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    return -torch.minimum(unclipped, clipped).mean(), ratio


class PPOTrainer:
    def __init__(
        self,
        policy: EvidenceSelectionPolicy,
        *,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 1.0,
        update_epochs: int = 4,
        minibatch_size: int = 32,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < clip_ratio <= 1.0:
            raise ValueError("clip_ratio must be in (0, 1]")
        if value_coefficient < 0 or entropy_coefficient < 0:
            raise ValueError("loss coefficients must be non-negative")
        if max_grad_norm <= 0 or update_epochs <= 0 or minibatch_size <= 0:
            raise ValueError("max_grad_norm, update_epochs, and minibatch_size must be positive")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        self.clip_ratio = float(clip_ratio)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.max_grad_norm = float(max_grad_norm)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.update_steps = 0

    @property
    def device(self) -> torch.device:
        return next(self.policy.parameters()).device

    def update(self, buffer: PPOBuffer) -> dict[str, float]:
        if not buffer.transitions:
            raise ValueError("Cannot update PPO with an empty buffer")
        rewards = np.asarray(
            [row.reward for row in buffer.transitions], dtype=np.float32
        )
        advantages_np, returns_np = compute_gae(
            rewards,
            [row.old_value for row in buffer.transitions],
            [row.done for row in buffer.transitions],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if len(advantages_np) > 1:
            advantages_np = (advantages_np - advantages_np.mean()) / (
                advantages_np.std() + 1e-8
            )
        metrics: list[dict[str, float]] = []
        for _ in range(self.update_epochs):
            indices = torch.randperm(len(buffer.transitions)).tolist()
            for start in range(0, len(indices), self.minibatch_size):
                batch_indices = indices[start : start + self.minibatch_size]
                metrics.append(
                    self._update_minibatch(
                        buffer.transitions,
                        batch_indices,
                        advantages_np,
                        returns_np,
                    )
                )
        self.update_steps += 1
        result = {
            key: float(np.mean([row[key] for row in metrics]))
            for key in metrics[0]
        }
        result.update(
            self._critic_diagnostics(buffer.transitions, returns_np)
        )
        result.update(
            {
                "pg_loss": result["policy_loss"],
                "entropy_loss": result["entropy"],
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "reward_mean": float(rewards.mean()),
                "reward_min": float(rewards.min()),
                "reward_max": float(rewards.max()),
                "batch_size": float(len(rewards)),
            }
        )
        return result

    def _update_minibatch(
        self,
        transitions: Sequence[PPOTransition],
        indices: Sequence[int],
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        current_log_probs: list[torch.Tensor] = []
        current_values: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for index in indices:
            row = transitions[index]
            step = self.policy.evaluate_actions(
                row.observation.to(self.device), row.actions
            )
            current_log_probs.append(step.joint_log_prob)
            current_values.append(step.value)
            entropies.append(step.entropy)
        new_log_prob = torch.stack(current_log_probs)
        value = torch.stack(current_values)
        entropy = torch.stack(entropies).mean()
        old_log_prob = torch.tensor(
            [transitions[index].old_log_prob for index in indices],
            dtype=torch.float32,
            device=self.device,
        )
        advantage = torch.as_tensor(
            advantages[list(indices)], dtype=torch.float32, device=self.device
        )
        target_return = torch.as_tensor(
            returns[list(indices)], dtype=torch.float32, device=self.device
        )
        policy_loss, ratio = clipped_policy_loss(
            new_log_prob, old_log_prob, advantage, self.clip_ratio
        )
        log_ratio = new_log_prob - old_log_prob
        ppo_kl = ((ratio - 1.0) - log_ratio).mean()
        pg_clipfrac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean()
        value_loss = nn.functional.mse_loss(value, target_return)
        loss = (
            policy_loss
            + self.value_coefficient * value_loss
            - self.entropy_coefficient * entropy
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
            "ratio": float(ratio.mean().detach().cpu()),
            "ppo_kl": float(ppo_kl.detach().cpu()),
            "pg_clipfrac": float(pg_clipfrac.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
        }

    def _critic_diagnostics(
        self,
        transitions: Sequence[PPOTransition],
        returns: np.ndarray,
    ) -> dict[str, float]:
        values: list[float] = []
        with torch.no_grad():
            for row in transitions:
                step = self.policy.evaluate_actions(
                    row.observation.to(self.device), row.actions
                )
                values.append(float(step.value.cpu()))
        predicted = np.asarray(values, dtype=np.float32)
        target = np.asarray(returns, dtype=np.float32)
        errors = target - predicted
        target_variance = float(np.var(target))
        explained_variance = (
            1.0 - float(np.var(errors)) / target_variance
            if target_variance > 1e-8
            else 0.0
        )
        return {
            "predicted_value_mean": float(predicted.mean()),
            "target_return_mean": float(target.mean()),
            "absolute_value_error": float(np.mean(np.abs(errors))),
            "explained_variance": explained_variance,
        }

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        config: dict[str, Any],
        epoch: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_steps": self.update_steps,
                "epoch": int(epoch),
                "config": config,
                "extra": extra or {},
                "action_schema_version": EVIDENCE_SCHEMA_VERSION,
                "evidence_order": [kind.value for kind in EVIDENCE_ORDER],
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        restore_random_state: bool = True,
    ) -> dict[str, Any]:
        # Keep RNG tensors on CPU while loading. ``torch.set_rng_state`` only
        # accepts a CPU ByteTensor; mapping the whole checkpoint to CUDA moves
        # this tensor as well and makes GPU resume fail before training starts.
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        _validate_action_schema(payload)
        self.policy.load_state_dict(payload["policy"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.update_steps = int(payload.get("update_steps", 0))
        if restore_random_state:
            random.setstate(payload["python_random_state"])
            np.random.set_state(payload["numpy_random_state"])
            torch.set_rng_state(payload["torch_random_state"].cpu())
            if torch.cuda.is_available() and payload.get("cuda_random_state") is not None:
                for device_index, state in enumerate(
                    payload["cuda_random_state"][: torch.cuda.device_count()]
                ):
                    torch.cuda.set_rng_state(state.cpu(), device=device_index)
        return {
            "epoch": int(payload.get("epoch", 0)),
            "config": payload.get("config", {}),
            "extra": payload.get("extra", {}),
        }


def load_policy_checkpoint(
    policy: EvidenceSelectionPolicy,
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    _validate_action_schema(payload)
    policy.load_state_dict(payload["policy"])
    return {
        "epoch": int(payload.get("epoch", 0)),
        "config": payload.get("config", {}),
        "extra": payload.get("extra", {}),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _observation_to_cpu(observation: PolicyObservation) -> PolicyObservation:
    return PolicyObservation(
        query_embedding=observation.query_embedding.detach().cpu().clone(),
        summary_embeddings=observation.summary_embeddings.detach().cpu().clone(),
        memory_ids=observation.memory_ids,
        evidence_availability_mask=(
            observation.evidence_availability_mask.detach().cpu().clone()
        ),
    )


def _validate_action_schema(payload: dict[str, Any]) -> None:
    version = int(payload.get("action_schema_version", 1))
    order = payload.get("evidence_order")
    expected_order = [kind.value for kind in EVIDENCE_ORDER]
    if version != EVIDENCE_SCHEMA_VERSION or order != expected_order:
        raise ValueError(
            "Checkpoint action schema is incompatible with the five-bit evidence policy: "
            f"version={version}, order={order!r}; expected "
            f"version={EVIDENCE_SCHEMA_VERSION}, order={expected_order!r}"
        )
