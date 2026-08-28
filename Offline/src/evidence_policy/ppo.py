from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .evidence import MAUEvidenceAction, PolicyObservation
from .policy import EvidenceSelectionPolicy


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
        visual_choice_mask=observation.visual_choice_mask.detach().cpu().clone(),
    )
