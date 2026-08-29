from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Bernoulli

from .evidence import (
    EVIDENCE_ORDER,
    MAUEvidenceAction,
    PolicyObservation,
    PolicyStep,
)


class EvidenceSelectionPolicy(nn.Module):
    """Shared per-MAU MLP with five independent binary evidence heads."""

    def __init__(
        self,
        embedding_dim: int = 2048,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        deterministic_threshold: float = 0.5,
    ):
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0 or hidden_layers <= 0:
            raise ValueError("Policy dimensions and hidden_layers must be positive")
        if not 0.0 <= deterministic_threshold <= 1.0:
            raise ValueError("deterministic_threshold must be in [0, 1]")
        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU()))
            input_dim = hidden_dim
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.deterministic_threshold = float(deterministic_threshold)
        self.encoder = nn.Sequential(*layers)
        self.evidence_head = nn.Linear(hidden_dim, len(EVIDENCE_ORDER))
        self.value_head = nn.Linear(hidden_dim, 1)

    def sample(self, observation: PolicyObservation) -> PolicyStep:
        return self._act(observation, deterministic=False)

    def select_deterministic(self, observation: PolicyObservation) -> PolicyStep:
        return self._act(observation, deterministic=True)

    def evaluate_actions(
        self,
        observation: PolicyObservation,
        actions: Sequence[MAUEvidenceAction],
    ) -> PolicyStep:
        logits, value = self._forward(observation)
        if len(actions) != len(observation.memory_ids):
            raise ValueError("Action count does not match observation Top-K")
        action_rows: list[tuple[bool, ...]] = []
        for index, (memory_id, action) in enumerate(zip(observation.memory_ids, actions)):
            if action.memory_id != memory_id:
                raise ValueError(f"Expected action for {memory_id}, got {action.memory_id}")
            unavailable = torch.as_tensor(
                action.mask,
                dtype=torch.bool,
                device=observation.evidence_availability_mask.device,
            ) & ~observation.evidence_availability_mask[index]
            if bool(unavailable.any()):
                raise ValueError(
                    f"MAU {memory_id} selected unavailable evidence bits: "
                    f"{torch.nonzero(unavailable, as_tuple=False).flatten().tolist()}"
                )
            action_rows.append(action.mask)
        action_tensor = torch.as_tensor(
            action_rows, dtype=logits.dtype, device=logits.device
        )
        distribution = Bernoulli(logits=logits)
        availability = observation.evidence_availability_mask.to(logits.dtype)
        joint_log_prob = (distribution.log_prob(action_tensor) * availability).sum()
        entropy = (distribution.entropy() * availability).sum()
        return PolicyStep(tuple(actions), joint_log_prob, entropy, value)

    def _act(self, observation: PolicyObservation, *, deterministic: bool) -> PolicyStep:
        logits, _ = self._forward(observation)
        if deterministic:
            threshold_logit = torch.logit(
                torch.tensor(
                    self.deterministic_threshold,
                    dtype=logits.dtype,
                    device=logits.device,
                ),
                eps=torch.finfo(logits.dtype).eps,
            )
            selected = logits >= threshold_logit
        else:
            selected = Bernoulli(logits=logits).sample().to(torch.bool)
        selected &= observation.evidence_availability_mask
        actions = tuple(
            MAUEvidenceAction.from_mask(memory_id, row.tolist())
            for memory_id, row in zip(observation.memory_ids, selected)
        )
        return self.evaluate_actions(observation, actions)

    def _forward(self, observation: PolicyObservation) -> tuple[torch.Tensor, torch.Tensor]:
        observation.validate()
        if observation.query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Policy expects embedding dim {self.embedding_dim}, "
                f"got {observation.query_embedding.shape[0]}"
            )
        query = observation.query_embedding.unsqueeze(0).expand(
            observation.summary_embeddings.shape[0], -1
        )
        hidden = self.encoder(torch.cat((query, observation.summary_embeddings), dim=-1))
        logits = self.evidence_head(hidden)
        value = self.value_head(hidden.mean(dim=0)).squeeze(-1)
        return logits, value
