from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from .evidence import (
    EvidenceTextAction,
    EvidenceVisualAction,
    MAUEvidenceAction,
    PolicyObservation,
    PolicyStep,
)


class EvidenceSelectionPolicy(nn.Module):
    """Shared per-MAU MLP with text, visual, and episode-value heads."""

    def __init__(
        self,
        embedding_dim: int = 2048,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ):
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0 or hidden_layers <= 0:
            raise ValueError("Policy dimensions and hidden_layers must be positive")
        layers: list[nn.Module] = []
        input_dim = embedding_dim * 2
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU()))
            input_dim = hidden_dim
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.encoder = nn.Sequential(*layers)
        self.text_head = nn.Linear(hidden_dim, len(EvidenceTextAction))
        self.visual_head = nn.Linear(hidden_dim, len(EvidenceVisualAction))
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
        hidden, text_dist, visual_dist, value = self._forward_distributions(observation)
        del hidden
        if len(actions) != len(observation.memory_ids):
            raise ValueError("Action count does not match observation Top-K")
        text_indices: list[int] = []
        visual_indices: list[int] = []
        normalized: list[MAUEvidenceAction] = []
        for index, (memory_id, action) in enumerate(zip(observation.memory_ids, actions)):
            if action.memory_id != memory_id:
                raise ValueError(f"Expected action for {memory_id}, got {action.memory_id}")
            text_indices.append(list(EvidenceTextAction).index(action.text))
            applies = bool(observation.visual_action_mask[index].item())
            has_visual = bool(observation.has_visual_evidence[index].item())
            if applies:
                if action.visual is None:
                    raise ValueError(f"MAU {memory_id} requires a visual action")
                visual_indices.append(list(EvidenceVisualAction).index(action.visual))
            elif has_visual and action.visual is not EvidenceVisualAction.CAPTION:
                raise ValueError(f"MAU {memory_id} must use fixed caption evidence")
            elif not has_visual and action.visual is not None:
                raise ValueError(f"MAU {memory_id} has no visual evidence")
            normalized.append(action)

        device = observation.summary_embeddings.device
        text_action = torch.tensor(text_indices, dtype=torch.long, device=device)
        joint_log_prob = text_dist.log_prob(text_action).sum()
        entropy = text_dist.entropy().sum()
        mask = observation.visual_action_mask
        if bool(mask.any()):
            assert visual_dist is not None
            visual_action = torch.tensor(visual_indices, dtype=torch.long, device=device)
            joint_log_prob = joint_log_prob + visual_dist.log_prob(visual_action).sum()
            entropy = entropy + visual_dist.entropy().sum()
        return PolicyStep(tuple(normalized), joint_log_prob, entropy, value)

    def _act(self, observation: PolicyObservation, *, deterministic: bool) -> PolicyStep:
        _, text_dist, visual_dist, value = self._forward_distributions(observation)
        text_indices = text_dist.logits.argmax(dim=-1) if deterministic else text_dist.sample()
        mask = observation.visual_action_mask
        visual_indices = (
            visual_dist.logits.argmax(dim=-1) if deterministic else visual_dist.sample()
        ) if visual_dist is not None else torch.empty(
            0, dtype=torch.long, device=observation.summary_embeddings.device
        )
        actions: list[MAUEvidenceAction] = []
        visual_index = 0
        for index, memory_id in enumerate(observation.memory_ids):
            text = list(EvidenceTextAction)[int(text_indices[index].item())]
            if bool(mask[index].item()):
                visual = list(EvidenceVisualAction)[int(visual_indices[visual_index].item())]
                visual_index += 1
            elif bool(observation.has_visual_evidence[index].item()):
                visual = EvidenceVisualAction.CAPTION
            else:
                visual = None
            actions.append(MAUEvidenceAction(memory_id, text, visual))
        return self.evaluate_actions(observation, actions)

    def _forward_distributions(
        self,
        observation: PolicyObservation,
    ) -> tuple[torch.Tensor, Categorical, Categorical | None, torch.Tensor]:
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
        text_dist = Categorical(logits=self.text_head(hidden))
        visual_dist = (
            Categorical(logits=self.visual_head(hidden[observation.visual_action_mask]))
            if bool(observation.visual_action_mask.any())
            else None
        )
        value = self.value_head(hidden.mean(dim=0)).squeeze(-1)
        return hidden, text_dist, visual_dist, value
