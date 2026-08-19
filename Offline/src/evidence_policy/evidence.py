from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from hive_mem.retriever import MemoryHit


MEMGALLERY_VISUAL_CATEGORIES = frozenset({"VS", "VR"})


class EvidenceTextAction(str, Enum):
    SUMMARY = "summary"
    DIALOGUE = "dialogue"


class EvidenceVisualAction(str, Enum):
    IMAGE = "image"
    CAPTION = "caption"


class EvidenceStrategy(str, Enum):
    FULL = "full-evidence"
    SUMMARY = "summary-only"
    RANDOM = "random"
    PPO = "ppo"


@dataclass(frozen=True)
class DialogueEvidence:
    dataset: str
    dialogue_id: str
    user: str
    assistant: str

    def render(self) -> str:
        return f"User: {self.user}\nAssistant: {self.assistant}"


@dataclass(frozen=True)
class MAUEvidenceAction:
    memory_id: str
    text: EvidenceTextAction
    visual: EvidenceVisualAction | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "memory_id": self.memory_id,
            "text": self.text.value,
            "visual": self.visual.value if self.visual is not None else None,
        }


@dataclass(frozen=True)
class PolicyObservation:
    query_embedding: torch.Tensor
    summary_embeddings: torch.Tensor
    memory_ids: tuple[str, ...]
    has_visual_evidence: torch.Tensor
    visual_action_mask: torch.Tensor

    def validate(self) -> None:
        if self.query_embedding.ndim != 1:
            raise ValueError("query_embedding must have shape [embedding_dim]")
        if self.summary_embeddings.ndim != 2:
            raise ValueError("summary_embeddings must have shape [top_k, embedding_dim]")
        top_k, embedding_dim = self.summary_embeddings.shape
        if self.query_embedding.shape[0] != embedding_dim:
            raise ValueError(
                f"Query dim {self.query_embedding.shape[0]} != summary dim {embedding_dim}"
            )
        if len(self.memory_ids) != top_k:
            raise ValueError(f"Expected {top_k} memory ids, got {len(self.memory_ids)}")
        for name, mask in (
            ("has_visual_evidence", self.has_visual_evidence),
            ("visual_action_mask", self.visual_action_mask),
        ):
            if mask.shape != (top_k,):
                raise ValueError(f"{name} must have shape [{top_k}]")
        if torch.any(self.visual_action_mask & ~self.has_visual_evidence):
            raise ValueError("visual_action_mask cannot enable a MAU without visual evidence")

    def to(self, device: torch.device | str) -> "PolicyObservation":
        return PolicyObservation(
            query_embedding=self.query_embedding.to(device),
            summary_embeddings=self.summary_embeddings.to(device),
            memory_ids=self.memory_ids,
            has_visual_evidence=self.has_visual_evidence.to(device),
            visual_action_mask=self.visual_action_mask.to(device),
        )


@dataclass
class PolicyStep:
    actions: tuple[MAUEvidenceAction, ...]
    joint_log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class DialogueStore:
    """Lazy, read-only lookup of original Mem-Gallery dialogue rounds."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._by_dataset: dict[str, dict[str, DialogueEvidence]] = {}

    def get(self, dataset: str, dialogue_id: str) -> DialogueEvidence:
        if dataset not in self._by_dataset:
            self._by_dataset[dataset] = self._load_dataset(dataset)
        try:
            return self._by_dataset[dataset][dialogue_id]
        except KeyError as exc:
            raise KeyError(f"Unknown dialogue {dataset}:{dialogue_id}") from exc

    def _load_dataset(self, dataset: str) -> dict[str, DialogueEvidence]:
        import json

        path = self.data_dir / "dialog" / f"{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Mem-Gallery dialogue file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, DialogueEvidence] = {}
        for session in payload.get("multi_session_dialogues", []) or []:
            for row in session.get("dialogues", []) or []:
                dialogue_id = str(row.get("round", "")).strip()
                if not dialogue_id:
                    continue
                if dialogue_id in result:
                    raise ValueError(f"Duplicate dialogue id in {path}: {dialogue_id}")
                result[dialogue_id] = DialogueEvidence(
                    dataset=dataset,
                    dialogue_id=dialogue_id,
                    user=str(row.get("user", "")),
                    assistant=str(row.get("assistant", "")),
                )
        return result

    def resolve_image_path(self, dataset: str, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        normalized = str(raw_path).replace("\\", "/")
        marker = "/image/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidate = self.data_dir / "image" / relative
        else:
            candidate = self.data_dir / "image" / dataset / path.name
        if not candidate.exists():
            raise FileNotFoundError(f"Cannot map stored image path {raw_path!r} to {candidate}")
        return candidate


class WMADialogueStore(DialogueStore):
    """Lazy round lookup for WorldMemArena's nested sample files."""

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)
        from embedding.chunk_builder import iter_wma_sample_files

        self._paths = {path.stem: path for path in iter_wma_sample_files(self.data_dir)}

    def _load_dataset(self, dataset: str) -> dict[str, DialogueEvidence]:
        import json
        from embedding.chunk_builder import _wma_rounds

        try:
            path = self._paths[dataset]
        except KeyError as exc:
            raise FileNotFoundError(f"Missing WorldMemArena sample: {dataset}") from exc
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, DialogueEvidence] = {}
        for session in payload.get("sessions", []) or []:
            session_id = str(session.get("_v2_session_id") or session.get("session_id") or "")
            for round_number, user, assistant in _wma_rounds(session.get("dialogue", []) or []):
                dialogue_id = f"{session_id}:R{round_number:04d}"
                result[dialogue_id] = DialogueEvidence(
                    dataset=dataset,
                    dialogue_id=dialogue_id,
                    user=str(user.get("content", "") or ""),
                    assistant=str(assistant.get("content", "") or ""),
                )
        return result

    def resolve_image_path(self, dataset: str, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        try:
            sample_dir = self._paths[dataset].parent
        except KeyError as exc:
            raise FileNotFoundError(f"Missing WorldMemArena sample: {dataset}") from exc
        candidate = sample_dir / str(raw_path).replace("\\", "/")
        if not candidate.exists():
            raise FileNotFoundError(f"Cannot map stored image path {raw_path!r} to {candidate}")
        return candidate


class EvidenceChainBuilder:
    """Convert per-MAU actions into the existing VLM ``memory_items`` shape."""

    def __init__(
        self,
        dialogue_store: DialogueStore,
        *,
        visual_categories: set[str] | frozenset[str] | None = None,
    ):
        self.dialogue_store = dialogue_store
        self.visual_categories = {
            str(value).upper()
            for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
        }

    def build(
        self,
        dataset: str,
        query_category: str,
        memory_hits: Sequence[MemoryHit],
        actions: Sequence[MAUEvidenceAction],
    ) -> list[dict[str, Any]]:
        if len(memory_hits) != len(actions):
            raise ValueError("Every retrieved MAU must have exactly one evidence action")
        items: list[dict[str, Any]] = []
        for hit, action in zip(memory_hits, actions):
            if action.memory_id != hit.item.id:
                raise ValueError(
                    f"Action for {action.memory_id} does not match retrieved MAU {hit.item.id}"
                )
            metadata = dict(hit.item.metadata or {})
            text = self._text_evidence(dataset, hit, action.text)
            image = None
            if action.visual is EvidenceVisualAction.CAPTION:
                caption = self._single_value(metadata, "image_captions")
                if not caption:
                    raise ValueError(f"MAU {hit.item.id} has no caption evidence")
                text = f"{text}\nImage caption: {caption}"
            elif action.visual is EvidenceVisualAction.IMAGE:
                if query_category.upper() not in self.visual_categories:
                    allowed = (
                        "VS/VR"
                        if self.visual_categories == set(MEMGALLERY_VISUAL_CATEGORIES)
                        else "/".join(sorted(self.visual_categories))
                    )
                    raise ValueError(
                        f"Memory images are only valid for {allowed} questions; "
                        f"got {query_category or 'unknown'}"
                    )
                raw_path = self._single_value(metadata, "image_paths")
                if not raw_path:
                    raise ValueError(f"MAU {hit.item.id} has no image evidence")
                image = {
                    "path": str(self.dialogue_store.resolve_image_path(dataset, raw_path)),
                    "img_id": self._single_value(metadata, "image_ids"),
                    "caption": "",
                }
            items.append(
                {
                    "text": text,
                    "image": image,
                    "chunk_id": hit.item.id,
                    "score": hit.score,
                    "metadata": metadata,
                }
            )
        return items

    def _text_evidence(
        self,
        dataset: str,
        hit: MemoryHit,
        action: EvidenceTextAction,
    ) -> str:
        if action is EvidenceTextAction.SUMMARY:
            return hit.item.summary
        source_ids = hit.item.metadata.get("source_dialogue_ids", [])
        if not isinstance(source_ids, list) or len(source_ids) != 1:
            raise ValueError(
                f"MAU {hit.item.id} must have exactly one source dialogue, got {source_ids!r}"
            )
        return self.dialogue_store.get(dataset, str(source_ids[0])).render()

    @staticmethod
    def _single_value(metadata: dict[str, Any], key: str) -> str:
        value = metadata.get(key, [])
        if isinstance(value, (list, tuple)):
            return str(value[0]) if value else ""
        return str(value or "")


def make_policy_observation(
    query_embedding: Sequence[float] | np.ndarray,
    memory_hits: Sequence[MemoryHit],
    category: str,
    visual_categories: set[str] | frozenset[str] | None = None,
) -> PolicyObservation:
    if not memory_hits:
        raise ValueError("Policy observation requires at least one retrieved MAU")
    allowed_visual = {
        str(value).upper()
        for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
    }
    has_visual = [
        bool(hit.item.metadata.get("image_paths"))
        and bool(hit.item.metadata.get("image_captions"))
        for hit in memory_hits
    ]
    observation = PolicyObservation(
        query_embedding=torch.as_tensor(np.asarray(query_embedding), dtype=torch.float32),
        summary_embeddings=torch.as_tensor(
            np.stack([hit.item.embedding for hit in memory_hits]), dtype=torch.float32
        ),
        memory_ids=tuple(hit.item.id for hit in memory_hits),
        has_visual_evidence=torch.as_tensor(has_visual, dtype=torch.bool),
        visual_action_mask=torch.as_tensor(
            [available and category.upper() in allowed_visual for available in has_visual],
            dtype=torch.bool,
        ),
    )
    observation.validate()
    return observation


def choose_baseline_actions(
    memory_hits: Sequence[MemoryHit],
    category: str,
    strategy: EvidenceStrategy,
    rng: random.Random | None = None,
    visual_categories: set[str] | frozenset[str] | None = None,
) -> tuple[MAUEvidenceAction, ...]:
    if strategy is EvidenceStrategy.PPO:
        raise ValueError("PPO actions must come from EvidenceSelectionPolicy")
    rng = rng or random.Random()
    allowed_visual = {
        str(value).upper()
        for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
    }
    actions: list[MAUEvidenceAction] = []
    for hit in memory_hits:
        has_visual = bool(hit.item.metadata.get("image_paths")) and bool(
            hit.item.metadata.get("image_captions")
        )
        if strategy is EvidenceStrategy.FULL:
            text = EvidenceTextAction.DIALOGUE
            visual = (
                EvidenceVisualAction.IMAGE
                if has_visual and category.upper() in allowed_visual
                else EvidenceVisualAction.CAPTION if has_visual else None
            )
        elif strategy is EvidenceStrategy.SUMMARY:
            text = EvidenceTextAction.SUMMARY
            visual = EvidenceVisualAction.CAPTION if has_visual else None
        else:
            text = rng.choice(list(EvidenceTextAction))
            if has_visual and category.upper() in allowed_visual:
                visual = rng.choice(list(EvidenceVisualAction))
            else:
                visual = EvidenceVisualAction.CAPTION if has_visual else None
        actions.append(MAUEvidenceAction(hit.item.id, text, visual))
    return tuple(actions)


def action_signature(actions: Sequence[MAUEvidenceAction]) -> str:
    return "|".join(
        f"{action.memory_id}:{action.text.value}:"
        f"{action.visual.value if action.visual is not None else '-'}"
        for action in actions
    )
