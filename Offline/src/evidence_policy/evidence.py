from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from hive_mem.retriever import MemoryHit

from .vp_store import VPArtifactIndex, VPPrimitive


MEMGALLERY_VISUAL_CATEGORIES = frozenset({"VS", "VR", "TTL"})


class EvidenceType(str, Enum):
    SUMMARY = "summary"
    DIALOGUE = "dialogue"
    CAPTION = "caption"
    IMAGE = "image"
    VP = "vp"


EVIDENCE_ORDER = tuple(EvidenceType)
EVIDENCE_SCHEMA_VERSION = 2


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
    selected: frozenset[EvidenceType] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected",
            frozenset(
                value if isinstance(value, EvidenceType) else EvidenceType(value)
                for value in self.selected
            ),
        )

    @classmethod
    def from_mask(
        cls, memory_id: str, mask: Sequence[bool | int]
    ) -> "MAUEvidenceAction":
        if len(mask) != len(EVIDENCE_ORDER):
            raise ValueError(f"Evidence mask must have {len(EVIDENCE_ORDER)} bits")
        return cls(
            memory_id,
            frozenset(kind for kind, enabled in zip(EVIDENCE_ORDER, mask) if enabled),
        )

    @property
    def mask(self) -> tuple[bool, ...]:
        return tuple(kind in self.selected for kind in EVIDENCE_ORDER)

    @property
    def bitmask(self) -> str:
        return "".join("1" if enabled else "0" for enabled in self.mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "mask": self.bitmask,
            "selected": [kind.value for kind in EVIDENCE_ORDER if kind in self.selected],
        }


@dataclass(frozen=True)
class PolicyObservation:
    query_embedding: torch.Tensor
    summary_embeddings: torch.Tensor
    memory_ids: tuple[str, ...]
    # Per-MAU availability in EVIDENCE_ORDER. Unavailable bits are fixed at 0
    # and excluded from policy log-probability and entropy.
    evidence_availability_mask: torch.Tensor

    @property
    def visual_action_mask(self) -> torch.Tensor:
        """Return per-memory availability for the image evidence action.

        This keeps WMA callers compatible with the pre-schema-v2 observation
        API while the policy internally uses the full evidence mask.
        """
        image_index = EVIDENCE_ORDER.index(EvidenceType.IMAGE)
        return self.evidence_availability_mask[:, image_index]

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
        if self.evidence_availability_mask.shape != (top_k, len(EVIDENCE_ORDER)):
            raise ValueError(
                "evidence_availability_mask must have shape "
                f"[{top_k}, {len(EVIDENCE_ORDER)}]"
            )
        if self.evidence_availability_mask.dtype is not torch.bool:
            raise ValueError("evidence_availability_mask must be boolean")

    def to(self, device: torch.device | str) -> "PolicyObservation":
        return PolicyObservation(
            query_embedding=self.query_embedding.to(device),
            summary_embeddings=self.summary_embeddings.to(device),
            memory_ids=self.memory_ids,
            evidence_availability_mask=self.evidence_availability_mask.to(device),
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
        normalized = str(raw_path).replace("\\", "/")
        candidates = []
        if not Path(normalized).is_absolute():
            candidates.append(self.data_dir / normalized)
        candidates.append(sample_dir / normalized)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Cannot map stored image path {raw_path!r}; tried "
            + ", ".join(str(candidate) for candidate in candidates)
        )


class EvidenceChainBuilder:
    """Convert independent per-MAU evidence masks into VLM memory items."""

    def __init__(
        self,
        dialogue_store: DialogueStore,
        *,
        vp_index: VPArtifactIndex | None = None,
        visual_categories: set[str] | frozenset[str] | None = None,
    ):
        self.dialogue_store = dialogue_store
        self.vp_index = vp_index
        self.visual_categories = {
            str(value).upper()
            for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
        }

    def availability(
        self,
        dataset: str,
        query_category: str,
        memory_hits: Sequence[MemoryHit],
    ) -> torch.Tensor:
        visual_allowed = query_category.upper() in self.visual_categories
        rows: list[list[bool]] = []
        for hit in memory_hits:
            metadata = dict(hit.item.metadata or {})
            image_paths = self._values(metadata, "image_paths")
            rows.append(
                [
                    bool(hit.item.summary.strip()),
                    len(self._values(metadata, "source_dialogue_ids")) == 1,
                    bool(self._values(metadata, "image_captions")),
                    bool(image_paths) and visual_allowed,
                    visual_allowed and any(self._vp_primitives(dataset, path) for path in image_paths),
                ]
            )
        return torch.as_tensor(rows, dtype=torch.bool)

    def build(
        self,
        dataset: str,
        query_category: str,
        memory_hits: Sequence[MemoryHit],
        actions: Sequence[MAUEvidenceAction],
    ) -> list[dict[str, Any]]:
        if len(memory_hits) != len(actions):
            raise ValueError("Every retrieved MAU must have exactly one evidence action")
        availability = self.availability(dataset, query_category, memory_hits)
        items: list[dict[str, Any]] = []
        for index, (hit, action) in enumerate(zip(memory_hits, actions)):
            if action.memory_id != hit.item.id:
                raise ValueError(
                    f"Action for {action.memory_id} does not match retrieved MAU {hit.item.id}"
                )
            unavailable = [
                kind.value
                for kind, selected, allowed in zip(
                    EVIDENCE_ORDER, action.mask, availability[index].tolist()
                )
                if selected and not allowed
            ]
            if unavailable:
                raise ValueError(
                    f"MAU {hit.item.id} selected unavailable evidence: {unavailable}"
                )
            if not action.selected:
                continue
            metadata = dict(hit.item.metadata or {})
            text_sections: list[str] = []
            images: list[dict[str, str]] = []
            if EvidenceType.SUMMARY in action.selected:
                text_sections.append(f"Summary:\n{hit.item.summary}")
            if EvidenceType.DIALOGUE in action.selected:
                text_sections.append(f"Dialogue:\n{self._dialogue(dataset, hit).render()}")
            if EvidenceType.CAPTION in action.selected:
                captions = self._values(metadata, "image_captions")
                text_sections.append(
                    "Image captions:\n"
                    + "\n".join(f"- {caption}" for caption in captions)
                )
            image_paths = self._values(metadata, "image_paths")
            image_ids = self._values(metadata, "image_ids")
            if EvidenceType.IMAGE in action.selected:
                resolved_paths = [
                    self._resolve_image_path(dataset, raw_path)
                    for raw_path in image_paths
                ]
                images.extend(
                    {
                        "kind": EvidenceType.IMAGE.value,
                        "path": str(path),
                        "img_id": image_ids[offset] if offset < len(image_ids) else "",
                    }
                    for offset, path in enumerate(resolved_paths)
                )
            if EvidenceType.VP in action.selected:
                for raw_path in image_paths:
                    images.extend(
                        {
                            "kind": EvidenceType.VP.value,
                            "path": str(primitive.crop_path),
                            "img_id": primitive.vp_id,
                        }
                        for primitive in self._vp_primitives(dataset, raw_path)
                    )
            items.append(
                {
                    "text": "\n\n".join(text_sections),
                    "images": images,
                    "chunk_id": hit.item.id,
                    "score": hit.score,
                    "metadata": metadata,
                }
            )
        return items

    def _dialogue(self, dataset: str, hit: MemoryHit) -> DialogueEvidence:
        source_ids = self._values(dict(hit.item.metadata or {}), "source_dialogue_ids")
        if len(source_ids) != 1:
            raise ValueError(
                f"MAU {hit.item.id} must have exactly one source dialogue, got {source_ids!r}"
            )
        return self.dialogue_store.get(dataset, source_ids[0])

    def _vp_primitives(self, dataset: str, raw_path: str) -> tuple[VPPrimitive, ...]:
        if self.vp_index is None:
            return ()
        record = self.vp_index.primitives_for(raw_path)
        if record:
            return record
        try:
            resolved = self.dialogue_store.resolve_image_path(dataset, raw_path)
        except FileNotFoundError:
            return ()
        return self.vp_index.primitives_for(resolved)

    def _resolve_image_path(self, dataset: str, raw_path: str) -> Path:
        try:
            return self.dialogue_store.resolve_image_path(dataset, raw_path)
        except FileNotFoundError as original_error:
            if self.vp_index is not None:
                record = self.vp_index.record_for(raw_path)
                if record is not None:
                    try:
                        return self.dialogue_store.resolve_image_path(
                            dataset, record.relative_path
                        )
                    except FileNotFoundError:
                        pass
            raise original_error

    @staticmethod
    def _values(metadata: dict[str, Any], key: str) -> list[str]:
        value = metadata.get(key, [])
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item)]
        return [str(value)] if value else []


def make_policy_observation(
    query_embedding: Sequence[float] | np.ndarray,
    memory_hits: Sequence[MemoryHit],
    category: str,
    visual_categories: set[str] | frozenset[str] | None = None,
    evidence_availability_mask: Sequence[Sequence[bool]] | torch.Tensor | None = None,
) -> PolicyObservation:
    if not memory_hits:
        raise ValueError("Policy observation requires at least one retrieved MAU")
    allowed_visual = {
        str(value).upper()
        for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
    }
    category_allows_images = category.upper() in allowed_visual
    if evidence_availability_mask is None:
        evidence_availability_mask = [
            [
                bool(hit.item.summary.strip()),
                len(hit.item.metadata.get("source_dialogue_ids", [])) == 1,
                bool(hit.item.metadata.get("image_captions")),
                bool(hit.item.metadata.get("image_paths")) and category_allows_images,
                False,
            ]
            for hit in memory_hits
        ]
    observation = PolicyObservation(
        query_embedding=torch.as_tensor(np.asarray(query_embedding), dtype=torch.float32),
        summary_embeddings=torch.as_tensor(
            np.stack([hit.item.embedding for hit in memory_hits]), dtype=torch.float32
        ),
        memory_ids=tuple(hit.item.id for hit in memory_hits),
        evidence_availability_mask=torch.as_tensor(
            evidence_availability_mask, dtype=torch.bool
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
    evidence_availability_mask: Sequence[Sequence[bool]] | torch.Tensor | None = None,
) -> tuple[MAUEvidenceAction, ...]:
    if strategy is EvidenceStrategy.PPO:
        raise ValueError("PPO actions must come from EvidenceSelectionPolicy")
    rng = rng or random.Random()
    allowed_visual = {
        str(value).upper()
        for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
    }
    if evidence_availability_mask is None:
        evidence_availability_mask = [
            [
                bool(hit.item.summary.strip()),
                len(hit.item.metadata.get("source_dialogue_ids", [])) == 1,
                bool(hit.item.metadata.get("image_captions")),
                bool(hit.item.metadata.get("image_paths"))
                and category.upper() in allowed_visual,
                False,
            ]
            for hit in memory_hits
        ]
    mask_rows = torch.as_tensor(evidence_availability_mask, dtype=torch.bool).tolist()
    actions: list[MAUEvidenceAction] = []
    for hit, available in zip(memory_hits, mask_rows):
        if strategy is EvidenceStrategy.FULL:
            selected_mask = available
        elif strategy is EvidenceStrategy.SUMMARY:
            selected_mask = [available[0], False, False, False, False]
        else:
            selected_mask = [rng.choice((False, True)) if value else False for value in available]
        actions.append(MAUEvidenceAction.from_mask(hit.item.id, selected_mask))
    return tuple(actions)


def action_signature(actions: Sequence[MAUEvidenceAction]) -> str:
    return "|".join(f"{action.memory_id}:{action.bitmask}" for action in actions)
