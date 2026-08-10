from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from .output_layout import DatasetLayout


class ModalityType(str, Enum):
    """text-only vs image-bearing memory. Any other stored value is a data
    bug and must fail loudly (ModalityType(...) raises ValueError)."""

    TEXT = "text"
    MULTIMODAL = "multimodal"


@dataclass
class MAU:
    """A compact MAU-shaped memory with legacy AgentMem aliases.

    ``summary``/``id`` are the canonical fields.  ``content``/``memory_id``
    remain as properties so the executor and retrieval adapters keep working.
    """

    summary: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    # LLM-extracted, normalized entity annotations of ``summary``:
    # {"name", "type", "aliases"?: [...], "attribute"?, "value"?}.
    # Owned by this MAU (re-extracted when the summary changes); entity
    # commonality edges are derived from this field at load time and are
    # deliberately not materialized.
    entities: List[Dict[str, Any]] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"mau_{int(time.time() * 1000)}_{uuid4().hex[:8]}")
    modality_type: ModalityType = ModalityType.TEXT
    # Memory-graph edges. ``prev``/``next`` hold the ids of temporal-chain
    # neighbours (built deterministically from metadata ordering, see
    # build_memory_edges.py). ``related`` holds typed edges such as
    # {"target": <mau_id>, "type": "SUPERSEDES" | "CAUSES" | ...}.
    links: Dict[str, Any] = field(
        default_factory=lambda: {"prev": None, "next": None, "related": []}
    )
    # "ACTIVE" or "ARCHIVED". Nothing produces ARCHIVED since the
    # mutable-memory path was removed (2026-08-07), but retrieval/edge
    # building still honour it as a read-side filter.
    status: str = "ACTIVE"

    @property
    def content(self) -> str:
        """Legacy AgentMem alias for the MAU summary."""
        return self.summary

    @property
    def memory_id(self) -> str:
        """Legacy AgentMem alias for the MAU id."""
        return self.id

#数据准成json格式的字典
    def to_dict(self) -> Dict[str, Any]:
        """Serialize using OmniSimpleMem's MAU field names.

        The embedding is intentionally omitted — vectors live in the
        dataset's ``vectors/`` directory.
        """
        return {
            "id": self.id,
            "modality_type": self.modality_type.value,
            "summary": self.summary,
            "entities": self.entities,
            "status": self.status,
            "metadata": self.metadata,
            "links": self.links,
        }

#设置memories这个列表，加mau到memories
class MAUBank:
    def __init__(self):
        self.memories: List[MAU] = []

    def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        entities: Optional[List[Dict[str, Any]]] = None,
    ):
        normalized_metadata = _normalize_metadata(metadata)
        image_paths = normalized_metadata.get("image_paths", [])
        modality_type = (
            ModalityType.MULTIMODAL if image_paths else ModalityType.TEXT
        )
        self.memories.append(
            MAU(
                summary=str(content).strip(),
                embedding=np.asarray(embedding, dtype=np.float32),
                metadata=normalized_metadata,
                entities=[e for e in (entities or []) if isinstance(e, dict)],#只保留字典类型的实体
                modality_type=modality_type,
            )
        )

    # NOTE (2026-08-07): the mutation/query methods update_memory,
    # supersede_memory (SUPERSEDES/SUPERSEDED_BY edge producer), delete_memory,
    # retrieve, get_contents and get_items were removed as dead code after the
    # insert-only + no-build-time-retrieval redesign. Restore from git if a
    # mutable-memory experiment ever needs them.

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        layout = DatasetLayout(directory)
        directory.mkdir(parents=True, exist_ok=True)
        layout.vectors_dir.mkdir(parents=True, exist_ok=True)
        with (directory / "memories.jsonl").open("w", encoding="utf-8") as handle:
            for item in self.memories:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        vectors = (
            np.vstack([item.embedding for item in self.memories]).astype(np.float32)
            if self.memories
            else np.zeros((0, 0), dtype=np.float32)
        )
        np.save(layout.text_vectors, vectors)

    @classmethod
    def load(cls, directory: str | Path) -> "MAUBank":
        directory = Path(directory)
        layout = DatasetLayout(directory)
        bank = cls()
        with (directory / "memories.jsonl").open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        vectors_path = layout.existing_vector_path("text.npy", "vectors.npy")
        if not vectors_path.exists():
            # Fail loudly: silently deriving vectors from rows would poison
            # retrieval with empty embeddings (rows carry no vectors by design).
            raise FileNotFoundError(f"Missing {vectors_path} next to memories.jsonl")
        vectors = np.load(vectors_path)
        if len(rows) != len(vectors):
            raise ValueError(f"Memory/vector count mismatch: {len(rows)} vs {len(vectors)}")
        for row, vector in zip(rows, vectors):
            # Accept both historical AgentMem rows and new MAU-shaped rows.
            summary = row.get("summary", row.get("content", ""))
            memory_id = row.get("id", row.get("memory_id"))
            metadata = _normalize_metadata(row.get("metadata"))
            item = MAU(
                id=memory_id or f"mau_{int(time.time() * 1000)}_{uuid4().hex[:8]}",
                modality_type=ModalityType(
                    row.get(
                        "modality_type",
                        "multimodal" if metadata.get("image_paths") else "text",
                    )
                ),
                summary=str(summary).strip(),
                embedding=np.asarray(vector, dtype=np.float32),
                entities=[e for e in (row.get("entities") or []) if isinstance(e, dict)],
                metadata=metadata,
                links=_normalize_links(row.get("links")),
                status=row.get("status", "ACTIVE"),
            )
            bank.memories.append(item)
        return bank

    def __len__(self):
        return len(self.memories)


_LIST_METADATA_KEYS = {
    "source_dialogue_ids",
    "source_chunk_ids",
    "image_ids",
    "image_paths",
    "image_captions",
}


def _normalize_links(links: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce stored links (including the legacy all-null schema) to the
    current {"prev", "next", "related"} shape."""
    links = dict(links or {})
    related = links.get("related")
    if not isinstance(related, list):
        related = []
    return {
        "prev": links.get("prev"),
        "next": links.get("next"),
        "related": related,
    }


def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(metadata or {})
    for key in _LIST_METADATA_KEYS:
        value = normalized.get(key, [])
        if not isinstance(value, (list, tuple, set)):
            value = [value] if value else []
        normalized[key] = list(dict.fromkeys(str(item) for item in value if str(item)))
    return normalized
