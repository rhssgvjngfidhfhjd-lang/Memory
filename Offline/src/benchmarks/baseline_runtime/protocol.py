from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from embedding.chunk_builder import Chunk


@dataclass(frozen=True)
class RetrievalRequest:
    query_id: str
    text: str
    category: str = ""
    top_k: int = 7
    query_image: str | None = None
    visible_session_ids: tuple[str, ...] = ()
    query_vector: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievalRequest":
        data = dict(value)
        data["visible_session_ids"] = tuple(str(x) for x in data.get("visible_session_ids") or ())
        vector = data.get("query_vector")
        data["query_vector"] = [float(x) for x in vector] if vector is not None else None
        return cls(**data)


@dataclass
class RetrievedMemory:
    memory_id: str
    text: str
    score: float | None = None
    session_id: str = ""
    source_dialogue_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievedMemory":
        return cls(**dict(value))

    def to_context_item(self) -> dict[str, Any]:
        image = None
        if self.image_paths:
            image = {
                "path": self.image_paths[0],
                "img_id": self.image_ids[0] if self.image_ids else "",
            }
        metadata = dict(self.metadata)
        metadata.update(
            {
                "session_id": self.session_id,
                "dialogue_id": self.source_dialogue_ids[0] if self.source_dialogue_ids else "",
                "source_dialogue_ids": list(self.source_dialogue_ids),
                "image_ids": list(self.image_ids),
                "image_paths": list(self.image_paths),
            }
        )
        return {"text": self.text, "image": image, "metadata": metadata}

    def to_trace(self, rank: int, *, via: str = "native") -> dict[str, Any]:
        return {
            "rank": rank,
            "memory_id": self.memory_id,
            "score": self.score,
            "via": via,
            "content": self.text,
            "session_id": self.session_id,
            "source_dialogue_ids": list(self.source_dialogue_ids),
            "image_ids": list(self.image_ids),
            "image_paths": list(self.image_paths),
        }


@dataclass
class RetrievalResult:
    items: list[RetrievedMemory] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"items": [item.to_dict() for item in self.items], "trace": dict(self.trace)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RetrievalResult":
        return cls(
            items=[RetrievedMemory.from_dict(row) for row in value.get("items") or []],
            trace=dict(value.get("trace") or {}),
        )


@dataclass
class MemoryRecord:
    memory_id: str
    text: str
    session_id: str = ""
    source_dialogue_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    backend_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRecord":
        return cls(**dict(value))


class BaselineAdapter(ABC):
    @abstractmethod
    def reset(self, sample_id: str, state_dir: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def ingest(self, chunk: Chunk) -> None:
        raise NotImplementedError

    @abstractmethod
    def end_session(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> list[MemoryRecord]:
        raise NotImplementedError

    def capabilities(self) -> dict[str, Any]:
        return {"backend": type(self).__name__, "available": True}

    def close(self) -> None:
        return None


def result_context_items(result: RetrievalResult) -> list[dict[str, Any]]:
    return [item.to_context_item() for item in result.items]


def result_trace_rows(result: RetrievalResult) -> list[dict[str, Any]]:
    default_via = str(result.trace.get("via") or "native")
    return [
        item.to_trace(
            rank,
            via=str(item.metadata.get("via") or default_via),
        )
        for rank, item in enumerate(result.items, start=1)
    ]
