"""Retrieval requests, memory records, and the HiVe_mem benchmark adapter."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from src.retriever import DEFAULT_HIVEMEM_GRAPH_OPTIONS


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
        images = [
            {
                "path": path,
                "img_id": self.image_ids[index] if index < len(self.image_ids) else "",
                "kind": "image",
            }
            for index, path in enumerate(self.image_paths)
        ]
        image = (
            {"path": images[0]["path"], "img_id": images[0]["img_id"]}
            if images
            else None
        )
        metadata = dict(self.metadata)
        metadata.update(
            {
                "session_id": self.session_id,
                "dialogue_id": self.source_dialogue_ids[0] if self.source_dialogue_ids else "",
                "source_dialogue_ids": list(self.source_dialogue_ids),
                "image_id": self.image_ids[0] if self.image_ids else "",
                "image_ids": list(self.image_ids),
                "image_paths": list(self.image_paths),
            }
        )
        return {
            "text": self.text,
            "image": image,
            "images": images,
            "metadata": metadata,
        }

    def to_trace(self, rank: int, *, via: str = "hivemem") -> dict[str, Any]:
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


def result_context_items(result: RetrievalResult) -> list[dict[str, Any]]:
    return [item.to_context_item() for item in result.items]


def result_trace_rows(result: RetrievalResult) -> list[dict[str, Any]]:
    default_via = str(result.trace.get("via") or "hivemem")
    return [
        item.to_trace(
            rank,
            via=str(item.metadata.get("via") or default_via),
        )
        for rank, item in enumerate(result.items, start=1)
    ]


class HiVeMemAdapter:
    def __init__(
        self,
        *,
        config: dict[str, Any],
    ) -> None:
        self.config = dict(config)
        raw_index_root = str(config.get("index_root") or "")
        self.index_root = Path(raw_index_root) if raw_index_root else None
        raw_graph_options = config.get("graph_options")
        if raw_graph_options is False:
            raise ValueError("HiVe_mem evaluation requires vector retrieval with graph append")
        self.graph_options = {**DEFAULT_HIVEMEM_GRAPH_OPTIONS, **dict(raw_graph_options or {})}
        self.visual_categories = set(config.get("visual_categories") or []) or None
        self.sample_id = ""
        self.directory = Path()
        self.index: Any = None

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del state_dir
        if self.index_root is None:
            raise ValueError("index_root is required for HiVe_mem")
        self.sample_id = sample_id
        self.directory = self.index_root / "datasets" / sample_id
        from src.retriever import HorizontalMemoryExpansionIndex
        options = dict(self.graph_options)
        if self.visual_categories and "visual_categories" not in options:
            options["visual_categories"] = self.visual_categories
        self.index = HorizontalMemoryExpansionIndex(self.directory, **options)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if self.index is None:
            raise RuntimeError("HiVe_mem adapter has not been reset")
        if request.query_vector is None:
            raise ValueError("HiVe_mem requires a cached query vector")
        allowed = set(request.visible_session_ids) if request.visible_session_ids else None
        vector_k = int(request.top_k)
        hits = self.index.search(request.query_vector, vector_k, category=request.category,
                                 allowed_session_ids=allowed)
        append_mode = getattr(self.index, "mode", "append") == "append"
        items = []
        for hit in hits:
            meta = hit.item.metadata
            text = str(hit.item.evidence_text)
            if append_mode and hit.via == "graph":
                text = f"(related background memory) {text}"
            items.append(
                RetrievedMemory(
                    memory_id=str(hit.item.id),
                    text=text,
                    score=float(hit.score),
                    session_id=str(meta.get("session_id") or ""),
                    source_dialogue_ids=[str(x) for x in meta.get("source_dialogue_ids") or []],
                    image_ids=[str(x) for x in meta.get("image_ids") or []],
                    image_paths=[str(x) for x in meta.get("image_paths") or []],
                    metadata={**meta, "via": hit.via},
                )
            )
        return RetrievalResult(
            items=items,
            trace={
                "method": "HiVe_mem",
                "via": "hivemem",
                "mode": getattr(self.index, "mode", "append"),
                "vector_k": vector_k,
                "graph_append_k": int(getattr(self.index, "append_k", 0)),
                "vector_count": sum(hit.via == "vector" for hit in hits),
                "graph_count": sum(hit.via == "graph" for hit in hits),
            },
        )

    def snapshot(self) -> list[MemoryRecord]:
        path = self.directory / "memories.jsonl"
        if not path.exists():
            return []
        records = []
        with path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = dict(row.get("metadata") or {})
                records.append(
                    MemoryRecord(
                        memory_id=str(row.get("memory_id") or row.get("id") or ""),
                        text=str(
                            row.get("chunk")
                            or row.get("raw_chunk")
                            or row.get("content")
                            or row.get("summary")
                            or ""
                        ),
                        session_id=str(meta.get("session_id") or ""),
                        source_dialogue_ids=[str(x) for x in meta.get("source_dialogue_ids") or []],
                        image_ids=[str(x) for x in meta.get("image_ids") or []],
                        image_paths=[str(x) for x in meta.get("image_paths") or []],
                        backend_type="hivemem",
                        metadata=meta,
                    )
                )
        return records

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "hivemem",
            "method": "HiVe_mem",
            "available": True,
            "prebuilt_index": True,
            "supports_session_filter": True,
            "supports_images": True,
        }

    def close(self) -> None:
        self.index = None
