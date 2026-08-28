from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from embedding.chunk_builder import Chunk


class HiveMemAdapter(BaselineAdapter):
    def __init__(
        self,
        *,
        baseline: str,
        source_root: Path,
        config: dict[str, Any],
    ) -> None:
        del source_root
        self.baseline = baseline
        self.config = dict(config)
        raw_index_root = str(config.get("index_root") or "")
        self.index_root = Path(raw_index_root) if raw_index_root else None
        self.graph_options = config.get("graph_options")
        categories = (
            self.graph_options.get("categories")
            if isinstance(self.graph_options, dict)
            else None
        )
        self.graph_categories = (
            {str(value).strip().upper() for value in categories}
            if categories
            else None
        )
        self.visual_categories = set(config.get("visual_categories") or []) or None
        self.sample_id = ""
        self.directory = Path()
        self.index: Any = None

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del state_dir
        if self.index_root is None:
            raise ValueError("index_root is required for HiveMem")
        self.sample_id = sample_id
        self.directory = self.index_root / "datasets" / sample_id
        if self.graph_options:
            from hive_mem.retriever import GraphExpandedIndex

            options = dict(self.graph_options)
            options.pop("categories", None)
            self.index = GraphExpandedIndex(self.directory, **options)
        else:
            from hive_mem.retriever import SimpleMemoryIndex

            kwargs = {"visual_categories": self.visual_categories} if self.visual_categories else {}
            self.index = SimpleMemoryIndex(self.directory, **kwargs)

    def ingest(self, chunk: Chunk) -> None:
        del chunk

    def end_session(self, session_id: str) -> None:
        del session_id

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if self.index is None:
            raise RuntimeError("HiveMem adapter has not been reset")
        if request.query_vector is None:
            raise ValueError("HiveMem requires a cached query vector")
        allowed = set(request.visible_session_ids) if request.visible_session_ids else None
        graph_gated_off = (
            self.graph_categories is not None
            and request.category.upper() not in self.graph_categories
        )
        if graph_gated_off:
            from hive_mem.retriever import SimpleMemoryIndex

            hits = SimpleMemoryIndex.search(
                self.index,
                request.query_vector,
                request.top_k,
                category=request.category,
                allowed_session_ids=allowed,
            )
        else:
            hits = self.index.search(
                request.query_vector,
                request.top_k,
                category=request.category,
                allowed_session_ids=allowed,
            )
        append_mode = getattr(self.index, "mode", "rerank") == "append"
        items = []
        for hit in hits:
            meta = hit.item.metadata
            text = str(hit.item.content)
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
        return RetrievalResult(items=items, trace={"baseline": self.baseline, "via": "hivemem"})

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
                        text=str(row.get("content") or row.get("summary") or ""),
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
            "baseline": self.baseline,
            "available": True,
            "prebuilt_index": True,
            "supports_session_filter": True,
            "supports_images": True,
        }
