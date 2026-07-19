from __future__ import annotations

from pathlib import Path
from typing import Any

from .index import SimpleMemoryIndex


class SimpleMemoryMemGalleryAdapter:
    def __init__(self, index_root: str | Path, dataset_name: str, top_k: int = 5):
        self.dataset_name = dataset_name
        self.top_k = int(top_k)
        self.index = SimpleMemoryIndex(Path(index_root) / "datasets" / dataset_name)
        self.last_retrieved_ids: list[str] = []
        self.last_retrieved_source_groups: list[list[str]] = []
        self.last_trace: list[dict[str, Any]] = []

    def reset(self) -> None:
        self.last_retrieved_ids = []
        self.last_retrieved_source_groups = []
        self.last_trace = []

    def recall(self, query: str | dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(query, dict) or query.get("vector") is None:
            raise ValueError("SimpleMemory adapter requires a cached query vector")
        category = str(query.get("category", ""))
        hits = self.index.search(query["vector"], self.top_k, category=category)
        groups = [list(hit.item.metadata.get("source_dialogue_ids", [])) for hit in hits]
        self.last_retrieved_source_groups = groups
        self.last_retrieved_ids = list(dict.fromkeys(source for group in groups for source in group))
        self.last_trace = [
            {
                "rank": hit.rank,
                "memory_id": hit.item.memory_id,
                "score": hit.score,
                "content": hit.item.content,
                "source_dialogue_ids": groups[index],
                "image_ids": hit.item.metadata.get("image_ids", []),
                "image_paths": hit.item.metadata.get("image_paths", []),
            }
            for index, hit in enumerate(hits)
        ]
        return [hit.to_context_item() for hit in hits]

    def store(self, observation: Any) -> None:
        return None
