from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from simple_memory_pipeline.memory_bank import MemoryBank, MemoryItem


@dataclass(frozen=True)
class MemoryHit:
    item: MemoryItem
    score: float
    rank: int

    def to_context_item(self) -> dict[str, Any]:
        metadata = self.item.metadata
        paths = metadata.get("image_paths", [])
        image_ids = metadata.get("image_ids", [])
        captions = metadata.get("image_captions", [])
        image = None
        if paths:
            image = {
                "path": paths[0],
                "img_id": image_ids[0] if image_ids else "",
                "caption": captions[0] if captions else "",
            }
        return {
            "text": self.item.content,
            "image": image,
            "chunk_id": self.item.memory_id,
            "score": self.score,
            "metadata": metadata,
        }


class SimpleMemoryIndex:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.bank = MemoryBank.load(self.directory)
        self.text_vectors = _normalize_rows(np.load(self.directory / "vectors.npy"))
        image_vectors_path = self.directory / "image_vectors.npy"
        image_mask_path = self.directory / "image_mask.npy"
        self.image_vectors = _normalize_rows(np.load(image_vectors_path)) if image_vectors_path.exists() else None
        self.image_mask = np.load(image_mask_path) if image_mask_path.exists() else None

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 5,
        *,
        category: str = "",
    ) -> list[MemoryHit]:
        if not len(self.bank):
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        query /= np.linalg.norm(query) + 1e-8
        if self.text_vectors.shape[1] != query.shape[0]:
            raise ValueError(f"Query dim {query.shape[0]} != memory dim {self.text_vectors.shape[1]}")
        scores = self.text_vectors @ query
        if category.upper() in {"VS", "VR", "TTL"} and self.image_vectors is not None:
            image_scores = self.image_vectors @ query
            scores = np.where(self.image_mask, np.maximum(scores, image_scores), scores)
        actual_k = min(int(top_k), len(scores))
        indices = np.argsort(scores)[::-1][:actual_k]
        return [
            MemoryHit(item=self.bank.memories[int(index)], score=float(scores[index]), rank=rank)
            for rank, index in enumerate(indices, start=1)
        ]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    return matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
