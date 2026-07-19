from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def make_query_id(
    *,
    dataset_name: str,
    qa_index: int,
    category: str,
    question: str,
    query_image: dict[str, Any] | None = None,
) -> str:
    image_path = ""
    image_caption = ""
    if query_image:
        image_path = str(query_image.get("path", "") or "")
        image_caption = str(query_image.get("caption", "") or "")
    digest = hashlib.sha1(
        "\n".join([category, question, image_path, image_caption]).encode("utf-8")
    ).hexdigest()[:16]
    return f"{dataset_name}::{qa_index}::{category}::{digest}"


class QueryEmbeddingCache:
    def __init__(self, cache_dir: str | Path, expected_dim: int = 2048):
        self.cache_dir = Path(cache_dir)
        self.expected_dim = int(expected_dim)
        self.vectors_path = self.cache_dir / "vectors.npy"
        self.metadata_path = self.cache_dir / "metadata.jsonl"
        self.manifest_path = self.cache_dir / "manifest.json"
        self._vectors: np.ndarray | None = None
        self._id_to_index: dict[str, int] = {}

    def load(self) -> None:
        if self._vectors is not None:
            return
        vectors = np.load(self.vectors_path)
        if vectors.ndim != 2 or vectors.shape[1] != self.expected_dim:
            raise ValueError(
                f"Expected query vectors shape (*, {self.expected_dim}), got {vectors.shape}"
            )
        self._vectors = vectors.astype(np.float32, copy=False)
        self._id_to_index = {}
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                self._id_to_index[str(row["query_id"])] = idx

    def get_by_id(self, query_id: str) -> list[float] | None:
        self.load()
        idx = self._id_to_index.get(query_id)
        if idx is None or self._vectors is None:
            return None
        return self._vectors[idx].astype(np.float32).tolist()

    def get(
        self,
        *,
        dataset_name: str,
        qa_index: int,
        category: str,
        question: str,
        query_image: dict[str, Any] | None = None,
    ) -> list[float] | None:
        query_id = make_query_id(
            dataset_name=dataset_name,
            qa_index=qa_index,
            category=category,
            question=question,
            query_image=query_image,
        )
        return self.get_by_id(query_id)
