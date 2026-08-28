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
    def __init__(
        self,
        cache_dir: str | Path,
        expected_dim: int = 2048,
        expected_model: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.expected_dim = int(expected_dim)
        self.expected_model = str(expected_model)
        self.vectors_path = self.cache_dir / "vectors.npy"
        self.metadata_path = self.cache_dir / "metadata.jsonl"
        self.manifest_path = self.cache_dir / "manifest.json"
        self._vectors: np.ndarray | None = None
        self._id_to_index: dict[str, int] = {}

    def load(self) -> None:
        if self._vectors is not None:
            return
        vectors = np.load(self.vectors_path, allow_pickle=False)
        if vectors.ndim != 2 or vectors.shape[1] != self.expected_dim:
            raise ValueError(
                f"Expected query vectors shape (*, {self.expected_dim}), got {vectors.shape}"
            )
        if not np.isfinite(vectors).all():
            raise ValueError(f"Query vectors contain NaN or Inf: {self.vectors_path}")
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest_count = int(manifest.get("count", len(vectors)))
            manifest_dim = int(manifest.get("dim", self.expected_dim))
            if manifest_count != len(vectors) or manifest_dim != self.expected_dim:
                raise ValueError(
                    "Query cache manifest does not match vectors: "
                    f"count={manifest_count}/{len(vectors)}, "
                    f"dim={manifest_dim}/{self.expected_dim}"
                )
            actual_model = str(manifest.get("model_name") or "")
            if self.expected_model and actual_model and actual_model != self.expected_model:
                raise ValueError(
                    f"Query cache model {actual_model!r} != expected {self.expected_model!r}"
                )
        self._vectors = vectors.astype(np.float32, copy=False)
        self._id_to_index = {}
        metadata_count = 0
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                query_id = str(row["query_id"])
                if query_id in self._id_to_index:
                    raise ValueError(
                        f"Duplicate query_id {query_id!r} at {self.metadata_path}:{line_number}"
                    )
                self._id_to_index[query_id] = metadata_count
                metadata_count += 1
        if metadata_count != len(vectors):
            raise ValueError(
                f"Query metadata/vector count mismatch: {metadata_count} vs {len(vectors)}"
            )

    def get_by_id(self, query_id: str) -> list[float] | None:
        self.load()
        idx = self._id_to_index.get(query_id)
        if idx is None or self._vectors is None:
            return None
        return self._vectors[idx].tolist()

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
