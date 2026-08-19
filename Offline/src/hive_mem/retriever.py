"""Memory retrieval: pure vector search (SimpleMemoryIndex) and graph-expanded
retrieval (GraphExpandedIndex, rerank/append modes). Merged from index.py +
graph_index.py on 2026-08-06."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hive_mem.build_memory_edges import (
    derive_attribute_pairs,
    derive_entity_pairs,
    load_alias_map,
)
from hive_mem.mau import MAUBank, MAU
from hive_mem.output_layout import DatasetLayout


@dataclass(frozen=True)
class MemoryHit:
    item: MAU
    score: float
    rank: int
    # "vector" for direct similarity hits, "graph" for hits pulled in by
    # graph expansion (GraphExpandedIndex).
    via: str = "vector"

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
    def __init__(
        self,
        directory: str | Path,
        *,
        visual_categories: set[str] | None = None,
    ):
        self.directory = Path(directory)
        self.visual_categories = {
            str(value).upper() for value in (visual_categories or {"VS", "VR", "TTL"})
        }
        layout = DatasetLayout(self.directory)
        self.bank = MAUBank.load(self.directory)
        text_vectors_path = layout.existing_vector_path("text.npy", "vectors.npy")
        self.text_vectors = _normalize_rows(np.load(text_vectors_path))
        image_vectors_path = layout.existing_vector_path("image.npy", "image_vectors.npy")
        image_mask_path = layout.existing_vector_path("image_mask.npy", "image_mask.npy")
        self.image_vectors = _normalize_rows(np.load(image_vectors_path)) if image_vectors_path.exists() else None
        self.image_mask = np.load(image_mask_path) if image_mask_path.exists() else None

    def _scores(
        self,
        query_vector: list[float] | np.ndarray,
        category: str = "",
        allowed_session_ids: set[str] | None = None,
    ) -> np.ndarray:
        """Per-memory similarity scores for a query; ARCHIVED rows are -inf."""
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        query /= np.linalg.norm(query) + 1e-8
        if self.text_vectors.shape[1] != query.shape[0]:
            raise ValueError(f"Query dim {query.shape[0]} != memory dim {self.text_vectors.shape[1]}")
        scores = self.text_vectors @ query
        if category.upper() in self.visual_categories and self.image_vectors is not None:
            image_scores = self.image_vectors @ query
            scores = np.where(self.image_mask, np.maximum(scores, image_scores), scores)
        archived = np.asarray(
            [item.status != "ACTIVE" for item in self.bank.memories], dtype=bool
        )
        disallowed = archived
        if allowed_session_ids is not None:
            allowed = {str(value) for value in allowed_session_ids}
            outside_checkpoint = np.asarray(
                [str(item.metadata.get("session_id", "")) not in allowed for item in self.bank.memories],
                dtype=bool,
            )
            disallowed = disallowed | outside_checkpoint
        return np.where(disallowed, -np.inf, scores)

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 5,
        *,
        category: str = "",
        allowed_session_ids: set[str] | None = None,
    ) -> list[MemoryHit]:
        if not len(self.bank):
            return []
        scores = self._scores(query_vector, category, allowed_session_ids)
        actual_k = min(int(top_k), int(np.isfinite(scores).sum()))
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


class GraphExpandedIndex(SimpleMemoryIndex):
    def __init__(
        self,
        directory: str | Path,
        *,
        seed_k: int = 0,
        expansion_bonus: float = 0.2,
        mode: str = "rerank",
        append_k: int = 2,
        expand_temporal: bool = True,
        expand_related: bool = True,
        expand_entity: bool = True,
        expand_attribute: bool = True,
        related_types: "set[str] | None" = None,
        df_max: float = 0.3,
        df_stop: float = 0.5,
        min_shared: int = 2,
        degree_cap: int = 10,
        visual_categories: set[str] | None = None,
    ):
        super().__init__(directory, visual_categories=visual_categories)
        self.seed_k = int(seed_k)
        self.expansion_bonus = float(expansion_bonus)
        if mode not in ("rerank", "append"):
            raise ValueError(f"Unknown graph retrieval mode: {mode}")
        # "rerank": neighbours compete with vector hits for the top_k slots.
        # "append": the vector top_k is returned untouched and up to append_k
        # graph neighbours are appended after it as extra context.
        self.mode = mode
        self.append_k = int(append_k)
        self.adjacency: dict[int, set[int]] = {}
        index_by_id = {item.id: position for position, item in enumerate(self.bank.memories)}

        def connect(a: int | None, b: int | None) -> None:
            if a is None or b is None or a == b:
                return
            if self.bank.memories[a].status != "ACTIVE" or self.bank.memories[b].status != "ACTIVE":
                return
            self.adjacency.setdefault(a, set()).add(b)
            self.adjacency.setdefault(b, set()).add(a)

        for position, item in enumerate(self.bank.memories):
            if item.status != "ACTIVE":
                continue
            links = item.links or {}
            if expand_temporal:
                connect(position, index_by_id.get(links.get("prev")))
                connect(position, index_by_id.get(links.get("next")))
            if expand_related:
                for edge in links.get("related") or []:
                    if related_types is not None and edge.get("type") not in related_types:
                        continue
                    connect(position, index_by_id.get(edge.get("target")))

        if expand_entity:
            alias_map = load_alias_map(Path(directory))
            for a, b in derive_entity_pairs(
                self.bank,
                alias_map=alias_map,
                df_max=df_max,
                df_stop=df_stop,
                min_shared=min_shared,
                degree_cap=degree_cap,
            ):
                connect(a, b)
        if expand_attribute:
            for a, b in derive_attribute_pairs(
                self.bank,
                df_max=df_max,
                df_stop=df_stop,
                min_shared=min_shared,
                degree_cap=degree_cap,
            ):
                connect(a, b)

    def search(
        self,
        query_vector: list[float] | np.ndarray,
        top_k: int = 5,
        *,
        category: str = "",
        allowed_session_ids: set[str] | None = None,
    ) -> list[MemoryHit]:
        if not len(self.bank):
            return []
        scores = self._scores(query_vector, category, allowed_session_ids)
        active_count = int(np.isfinite(scores).sum())
        seed_count = min(self.seed_k or int(top_k), active_count)
        seed_indices = [int(i) for i in np.argsort(scores)[::-1][:seed_count]]

        if self.mode == "append":
            return self._search_append(scores, seed_indices, int(top_k), active_count)

        final: dict[int, float] = {}
        via: dict[int, str] = {}
        for seed in seed_indices:
            final[seed] = float(scores[seed])
            via[seed] = "vector"
        for seed in seed_indices:
            for neighbour in self.adjacency.get(seed, ()):
                if via.get(neighbour) == "vector":
                    continue
                if not np.isfinite(scores[neighbour]):
                    continue
                candidate = float(scores[neighbour]) + self.expansion_bonus * float(scores[seed])
                if candidate > final.get(neighbour, float("-inf")):
                    final[neighbour] = candidate
                    via[neighbour] = "graph"

        ranked = sorted(final.items(), key=lambda kv: -kv[1])[: min(int(top_k), len(final))]
        return [
            MemoryHit(
                item=self.bank.memories[index],
                score=score,
                rank=rank,
                via=via[index],
            )
            for rank, (index, score) in enumerate(ranked, start=1)
        ]

    def _search_append(self, scores, seed_indices, top_k, active_count):
        """Append mode: vector top_k untouched + up to append_k neighbours."""
        kept = seed_indices[: min(top_k, active_count)]
        kept_set = set(kept)
        hits = [
            MemoryHit(item=self.bank.memories[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(kept, start=1)
        ]
        candidate: dict[int, float] = {}
        for seed in kept:
            for neighbour in self.adjacency.get(seed, ()):
                if neighbour in kept_set or not np.isfinite(scores[neighbour]):
                    continue
                value = float(scores[neighbour]) + self.expansion_bonus * float(scores[seed])
                if value > candidate.get(neighbour, float("-inf")):
                    candidate[neighbour] = value
        extra = sorted(candidate.items(), key=lambda kv: -kv[1])[: self.append_k]
        for offset, (index, score) in enumerate(extra, start=1):
            hits.append(
                MemoryHit(
                    item=self.bank.memories[index],
                    score=score,
                    rank=len(kept) + offset,
                    via="graph",
                )
            )
        return hits
