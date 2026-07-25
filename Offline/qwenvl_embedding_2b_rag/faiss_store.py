from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .schema import Chunk, SearchHit


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise ValueError("Cannot normalize an all-zero embedding")
    return vec / (norm + 1e-8)


class FaissChunkStore:
    """Single-index FAISS store for Mem-Gallery chunks."""

    def __init__(self, index_dir: str | Path, dim: int = 2048):
        self.index_dir = Path(index_dir)
        self.dim = int(dim)
        self.index_path = self.index_dir / "vectors.index"
        self.id_mapping_path = self.index_dir / "id_mapping.json"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.metadata_path = self.index_dir / "metadata.jsonl"
        self.index = None
        self.id_mapping: list[str] = []
        self.chunks: dict[str, Chunk] = {}

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        import faiss

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks/embeddings mismatch: {len(chunks)} vs {len(embeddings)}")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings shape (*, {self.dim}), got {embeddings.shape}")

        vectors = np.vstack([normalize_vector(v) for v in embeddings]).astype(np.float32)
        index = faiss.IndexFlatIP(self.dim)
        index.add(vectors)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.id_mapping = [chunk.chunk_id for chunk in chunks]
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.index = index
        self._write_json()

    def load(self) -> None:
        import faiss

        self.index = faiss.read_index(str(self.index_path))
        self.id_mapping = json.loads(self.id_mapping_path.read_text(encoding="utf-8"))
        self.chunks = {}
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = Chunk.from_dict(json.loads(line))
                    self.chunks[chunk.chunk_id] = chunk

    def search(
        self,
        embedding: Iterable[float],
        top_k: int = 5,
        *,
        category: str = "",
        query_text: str = "",
        filters: dict[str, object] | None = None,
        candidate_multiplier: int = 80,
    ) -> list[SearchHit]:
        if self.index is None:
            self.load()
        query = normalize_vector(np.asarray(list(embedding), dtype=np.float32))
        if query.shape[0] != self.dim:
            raise ValueError(f"Expected query dim {self.dim}, got {query.shape[0]}")

        candidate_k = min(max(top_k * candidate_multiplier, top_k), len(self.id_mapping))
        scores, indices = self.index.search(query.reshape(1, -1), candidate_k)
        raw_hits: list[SearchHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            chunk_id = self.id_mapping[int(idx)]
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            if filters and not self._matches_filters(chunk, filters):
                continue
            boosted = float(score) + self._metadata_bonus(chunk, category, query_text)
            raw_hits.append(SearchHit(chunk=chunk, score=boosted, rank=rank))
        raw_hits.sort(key=lambda h: h.score, reverse=True)
        return raw_hits[:top_k]

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict[str, object]) -> bool:
        md = chunk.metadata
        for key, expected in filters.items():
            if expected is None or expected == "":
                continue
            actual = md.get(key)
            if isinstance(expected, (set, list, tuple)):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _metadata_bonus(self, chunk: Chunk, category: str, query_text: str) -> float:
        md = chunk.metadata
        category = (category or "").upper()
        q = query_text or ""
        bonus = 0.0
        if category in {"VS", "VR", "TTL"} and md.get("has_image"):
            bonus += 0.08
        image_id = str(md.get("image_id", ""))
        if image_id and image_id in q:
            bonus += 0.25
        session_id = str(md.get("session_id", ""))
        if session_id and session_id in q:
            bonus += 0.04
        return bonus

    def _write_json(self) -> None:
        self.id_mapping_path.write_text(
            json.dumps(self.id_mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.chunks_path.open("w", encoding="utf-8") as cf, self.metadata_path.open(
            "w", encoding="utf-8"
        ) as mf:
            for chunk_id in self.id_mapping:
                chunk = self.chunks[chunk_id]
                cf.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                row = {"chunk_id": chunk.chunk_id, **chunk.metadata}
                mf.write(json.dumps(row, ensure_ascii=False) + "\n")
