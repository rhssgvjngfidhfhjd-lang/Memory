from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np


@dataclass
class MemoryItem:
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "metadata": self.metadata,
        }


class MemoryBank:
    def __init__(self):
        self.memories: List[MemoryItem] = []
        self.timestep = 0

    def add_memory(
        self,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        memory_id: Optional[str] = None,
    ):
        self.memories.append(
            MemoryItem(
                memory_id=memory_id or f"mem_{uuid4().hex}",
                content=str(content).strip(),
                embedding=np.asarray(embedding, dtype=np.float32),
                metadata=_normalize_metadata(metadata),
            )
        )

    def update_memory(
        self,
        index: int,
        content: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not 0 <= index < len(self.memories):
            raise IndexError(f"Memory index out of range: {index}")
        self.memories[index].content = str(content).strip()
        self.memories[index].embedding = np.asarray(embedding, dtype=np.float32)
        self.memories[index].metadata = _merge_metadata(self.memories[index].metadata, metadata)

    def delete_memory(self, index: int):
        if not 0 <= index < len(self.memories):
            raise IndexError(f"Memory index out of range: {index}")
        del self.memories[index]

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[List[str], List[int]]:
        if top_k <= 0 or not self.memories:
            return [], []

        query = _normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        memory_matrix = np.vstack([item.embedding for item in self.memories]).astype(np.float32)
        memory_matrix = _normalize(memory_matrix)
        similarities = np.dot(query, memory_matrix.T)[0]

        actual_k = min(int(top_k), len(self.memories))
        top_indices = np.argsort(similarities)[::-1][:actual_k].tolist()
        return [self.memories[index].content for index in top_indices], top_indices

    def get_contents(self) -> List[str]:
        return [item.content for item in self.memories]

    def get_items(self) -> List[MemoryItem]:
        return list(self.memories)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "memories.jsonl").open("w", encoding="utf-8") as handle:
            for item in self.memories:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        vectors = (
            np.vstack([item.embedding for item in self.memories]).astype(np.float32)
            if self.memories
            else np.zeros((0, 0), dtype=np.float32)
        )
        np.save(directory / "vectors.npy", vectors)
        (directory / "state.json").write_text(
            json.dumps({"timestep": self.timestep, "count": len(self.memories)}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> "MemoryBank":
        directory = Path(directory)
        bank = cls()
        with (directory / "memories.jsonl").open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        vectors = np.load(directory / "vectors.npy")
        if len(rows) != len(vectors):
            raise ValueError(f"Memory/vector count mismatch: {len(rows)} vs {len(vectors)}")
        for row, vector in zip(rows, vectors):
            bank.add_memory(
                row["content"],
                vector,
                metadata=row.get("metadata"),
                memory_id=row.get("memory_id"),
            )
        state_path = directory / "state.json"
        if state_path.exists():
            bank.timestep = int(json.loads(state_path.read_text(encoding="utf-8")).get("timestep", 0))
        return bank

    def step(self):
        self.timestep += 1

    def __len__(self):
        return len(self.memories)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-8)


_LIST_METADATA_KEYS = {
    "source_dialogue_ids",
    "source_chunk_ids",
    "image_ids",
    "image_paths",
    "image_captions",
}


def _normalize_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(metadata or {})
    for key in _LIST_METADATA_KEYS:
        value = normalized.get(key, [])
        if not isinstance(value, (list, tuple, set)):
            value = [value] if value else []
        normalized[key] = list(dict.fromkeys(str(item) for item in value if str(item)))
    return normalized


def _merge_metadata(existing: Dict[str, Any], incoming: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = _normalize_metadata(existing)
    incoming = _normalize_metadata(incoming)
    for key, value in incoming.items():
        if key in _LIST_METADATA_KEYS:
            merged[key] = list(dict.fromkeys(merged.get(key, []) + value))
        elif value not in (None, "", []):
            merged[key] = value
    return merged
