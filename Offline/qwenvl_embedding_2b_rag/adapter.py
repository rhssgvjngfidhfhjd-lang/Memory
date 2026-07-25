from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .faiss_store import FaissChunkStore
from .qwen3vl_embedding import Qwen3VLEmbeddingService


class QwenChunkMemory:
    """Small recall adapter backed by a prebuilt Qwen3-VL FAISS index.

    This returns the multimodal list format used by Mem-Gallery's VLM runner:
    [{"text": ..., "image": {"path": ..., "img_id": ...}}, ...]
    """

    def __init__(
        self,
        index_dir: str | Path = "artifacts/faiss_index",
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        dim: int = 2048,
        device: str | None = None,
        dtype: str = "auto",
        top_k: int = 5,
        local_files_only: bool = False,
    ):
        self.index_dir = Path(index_dir)
        self.dim = dim
        self.top_k = top_k
        self.store = FaissChunkStore(self.index_dir, dim=dim)
        self.embedder = Qwen3VLEmbeddingService(
            model_name=model_name,
            device=device,
            expected_dim=dim,
            dtype=dtype,
            local_files_only=local_files_only,
        )
        self.last_retrieved_ids: list[str] = []

    def reset(self) -> None:
        self.last_retrieved_ids = []

    def store_observation(self, observation: Any) -> None:
        """No-op for prebuilt offline indexes."""
        return None

    def recall(self, query: str | dict[str, Any]) -> list[dict[str, Any]]:
        if isinstance(query, dict):
            text = str(query.get("text", ""))
            q_images = []
            image = query.get("image")
            if isinstance(image, dict) and image.get("path"):
                q_images.append(str(image["path"]))
        else:
            text = str(query)
            q_images = []
        category, clean = self._category_from_query(text)
        vector = self.embedder.embed_query(clean, q_images)
        top_k = self._top_k_for_category(category, clean)
        hits = self.store.search(vector, top_k=top_k, category=category, query_text=clean)
        self.last_retrieved_ids = [h.chunk.metadata.get("dialogue_id", h.chunk.chunk_id) for h in hits]
        if category in {"TR", "CD"}:
            hits = sorted(
                hits,
                key=lambda h: (
                    str(h.chunk.metadata.get("session_id", "")),
                    int(h.chunk.metadata.get("round_id") or 0),
                ),
            )
        return [hit.to_context_item() for hit in hits]

    @staticmethod
    def _category_from_query(query: str) -> tuple[str, str]:
        match = re.match(r"^\[([A-Z]+)\]\s*", query)
        if not match:
            return "", query
        return match.group(1), query[match.end():]

    def _top_k_for_category(self, category: str, query: str) -> int:
        category = category.upper()
        if category in {"FR", "CD", "KR"}:
            return max(self.top_k, 8)
        if category in {"VS", "VR", "TTL"}:
            return self.top_k
        if "how many" in query.lower() or "list all" in query.lower():
            return max(self.top_k, 8)
        return self.top_k
