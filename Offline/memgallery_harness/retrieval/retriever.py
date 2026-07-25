from __future__ import annotations

import re

from memgallery_harness.core.config import OfflineOmniConfig
from qwenvl_embedding_2b_rag.faiss_store import FaissChunkStore as ChunkStore
from qwenvl_embedding_2b_rag.qwen3vl_embedding import Qwen3VLEmbeddingService as Qwen3VLEmbedder
from .formatter import format_hits_as_multimodal_items


class MemGalleryRetriever:
    def __init__(self, config: OfflineOmniConfig, dataset_name: str | None = None):
        self.config = config
        self.dataset_name = dataset_name
        self.store = ChunkStore(config.index_dir, dim=config.embedding_dim)
        self.embedder: Qwen3VLEmbedder | None = None
        self.last_retrieved_ids: list[str] = []

    def retrieve(
        self,
        query: str,
        query_images: list[str] | None = None,
        query_vector: list[float] | None = None,
    ) -> list[dict]:
        category, clean_query = self._parse_category(query)
        top_k = self._top_k(category, clean_query)
        vector = query_vector if query_vector is not None else self._embed_query(clean_query, query_images or [])
        filters = {"dataset": self.dataset_name} if self.dataset_name else None
        hits = self.store.search(
            vector,
            top_k=top_k * 3,
            category=category,
            query_text=clean_query,
            filters=filters,
            candidate_multiplier=120,
        )
        hits = self._dedupe_by_dialogue_id(hits, top_k)
        if category in {"TR", "CD"}:
            hits = sorted(
                hits,
                key=lambda h: (
                    str(h.chunk.metadata.get("session_id", "")),
                    int(h.chunk.metadata.get("round_id") or 0),
                ),
            )
        self.last_retrieved_ids = [
            str(hit.chunk.metadata.get("dialogue_id") or hit.chunk.chunk_id) for hit in hits
        ]
        return format_hits_as_multimodal_items(hits)

    def _embed_query(self, query: str, query_images: list[str]) -> list[float]:
        if self.embedder is None:
            self.embedder = Qwen3VLEmbedder(
                model_name=self.config.embedding_model,
                device=self.config.device,
                expected_dim=self.config.embedding_dim,
                dtype=self.config.dtype,
            )
        return self.embedder.embed_query(query, query_images)

    @staticmethod
    def _dedupe_by_dialogue_id(hits: list, top_k: int) -> list:
        """Collapse hits that share a dialogue_id (e.g. a round's text row and
        its separately-embedded image-companion row) so one round doesn't
        occupy two top-k slots. Hits are assumed sorted by score descending,
        so the first occurrence per dialogue_id is the highest scoring."""
        seen: set[str] = set()
        deduped = []
        for hit in hits:
            dialogue_id = str(hit.chunk.metadata.get("dialogue_id") or hit.chunk.chunk_id)
            if dialogue_id in seen:
                continue
            seen.add(dialogue_id)
            deduped.append(hit)
            if len(deduped) >= top_k:
                break
        return deduped

    @staticmethod
    def _parse_category(query: str) -> tuple[str, str]:
        match = re.match(r"^\[([A-Z]+)\]\s*", query)
        if match:
            return match.group(1), query[match.end():]
        return "", query

    def _top_k(self, category: str, query: str) -> int:
        return self.config.top_k_default
