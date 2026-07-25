from __future__ import annotations

from typing import Any

from memgallery_harness.core.config import OfflineOmniConfig
from memgallery_harness.retrieval.retriever import MemGalleryRetriever


class MemGalleryOfflineAdapter:
    """Mem-Gallery explicit-memory adapter backed by prebuilt offline chunks."""

    def __init__(self, config: OfflineOmniConfig, dataset_name: str | None = None):
        self.config = config
        self.dataset_name = dataset_name
        self.retriever = MemGalleryRetriever(config, dataset_name=dataset_name)
        self.last_retrieved_ids: list[str] = []

    def reset(self) -> None:
        self.last_retrieved_ids = []

    def store(self, observation: Any) -> None:
        # Memory is prebuilt in artifacts/faiss_index.
        return None

    def recall(self, query: str | dict[str, Any]) -> list[dict]:
        vector = None
        if isinstance(query, dict):
            text = str(query.get("text", ""))
            raw_vector = query.get("vector")
            if raw_vector is not None:
                vector = [float(x) for x in raw_vector]
            images = []
            image = query.get("image")
            if isinstance(image, dict) and image.get("path"):
                images.append(str(image["path"]))
        else:
            text = str(query)
            images = []
        items = self.retriever.retrieve(text, images, query_vector=vector)
        self.last_retrieved_ids = list(self.retriever.last_retrieved_ids)
        return items

    def display(self) -> None:
        print(f"MemGalleryOfflineAdapter(dataset={self.dataset_name})")

    def manage(self, operation, **kwargs) -> None:
        return None

    def optimize(self, **kwargs) -> None:
        return None
