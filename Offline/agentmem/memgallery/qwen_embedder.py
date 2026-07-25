from __future__ import annotations

from typing import Sequence

import numpy as np

from qwenvl_embedding_2b_rag.qwen3vl_embedding import Qwen3VLEmbeddingService


class QwenMemoryEmbedder:
    """Expose the simple pipeline embedder API using Qwen3-VL Embedding."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = "cuda:0",
        expected_dim: int = 2048,
        dtype: str = "auto",
        local_files_only: bool = False,
    ):
        self.expected_dim = int(expected_dim)
        self.service = Qwen3VLEmbeddingService(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    def embed_texts(self, texts: str | Sequence[str], mode: str = "context") -> np.ndarray:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        vectors = []
        for text in values:
            if mode == "query":
                vector = self.service.embed_query(str(text), [])
            else:
                vector = self.service.embed_chunk(str(text), [])
            vectors.append(vector)
        result = np.asarray(vectors, dtype=np.float32)
        if not vectors:
            result = np.zeros((0, self.expected_dim), dtype=np.float32)
        return result[0] if single else result

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        vectors = [self.service.embed_chunk("Represent this memory image.", [path]) for path in image_paths]
        if not vectors:
            return np.zeros((0, self.expected_dim), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)
