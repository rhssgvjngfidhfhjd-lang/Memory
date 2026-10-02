from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import urllib.request

import numpy as np


class OpenAIMemoryEmbedder:
    """Memory-pipeline embedder backed by an OpenAI-compatible endpoint."""

    supports_images = True
    text_batch_size = 128

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        expected_dim: int,
        api_key: str = "EMPTY",
        timeout: float = 180,
    ) -> None:
        self.endpoint = base_url.rstrip("/")
        if not self.endpoint.endswith("/embeddings"):
            self.endpoint += "/embeddings"
        self.model_name = model_name
        self.expected_dim = int(expected_dim)
        self.api_key = api_key
        self.timeout = float(timeout)

    def embed_texts(
        self, texts: str | Sequence[str], mode: str = "context"
    ) -> np.ndarray:
        single = isinstance(texts, str)
        values = [str(texts)] if single else [str(text) for text in texts]
        batches = [
            self._request(
                {"input": values[start : start + self.text_batch_size], "mode": mode}
            )
            for start in range(0, len(values), self.text_batch_size)
        ]
        vectors = (
            np.concatenate(batches, axis=0)
            if batches
            else np.zeros((0, self.expected_dim), dtype=np.float32)
        )
        return vectors[0] if single else vectors

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        vectors = []
        for path in image_paths:
            vectors.append(
                self.embed_multimodal(
                    "Represent this memory image.", [path], mode="context"
                )
            )
        if not vectors:
            return np.zeros((0, self.expected_dim), dtype=np.float32)
        return np.asarray(vectors, dtype=np.float32)

    def embed_multimodal(
        self,
        text: str,
        image_paths: Sequence[str] = (),
        *,
        mode: str = "context",
    ) -> np.ndarray:
        """Embed one text/image item through the shared Qwen3-VL service."""
        content: list[dict[str, Any]] = []
        for raw_path in image_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"embedding image does not exist: {path}")
            content.append(
                {"type": "image_url", "image_url": {"url": str(path)}}
            )
        content.append({"type": "text", "text": str(text or " ")})
        vectors = self._request(
            {
                "mode": mode,
                "messages": [{"role": "user", "content": content}],
            }
        )
        if len(vectors) != 1:
            raise ValueError(
                f"multimodal embedding expected one vector, received {len(vectors)}"
            )
        return vectors[0]

    def _request(self, payload: dict[str, Any]) -> np.ndarray:
        body = json.dumps(
            {"model": self.model_name, **payload}, ensure_ascii=True
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint, data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        rows = sorted(result.get("data") or [], key=lambda row: int(row.get("index", 0)))
        vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.expected_dim:
            raise ValueError(
                f"embedding shape mismatch: expected (*, {self.expected_dim}), got {vectors.shape}"
            )
        return vectors
