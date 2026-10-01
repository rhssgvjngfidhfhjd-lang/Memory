from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
import re
from typing import Any
import urllib.request

import numpy as np

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from embedding.chunk_builder import Chunk
from embedding.openai_memory_embedder import OpenAIMemoryEmbedder


SUPPORTED_BASELINES = frozenset({"NaiveRAG", "MuRAG", "UniversalRAG"})


class RAGFamilyAdapter(BaselineAdapter):
    """Fixed-chunk adapters for the three retrieval-only RAG baselines.

    Final QA is intentionally not implemented here.  The benchmark harness owns
    the shared QA prompt and answer call, so every baseline sees byte-identical
    benchmark-specific answer messages after retrieval.
    """

    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        if baseline not in SUPPORTED_BASELINES:
            raise ValueError(f"unsupported RAG-family baseline: {baseline}")
        self.baseline = baseline
        self.source_root = Path(source_root)
        self.config = dict(config)
        self.top_k = int(config.get("top_k") or 7)
        self.dim = int(config["embedding_dim"])
        self.embedder = OpenAIMemoryEmbedder(
            base_url=str(config["embedding_base_url"]),
            model_name=str(config["embedding_model"]),
            expected_dim=self.dim,
            api_key=(
                os.getenv(str(config.get("embedding_api_key_env") or "")) or "EMPTY"
            ),
            timeout=float(config.get("request_timeout") or 180),
        )
        self.sample_id = ""
        self.state_dir = Path()
        self._chunks: list[Chunk] = []
        self._text_vectors: list[np.ndarray] = []
        self._image_vectors: list[np.ndarray | None] = []
        self._pending: list[Chunk] = []

    def reset(self, sample_id: str, state_dir: Path) -> None:
        self.sample_id = str(sample_id)
        self.state_dir = Path(state_dir)
        self._chunks = []
        self._text_vectors = []
        self._image_vectors = []
        self._pending = []

    def ingest(self, chunk: Chunk) -> None:
        # The harness supplies the immutable shared chunks.  Never re-chunk here.
        self._pending.append(chunk)

    def end_session(self, session_id: str) -> None:
        del session_id
        self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        if self.baseline == "NaiveRAG":
            vectors = self.embedder.embed_texts(
                [chunk.text for chunk in pending], mode="context"
            )
            for chunk, vector in zip(pending, vectors, strict=True):
                self._append(chunk, vector, None)
            return
        for chunk in pending:
            if self.baseline == "MuRAG":
                vector = self.embedder.embed_multimodal(
                    chunk.text, chunk.images, mode="context"
                )
                self._append(chunk, vector, vector if chunk.images else None)
                continue
            text_vector = self.embedder.embed_texts(chunk.text, mode="context")
            image_vector = (
                self.embedder.embed_multimodal(
                    chunk.text, chunk.images, mode="context"
                )
                if chunk.images
                else None
            )
            self._append(chunk, text_vector, image_vector)

    def _append(
        self,
        chunk: Chunk,
        text_vector: np.ndarray | list[float],
        image_vector: np.ndarray | list[float] | None,
    ) -> None:
        self._chunks.append(chunk)
        self._text_vectors.append(self._normalize(text_vector))
        self._image_vectors.append(
            self._normalize(image_vector) if image_vector is not None else None
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self._flush_pending()
        requested = min(int(request.top_k or self.top_k), self.top_k)
        route = "document"
        if self.baseline == "UniversalRAG":
            route = self._route(request)
            if route == "no":
                return RetrievalResult(
                    items=[],
                    trace={
                        "baseline": self.baseline,
                        "via": "universal_routing",
                        "route": route,
                        "requested_top_k": requested,
                    },
                )

        if self.baseline == "NaiveRAG":
            query_vector = self.embedder.embed_texts(request.text, mode="query")
            vectors = self._text_vectors
            indices = list(range(len(self._chunks)))
            via = "text_cosine"
        elif self.baseline == "MuRAG":
            query_vector = self.embedder.embed_multimodal(
                request.text,
                [request.query_image] if request.query_image else [],
                mode="query",
            )
            vectors = self._text_vectors
            indices = list(range(len(self._chunks)))
            via = "multimodal_cosine"
        elif route == "image":
            query_vector = self.embedder.embed_multimodal(
                request.text,
                [request.query_image] if request.query_image else [],
                mode="query",
            )
            indices = [
                index
                for index, vector in enumerate(self._image_vectors)
                if vector is not None
            ]
            vectors = [self._image_vectors[index] for index in indices]
            via = "universal_image_cosine"
        else:
            query_vector = self.embedder.embed_texts(request.text, mode="query")
            vectors = self._text_vectors
            indices = list(range(len(self._chunks)))
            via = "universal_document_cosine"

        visible = {str(value) for value in request.visible_session_ids if value}
        if visible:
            selected = [
                (index, vector)
                for index, vector in zip(indices, vectors, strict=True)
                if self._session_id(self._chunks[index]) in visible
            ]
            indices = [index for index, _vector in selected]
            vectors = [vector for _index, vector in selected]

        ranked = self._rank(query_vector, indices, vectors, requested)
        items = [self._retrieved(index, score, via=via) for index, score in ranked]
        return RetrievalResult(
            items=items,
            trace={
                "baseline": self.baseline,
                "via": via,
                "route": route,
                "requested_top_k": requested,
                "candidate_count": len(indices),
                "visible_session_filter": sorted(visible),
            },
        )

    def _rank(
        self,
        query_vector: np.ndarray | list[float],
        indices: list[int],
        vectors: list[np.ndarray | None],
        top_k: int,
    ) -> list[tuple[int, float]]:
        if not indices:
            return []
        query = self._normalize(query_vector)
        matrix = np.stack([self._normalize(vector) for vector in vectors])
        scores = matrix @ query
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(indices[int(offset)], float(scores[int(offset)])) for offset in order]

    def _retrieved(self, index: int, score: float, *, via: str) -> RetrievedMemory:
        chunk = self._chunks[index]
        metadata = dict(chunk.metadata)
        source_ids = [
            str(value)
            for value in metadata.get("source_dialogue_ids") or []
            if value
        ]
        dialogue_id = str(metadata.get("dialogue_id") or chunk.chunk_id)
        if dialogue_id and dialogue_id not in source_ids:
            source_ids.append(dialogue_id)
        image_ids = [
            str(value) for value in metadata.get("image_ids") or [] if value
        ]
        if not image_ids and metadata.get("image_id"):
            image_ids = [str(metadata["image_id"])]
        return RetrievedMemory(
            memory_id=f"{self.baseline}:{self.sample_id}:{chunk.chunk_id}",
            text=chunk.text,
            score=score,
            session_id=self._session_id(chunk),
            source_dialogue_ids=source_ids,
            image_ids=image_ids,
            image_paths=list(chunk.images) if self.baseline != "NaiveRAG" else [],
            metadata={
                "via": via,
                "chunk_id": chunk.chunk_id,
                "has_image": bool(chunk.images),
            },
        )

    def _route(self, request: RetrievalRequest) -> str:
        model = str(
            self.config.get("universalrag_router_model")
            or self.config.get("executor_model")
            or self.config["answer_model"]
        )
        base_url = str(
            self.config.get("universalrag_router_base_url")
            or self.config.get("executor_base_url")
            or self.config["answer_base_url"]
        ).rstrip("/")
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        prompt = (
            "Classify this memory question into exactly one retrieval category: "
            "Document, Image, or No. Document means stored textual conversation "
            "facts are needed. Image means visual appearance, spatial content, or "
            "an image is needed. No means no stored memory is needed. Return only "
            f"the category.\n\nQuery: {request.text}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if request.query_image:
            image_path = Path(request.query_image).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"router query image does not exist: {image_path}")
            mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            image_url = (
                f"data:{mime};base64,"
                + base64.b64encode(image_path.read_bytes()).decode("ascii")
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                }
            )
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 16,
        }
        reasoning_effort = str(self.config.get("reasoning_effort") or "").strip()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        request_http = urllib.request.Request(
            base_url + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request_http, timeout=float(self.config.get("request_timeout") or 180)
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
        choices = result.get("choices") or []
        if not choices:
            raise RuntimeError("UniversalRAG router returned no choices")
        raw = choices[0].get("message", {}).get("content", "")
        if isinstance(raw, list):
            raw = " ".join(
                str(item.get("text") or "") if isinstance(item, dict) else str(item)
                for item in raw
            )
        normalized = str(raw).strip().casefold()
        for category in ("image", "document", "no"):
            if re.search(rf"\b{category}\b", normalized):
                return category
        # The upstream implementation falls back to document for an invalid route.
        return "document"

    def snapshot(self) -> list[MemoryRecord]:
        self._flush_pending()
        return [
            MemoryRecord(
                memory_id=f"{self.baseline}:{self.sample_id}:{chunk.chunk_id}",
                text=chunk.text,
                session_id=self._session_id(chunk),
                source_dialogue_ids=[
                    str(value)
                    for value in chunk.metadata.get("source_dialogue_ids")
                    or [chunk.metadata.get("dialogue_id") or chunk.chunk_id]
                    if value
                ],
                image_ids=[
                    str(value)
                    for value in chunk.metadata.get("image_ids") or []
                    if value
                ],
                image_paths=list(chunk.images),
                backend_type=self.baseline,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "embedding_model": self.config["embedding_model"],
                    "fixed_chunk": True,
                },
            )
            for chunk in self._chunks
        ]

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "rag_family",
            "baseline": self.baseline,
            "available": True,
            "supports_images": self.baseline != "NaiveRAG",
            "supports_session_filter": True,
            "fixed_chunks": True,
            "final_answer_owner": "benchmark_harness",
        }

    @staticmethod
    def _normalize(vector: np.ndarray | list[float] | None) -> np.ndarray:
        if vector is None:
            raise ValueError("cannot normalize an empty embedding")
        value = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("embedding has zero or non-finite norm")
        return value / norm

    @staticmethod
    def _session_id(chunk: Chunk) -> str:
        return str(chunk.metadata.get("session_id") or "")
