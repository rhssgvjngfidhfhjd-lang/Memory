from __future__ import annotations

from typing import Sequence

import numpy as np


QWEN3_TEXT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_TEXT_EMBEDDING_DIM = 1024
DEFAULT_QUERY_INSTRUCTION = (
    "Given a memory-related question, retrieve relevant memory passages that answer the question"
)


def is_qwen3_text_embedding_model(model_name: str) -> bool:
    return str(model_name).rstrip("/").casefold() == QWEN3_TEXT_EMBEDDING_MODEL.casefold()


class Qwen3TextEmbeddingService:
    """Text-only Qwen3 embedding backend with official last-token pooling."""

    supports_images = False

    def __init__(
        self,
        model_name: str = QWEN3_TEXT_EMBEDDING_MODEL,
        device: str | None = None,
        expected_dim: int = QWEN3_TEXT_EMBEDDING_DIM,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        local_files_only: bool = False,
        max_length: int = 8192,
        batch_size: int = 16,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ):
        if not is_qwen3_text_embedding_model(model_name):
            raise ValueError(f"Unsupported text embedding model: {model_name}")
        self.model_name = model_name
        self.device = device
        self.expected_dim = int(expected_dim)
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.max_length = int(max_length)
        self.batch_size = max(1, int(batch_size))
        self.query_instruction = str(query_instruction).strip()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            dtype=self._resolve_dtype(torch),
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self._model.eval()

    def embed_chunk(self, text: str, images: list[str] | None = None) -> list[float]:
        self._reject_images(images)
        return self.embed_chunks([text])[0].tolist()

    def embed_query(self, query: str, images: list[str] | None = None) -> list[float]:
        self._reject_images(images)
        return self.embed_queries([query])[0].tolist()

    def embed_chunks(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed_texts([str(text) for text in texts])

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        values = [self._format_query(str(query)) for query in queries]
        return self._embed_texts(values)

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.expected_dim), dtype=np.float32)
        self.load()
        import torch
        import torch.nn.functional as functional

        batches = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs, return_dict=True)
                vectors = self._last_token_pool(
                    outputs.last_hidden_state,
                    inputs["attention_mask"],
                )
                vectors = functional.normalize(vectors.float(), p=2, dim=1)
            batches.append(vectors.detach().cpu().float().numpy())
        result = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
        if result.shape != (len(texts), self.expected_dim):
            raise ValueError(
                f"Embedding shape mismatch for {self.model_name}: "
                f"expected ({len(texts)}, {self.expected_dim}), got {result.shape}"
            )
        return result

    def _format_query(self, query: str) -> str:
        if not self.query_instruction:
            return query
        return f"Instruct: {self.query_instruction}\nQuery: {query}"

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        import torch

        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if bool(left_padding):
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _resolve_dtype(self, torch):
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.dtype, "auto")

    def _reject_images(self, images: Sequence[str] | None) -> None:
        if images:
            raise ValueError(
                f"{self.model_name} is text-only; pass an image caption as text instead"
            )


class Qwen3TextMemoryEmbedder:
    """Expose the memory-pipeline API for Qwen3 text embeddings."""

    supports_images = False

    def __init__(
        self,
        model_name: str = QWEN3_TEXT_EMBEDDING_MODEL,
        device: str = "cuda:0",
        expected_dim: int = QWEN3_TEXT_EMBEDDING_DIM,
        dtype: str = "auto",
        local_files_only: bool = False,
        batch_size: int = 16,
    ):
        self.expected_dim = int(expected_dim)
        self.service = Qwen3TextEmbeddingService(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
            batch_size=batch_size,
        )

    def embed_texts(self, texts: str | Sequence[str], mode: str = "context") -> np.ndarray:
        single = isinstance(texts, str)
        values = [str(texts)] if single else [str(text) for text in texts]
        if mode == "query":
            result = self.service.embed_queries(values)
        else:
            result = self.service.embed_chunks(values)
        return result[0] if single else result

    def embed_images(self, image_paths: Sequence[str]) -> np.ndarray:
        raise ValueError(
            f"{self.service.model_name} is text-only and cannot create image vectors"
        )


def create_embedding_service(
    *,
    model_name: str,
    device: str | None,
    expected_dim: int,
    dtype: str = "auto",
    local_files_only: bool = False,
    batch_size: int = 16,
):
    if is_qwen3_text_embedding_model(model_name):
        return Qwen3TextEmbeddingService(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
            batch_size=batch_size,
        )
    from .qwen3vl_embedding import Qwen3VLEmbeddingService

    return Qwen3VLEmbeddingService(
        model_name=model_name,
        device=device,
        expected_dim=expected_dim,
        dtype=dtype,
        local_files_only=local_files_only,
    )


def create_memory_embedder(
    *,
    model_name: str,
    device: str,
    expected_dim: int,
    dtype: str = "auto",
    local_files_only: bool = False,
):
    if is_qwen3_text_embedding_model(model_name):
        return Qwen3TextMemoryEmbedder(
            model_name=model_name,
            device=device,
            expected_dim=expected_dim,
            dtype=dtype,
            local_files_only=local_files_only,
        )
    from .qwen3vl_embedding import QwenMemoryEmbedder

    return QwenMemoryEmbedder(
        model_name=model_name,
        device=device,
        expected_dim=expected_dim,
        dtype=dtype,
        local_files_only=local_files_only,
    )
