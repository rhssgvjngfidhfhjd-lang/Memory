from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class OfflineOmniConfig:
    data_dir: str = "/data/haozhen/Offline/Mem-Gallery/benchmark/data"
    index_dir: str = "artifacts/faiss_index"
    embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    embedding_dim: int = 2048
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    top_k_default: int = 5
    top_k_broad: int = 5
    answer_base_url: str = "http://localhost:8000/v1"
    answer_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    answer_api_key: str = "EMPTY"
    answer_backend: str = "openai"

    @classmethod
    def from_env(cls, **overrides):
        cfg = cls(
            answer_base_url=(
                os.getenv("OPENAI_BASE_URL")
                or os.getenv("ANSWER_BASE_URL")
                or cls.answer_base_url
            ),
            answer_model=(
                os.getenv("OPENAI_MODEL")
                or os.getenv("ANSWER_MODEL")
                or cls.answer_model
            ),
            answer_api_key=(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("ANSWER_API_KEY")
                or cls.answer_api_key
            ),
            answer_backend=os.getenv("ANSWER_BACKEND", cls.answer_backend),
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg
