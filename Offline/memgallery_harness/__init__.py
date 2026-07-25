"""Mem-Gallery evaluation harness: offline RAG retrieval over prebuilt Qwen3-VL
FAISS chunks, plus shared scoring utilities (VLM answer client, F1/EM metrics,
query-embedding cache) reused by agentmem's Mem-Gallery runs."""

from .core.config import OfflineOmniConfig

__all__ = ["OfflineOmniConfig"]
