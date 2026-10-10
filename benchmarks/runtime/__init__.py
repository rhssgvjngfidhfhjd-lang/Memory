"""Shared runtime for HiVe_mem benchmark evaluation."""

from typing import Any

from benchmarks.runtime.adapter import (
    HiVeMemAdapter, MemoryRecord, RetrievalRequest, RetrievalResult, RetrievedMemory,
)


def create_adapter(*, config_overrides: dict[str, Any]) -> HiVeMemAdapter:
    return HiVeMemAdapter(config=config_overrides)


def method_metadata() -> dict[str, Any]:
    return {"method": "HiVe_mem", "prebuilt_index": True, "supports_images": True, "supports_session_filter": True}


def __getattr__(name: str):
    if name == "OutputLayout":
        from benchmarks.runtime.evaluation import OutputLayout
        globals()[name] = OutputLayout
        return OutputLayout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["HiVeMemAdapter", "OutputLayout", "MemoryRecord", "RetrievalRequest", "RetrievalResult", "RetrievedMemory", "create_adapter", "method_metadata"]
