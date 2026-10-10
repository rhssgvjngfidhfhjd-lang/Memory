"""Shared benchmark chunking and embedding tools."""

from importlib import import_module

__all__ = [
    "Chunk",
    "build_chunks_from_data",
    "build_chunks_from_file",
    "build_h2h_chunks_from_data",
    "build_h2h_chunks_from_directory",
    "iter_h2h_session_files",
    "write_chunks_jsonl",
]


def __getattr__(name: str):
    if name in __all__:
        value = getattr(import_module(".chunks", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
