"""Read the immutable benchmark chunk JSONL inputs used by every baseline."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

from benchmarks.baseline_runtime.config import OFFLINE_ROOT, load_defaults
from embedding.chunk_builder import Chunk, read_chunks_jsonl


_CONFIG_KEYS = {
    "memgallery": "memgallery_chunks_file",
    "h2hmem_dyadic": "h2hmem_dyadic_chunks_file",
    "h2hmem_multiparty": "h2hmem_multiparty_chunks_file",
    "wma_lifelong": "wma_lifelong_chunks_file",
}


def resolve_chunk_file(config: dict[str, Any], source: str) -> Path:
    try:
        key = _CONFIG_KEYS[source]
    except KeyError as exc:
        raise KeyError(f"unknown fixed chunk source: {source}") from exc
    raw = str(config.get(key) or load_defaults().get(key) or "").strip()
    if not raw:
        raise ValueError(f"missing fixed chunk configuration: {key}")
    path = Path(raw).expanduser()
    path = path.resolve() if path.is_absolute() else (OFFLINE_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fixed chunk file does not exist: {path}")
    return path


@lru_cache(maxsize=8)
def _read_cached(path: str) -> tuple[Chunk, ...]:
    return tuple(read_chunks_jsonl(path))


def _rows(config: dict[str, Any], source: str) -> tuple[Path, tuple[Chunk, ...]]:
    path = resolve_chunk_file(config, source)
    return path, _read_cached(str(path))


def memgallery_chunks(config: dict[str, Any], dataset_name: str) -> list[Chunk]:
    _path, rows = _rows(config, "memgallery")
    selected = [
        row
        for row in rows
        if str(row.metadata.get("dataset") or "") == dataset_name
    ]
    return _require_sample(selected, "Mem-Gallery", dataset_name)


def h2hmem_chunks(
    config: dict[str, Any], *, variant: str, conversation_id: str
) -> list[Chunk]:
    source = f"h2hmem_{variant}"
    _path, rows = _rows(config, source)
    selected = [
        row
        for row in rows
        if str(row.metadata.get("variant") or "") == variant
        and str(row.metadata.get("conversation_id") or "") == conversation_id
    ]
    return _require_sample(selected, f"H2HMem/{variant}", conversation_id)


def wma_chunks(config: dict[str, Any], sample_id: str) -> list[Chunk]:
    _path, rows = _rows(config, "wma_lifelong")
    selected = [
        row
        for row in rows
        if str(row.metadata.get("dataset") or "") == sample_id
    ]
    return _require_sample(selected, "WorldMemArena/lifelong", sample_id)


def chunk_source_manifest(config: dict[str, Any], source: str) -> dict[str, Any]:
    path, rows = _rows(config, source)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "source": source,
        "path": str(path),
        "sha256": digest,
        "chunk_count": len(rows),
        "format": "Chunk JSONL",
    }


def _require_sample(rows: list[Chunk], benchmark: str, sample_id: str) -> list[Chunk]:
    if not rows:
        raise KeyError(f"fixed {benchmark} chunk file has no sample {sample_id!r}")
    ids = [row.chunk_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"fixed {benchmark} chunks contain duplicate IDs for {sample_id}")
    return rows
