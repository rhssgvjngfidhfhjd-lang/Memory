from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable
from contextlib import contextmanager


def write_text_atomic(path: str | Path, content: str) -> None:
    """Replace a text file atomically without leaving a shared ``.tmp`` file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


@contextmanager
def atomic_binary_writer(path: str | Path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def write_json_atomic(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    write_text_atomic(
        path,
        json.dumps(payload, ensure_ascii=False, indent=indent),
    )


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    write_text_atomic(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash existing files using resolved paths as stable manifest keys."""
    manifest: dict[str, str] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            manifest[str(path)] = sha256_file(path)
    return dict(sorted(manifest.items()))
