from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schema import Chunk


def write_chunks_jsonl(chunks: Iterable[Chunk], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_chunks_jsonl(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
