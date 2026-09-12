#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def request_embeddings(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout: int,
    retries: int,
) -> list[list[float]]:
    body = json.dumps(payload).encode("utf-8")
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer EMPTY",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            return [row["embedding"] for row in result.get("data") or []]
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"embedding request failed: {exc}") from exc
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def scan_input(input_path: Path, metadata_path: Path) -> int:
    temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    count = 0
    with input_path.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            output.write(
                json.dumps(
                    {
                        "index": count,
                        "chunk_id": row.get("chunk_id"),
                        "dataset": metadata.get("dataset"),
                        "session_id": metadata.get("session_id"),
                        "source_dialogue_ids": metadata.get("source_dialogue_ids")
                        or [metadata.get("dialogue_id") or row.get("chunk_id")],
                        "image_count": len(row.get("images") or []),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    temporary.replace(metadata_path)
    return count


def embedding_payload(
    rows: list[dict[str, Any]],
    *,
    model: str,
    include_images: bool,
) -> dict[str, Any]:
    if len(rows) == 1 and include_images and rows[0].get("images"):
        content = [
            {"type": "image", "image": str(path)}
            for path in rows[0].get("images") or []
        ]
        content.append({"type": "text", "text": str(rows[0].get("text") or " ")})
        return {
            "model": model,
            "mode": "context",
            "messages": [{"role": "user", "content": content}],
        }
    return {
        "model": model,
        "mode": "context",
        "input": [str(row.get("text") or " ") for row in rows],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable embeddings for chunk JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--include-images", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    manifest_path = output_dir / "manifest.json"
    vectors_path = output_dir / "vectors.npy"
    input_hash = file_sha256(input_path)
    total = scan_input(input_path, metadata_path)
    signature = {
        "input": str(input_path),
        "input_sha256": input_hash,
        "model": args.model,
        "dim": args.dim,
        "include_images": args.include_images,
        "total": total,
    }

    completed = 0
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatched = [key for key, value in signature.items() if previous.get(key) != value]
        if mismatched:
            raise RuntimeError(f"existing cache signature mismatch: {', '.join(mismatched)}")
        completed = int(previous.get("completed") or 0)
    if completed == total and vectors_path.is_file():
        print(f"already complete: {total}/{total} {output_dir}", flush=True)
        return

    if vectors_path.is_file():
        vectors = np.lib.format.open_memmap(vectors_path, mode="r+")
        if vectors.shape != (total, args.dim):
            raise RuntimeError(f"unexpected vector shape: {vectors.shape}")
    else:
        vectors = np.lib.format.open_memmap(
            vectors_path, mode="w+", dtype=np.float32, shape=(total, args.dim)
        )

    manifest = {**signature, "completed": completed, "status": "running"}
    write_json_atomic(manifest_path, manifest)
    endpoint = args.base_url.rstrip("/") + "/embeddings"
    started = time.time()
    pending: list[tuple[int, dict[str, Any]]] = []

    def commit(batch: list[tuple[int, dict[str, Any]]]) -> None:
        nonlocal completed
        if not batch:
            return
        rows = [row for _, row in batch]
        payload = embedding_payload(
            rows, model=args.model, include_images=args.include_images
        )
        values = request_embeddings(
            endpoint, payload, timeout=args.timeout, retries=args.retries
        )
        if len(values) != len(batch):
            raise RuntimeError(f"expected {len(batch)} vectors, received {len(values)}")
        for (index, _), value in zip(batch, values):
            vector = np.asarray(value, dtype=np.float32)
            if vector.shape != (args.dim,):
                raise RuntimeError(f"vector {index} has shape {vector.shape}")
            vectors[index] = vector
            completed = index + 1
        vectors.flush()
        manifest.update(
            {
                "completed": completed,
                "status": "running",
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
        write_json_atomic(manifest_path, manifest)
        if completed % 100 < len(batch) or completed == total:
            print(f"progress {completed}/{total}", flush=True)

    with input_path.open(encoding="utf-8") as source:
        index = 0
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if index < completed:
                index += 1
                continue
            has_images = bool(args.include_images and row.get("images"))
            if has_images:
                commit(pending)
                pending = []
                commit([(index, row)])
            else:
                pending.append((index, row))
                if len(pending) >= args.batch_size:
                    commit(pending)
                    pending = []
            index += 1
        commit(pending)

    manifest.update(
        {
            "completed": completed,
            "status": "complete",
            "elapsed_seconds": round(time.time() - started, 2),
            "vectors": str(vectors_path),
            "metadata": str(metadata_path),
        }
    )
    write_json_atomic(manifest_path, manifest)
    print(f"complete {completed}/{total}: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
