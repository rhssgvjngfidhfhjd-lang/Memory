#!/usr/bin/env python3
"""Prepare an isolated, audited MMA memory-bank copy for QA-only replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time
from typing import Any


TABLE_PREFIXES = {
    "episodic_memory": "episodic_memory_manager",
    "semantic_memory": "semantic_memory_manager",
    "procedural_memory": "procedural_memory_manager",
    "resource_memory": "resource_memory_manager",
    "knowledge_vault": "knowledge_vault_manager",
}
TEXT_FIELDS = {
    "episodic_memory": ("summary", "details", "actor", "event_type"),
    "semantic_memory": ("name", "summary", "details", "source"),
    "procedural_memory": ("entry_type", "summary", "steps"),
    "resource_memory": ("title", "summary", "resource_type", "content"),
    "knowledge_vault": ("entry_type", "source", "secret_value", "caption"),
}
PROVENANCE_NAME = ".mma_reuse_provenance.json"
MANIFEST_NAME = ".mma_reuse_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_snapshot(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            memory_id = str(row.get("memory_id") or "")
            if not memory_id:
                raise RuntimeError(f"snapshot line {line_number} has no memory_id")
            rows.setdefault(memory_id, []).append(row)
    return rows


def database_rows(database: Path) -> list[tuple[str, list[str]]]:
    output: list[tuple[str, list[str]]] = []
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        check = connection.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise RuntimeError(f"SQLite quick_check failed for {database}: {check}")
        for table, prefix in TABLE_PREFIXES.items():
            columns = {
                str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            fields = [field for field in TEXT_FIELDS[table] if field in columns]
            selected = ", ".join(['"id"', *[f'"{field}"' for field in fields]])
            deleted_filter = (
                ' WHERE "is_deleted" = 0 OR "is_deleted" IS NULL'
                if "is_deleted" in columns
                else ""
            )
            for row in connection.execute(
                f'SELECT {selected} FROM "{table}"{deleted_filter}'
            ):
                memory_id = f"{prefix}:{row[0]}"
                values = [str(value) for value in row[1:] if value not in (None, "")]
                output.append((memory_id, values))
    finally:
        connection.close()
    return output


def choose_snapshot_row(
    memory_id: str,
    values: list[str],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(candidates) == 1:
        return candidates[0]
    scored = []
    for candidate in candidates:
        text = str(candidate.get("text") or "")
        score = sum(len(value) for value in values if len(value) >= 8 and value in text)
        scored.append((score, candidate))
    scored.sort(key=lambda row: row[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        raise RuntimeError(
            f"cannot disambiguate duplicate snapshot memory_id {memory_id!r}"
        )
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise RuntimeError(
            f"ambiguous duplicate snapshot memory_id {memory_id!r}: score={scored[0][0]}"
        )
    return scored[0][1]


def provenance_for_database(
    database: Path,
    snapshots: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for memory_id, values in database_rows(database):
        candidates = snapshots.get(memory_id) or []
        if not candidates:
            raise RuntimeError(
                f"memory {memory_id!r} in {database} is absent from the snapshot"
            )
        row = choose_snapshot_row(memory_id, values, candidates)
        session_id = str(row.get("session_id") or "")
        provenance[memory_id] = {
            "session_id": session_id,
            "session_ids": [session_id] if session_id else [],
            "source_dialogue_ids": [
                str(value) for value in row.get("source_dialogue_ids") or []
            ],
            "image_ids": [str(value) for value in row.get("image_ids") or []],
            "image_paths": [str(value) for value in row.get("image_paths") or []],
        }
    return provenance


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError(f"destination is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cp", "-a", "--reflink=auto", f"{source}/.", str(destination)],
        check=True,
    )


def prepare(source_memory: Path, destination: Path) -> dict[str, Any]:
    source_memory = source_memory.resolve()
    destination = destination.resolve()
    snapshot = source_memory / "memory_snapshot.jsonl"
    source_datasets = source_memory / "datasets"
    if not snapshot.is_file() or not source_datasets.is_dir():
        raise RuntimeError(
            f"expected memory_snapshot.jsonl and datasets/ under {source_memory}"
        )
    if destination == source_datasets or source_datasets in destination.parents:
        raise RuntimeError("destination must not be inside the source memory bank")

    snapshots = read_snapshot(snapshot)
    source_databases = [
        path
        for path in sorted(source_datasets.rglob("sqlite.db"))
        if ".orphaned_pristine" not in str(path.parent)
    ]
    if not source_databases:
        raise RuntimeError(f"no SQLite memory banks found under {source_datasets}")
    copy_tree(source_datasets, destination)

    samples = []
    for source_database in source_databases:
        relative = source_database.relative_to(source_datasets)
        copied_database = destination / relative
        if sha256_file(source_database) != sha256_file(copied_database):
            raise RuntimeError(f"copied SQLite hash mismatch: {relative}")
        provenance = provenance_for_database(source_database, snapshots)
        state_dir = copied_database.parent
        payload = {
            "version": 1,
            "mode": "qa_only_isolated_memory_reuse",
            "source_database": str(source_database),
            "source_database_sha256": sha256_file(source_database),
            "memory_count": len(provenance),
            "completed_session_ids": [],
            "provenance": provenance,
        }
        (state_dir / PROVENANCE_NAME).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        samples.append(
            {
                "sample": str(relative.parent),
                "source_database": str(source_database),
                "copied_database": str(copied_database),
                "source_database_sha256": payload["source_database_sha256"],
                "memory_count": len(provenance),
                "provenance_file": str(state_dir / PROVENANCE_NAME),
            }
        )

    manifest = {
        "version": 1,
        "mode": "qa_only_isolated_memory_reuse",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_memory": str(source_memory),
        "source_snapshot": str(snapshot),
        "source_snapshot_sha256": sha256_file(snapshot),
        "destination": str(destination),
        "sample_count": len(samples),
        "memory_count": sum(int(row["memory_count"]) for row in samples),
        "samples": samples,
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_memory", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    manifest = prepare(args.source_memory, args.destination)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
