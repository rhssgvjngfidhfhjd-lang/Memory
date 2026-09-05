from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np

from benchmarks.io_utils import atomic_binary_writer, sha256_file, write_json_atomic
from hive_mem.build_memory_edges import process_dataset_dir
from hive_mem.mau import MAUBank
from hive_mem.output_layout import DatasetLayout


PREFIX_GRAPH_SCHEMA_VERSION = 1


def materialize_prefix_graph(
    source_dataset_dir: str | Path,
    checkpoint_index_root: str | Path,
    *,
    sample_id: str,
    checkpoint_id: str,
    visible_session_ids: Sequence[str],
    graph_options: dict[str, Any] | None = None,
) -> Path:
    """Create a checkpoint-local HiveMem index using only visible sessions.

    MAUs and their embeddings are reused. Temporal links and deterministic
    entity/attribute graph statistics are rebuilt from the cumulative prefix,
    so no full-sample graph structure is inherited.
    """
    source_dataset_dir = Path(source_dataset_dir)
    checkpoint_index_root = Path(checkpoint_index_root)
    visible = tuple(dict.fromkeys(str(value) for value in visible_session_ids))
    if not visible:
        raise ValueError("A prefix graph requires at least one visible session")

    source_layout = DatasetLayout(source_dataset_dir)
    text_path = source_layout.existing_vector_path("text.npy", "vectors.npy")
    image_path = source_layout.existing_vector_path("image.npy", "image_vectors.npy")
    image_mask_path = source_layout.existing_vector_path("image_mask.npy", "image_mask.npy")
    source_paths = [source_dataset_dir / "memories.jsonl", text_path]
    if image_path.exists() or image_mask_path.exists():
        if image_path.exists() != image_mask_path.exists():
            raise ValueError(
                "Image vectors and image mask must both exist in the source index"
            )
        source_paths.extend((image_path, image_mask_path))
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing prefix-graph source file: {path}")

    options = dict(graph_options or {})
    signature_payload = {
        "schema_version": PREFIX_GRAPH_SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "checkpoint_id": str(checkpoint_id),
        "visible_session_ids": list(visible),
        "graph_options": options,
        "source_files": {
            path.name: sha256_file(path) for path in source_paths
        },
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    manifest_path = checkpoint_index_root / "prefix_manifest.json"
    prefix_dataset_dir = checkpoint_index_root / "datasets" / str(sample_id)
    cached_paths = [
        prefix_dataset_dir / "memories.jsonl",
        DatasetLayout(prefix_dataset_dir).text_vectors,
    ]
    if image_path.exists():
        cached_paths.extend(
            (
                DatasetLayout(prefix_dataset_dir).image_vectors,
                DatasetLayout(prefix_dataset_dir).image_mask,
            )
        )
    if manifest_path.is_file() and all(path.is_file() for path in cached_paths):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("signature") == signature:
            return checkpoint_index_root

    source_bank = MAUBank.load(source_dataset_dir)
    allowed = set(visible)
    selected_indices = [
        index
        for index, item in enumerate(source_bank.memories)
        if str((item.metadata or {}).get("session_id") or "") in allowed
    ]
    if not selected_indices:
        raise ValueError(
            f"No memories from {sample_id} belong to checkpoint {checkpoint_id} "
            f"visible sessions {list(visible)!r}"
        )

    checkpoint_index_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{checkpoint_index_root.name}.",
            dir=checkpoint_index_root.parent,
        )
    )
    try:
        temporary_dataset_dir = temporary_root / "datasets" / str(sample_id)
        prefix_bank = MAUBank()
        prefix_bank.memories = [
            deepcopy(source_bank.memories[index]) for index in selected_indices
        ]
        for item in prefix_bank.memories:
            item.links = {"prev": None, "next": None, "related": []}
        prefix_bank.save(temporary_dataset_dir)
        _slice_image_vectors(
            source_layout,
            DatasetLayout(temporary_dataset_dir),
            selected_indices,
            source_memory_count=len(source_bank),
        )
        edge_report = process_dataset_dir(
            temporary_dataset_dir,
            llm_client=None,
            event_relations=False,
            session_window=1,
            min_confidence=0.7,
            max_pairs=0,
            df_max=float(options.get("df_max", 0.3)),
            df_stop=float(options.get("df_stop", 0.5)),
            min_shared=int(options.get("min_shared", 2)),
            degree_cap=int(options.get("degree_cap", 10)),
        )
        edge_report["dataset_dir"] = str(prefix_dataset_dir.resolve())
        write_json_atomic(
            DatasetLayout(temporary_dataset_dir).edges_manifest,
            edge_report,
        )
        write_json_atomic(
            temporary_root / "prefix_manifest.json",
            {
                **signature_payload,
                "signature": signature,
                "source_dataset_dir": str(source_dataset_dir.resolve()),
                "memory_count": len(prefix_bank),
                "edge_report": edge_report,
            },
        )
        if checkpoint_index_root.exists():
            shutil.rmtree(checkpoint_index_root)
        os.replace(temporary_root, checkpoint_index_root)
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
    return checkpoint_index_root


def _slice_image_vectors(
    source: DatasetLayout,
    destination: DatasetLayout,
    selected_indices: Sequence[int],
    *,
    source_memory_count: int,
) -> None:
    image_path = source.existing_vector_path("image.npy", "image_vectors.npy")
    mask_path = source.existing_vector_path("image_mask.npy", "image_mask.npy")
    if not image_path.exists() and not mask_path.exists():
        return
    if image_path.exists() != mask_path.exists():
        raise ValueError("Image vectors and image mask must both exist")
    image_vectors = np.load(image_path, mmap_mode="r", allow_pickle=False)
    image_mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if len(image_vectors) != source_memory_count or len(image_mask) != source_memory_count:
        raise ValueError(
            "Source image-vector rows do not match the source memory count"
        )
    indices = np.asarray(selected_indices, dtype=np.int64)
    with atomic_binary_writer(destination.image_vectors) as handle:
        np.save(handle, np.asarray(image_vectors[indices], dtype=np.float32))
    with atomic_binary_writer(destination.image_mask) as handle:
        np.save(handle, np.asarray(image_mask[indices], dtype=bool))
