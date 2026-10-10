"""Cross-Episode Affinity Graph construction and visible-session prefix graphs."""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Sequence, Any

import numpy as np

from .utils import DatasetLayout
from .utils import atomic_binary_writer, sha256_file, write_json_atomic
from .memory import MemoryBank, iter_node_attributes, serialize_attribute


# Cross-Episode Affinity Graph

ATTRIBUTE_WEIGHTING_MODES = ("idf",)


def _attribute_record(attribute: tuple[str, str]) -> Dict[str, str]:
    return {"attribute": attribute[0], "value": attribute[1]}


def build_affinity_graph(
    bank: MemoryBank,
    *,
    degree_cap: int = 4,
    attribute_weighting: str = "idf",
) -> Dict[str, object]:
    """Build and prune an undirected Cross-Episode Affinity Graph from Ti/Vi anchors."""
    if degree_cap < 0:
        raise ValueError("degree_cap cannot be negative")
    if attribute_weighting not in ATTRIBUTE_WEIGHTING_MODES:
        raise ValueError(
            f"attribute_weighting must be one of {ATTRIBUTE_WEIGHTING_MODES}, "
            f"got {attribute_weighting!r}"
        )
    node_count = len(bank.memories)
    text_by_node = [set(iter_node_attributes(item.textual_anchors)) for item in bank.memories]
    visual_by_node = [set(iter_node_attributes(item.visual_anchors)) for item in bank.memories]

    members: Dict[tuple[str, str], set[int]] = {}
    text_members: Dict[tuple[str, str], list[int]] = {}
    visual_members: Dict[tuple[str, str], list[int]] = {}
    for position, (textual_anchors, visual_anchors) in enumerate(
        zip(text_by_node, visual_by_node)
    ):
        for attribute in textual_anchors | visual_anchors:
            members.setdefault(attribute, set()).add(position)
        for attribute in textual_anchors:
            text_members.setdefault(attribute, []).append(position)
        for attribute in visual_anchors:
            visual_members.setdefault(attribute, []).append(position)

    idf = {
        attribute: math.log((node_count + 1) / (len(positions) + 1))
        for attribute, positions in members.items()
        if node_count and positions
    }
    attribute_weights = {
        attribute: idf[attribute]
        for attribute in idf
    }
    shared_text: Dict[tuple[int, int], set[tuple[str, str]]] = {}
    shared_cross: Dict[tuple[int, int], set[tuple[str, str]]] = {}
    for attribute in sorted(members):
        textual = text_members.get(attribute, [])
        visual = visual_members.get(attribute, [])
        for left_index, left in enumerate(textual):
            for right in textual[left_index + 1:]:
                pair = (left, right) if left < right else (right, left)
                shared_text.setdefault(pair, set()).add(attribute)
        for left in textual:
            for right in visual:
                if left == right:
                    continue
                pair = (left, right) if left < right else (right, left)
                shared_cross.setdefault(pair, set()).add(attribute)

    candidate_pairs = list(dict.fromkeys([*shared_text, *shared_cross]))
    candidates = []
    for pair in candidate_pairs:
        textual_anchors = shared_text.get(pair, set())
        cross_attributes = shared_cross.get(pair, set())
        weight = sum(attribute_weights[value] for value in textual_anchors) + sum(
            attribute_weights[value] for value in cross_attributes
        )
        candidates.append((weight, pair, textual_anchors, cross_attributes))
    candidates.sort(key=lambda row: -row[0])

    degree = [0] * node_count
    edges = []
    for weight, (left, right), textual_anchors, cross_attributes in candidates:
        if degree[left] >= degree_cap or degree[right] >= degree_cap:
            continue
        degree[left] += 1
        degree[right] += 1
        edges.append(
            {
                "source": bank.memories[left].id,
                "target": bank.memories[right].id,
                "weight": float(weight),
                "shared_text": [
                    _attribute_record(value) for value in sorted(textual_anchors)
                ],
                "shared_cross": [
                    _attribute_record(value) for value in sorted(cross_attributes)
                ],
            }
        )

    # New indexes contain no temporal or event-relation graph state.
    for item in bank.memories:
        item.links = {"prev": None, "next": None, "related": []}
    return {
        "schema_version": 2,
        "attribute_weighting": attribute_weighting,
        "nodes": node_count,
        "degree_cap": degree_cap,
        "candidate_edges": len(candidates),
        "edges_kept": len(edges),
        "attribute_weights": [
            {
                **_attribute_record(attribute),
                "text": serialize_attribute(attribute),
                "df": len(members[attribute]),
                "weight": float(attribute_weights[attribute]),
            }
            for attribute in sorted(idf)
        ],
        # Preserve the established schema for existing IDF graph consumers.
        "idf": [
            {
                **_attribute_record(attribute),
                "text": serialize_attribute(attribute),
                "df": len(members[attribute]),
                "idf": float(idf[attribute]),
            }
            for attribute in sorted(idf)
        ],
        "edges": edges,
    }


def process_dataset_dir(
    dataset_dir: Path,
    *,
    degree_cap: int = 4,
    attribute_weighting: str = "idf",
) -> Dict[str, object]:
    """Persist the Cross-Episode Affinity Graph for one memory bank."""
    layout = DatasetLayout(dataset_dir)
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    bank = MemoryBank.load(dataset_dir)
    summary = {
        "dataset_dir": str(dataset_dir),
        **build_affinity_graph(
            bank,
            degree_cap=degree_cap,
            attribute_weighting=attribute_weighting,
        ),
    }
    bank.save(dataset_dir)
    write_json_atomic(layout.edges_manifest, summary, trailing_newline=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Cross-Episode Affinity Graph.")
    parser.add_argument("dataset_dirs", nargs="+", help="Dataset dirs containing memories.jsonl + vectors/text.npy")
    parser.add_argument("--degree-cap", type=int, default=4, help="Maximum final undirected degree per memory episode")
    args = parser.parse_args()
    for dataset_dir in args.dataset_dirs:
        summary = process_dataset_dir(Path(dataset_dir), degree_cap=args.degree_cap)
        compact = {
            key: value
            for key, value in summary.items()
            if key not in {"idf", "attribute_weights", "edges"}
        }
        print(json.dumps(compact, ensure_ascii=False))


# Visible-session prefix graphs

PREFIX_GRAPH_SCHEMA_VERSION = 2


def materialize_prefix_graph(
    source_dataset_dir: str | Path,
    checkpoint_index_root: str | Path,
    *,
    sample_id: str,
    checkpoint_id: str,
    visible_session_ids: Sequence[str],
    graph_options: dict[str, Any] | None = None,
) -> Path:
    """Create a checkpoint-local HiVe_mem index using only visible sessions.

    Memory episodes and their embeddings are reused. Cross-episode affinity
    graph statistics are rebuilt from the cumulative prefix,
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
    if source_layout.attributes.exists() or source_layout.attribute_vectors.exists():
        if not source_layout.attributes.is_file() or not source_layout.attribute_vectors.is_file():
            raise ValueError(
                "Attribute metadata and vectors must both exist in the source index"
            )
        source_paths.extend((source_layout.attributes, source_layout.attribute_vectors))
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
        DatasetLayout(prefix_dataset_dir).edges_manifest,
    ]
    if source_layout.attributes.is_file():
        cached_paths.extend(
            (
                DatasetLayout(prefix_dataset_dir).attributes,
                DatasetLayout(prefix_dataset_dir).attribute_vectors,
            )
        )
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
        if isinstance(manifest, dict) and manifest.get("signature") == signature and _valid_prefix_cache(prefix_dataset_dir, options, set(visible), manifest.get("edge_report")):
            return checkpoint_index_root

    source_bank = MemoryBank.load(source_dataset_dir)
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
        prefix_bank = MemoryBank()
        prefix_bank.memories = [
            deepcopy(source_bank.memories[index]) for index in selected_indices
        ]
        for item in prefix_bank.memories:
            item.links = {"prev": None, "next": None, "related": []}
        prefix_bank.save(temporary_dataset_dir)
        if source_layout.attributes.is_file():
            destination_layout = DatasetLayout(temporary_dataset_dir)
            shutil.copy2(source_layout.attributes, destination_layout.attributes)
            shutil.copy2(
                source_layout.attribute_vectors,
                destination_layout.attribute_vectors,
            )
        _slice_image_vectors(
            source_layout,
            DatasetLayout(temporary_dataset_dir),
            selected_indices,
            source_memory_count=len(source_bank),
        )
        edge_report = process_dataset_dir(
            temporary_dataset_dir,
            degree_cap=int(options.get("degree_cap", 4)),
            attribute_weighting=str(options.get("attribute_weighting", "idf")),
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


def _valid_prefix_cache(directory: Path, options: dict[str, Any], visible: set[str], expected_report: Any) -> bool:
    """Reuse only a complete, readable index for the requested visible prefix."""
    from .retriever import HorizontalMemoryExpansionIndex

    try:
        index = HorizontalMemoryExpansionIndex(
            directory,
            degree_cap=int(options.get("degree_cap", 4)),
            attribute_weighting=str(options.get("attribute_weighting", "idf")),
        )
        if not len(index.bank) or any(str(item.metadata.get("session_id", "")) not in visible for item in index.bank.memories):
            return False
        layout = DatasetLayout(directory)
        report = json.loads(layout.edges_manifest.read_text(encoding="utf-8"))
        if report != expected_report or report.get("nodes") != len(index.bank):
            return False
        if layout.image_mask.is_file() and np.load(layout.image_mask, allow_pickle=False).dtype != np.bool_:
            return False
        if layout.attributes.is_file():
            rows = json.loads(layout.attributes.read_text(encoding="utf-8"))
            vectors = np.load(layout.attribute_vectors, allow_pickle=False)
            if not isinstance(rows, list) or vectors.shape != (len(rows), index.text_vectors.shape[1]) or not np.isfinite(vectors).all():
                return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return True


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


if __name__ == "__main__":
    main()
