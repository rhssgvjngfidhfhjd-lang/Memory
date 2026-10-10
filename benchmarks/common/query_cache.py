"""Shared query-vector cache and portable Mem-Gallery query identifiers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import posixpath
import threading
from typing import Any

import numpy as np

from src.utils import sha256_file, source_reference


QUERY_ID_SCHEME = "memgallery-portable-v2"


def h2hmem_question_id(
    variant: str, conversation_id: str, session_id: str, qa_index: int,
    qa: dict[str, Any], *, expected_question_ids=None,
) -> str:
    """Use the frozen manifest to resolve historical H2HMem ID conventions."""
    original_id = str(qa.get("original_question_id") or "").strip()
    if original_id:
        # A changed explicit source ID must fail the existing completeness check.
        return f"h2hmem:{variant}:{conversation_id}:{session_id}:{original_id}"
    codes = [str(qa.get("question_id") or "").strip()]
    codes.append(f"Q{qa_index:03d}")
    candidates = list(dict.fromkeys(
        f"h2hmem:{variant}:{conversation_id}:{session_id}:{code}"
        for code in codes if code
    ))
    if expected_question_ids is not None:
        matches = [value for value in candidates if value in expected_question_ids]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous H2HMEM question IDs in the split manifest: {matches}")
        if matches:
            return matches[0]
    return candidates[0]


def _image_identity(query_image: dict[str, Any] | None) -> str:
    if not query_image:
        return ""
    identifier = str(query_image.get("img_id") or query_image.get("id") or "")
    if identifier and not identifier.startswith("/"):
        if identifier.startswith("dataset://"):
            return identifier
        relative = identifier.replace("\\", "/")
        relative = relative.removeprefix("../image/")
        return "dataset://memgallery/" + posixpath.normpath("image/" + relative)
    image_path = str(query_image.get("path") or "")
    if not image_path:
        return ""
    portable = source_reference(image_path)
    if not portable.startswith("/"):
        return portable
    path = Path(image_path)
    if path.is_file():
        # Custom callers without a dataset/image ID can still identify image
        # content independently of the absolute directory in which it lives.
        return "sha256://" + sha256_file(path)
    raise ValueError(
        "A portable query image requires img_id/id, a configured dataset root, "
        "or an existing image file."
    )


def make_query_id(
    *,
    dataset_name: str,
    qa_index: int,
    category: str,
    question: str,
    query_image: dict[str, Any] | None = None,
) -> str:
    caption = str((query_image or {}).get("caption") or "")
    digest = hashlib.sha1(
        "\n".join([category, question, _image_identity(query_image), caption]).encode("utf-8")
    ).hexdigest()[:16]
    return f"{dataset_name}::{qa_index}::{category}::v2::{digest}"


def legacy_query_id(
    *,
    dataset_name: str,
    qa_index: int,
    category: str,
    question: str,
    query_image: dict[str, Any] | None = None,
) -> str:
    """Original path-sensitive ID, retained solely for existing cache readers."""
    image_path = ""
    image_caption = ""
    if query_image:
        image_path = str(query_image.get("path", "") or "")
        image_caption = str(query_image.get("caption", "") or "")
    digest = hashlib.sha1(
        "\n".join([category, question, image_path, image_caption]).encode("utf-8")
    ).hexdigest()[:16]
    return f"{dataset_name}::{qa_index}::{category}::{digest}"


class QueryEmbeddingCache:
    def __init__(
        self,
        cache_dir: str | Path,
        expected_dim: int = 2048,
        expected_model: str = "",
        expected_revision: str = "",
    ):
        self.cache_dir = Path(cache_dir)
        self.expected_dim = int(expected_dim)
        self.expected_model = str(expected_model or "").strip()
        self.expected_revision = str(expected_revision or "").strip()
        self.vectors_path = self.cache_dir / "vectors.npy"
        self.metadata_path = self.cache_dir / "metadata.jsonl"
        self.manifest_path = self.cache_dir / "manifest.json"
        self._vectors: np.ndarray | None = None
        self._id_to_index: dict[str, int] = {}
        self._semantic_to_index: dict[tuple[str, int, str, str], int] = {}
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self._vectors is not None:
            return
        with self._load_lock:
            if self._vectors is not None:
                return
            vectors = np.load(self.vectors_path, allow_pickle=False)
            if vectors.ndim != 2 or vectors.shape[1] != self.expected_dim:
                raise ValueError(
                    f"Expected query vectors shape (*, {self.expected_dim}), got {vectors.shape}"
                )
            if not np.isfinite(vectors).all():
                raise ValueError(f"Query vectors contain NaN or Inf: {self.vectors_path}")
            if self.manifest_path.exists():
                manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                manifest_count = int(manifest.get("count", len(vectors)))
                manifest_dim = int(manifest.get("dim", self.expected_dim))
                if manifest_count != len(vectors) or manifest_dim != self.expected_dim:
                    raise ValueError(
                        "Query cache manifest does not match vectors: "
                        f"count={manifest_count}/{len(vectors)}, "
                        f"dim={manifest_dim}/{self.expected_dim}"
                    )
                actual_model = str(manifest.get("model_name") or "")
                if self.expected_model and actual_model != self.expected_model:
                    raise ValueError(
                        f"Query cache model {actual_model!r} != expected {self.expected_model!r}"
                    )
                actual_revision = str(manifest.get("model_revision") or "").strip()
                if self.expected_revision and actual_revision != self.expected_revision:
                    raise ValueError(f"Query cache revision {actual_revision!r} != expected {self.expected_revision!r}")
            elif self.expected_model or self.expected_revision:
                raise ValueError(f"Query cache manifest is required to verify model/revision: {self.manifest_path}")
            id_to_index: dict[str, int] = {}
            semantic_to_index: dict[tuple[str, int, str, str], int] = {}
            seen_semantic_keys: set[tuple[str, int, str, str]] = set()
            metadata_count = 0
            with self.metadata_path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    query_id = str(row["query_id"])
                    if query_id in id_to_index:
                        raise ValueError(
                            f"Duplicate query_id {query_id!r} at {self.metadata_path}:{line_number}"
                        )
                    id_to_index[query_id] = metadata_count
                    aliases = [str(row.get("legacy_query_id") or "")]
                    is_memgallery_id = "::v2::" in query_id or len(query_id.split("::")) == 4
                    if is_memgallery_id and "dataset" in row and "qa_index" in row:
                        if "::v2::" not in query_id:
                            try:
                                aliases.append(make_query_id(
                                    dataset_name=str(row["dataset"]),
                                    qa_index=int(row["qa_index"]),
                                    category=str(row.get("category") or ""),
                                    question=str(row.get("question") or ""),
                                    query_image=row.get("query_image"),
                                ))
                            except ValueError:
                                # Older minimal caches may have no stable image
                                # ID and refer to an unavailable machine.
                                pass
                        semantic_key = _semantic_key(
                            row["dataset"],
                            row["qa_index"],
                            row.get("category", ""),
                            row.get("question", ""),
                        )
                        if semantic_key in seen_semantic_keys:
                            raise ValueError(
                                f"Duplicate semantic query key at {self.metadata_path}:{line_number}"
                            )
                        seen_semantic_keys.add(semantic_key)
                        # A v2 cache must not silently reuse an embedding after
                        # the query image/caption changes. This fallback belongs
                        # to legacy rows whose hashes included absolute paths.
                        if "::v2::" not in query_id:
                            semantic_to_index[semantic_key] = metadata_count
                    for alias in filter(None, aliases):
                        previous = id_to_index.get(alias)
                        if previous is not None and previous != metadata_count:
                            raise ValueError(
                                f"Duplicate query alias at {self.metadata_path}:{line_number}"
                            )
                        id_to_index[alias] = metadata_count
                    metadata_count += 1
            if metadata_count != len(vectors):
                raise ValueError(
                    f"Query metadata/vector count mismatch: {metadata_count} vs {len(vectors)}"
                )
            self._id_to_index = id_to_index
            self._semantic_to_index = semantic_to_index
            self._vectors = vectors.astype(np.float32, copy=False)

    def get_by_id(self, query_id: str) -> list[float] | None:
        self.load()
        idx = self._id_to_index.get(query_id)
        if idx is None or self._vectors is None:
            return None
        return self._vectors[idx].tolist()

    def get(
        self,
        *,
        dataset_name: str,
        qa_index: int,
        category: str,
        question: str,
        query_image: dict[str, Any] | None = None,
    ) -> list[float] | None:
        arguments = dict(
            dataset_name=dataset_name, qa_index=qa_index, category=category,
            question=question, query_image=query_image,
        )
        identifiers = [legacy_query_id(**arguments)]
        try:
            identifiers.insert(0, make_query_id(**arguments))
        except ValueError:
            pass
        for query_id in identifiers:
            vector = self.get_by_id(query_id)
            if vector is not None:
                return vector
        self.load()
        index = self._semantic_to_index.get(
            _semantic_key(dataset_name, qa_index, category, question)
        )
        if index is None or self._vectors is None:
            return None
        return self._vectors[index].tolist()


def _semantic_key(
    dataset_name: Any,
    qa_index: Any,
    category: Any,
    question: Any,
) -> tuple[str, int, str, str]:
    """Identify a QA independently of machine-specific absolute image paths."""
    return str(dataset_name), int(qa_index), str(category), str(question)
