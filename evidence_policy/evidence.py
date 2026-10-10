"""Evidence-selection inputs, artifacts, Vertical Evidence Composition, and actions.

Data readers and GVV indexes do not load the training stack. Tensor operations
import PyTorch only when called.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.utils import dataset_root
from src.graph import materialize_prefix_graph
from src.utils import sha256_file as _file_sha256
from src.retriever import (
    DEFAULT_HIVEMEM_GRAPH_OPTIONS,
    HorizontalMemoryExpansionIndex,
    MemoryHit,
)

if TYPE_CHECKING:
    import torch


# Split manifests

_SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


def normalize_split_name(value: str) -> str:
    try:
        return _SPLIT_ALIASES[str(value).strip().lower()]
    except KeyError as exc:
        allowed = ", ".join(sorted(_SPLIT_ALIASES))
        raise ValueError(f"Unknown split {value!r}; expected one of: {allowed}") from exc


@dataclass(frozen=True)
class SplitConversation:
    data_source: str
    split: str
    conversation_id: str
    source_id: str
    variant: str
    question_ids: tuple[str, ...]


class SplitManifestIndex:
    """Validated, read-only index over a conversation-level split manifest."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Only split manifest schema_version=1 is supported")
        if payload.get("split_unit") != "conversation":
            raise ValueError("Evidence-policy splits must use split_unit='conversation'")

        self.payload: dict[str, Any] = payload
        self.file_sha256 = _file_sha256(self.path)
        self._conversations: dict[str, SplitConversation] = {}
        self._questions: dict[str, tuple[str, str, str]] = {}
        self._by_source_split: dict[tuple[str, str], list[SplitConversation]] = {}
        self._load()

    @property
    def data_sources(self) -> tuple[str, ...]:
        return tuple(str(row["data_source"]) for row in self.payload["datasets"])

    def _load(self) -> None:
        expected_splits = {"train", "val", "test"}
        for dataset in self.payload.get("datasets", []):
            data_source = str(dataset.get("data_source", "")).strip()
            if not data_source:
                raise ValueError("Manifest dataset is missing data_source")
            splits = dataset.get("splits") or {}
            if set(splits) != expected_splits:
                raise ValueError(
                    f"{data_source} must define exactly {sorted(expected_splits)}, "
                    f"got {sorted(splits)}"
                )
            for raw_split, split_payload in splits.items():
                split = normalize_split_name(raw_split)
                rows = split_payload.get("conversations") or []
                actual_question_count = 0
                for row in rows:
                    question_ids = tuple(str(value) for value in row.get("question_ids", []))
                    conversation = SplitConversation(
                        data_source=data_source,
                        split=split,
                        conversation_id=str(row.get("conversation_id", "")),
                        source_id=str(row.get("source_id", "")),
                        variant=str(row.get("variant", "")),
                        question_ids=question_ids,
                    )
                    if not conversation.conversation_id or not conversation.source_id:
                        raise ValueError(f"Incomplete conversation entry in {data_source}/{split}")
                    if conversation.conversation_id in self._conversations:
                        previous = self._conversations[conversation.conversation_id]
                        raise ValueError(
                            f"Conversation {conversation.conversation_id!r} appears in both "
                            f"{previous.split} and {split}"
                        )
                    self._conversations[conversation.conversation_id] = conversation
                    self._by_source_split.setdefault((data_source, split), []).append(conversation)
                    for question_id in question_ids:
                        if question_id in self._questions:
                            previous = self._questions[question_id]
                            raise ValueError(
                                f"Question {question_id!r} appears in both "
                                f"{previous[1]} and {split}"
                            )
                        self._questions[question_id] = (
                            data_source,
                            split,
                            conversation.conversation_id,
                        )
                    actual_question_count += len(question_ids)
                if int(split_payload.get("conversation_count", -1)) != len(rows):
                    raise ValueError(f"Conversation count mismatch for {data_source}/{split}")
                if int(split_payload.get("question_count", -1)) != actual_question_count:
                    raise ValueError(f"Question count mismatch for {data_source}/{split}")

        if not self._conversations or not self._questions:
            raise ValueError("Split manifest is empty")

    def conversations(
        self, split: str, *, data_source: str | None = None
    ) -> tuple[SplitConversation, ...]:
        normalized = normalize_split_name(split)
        if data_source is not None:
            return tuple(self._by_source_split.get((data_source, normalized), ()))
        return tuple(
            row for row in self._conversations.values() if row.split == normalized
        )

    def source_ids(self, split: str, data_source: str) -> tuple[str, ...]:
        return tuple(row.source_id for row in self.conversations(split, data_source=data_source))

    def ordered_question_ids(
        self, split: str, *, data_source: str | None = None
    ) -> tuple[str, ...]:
        """Return question IDs in the exact order recorded by the manifest."""
        return tuple(
            question_id
            for conversation in self.conversations(split, data_source=data_source)
            for question_id in conversation.question_ids
        )

    def question_ids_for_conversation(
        self,
        split: str,
        data_source: str,
        *,
        conversation_id: str | None = None,
        source_id: str | None = None,
    ) -> tuple[str, ...]:
        """Return one conversation's question IDs without losing manifest order."""
        if (conversation_id is None) == (source_id is None):
            raise ValueError("Specify exactly one of conversation_id or source_id")
        matches = [
            row
            for row in self.conversations(split, data_source=data_source)
            if (
                row.conversation_id == conversation_id
                if conversation_id is not None
                else row.source_id == source_id
            )
        ]
        if len(matches) != 1:
            key = conversation_id if conversation_id is not None else source_id
            raise KeyError(
                f"Expected one manifest conversation for {data_source}/{split}/{key}, "
                f"found {len(matches)}"
            )
        return matches[0].question_ids

    def question_ids(self, split: str, *, data_source: str | None = None) -> frozenset[str]:
        return frozenset(self.ordered_question_ids(split, data_source=data_source))

    def contains_question(self, split: str, data_source: str, question_id: str) -> bool:
        row = self._questions.get(str(question_id))
        return row is not None and row[:2] == (data_source, normalize_split_name(split))

    def split_for_question(self, question_id: str) -> str:
        try:
            return self._questions[str(question_id)][1]
        except KeyError as exc:
            raise KeyError(f"Question is not present in split manifest: {question_id}") from exc

    def iter_question_assignments(
        self, *, data_source: str | None = None, split: str | None = None
    ) -> Iterator[tuple[str, str, str, str]]:
        normalized = normalize_split_name(split) if split is not None else None
        for question_id, (current_source, current_split, conversation_id) in self._questions.items():
            if data_source is not None and current_source != data_source:
                continue
            if normalized is not None and current_split != normalized:
                continue
            yield question_id, current_source, current_split, conversation_id

    def summary(self, *, excluded_question_ids: Iterable[str] = ()) -> dict[str, Any]:
        excluded = {str(value) for value in excluded_question_ids}
        result: dict[str, Any] = {
            "manifest": str(self.path),
            "manifest_file_sha256": self.file_sha256,
            "split_unit": "conversation",
            "splits": {},
            "data_sources": {},
        }
        for split in ("train", "val", "test"):
            conversations = self.conversations(split)
            questions = self.question_ids(split)
            result["splits"][split] = {
                "conversation_count": len(conversations),
                "question_count": len(questions),
                "effective_question_count": len(questions.difference(excluded)),
            }
        for source in self.data_sources:
            result["data_sources"][source] = {
                split: {
                    "conversation_count": len(self.conversations(split, data_source=source)),
                    "question_count": len(self.question_ids(split, data_source=source)),
                }
                for split in ("train", "val", "test")
            }
        return result


# Benchmark question sources

@dataclass(frozen=True)
class SourceQuestion:
    split: str
    data_source: str
    conversation_id: str
    source_id: str
    question_id: str
    question: str
    answer: str
    category: str
    source_path: str
    question_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _question_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return str(value or "")


def _h2h_category(question: dict[str, Any]) -> str:
    value = question.get("question_type") or {}
    if not isinstance(value, dict):
        return str(value or "")
    return str(value.get("subsub_type") or value.get("sub_type") or value.get("main_type") or "")


def _session_sort_key(path: Path) -> tuple[int, str]:
    suffix = "".join(character for character in path.name if character.isdigit())
    return (int(suffix) if suffix else 10**9, path.name)


def _iter_h2h_questions(
    workspace_root: Path | None,
    conversation: SplitConversation,
    *,
    data_dir: Path | None = None,
) -> Iterator[SourceQuestion]:
    from benchmarks.common.query_cache import h2hmem_question_id

    folder = "dyadic" if conversation.data_source == "h2hmem_dyadic" else "multi-party"
    variant = "dyadic" if folder == "dyadic" else "multiparty"
    data_root = data_dir if data_dir is not None else dataset_root("h2hmem", workspace_root)
    if (data_root / "dataset").is_dir():
        data_root = data_root / "dataset"
    scenes = (
        data_root
        / folder
        / conversation.source_id
        / "scenes"
    )
    if not scenes.is_dir():
        raise FileNotFoundError(f"Missing H2HMem conversation: {scenes}")
    allowed = set(conversation.question_ids)
    for session in sorted((row for row in scenes.iterdir() if row.is_dir()), key=_session_sort_key):
        path = session / "questions.json"
        if not path.is_file():
            # Some H2HMem sessions contain dialogue/image evidence but no local
            # QA file. They remain part of the conversation context; completeness
            # is checked below against the manifest's exact question allowlist.
            continue
        payload = _read_json(path)
        for index, question in enumerate(payload.get("questions", []) or []):
            question_id = h2hmem_question_id(
                variant, conversation.source_id, session.name, index + 1, question,
                expected_question_ids=allowed,
            )
            if question_id not in allowed:
                continue
            question_value = question.get("question") or {}
            yield SourceQuestion(
                split=conversation.split,
                data_source=conversation.data_source,
                conversation_id=conversation.conversation_id,
                source_id=conversation.source_id,
                question_id=question_id,
                question=_question_text(question_value),
                answer=str(question.get("original_answer", "")),
                category=_h2h_category(question),
                source_path=str(path.resolve()),
                question_index=index,
                metadata={
                    "variant": conversation.variant,
                    "session_id": session.name,
                    "difficulty": question.get("difficulty", ""),
                    "question_image": (
                        str(question_value.get("image", ""))
                        if isinstance(question_value, dict)
                        else ""
                    ),
                    "answer_session": question.get("answer_session", []),
                    "question_type": question.get("question_type", {}),
                },
            )


def _iter_mem_gallery_questions(
    workspace_root: Path | None,
    conversation: SplitConversation,
    *,
    data_dir: Path | None = None,
) -> Iterator[SourceQuestion]:
    path = (
        (data_dir if data_dir is not None else dataset_root("memgallery", workspace_root))
        / "dialog"
        / f"{conversation.source_id}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing Mem-Gallery conversation: {path}")
    payload = _read_json(path)
    allowed = set(conversation.question_ids)
    for index, question in enumerate(payload.get("human-annotated QAs", []) or []):
        question_id = f"{conversation.source_id}_q{index:04d}"
        if question_id not in allowed:
            continue
        yield SourceQuestion(
            split=conversation.split,
            data_source=conversation.data_source,
            conversation_id=conversation.conversation_id,
            source_id=conversation.source_id,
            question_id=question_id,
            question=str(question.get("question", "")),
            answer=str(question.get("answer", "")),
            category=str(question.get("point", "")),
            source_path=str(path.resolve()),
            question_index=index,
            metadata={
                "session_id": question.get("session_id", []),
                "clue": question.get("clue", []),
                "question_image": question.get("question_image", ""),
                "image_caption": question.get("image_caption", ""),
            },
        )


def _worldmemarena_paths(
    workspace_root: Path | None, *, data_dir: Path | None = None
) -> dict[str, Path]:
    root = data_dir if data_dir is not None else dataset_root("wma", workspace_root)
    result: dict[str, Path] = {}
    for path in root.rglob("*.json"):
        payload = _read_json(path)
        sample_id = str(payload.get("sample_id", ""))
        if not sample_id:
            continue
        if sample_id in result:
            raise ValueError(f"Duplicate WorldMemArena sample_id {sample_id!r}")
        result[sample_id] = path
    return result


def _iter_worldmemarena_questions(
    paths: dict[str, Path],
    conversation: SplitConversation,
) -> Iterator[SourceQuestion]:
    try:
        path = paths[conversation.source_id]
    except KeyError as exc:
        raise FileNotFoundError(
            f"Missing WorldMemArena lifelong sample: {conversation.source_id}"
        ) from exc
    payload = _read_json(path)
    allowed = set(conversation.question_ids)
    for checkpoint in payload.get("qa_checkpoints", []) or []:
        checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
        for index, question in enumerate(checkpoint.get("questions", []) or []):
            question_id = f"{conversation.source_id}:{checkpoint_id}:Q{index + 1:03d}"
            if question_id not in allowed:
                continue
            yield SourceQuestion(
                split=conversation.split,
                data_source=conversation.data_source,
                conversation_id=conversation.conversation_id,
                source_id=conversation.source_id,
                question_id=question_id,
                question=str(question.get("question", "")),
                answer=str(question.get("answer", "")),
                category=str(question.get("question_type_abbrev", "")),
                source_path=str(path.resolve()),
                question_index=index,
                metadata={
                    "checkpoint_id": checkpoint_id,
                    "covered_sessions": checkpoint.get("covered_sessions", []),
                    "difficulty": question.get("difficulty", ""),
                    "question_type": question.get("question_type", ""),
                    "evidence": question.get("evidence", []),
                },
            )


def iter_source_questions(
    manifest: SplitManifestIndex,
    workspace_root: str | Path | None = None,
    *,
    split: str | None = None,
    data_sources: Iterable[str] | None = None,
    dataset_roots: Mapping[str, str | Path] | None = None,
) -> Iterator[SourceQuestion]:
    """Read manifest questions, with optional explicit per-benchmark data roots.

    Exhaust this iterator to check completeness before applying episode limits
    or category exclusions. Explicit roots do not modify process environment.
    """
    root = Path(workspace_root).expanduser().resolve() if workspace_root is not None else None
    roots = {
        name: Path(path).expanduser().resolve()
        for name, path in (dataset_roots or {}).items()
    }
    unknown_roots = set(roots).difference({"memgallery", "h2hmem", "wma"})
    if unknown_roots:
        raise ValueError(f"Unknown benchmark data roots: {sorted(unknown_roots)}")
    selected_sources = set(data_sources or manifest.data_sources)
    unknown = selected_sources.difference(manifest.data_sources)
    if unknown:
        raise ValueError(f"Unknown manifest data sources: {sorted(unknown)}")
    normalized_split = normalize_split_name(split) if split is not None else None
    wma_paths: dict[str, Path] | None = None
    found: set[str] = set()
    expected = {
        question_id
        for question_id, source, current_split, _ in manifest.iter_question_assignments()
        if source in selected_sources
        and (normalized_split is None or current_split == normalized_split)
    }
    for source in manifest.data_sources:
        if source not in selected_sources:
            continue
        splits = (normalized_split,) if normalized_split is not None else ("train", "val", "test")
        for current_split in splits:
            for conversation in manifest.conversations(current_split, data_source=source):
                if source.startswith("h2hmem_"):
                    rows = _iter_h2h_questions(root, conversation, data_dir=roots.get("h2hmem"))
                elif source == "mem_gallery":
                    rows = _iter_mem_gallery_questions(root, conversation, data_dir=roots.get("memgallery"))
                elif source == "worldmemarena_lifelong":
                    if wma_paths is None:
                        wma_paths = _worldmemarena_paths(root, data_dir=roots.get("wma"))
                    rows = _iter_worldmemarena_questions(wma_paths, conversation)
                else:
                    raise ValueError(f"No source adapter for {source}")
                for row in rows:
                    if row.question_id in found:
                        raise ValueError(f"Duplicate source question id: {row.question_id}")
                    found.add(row.question_id)
                    yield row
    missing = sorted(expected.difference(found))
    extra = sorted(found.difference(expected))
    if missing or extra:
        raise ValueError(
            "Source data does not match split manifest: "
            f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
        )




# Grounded visual view artifacts

@dataclass(frozen=True)
class GroundedVisualView:
    gvv_id: str
    label: str
    crop_path: Path
    bbox_norm: tuple[int, int, int, int]


@dataclass(frozen=True)
class GVVImageRecord:
    image_id: str
    dataset: str
    relative_path: str
    source_sha256: str
    status: str
    grounded_visual_views: tuple[GroundedVisualView, ...]


class GVVArtifactIndex:
    """Read-only lookup over one gvv_extractor artifact run."""

    def __init__(self, run_dir: str | Path, *, max_views_per_image: int = 0):
        self.run_dir = Path(run_dir).expanduser().resolve()
        if max_views_per_image < 0:
            raise ValueError("max_views_per_image must be non-negative")
        self.max_views_per_image = int(max_views_per_image)
        self.run_path = self.run_dir / "run.json"
        self.images_path = self.run_dir / "exports" / "images.jsonl"
        if not self.run_path.is_file():
            raise FileNotFoundError(f"Missing GVV run metadata: {self.run_path}")
        if not self.images_path.is_file():
            raise FileNotFoundError(f"Missing GVV image index: {self.images_path}")
        self.run_metadata: dict[str, Any] = json.loads(
            self.run_path.read_text(encoding="utf-8")
        )
        self.run_id = str(self.run_metadata.get("run_id", ""))
        self.signature = self._signature()
        self._by_sha256: dict[str, GVVImageRecord] = {}
        self._by_dataset_relative: dict[tuple[str, str], GVVImageRecord] = {}
        self._by_basename: dict[str, list[GVVImageRecord]] = {}
        self._path_cache: dict[tuple[str, str], GVVImageRecord | None] = {}
        self._load()

    def _signature(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.run_path.read_bytes())
        digest.update(self.images_path.read_bytes())
        return digest.hexdigest()

    def _load(self) -> None:
        with self.images_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if str(raw.get("schema_version", "")) != "1.0":
                    raise ValueError(
                        f"Unsupported GVV schema at {self.images_path}:{line_number}"
                    )
                source = raw.get("source") or {}
                dataset = str(source.get("dataset", "")).strip()
                relative = _normalize_path(source.get("relative_path", ""))
                source_sha256 = str(source.get("sha256", "")).lower()
                grounded_visual_views: list[GroundedVisualView] = []
                for item in raw.get("grounded_visual_views", []) or []:
                    raw_crop_path = str(item.get("crop_path", "")).strip()
                    if not raw_crop_path:
                        raise ValueError(
                            f"Missing GVV crop path at {self.images_path}:{line_number}"
                        )
                    crop_path = (self.run_dir / raw_crop_path).resolve()
                    bbox = tuple(int(value) for value in item.get("bbox_norm", []))
                    if len(bbox) != 4:
                        raise ValueError(
                            f"Invalid GVV bbox at {self.images_path}:{line_number}"
                        )
                    grounded_visual_views.append(
                        GroundedVisualView(
                            gvv_id=str(item.get("gvv_id", "")),
                            label=str(item.get("label", "")),
                            crop_path=crop_path,
                            bbox_norm=(bbox[0], bbox[1], bbox[2], bbox[3]),
                        )
                    )
                if self.max_views_per_image:
                    grounded_visual_views = grounded_visual_views[: self.max_views_per_image]
                record = GVVImageRecord(
                    image_id=str(raw.get("image_id", "")),
                    dataset=dataset,
                    relative_path=relative,
                    source_sha256=source_sha256,
                    status=str(raw.get("status", "")),
                    grounded_visual_views=tuple(grounded_visual_views),
                )
                key = (dataset.lower(), relative)
                if not dataset or not relative or not record.image_id:
                    raise ValueError(
                        f"Incomplete GVV record at {self.images_path}:{line_number}"
                    )
                if key in self._by_dataset_relative:
                    raise ValueError(f"Duplicate GVV source record: {dataset}:{relative}")
                self._by_dataset_relative[key] = record
                if source_sha256:
                    self._by_sha256.setdefault(source_sha256, record)
                self._by_basename.setdefault(Path(relative).name.lower(), []).append(record)

    def record_for(
        self, image_path: str | Path, *, dataset: str | None = None
    ) -> GVVImageRecord | None:
        raw_path = str(image_path)
        dataset_key = str(dataset or "").lower()
        cache_key = (dataset_key, raw_path)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]
        normalized = _normalize_path(raw_path)
        basename = Path(normalized).name.lower()
        blob_sha256 = (
            basename
            if len(basename) == 64
            and all(character in "0123456789abcdef" for character in basename)
            else ""
        )
        if blob_sha256:
            record = self._by_sha256.get(blob_sha256)
            if record is not None:
                self._path_cache[cache_key] = record
                return record
        candidates = self._by_basename.get(basename, ())
        suffix_matches = [
            row
            for row in candidates
            if (not dataset_key or row.dataset.lower() == dataset_key)
            and (normalized == row.relative_path or normalized.endswith("/" + row.relative_path))
        ]
        if len(suffix_matches) == 1:
            record = suffix_matches[0]
        else:
            path = Path(image_path)
            record = self._by_sha256.get(_file_sha256(path)) if path.is_file() else None
        self._path_cache[cache_key] = record
        return record

    def views_for(
        self, image_path: str | Path, *, dataset: str | None = None
    ) -> tuple[GroundedVisualView, ...]:
        record = self.record_for(image_path, dataset=dataset)
        return record.grounded_visual_views if record is not None else ()

    def has_views(self, image_path: str | Path, *, dataset: str | None = None) -> bool:
        return bool(self.views_for(image_path, dataset=dataset))

    def audit(self, image_paths: Iterable[str | Path]) -> dict[str, int]:
        total = matched = with_views = missing_crops = 0
        for image_path in dict.fromkeys(str(value) for value in image_paths):
            total += 1
            record = self.record_for(image_path)
            if record is None:
                continue
            matched += 1
            if record.grounded_visual_views:
                with_views += 1
            missing_crops += sum(not row.crop_path.is_file() for row in record.grounded_visual_views)
        return {
            "image_count": total,
            "matched_records": matched,
            "missing_records": total - matched,
            "with_views": with_views,
            "without_views": matched - with_views,
            "missing_crop_files": missing_crops,
        }


def _normalize_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").strip().lstrip("./")


# Retrieval input preparation

GRAPH_OPTION_KEYS = {
    "seed_k",
    "mode",
    "append_k",
    "degree_cap",
    "attribute_weighting",
}


def resolve_retrieval_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the vector seeds and affinity-graph append configuration."""
    mode = str(config.get("retrieval_mode") or "graph_append").strip().lower()
    if mode != "graph_append":
        raise ValueError("Evidence-policy retrieval requires retrieval_mode='graph_append'")
    vector_k = int(config.get("top_k", 0))
    if vector_k < 1:
        raise ValueError("top_k must be at least 1")
    graph_options = resolve_graph_options(config)
    return {
        "mode": mode,
        "vector_k": vector_k,
        "append_k": int(graph_options["append_k"]),
        "graph_options": graph_options,
    }


def resolve_graph_options(config: dict[str, Any]) -> dict[str, Any]:
    """Return shared affinity-graph defaults plus explicit policy overrides."""
    raw = config.get("graph_options")
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("graph_options must be an object")
    options = {**DEFAULT_HIVEMEM_GRAPH_OPTIONS, **dict(raw or {})}
    unknown = sorted(set(options) - GRAPH_OPTION_KEYS)
    if unknown:
        raise ValueError(f"Unknown graph_options: {', '.join(unknown)}")
    if options.get("mode") != "append":
        raise ValueError("Evidence-policy graph retrieval requires mode='append'")
    if int(options.get("append_k", 0)) < 1:
        raise ValueError("Evidence-policy graph retrieval requires append_k>=1")
    if int(options.get("seed_k", 0)) < 0:
        raise ValueError("Evidence-policy graph retrieval requires seed_k>=0")
    if int(options.get("degree_cap", 0)) < 0:
        raise ValueError("Evidence-policy graph retrieval requires degree_cap>=0")
    if options.get("attribute_weighting") != "idf":
        raise ValueError("Evidence-policy graph retrieval requires attribute_weighting='idf'")
    return options


def validate_graph_config(config: dict[str, Any]) -> None:
    resolve_retrieval_settings(config)


def retrieve_hits(
    index: HorizontalMemoryExpansionIndex,
    query_vector: list[float] | np.ndarray,
    settings: dict[str, Any],
    *,
    category: str = "",
    allowed_session_ids: set[str] | None = None,
) -> tuple[list[MemoryHit], dict[str, Any]]:
    """Retrieve one question and return both hits and audit metadata."""

    vector_k = int(settings["vector_k"])
    hits = list(
        index.search(
            query_vector,
            top_k=vector_k,
            category=category,
            allowed_session_ids=allowed_session_ids,
        )
    )
    mode = str(settings["mode"])
    requested = int(settings["append_k"])
    vector_ids = [str(hit.item.id) for hit in hits if hit.via == "vector"]
    appended = [hit for hit in hits if hit.via != "vector"]
    actual = len(appended)
    return hits, {
        "retrieval_mode": mode,
        "vector_k": vector_k,
        "append_k_requested": requested,
        "append_k_actual": actual,
        "retrieval_vector_ids": vector_ids,
        "retrieval_append_ids": [str(hit.item.id) for hit in appended],
        "retrieval_final_ids": [str(hit.item.id) for hit in hits],
        "append_shortfall_reason": (
            "no_eligible_graph_candidates"
            if mode == "graph_append" and actual < requested
            else ""
        ),
    }


def build_graph_index(
    dataset_dir: str | Path,
    options: dict[str, Any],
    *,
    visual_categories: set[str] | None = None,
) -> HorizontalMemoryExpansionIndex:
    kwargs = dict(options)
    if visual_categories:
        kwargs["visual_categories"] = visual_categories
    return HorizontalMemoryExpansionIndex(dataset_dir, **kwargs)


def build_wma_prefix_graph_index(
    source_dataset_dir: str | Path,
    cache_root: str | Path,
    *,
    sample_id: str,
    checkpoint_id: str,
    visible_session_ids: Iterable[str],
    options: dict[str, Any],
    visual_categories: set[str] | None = None,
) -> tuple[HorizontalMemoryExpansionIndex, str]:
    checkpoint_root = Path(cache_root) / sample_id / checkpoint_id
    prefix_root = materialize_prefix_graph(
        source_dataset_dir,
        checkpoint_root,
        sample_id=sample_id,
        checkpoint_id=checkpoint_id,
        visible_session_ids=tuple(visible_session_ids),
        graph_options=options,
    )
    manifest = json.loads(
        (prefix_root / "prefix_manifest.json").read_text(encoding="utf-8")
    )
    index = build_graph_index(
        prefix_root / "datasets" / sample_id,
        options,
        visual_categories=visual_categories,
    )
    return index, str(manifest["signature"])


def retrieval_signature(
    dataset_dir: str | Path,
    options: dict[str, Any],
    *,
    prefix_signature: str = "",
    vector_k: int | None = None,
    append_k: int = 0,
) -> str:
    payload = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "retrieval_mode": "graph_append",
        "vector_k": vector_k,
        "append_k": int(append_k),
        "graph_options": options,
        "prefix_signature": prefix_signature,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def retrieval_trace(hits: Iterable[MemoryHit]) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": str(hit.item.id),
            "rank": int(hit.rank),
            "score": float(hit.score),
            "via": str(hit.via),
            "session_id": str(hit.item.metadata.get("session_id", "")),
            "source_dialogue_ids": [
                str(value)
                for value in hit.item.metadata.get("source_dialogue_ids", [])
            ],
        }
        for hit in hits
    ]


# Evidence types, actions, and observations

MEMGALLERY_VISUAL_CATEGORIES = frozenset({"VS", "VR", "TTL"})


class EvidenceType(str, Enum):
    SUMMARY = "summary"
    RAW_INTERACTION_TEXT = "raw_interaction_text"
    CAPTION = "caption"
    IMAGE = "image"
    GVV = "gvv"


EVIDENCE_ORDER = tuple(EvidenceType)
EVIDENCE_SCHEMA_VERSION = 2


class EvidenceStrategy(str, Enum):
    FULL = "full-evidence"
    PPO = "ppo"


@dataclass(frozen=True)
class DialogueEvidence:
    dataset: str
    dialogue_id: str
    user: str
    assistant: str

    def render(self) -> str:
        return f"User: {self.user}\nAssistant: {self.assistant}"


@dataclass(frozen=True)
class MemoryEvidenceAction:
    memory_id: str
    selected: frozenset[EvidenceType] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected",
            frozenset(
                value if isinstance(value, EvidenceType) else EvidenceType(value)
                for value in self.selected
            ),
        )

    @classmethod
    def from_mask(
        cls, memory_id: str, mask: Sequence[bool | int]
    ) -> "MemoryEvidenceAction":
        if len(mask) != len(EVIDENCE_ORDER):
            raise ValueError(f"Evidence mask must have {len(EVIDENCE_ORDER)} bits")
        return cls(
            memory_id,
            frozenset(kind for kind, enabled in zip(EVIDENCE_ORDER, mask) if enabled),
        )

    @property
    def mask(self) -> tuple[bool, ...]:
        return tuple(kind in self.selected for kind in EVIDENCE_ORDER)

    @property
    def bitmask(self) -> str:
        return "".join("1" if enabled else "0" for enabled in self.mask)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "mask": self.bitmask,
            "selected": [kind.value for kind in EVIDENCE_ORDER if kind in self.selected],
        }


@dataclass(frozen=True)
class PolicyObservation:
    query_embedding: torch.Tensor
    summary_embeddings: torch.Tensor
    memory_ids: tuple[str, ...]
    # Per-MemoryEpisode availability in EVIDENCE_ORDER. Unavailable bits are fixed at 0
    # and excluded from policy log-probability and entropy.
    evidence_availability_mask: torch.Tensor

    def validate(self) -> None:
        import torch

        if self.query_embedding.ndim != 1:
            raise ValueError("query_embedding must have shape [embedding_dim]")
        if self.summary_embeddings.ndim != 2:
            raise ValueError("summary_embeddings must have shape [top_k, embedding_dim]")
        top_k, embedding_dim = self.summary_embeddings.shape
        if self.query_embedding.shape[0] != embedding_dim:
            raise ValueError(
                f"Query dim {self.query_embedding.shape[0]} != summary dim {embedding_dim}"
            )
        if len(self.memory_ids) != top_k:
            raise ValueError(f"Expected {top_k} memory ids, got {len(self.memory_ids)}")
        if self.evidence_availability_mask.shape != (top_k, len(EVIDENCE_ORDER)):
            raise ValueError(
                "evidence_availability_mask must have shape "
                f"[{top_k}, {len(EVIDENCE_ORDER)}]"
            )
        if self.evidence_availability_mask.dtype is not torch.bool:
            raise ValueError("evidence_availability_mask must be boolean")

    def to(self, device: torch.device | str) -> "PolicyObservation":
        return PolicyObservation(
            query_embedding=self.query_embedding.to(device),
            summary_embeddings=self.summary_embeddings.to(device),
            memory_ids=self.memory_ids,
            evidence_availability_mask=self.evidence_availability_mask.to(device),
        )


@dataclass
class PolicyStep:
    actions: tuple[MemoryEvidenceAction, ...]
    joint_log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class DialogueStore:
    """Lazy, read-only lookup of original Mem-Gallery dialogue rounds."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self._by_dataset: dict[str, dict[str, DialogueEvidence]] = {}

    def get(self, dataset: str, dialogue_id: str) -> DialogueEvidence:
        if dataset not in self._by_dataset:
            self._by_dataset[dataset] = self._load_dataset(dataset)
        try:
            return self._by_dataset[dataset][dialogue_id]
        except KeyError as exc:
            raise KeyError(f"Unknown dialogue {dataset}:{dialogue_id}") from exc

    def _load_dataset(self, dataset: str) -> dict[str, DialogueEvidence]:
        import json

        path = self.data_dir / "dialog" / f"{dataset}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Mem-Gallery dialogue file: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, DialogueEvidence] = {}
        for session in payload.get("multi_session_dialogues", []) or []:
            for row in session.get("dialogues", []) or []:
                dialogue_id = str(row.get("round", "")).strip()
                if not dialogue_id:
                    continue
                if dialogue_id in result:
                    raise ValueError(f"Duplicate dialogue id in {path}: {dialogue_id}")
                result[dialogue_id] = DialogueEvidence(
                    dataset=dataset,
                    dialogue_id=dialogue_id,
                    user=str(row.get("user", "")),
                    assistant=str(row.get("assistant", "")),
                )
        return result

    def resolve_image_path(self, dataset: str, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        normalized = str(raw_path).replace("\\", "/")
        marker = "/image/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            candidate = self.data_dir / "image" / relative
        else:
            candidate = self.data_dir / "image" / dataset / path.name
        if not candidate.exists():
            raise FileNotFoundError(f"Cannot map stored image path {raw_path!r} to {candidate}")
        return candidate


class WMADialogueStore(DialogueStore):
    """Lazy round lookup for WorldMemArena's nested sample files."""

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir)
        self._paths = _worldmemarena_paths(None, data_dir=self.data_dir)

    def _load_dataset(self, dataset: str) -> dict[str, DialogueEvidence]:
        import json
        from embedding.chunks import _wma_rounds

        try:
            path = self._paths[dataset]
        except KeyError as exc:
            raise FileNotFoundError(f"Missing WorldMemArena sample: {dataset}") from exc
        payload = json.loads(path.read_text(encoding="utf-8"))
        result: dict[str, DialogueEvidence] = {}
        for session in payload.get("sessions", []) or []:
            session_id = str(session.get("_v2_session_id") or session.get("session_id") or "")
            for round_number, user, assistant in _wma_rounds(session.get("dialogue", []) or []):
                dialogue_id = f"{session_id}:R{round_number:04d}"
                result[dialogue_id] = DialogueEvidence(
                    dataset=dataset,
                    dialogue_id=dialogue_id,
                    user=str(user.get("content", "") or ""),
                    assistant=str(assistant.get("content", "") or ""),
                )
        return result

    def resolve_image_path(self, dataset: str, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.exists():
            return path
        try:
            sample_dir = self._paths[dataset].parent
        except KeyError as exc:
            raise FileNotFoundError(f"Missing WorldMemArena sample: {dataset}") from exc
        normalized = str(raw_path).replace("\\", "/")
        candidates = []
        if not Path(normalized).is_absolute():
            candidates.append(self.data_dir / normalized)
        candidates.append(sample_dir / normalized)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Cannot map stored image path {raw_path!r}; tried "
            + ", ".join(str(candidate) for candidate in candidates)
        )


class H2HMemDialogueStore(DialogueStore):
    """Lazy round lookup for H2HMem dyadic and multiparty conversations."""

    def __init__(self, data_dir: str | Path):
        root = Path(data_dir).expanduser()
        if (root / "dataset").is_dir():
            root = root / "dataset"
        super().__init__(root)

    def _load_dataset(self, dataset: str) -> dict[str, DialogueEvidence]:
        import json

        from embedding.chunks import _h2h_speaker_blocks, iter_h2h_session_files

        try:
            variant, conversation_id = dataset.split("_", 1)
        except ValueError as exc:
            raise ValueError(
                "H2HMem memory dataset names must be '<variant>_<conversation>', "
                f"got {dataset!r}"
            ) from exc
        if variant not in {"dyadic", "multiparty"}:
            raise ValueError(f"Unknown H2HMem memory dataset: {dataset!r}")

        result: dict[str, DialogueEvidence] = {}
        for path in iter_h2h_session_files(self.data_dir, variant=variant):
            if path.parents[2].name != conversation_id:
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            blocks = _h2h_speaker_blocks(payload.get("dialogue", []) or [], path.parent)
            for offset in range(0, len(blocks), 2):
                first = blocks[offset]
                second = blocks[offset + 1] if offset + 1 < len(blocks) else None
                dialogue_id = f"{path.parent.name}:R{offset // 2 + 1:04d}"

                def render(block: dict[str, Any] | None) -> str:
                    if block is None:
                        return ""
                    text = "\n".join(str(value) for value in block["texts"] if value)
                    return f"{block['speaker']}: {text}" if text else str(block["speaker"])

                result[dialogue_id] = DialogueEvidence(
                    dataset=dataset,
                    dialogue_id=dialogue_id,
                    user=render(first),
                    assistant=render(second),
                )
        if not result:
            raise FileNotFoundError(
                f"Missing H2HMem conversation for memory dataset {dataset!r}"
            )
        return result

    def resolve_image_path(self, dataset: str, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_file():
            return path
        try:
            variant, conversation_id = dataset.split("_", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid H2HMem memory dataset: {dataset!r}") from exc
        variant_dir = "multi-party" if variant == "multiparty" else variant
        normalized = str(raw_path).replace("\\", "/")
        marker = "/scenes/"
        candidates: list[Path] = []
        if marker in normalized:
            candidates.append(
                self.data_dir
                / variant_dir
                / conversation_id
                / "scenes"
                / normalized.split(marker, 1)[1]
            )
        candidates.append(
            self.data_dir / variant_dir / conversation_id / "scenes" / path.name
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"Cannot map stored H2HMem image path {raw_path!r}; tried "
            + ", ".join(str(candidate) for candidate in candidates)
        )


class EvidenceComposer:
    """Vertical Evidence Composition from independent per-memory evidence masks."""

    def __init__(
        self,
        dialogue_store: DialogueStore,
        *,
        gvv_index: GVVArtifactIndex | None = None,
        visual_categories: set[str] | frozenset[str] | None = None,
    ):
        self.dialogue_store = dialogue_store
        self.gvv_index = gvv_index
        self.visual_categories = {
            str(value).upper()
            for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
        }

    def availability(
        self,
        dataset: str,
        query_category: str,
        memory_hits: Sequence[MemoryHit],
    ) -> torch.Tensor:
        import torch

        visual_allowed = query_category.upper() in self.visual_categories
        rows: list[list[bool]] = []
        for hit in memory_hits:
            metadata = dict(hit.item.metadata or {})
            image_paths = self._values(metadata, "image_paths")
            rows.append(
                [
                    bool(hit.item.summary.strip()),
                    len(self._values(metadata, "source_dialogue_ids")) == 1,
                    bool(self._values(metadata, "image_captions")),
                    bool(image_paths) and visual_allowed,
                    visual_allowed and any(self._grounded_visual_views(dataset, path) for path in image_paths),
                ]
            )
        return torch.as_tensor(rows, dtype=torch.bool)

    def build(
        self,
        dataset: str,
        query_category: str,
        memory_hits: Sequence[MemoryHit],
        actions: Sequence[MemoryEvidenceAction],
    ) -> list[dict[str, Any]]:
        if len(memory_hits) != len(actions):
            raise ValueError("Every retrieved MemoryEpisode must have exactly one evidence action")
        availability = self.availability(dataset, query_category, memory_hits)
        items: list[dict[str, Any]] = []
        for index, (hit, action) in enumerate(zip(memory_hits, actions)):
            if action.memory_id != hit.item.id:
                raise ValueError(
                    f"Action for {action.memory_id} does not match retrieved MemoryEpisode {hit.item.id}"
                )
            unavailable = [
                kind.value
                for kind, selected, allowed in zip(
                    EVIDENCE_ORDER, action.mask, availability[index].tolist()
                )
                if selected and not allowed
            ]
            if unavailable:
                raise ValueError(
                    f"MemoryEpisode {hit.item.id} selected unavailable evidence: {unavailable}"
                )
            if not action.selected:
                continue
            metadata = dict(hit.item.metadata or {})
            text_sections: list[str] = []
            images: list[dict[str, str]] = []
            if EvidenceType.SUMMARY in action.selected:
                text_sections.append(f"Summary:\n{hit.item.summary}")
            if EvidenceType.RAW_INTERACTION_TEXT in action.selected:
                text_sections.append(f"Raw Interaction Text:\n{self._dialogue(dataset, hit).render()}")
            if EvidenceType.CAPTION in action.selected:
                captions = self._values(metadata, "image_captions")
                text_sections.append(
                    "Image captions:\n"
                    + "\n".join(f"- {caption}" for caption in captions)
                )
            image_paths = self._values(metadata, "image_paths")
            image_ids = self._values(metadata, "image_ids")
            if EvidenceType.IMAGE in action.selected:
                resolved_paths = [
                    self._resolve_image_path(dataset, raw_path)
                    for raw_path in image_paths
                ]
                images.extend(
                    {
                        "kind": EvidenceType.IMAGE.value,
                        "path": str(path),
                        "img_id": image_ids[offset] if offset < len(image_ids) else "",
                    }
                    for offset, path in enumerate(resolved_paths)
                )
            if EvidenceType.GVV in action.selected:
                for raw_path in image_paths:
                    images.extend(
                        {
                            "kind": EvidenceType.GVV.value,
                            "path": str(view.crop_path),
                            "img_id": view.gvv_id,
                        }
                        for view in self._grounded_visual_views(dataset, raw_path)
                    )
            items.append(
                {
                    "text": "\n\n".join(text_sections),
                    "images": images,
                    "chunk_id": hit.item.id,
                    "score": hit.score,
                    "metadata": metadata,
                }
            )
        return items

    def _dialogue(self, dataset: str, hit: MemoryHit) -> DialogueEvidence:
        source_ids = self._values(dict(hit.item.metadata or {}), "source_dialogue_ids")
        if len(source_ids) != 1:
            raise ValueError(
                f"MemoryEpisode {hit.item.id} must have exactly one source dialogue, got {source_ids!r}"
            )
        return self.dialogue_store.get(dataset, source_ids[0])

    def _grounded_visual_views(self, dataset: str, raw_path: str) -> tuple[GroundedVisualView, ...]:
        if self.gvv_index is None:
            return ()
        record = self.gvv_index.views_for(raw_path)
        if record:
            return record
        try:
            resolved = self.dialogue_store.resolve_image_path(dataset, raw_path)
        except FileNotFoundError:
            return ()
        return self.gvv_index.views_for(resolved)

    def _resolve_image_path(self, dataset: str, raw_path: str) -> Path:
        try:
            return self.dialogue_store.resolve_image_path(dataset, raw_path)
        except FileNotFoundError as original_error:
            if self.gvv_index is not None:
                record = self.gvv_index.record_for(raw_path)
                if record is not None:
                    try:
                        return self.dialogue_store.resolve_image_path(
                            dataset, record.relative_path
                        )
                    except FileNotFoundError:
                        pass
            raise original_error

    @staticmethod
    def _values(metadata: dict[str, Any], key: str) -> list[str]:
        value = metadata.get(key, [])
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if str(item)]
        return [str(value)] if value else []


def make_policy_observation(
    query_embedding: Sequence[float] | np.ndarray,
    memory_hits: Sequence[MemoryHit],
    category: str,
    visual_categories: set[str] | frozenset[str] | None = None,
    evidence_availability_mask: Sequence[Sequence[bool]] | torch.Tensor | None = None,
) -> PolicyObservation:
    import torch

    if not memory_hits:
        raise ValueError("Policy observation requires at least one retrieved MemoryEpisode")
    allowed_visual = {
        str(value).upper()
        for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
    }
    category_allows_images = category.upper() in allowed_visual
    if evidence_availability_mask is None:
        evidence_availability_mask = [
            [
                bool(hit.item.summary.strip()),
                len(hit.item.metadata.get("source_dialogue_ids", [])) == 1,
                bool(hit.item.metadata.get("image_captions")),
                bool(hit.item.metadata.get("image_paths")) and category_allows_images,
                False,
            ]
            for hit in memory_hits
        ]
    observation = PolicyObservation(
        query_embedding=torch.as_tensor(np.asarray(query_embedding), dtype=torch.float32),
        summary_embeddings=torch.as_tensor(
            np.stack([hit.item.embedding for hit in memory_hits]), dtype=torch.float32
        ),
        memory_ids=tuple(hit.item.id for hit in memory_hits),
        evidence_availability_mask=torch.as_tensor(
            evidence_availability_mask, dtype=torch.bool
        ),
    )
    observation.validate()
    return observation


def choose_full_evidence_actions(
    memory_hits: Sequence[MemoryHit],
    category: str,
    visual_categories: set[str] | frozenset[str] | None = None,
    evidence_availability_mask: Sequence[Sequence[bool]] | torch.Tensor | None = None,
) -> tuple[MemoryEvidenceAction, ...]:
    """Select every available evidence type for every retrieved MemoryEpisode."""
    import torch

    if not memory_hits:
        raise ValueError("Evidence selection requires at least one retrieved MemoryEpisode")
    if evidence_availability_mask is None:
        allowed_visual = {
            str(value).upper()
            for value in (visual_categories or MEMGALLERY_VISUAL_CATEGORIES)
        }
        evidence_availability_mask = [
            [
                bool(hit.item.summary.strip()),
                len(hit.item.metadata.get("source_dialogue_ids", [])) == 1,
                bool(hit.item.metadata.get("image_captions")),
                bool(hit.item.metadata.get("image_paths"))
                and category.upper() in allowed_visual,
                False,
            ]
            for hit in memory_hits
        ]
    rows = torch.as_tensor(evidence_availability_mask, dtype=torch.bool)
    if rows.shape != (len(memory_hits), len(EVIDENCE_ORDER)):
        raise ValueError(
            "Evidence availability must have shape "
            f"({len(memory_hits)}, {len(EVIDENCE_ORDER)}), got {tuple(rows.shape)}"
        )
    return tuple(
        MemoryEvidenceAction.from_mask(hit.item.id, available)
        for hit, available in zip(memory_hits, rows.tolist())
    )


def action_signature(actions: Sequence[MemoryEvidenceAction]) -> str:
    return "|".join(f"{action.memory_id}:{action.bitmask}" for action in actions)
