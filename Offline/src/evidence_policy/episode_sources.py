from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .split_manifest import SplitConversation, SplitManifestIndex, normalize_split_name


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
    workspace_root: Path,
    conversation: SplitConversation,
) -> Iterator[SourceQuestion]:
    folder = "dyadic" if conversation.data_source == "h2hmem_dyadic" else "multi-party"
    variant = "dyadic" if folder == "dyadic" else "multiparty"
    scenes = (
        workspace_root
        / "H2HMEM-main"
        / "dataset"
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
            original_id = str(question.get("original_question_id") or "")
            local_id = original_id or f"Q{index + 1:03d}"
            question_id = (
                f"h2hmem:{variant}:{conversation.source_id}:{session.name}:{local_id}"
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
    workspace_root: Path,
    conversation: SplitConversation,
) -> Iterator[SourceQuestion]:
    path = (
        workspace_root
        / "Mem-Gallery"
        / "benchmark"
        / "data"
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


def _worldmemarena_paths(workspace_root: Path) -> dict[str, Path]:
    root = workspace_root / "WorldMemArena" / "WorldMemArena" / "lifelong"
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
    workspace_root: str | Path,
    *,
    split: str | None = None,
    data_sources: Iterable[str] | None = None,
) -> Iterator[SourceQuestion]:
    root = Path(workspace_root).expanduser().resolve()
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
                    rows = _iter_h2h_questions(root, conversation)
                elif source == "mem_gallery":
                    rows = _iter_mem_gallery_questions(root, conversation)
                elif source == "worldmemarena_lifelong":
                    if wma_paths is None:
                        wma_paths = _worldmemarena_paths(root)
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


def materialize_split_indexes(
    manifest: SplitManifestIndex,
    workspace_root: str | Path,
    output_dir: str | Path,
    *,
    data_sources: Iterable[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = manifest.summary()
    report["workspace_root"] = str(Path(workspace_root).expanduser().resolve())
    report["materialized"] = {}
    all_rows = list(
        iter_source_questions(
            manifest, workspace_root, data_sources=data_sources
        )
    )
    for split in ("train", "val", "test"):
        rows = [row for row in all_rows if row.split == split]
        path = output / f"{split}.jsonl"
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
        temporary.replace(path)
        report["materialized"][split] = {"path": str(path), "question_count": len(rows)}
    report_path = output / "audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize evidence-policy train/val/test indexes from source data"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-source", action="append", dest="data_sources")
    args = parser.parse_args()
    manifest = SplitManifestIndex(args.manifest)
    report = materialize_split_indexes(
        manifest,
        args.workspace_root,
        args.output,
        data_sources=args.data_sources,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
