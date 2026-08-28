from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    """One Mem-Gallery dialogue-round chunk."""

    chunk_id: str
    text: str
    images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data.get("text", "")),
            images=[str(p) for p in data.get("images", [])],
            metadata=dict(data.get("metadata", {})),
        )


def write_chunks_jsonl(chunks, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_chunks_jsonl(path):
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.from_dict(json.loads(line)))
    return chunks


def write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")





_SPACE_RE = re.compile(r"\s+")


def compact_text(text: Any) -> str:
    """Normalize whitespace while preserving readable text."""
    if text is None:
        return ""
    value = str(text).encode("utf-8", errors="replace").decode("utf-8")
    return _SPACE_RE.sub(" ", value).strip()


def estimate_tokens(text: str) -> int:
    """Cheap GPT-style estimate, good enough for chunk trimming."""
    return max(1, len(text) // 4)


def _profile_summary(profile: dict[str, Any], max_chars: int = 900) -> str:
    parts: list[str] = []
    name = compact_text(profile.get("name"))
    persona = compact_text(profile.get("persona_summary"))
    traits = profile.get("traits") or []
    style = compact_text(profile.get("conversation_style"))
    if name:
        parts.append(f"name: {name}")
    if persona:
        parts.append(f"persona: {persona}")
    if traits:
        parts.append("traits: " + ", ".join(str(t) for t in traits))
    if style:
        parts.append(f"style: {style}")
    summary = "; ".join(parts)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def _resolve_image_path(raw_path: str, data_dir: Path) -> str:
    if not raw_path:
        return ""
    p = Path(raw_path)
    if p.is_absolute():
        return str(p)
    if raw_path.startswith("../image/"):
        return str((data_dir / "image" / raw_path.replace("../image/", "")).resolve())
    return str((data_dir / raw_path).resolve())


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_round_number(round_id: str) -> int | None:
    if ":" not in round_id:
        return None
    try:
        return int(round_id.split(":", 1)[1])
    except ValueError:
        return None


def _trim_chunk_text(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    # Preserve line order and trim the longest content-bearing lines first.
    lines = text.splitlines()
    priority_prefixes = (
        "profile_summary:",
        "previous_round_summary:",
        "assistant:",
        "user:",
    )
    target_chars = max_tokens * 4
    while len("\n".join(lines)) > target_chars and lines:
        changed = False
        for prefix in priority_prefixes:
            for idx, line in enumerate(lines):
                if line.startswith(prefix) and len(line) > 240:
                    keep = max(160, int(len(line) * 0.75))
                    lines[idx] = line[:keep].rstrip() + "..."
                    changed = True
                    break
            if changed:
                break
        if not changed:
            joined = "\n".join(lines)
            return joined[: target_chars - 3].rstrip() + "..."
    return "\n".join(lines)


def _make_chunk_text(
    *,
    profile_summary: str,
    session_id: str,
    date: str,
    round_id: str,
    user_text: str,
    assistant_text: str,
    image_ids: list[str],
    captions: list[str],
    previous_round_summary: str,
    max_tokens: int,
    include_captions: bool = True,
) -> str:
    lines = [
        f"profile_summary: {profile_summary}",
        f"session: {session_id}",
        f"date: {date}",
        f"round: {round_id}",
        f"user: {compact_text(user_text)}",
        f"assistant: {compact_text(assistant_text)}",
    ]
    for image_id, caption in zip(image_ids, captions):
        if image_id:
            lines.append(f"image_id: {image_id}")
        if include_captions and caption:
            lines.append(f"image_caption: {compact_text(caption)}")
    if previous_round_summary:
        lines.append(f"previous_round_summary: {previous_round_summary}")
    return _trim_chunk_text("\n".join(line for line in lines if line.strip()), max_tokens)


def build_chunks_from_file(
    dialog_file: str | Path,
    data_dir: str | Path,
    *,
    max_tokens: int = 800,
    include_previous_summary: bool = True,
    include_captions: bool = True,
    include_images: bool = True,
) -> list[Chunk]:
    path = Path(dialog_file)
    with path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset_name = path.stem
    return build_chunks_from_data(
        dataset,
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_tokens=max_tokens,
        include_previous_summary=include_previous_summary,
        include_captions=include_captions,
        include_images=include_images,
    )


def build_chunks_from_data(
    dataset: dict[str, Any],
    data_dir: str | Path,
    dataset_name: str,
    *,
    max_tokens: int = 800,
    include_previous_summary: bool = True,
    include_captions: bool = True,
    include_images: bool = True,
) -> list[Chunk]:
    data_dir = Path(data_dir)
    profile = dataset.get("character_profile") or {}
    profile_summary = _profile_summary(profile)
    chunks: list[Chunk] = []

    for session in dataset.get("multi_session_dialogues", []) or []:
        session_id = str(session.get("session_id", ""))
        date = str(session.get("date", ""))
        previous_summary = ""
        for dialog in session.get("dialogues", []) or []:
            round_id = str(dialog.get("round", ""))
            user_text = str(dialog.get("user", "") or "")
            assistant_text = str(dialog.get("assistant", "") or "")
            raw_image_paths = [str(x) for x in _as_list(dialog.get("input_image")) if x]
            image_paths = [
                p for p in (_resolve_image_path(raw, data_dir) for raw in raw_image_paths) if p
            ]
            image_ids = [str(x) for x in _as_list(dialog.get("image_id")) if x]
            captions = [str(x) for x in _as_list(dialog.get("image_caption")) if x]
            # Keep zip stable when one side is missing.
            max_images = max(len(image_paths), len(image_ids), len(captions), 0)
            image_paths = image_paths + [""] * (max_images - len(image_paths))
            image_ids = image_ids + [""] * (max_images - len(image_ids))
            captions = captions + [""] * (max_images - len(captions))

            chunk_text = _make_chunk_text(
                profile_summary=profile_summary,
                session_id=session_id,
                date=date,
                round_id=round_id,
                user_text=user_text,
                assistant_text=assistant_text,
                image_ids=image_ids,
                captions=captions,
                previous_round_summary=previous_summary if include_previous_summary else "",
                max_tokens=max_tokens,
                include_captions=include_captions,
            )
            round_number = _parse_round_number(round_id)
            chunk_id = f"{dataset_name}:{round_id or len(chunks) + 1}"
            metadata = {
                "dataset": dataset_name,
                "profile_name": profile.get("name", ""),
                "session_id": session_id,
                "date": date,
                "round_id": round_number,
                "dialogue_id": round_id,
                "image_id": image_ids[0] if image_ids else "",
                "image_ids": [x for x in image_ids if x],
                "image_caption": captions[0] if include_captions and captions else "",
                "image_captions": [x for x in captions if x] if include_captions else [],
                "timestamp": date,
                "category": "",
                "has_image": any(image_paths),
                "token_estimate": estimate_tokens(chunk_text),
            }
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    images=[p for p in image_paths if p] if include_images else [],
                    metadata=metadata,
                )
            )
            previous_summary = _make_previous_summary(
                round_id,
                user_text,
                assistant_text,
                captions if include_captions else [],
            )
    return chunks


def _make_previous_summary(
    round_id: str,
    user_text: str,
    assistant_text: str,
    captions: Iterable[str],
    max_chars: int = 260,
) -> str:
    parts = [f"{round_id}:"] if round_id else []
    if user_text:
        parts.append("user " + compact_text(user_text))
    if assistant_text:
        parts.append("assistant " + compact_text(assistant_text))
    for caption in captions:
        if caption:
            parts.append("image_caption " + compact_text(caption))
            break
    out = "; ".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    return out


def iter_dialog_files(data_dir: str | Path) -> list[Path]:
    return sorted((Path(data_dir) / "dialog").glob("*.json"))


def build_chunks_from_directory(
    data_dir: str | Path,
    *,
    max_tokens: int = 800,
    include_previous_summary: bool = True,
    include_captions: bool = True,
    include_images: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for dialog_file in iter_dialog_files(data_dir):
        chunks.extend(
            build_chunks_from_file(
                dialog_file,
                data_dir=data_dir,
                max_tokens=max_tokens,
                include_previous_summary=include_previous_summary,
                include_captions=include_captions,
                include_images=include_images,
            )
        )
    return chunks


def iter_wma_sample_files(data_dir: str | Path) -> list[Path]:
    """Return lifelong WorldMemArena sample JSON files in a stable order."""
    root = Path(data_dir)
    search_root = root / "lifelong" if (root / "lifelong").is_dir() else root
    paths = list(search_root.rglob("*.json"))
    return sorted(paths, key=lambda item: (item.stem, str(item)))


def _wma_rounds(dialogue: list[dict[str, Any]]) -> Iterable[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Pair WMA user/assistant messages using the MemGallery round unit."""
    pending_user: dict[str, Any] | None = None
    round_number = 0
    for row in dialogue:
        role = str(row.get("role", "")).lower()
        if role == "user":
            if pending_user is not None:
                round_number += 1
                yield round_number, pending_user, {}
            pending_user = row
        elif role == "assistant":
            round_number += 1
            yield round_number, pending_user or {}, row
            pending_user = None
    if pending_user is not None:
        round_number += 1
        yield round_number, pending_user, {}


def _wma_attachments(*messages: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        for attachment in message.get("attachments", []) or []:
            if isinstance(attachment, dict) and attachment.get("file_path"):
                result.append(attachment)
    return result


def build_wma_chunks_from_data(
    sample: dict[str, Any],
    data_dir: str | Path,
    *,
    sample_path: str | Path | None = None,
    max_tokens: int = 800,
    include_previous_summary: bool = True,
    include_captions: bool = True,
    include_images: bool = True,
) -> list[Chunk]:
    """Convert one WMA sample to the same round-level ``Chunk`` schema.

    Gold ``memory_points`` and ``qa_checkpoints`` are intentionally ignored.
    """
    root = Path(data_dir)
    image_base_dir = Path(sample_path).parent if sample_path is not None else root
    sample_id = str(sample["sample_id"])
    chunks: list[Chunk] = []
    for session in sample.get("sessions", []) or []:
        session_id = str(session.get("_v2_session_id") or session.get("session_id") or "")
        previous_summary = ""
        for round_number, user_row, assistant_row in _wma_rounds(session.get("dialogue", []) or []):
            dialogue_id = f"{session_id}:R{round_number:04d}"
            attachments = _wma_attachments(user_row, assistant_row)
            image_ids = [str(row.get("image_id", "")) for row in attachments]
            captions = [str(row.get("caption", "")) for row in attachments]
            image_paths = [
                str(
                    (
                        Path(str(row["file_path"]))
                        if Path(str(row["file_path"])).is_absolute()
                        else image_base_dir / str(row["file_path"])
                    ).resolve()
                )
                for row in attachments
            ]
            timestamp = str(user_row.get("timestamp") or assistant_row.get("timestamp") or "")
            chunk_text = _make_chunk_text(
                profile_summary="",
                session_id=session_id,
                date=timestamp,
                round_id=dialogue_id,
                user_text=str(user_row.get("content", "") or ""),
                assistant_text=str(assistant_row.get("content", "") or ""),
                image_ids=image_ids,
                captions=captions,
                previous_round_summary=previous_summary if include_previous_summary else "",
                max_tokens=max_tokens,
                include_captions=include_captions,
            )
            metadata = {
                "dataset": sample_id,
                "profile_name": "",
                "session_id": session_id,
                "date": timestamp,
                "round_id": round_number,
                "dialogue_id": dialogue_id,
                "image_id": image_ids[0] if image_ids else "",
                "image_ids": image_ids,
                "image_caption": captions[0] if include_captions and captions else "",
                "image_captions": captions if include_captions else [],
                "timestamp": timestamp,
                "category": "",
                "has_image": bool(image_paths),
                "token_estimate": estimate_tokens(chunk_text),
            }
            chunks.append(
                Chunk(
                    chunk_id=f"{sample_id}:{dialogue_id}",
                    text=chunk_text,
                    images=image_paths if include_images else [],
                    metadata=metadata,
                )
            )
            previous_summary = _make_previous_summary(
                dialogue_id,
                str(user_row.get("content", "") or ""),
                str(assistant_row.get("content", "") or ""),
                captions if include_captions else [],
            )
    return chunks


def build_wma_chunks_from_directory(
    data_dir: str | Path,
    *,
    sample_ids: set[str] | None = None,
    max_tokens: int = 800,
    include_previous_summary: bool = True,
    include_captions: bool = True,
    include_images: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_wma_sample_files(data_dir):
        sample = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(sample["sample_id"])
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        chunks.extend(
            build_wma_chunks_from_data(
                sample,
                data_dir,
                sample_path=path,
                max_tokens=max_tokens,
                include_previous_summary=include_previous_summary,
                include_captions=include_captions,
                include_images=include_images,
            )
        )
    return chunks


_H2H_VARIANT_DIRS = {
    "dyadic": "dyadic",
    "multiparty": "multi-party",
}


def _natural_path_key(path: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    for part in path.parts:
        for value in re.split(r"(\d+)", part.lower()):
            parts.append(int(value) if value.isdigit() else value)
    return tuple(parts)


def iter_h2h_session_files(
    data_dir: str | Path,
    *,
    variant: str | None = None,
) -> list[Path]:
    """Return H2HMem session files in conversation/session numeric order."""
    root = Path(data_dir)
    if (root / "dataset").is_dir():
        root = root / "dataset"
    variants = [variant] if variant else list(_H2H_VARIANT_DIRS)
    invalid = [value for value in variants if value not in _H2H_VARIANT_DIRS]
    if invalid:
        raise ValueError(f"unknown H2HMem variant: {invalid[0]}")
    paths = [
        path
        for value in variants
        for path in (root / _H2H_VARIANT_DIRS[value]).glob(
            "*/scenes/session*/session.json"
        )
    ]
    return sorted(paths, key=_natural_path_key)


def _h2h_speaker_blocks(
    dialogue: list[dict[str, Any]],
    session_dir: Path,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for row in dialogue:
        speaker = compact_text(row.get("role"))
        content = row.get("content") or {}
        if not isinstance(content, dict):
            content = {"text": content}
        text = compact_text(content.get("text"))
        raw_images = [str(value) for value in _as_list(content.get("image")) if value]
        image_paths = [
            str(
                (
                    Path(raw)
                    if Path(raw).is_absolute()
                    else session_dir / "image" / raw
                ).resolve()
            )
            for raw in raw_images
        ]
        if blocks and blocks[-1]["speaker"] == speaker:
            if text:
                blocks[-1]["texts"].append(text)
            blocks[-1]["image_names"].extend(raw_images)
            blocks[-1]["image_paths"].extend(image_paths)
            continue
        blocks.append(
            {
                "speaker": speaker,
                "texts": [text] if text else [],
                "image_names": raw_images,
                "image_paths": image_paths,
            }
        )
    return blocks


def _make_h2h_chunk_text(
    *,
    variant: str,
    conversation_id: str,
    session_id: str,
    native_session_id: str,
    date: str,
    round_id: str,
    session_title: str,
    theme: str,
    first: dict[str, Any],
    second: dict[str, Any] | None,
    previous_round_summary: str,
    max_tokens: int,
) -> str:
    lines = [
        "benchmark: H2HMem",
        f"variant: {variant}",
        f"conversation: {conversation_id}",
        f"session: {session_id}",
        f"native_session: {native_session_id}",
        f"date: {date}",
        f"round: {round_id}",
    ]
    if session_title:
        lines.append(f"session_title: {session_title}")
    if theme:
        lines.append(f"theme: {theme}")
    for label, block in (("speaker_a", first), ("speaker_b", second)):
        if block is None:
            continue
        lines.append(f"{label}_name: {block['speaker']}")
        for text in block["texts"]:
            lines.append(f"{label}: {text}")
        for image_name in block["image_names"]:
            lines.append(f"{label}_image: {image_name}")
    if previous_round_summary:
        lines.append(f"previous_round_summary: {previous_round_summary}")
    return _trim_chunk_text("\n".join(lines), max_tokens)


def _h2h_previous_summary(
    round_id: str,
    first: dict[str, Any],
    second: dict[str, Any] | None,
    *,
    max_chars: int = 260,
) -> str:
    parts = [f"{round_id}:"]
    for block in (first, second):
        if block is None:
            continue
        text = " ".join(block["texts"])
        if text:
            parts.append(f"{block['speaker']} {text}")
    value = "; ".join(parts)
    return value if len(value) <= max_chars else value[: max_chars - 3].rstrip() + "..."


def build_h2h_chunks_from_data(
    session: dict[str, Any],
    *,
    session_path: str | Path,
    variant: str,
    conversation_id: str,
    max_tokens: int = 800,
    include_previous_summary: bool = True,
    include_images: bool = True,
) -> list[Chunk]:
    """Convert one H2HMem session into adjacent two-speaker exchange chunks."""
    if variant not in _H2H_VARIANT_DIRS:
        raise ValueError(f"unknown H2HMem variant: {variant}")
    path = Path(session_path)
    scene_id = path.parent.name
    native_session_id = str(session.get("session_id") or "")
    date = str(session.get("timeline_date") or "")
    blocks = _h2h_speaker_blocks(session.get("dialogue", []) or [], path.parent)
    chunks: list[Chunk] = []
    previous_summary = ""
    for offset in range(0, len(blocks), 2):
        first = blocks[offset]
        second = blocks[offset + 1] if offset + 1 < len(blocks) else None
        round_number = len(chunks) + 1
        round_id = f"{scene_id}:R{round_number:04d}"
        all_blocks = [block for block in (first, second) if block is not None]
        image_names = [name for block in all_blocks for name in block["image_names"]]
        image_paths = [value for block in all_blocks for value in block["image_paths"]]
        image_ids = [
            f"h2hmem:{variant}:{conversation_id}:{scene_id}:{name}"
            for name in image_names
        ]
        text = _make_h2h_chunk_text(
            variant=variant,
            conversation_id=conversation_id,
            session_id=scene_id,
            native_session_id=native_session_id,
            date=date,
            round_id=round_id,
            session_title=compact_text(session.get("session_title")),
            theme=compact_text(session.get("theme")),
            first=first,
            second=second,
            previous_round_summary=(
                previous_summary if include_previous_summary else ""
            ),
            max_tokens=max_tokens,
        )
        chunks.append(
            Chunk(
                chunk_id=f"h2hmem:{variant}:{conversation_id}:{round_id}",
                text=text,
                images=image_paths if include_images else [],
                metadata={
                    "benchmark": "h2hmem",
                    "variant": variant,
                    # This value becomes an on-disk HiveMem dataset directory.
                    # Keep it Windows-safe and distinguish dyadic/multiparty
                    # conversations that share names such as ``dialogue1``.
                    "dataset": f"{variant}_{conversation_id}",
                    "conversation_id": conversation_id,
                    "scene_id": scene_id,
                    "session_id": scene_id,
                    "native_session_id": native_session_id,
                    "dialogue_id": round_id,
                    "round_id": round_number,
                    "speaker_a": first["speaker"],
                    "speaker_b": second["speaker"] if second else "",
                    "date": date,
                    "timestamp": date,
                    "session_title": session.get("session_title") or "",
                    "theme": session.get("theme") or "",
                    "group_id": session.get("group_id") or "",
                    "image_id": image_ids[0] if image_ids else "",
                    "image_ids": image_ids,
                    "image_captions": [],
                    "has_image": bool(image_paths),
                    "token_estimate": estimate_tokens(text),
                },
            )
        )
        previous_summary = _h2h_previous_summary(round_id, first, second)
    return chunks


def build_h2h_chunks_from_directory(
    data_dir: str | Path,
    *,
    variant: str,
    conversation_ids: set[str] | None = None,
    max_tokens: int = 800,
    include_previous_summary: bool = True,
    include_images: bool = True,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_h2h_session_files(data_dir, variant=variant):
        conversation_id = path.parents[2].name
        if conversation_ids is not None and conversation_id not in conversation_ids:
            continue
        session = json.loads(path.read_text(encoding="utf-8"))
        chunks.extend(
            build_h2h_chunks_from_data(
                session,
                session_path=path,
                variant=variant,
                conversation_id=conversation_id,
                max_tokens=max_tokens,
                include_previous_summary=include_previous_summary,
                include_images=include_images,
            )
        )
    return chunks
