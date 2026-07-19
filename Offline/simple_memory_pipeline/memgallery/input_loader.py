from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MemoryEvent:
    text: str
    dataset: str
    dialogue_id: str
    session_id: str
    round_id: int
    source_chunk_id: str
    image_ids: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    image_captions: list[str] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "session_id": self.session_id,
            "round_id": self.round_id,
            "dialogue_id": self.dialogue_id,
            "source_dialogue_ids": [self.dialogue_id],
            "source_chunk_ids": [self.source_chunk_id],
            "image_id": self.image_ids[0] if self.image_ids else "",
            "image_ids": self.image_ids,
            "image_paths": self.image_paths,
            "image_captions": self.image_captions,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_events(path: str | Path, dataset: str | None = None) -> list[MemoryEvent]:
    events = []
    for row in read_jsonl(path):
        metadata = dict(row.get("metadata") or {})
        event_dataset = str(metadata.get("dataset", ""))
        if dataset and event_dataset != dataset:
            continue
        events.append(
            MemoryEvent(
                text=str(row.get("text", "")).strip(),
                dataset=event_dataset,
                dialogue_id=str(metadata.get("dialogue_id", "")),
                session_id=str(metadata.get("session_id", "")),
                round_id=int(metadata.get("round_id", 0)),
                source_chunk_id=str(row.get("chunk_id", "")),
                image_ids=_strings(metadata.get("image_ids") or [metadata.get("image_id", "")]),
                image_paths=_strings(row.get("images") or []),
                image_captions=_strings(
                    metadata.get("image_captions") or [metadata.get("image_caption", "")]
                ),
            )
        )
    events.sort(key=lambda item: (item.dataset, _session_key(item.session_id), item.round_id, item.source_chunk_id))
    return events


def audit_ablation_chunks(
    text_only_path: str | Path,
    text_caption_path: str | Path,
    multimodal_path: str | Path,
) -> dict[str, Any]:
    rows_by_mode = {
        "a": read_jsonl(text_only_path),
        "b": read_jsonl(text_caption_path),
        "c": read_jsonl(multimodal_path),
    }
    ids = {mode: [str(row.get("chunk_id", "")) for row in rows] for mode, rows in rows_by_mode.items()}
    reference = set(ids["a"])
    errors = []
    for mode in ("a", "b", "c"):
        if len(ids[mode]) != len(set(ids[mode])):
            errors.append(f"{mode}: duplicate chunk_id")
        if set(ids[mode]) != reference:
            errors.append(f"{mode}: chunk_id set differs from a")

    stats = {}
    for mode, rows in rows_by_mode.items():
        captions = sum(bool((row.get("metadata") or {}).get("image_caption")) for row in rows)
        images = sum(bool(row.get("images")) for row in rows)
        datasets = {str((row.get("metadata") or {}).get("dataset", "")) for row in rows}
        stats[mode] = {
            "count": len(rows),
            "datasets": len(datasets - {""}),
            "caption_rows": captions,
            "image_rows": images,
        }
    if stats["a"]["caption_rows"] or stats["a"]["image_rows"]:
        errors.append("a must not contain captions or image paths")
    if stats["b"]["image_rows"]:
        errors.append("b must not contain image paths")
    if stats["b"]["caption_rows"] == 0:
        errors.append("b contains no captions")
    if stats["c"]["caption_rows"] == 0 or stats["c"]["image_rows"] == 0:
        errors.append("c must contain captions and image paths")
    return {"ok": not errors, "errors": errors, "modes": stats}


def write_audit(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value and str(value)))


def _session_key(session_id: str) -> tuple[int, str]:
    digits = "".join(character for character in session_id if character.isdigit())
    return (int(digits) if digits else 0, session_id)
