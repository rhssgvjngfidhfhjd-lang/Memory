"""Map MEMLENS question-specific haystacks to the shared Chunk protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from benchmarks.multimodal_dataset_harness.runner import HarnessQuestion, HarnessSample
from embedding.chunk_builder import Chunk, compact_text, estimate_tokens


def _image_path(root: Path, image: dict[str, Any]) -> str:
    raw = str(image.get("file") or "").strip()
    return str((root / "release_images" / raw).resolve()) if raw else ""


def _pairs(turns: list[dict[str, Any]]) -> Iterable[list[dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    for turn in turns:
        role = str(turn.get("role") or "").casefold()
        if role == "user" and pending:
            yield pending
            pending = []
        pending.append(turn)
        if role == "assistant":
            yield pending
            pending = []
    if pending:
        yield pending


def build_memlens_sample(row: dict[str, Any], root: Path, source_path: Path) -> HarnessSample:
    question_id = str(row.get("question_id") or "").strip()
    if not question_id:
        raise ValueError("MEMLENS row has no question_id")
    sessions = list(row.get("haystack_sessions") or [])
    session_ids = [str(value) for value in row.get("haystack_session_ids") or []]
    dates = [str(value) for value in row.get("haystack_dates") or []]
    if not (len(sessions) == len(session_ids) == len(dates)):
        raise ValueError(f"unaligned MEMLENS session arrays for {question_id}")
    chunks: list[Chunk] = []
    evidence_ids: list[str] = []
    for session_index, (session_id, date, turns) in enumerate(
        zip(session_ids, dates, sessions), start=1
    ):
        for round_index, pair in enumerate(_pairs(list(turns or [])), start=1):
            dialogue_id = f"{session_id}:R{round_index:04d}"
            text_lines = [
                f"session: {session_id}",
                f"date: {date}",
                f"round: {dialogue_id}",
            ]
            images: list[str] = []
            image_ids: list[str] = []
            captions: list[str] = []
            structured_turns: list[dict[str, Any]] = []
            has_answer = False
            for turn_index, turn in enumerate(pair, start=1):
                role = str(turn.get("role") or "speaker")
                turn_text = compact_text(turn.get("content"))
                text_lines.append(f"{role}: {turn_text}")
                turn_images: list[str] = []
                turn_image_ids: list[str] = []
                turn_captions: list[str] = []
                for image in turn.get("images") or []:
                    if not isinstance(image, dict):
                        continue
                    path = _image_path(root, image)
                    image_id = Path(str(image.get("file") or "")).name
                    caption = str(image.get("blip_caption") or "").strip()
                    if path:
                        images.append(path)
                        turn_images.append(path)
                    if image_id:
                        image_ids.append(image_id)
                        turn_image_ids.append(image_id)
                        text_lines.append(f"image_id: {image_id}")
                    if caption:
                        captions.append(caption)
                        turn_captions.append(caption)
                        text_lines.append(f"image_caption: {compact_text(caption)}")
                structured_turns.append(
                    {
                        "turn_id": f"{question_id}:{dialogue_id}:T{turn_index:02d}",
                        "source_dialogue_id": dialogue_id,
                        "role": role,
                        "speaker": role,
                        "text": turn_text,
                        "timestamp": date,
                        "images": list(dict.fromkeys(turn_images)),
                        "image_ids": list(dict.fromkeys(turn_image_ids)),
                        "image_captions": turn_captions,
                    }
                )
                has_answer = has_answer or bool(turn.get("has_answer"))
            if has_answer:
                evidence_ids.append(dialogue_id)
            text = "\n".join(text_lines)
            chunks.append(
                Chunk(
                    chunk_id=f"{question_id}:{dialogue_id}",
                    text=text,
                    images=list(dict.fromkeys(images)),
                    metadata={
                        "dataset": question_id,
                        "session_id": session_id,
                        "session_index": session_index,
                        "date": date,
                        "timestamp": date,
                        "round_id": round_index,
                        "dialogue_id": dialogue_id,
                        "image_id": image_ids[0] if image_ids else "",
                        "image_ids": list(dict.fromkeys(image_ids)),
                        "image_caption": captions[0] if captions else "",
                        "image_captions": captions,
                        "has_image": bool(images),
                        "token_estimate": estimate_tokens(text),
                        "source_format": "memlens",
                        # Keep the benchmark-native turns available to memory
                        # systems such as M3 without parsing rendered text.
                        "m2a_turns": structured_turns,
                        "source_dialogue_ids": [dialogue_id],
                    },
                )
            )
    if not evidence_ids:
        answer_sessions = {str(value) for value in row.get("answer_session_ids") or []}
        evidence_ids = [
            str(chunk.metadata["dialogue_id"])
            for chunk in chunks
            if str(chunk.metadata.get("session_id") or "") in answer_sessions
        ]
    question = HarnessQuestion(
        question_id=question_id,
        question=str(row.get("question") or ""),
        answer=str(row.get("answer") or ""),
        category=str(row.get("question_type") or ""),
        clue_ids=list(dict.fromkeys(evidence_ids)),
        session_ids=[str(value) for value in row.get("answer_session_ids") or []],
        metadata={
            "question_date": str(row.get("question_date") or ""),
            "question_subtype": str(row.get("question_subtype") or ""),
            "answer_session_ids": [str(value) for value in row.get("answer_session_ids") or []],
        },
    )
    return HarnessSample(
        sample_id=question_id,
        source_name=source_path.stem,
        source_path=source_path,
        chunks=chunks,
        questions=[question],
    )


def load_memlens_samples(root: Path, dataset_file: str = "dataset_32k.json") -> list[HarnessSample]:
    source_path = Path(dataset_file)
    if not source_path.is_absolute():
        source_path = root / source_path
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("MEMLENS dataset JSON must contain a list")
    return [build_memlens_sample(row, root, source_path) for row in payload]
