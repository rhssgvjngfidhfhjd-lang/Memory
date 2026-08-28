from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


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
        self.file_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
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

    def question_ids(self, split: str, *, data_source: str | None = None) -> frozenset[str]:
        normalized = normalize_split_name(split)
        return frozenset(
            question_id
            for question_id, (current_source, current_split, _) in self._questions.items()
            if current_split == normalized
            and (data_source is None or current_source == data_source)
        )

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
