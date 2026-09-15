from __future__ import annotations

import json
from typing import Any


def normalize_evidence_ranges(value: Any) -> list[list[int]]:
    """Canonicalize M2A evidence references without dropping raw message IDs."""
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"evidence_ids must be a list, got {type(value).__name__}")

    # Some OpenAI-compatible models emit [1, 3, 5] or [[1, 3, 5]] when they
    # mean explicit raw-message IDs. Treat those as singleton intervals. A
    # two-item list retains the official M2A [start, end] range semantics.
    entries: list[Any]
    if value and all(_is_integer(item) for item in value):
        entries = [[item] for item in value]
    else:
        entries = list(value)

    intervals: list[tuple[int, int]] = []
    for entry in entries:
        if _is_integer(entry):
            entry = [entry]
        if not isinstance(entry, (list, tuple)) or not entry:
            raise ValueError(f"invalid M2A evidence range: {entry!r}")
        numbers = [_positive_integer(item) for item in entry]
        if len(numbers) == 2:
            start, end = numbers
            if end < start:
                raise ValueError(f"invalid M2A evidence range: {entry!r}")
            intervals.append((start, end))
        elif len(numbers) == 1:
            intervals.append((numbers[0], numbers[0]))
        else:
            intervals.extend((number, number) for number in numbers)

    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_integer(value: Any) -> int:
    if not _is_integer(value) or value < 1:
        raise ValueError(f"invalid M2A evidence ID: {value!r}")
    return value
