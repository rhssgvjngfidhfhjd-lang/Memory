"""Explicit prompt-layer handling for retrievals with no usable evidence."""

from __future__ import annotations

from collections.abc import Sequence


ZERO_HIT_PROMPT_MARKER = "Retrieved memory evidence count: 0."


def evidence_with_zero_hit_marker(
    evidence: Sequence[str],
) -> tuple[list[str], bool]:
    """Add a prompt-only marker without changing memory, provenance, or Top-K."""

    normalized = [str(item).strip() for item in evidence if str(item).strip()]
    if normalized:
        return normalized, False
    return [ZERO_HIT_PROMPT_MARKER], True
