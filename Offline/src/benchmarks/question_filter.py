from __future__ import annotations

from collections.abc import Iterable


def parse_excluded_categories(
    value: str | Iterable[str] | None,
) -> frozenset[str]:
    """Return normalized QA category labels from CLI/config input."""
    if value is None:
        return frozenset()
    values = value.split(",") if isinstance(value, str) else value
    return frozenset(
        str(item).strip().casefold()
        for item in values
        if str(item).strip()
    )


def is_excluded_category(
    category: object,
    excluded_categories: str | Iterable[str] | None,
) -> bool:
    excluded = (
        excluded_categories
        if isinstance(excluded_categories, frozenset)
        else parse_excluded_categories(excluded_categories)
    )
    return str(category or "").strip().casefold() in excluded
