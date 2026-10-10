"""Shared question filtering and inference-service URL validation."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from urllib.parse import urlsplit


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


def require_service_url(
    parser: argparse.ArgumentParser, value: str | None, *, flag: str, env_name: str,
) -> str:
    url = str(value or '').strip()
    if not url:
        parser.error(f'{flag} is required; set {env_name}, provide {flag}, or configure the service URL in HIVE_CONFIG')
    try:
        parsed = urlsplit(url)
        parsed.port
    except ValueError:
        parser.error(f'{flag} must be an HTTP or HTTPS service URL')
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        parser.error(f'{flag} must be an HTTP or HTTPS service URL')
    return url.rstrip('/')
