from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any


DEFAULT_EXCLUDED_QUESTION_TYPES = "Answer Refusal"
ENV_NAME = "H2HMEM_EXCLUDED_QUESTION_TYPES"


def parse_excluded_question_types(
    value: str | Iterable[str] | None = None,
) -> frozenset[str]:
    if value is None:
        value = os.getenv(ENV_NAME, DEFAULT_EXCLUDED_QUESTION_TYPES)
    values = value.split(",") if isinstance(value, str) else value
    return frozenset(
        str(item).strip().casefold()
        for item in values
        if str(item).strip()
    )


def question_category(question: dict[str, Any]) -> str:
    question_type = question.get("question_type", {})
    if isinstance(question_type, dict):
        return str(
            question_type.get("subsub_type")
            or question_type.get("sub_type")
            or question_type.get("main_type")
            or ""
        )
    return str(question_type or "")


def is_excluded_question(
    question: dict[str, Any],
    excluded_question_types: str | Iterable[str] | None = None,
) -> bool:
    return question_category(question).strip().casefold() in parse_excluded_question_types(
        excluded_question_types
    )


def filter_questions(
    questions: Iterable[dict[str, Any]],
    excluded_question_types: str | Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = parse_excluded_question_types(excluded_question_types)
    return [
        question
        for question in questions
        if not is_excluded_question(question, excluded)
    ]


def is_excluded_category(
    category: object,
    excluded_question_types: str | Iterable[str] | None = None,
) -> bool:
    return str(category or "").strip().casefold() in parse_excluded_question_types(
        excluded_question_types
    )
