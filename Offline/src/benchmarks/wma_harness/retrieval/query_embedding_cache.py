from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from benchmarks.memgallery_harness.retrieval.query_embedding_cache import QueryEmbeddingCache
from benchmarks.question_filter import is_excluded_category
from embedding.chunk_builder import iter_wma_sample_files


def session_ids(payload: dict[str, Any]) -> list[str]:
    return [
        str(session.get("_v2_session_id") or session.get("session_id") or "")
        for session in payload.get("sessions", []) or []
    ]


def visible_sessions_for_checkpoint(
    ordered_session_ids: list[str], covered_sessions: list[str] | set[str] | tuple[str, ...]
) -> list[str]:
    """Return the cumulative history visible when a WMA checkpoint fires.

    ``covered_sessions`` is the checkpoint's trigger/stage set, not a retrieval
    whitelist. The official runner keeps one cumulative adapter and triggers QA
    after the last covered session has completed.
    """
    covered = {str(value) for value in covered_sessions}
    if not covered:
        raise ValueError("WMA checkpoint has no covered_sessions")
    positions = {value: index for index, value in enumerate(ordered_session_ids)}
    missing = sorted(covered.difference(positions))
    if missing:
        raise ValueError(f"Unknown covered session ids: {missing}")
    trigger_index = max(positions[value] for value in covered)
    return ordered_session_ids[: trigger_index + 1]


def build_gold_evidence_map(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Map WMA memory/image ids to session and evaluator-only content."""
    result: dict[str, dict[str, str]] = {}
    for session in payload.get("sessions", []) or []:
        current_session = str(
            session.get("_v2_session_id") or session.get("session_id") or ""
        )
        for point in session.get("memory_points", []) or []:
            memory_id = str(point.get("memory_id", ""))
            if memory_id:
                result[memory_id] = {
                    "session_id": str(point.get("_session_id") or current_session),
                    "content": str(point.get("memory_content", "")),
                    "kind": "memory_id",
                }
        for turn in session.get("dialogue", []) or []:
            for attachment in turn.get("attachments", []) or []:
                image_id = str(attachment.get("image_id", ""))
                if image_id:
                    result[image_id] = {
                        "session_id": current_session,
                        "content": str(attachment.get("caption", "")),
                        "kind": "image_id",
                    }
    for block in payload.get("memory_points", []) or []:
        if not isinstance(block, dict):
            continue
        block_session = str(block.get("session_id", ""))
        for point in block.get("memory_points", []) or []:
            if not isinstance(point, dict):
                continue
            memory_id = str(point.get("memory_id", ""))
            if memory_id:
                result[memory_id] = {
                    "session_id": str(point.get("_session_id") or block_session),
                    "content": str(point.get("memory_content", "")),
                    "kind": "memory_id",
                }
    return result


def make_query_id(
    *, sample_id: str,
    checkpoint_id: str,
    qa_index: int,
    category: str,
    question: str,
) -> str:
    digest = hashlib.sha1(
        "\n".join([checkpoint_id, category, question]).encode("utf-8")
    ).hexdigest()[:16]
    return f"{sample_id}::{checkpoint_id}::{qa_index}::{category}::{digest}"


def iter_qa_items(
    data_dir: str | Path,
    *,
    sample_ids: set[str] | None = None,
    excluded_categories: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in iter_wma_sample_files(data_dir):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample_id = str(payload["sample_id"])
        if sample_ids is not None and sample_id not in sample_ids:
            continue
        ordered_sessions = session_ids(payload)
        for checkpoint in payload.get("qa_checkpoints", []) or []:
            checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
            covered_sessions = [str(value) for value in checkpoint.get("covered_sessions", [])]
            visible_sessions = visible_sessions_for_checkpoint(
                ordered_sessions, covered_sessions
            )
            for qa_index, qa in enumerate(checkpoint.get("questions", []) or [], start=1):
                category = str(qa.get("question_type_abbrev", ""))
                if is_excluded_category(category, excluded_categories):
                    continue
                question = str(qa.get("question", ""))
                items.append(
                    {
                        "query_id": make_query_id(
                            sample_id=sample_id,
                            checkpoint_id=checkpoint_id,
                            qa_index=qa_index,
                            category=category,
                            question=question,
                        ),
                        "dataset": sample_id,
                        "sample_id": sample_id,
                        "checkpoint_id": checkpoint_id,
                        "covered_sessions": covered_sessions,
                        "visible_sessions": visible_sessions,
                        "qa_index": qa_index,
                        "category": category,
                        "question": question,
                        "query_image": None,
                        "answer": qa.get("answer", ""),
                        "difficulty": qa.get("difficulty", ""),
                        "evidence": qa.get("evidence", []),
                    }
                )
    return items
