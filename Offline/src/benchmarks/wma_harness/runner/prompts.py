"""WorldMemArena QA prompt matching the repository-root answer_prompts.py."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from benchmarks.answer_response import parse_answer_block


PROMPT_VERSION = "answer-prompts-custom-20260909-v1"
PROMPT_SOURCE = "answer_prompts.py:build_benchmark_answer_messages[worldmemarena]"
ANSWER_TAG_CONTRACT = (
    "Return only one non-empty <answer>...</answer> block, with the answer text inside the tags."
)
TASK_RULES = (
    "Answer from the provided retrieved memories. Synthesize multiple entries when needed, use timestamps "
    "and the most recent evidence to resolve conflicts, and ground every claim in the memories. Convert "
    "relative dates using the memory timestamps. When images are requested, preserve exact image_id "
    "values. Keep the answer concise; if no relevant information is present, the answer text must be "
    'exactly "Not mentioned in memory."'
)
SYSTEM_PROMPT = f"{TASK_RULES} {ANSWER_TAG_CONTRACT}"


def build_answer_messages(
    *,
    question: str,
    memory_evidence: Sequence[str],
    query_images: Any = None,
    question_type: str = "",
) -> list[dict[str, str]]:
    del question_type
    evidence = _validated_evidence(memory_evidence)
    question = str(question or "").strip()
    if not question:
        raise ValueError("Direct answer generation requires sample_metadata.question.")
    evidence_text = "\n\n".join(
        f"[Evidence {index}]\n{item}" for index, item in enumerate(evidence, start=1)
    )
    sections = [f"Retrieved memories:\n{evidence_text}"]
    image_context = _format_query_images(query_images)
    if image_context:
        sections.append("Question Image:\n" + image_context)
    sections.append(f"Question: {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def parse_answer_response(raw: str) -> str:
    return parse_answer_block(raw)


def prompt_sha256() -> str:
    source = json.dumps(
        {
            "version": PROMPT_VERSION,
            "source": PROMPT_SOURCE,
            "system": SYSTEM_PROMPT,
            "evidence_heading": "Retrieved memories",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _validated_evidence(memory_evidence: Sequence[str]) -> list[str]:
    if isinstance(memory_evidence, (str, bytes)):
        raise ValueError("memory_evidence must be a sequence of evidence strings.")
    evidence = [str(item).strip() for item in memory_evidence if str(item).strip()]
    if not evidence:
        raise ValueError(
            "Direct answer generation requires at least one non-empty memory evidence item."
        )
    return evidence


def _format_query_images(raw_images: Any) -> str:
    if isinstance(raw_images, Mapping):
        images = [raw_images]
    elif isinstance(raw_images, Sequence) and not isinstance(raw_images, (str, bytes)):
        images = [image for image in raw_images if isinstance(image, Mapping)]
    else:
        images = []
    lines = []
    for image in images:
        image_id = " ".join(str(image.get("id") or "").split())
        caption = " ".join(str(image.get("caption") or "").split())
        if not image_id:
            continue
        lines.append(f"- image_id: {image_id}")
        if caption:
            lines.append(f"  image_caption: {caption}")
    return "\n".join(lines)
