"""H2HMem QA prompt matching the repository-root answer_prompts.py."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from benchmarks.answer_response import parse_answer_block


PROMPT_VERSION = "answer-prompts-custom-20260909-v1"
PROMPT_SOURCE = "answer_prompts.py:build_benchmark_answer_messages[h2hmem]"
ANSWER_TAG_CONTRACT = (
    "Return only one non-empty <answer>...</answer> block, with the answer text inside the tags."
)
INSTRUCTIONS = {
    "unimodal precise recall": "Accurately recall the requested information from the conversation memory.",
    "cross-modal related retrieval": "Use related textual and visual information from the conversation memory.",
    "knowledge resolution": "Resolve knowledge consistently across the conversation memory.",
    "temporal reasoning": "Reason about temporal relationships in the conversation memory.",
    "multimodal causal reasoning": "Use textual and visual evidence for the requested causal reasoning.",
    "reference & evolution tracking": "Track references and how they evolve across the conversation memory.",
    "test-time learning": "Use knowledge introduced in the conversation memory to answer the question.",
    "conflict detection": "Determine whether the information in the question conflicts with the conversation memory.",
    "answer refusal": "Determine whether the question can be answered from the conversation memory.",
}


def build_answer_messages(
    *,
    question: str,
    question_type: str,
    memory_evidence: Sequence[str],
    query_images: Any = None,
) -> list[dict[str, str]]:
    evidence = _validated_evidence(memory_evidence)
    question = str(question or "").strip()
    if not question:
        raise ValueError("Direct answer generation requires sample_metadata.question.")
    normalized_type = str(question_type or "").strip().casefold()
    type_instruction = INSTRUCTIONS.get(
        normalized_type, INSTRUCTIONS["unimodal precise recall"]
    )
    task_rules = (
        f"You are a memory testing system. {type_instruction} Answer directly in English without reasoning, "
        "using no more than 100 words."
    )
    if normalized_type == "conflict detection":
        task_rules += " The answer text must be strictly either Yes or No."
    elif normalized_type == "answer refusal":
        task_rules += ' If the information is absent, the answer text must be exactly "Not mentioned."'
    system_prompt = f"{task_rules} {ANSWER_TAG_CONTRACT}"
    evidence_text = "\n\n".join(
        f"[Evidence {index}]\n{item}" for index, item in enumerate(evidence, start=1)
    )
    sections = [f"Conversation memory:\n{evidence_text}"]
    image_context = _format_query_images(query_images)
    if image_context:
        sections.append("Question Image:\n" + image_context)
    sections.append(f"Question: {question}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def parse_answer_response(raw: str) -> str:
    return parse_answer_block(raw)


def prompt_sha256() -> str:
    source = json.dumps(
        {
            "version": PROMPT_VERSION,
            "source": PROMPT_SOURCE,
            "answer_tag_contract": ANSWER_TAG_CONTRACT,
            "instructions": INSTRUCTIONS,
            "evidence_heading": "Conversation memory",
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
