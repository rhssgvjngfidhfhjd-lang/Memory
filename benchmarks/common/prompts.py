"""Shared answer contracts and dataset-specific benchmark QA prompts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


ANSWER_TAG_CONTRACT = (
    "Return only one non-empty <answer>...</answer> block, with the answer text inside the tags."
)
PPO_EMPTY_PROMPT_VERSION = "ppo-empty-evidence-20260911-v1"
PPO_EMPTY_EVIDENCE_INSTRUCTION = (
    "No conversation-memory evidence was selected. Answer using only the question and its "
    "Question Image, if present; do not invent missing memory facts."
)


def validate_evidence(
    memory_evidence: Sequence[str], *, allow_empty_evidence: bool = False
) -> list[str]:
    if isinstance(memory_evidence, (str, bytes)):
        raise ValueError("memory_evidence must be a sequence of evidence strings.")
    evidence = [str(item).strip() for item in memory_evidence if str(item).strip()]
    if not evidence and not allow_empty_evidence:
        raise ValueError(
            "Direct answer generation requires at least one non-empty memory evidence item."
        )
    return evidence


def format_query_images(raw_images: Any) -> str:
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


class AnswerFormatError(ValueError):
    """Raised when a model response violates the shared answer-tag contract."""


ANSWER_BLOCK_REGEX = r"<answer>[^<]+</answer>"
# Used only after a response exhausts the 512-token budget before closing its
# tag. Keeping the common path unbounded avoids the large performance penalty
# of bounded regex decoding; the rare retry is capped without changing Prompt.
ANSWER_BLOCK_RETRY_REGEX = r"<answer>[^<]{1,800}</answer>"


_ANSWER_BLOCK = re.compile(r"\s*<answer>(.*?)</answer>\s*", re.DOTALL)
_EMBEDDED_ANSWER_BLOCK = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_answer_block(raw: str) -> str:
    """Extract exactly one non-empty ``<answer>...</answer>`` block."""
    text = str(raw or "")
    match = _ANSWER_BLOCK.fullmatch(text)
    if match is None:
        raise AnswerFormatError(
            "Response must contain only one <answer>...</answer> block"
        )
    answer = match.group(1).strip()
    if not answer or "<answer>" in answer or "</answer>" in answer:
        raise AnswerFormatError("The <answer> block must be unique and non-empty")
    return answer


def recover_unique_answer_block(raw: str) -> str:
    """Recover one valid answer block while preserving strict normal parsing."""

    matches = _EMBEDDED_ANSWER_BLOCK.findall(str(raw or ""))
    if len(matches) != 1:
        raise AnswerFormatError("Response does not contain exactly one recoverable answer block")
    answer = matches[0].strip()
    if not answer or "<answer>" in answer or "</answer>" in answer:
        raise AnswerFormatError("The recoverable <answer> block must be unique and non-empty")
    return answer


ZERO_HIT_PROMPT_MARKER = "Retrieved memory evidence count: 0."


def evidence_with_zero_hit_marker(
    evidence: Sequence[str],
) -> tuple[list[str], bool]:
    """Add a prompt-only marker without changing memory, provenance, or Top-K."""

    normalized = [str(item).strip() for item in evidence if str(item).strip()]
    if normalized:
        return normalized, False
    return [ZERO_HIT_PROMPT_MARKER], True


# Source IDs are stable protocol identifiers; retain them across file moves so
# saved prompt hashes and evaluation checkpoints remain compatible.
MEMGALLERY_PROMPT_VERSION = "answer-prompts-custom-20260909-v1"
MEMGALLERY_PROMPT_SOURCE = "answer_prompts.py:build_benchmark_answer_messages[mem_gallery]"
MEMGALLERY_TASK_RULES = (
    "Answer from the provided multimodal conversation memory. Ground the answer in that memory, prefer "
    "the latest information when details change over time, and keep the response concise but complete. "
    "When the question asks for images, preserve the exact image_id values."
)
MEMGALLERY_SYSTEM_PROMPT = f"{MEMGALLERY_TASK_RULES} {ANSWER_TAG_CONTRACT}"


def build_memgallery_answer_messages(
    *,
    question: str,
    question_type: str,
    memory_evidence: Sequence[str],
    query_images: Any = None,
    allow_empty_evidence: bool = False,
) -> list[dict[str, str]]:
    evidence = validate_evidence(
        memory_evidence, allow_empty_evidence=allow_empty_evidence
    )
    question = str(question or "").strip()
    if not question:
        raise ValueError("Direct answer generation requires sample_metadata.question.")
    sections = []
    system_prompt = MEMGALLERY_SYSTEM_PROMPT
    if evidence:
        evidence_text = "\n\n".join(
            f"[Evidence {index}]\n{item}"
            for index, item in enumerate(evidence, start=1)
        )
        sections.append(f"Conversation memory:\n{evidence_text}")
    else:
        system_prompt = f"{PPO_EMPTY_EVIDENCE_INSTRUCTION} {ANSWER_TAG_CONTRACT}"
    image_context = format_query_images(query_images)
    if image_context:
        sections.append("Question Image:\n" + image_context)
    sections.append(f"Question: {question}")
    normalized_type = str(question_type or "").strip().upper()
    if normalized_type == "CD":
        sections.append(
            'The text inside <answer>...</answer> must be exactly "Yes." or "No."'
        )
    elif normalized_type == "VS":
        sections.append(
            "Return the exact matching image_id value or values. If several apply, sort them in ascending "
            "order and separate them with commas."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def parse_answer_response(raw: str) -> str:
    return parse_answer_block(raw)


def memgallery_prompt_sha256() -> str:
    source = json.dumps(
        {
            "version": MEMGALLERY_PROMPT_VERSION,
            "source": MEMGALLERY_PROMPT_SOURCE,
            "system": MEMGALLERY_SYSTEM_PROMPT,
            "cd": 'The text inside <answer>...</answer> must be exactly "Yes." or "No."',
            "vs": (
                "Return the exact matching image_id value or values. If several apply, sort them in ascending "
                "order and separate them with commas."
            ),
            "evidence_heading": "Conversation memory",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def memgallery_ppo_empty_prompt_sha256() -> str:
    source = json.dumps(
        {
            "version": PPO_EMPTY_PROMPT_VERSION,
            "base_prompt_sha256": memgallery_prompt_sha256(),
            "system": f"{PPO_EMPTY_EVIDENCE_INSTRUCTION} {ANSWER_TAG_CONTRACT}",
            "evidence_section": "omitted",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def memgallery_prompt_manifest() -> dict[str, object]:
    return {
        "prompt_version": MEMGALLERY_PROMPT_VERSION,
        "prompt_source": MEMGALLERY_PROMPT_SOURCE,
        "prompt_sha256": memgallery_prompt_sha256(),
    }


def resolve_question_image(data_dir: Path, qa: dict) -> dict | None:
    raw = qa.get("question_image")
    if not raw:
        return None
    if os.path.isabs(raw):
        path = raw
    elif raw.startswith("../image/"):
        path = str((data_dir / "image" / raw.replace("../image/", "")).resolve())
    else:
        path = str((data_dir / "image" / raw).resolve())
    out = {"path": path, "img_id": str(raw)}
    if qa.get("image_caption"):
        out["caption"] = qa["image_caption"]
    return out


H2HMEM_PROMPT_VERSION = "answer-prompts-custom-20260909-v1"
H2HMEM_PROMPT_SOURCE = "answer_prompts.py:build_benchmark_answer_messages[h2hmem]"
H2HMEM_INSTRUCTIONS = {
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

H2HMEM_QUESTION_TYPE_ALIASES = {
    "multimodal causal inference": "multimodal causal reasoning",
}


def build_h2hmem_answer_messages(
    *,
    question: str,
    question_type: str,
    memory_evidence: Sequence[str],
    query_images: Any = None,
    allow_empty_evidence: bool = False,
) -> list[dict[str, str]]:
    evidence = validate_evidence(
        memory_evidence, allow_empty_evidence=allow_empty_evidence
    )
    question = str(question or "").strip()
    if not question:
        raise ValueError("Direct answer generation requires sample_metadata.question.")
    normalized_type = str(question_type or "").strip().casefold()
    normalized_type = H2HMEM_QUESTION_TYPE_ALIASES.get(normalized_type, normalized_type)
    type_instruction = H2HMEM_INSTRUCTIONS.get(
        normalized_type, H2HMEM_INSTRUCTIONS["unimodal precise recall"]
    )
    task_instruction = type_instruction if evidence else PPO_EMPTY_EVIDENCE_INSTRUCTION
    task_rules = (
        f"You are a memory testing system. {task_instruction} Answer directly in English without reasoning, "
        "using no more than 100 words."
    )
    if normalized_type == "conflict detection":
        task_rules += " The answer text must be strictly either Yes or No."
    elif normalized_type == "answer refusal":
        task_rules += ' If the information is absent, the answer text must be exactly "Not mentioned."'
    system_prompt = f"{task_rules} {ANSWER_TAG_CONTRACT}"
    sections = []
    if evidence:
        evidence_text = "\n\n".join(
            f"[Evidence {index}]\n{item}"
            for index, item in enumerate(evidence, start=1)
        )
        sections.append(f"Conversation memory:\n{evidence_text}")
    image_context = format_query_images(query_images)
    if image_context:
        sections.append("Question Image:\n" + image_context)
    sections.append(f"Question: {question}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]






def h2hmem_prompt_sha256() -> str:
    source = json.dumps(
        {
            "version": H2HMEM_PROMPT_VERSION,
            "source": H2HMEM_PROMPT_SOURCE,
            "answer_tag_contract": ANSWER_TAG_CONTRACT,
            "instructions": H2HMEM_INSTRUCTIONS,
            "question_type_aliases": H2HMEM_QUESTION_TYPE_ALIASES,
            "evidence_heading": "Conversation memory",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def h2hmem_ppo_empty_prompt_sha256() -> str:
    source = json.dumps(
        {
            "version": PPO_EMPTY_PROMPT_VERSION,
            "base_prompt_sha256": h2hmem_prompt_sha256(),
            "empty_instruction": PPO_EMPTY_EVIDENCE_INSTRUCTION,
            "answer_tag_contract": ANSWER_TAG_CONTRACT,
            "evidence_section": "omitted",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


WMA_PROMPT_VERSION = "answer-prompts-custom-20260909-v1"
WMA_PROMPT_SOURCE = "answer_prompts.py:build_benchmark_answer_messages[worldmemarena]"
WMA_TASK_RULES = (
    "Answer from the provided retrieved memories. Synthesize multiple entries when needed, use timestamps "
    "and the most recent evidence to resolve conflicts, and ground every claim in the memories. Convert "
    "relative dates using the memory timestamps. When images are requested, preserve exact image_id "
    "values. Keep the answer concise; if no relevant information is present, the answer text must be "
    'exactly "Not mentioned in memory."'
)
WMA_SYSTEM_PROMPT = f"{WMA_TASK_RULES} {ANSWER_TAG_CONTRACT}"


def build_wma_answer_messages(
    *,
    question: str,
    memory_evidence: Sequence[str],
    query_images: Any = None,
    question_type: str = "",
    allow_empty_evidence: bool = False,
) -> list[dict[str, str]]:
    del question_type
    evidence = validate_evidence(
        memory_evidence, allow_empty_evidence=allow_empty_evidence
    )
    question = str(question or "").strip()
    if not question:
        raise ValueError("Direct answer generation requires sample_metadata.question.")
    sections = []
    system_prompt = WMA_SYSTEM_PROMPT
    if evidence:
        evidence_text = "\n\n".join(
            f"[Evidence {index}]\n{item}"
            for index, item in enumerate(evidence, start=1)
        )
        sections.append(f"Retrieved memories:\n{evidence_text}")
    else:
        system_prompt = f"{PPO_EMPTY_EVIDENCE_INSTRUCTION} {ANSWER_TAG_CONTRACT}"
    image_context = format_query_images(query_images)
    if image_context:
        sections.append("Question Image:\n" + image_context)
    sections.append(f"Question: {question}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n\n".join(sections)},
    ]






def wma_prompt_sha256() -> str:
    source = json.dumps(
        {
            "version": WMA_PROMPT_VERSION,
            "source": WMA_PROMPT_SOURCE,
            "system": WMA_SYSTEM_PROMPT,
            "evidence_heading": "Retrieved memories",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def wma_ppo_empty_prompt_sha256() -> str:
    source = json.dumps(
        {
            "version": PPO_EMPTY_PROMPT_VERSION,
            "base_prompt_sha256": wma_prompt_sha256(),
            "system": f"{PPO_EMPTY_EVIDENCE_INSTRUCTION} {ANSWER_TAG_CONTRACT}",
            "evidence_section": "omitted",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
