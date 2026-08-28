"""Mem-Gallery answer-prompt helpers (SYSTEM_PROMPT, question formatting,
question-image resolution), consumed by benchmarks.memgallery_harness.eval_memgallery.

History: this file was ``run_memgallery.py``, the chunk-RAG era's full
benchmark runner; the legacy runner was removed on 2026-08-06 and the module
renamed to ``prompts.py``.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path

OFFLINE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROMPT_DIR = OFFLINE_ROOT.parent / "Mem-Gallery" / "benchmark" / "prompt"
PROMPT_DIR = Path(os.getenv("MEMGALLERY_PROMPT_DIR", DEFAULT_PROMPT_DIR)).expanduser().resolve()
REQUIRED_PROMPTS = {
    "system": "sys_prompt.txt",
    "AR": "ar_prompt.txt",
    "CD": "cd_prompt.txt",
    "VS": "vs_prompt.txt",
}


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Mem-Gallery prompt is missing: {path}. "
            "Set MEMGALLERY_PROMPT_DIR to override the prompt directory."
        )
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Required Mem-Gallery prompt is empty: {path}")
    return content


SYSTEM_PROMPT = load_prompt(REQUIRED_PROMPTS["system"])
CATEGORY_PROMPTS = {
    category: load_prompt(filename)
    for category, filename in REQUIRED_PROMPTS.items()
    if category != "system"
}


def prompt_manifest() -> dict[str, object]:
    """Return reproducibility metadata for the exact prompts used by QA."""
    return {
        "prompt_dir": str(PROMPT_DIR),
        "prompt_sha256": {
            filename: hashlib.sha256((PROMPT_DIR / filename).read_bytes()).hexdigest()
            for filename in REQUIRED_PROMPTS.values()
        },
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
    out = {"path": path}
    if qa.get("image_caption"):
        out["caption"] = qa["image_caption"]
    return out


def format_question_prompt(question: str, category: str, speaker_a: str, speaker_b: str) -> str:
    constraint = CATEGORY_PROMPTS.get(category, "")
    if constraint:
        constraint = "\n\n" + constraint
    return (
        f"Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} "
        "in a concise manner with the help of memory content.\n"
        "Please only provide the content of the answer, without including introductory phrases like 'answer:'.\n"
        "For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.\n\n"
        f"The current question is as follows:\n{question}{constraint}"
    )
