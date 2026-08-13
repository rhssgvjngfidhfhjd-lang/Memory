"""Mem-Gallery answer-prompt helpers (SYSTEM_PROMPT, question formatting,
question-image resolution), consumed by benchmarks.memgallery_harness.eval_memgallery.

History: this file was ``run_memgallery.py``, the chunk-RAG era's full
benchmark runner; the legacy runner was removed on 2026-08-06 and the module
renamed to ``prompts.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

PROMPT_DIR = Path("/data/haozhen/Memory/Mem-Gallery/benchmark/prompt")


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


SYSTEM_PROMPT = load_prompt("sys_prompt.txt")


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
    constraint = ""
    if category == "AR":
        constraint = load_prompt("ar_prompt.txt")
    elif category == "CD":
        constraint = load_prompt("cd_prompt.txt")
    elif category == "VS":
        constraint = load_prompt("vs_prompt.txt")
    if constraint:
        constraint = "\n\n" + constraint
    return (
        f"Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} "
        "in a concise manner with the help of memory content.\n"
        "Please only provide the content of the answer, without including introductory phrases like 'answer:'.\n"
        "For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.\n\n"
        f"The current question is as follows:\n{question}{constraint}"
    )
