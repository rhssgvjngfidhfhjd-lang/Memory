"""Map MemEye task JSONs to the shared Chunk and question protocols."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.multimodal_dataset_harness.runner import HarnessQuestion, HarnessSample
from embedding.chunk_builder import build_chunks_from_data


def _category(qa: dict[str, Any]) -> str:
    axes = []
    for group in qa.get("point") or []:
        if isinstance(group, list):
            axes.extend(str(value) for value in group if str(value))
        elif str(group):
            axes.append(str(group))
    return "/".join(axes)


def _query_image(image_root: Path, qa: dict[str, Any]) -> dict[str, str] | None:
    raw = str(qa.get("question_image") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = image_root / raw
    result = {"path": str(path.resolve()), "img_id": raw}
    if qa.get("image_caption"):
        result["caption"] = str(qa["image_caption"])
    return result


def load_memeye_samples(root: Path) -> list[HarnessSample]:
    # Accept both the MemEye repository root and its ``data`` directory.  The
    # CLI default points at ``data``, while an explicit path commonly points at
    # the newly cloned repository itself.
    if not (root / "dialog").is_dir() and (root / "data" / "dialog").is_dir():
        root = root / "data"
    dialog_root = root / "dialog"
    image_root = root / "image"
    paths = sorted(dialog_root.glob("*_Open.json"))
    if not paths:
        raise FileNotFoundError(f"no MemEye *_Open.json files under {dialog_root}")
    samples: list[HarnessSample] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample_id = path.stem.removesuffix("_Open")
        # MemEye image references are relative to data/image, unlike Mem-Gallery's
        # ../image-prefixed references. Passing image_root preserves source paths.
        chunks = build_chunks_from_data(payload, image_root, sample_id)
        questions = []
        for index, qa in enumerate(payload.get("human-annotated QAs") or [], start=1):
            raw_id = str(qa.get("question_id") or f"Q{index:03d}")
            questions.append(
                HarnessQuestion(
                    question_id=raw_id,
                    question=str(qa.get("question") or ""),
                    answer=str(qa.get("answer") or ""),
                    category=_category(qa),
                    clue_ids=[str(value) for value in qa.get("clue") or []],
                    session_ids=[str(value) for value in qa.get("session_id") or []],
                    query_image=_query_image(image_root, qa),
                    metadata={"point": qa.get("point") or []},
                )
            )
        samples.append(
            HarnessSample(
                sample_id=sample_id,
                source_name=path.stem,
                source_path=path,
                chunks=chunks,
                questions=questions,
            )
        )
    return samples
