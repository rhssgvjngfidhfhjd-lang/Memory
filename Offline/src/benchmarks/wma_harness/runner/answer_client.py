from __future__ import annotations

from typing import Any

from benchmarks.memgallery_harness.runner.answer_client import VLMAnswerClient as _BaseClient


# Agent-domain rows labelled question_type="unknown" use an empty abbreviation,
# including some questions whose gold evidence is an image.
MEMORY_IMAGE_CATEGORIES = frozenset({"VFR", "VS", "VU", "CMR", ""})


def build_retrieved_memory_context(
    memory_items: list[dict[str, Any]], category: str = ""
) -> tuple[str, list[str]]:
    lines = ["The retrieved memory contents are as follows:"]
    image_paths: list[str] = []
    include_images = category.upper() in MEMORY_IMAGE_CATEGORIES
    for rank, item in enumerate(memory_items, start=1):
        metadata = item.get("metadata", {}) or {}
        image = item.get("image")
        attached = include_images and isinstance(image, dict) and bool(image.get("path"))
        header = (
            f"[{rank}] SESSION:{metadata.get('session_id', '')} "
            f"ROUND:{metadata.get('dialogue_id', '')}"
        )
        if attached and metadata.get("image_id"):
            header += f" IMG:{metadata['image_id']}"
        text = str(item.get("text", ""))
        if not attached:
            for image_id in metadata.get("image_ids", []) or []:
                if image_id:
                    text = text.replace(str(image_id), "[IMAGE_ID_REDACTED]")
        lines.extend((header, text))
        if attached:
            image_paths.append(str(image["path"]))
            lines.append(
                f"Attached memory image {len(image_paths)}: {image.get('img_id', '')}"
            )
    return "\n\n".join(lines), image_paths


class VLMAnswerClient(_BaseClient):
    def _build_text_and_image_paths(
        self,
        memory_items: list[dict[str, Any]],
        question_prompt: str,
        query_image: dict[str, Any] | None,
        category: str = "",
    ) -> tuple[str, list[str]]:
        memory_text, image_paths = build_retrieved_memory_context(memory_items, category)
        lines = [memory_text, "", question_prompt]
        if query_image and query_image.get("path"):
            lines.append(f"Attached question image {len(image_paths) + 1}.")
            image_paths.append(str(query_image["path"]))
        return "\n\n".join(lines), image_paths

    @staticmethod
    def _include_memory_images(category: str) -> bool:
        return category.upper() in MEMORY_IMAGE_CATEGORIES
