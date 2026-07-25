from __future__ import annotations

from qwenvl_embedding_2b_rag.schema import SearchHit


def format_hits_as_multimodal_items(hits: list[SearchHit]) -> list[dict]:
    return [hit.to_context_item() for hit in hits]


def format_hits_as_text_context(hits: list[SearchHit]) -> str:
    lines = []
    for idx, hit in enumerate(hits, start=1):
        md = hit.chunk.metadata
        header_parts = [
            f"SESSION:{md.get('session_id', '')}",
            f"ROUND:{md.get('dialogue_id', '')}",
            f"DATE:{md.get('date', '')}",
        ]
        if md.get("image_id"):
            header_parts.append(f"IMG:{md.get('image_id')}")
        lines.append(f"[{idx}] " + " | ".join(p for p in header_parts if p))
        lines.append(hit.chunk.text)
    return "\n\n".join(lines)
