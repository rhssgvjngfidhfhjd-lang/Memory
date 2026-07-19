from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Chunk:
    """One Mem-Gallery dialogue-round chunk."""

    chunk_id: str
    text: str
    images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(data["chunk_id"]),
            text=str(data.get("text", "")),
            images=[str(p) for p in data.get("images", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SearchHit:
    """A retrieved chunk plus its FAISS score."""

    chunk: Chunk
    score: float
    rank: int

    def to_context_item(self) -> dict[str, Any]:
        """Return the multimodal memory format expected by Mem-Gallery's runner."""
        md = self.chunk.metadata
        image = None
        if self.chunk.images:
            image = {
                "path": self.chunk.images[0],
                "img_id": md.get("image_id", ""),
                "caption": md.get("image_caption", ""),
            }
        return {
            "text": self.chunk.text,
            "image": image,
            "chunk_id": self.chunk.chunk_id,
            "score": self.score,
            "metadata": md,
        }
