from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from benchmarks.baseline_runtime.provenance import ProvenanceIndex
from embedding.chunk_builder import Chunk


class OmniSimpleMemAdapter(BaselineAdapter):
    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root
        self.config = dict(config)
        self.backend: Any = None
        self.provenance = ProvenanceIndex()
        self._current_session = ""
        self._reuse_existing_state = False
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    def _build_config(self) -> Any:
        from omni_memory import OmniMemoryConfig

        config = OmniMemoryConfig.create_default()
        config.embedding.model_name = str(self.config["embedding_model"])
        config.embedding.embedding_dim = int(self.config["embedding_dim"])
        config.embedding.visual_embedding_model = str(self.config["embedding_model"])
        config.embedding.visual_embedding_dim = int(self.config["embedding_dim"])
        config.embedding.api_base_url = str(self.config["embedding_base_url"])
        config.embedding.api_key = (
            os.getenv(str(self.config.get("embedding_api_key_env") or "")) or "EMPTY"
        )
        config.embedding.remote = True
        config.llm.model = str(self.config["executor_model"])
        config.llm.api_base_url = str(self.config["executor_base_url"])
        config.llm.api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        config.llm.temperature = float(self.config.get("executor_temperature") or 0.0)
        # OmniSimpleMem's entity extractor owns additional executor calls; use
        # the same retry budget as the unified benchmark configuration.
        config.llm.retries = int(self.config.get("retries") or 0)
        # The benchmark adapter consumes only ``RetrievalResult.items``.  The
        # upstream graph traversal populates ``graph_entities`` but never
        # contributes those entities to ``items``, so enabling it here adds an
        # executor call and potentially unbounded graph work without changing
        # the evidence sent to the answer model.  Keep an explicit opt-in for
        # experiments that inspect the native graph payload directly.
        config.retrieval.enable_graph_traversal = bool(
            self.config.get("omni_enable_graph_traversal", False)
        )
        return config

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del sample_id
        if self.backend is not None:
            self.backend.close()
        from omni_memory import OmniMemoryOrchestrator

        mau_store_dir = state_dir / "index" / "mau_store"
        self._reuse_existing_state = (
            os.getenv("OMNI_SIMPLEMEM_REUSE_STATE", "0") == "1"
            and mau_store_dir.is_dir()
            and any(mau_store_dir.glob("*.jsonl"))
        )
        if state_dir.exists() and not self._reuse_existing_state:
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.backend = OmniMemoryOrchestrator(
            config=self._build_config(), data_dir=str(state_dir)
        )
        self.provenance.clear()
        if self._reuse_existing_state:
            self._restore_persisted_provenance()
        self._current_session = ""

    def _restore_persisted_provenance(self) -> None:
        """Rebuild adapter-only provenance when resuming a complete Omni bank."""
        for mau in self.backend.mau_store.get_active(limit=1_000_000):
            metadata = mau.metadata.to_dict()
            tags = [str(value) for value in metadata.get("tags") or []]
            dialogue_id = next(
                (tag.split(":", 1)[1] for tag in tags if tag.startswith("dialogue_id:")),
                str(mau.id),
            )
            session_id = str(metadata.get("session_id") or "")
            images = [str(mau.raw_pointer)] if getattr(mau, "raw_pointer", None) else []
            self.provenance.register(
                str(mau.id),
                Chunk(
                    chunk_id=dialogue_id,
                    text=str(mau.summary or ""),
                    images=images,
                    metadata={
                        "dialogue_id": dialogue_id,
                        "session_id": session_id,
                    },
                ),
            )

    def ingest(self, chunk: Chunk) -> None:
        if self._reuse_existing_state:
            return
        session_id = str(chunk.metadata.get("session_id") or "")
        if session_id != self._current_session:
            if self._current_session:
                self.backend.end_session()
            self.backend.start_session(session_id or None)
            self._current_session = session_id
        dialogue_id = str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
        tags = [f"dialogue_id:{dialogue_id}", f"session_id:{session_id}"]
        result = self.backend.add_text(
            chunk.text, session_id=session_id, tags=tags, force=True
        )
        if not result.success:
            raise RuntimeError(f"OmniSimpleMem add_text failed: {result.error}")
        mau = getattr(result, "mau", None)
        if mau is not None:
            self.provenance.register(str(mau.id), chunk)
        if str(self.config.get("executor_visual_input") or "image") == "image":
            for image in chunk.images:
                captions = [
                    str(value)
                    for value in chunk.metadata.get("image_captions") or []
                    if str(value).strip()
                ]
                caption = captions[0] if captions else chunk.text
                visual = self.backend.add_visual_with_caption_embedding(
                    image,
                    caption,
                    session_id=session_id,
                    tags=tags,
                    force=True,
                )
                if not visual.success:
                    raise RuntimeError(
                        f"OmniSimpleMem add_image failed for {image}: {visual.error}"
                    )
                visual_mau = getattr(visual, "mau", None)
                if visual_mau is not None:
                    self.provenance.register(str(visual_mau.id), chunk)

    def end_session(self, session_id: str) -> None:
        if self._reuse_existing_state:
            return
        if hasattr(self.backend, "end_session") and (
            not session_id or session_id == self._current_session
        ):
            self.backend.end_session()
            self._current_session = ""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        native = self.backend.query(
            request.text,
            top_k=request.top_k,
            benchmark_safe=True,
            query_embedding=request.query_vector,
        )
        items = []
        for row in native.items:
            memory_id = str(row.get("id") or row.get("memory_id") or len(items))
            provenance = self.provenance.get(memory_id)
            if request.visible_session_ids and not self.provenance.visible(
                memory_id, request.visible_session_ids
            ):
                continue
            metadata = dict(row.get("metadata") or {})
            image_paths = [str(x) for x in provenance.get("image_paths") or []]
            items.append(
                RetrievedMemory(
                    memory_id=memory_id,
                    text=str(row.get("summary") or row.get("text") or ""),
                    score=float(row["score"]) if row.get("score") is not None else None,
                    session_id=str(provenance.get("session_id") or ""),
                    source_dialogue_ids=list(provenance.get("source_dialogue_ids") or []),
                    image_ids=list(provenance.get("image_ids") or []),
                    image_paths=image_paths,
                    metadata=metadata,
                )
            )
            if len(items) >= request.top_k:
                break
        return RetrievalResult(items=items, trace={"baseline": self.baseline, "via": "omni_query"})

    def snapshot(self) -> list[MemoryRecord]:
        records = []
        for mau in self.backend.mau_store.get_active(limit=1_000_000):
            provenance = self.provenance.get(str(mau.id))
            records.append(
                MemoryRecord(
                    memory_id=str(mau.id),
                    text=str(mau.summary or ""),
                    session_id=str(provenance.get("session_id") or ""),
                    source_dialogue_ids=list(provenance.get("source_dialogue_ids") or []),
                    image_ids=list(provenance.get("image_ids") or []),
                    image_paths=list(provenance.get("image_paths") or []),
                    backend_type="omni_mau",
                    metadata=mau.metadata.to_dict(),
                )
            )
        return records

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None
        self._current_session = ""

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "omni_simplemem",
            "baseline": self.baseline,
            "available": True,
            "supports_images": True,
            "supports_session_filter": True,
        }
