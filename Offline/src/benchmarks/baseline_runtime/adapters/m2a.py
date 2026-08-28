from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime
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


class M2AAdapter(BaselineAdapter):
    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root
        self.config = dict(config)
        self.backend: Any = None
        self.state_dir: Path | None = None
        self.provenance = ProvenanceIndex()
        self._known_ids: set[str] = set()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    def _build_config(self, state_dir: Path) -> Any:
        from agent.config import (
            ChatAgentConfig,
            LLMConfig,
            M2AConfig,
            MemoryConfig,
            MemoryManagerConfig,
            MultimodalEmbeddingConfig,
            TextEmbeddingConfig,
        )

        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        llm = LLMConfig(
            model=str(self.config["executor_model"]),
            api_key=api_key,
            base_url=str(self.config["executor_base_url"]),
            temperature=float(self.config.get("executor_temperature") or 0.0),
            max_tokens=int(self.config.get("num_predict") or 512),
            timeout=int(self.config.get("request_timeout") or 180),
        )
        embedding_url = str(self.config.get("embedding_base_url") or "")
        text_embedding = TextEmbeddingConfig(
            model=str(self.config["embedding_model"]),
            api_key=os.getenv(str(self.config.get("embedding_api_key_env") or "")) or "EMPTY",
            base_url=embedding_url,
            dimension=int(self.config["embedding_dim"]),
        )
        multimodal_embedding = MultimodalEmbeddingConfig(
            model=str(self.config["embedding_model"]),
            api_key=os.getenv(str(self.config.get("embedding_api_key_env") or "")) or "EMPTY",
            base_url=embedding_url,
            dimension=int(self.config["embedding_dim"]),
        )
        memory = MemoryConfig(
            raw_db_path=str(state_dir / "raw.db"),
            semantic_db_path=str(state_dir / "semantic.db"),
            reuse_db=False,
            max_raw_messages_return=20,
        )
        return M2AConfig(
            llm=llm,
            text_embedding=text_embedding,
            multimodal_embedding=multimodal_embedding,
            memory=memory,
            chat_agent=ChatAgentConfig(),
            memory_manager=MemoryManagerConfig(),
        )

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del sample_id
        self.close()
        self.state_dir = state_dir
        if state_dir.exists():
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True)
        from agent.m2a import M2ASystem
        from agent.agents.chat_agent import ChatAgent

        self.backend = M2ASystem(config=self._build_config(state_dir))
        self.backend.chat_agent = ChatAgent(
            memory_manager=self.backend.memory_manager,
            raw_store=self.backend.raw_store,
            llm=self.backend.llm,
            image_manager=self.backend.image_manager,
            update_memory=True,
            config=self.backend.config.chat_agent,
            update_only=True,
        )
        self.backend.chat_agent.init_conversation(
            "Store durable information from every conversation round. "
            "Do not answer the conversation; update long-term memory when appropriate."
        )
        self.provenance.clear()
        self._known_ids = set()

    def ingest(self, chunk: Chunk) -> None:
        before = {
            str(row.get("id") or ""): (
                str(row.get("text") or ""),
                str(row.get("image_caption") or ""),
                str(row.get("image_path") or ""),
            )
            for row in self._memory_rows()
            if row.get("id")
        }
        timestamp = _parse_timestamp(str(chunk.metadata.get("timestamp") or ""))
        if self.config.get("m2a_native_tools", False):
            try:
                self.backend.chat_agent.chat(
                    user_text=chunk.text,
                    user_image_path_or_url=chunk.images[0] if chunk.images else None,
                    timestamp=timestamp,
                    role="conversation",
                )
            except KeyError as exc:
                # M2A's update-only graph intentionally has no response node,
                # while ChatAgent.chat still reads result["response"].
                if exc.args != ("response",):
                    raise
        else:
            # The shared vLLM endpoints do not expose auto-tool parsing. Keep
            # M2A's own semantic store and hybrid retriever, but write the
            # benchmark chunk directly instead of dropping it silently.
            from agent.stores.semantic import SemanticMemory

            self.backend.semantic_store.add(SemanticMemory(text=chunk.text))
        current = self._memory_rows()
        for row in current:
            memory_id = str(row.get("id") or "")
            fingerprint = (
                str(row.get("text") or ""),
                str(row.get("image_caption") or ""),
                str(row.get("image_path") or ""),
            )
            if memory_id and (
                memory_id not in self._known_ids
                or before.get(memory_id) != fingerprint
            ):
                self.provenance.register(memory_id, chunk)
                self._known_ids.add(memory_id)

    def end_session(self, session_id: str) -> None:
        del session_id

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        hits = self.backend.semantic_store.hybrid_search(
            query_text=request.text,
            query_image_path=request.query_image,
            top_k=request.top_k,
        )
        items = []
        for rank, hit in enumerate(hits):
            memory_id = str(getattr(hit, "memory_id", None) or getattr(hit, "id", rank))
            provenance = self.provenance.get(memory_id)
            if request.visible_session_ids and not self.provenance.visible(
                memory_id, request.visible_session_ids
            ):
                continue
            text = str(getattr(hit, "text", "") or "")
            caption = str(getattr(hit, "image_caption", "") or "")
            if caption:
                text = f"{text}\nimage_caption: {caption}"
            items.append(
                RetrievedMemory(
                    memory_id=f"m2a:{memory_id}",
                    text=text,
                    score=None,
                    session_id=str(provenance.get("session_id") or ""),
                    source_dialogue_ids=list(provenance.get("source_dialogue_ids") or []),
                    image_ids=list(provenance.get("image_ids") or []),
                    image_paths=list(provenance.get("image_paths") or []),
                )
            )
            if len(items) >= request.top_k:
                break
        return RetrievalResult(items=items, trace={"baseline": self.baseline, "via": "hybrid_search"})

    def _memory_rows(self) -> list[dict[str, Any]]:
        try:
            return list(
                self.backend.semantic_store.db.query(
                    collection_name="memory", filter="id > 0", limit=16_384
                )
                or []
            )
        except Exception:
            return []

    def snapshot(self) -> list[MemoryRecord]:
        records = []
        for row in self._memory_rows():
            memory_id = str(row.get("id") or "")
            provenance = self.provenance.get(memory_id)
            text = str(row.get("text") or "")
            if row.get("image_caption"):
                text += f"\nimage_caption: {row['image_caption']}"
            records.append(
                MemoryRecord(
                    memory_id=f"m2a:{memory_id}",
                    text=text,
                    session_id=str(provenance.get("session_id") or ""),
                    source_dialogue_ids=list(provenance.get("source_dialogue_ids") or []),
                    image_ids=list(provenance.get("image_ids") or []),
                    image_paths=list(provenance.get("image_paths") or []),
                    backend_type="m2a_semantic",
                    metadata={"image_path": row.get("image_path") or ""},
                )
            )
        return records

    def close(self) -> None:
        if self.backend is not None:
            for name in ("semantic_store", "raw_store"):
                try:
                    getattr(self.backend, name).close()
                except Exception:
                    pass
            self.backend = None

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "m2a",
            "baseline": self.baseline,
            "available": True,
            "supports_images": True,
            "supports_session_filter": True,
            "requires_python": ">=3.12",
        }


def _parse_timestamp(value: str) -> datetime:
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        try:
            return datetime.strptime(value, "%Y-%m-%d")
        except ValueError:
            return datetime.now()
