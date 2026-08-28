from __future__ import annotations

import asyncio
import importlib
import json
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
from embedding.chunk_builder import Chunk


class MemVerseAdapter(BaselineAdapter):
    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root
        self.config = dict(config)
        self.module: Any = None
        self.build_memory: Any = None
        self.state_dir: Path | None = None
        self._records: dict[str, MemoryRecord] = {}
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del sample_id
        if state_dir.exists():
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True)
        self.state_dir = state_dir
        os.environ["OPENAI_API_BASE"] = str(self.config["executor_base_url"])
        os.environ["OPENAI_BASE_URL"] = str(self.config["executor_base_url"])
        os.environ["OPENAI_MODEL"] = str(self.config["executor_model"])
        os.environ.setdefault("OPENAI_API_KEY", str(self.config.get("executor_api_key") or "EMPTY"))
        os.environ["OPENAI_EMBEDDING_BASE_URL"] = str(self.config["embedding_base_url"])
        os.environ["OPENAI_EMBEDDING_MODEL"] = str(self.config["embedding_model"])
        embedding_key_env = str(self.config.get("embedding_api_key_env") or "")
        os.environ["OPENAI_EMBEDDING_API_KEY"] = os.getenv(embedding_key_env) or "EMPTY"
        os.environ["LOG_DIR"] = str(state_dir / "logs")
        old_cwd = Path.cwd()
        try:
            os.chdir(self.source_root)
            self.module = importlib.import_module("orchestrator")
            self.build_memory = importlib.import_module("MemoryKB.build_memory")
        finally:
            os.chdir(old_cwd)
        self._configure_paths(state_dir)
        self._configure_models()
        _run(self.module.initialize_rag())
        self._records = {}

    def _configure_models(self) -> None:
        from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag.llm.openai import (
            openai_complete_if_cache,
            openai_embed,
        )
        from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag.utils import (
            wrap_embedding_func_with_attrs,
        )

        config = self.config

        async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
            kwargs.pop("keyword_extraction", None)
            return await openai_complete_if_cache(
                str(config["executor_model"]),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                base_url=str(config["executor_base_url"]),
                api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                **kwargs,
            )

        async def embed(texts: list[str]):
            return await openai_embed.func(
                texts,
                model=str(config["embedding_model"]),
                base_url=str(config["embedding_base_url"]),
                api_key=os.getenv("OPENAI_EMBEDDING_API_KEY") or "EMPTY",
            )

        self.module.gpt_4o_mini_complete = complete
        self.module.openai_embed = wrap_embedding_func_with_attrs(
            embedding_dim=int(config["embedding_dim"]),
            max_token_size=8192,
        )(embed)

    def _configure_paths(self, state_dir: Path) -> None:
        graph_root = state_dir / "graph"
        chunks_root = state_dir / "memory_chunks"
        conversation_root = state_dir / "conversation"
        self.module.BASE_DIR = str(graph_root)
        self.module.CORE_DIR = str(graph_root / "core")
        self.module.EPISODIC_DIR = str(graph_root / "episodic")
        self.module.SEMANTIC_DIR = str(graph_root / "semantic")
        self.module.MEMORY_JSON_DIR = str(chunks_root)
        self.module.CORE_JSON = str(chunks_root / "core_memory.json")
        self.module.EPISODIC_JSON = str(chunks_root / "episodic_memory.json")
        self.module.SEMANTIC_JSON = str(chunks_root / "semantic_memory.json")
        self.module.USER_CONV_DIR = conversation_root
        self.module.CONV_JSON = conversation_root / "conversation.json"
        conversation_root.mkdir(parents=True, exist_ok=True)
        prompts = self.source_root / "MemoryKB" / "Long_Term_Memory" / "system"
        self.build_memory.memory_files = {
            str(prompts / "core_memory_agent.txt"): str(chunks_root / "core_memory.json"),
            str(prompts / "episodic_memory_agent.txt"): str(chunks_root / "episodic_memory.json"),
            str(prompts / "semantic_memory_agent.txt"): str(chunks_root / "semantic_memory.json"),
        }

    def ingest(self, chunk: Chunk) -> None:
        dialogue_id = str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
        entry = {
            "id": dialogue_id,
            "query": chunk.text,
            "videocaption": None,
            "audiocaption": None,
            "imagecaption": "\n".join(str(x) for x in chunk.metadata.get("image_captions") or []) or None,
        }
        self.module.append_to_conversation(entry)
        _run(self.module.update_long_term_memory(entry))
        self._reload_records(chunk)

    def _reload_records(self, chunk: Chunk) -> None:
        assert self.state_dir is not None
        session_id = str(chunk.metadata.get("session_id") or "")
        for memory_type in ("core", "episodic", "semantic"):
            path = self.state_dir / "memory_chunks" / f"{memory_type}_memory.json"
            if not path.exists():
                continue
            for row in _read_jsonl(path):
                raw_id = str(row.get("id") or "")
                memory_id = f"memverse:{memory_type}:{raw_id}"
                existing = self._records.get(memory_id)
                self._records[memory_id] = MemoryRecord(
                    memory_id=memory_id,
                    text=str(row.get("output_text") or row.get("input_text") or ""),
                    session_id=existing.session_id if existing else session_id,
                    source_dialogue_ids=(
                        existing.source_dialogue_ids
                        if existing else ([raw_id] if raw_id else [])
                    ),
                    image_ids=(
                        existing.image_ids
                        if existing
                        else [str(x) for x in chunk.metadata.get("image_ids") or []]
                    ),
                    image_paths=existing.image_paths if existing else list(chunk.images),
                    backend_type=f"memverse_{memory_type}",
                    metadata={"memory_type": memory_type},
                )

    def end_session(self, session_id: str) -> None:
        del session_id

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag import QueryParam

        visible = set(request.visible_session_ids)
        stores = (
            ("core", self.module.mem_core),
            ("episodic", self.module.mem_epi),
            ("semantic", self.module.mem_sem),
        )
        items = []
        for memory_type, store in stores:
            value = _run(
                store.aquery(
                    request.text,
                    param=QueryParam(mode="hybrid", top_k=request.top_k),
                )
            )
            text = str(value or "").strip()
            if not text:
                continue
            related = [
                row
                for row in self._records.values()
                if row.metadata.get("memory_type") == memory_type
                and (not visible or row.session_id in visible)
            ]
            source_ids = list(
                dict.fromkeys(
                    source
                    for row in related
                    for source in row.source_dialogue_ids
                )
            )
            items.append(
                RetrievedMemory(
                    memory_id=f"memverse:{memory_type}:{request.query_id}",
                    text=text,
                    score=None,
                    session_id=related[-1].session_id if related else "",
                    source_dialogue_ids=source_ids,
                    image_ids=list(dict.fromkeys(x for row in related for x in row.image_ids)),
                    image_paths=list(dict.fromkeys(x for row in related for x in row.image_paths)),
                    metadata={"memory_type": memory_type, "aggregated_context": True},
                )
            )
        return RetrievalResult(
            items=items[: request.top_k],
            trace={"baseline": self.baseline, "via": "lightrag_hybrid"},
        )

    def snapshot(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def close(self) -> None:
        if self.module is None:
            return
        for store in (self.module.mem_core, self.module.mem_epi, self.module.mem_sem):
            finalize = getattr(store, "finalize_storages", None)
            if finalize is not None:
                try:
                    _run(finalize())
                except Exception:
                    pass

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "memverse",
            "baseline": self.baseline,
            "available": True,
            "supports_images": True,
            "supports_session_filter": True,
            "parametric_memory": False,
        }


def _run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
