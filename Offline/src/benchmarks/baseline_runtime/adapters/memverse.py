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
from benchmarks.io_utils import write_json_atomic, write_jsonl_atomic
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
        self._current_session = ""
        self._graph_rows: dict[str, int] = {}
        self._completed_chunk_ids: set[str] = set()
        self._completed_session_ids: set[str] = set()
        self._memory_rows: dict[str, dict[str, dict[str, Any]]] = {}
        self._conversation_ids: set[str] = set()
        self._state_path: Path | None = None
        self._reuse_existing_state = False
        self._loop = asyncio.new_event_loop()
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    def reset(self, sample_id: str, state_dir: Path) -> None:
        del sample_id
        self._reuse_existing_state = (
            os.getenv("MEMVERSE_REUSE_STATE", "0") == "1" and state_dir.is_dir()
        )
        if state_dir.exists() and not self._reuse_existing_state:
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir = state_dir
        self._state_path = state_dir / "adapter_state.json"
        if self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
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
        self._run(self.module.initialize_rag())
        self._current_session = ""
        self._load_memory_rows()
        self._load_conversation_ids()
        self._restore_state()

    def _configure_models(self) -> None:
        from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag.llm.openai import (
            openai_complete_if_cache,
            openai_embed,
        )
        from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag.utils import (
            TiktokenTokenizer,
            wrap_embedding_func_with_attrs,
        )

        config = self.config
        tokenizer = TiktokenTokenizer()

        async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
            kwargs.pop("keyword_extraction", None)
            kwargs.setdefault(
                "max_tokens",
                int(
                    config.get("executor_max_tokens")
                    or config.get("num_predict")
                    or 512
                ),
            )
            history_messages = _bounded_history_messages(
                history_messages or [],
                prompt=str(prompt),
                system_prompt=str(system_prompt or ""),
                tokenizer=tokenizer,
                max_input_tokens=int(
                    config.get("memverse_max_input_tokens") or 24000
                ),
            )
            return await openai_complete_if_cache(
                str(config["executor_model"]),
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
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
        session_id = str(chunk.metadata.get("session_id") or "")
        if self._current_session and session_id != self._current_session:
            self._flush_graph()
        self._current_session = session_id
        dialogue_id = str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
        existing_types = {
            memory_type
            for memory_type, rows in self._memory_rows.items()
            if dialogue_id in rows
        }
        if existing_types == set(self._memory_types()):
            self._completed_chunk_ids.add(dialogue_id)
            self._reload_record_for_chunk(chunk)
            self._persist_state()
            return
        if existing_types:
            # A worker may be terminated between the three native summary
            # writes.  Those partial rows have not reached the graph yet, so
            # remove them and regenerate the complete triplet atomically from
            # the adapter's point of view.
            self._remove_partial_memory_rows(dialogue_id)
        entry = {
            "id": dialogue_id,
            "query": chunk.text,
            "videocaption": None,
            "audiocaption": None,
            "imagecaption": "\n".join(str(x) for x in chunk.metadata.get("image_captions") or []) or None,
        }
        if dialogue_id not in self._conversation_ids:
            self.module.append_to_conversation(entry)
            self._conversation_ids.add(dialogue_id)
        # Generate all three native MemVerse memory summaries per chunk, but
        # defer LightRAG graph construction until the session boundary.  WMA
        # and H2HMem expose questions only after that boundary, so this retains
        # prefix safety while allowing LightRAG to batch the session documents.
        self._run(self.module.update_long_term_memory(entry, insert_graph=False))
        self._refresh_memory_rows(dialogue_id)
        missing = [
            memory_type
            for memory_type in self._memory_types()
            if dialogue_id not in self._memory_rows[memory_type]
        ]
        if missing:
            raise RuntimeError(
                f"MemVerse failed to persist {dialogue_id} for: {', '.join(missing)}"
            )
        self._completed_chunk_ids.add(dialogue_id)
        self._reload_record_for_chunk(chunk)
        self._persist_state()

    def _reload_record_for_chunk(self, chunk: Chunk) -> None:
        session_id = str(chunk.metadata.get("session_id") or "")
        dialogue_id = str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
        for memory_type in self._memory_types():
            row = self._memory_rows.get(memory_type, {}).get(dialogue_id)
            if row is None:
                continue
            raw_id = str(row.get("id") or dialogue_id)
            memory_id = f"memverse:{memory_type}:{raw_id}"
            existing = self._records.get(memory_id)
            self._records[memory_id] = MemoryRecord(
                memory_id=memory_id,
                text=str(row.get("output_text") or row.get("input_text") or ""),
                session_id=session_id or (existing.session_id if existing else ""),
                source_dialogue_ids=list(
                    dict.fromkeys(
                        [
                            *(existing.source_dialogue_ids if existing else []),
                            *([raw_id] if raw_id else []),
                        ]
                    )
                ),
                image_ids=list(
                    dict.fromkeys(
                        [
                            *(existing.image_ids if existing else []),
                            *[str(x) for x in chunk.metadata.get("image_ids") or []],
                        ]
                    )
                ),
                image_paths=list(
                    dict.fromkeys(
                        [*(existing.image_paths if existing else []), *list(chunk.images)]
                    )
                ),
                backend_type=f"memverse_{memory_type}",
                metadata={"memory_type": memory_type},
            )

    def end_session(self, session_id: str) -> None:
        if not session_id or session_id == self._current_session:
            self._flush_graph()
            if session_id:
                self._completed_session_ids.add(str(session_id))
            self._current_session = ""
            self._persist_state()

    def _flush_graph(self) -> None:
        if self.module is None or not self._current_session:
            return
        stores = self._graph_stores()
        # LightRAG's document-processing queue is process-global even when the
        # stores use different working directories. Keep graph insertion
        # sequential so core/episodic/semantic documents cannot cross stores.
        for memory_type, store, path in stores:
            row_count = self.module.count_jsonl_rows(path)
            start_row = min(self._graph_rows.get(memory_type, 0), row_count)
            if start_row >= row_count:
                continue
            self._run(
                self.module.insert_chunks_from_json(
                    store,
                    path,
                    start_row=start_row,
                )
            )
            self._graph_rows[memory_type] = row_count
            # Persist after each store.  If a later store times out, a resumed
            # worker starts from the last confirmed boundary and LightRAG
            # content hashes make a partially completed store idempotent.
            self._persist_state()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        from MemoryKB.Long_Term_Memory.Graph_Construction.lightrag import QueryParam

        visible = set(request.visible_session_ids)
        stores = (
            ("core", self.module.mem_core),
            ("episodic", self.module.mem_epi),
            ("semantic", self.module.mem_sem),
        )
        async def query_stores() -> list[Any]:
            return await asyncio.gather(
                *(
                    store.aquery(
                        request.text,
                        param=QueryParam(mode="hybrid", top_k=request.top_k),
                    )
                    for _, store in stores
                )
            )

        values = self._run(query_stores())
        items = []
        for (memory_type, _), value in zip(stores, values):
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
        self._flush_graph()
        for store in (self.module.mem_core, self.module.mem_epi, self.module.mem_sem):
            finalize = getattr(store, "finalize_storages", None)
            if finalize is not None:
                try:
                    self._run(finalize())
                except Exception:
                    pass
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self._loop.close()
        self._current_session = ""
        self.module = None

    def _run(self, awaitable: Any) -> Any:
        """Keep LightRAG queues and workers bound to one persistent event loop."""
        return self._loop.run_until_complete(awaitable)

    @staticmethod
    def _memory_types() -> tuple[str, str, str]:
        return ("core", "episodic", "semantic")

    def _memory_paths(self) -> dict[str, Path]:
        assert self.state_dir is not None
        return {
            memory_type: self.state_dir / "memory_chunks" / f"{memory_type}_memory.json"
            for memory_type in self._memory_types()
        }

    def _graph_stores(self) -> tuple[tuple[str, Any, str], ...]:
        return (
            ("core", self.module.mem_core, str(self.module.CORE_JSON)),
            ("episodic", self.module.mem_epi, str(self.module.EPISODIC_JSON)),
            ("semantic", self.module.mem_sem, str(self.module.SEMANTIC_JSON)),
        )

    def _load_memory_rows(self) -> None:
        self._memory_rows = {memory_type: {} for memory_type in self._memory_types()}
        for memory_type, path in self._memory_paths().items():
            if not path.is_file():
                continue
            for row in _read_jsonl(path):
                raw_id = str(row.get("id") or "")
                if raw_id:
                    self._memory_rows[memory_type][raw_id] = row

    def _refresh_memory_rows(self, dialogue_id: str) -> None:
        for memory_type, path in self._memory_paths().items():
            if not path.is_file():
                continue
            match = _last_jsonl_row(path)
            if match is not None and str(match.get("id") or "") != dialogue_id:
                match = next(
                    (
                        row
                        for row in reversed(list(_read_jsonl(path)))
                        if str(row.get("id") or "") == dialogue_id
                    ),
                    None,
                )
            if match is not None:
                self._memory_rows[memory_type][dialogue_id] = match

    def _remove_partial_memory_rows(self, dialogue_id: str) -> None:
        for memory_type, path in self._memory_paths().items():
            if not path.is_file():
                continue
            rows = [
                row
                for row in _read_jsonl(path)
                if str(row.get("id") or "") != dialogue_id
            ]
            write_jsonl_atomic(path, rows)
            self._memory_rows[memory_type].pop(dialogue_id, None)
        self._completed_chunk_ids.discard(dialogue_id)

    def _load_conversation_ids(self) -> None:
        self._conversation_ids = set()
        if self.state_dir is None:
            return
        path = self.state_dir / "conversation" / "conversation.json"
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._conversation_ids = {
            str(row.get("id") or "")
            for row in payload
            if isinstance(row, dict) and row.get("id")
        }

    def _restore_state(self) -> None:
        self._records = {}
        self._graph_rows = {memory_type: 0 for memory_type in self._memory_types()}
        self._completed_chunk_ids = set.intersection(
            *(set(rows) for rows in self._memory_rows.values())
        ) if self._memory_rows else set()
        self._completed_session_ids = set()
        if not self._reuse_existing_state or self._state_path is None:
            self._persist_state()
            return
        try:
            payload = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._persist_state()
            return
        stored_completed = {
            str(value) for value in payload.get("completed_chunk_ids") or []
        }
        self._completed_chunk_ids &= stored_completed or self._completed_chunk_ids
        self._completed_session_ids = {
            str(value) for value in payload.get("completed_session_ids") or []
        }
        stored_rows = dict(payload.get("graph_rows") or {})
        for memory_type in self._memory_types():
            maximum = len(self._memory_rows[memory_type])
            self._graph_rows[memory_type] = min(
                max(int(stored_rows.get(memory_type, 0) or 0), 0), maximum
            )
        for row in payload.get("records") or []:
            try:
                record = MemoryRecord.from_dict(row)
            except (TypeError, ValueError):
                continue
            self._records[record.memory_id] = record

    def _persist_state(self) -> None:
        if self._state_path is None:
            return
        write_json_atomic(
            self._state_path,
            {
                "version": 1,
                "completed_chunk_ids": sorted(self._completed_chunk_ids),
                "completed_session_ids": sorted(self._completed_session_ids),
                "graph_rows": dict(self._graph_rows),
            },
        )

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "memverse",
            "baseline": self.baseline,
            "available": True,
            "supports_images": True,
            "supports_session_filter": True,
            "parametric_memory": False,
        }


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _last_jsonl_row(path: Path) -> dict[str, Any] | None:
    """Read the last non-empty JSONL row without rescanning a growing file."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        suffix = b""
        while position > 0:
            block_size = min(65536, position)
            position -= block_size
            handle.seek(position)
            suffix = handle.read(block_size) + suffix
            lines = [line for line in suffix.splitlines() if line.strip()]
            if lines and (position == 0 or len(lines) >= 2):
                return json.loads(lines[-1].decode("utf-8-sig"))
    return None


def _bounded_history_messages(
    history_messages: list[dict[str, Any]],
    *,
    prompt: str,
    system_prompt: str,
    tokenizer: Any,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    """Keep recent LightRAG history inside the executor context window.

    Entity gleaning includes the complete previous extraction response.  A
    verbose response can otherwise make the next request exceed the 32k vLLM
    limit even though each source chunk is small.  Reserve room for chat
    framing and retain the newest history first, clipping only the oldest part
    that still fits.
    """
    if not history_messages:
        return []
    framing_reserve = 256 + 16 * (len(history_messages) + 2)
    available = max_input_tokens - framing_reserve
    available -= len(tokenizer.encode(system_prompt))
    available -= len(tokenizer.encode(prompt))
    if available <= 0:
        return []

    bounded: list[dict[str, Any]] = []
    for message in reversed(history_messages):
        content = str(message.get("content") or "")
        tokens = tokenizer.encode(content)
        if len(tokens) <= available:
            bounded.append(dict(message))
            available -= len(tokens)
            continue
        if available > 0:
            clipped = dict(message)
            clipped["content"] = tokenizer.decode(tokens[-available:])
            bounded.append(clipped)
        break
    bounded.reverse()
    return bounded
