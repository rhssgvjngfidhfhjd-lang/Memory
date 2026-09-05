from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import uuid
from contextlib import contextmanager
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


_PARTITIONS: tuple[tuple[str, str, str], ...] = (
    ("episodic_memory_manager", "episodic_memory_agent_state", "list_episodic_memory"),
    ("semantic_memory_manager", "semantic_memory_agent_state", "list_semantic_items"),
    ("procedural_memory_manager", "procedural_memory_agent_state", "list_procedures"),
    ("resource_memory_manager", "resource_memory_agent_state", "list_resources"),
    ("knowledge_vault_manager", "knowledge_vault_agent_state", "list_knowledge"),
)


class MirixFamilyAdapter(BaselineAdapter):
    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root
        self.config = dict(config)
        self.package = "mma" if baseline == "MMA" else "mirix"
        self.backend: Any = None
        self.provenance = ProvenanceIndex()
        self._known_ids: set[str] = set()
        self._last_chunk: Chunk | None = None
        self._ingested_chunks = 0
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    def reset(self, sample_id: str, state_dir: Path) -> None:
        if state_dir.exists():
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True)
        temp_dir = state_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        env_name = "MMA_DIR" if self.baseline == "MMA" else "MIRIX_DIR"
        os.environ[env_name] = str(state_dir)
        os.environ["TMPDIR"] = str(temp_dir)
        os.environ["SQLITE_TMPDIR"] = str(temp_dir)
        os.environ["OPENAI_API_BASE"] = str(self.config["executor_base_url"])
        os.environ["OPENAI_BASE_URL"] = str(self.config["executor_base_url"])
        os.environ.setdefault("OPENAI_API_KEY", str(self.config.get("executor_api_key") or "EMPTY"))

        self._ensure_package_importable()
        self._patch_openai_tool_compat()
        constants = importlib.import_module(f"{self.package}.agent.app_constants")
        wrapper_module = importlib.import_module(f"{self.package}.agent.agent_wrapper")
        model = str(self.config["executor_model"])
        if model not in constants.OPENAI_MODELS:
            constants.OPENAI_MODELS = list(constants.OPENAI_MODELS) + [model]
            wrapper_module.OPENAI_MODELS = constants.OPENAI_MODELS

        config_path = state_dir / f"{self.package}.yaml"
        config_path.write_text(
            json.dumps({"agent_name": f"{self.package}_{sample_id}_{uuid.uuid4().hex[:8]}", "model_name": model}),
            encoding="utf-8",
        )
        agent_module = importlib.import_module(f"{self.package}.agent")
        llm, embedding = self._model_configs()
        with self._patched_defaults(llm, embedding):
            self.backend = agent_module.AgentWrapper(str(config_path))
        self._apply_model_config(llm, embedding)
        self.provenance.clear()
        self._known_ids = set()
        self._last_chunk = None
        self._ingested_chunks = 0

    def _patch_openai_tool_compat(self) -> None:
        """Normalize Qwen tool tags when vLLM auto-tool parsing is disabled."""
        module = importlib.import_module(f"{self.package}.llm_api.openai_client")
        client_class = module.OpenAIClient
        if getattr(client_class, "_offline_tool_compat", False):
            return
        original_build = client_class.build_request_data
        original_request = client_class.request
        original_request_async = client_class.request_async
        original_convert = client_class.convert_response_to_chat_completion

        def build_request(client: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return _normalize_openai_tool_request(
                original_build(client, *args, **kwargs)
            )

        def request(client: Any, request_data: dict[str, Any]) -> dict[str, Any]:
            return _normalize_openai_tool_tags(original_request(client, request_data))

        async def request_async(
            client: Any, request_data: dict[str, Any]
        ) -> dict[str, Any]:
            response = await original_request_async(client, request_data)
            return _normalize_openai_tool_tags(response)

        def convert_response_to_chat_completion(
            client: Any, response_data: dict[str, Any], *args: Any, **kwargs: Any
        ) -> Any:
            # Keep the normalization at the final common conversion boundary
            # as well. Some MIRIX/MMA agent paths obtain a raw response through
            # an alternate async helper and reach this method without calling
            # the patched request methods above.
            return original_convert(
                client,
                _normalize_openai_tool_tags(response_data),
                *args,
                **kwargs,
            )

        client_class.build_request_data = build_request
        client_class.request = request
        client_class.request_async = request_async
        client_class.convert_response_to_chat_completion = (
            convert_response_to_chat_completion
        )
        client_class._offline_tool_compat = True

    def _ensure_package_importable(self) -> None:
        """Load MMA's uppercase source directory under its expected lowercase name."""
        if self.baseline != "MMA" or self.package in sys.modules:
            return
        package_dir = self.source_root / "MMA"
        init_path = package_dir / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            self.package,
            init_path,
            submodule_search_locations=[str(package_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load MMA package from {init_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.package] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(self.package, None)
            raise

    def _model_configs(self) -> tuple[Any, Any]:
        package = importlib.import_module(self.package)
        llm = package.LLMConfig(
            model=str(self.config["executor_model"]),
            model_endpoint_type="openai",
            model_endpoint=str(self.config["executor_base_url"]),
            model_wrapper=None,
            context_window=int(self.config.get("context_window") or 24000),
            temperature=float(self.config.get("executor_temperature") or 0.0),
            max_tokens=int(self.config.get("num_predict") or 512),
        )
        embedding = package.EmbeddingConfig(
            embedding_model=str(self.config["embedding_model"]),
            embedding_endpoint_type="openai",
            embedding_endpoint=str(self.config["embedding_base_url"]),
            embedding_dim=int(self.config["embedding_dim"]),
            embedding_chunk_size=300,
        )
        return llm, embedding

    @contextmanager
    def _patched_defaults(self, llm: Any, embedding: Any):
        package = importlib.import_module(self.package)
        original_llm = package.LLMConfig.__dict__["default_config"]
        original_embedding = package.EmbeddingConfig.__dict__["default_config"]
        package.LLMConfig.default_config = classmethod(lambda cls, *args, **kwargs: llm)
        package.EmbeddingConfig.default_config = classmethod(
            lambda cls, *args, **kwargs: embedding
        )
        try:
            yield
        finally:
            package.LLMConfig.default_config = original_llm
            package.EmbeddingConfig.default_config = original_embedding

    def _apply_model_config(self, llm: Any, embedding: Any) -> None:
        try:
            self.backend.client.set_default_llm_config(llm)
            self.backend.client.set_default_embedding_config(embedding)
            state_by_id = {}
            for state in self.backend.client.list_agents():
                updated = self.backend.client.update_agent(
                    agent_id=state.id,
                    llm_config=llm,
                    embedding_config=embedding,
                )
                state_by_id[str(state.id)] = updated
            for name, state in vars(self.backend.agent_states).items():
                state_id = str(getattr(state, "id", ""))
                if state_id in state_by_id:
                    setattr(self.backend.agent_states, name, state_by_id[state_id])
        except Exception:
            if self.config.get("baseline_strict_config", True):
                raise

    def ingest(self, chunk: Chunk) -> None:
        before = {
            row["memory_id"]: row["text"] for row in self._memory_rows()
        }
        kwargs = {
            "message": chunk.text,
            "image_uris": list(chunk.images) or None,
            "memorizing": True,
            "async_upload": False,
        }
        timestamp = str(chunk.metadata.get("timestamp") or "")
        if timestamp:
            kwargs["specific_timestamps"] = [timestamp]
        self._last_chunk = chunk
        if self._should_direct_insert(chunk):
            self._insert_fallback_memory(chunk)
        else:
            try:
                self.backend.send_message(**kwargs)
            except Exception as exc:
                if not _is_context_overflow_exception(exc):
                    raise
                if not self._has_new_or_changed_memory(before):
                    self._insert_fallback_memory(chunk)
        current = self._memory_rows()
        if not current:
            self._insert_fallback_memory(chunk)
            current = self._memory_rows()
        for row in current:
            memory_id = row["memory_id"]
            if memory_id not in before or before[memory_id] != row["text"]:
                self.provenance.register(memory_id, chunk)
                self._known_ids.add(memory_id)
        self._ingested_chunks += 1

    def _has_new_or_changed_memory(self, before: dict[str, str]) -> bool:
        for row in self._memory_rows():
            memory_id = row["memory_id"]
            if memory_id not in before or before[memory_id] != row["text"]:
                return True
        return False

    def _should_direct_insert(self, chunk: Chunk) -> bool:
        """Avoid native agent context spirals on very long WMA samples.

        MIRIX/MMA keep their memory-manager dialogue history inside each
        per-sample database.  WorldMemArena lifelong samples have 400+ chunks;
        after a few hundred native LLM tool turns, some manager agents hit
        their own context-recovery recursion and can hang until the outer
        worker watchdog kills the whole sample.  Direct semantic insertion is
        only used after the configurable native-ingest prefix, preserving
        chronological writes without letting late-session bookkeeping poison
        the full run.
        """
        benchmark = str(chunk.metadata.get("benchmark") or "").lower()
        is_wma_chunk = benchmark in {"worldmemarena", "wma"} or all(
            key in chunk.metadata
            for key in ("dataset", "dialogue_id", "round_id", "date")
        )
        if not is_wma_chunk:
            return False
        limit = int(self.config.get("native_ingest_chunk_limit") or 40)
        return self._ingested_chunks >= limit

    def _insert_fallback_memory(self, chunk: Chunk) -> None:
        """Use the native semantic store only when the agent produced no memory."""
        server = self.backend.client.server
        manager = server.semantic_memory_manager
        state = self.backend.agent_states.semantic_memory_agent_state
        organization_id = str(
            getattr(state, "organization_id", None)
            or getattr(state, "created_by_id", None)
            or ""
        )
        manager.insert_semantic_item(
            agent_state=state,
            name=str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)[:255],
            summary=chunk.text[:1000],
            details=chunk.text,
            source="benchmark_chunk",
            tree_path=[
                "benchmark",
                str(chunk.metadata.get("benchmark") or "memory"),
            ],
            organization_id=organization_id,
        )

    def end_session(self, session_id: str) -> None:
        del session_id
        before = {row["memory_id"] for row in self._memory_rows()}
        try:
            self.backend.send_message(
                message="",
                memorizing=True,
                force_absorb_content=True,
                async_upload=False,
            )
        except Exception:
            pass
        if self._last_chunk is not None:
            for row in self._memory_rows():
                memory_id = row["memory_id"]
                if memory_id not in before:
                    self.provenance.register(memory_id, self._last_chunk)
                    self._known_ids.add(memory_id)

    def _memory_rows(self) -> list[dict[str, Any]]:
        rows = []
        server = self.backend.client.server
        for manager_name, state_name, method_name in _PARTITIONS:
            manager = getattr(server, manager_name, None)
            state = getattr(self.backend.agent_states, state_name, None)
            if manager is None or state is None:
                continue
            try:
                values = getattr(manager, method_name)(state, limit=None) or []
            except Exception:
                values = []
            for value in values:
                raw_id = str(getattr(value, "id", None) or uuid.uuid4().hex[:12])
                rows.append(
                    {
                        "memory_id": f"{manager_name}:{raw_id}",
                        "text": _render_memory(manager_name, value),
                        "partition": manager_name.removesuffix("_manager"),
                        "raw": value,
                    }
                )
        return rows

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        candidates: list[tuple[float, dict[str, Any]]] = []
        visible = set(request.visible_session_ids)
        server = self.backend.client.server
        for manager_name, state_name, method_name in _PARTITIONS:
            manager = getattr(server, manager_name, None)
            state = getattr(self.backend.agent_states, state_name, None)
            if manager is None or state is None:
                continue
            hits = []
            for field in ("summary", "name", "description", "details"):
                try:
                    hits = getattr(manager, method_name)(
                        state,
                        query=request.text,
                        search_method="embedding",
                        search_field=field,
                        limit=max(request.top_k * 2, 8),
                    ) or []
                except Exception:
                    hits = []
                if hits:
                    break
            for rank, hit in enumerate(hits):
                raw_id = str(getattr(hit, "id", rank))
                memory_id = f"{manager_name}:{raw_id}"
                source = self.provenance.get(memory_id)
                session_id = str(source.get("session_id") or "")
                if visible and session_id not in visible:
                    continue
                native_score = getattr(hit, "score", None)
                confidence = getattr(hit, "confidence", None) if self.baseline == "MMA" else None
                score = float(confidence if confidence is not None else native_score if native_score is not None else 1.0 / (rank + 1))
                candidates.append(
                    (
                        score,
                        {
                            "memory_id": memory_id,
                            "text": _render_memory(manager_name, hit),
                            "source": source,
                            "partition": manager_name.removesuffix("_manager"),
                        },
                    )
                )
        candidates.sort(key=lambda value: value[0], reverse=True)
        items = []
        for score, row in candidates[: request.top_k]:
            source = row["source"]
            items.append(
                RetrievedMemory(
                    memory_id=row["memory_id"],
                    text=row["text"],
                    score=score,
                    session_id=str(source.get("session_id") or ""),
                    source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
                    image_ids=list(source.get("image_ids") or []),
                    image_paths=[],
                    metadata={"partition": row["partition"]},
                )
            )
        return RetrievalResult(
            items=items,
            trace={
                "baseline": self.baseline,
                "via": "confidence" if self.baseline == "MMA" else "partition_search",
            },
        )

    def snapshot(self) -> list[MemoryRecord]:
        records = []
        for row in self._memory_rows():
            source = self.provenance.get(row["memory_id"])
            records.append(
                MemoryRecord(
                    memory_id=row["memory_id"],
                    text=row["text"],
                    session_id=str(source.get("session_id") or ""),
                    source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
                    image_ids=list(source.get("image_ids") or []),
                    image_paths=list(source.get("image_paths") or []),
                    backend_type=f"{self.package}_{row['partition']}",
                    metadata={"partition": row["partition"]},
                )
            )
        return records

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.package,
            "baseline": self.baseline,
            "available": True,
            "supports_images": True,
            "supports_session_filter": True,
            "confidence_ranking": self.baseline == "MMA",
        }

    def close(self) -> None:
        self.backend = None


def _render_memory(manager: str, value: Any) -> str:
    fields = []
    for key in ("summary", "details", "description", "steps", "content", "name", "secret_value", "source"):
        item = getattr(value, key, None)
        if item:
            fields.append(f"{key}: {item}")
    return f"[{manager.removesuffix('_manager')}] " + " | ".join(fields)


def _normalize_openai_tool_request(data: dict[str, Any]) -> dict[str, Any]:
    tools = data.get("tools") or []
    if data.get("tool_choice") == "required" and tools:
        # These vLLM servers intentionally run without the automatic tool
        # parser. A named choice would force tools[0], preventing the model
        # from selecting insert/merge/finish and potentially creating an
        # endless search_in_memory({}) loop. With "none", Qwen still sees the
        # tool schemas and emits its selection as a textual <tool_call> tag,
        # which _normalize_openai_tool_tags converts below.
        data["tool_choice"] = "none"
    return data


def _normalize_openai_tool_tags(response: dict[str, Any]) -> dict[str, Any]:
    """Turn Qwen's textual tool envelope into valid OpenAI tool arguments."""
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            raw_arguments = str(function.get("arguments") or "")
            payload = _tool_payload(raw_arguments)
            if payload is None:
                # Qwen can occasionally return a correctly named forced tool
                # call with an empty arguments string. MIRIX tries to unpack
                # inner_thoughts before its normal tool-error recovery runs,
                # so an empty string would abort the complete memory update.
                # An empty object lets MIRIX report missing required arguments
                # through its existing model-recovery path.
                # Any non-JSON arguments would fail before MIRIX reaches its
                # normal tool validation/recovery path.  Normalize those to an
                # empty object as well; the tool schema can then request the
                # missing required fields on the next model turn.
                function["arguments"] = json.dumps(
                    _normalize_native_tool_arguments(
                        str(function.get("name") or ""), {}
                    ),
                    ensure_ascii=False,
                )
                continue
            is_envelope = "name" in payload and (
                "arguments" in payload or "args" in payload
            )
            if is_envelope:
                function["name"] = str(
                    payload.get("name") or function.get("name") or ""
                )
                arguments = payload.get("arguments") or payload.get("args") or {}
            else:
                # A normal OpenAI structured tool call already stores only its
                # argument object here.  Preserve it instead of discarding all
                # fields while looking for an outer Qwen envelope.
                arguments = payload
            arguments = _normalize_native_tool_arguments(
                str(function.get("name") or ""), arguments
            )
            function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        if message.get("tool_calls"):
            continue
        payload = _tool_payload(str(message.get("content") or ""))
        if payload is None:
            continue
        arguments = payload.get("arguments") or payload.get("args") or {}
        function_name = str(payload.get("name") or "")
        arguments = _normalize_native_tool_arguments(function_name, arguments)
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ]
    return response


def _is_context_overflow_exception(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "context_window_exceeded",
        "context length",
        "context_length_exceeded",
        "maximum context",
        "decoder prompt",
        "prompt is too long",
        "not enough messages to compress",
    )
    return any(marker in text for marker in markers)


def _normalize_native_tool_arguments(name: str, arguments: Any) -> Any:
    """Repair unsupported MIRIX/MMA search field/method combinations.

    Qwen occasionally chooses embedding search over a field for which the
    native schema stores no embedding.  Preserve the requested semantic search
    by switching to that partition's supported summary field instead of
    persisting a failed native tool call.
    """
    if name != "search_in_memory" or not isinstance(arguments, dict):
        return arguments
    normalized = dict(arguments)
    normalized.setdefault("memory_type", "all")
    normalized.setdefault("query", "")
    normalized.setdefault("search_method", "embedding")
    memory_type = normalized.get("memory_type")
    search_field = normalized.get("search_field")
    defaults = {
        "all": "null",
        "episodic": "summary",
        "resource": "summary" if normalized.get("search_method") == "embedding" else "content",
        "procedural": "summary",
        "knowledge_vault": "caption" if normalized.get("search_method") == "embedding" else "secret_value",
        "semantic": "summary",
    }
    valid_fields = {
        "all": {"null"},
        "episodic": {"summary", "details"},
        "resource": {"summary", "content"},
        "procedural": {"summary", "steps", "description"},
        "knowledge_vault": {"caption", "secret_value"},
        "semantic": {"name", "summary", "details"},
    }
    if memory_type not in valid_fields:
        normalized["memory_type"] = "all"
        memory_type = "all"
    if not search_field or search_field == "None":
        normalized["search_field"] = defaults[memory_type]
    elif search_field not in valid_fields[memory_type]:
        normalized["search_field"] = defaults[memory_type]
    if normalized.get("search_method") == "embedding":
        if memory_type == "resource" and normalized.get("search_field") == "content":
            normalized["search_field"] = "summary"
        elif memory_type == "knowledge_vault" and normalized.get("search_field") == "secret_value":
            normalized["search_field"] = "caption"
    return normalized


def _tool_payload(text: str) -> dict[str, Any] | None:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    candidate = match.group(1) if match else text
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None
