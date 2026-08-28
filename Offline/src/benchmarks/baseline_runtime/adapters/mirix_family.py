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
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    def reset(self, sample_id: str, state_dir: Path) -> None:
        if state_dir.exists():
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True)
        env_name = "MMA_DIR" if self.baseline == "MMA" else "MIRIX_DIR"
        os.environ[env_name] = str(state_dir)
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

    def _patch_openai_tool_compat(self) -> None:
        """Normalize Qwen tool tags when vLLM auto-tool parsing is disabled."""
        module = importlib.import_module(f"{self.package}.llm_api.openai_client")
        client_class = module.OpenAIClient
        if getattr(client_class, "_offline_tool_compat", False):
            return
        original_build = client_class.build_request_data
        original_request = client_class.request

        def build_request(client: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            data = original_build(client, *args, **kwargs)
            tools = data.get("tools") or []
            if data.get("tool_choice") == "required" and tools:
                function = tools[0].get("function") or {}
                data["tool_choice"] = {
                    "type": "function",
                    "function": {"name": function.get("name")},
                }
            return data

        def request(client: Any, request_data: dict[str, Any]) -> dict[str, Any]:
            return _normalize_openai_tool_tags(original_request(client, request_data))

        client_class.build_request_data = build_request
        client_class.request = request
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
            context_window=int(self.config.get("context_window") or 128000),
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
        self.backend.send_message(**kwargs)
        self._last_chunk = chunk
        current = self._memory_rows()
        if not current:
            self._insert_fallback_memory(chunk)
            current = self._memory_rows()
        for row in current:
            memory_id = row["memory_id"]
            if memory_id not in before or before[memory_id] != row["text"]:
                self.provenance.register(memory_id, chunk)
                self._known_ids.add(memory_id)

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


def _normalize_openai_tool_tags(response: dict[str, Any]) -> dict[str, Any]:
    """Turn Qwen's textual tool envelope into valid OpenAI tool arguments."""
    for choice in response.get("choices") or []:
        message = choice.get("message") or {}
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            payload = _tool_payload(str(function.get("arguments") or ""))
            if payload is None:
                continue
            function["name"] = str(payload.get("name") or function.get("name") or "")
            arguments = payload.get("arguments") or payload.get("args") or {}
            function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        if message.get("tool_calls"):
            continue
        payload = _tool_payload(str(message.get("content") or ""))
        if payload is None:
            continue
        arguments = payload.get("arguments") or payload.get("args") or {}
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": str(payload.get("name") or ""),
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ]
    return response


def _tool_payload(text: str) -> dict[str, Any] | None:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    candidate = match.group(1) if match else text
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None
