from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import base64
import contextvars
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import sqlite3
import sys
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from json_repair import repair_json

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    NativeAnswerRequest,
    NativeAnswerResult,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from benchmarks.baseline_runtime.provenance import ProvenanceIndex
from embedding.chunk_builder import Chunk, compact_text


_PARTITIONS: tuple[tuple[str, str, str], ...] = (
    ("episodic_memory_manager", "episodic_memory_agent_state", "list_episodic_memory"),
    ("semantic_memory_manager", "semantic_memory_agent_state", "list_semantic_items"),
    ("procedural_memory_manager", "procedural_memory_agent_state", "list_procedures"),
    ("resource_memory_manager", "resource_memory_agent_state", "list_resources"),
    ("knowledge_vault_manager", "knowledge_vault_agent_state", "list_knowledge"),
)

_ACTIVE_MIRIX_RETRIEVAL: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "offline_active_mirix_retrieval", default=None
)
_ACTIVE_MIRIX_QA_TRUNCATION: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("offline_active_mirix_qa_truncation", default=None)
)
_ACTIVE_MIRIX_NATIVE_MEMORY_TOOL_REQUIRED: contextvars.ContextVar[bool] = (
    contextvars.ContextVar(
        "offline_active_mirix_native_memory_tool_required", default=False
    )
)
_ACTIVE_MIRIX_NATIVE_MEMORY_MAX_OUTPUT_TOKENS: contextvars.ContextVar[int] = (
    contextvars.ContextVar(
        "offline_active_mirix_native_memory_max_output_tokens", default=0
    )
)

# MIRIX runs one benchmark sample per worker process. Native memory agents may
# embed fields concurrently in child threads, so a process-local context (not a
# ContextVar) is used to make the current ingestion images visible to all of
# them. The value is installed only around a synchronous native build/answer
# lifecycle and restored before the worker advances to the next point.
_ACTIVE_MIRIX_MULTIMODAL_EMBEDDING: dict[str, Any] | None = None


class _MirixEmbeddingProxy:
    """Add optional image inputs without changing MIRIX's text-only default."""

    def __init__(
        self,
        delegate: Any,
        *,
        model: str,
        endpoint: str,
        dimensions: int,
    ) -> None:
        self._delegate = delegate
        self._model = str(model)
        self._endpoint = str(endpoint)
        self._dimensions = int(dimensions)

    def get_text_embedding(self, text: str) -> list[float]:
        active = _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING
        images = list((active or {}).get("images") or [])
        if not images:
            return list(self._delegate.get_text_embedding(text))
        return _request_mirix_multimodal_embedding(
            endpoint=self._endpoint,
            model=self._model,
            dimensions=self._dimensions,
            text=str(text),
            image_paths=images,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

_MIRIX_BOUNDED_MEMORY_PROMPTS = {
    "semantic_memory_agent": (
        "[Offline bounded memory-update protocol]\n"
        "Keep the complete function-call arguments below 700 tokens. For "
        "semantic_memory_insert, consolidate related facts and return at most "
        "three of the most important concepts. Keep name at most 120 characters, "
        "summary at most 240 characters, details at most 500 characters, and "
        "source at most 160 characters. Never transcribe whole sessions."
    ),
    "episodic_memory_agent": (
        "[Offline bounded memory-update protocol]\n"
        "Keep the complete function-call arguments below 650 tokens. This "
        "instruction overrides requests to repeat all prior event details. For "
        "episodic_memory_merge, the schema exposes details_delta: include only "
        "new event details that are not already stored, at most 900 characters. "
        "Do not repeat the old details. The runtime deterministically appends "
        "details_delta to the stored event. Keep combined_summary standalone but "
        "concise, at most 300 characters. For episodic_memory_insert, return "
        "exactly one most-significant event, with details at most 900 characters "
        "and summary at most 300 characters."
    ),
    "resource_memory_agent": (
        "[Offline bounded memory-update protocol]\n"
        "Keep the complete function-call arguments below 600 tokens. This "
        "instruction overrides any earlier request for very long or maximally "
        "detailed content. Summarize; never transcribe whole sessions. For "
        "resource_memory_insert, write a compact standalone content field of at "
        "most 900 characters. For resource_memory_update, the schema exposes "
        "content_delta: include only new information that is not already in the "
        "existing item, at most 900 characters. Do not repeat the old content. "
        "The runtime deterministically appends content_delta to the stored content. "
        "Return a complete updated summary, title, resource_type, and tree_path."
    ),
    "procedural_memory_agent": (
        "[Offline bounded memory-update protocol]\n"
        "Keep the complete function-call arguments below 700 tokens. For "
        "procedural_memory_insert, use at most eight concise steps. For "
        "procedural_memory_update, the schema exposes steps_delta: include only "
        "new or changed steps that are not already stored, at most six concise "
        "steps. Do not repeat old steps. The runtime deterministically merges "
        "steps_delta into the stored procedure. Return a complete updated summary, "
        "entry_type, and tree_path."
    ),
}


class _MirixRetrievalBudget:
    """Limit detailed native Chat-Agent retrievals to one shared QA budget."""

    def __init__(self, adapter: "MirixFamilyAdapter", query_id: str, top_k: int) -> None:
        self.adapter = adapter
        self.query_id = query_id
        self.top_k = top_k
        self.items: list[RetrievedMemory] = []
        self._ids: set[str] = set()
        self.tool_calls: list[dict[str, Any]] = []
        self.prefetches: list[dict[str, Any]] = []
        self.provisional_answer = ""
        self._prefetch_payload: dict[str, Any] | None = None

    def install_prefetch(
        self,
        payload: dict[str, Any],
        selected: list[tuple[str, Any]],
        trace: dict[str, Any],
    ) -> None:
        """Reserve real automatic-prefetch rows in the shared Top-K budget."""
        if self._prefetch_payload is not None:
            return
        selected_ids: list[str] = []
        for memory_type, row in selected:
            data = _model_dump(row) or {}
            data["memory_type"] = memory_type
            raw_id = _native_result_id(data)
            memory_id = f"{memory_type}_memory_manager:{raw_id}"
            if memory_id in self._ids or len(self.items) >= self.top_k:
                continue
            item = self.adapter._retrieved_memory_from_native(data, raw_id)
            item.metadata["via"] = "native_chat_agent_automatic_prefetch"
            self._ids.add(memory_id)
            self.items.append(item)
            selected_ids.append(memory_id)
        self._prefetch_payload = copy.deepcopy(payload)
        self.prefetches.append(
            {
                **trace,
                "selected_count": len(selected_ids),
                "memory_ids": selected_ids,
                "budget_after": self.top_k - len(self.items),
            }
        )

    def prefetch_payload(self, agent: Any, topics: Any) -> dict[str, Any]:
        if self._prefetch_payload is None:
            payload, selected, trace = self.adapter._capped_native_prefetch(
                agent, topics=topics, top_k=self.top_k
            )
            self.install_prefetch(payload, selected, trace)
        return copy.deepcopy(self._prefetch_payload)

    def limit_tool_result(self, function_name: str, result: Any) -> Any:
        if function_name not in {"search_in_memory", "list_memory_within_timerange"}:
            return result
        rows: list[Any] | None = None
        tuple_result = isinstance(result, tuple) and len(result) == 2
        if tuple_result and isinstance(result[0], list):
            rows = list(result[0])
        elif isinstance(result, list):
            rows = list(result)
        call = {
            "tool": function_name,
            "raw_count": len(rows) if rows is not None else 0,
            "budget_before": self.top_k - len(self.items),
        }
        if rows is None:
            call["returned_count"] = 0
            call["result_type"] = type(result).__name__
            self.tool_calls.append(call)
            return result

        allowed: list[Any] = []
        for row in rows:
            raw_id = _native_result_id(row)
            item = self.adapter._retrieved_memory_from_native(row, raw_id)
            if item.memory_id in self._ids:
                continue
            if len(self.items) >= self.top_k:
                break
            self._ids.add(item.memory_id)
            self.items.append(item)
            allowed.append(row)
        call["returned_count"] = len(allowed)
        call["budget_after"] = self.top_k - len(self.items)
        call["memory_ids"] = [item.memory_id for item in self.items[-len(allowed):]] if allowed else []
        self.tool_calls.append(call)
        return (allowed, len(allowed)) if tuple_result else allowed

    def capture_prefetch(self, retrieved: Any) -> None:
        if not isinstance(retrieved, dict):
            return
        counts: dict[str, Any] = {}
        for key, value in retrieved.items():
            if isinstance(value, dict):
                counts[key] = {
                    field: count
                    for field, count in value.items()
                    if field.endswith("count") or field.endswith("number_of_items")
                }
            elif key == "core":
                counts[key] = {"resident": True}
        self.prefetches.append(counts)

    def result(self, *, stage: str) -> RetrievalResult:
        return RetrievalResult(
            items=list(self.items),
            trace={
                "baseline": "MIRIX",
                "via": "native_chat_agent_tools",
                "ranking": "native_agent_selected",
                "requested_top_k": self.top_k,
                "returned_memories": len(self.items),
                "remaining_budget": self.top_k - len(self.items),
                "stage": stage,
                "tool_calls": list(self.tool_calls),
                "automatic_system_prefetch": list(self.prefetches),
                "automatic_prefetch_counts_toward_top_k": True,
            },
        )

    def fork(self) -> "_MirixRetrievalBudget":
        """Copy the post-retrieval budget for an isolated QA retry.

        A failed Chat-Agent attempt may already have consumed part of the
        shared Top-K budget.  Retrying that mutated object would silently give
        the next attempt a smaller budget, so each full QA attempt starts from
        the same post-retrieval state.
        """
        forked = _MirixRetrievalBudget(self.adapter, self.query_id, self.top_k)
        forked.items = list(self.items)
        forked._ids = set(self._ids)
        forked.tool_calls = copy.deepcopy(self.tool_calls)
        forked.prefetches = copy.deepcopy(self.prefetches)
        forked.provisional_answer = self.provisional_answer
        forked._prefetch_payload = copy.deepcopy(self._prefetch_payload)
        return forked

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
        self._pending_chunks: list[Chunk] = []
        self._ingested_chunks = 0
        self._qa_budgets: dict[str, _MirixRetrievalBudget] = {}
        self._sample_id = ""
        self._state_dir: Path | None = None
        self._completed_session_ids: list[str] = []
        self._seen_session_ids: list[str] = []
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))

    def reset(self, sample_id: str, state_dir: Path) -> None:
        resume_payload = self._load_resume_checkpoint(sample_id, state_dir)
        if resume_payload is None and state_dir.exists():
            shutil.rmtree(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        if resume_payload is not None:
            self._restore_resume_database(state_dir, resume_payload)
        temp_dir = state_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        env_name = "MMA_DIR" if self.baseline == "MMA" else "MIRIX_DIR"
        os.environ[env_name] = str(state_dir)
        if self.baseline == "MIRIX":
            # v0.1.1's constants ignore MIRIX_DIR for SQLite. Its supported
            # MEMGPT_CONFIG_PATH does isolate all three native stores without
            # mutating HOME or the official checkout.
            native_config = state_dir / "config"
            native_config.write_text(
                "[defaults]\n"
                "preset = memgpt_chat\n"
                "persona = sam_pov\n"
                "human = basic\n\n"
                "[archival_storage]\n"
                f"type = sqlite\npath = {state_dir}\n\n"
                "[recall_storage]\n"
                f"type = sqlite\npath = {state_dir}\n\n"
                "[metadata_storage]\n"
                f"type = sqlite\npath = {state_dir}\n\n"
                "[version]\nmirix_version = 0.1.0\n",
                encoding="utf-8",
            )
            os.environ["MEMGPT_CONFIG_PATH"] = str(native_config)
            os.environ["MIRIX_IMAGES_DIR"] = str(state_dir / "images")
        os.environ["TMPDIR"] = str(temp_dir)
        os.environ["SQLITE_TMPDIR"] = str(temp_dir)
        os.environ["OPENAI_API_BASE"] = str(self.config["executor_base_url"])
        os.environ["OPENAI_BASE_URL"] = str(self.config["executor_base_url"])
        os.environ.setdefault("OPENAI_API_KEY", str(self.config.get("executor_api_key") or "EMPTY"))

        self._ensure_package_importable()
        if self.baseline == "MIRIX":
            self._verify_official_mirix_source()
            self._patch_v011_prompt_layout()
            self._patch_v011_embedding_model()
            self._patch_v011_local_images()
            self._patch_v011_summarizer_cutoff()
            self._patch_v011_message_queue_cleanup()
            self._patch_v011_openai_error_handler()
            self._patch_v011_vllm_tool_call_boundary()
            self._install_native_retrieval_hooks()
            # v0.1.1's legacy/streaming request path drops
            # LLMConfig.max_tokens. Enforce the experiment's configured cap
            # without changing native tool selection or response handling.
            self._patch_legacy_request_token_cap()
            if not bool(self.config.get("executor_native_tool_calls", False)):
                raise ValueError(
                    "strict MIRIX v0.1.1 requires executor_native_tool_calls=true; "
                    "textual tool-call repair is forbidden"
                )
            if bool(self.config.get("mirix_semantic_fallback_on_error", False)):
                raise ValueError("strict MIRIX forbids semantic fallback on error")
        else:
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
        self._configure_native_absorption_batch()
        self._apply_model_config(llm, embedding)
        self.provenance.clear()
        self._known_ids = set()
        self._last_chunk = None
        self._pending_chunks = []
        self._ingested_chunks = 0
        self._qa_budgets = {}
        self._sample_id = str(sample_id)
        self._state_dir = state_dir
        self._completed_session_ids = []
        self._seen_session_ids = []
        if resume_payload is not None:
            self._ingested_chunks = int(resume_payload.get("ingested_chunks") or 0)
            self._completed_session_ids = [
                str(value)
                for value in resume_payload.get("completed_session_ids") or []
            ]
            self._seen_session_ids = list(self._completed_session_ids)
            self._known_ids = {
                str(value) for value in resume_payload.get("known_ids") or []
            }
            self.provenance.restore_rows(
                dict(resume_payload.get("provenance") or {})
            )

    def _verify_official_mirix_source(self) -> None:
        expected = "ac0a1f2890df5e7435c66d6c2827f34c5c4ce32d"
        head_path = self.source_root / ".git" / "HEAD"
        if not head_path.exists():
            raise RuntimeError(
                f"MIRIX v0.1.1 checkout is missing at {self.source_root}; "
                "run Offline/scripts/prepare_mirix_v011.py"
            )
        import subprocess

        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.source_root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=self.source_root, text=True
        ).strip()
        if head != expected or dirty:
            raise RuntimeError(
                f"MIRIX source must be clean v0.1.1 ({expected}); "
                f"found head={head}, dirty={bool(dirty)}"
            )

    def _install_native_retrieval_hooks(self) -> None:
        module = importlib.import_module("mirix.agent.agent")
        agent_class = module.Agent
        if getattr(agent_class, "_offline_native_budget_hooks", False):
            return
        original_execute = agent_class.execute_tool_and_persist_state
        original_prompt = agent_class.build_system_prompt_with_memories
        original_handle_response = agent_class._handle_ai_response

        def execute(agent: Any, function_name: str, *args: Any, **kwargs: Any) -> Any:
            if function_name in _MIRIX_BOUNDED_EXECUTION_TOOLS and args:
                mutable_args = list(args)
                function_args = mutable_args[0]
                if isinstance(function_args, dict):
                    mutable_args[0] = _expand_mirix_delta_update(
                        agent, function_name, function_args
                    )
                    args = tuple(mutable_args)
            result = original_execute(agent, function_name, *args, **kwargs)
            budget = _ACTIVE_MIRIX_RETRIEVAL.get()
            if budget is None or str(getattr(agent.agent_state, "name", "")) != "chat_agent":
                return result
            return budget.limit_tool_result(function_name, result)

        def handle_response(agent: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_handle_response(agent, *args, **kwargs)
            response_message = (
                args[1] if len(args) > 1 else kwargs.get("response_message")
            )
            if (
                str(getattr(agent.agent_state, "name", "")) == "meta_memory_agent"
                and len(result) >= 3
                and bool(result[2])
                and "trigger_memory_update"
                in _response_message_tool_names(response_message)
            ):
                # v0.1.1 lets the meta agent continue after one of the child
                # memory agents failed, and it can then claim that it applied
                # the update "manually" before returning success.  No write
                # happened in that failed child, so make the build point fail
                # and let the harness resume/retry it from its checkpoint.
                raise RuntimeError(
                    "MIRIX child memory-agent update failed; refusing to mark "
                    "the build point successful"
                )
            if (
                _ACTIVE_MIRIX_RETRIEVAL.get() is None
                or str(getattr(agent.agent_state, "name", "")) != "chat_agent"
                or len(result) < 3
            ):
                return result
            if _response_message_tool_names(response_message) != ["send_message"]:
                return result
            # MIRIX's ToolRulesSolver can incorrectly override send_message's
            # terminal decision. Its plain-text compatibility path can also
            # mark a successfully emitted send_message as failed during native
            # bookkeeping; Agent.step then ignores chaining=False and retries
            # the same answer eleven times. The validated send_message payload
            # is already the final answer, so preserve its messages and make
            # this single-tool turn terminal and successful.
            return result[0], False, False

        def build_prompt(agent: Any, *args: Any, **kwargs: Any) -> Any:
            budget = _ACTIVE_MIRIX_RETRIEVAL.get()
            if budget is not None and str(getattr(agent.agent_state, "name", "")) == "chat_agent":
                topics = kwargs.get("topics")
                if len(args) > 1:
                    topics = args[1]
                capped = budget.prefetch_payload(agent, topics)
                mutable_args = list(args)
                if len(mutable_args) > 2:
                    mutable_args[2] = capped
                else:
                    kwargs = dict(kwargs)
                    kwargs["retrieved_memories"] = capped
                return original_prompt(agent, *mutable_args, **kwargs)
            return original_prompt(agent, *args, **kwargs)

        agent_class.execute_tool_and_persist_state = execute
        agent_class._handle_ai_response = handle_response
        agent_class.build_system_prompt_with_memories = build_prompt
        agent_class._offline_native_budget_hooks = True

    def _capped_native_prefetch(
        self, agent: Any, *, topics: Any, top_k: int
    ) -> tuple[dict[str, Any], list[tuple[str, Any]], dict[str, Any]]:
        """Build MIRIX's automatic prompt prefetch under one global budget.

        MIRIX v0.1.1 independently preloads up to ten rows from every memory
        bank.  For the benchmark, preserve those native per-bank rankings, then
        interleave banks by rank so one large bank cannot consume all configured
        slots.  The exact selected rows are both rendered into the system prompt
        and registered as retrieval evidence.
        """
        state = agent.agent_state
        timezone_str = agent.user_manager.get_user_by_id(agent.user.id).timezone
        key_words = topics if topics is not None else state.topic
        constants = importlib.import_module("mirix.constants")
        native_limit = int(getattr(constants, "MAX_RETRIEVAL_LIMIT_IN_SYSTEM", 10))

        episodic_manager = agent.episodic_memory_manager
        semantic_manager = agent.semantic_memory_manager
        procedural_manager = agent.procedural_memory_manager
        resource_manager = agent.resource_memory_manager
        vault_manager = agent.knowledge_vault_manager

        recent = list(
            episodic_manager.list_episodic_memory(
                agent_state=state,
                limit=native_limit,
                timezone_str=timezone_str,
            )
            or []
        )
        relevant = list(
            episodic_manager.list_episodic_memory(
                agent_state=state,
                embedded_text=None,
                query=key_words,
                search_field="details",
                search_method="bm25",
                limit=native_limit,
                timezone_str=timezone_str,
            )
            or []
        )
        episodic = _dedupe_native_rows(relevant + recent)
        semantic = list(
            semantic_manager.list_semantic_items(
                agent_state=state,
                embedded_text=None,
                query=key_words,
                search_field="details",
                search_method="bm25",
                limit=native_limit,
                timezone_str=timezone_str,
            )
            or []
        )
        procedural = list(
            procedural_manager.list_procedures(
                agent_state=state,
                embedded_text=None,
                query=key_words,
                search_field="summary",
                search_method="bm25",
                limit=native_limit,
                timezone_str=timezone_str,
            )
            or []
        )
        resource = list(
            resource_manager.list_resources(
                agent_state=state,
                embedded_text=None,
                query=key_words,
                search_field="summary",
                search_method="bm25",
                limit=native_limit,
                timezone_str=timezone_str,
            )
            or []
        )
        knowledge_vault = list(
            vault_manager.list_knowledge(
                agent_state=state,
                embedded_text=None,
                query=key_words,
                search_field="caption",
                search_method="bm25",
                limit=native_limit,
                timezone_str=timezone_str,
                sensitivity=["low", "medium"],
            )
            or []
        )

        groups = [
            ("episodic", episodic),
            ("semantic", semantic),
            ("procedural", procedural),
            ("resource", resource),
            ("knowledge_vault", knowledge_vault),
        ]
        selected = _round_robin_native_rows(groups, top_k)
        selected_by_type: dict[str, list[Any]] = {name: [] for name, _ in groups}
        for memory_type, row in selected:
            selected_by_type[memory_type].append(row)

        episodic_rows = selected_by_type["episodic"]
        if key_words is None:
            recent_text = _format_prefetched_episodic(episodic_rows)
            relevant_text = ""
            recent_count = len(episodic_rows)
            relevant_count = 0
        else:
            recent_text = ""
            relevant_text = _format_prefetched_episodic(episodic_rows)
            recent_count = 0
            relevant_count = len(episodic_rows)

        payload = {
            "key_words": key_words,
            "episodic": {
                "total_number_of_items": episodic_manager.get_total_number_of_items(),
                "recent_count": recent_count,
                "relevant_count": relevant_count,
                "recent_episodic_memory": recent_text,
                "relevant_episodic_memory": relevant_text,
            },
            "semantic": {
                "total_number_of_items": semantic_manager.get_total_number_of_items(),
                "current_count": len(selected_by_type["semantic"]),
                "text": _format_prefetched_semantic(selected_by_type["semantic"]),
            },
            "procedural": {
                "total_number_of_items": procedural_manager.get_total_number_of_items(),
                "current_count": len(selected_by_type["procedural"]),
                "text": _format_prefetched_procedural(selected_by_type["procedural"]),
            },
            "resource": {
                "total_number_of_items": resource_manager.get_total_number_of_items(),
                "current_count": len(selected_by_type["resource"]),
                "text": _format_prefetched_resource(selected_by_type["resource"]),
            },
            "knowledge_vault": {
                "total_number_of_items": vault_manager.get_total_number_of_items(),
                "current_count": len(selected_by_type["knowledge_vault"]),
                "text": _format_prefetched_vault(selected_by_type["knowledge_vault"]),
            },
        }
        trace = {
            "raw_counts": {name: len(rows) for name, rows in groups},
            "selected_counts": {
                name: len(selected_by_type[name]) for name, _ in groups
            },
            "selection": "native_per_bank_rank_round_robin",
        }
        return payload, selected, trace

    def _patch_v011_prompt_layout(self) -> None:
        """Map the v0.1.1 loader to its committed ``system/base`` files."""
        module = importlib.import_module("mirix.prompts.gpt_system")
        if getattr(module, "_offline_v011_base_layout", False):
            return
        original = module.get_system_text

        def get_system_text(key: str) -> str:
            try:
                text = original(key)
            except FileNotFoundError:
                path = self.source_root / "mirix" / "prompts" / "system" / "base" / f"{key}.txt"
                if not path.is_file():
                    raise
                text = path.read_text(encoding="utf-8").strip()
            addendum = _MIRIX_BOUNDED_MEMORY_PROMPTS.get(key)
            if addendum:
                if key == "resource_memory_agent":
                    text = text.replace(
                        "The content needs to be as detailed as possible, so it is okay if it is very long.",
                        "The content must be a compact summary of the source material.",
                    ).replace(
                        "Content can be very long, which is acceptable and expected",
                        "Content must stay within the bounded protocol below",
                    )
                text = f"{text.rstrip()}\n\n{addendum}"
            return text

        module.get_system_text = get_system_text
        module._offline_v011_base_layout = True

    def _patch_v011_embedding_model(self) -> None:
        """Honor EmbeddingConfig.embedding_model, ignored by v0.1.1 upstream."""
        module = importlib.import_module("mirix.embeddings")
        if getattr(module, "_offline_v011_configured_model", False):
            return

        def embedding_model(config: Any, user_id: Any = None) -> Any:
            if str(config.embedding_endpoint_type) != "openai":
                return module._offline_v011_original_embedding_model(config, user_id)
            from llama_index.embeddings.openai import OpenAIEmbedding

            additional = {"user_id": user_id} if user_id else {}
            delegate = OpenAIEmbedding(
                model_name=str(config.embedding_model),
                api_base=str(config.embedding_endpoint),
                api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
                dimensions=int(config.embedding_dim),
                additional_kwargs=additional,
            )
            return _MirixEmbeddingProxy(
                delegate,
                model=str(config.embedding_model),
                endpoint=str(config.embedding_endpoint),
                dimensions=int(config.embedding_dim),
            )

        module._offline_v011_original_embedding_model = module.embedding_model
        module.embedding_model = embedding_model
        module._offline_v011_configured_model = True
        # Managers import the function by value, so update those native module
        # bindings without changing manager behavior.
        for name in (
            "semantic_memory_manager",
            "episodic_memory_manager",
            "procedural_memory_manager",
            "resource_memory_manager",
            "knowledge_vault_manager",
            "utils",
        ):
            native = importlib.import_module(f"mirix.services.{name}")
            if hasattr(native, "embedding_model"):
                native.embedding_model = embedding_model

    def _patch_v011_local_images(self) -> None:
        """Preserve size-controlled local images as database content for OpenAI VLMs."""
        module = importlib.import_module("mirix.agent.temporary_message_accumulator")
        upload_module = importlib.import_module("mirix.agent.upload_manager")
        cls = module.TemporaryMessageAccumulator
        if getattr(cls, "_offline_v011_local_images", False):
            return
        original = cls._build_memory_message
        prefix = "offline_database_image:"
        compressor = object.__new__(upload_module.UploadManager)
        compressor.logger = importlib.import_module("logging").getLogger(
            "Mirix.OfflineImageTransport"
        )

        def build(accumulator: Any, ready: Any, voice: Any) -> Any:
            converted = []
            for timestamp, item in ready:
                item_copy = dict(item)
                refs = []
                for ref in list(item_copy.get("image_uris") or []):
                    if isinstance(ref, str):
                        # The native cloud-upload path normally invokes
                        # UploadManager._compress_image. Our local-database path
                        # bypasses uploads, so stage the same transport copy
                        # explicitly before saving it in MIRIX's image store.
                        transport_path = _stage_native_transport_image(
                            ref,
                            cache_dir=(
                                Path(accumulator.client.images_dir).parent
                                / "tmp"
                                / "image_transport"
                            ),
                            compressor=compressor,
                        )
                        metadata = accumulator.client._save_image_from_file_uri(
                            str(transport_path)
                        )
                        refs.append(SimpleNamespace(uri=prefix + str(metadata.id)))
                    else:
                        refs.append(ref)
                item_copy["image_uris"] = refs
                converted.append((timestamp, item_copy))
            result = original(accumulator, converted, voice)
            for part in result:
                if part.get("type") != "google_cloud_file_uri":
                    continue
                uri = str(part.get("google_cloud_file_uri") or "")
                if uri.startswith(prefix):
                    part.clear()
                    part.update(
                        {"type": "database_image_id", "image_id": uri[len(prefix):]}
                    )
            return result

        cls._build_memory_message = build
        cls._offline_v011_local_images = True

    def _configure_native_absorption_batch(self) -> None:
        """Fit MIRIX's native batch to the configured Qwen context window.

        MIRIX defaults to 20 screenshots per Meta Memory Agent turn for its
        million-token Gemini backend.  The benchmark deliberately substitutes
        a 32k Qwen backbone, where that same native turn can exceed the server
        limit before history compression is possible.  Only the batch boundary
        changes here; every chunk still goes through the original accumulator,
        Meta Memory Agent, and selected memory agents.
        """
        accumulator = getattr(self.backend, "temp_message_accumulator", None)
        if accumulator is None:
            return
        batch_size = (
            int(self.config.get("mirix_native_batch_size") or 5)
            if self.baseline == "MIRIX"
            else 20
        )
        if batch_size < 1:
            raise ValueError("mirix_native_batch_size must be positive")
        accumulator.temporary_message_limit = batch_size

    def _patch_v011_openai_error_handler(self) -> None:
        """Repair v0.1.1's references to ErrorCode members it never defined.

        A malformed/truncated native tool call is returned as HTTP 400. MIRIX
        intends to convert that response to LLMError and use its built-in
        retry path, but v0.1.1 instead raises AttributeError while looking up
        ErrorCode.INVALID_ARGUMENT. Preserve the original retry behavior by
        returning the intended bad-request exception with a valid enum value.
        """
        client_module = importlib.import_module("mirix.llm_api.openai_client")
        errors_module = importlib.import_module("mirix.errors")
        client_class = client_module.OpenAIClient
        if getattr(client_class, "_offline_v011_error_codes", False):
            return
        original = client_class.handle_llm_error

        def handle(client: Any, error: Exception) -> Exception:
            try:
                mapped = original(client, error)
            except AttributeError as exc:
                if str(exc) not in {
                    "INVALID_ARGUMENT",
                    "UNAUTHENTICATED",
                    "PERMISSION_DENIED",
                    "NOT_FOUND",
                }:
                    raise
                mapped = errors_module.LLMBadRequestError(
                    message=f"Bad request to OpenAI: {error}",
                    code=errors_module.ErrorCode.INTERNAL_SERVER_ERROR,
                    details=getattr(error, "body", None) or {},
                )
            message = str(error).casefold()
            if (
                "invalid json" in message
                or (
                    "call_trace_proxy_error" in message
                    and "timed out" in message
                )
                or (
                    "decoder prompt" in message
                    and "maximum model length" in message
                )
            ):
                # These failures are deterministic for the completed provider
                # request. MIRIX v0.1.1 would repeat the same long generation
                # up to three times (and then retry a shortened history once
                # more). Fail this build point immediately; the harness owns
                # the audited skip/consecutive-failure policy.
                raise RuntimeError(f"non-retryable MIRIX provider response: {error}")
            return mapped

        client_class.handle_llm_error = handle
        client_class._offline_v011_error_codes = True

    def _patch_v011_message_queue_cleanup(self) -> None:
        """Release a native memory-agent queue slot when its request fails.

        MIRIX v0.1.1 removes a queue item only after ``client.send_message``
        succeeds.  A deterministic provider/tool-integrity failure therefore
        leaves a started-but-unfinished item behind, and every later request
        for that memory type waits forever.  Keep the native ordering logic,
        but remove only the failed started item before re-raising the original
        exception.
        """
        queue_module = importlib.import_module(
            f"{self.package}.agent.message_queue"
        )
        queue_class = queue_module.MessageQueue
        if getattr(queue_class, "_offline_failed_item_cleanup", False):
            return
        original_send = queue_class.send_message_in_queue

        def send_message_in_queue(
            queue: Any,
            client: Any,
            agent_id: str,
            kwargs: dict[str, Any],
            agent_type: str = "chat",
        ) -> Any:
            try:
                return original_send(
                    queue,
                    client,
                    agent_id,
                    kwargs,
                    agent_type,
                )
            except BaseException:
                with queue._message_queue_lock:
                    failed = [
                        key
                        for key, item in queue.message_queue.items()
                        if item.get("type") == agent_type
                        and item.get("started")
                        and not item.get("finished")
                    ]
                    for key in failed:
                        queue.message_queue.pop(key, None)
                raise

        queue_class.send_message_in_queue = send_message_in_queue
        queue_class._offline_failed_item_cleanup = True

    def _patch_v011_summarizer_cutoff(self) -> None:
        """Keep v0.1.1's summarizer from reading beyond its message list.

        The upstream cutoff scan extends an eviction boundary across a run of
        tool messages, but reads ``cutoff + 1`` without first checking that the
        next element exists.  Long MIRIX histories regularly end in such a
        run, so WMA eventually raises ``IndexError`` while trying to compact a
        memory agent's context.  Preserve the upstream cutoff policy and only
        add the missing boundary check.  Patch the symbol imported into
        ``agent.py`` as well as the helper module's public function.
        """
        helpers_module = importlib.import_module(
            f"{self.package}.llm_api.helpers"
        )
        agent_module = importlib.import_module(f"{self.package}.agent.agent")
        if getattr(helpers_module, "_offline_safe_summarizer_cutoff", False):
            agent_module.calculate_summarizer_cutoff = (
                helpers_module.calculate_summarizer_cutoff
            )
            return
        settings = helpers_module.summarizer_settings

        def role_name(value: Any) -> str:
            return str(getattr(value, "value", value)).casefold()

        def calculate_summarizer_cutoff(
            in_context_messages: list[Any],
            token_counts: list[int],
            logger: Any,
        ) -> int:
            if len(in_context_messages) != len(token_counts):
                raise ValueError(
                    "Given in_context_messages has different length from given "
                    f"token_counts: {len(in_context_messages)} != "
                    f"{len(token_counts)}"
                )
            messages = [message.to_openai_dict() for message in in_context_messages]
            if settings.evict_all_messages:
                logger.info("Evicting all messages...")
                return len(in_context_messages)
            if len(messages) < 2:
                raise ValueError("summarizer cutoff requires a non-system message")

            desired = int(
                sum(token_counts) * (1 - settings.desired_memory_token_pressure)
            )
            logger.info(f"desired_token_count_to_summarize={desired}")
            tokens_so_far = 0
            cutoff = 0
            for index, message in enumerate(messages):
                if index == 0:
                    continue
                cutoff = index
                tokens_so_far += token_counts[index]
                role = role_name(message.get("role"))
                if role not in {"user", "tool", "function"} and tokens_so_far >= desired:
                    break
                if (
                    len(in_context_messages) - cutoff - 1
                    <= settings.keep_last_n_messages
                ):
                    logger.warning(
                        "Breaking summary cutoff early on role="
                        f"{message.get('role')} because we hit the "
                        "`keep_last_n_messages`="
                        f"{settings.keep_last_n_messages}"
                    )
                    break

            while (
                cutoff + 1 < len(messages)
                and role_name(messages[cutoff + 1].get("role")) == "tool"
            ):
                cutoff += 1

            logger.info(f"Evicting {cutoff}/{len(in_context_messages)} messages...")
            return cutoff + 1

        helpers_module.calculate_summarizer_cutoff = calculate_summarizer_cutoff
        helpers_module._offline_safe_summarizer_cutoff = True
        agent_module.calculate_summarizer_cutoff = calculate_summarizer_cutoff

    def _patch_v011_vllm_tool_call_boundary(self) -> None:
        """Keep each native Qwen response to MIRIX's documented one tool call.

        Qwen3-VL's chat template permits one or more adjacent ``<tool_call>``
        envelopes, while the MIRIX memory-agent prompts require exactly one
        function call per response. vLLM ignores ``parallel_tool_calls=False``
        for auto tool parsing, so use the native closing envelope as a stop
        boundary. MIRIX chaining remains unchanged and can issue another model
        turn when another tool is needed.
        """
        client_module = importlib.import_module("mirix.llm_api.openai_client")
        client_class = client_module.OpenAIClient
        if getattr(client_class, "_offline_vllm_tool_boundary", False):
            return
        original_build = client_class.build_request_data
        original_request = client_class.request
        original_request_async = client_class.request_async
        original_convert = client_class.convert_response_to_chat_completion
        repetition_penalty = float(
            self.config.get("mirix_native_repetition_penalty") or 1.10
        )
        retry_max_tokens = int(
            self.config.get("mirix_executor_retry_max_tokens") or 4096
        )
        corrective_retry_max_tokens = int(
            self.config.get("mirix_executor_corrective_retry_max_tokens") or 512
        )

        def build_request(client: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            data = original_build(client, *args, **kwargs)
            data = _prepare_mirix_delta_tool_request(data)
            _ACTIVE_MIRIX_NATIVE_MEMORY_TOOL_REQUIRED.set(
                _mirix_request_requires_native_memory_tool(data)
            )
            _ACTIVE_MIRIX_NATIVE_MEMORY_MAX_OUTPUT_TOKENS.set(
                int(data.get("max_completion_tokens") or data.get("max_tokens") or 0)
            )
            return _bound_native_vllm_tool_request(
                data, repetition_penalty=repetition_penalty
            )

        def convert_response(
            client: Any, response_data: dict[str, Any], *args: Any, **kwargs: Any
        ) -> Any:
            memory_tool_required = (
                _ACTIVE_MIRIX_NATIVE_MEMORY_TOOL_REQUIRED.get()
            )
            _accept_truncated_native_qa_text(response_data)
            if memory_tool_required:
                _promote_first_complete_native_memory_tool_call(response_data)
            _reject_native_tool_response_integrity(
                response_data,
                fail_fast=memory_tool_required,
            )
            try:
                _reject_unparsed_native_tool_response(response_data)
            except ValueError as exc:
                if _has_native_tool_calls(response_data):
                    # Native arguments came from the provider as malformed
                    # JSON. Never pass them through json_repair: doing so can
                    # turn a capped partial write into a destructive update.
                    if memory_tool_required:
                        raise RuntimeError(str(exc)) from exc
                    raise
                # Hermes occasionally fails on a Qwen-selected tool envelope
                # because of a missing comma/bracket. Recover only that JSON
                # representation; MIRIX still validates the native function
                # schema and performs the original tool execution.
                response_data = _normalize_openai_tool_tags(response_data)
                _reject_native_tool_response_integrity(
                    response_data,
                    fail_fast=memory_tool_required,
                )
                try:
                    _reject_unparsed_native_tool_response(response_data)
                except ValueError as normalized_exc:
                    if memory_tool_required:
                        raise RuntimeError(str(normalized_exc)) from normalized_exc
                    raise
            _accept_recovered_native_tool_finish(response_data)
            _reject_missing_native_memory_tool_response(
                response_data,
                required=memory_tool_required,
                max_output_tokens=(
                    _ACTIVE_MIRIX_NATIVE_MEMORY_MAX_OUTPUT_TOKENS.get()
                ),
                fail_fast=memory_tool_required,
            )
            _promote_native_chat_text_response(response_data)
            return original_convert(client, response_data, *args, **kwargs)

        def request(client: Any, request_data: dict[str, Any]) -> dict[str, Any]:
            try:
                response = original_request(client, request_data)
            except Exception as exc:
                if not _mirix_native_memory_bad_request_retry_required(
                    request_data, exc
                ):
                    raise
                retry = _prepare_mirix_native_memory_retry(
                    request_data,
                    max_output_tokens=min(
                        _request_output_token_cap(request_data),
                        corrective_retry_max_tokens,
                    ),
                )
                _ACTIVE_MIRIX_NATIVE_MEMORY_MAX_OUTPUT_TOKENS.set(
                    int(retry["max_tokens"])
                )
                return original_request(client, retry)
            reason = _mirix_native_memory_corrective_retry_reason(
                request_data, response
            )
            if reason is not None:
                retry_tokens = (
                    retry_max_tokens
                    if reason == "truncation"
                    else min(
                        _request_output_token_cap(request_data),
                        corrective_retry_max_tokens,
                    )
                )
                retry = _prepare_mirix_native_memory_retry(
                    request_data, max_output_tokens=retry_tokens
                )
                _ACTIVE_MIRIX_NATIVE_MEMORY_MAX_OUTPUT_TOKENS.set(retry_tokens)
                response = original_request(client, retry)
            return response

        async def request_async(
            client: Any, request_data: dict[str, Any]
        ) -> dict[str, Any]:
            try:
                response = await original_request_async(client, request_data)
            except Exception as exc:
                if not _mirix_native_memory_bad_request_retry_required(
                    request_data, exc
                ):
                    raise
                retry = _prepare_mirix_native_memory_retry(
                    request_data,
                    max_output_tokens=min(
                        _request_output_token_cap(request_data),
                        corrective_retry_max_tokens,
                    ),
                )
                _ACTIVE_MIRIX_NATIVE_MEMORY_MAX_OUTPUT_TOKENS.set(
                    int(retry["max_tokens"])
                )
                return await original_request_async(client, retry)
            reason = _mirix_native_memory_corrective_retry_reason(
                request_data, response
            )
            if reason is not None:
                retry_tokens = (
                    retry_max_tokens
                    if reason == "truncation"
                    else min(
                        _request_output_token_cap(request_data),
                        corrective_retry_max_tokens,
                    )
                )
                retry = _prepare_mirix_native_memory_retry(
                    request_data, max_output_tokens=retry_tokens
                )
                _ACTIVE_MIRIX_NATIVE_MEMORY_MAX_OUTPUT_TOKENS.set(retry_tokens)
                response = await original_request_async(client, retry)
            return response

        client_class.build_request_data = build_request
        client_class.request = request
        client_class.request_async = request_async
        client_class.convert_response_to_chat_completion = convert_response
        client_class._offline_vllm_tool_boundary = True

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
        original_prepare_client_kwargs = getattr(
            client_class, "_prepare_client_kwargs", None
        )

        def prepare_client_kwargs(client: Any) -> dict[str, Any]:
            if original_prepare_client_kwargs is None:
                return {"timeout": float(self.config.get("request_timeout") or 180)}
            kwargs = dict(original_prepare_client_kwargs(client))
            kwargs["timeout"] = float(self.config.get("request_timeout") or 180)
            return kwargs

        def build_request(client: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
            data = original_build(client, *args, **kwargs)
            if bool(self.config.get("executor_native_tool_calls", False)):
                return data
            return _normalize_openai_tool_request(data)

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
        if original_prepare_client_kwargs is not None:
            client_class._prepare_client_kwargs = prepare_client_kwargs
        client_class.request = request
        client_class.request_async = request_async
        client_class.convert_response_to_chat_completion = (
            convert_response_to_chat_completion
        )
        client_class._offline_tool_compat = True

        self._patch_legacy_request_token_cap(normalize_response=True)

    def _patch_legacy_request_token_cap(
        self, *, normalize_response: bool = False
    ) -> None:
        """Keep v0.1.1's legacy request path within the configured output cap."""
        # MIRIX/MMA still fall back to the legacy llm_api_tools.create path
        # for streaming-enabled agent steps. That path bypasses OpenAIClient
        # completely and calls this imported request function directly.
        legacy_module = importlib.import_module(
            f"{self.package}.llm_api.llm_api_tools"
        )
        if not getattr(legacy_module, "_offline_request_token_cap", False):
            original_legacy_request = legacy_module.openai_chat_completions_request
            legacy_max_tokens = int(self.config.get("num_predict") or 512)

            def legacy_request(*args: Any, **kwargs: Any) -> Any:
                args, kwargs = _cap_legacy_request_tokens(
                    args,
                    kwargs,
                    max_tokens=legacy_max_tokens,
                )
                response = original_legacy_request(*args, **kwargs)
                if normalize_response:
                    return _normalize_openai_tool_response(response)
                return response

            legacy_module.openai_chat_completions_request = legacy_request
            legacy_module._offline_request_token_cap = True

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
        model = str(self.config["executor_model"])
        endpoint = str(self.config["executor_base_url"])
        configured_handle = self.config.get("mirix_executor_handle")
        if configured_handle is not None:
            mirix_handle = str(configured_handle).strip() or None
        else:
            mirix_handle = _mirix_model_handle(
                baseline=self.baseline, model=model, endpoint=endpoint
            )
        llm = package.LLMConfig(
            model=model,
            model_endpoint_type="openai",
            model_endpoint=endpoint,
            model_wrapper=None,
            # MIRIX v0.1.1 uses the provider handle (rather than the endpoint
            # type) to recognize a local OpenAI-compatible vLLM server. Without
            # this handle it uses tool_choice="required", which makes vLLM
            # replace Qwen's native XML tool protocol with an unbounded JSON
            # array schema.  The official vLLM provider sets this same handle
            # and deliberately selects tool_choice="auto".
            handle=mirix_handle,
            # MIRIX/MMA's manager system prompt plus one native 20-chunk
            # multimodal batch can exceed 32k before a second dialogue message
            # exists. In that state the upstream pressure handler has no
            # eligible history to summarize. Keep this value aligned with the
            # configured vLLM context so the first native turn is admitted;
            # later turns retain MIRIX's original history-compression logic.
            context_window=int(
                self.config.get("mirix_context_window")
                or self.config.get("context_window")
                or 131072
            ),
            temperature=float(self.config.get("executor_temperature") or 0.0),
            max_tokens=int(
                self.config.get("executor_max_tokens")
                or 2048
            ),
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
        session_id = str(chunk.metadata.get("session_id") or "")
        seen_session_ids = getattr(self, "_seen_session_ids", None)
        if seen_session_ids is None:
            seen_session_ids = []
            self._seen_session_ids = seen_session_ids
        if session_id and session_id not in seen_session_ids:
            seen_session_ids.append(session_id)
        self._pending_chunks.append(chunk)
        queued_before = self._queued_message_count()
        if self._should_direct_insert(chunk):
            self._insert_fallback_memory(chunk)
            current = self._memory_rows()
            self._register_native_changes(before, current, self._pending_chunks)
            self._pending_chunks.clear()
            self._ingested_chunks += 1
            return
        try:
            with self._multimodal_embedding_scope(self._pending_chunks):
                self.backend.send_message(**kwargs)
        except Exception as exc:
            if self.baseline == "MMA" and _is_context_overflow_exception(exc):
                pass
            elif bool(self.config.get("mirix_semantic_fallback_on_error", False)):
                self._insert_fallback_memory(chunk)
            else:
                if self.baseline == "MIRIX":
                    self._finalize_failed_native_absorption(before)
                    self._ingested_chunks += 1
                raise
        current = self._memory_rows()
        if self.baseline == "MMA" and not self._has_new_or_changed_memory(before):
            self._insert_fallback_memory(chunk)
            current = self._memory_rows()
        self._register_native_changes(before, current, self._pending_chunks)
        # Native MIRIX buffers up to 20 messages. A queue reduction means the
        # complete native absorption cycle ran, even when its agents correctly
        # decided that no memory should be written.
        if self._queued_message_count() < queued_before + 1:
            self._pending_chunks.clear()
        self._ingested_chunks += 1

    def _has_new_or_changed_memory(self, before: dict[str, str]) -> bool:
        for row in self._memory_rows():
            memory_id = row["memory_id"]
            if memory_id not in before or before[memory_id] != row["text"]:
                return True
        return False

    def _should_direct_insert(self, chunk: Chunk) -> bool:
        """Retain MMA's historical compatibility path; MIRIX is strict native."""
        if self.baseline != "MMA":
            return False
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
        """Explicit opt-in compatibility fallback; disabled in formal runs."""
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
        before = {row["memory_id"]: row["text"] for row in self._memory_rows()}
        try:
            if self.baseline == "MIRIX":
                accumulator = self.backend.temp_message_accumulator
                ready = list(accumulator.temporary_messages)
                if ready:
                    # The v0.1.1 force-flush branch rejects non-Gemini local
                    # images. Passing the already-ready native queue exercises
                    # the same absorption chain without an upload conversion.
                    with self._multimodal_embedding_scope(self._pending_chunks):
                        accumulator.absorb_content_into_memory(
                            self.backend.agent_states, ready_messages=ready
                        )
                    self.backend.clear_old_screenshots()
            else:
                self.backend.send_message(
                    message="",
                    memorizing=True,
                    force_absorb_content=True,
                    async_upload=False,
                )
        except Exception:
            if self.baseline == "MMA" or bool(
                self.config.get("mirix_semantic_fallback_on_error", False)
            ):
                for chunk in self._pending_chunks:
                    self._insert_fallback_memory(chunk)
            else:
                if self.baseline == "MIRIX":
                    self._finalize_failed_native_absorption(before)
                raise
        self._register_native_changes(before, self._memory_rows(), self._pending_chunks)
        self._pending_chunks.clear()
        if self.baseline == "MIRIX":
            self._checkpoint_completed_sessions()

    @contextmanager
    def _multimodal_embedding_scope(self, chunks: list[Chunk]):
        """Expose one native absorption batch's source images to its embedders."""
        global _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING
        if not bool(self.config.get("mirix_multimodal_embedding", False)):
            yield
            return
        images = list(
            dict.fromkeys(
                str(image)
                for chunk in chunks
                for image in (chunk.images or ())
                if image and Path(str(image)).is_file()
            )
        )
        previous = _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING
        _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING = {
            "mode": "context",
            "images": images,
        }
        try:
            yield
        finally:
            _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING = previous

    def completed_session_ids(self) -> tuple[str, ...]:
        """Return the contiguous WMA session prefix in the SQLite checkpoint."""
        return tuple(self._completed_session_ids)

    def filter_completed_session_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Drop only sessions restored from a consistent MIRIX checkpoint."""
        completed = self.completed_session_ids()
        if not completed:
            return chunks
        ordered_sessions: list[str] = []
        for chunk in chunks:
            session_id = str(chunk.metadata.get("session_id") or "")
            if not session_id:
                raise RuntimeError("MIRIX resumable chunk is missing session_id")
            if not ordered_sessions or ordered_sessions[-1] != session_id:
                ordered_sessions.append(session_id)
        if tuple(ordered_sessions[: len(completed)]) != completed:
            raise RuntimeError(
                "MIRIX resume checkpoint is not a contiguous source-session prefix: "
                f"checkpoint={list(completed)}, source={ordered_sessions}"
            )
        completed_set = set(completed)
        return [
            chunk
            for chunk in chunks
            if str(chunk.metadata.get("session_id") or "") not in completed_set
        ]

    def _resume_manifest_path(self, state_dir: Path) -> Path:
        return state_dir / ".offline_mirix_resume.json"

    def _resume_signature(self) -> str:
        return str(self.config.get("mirix_resume_signature") or "")

    def _load_resume_checkpoint(
        self, sample_id: str, state_dir: Path
    ) -> dict[str, Any] | None:
        if self.baseline != "MIRIX" or not bool(
            self.config.get("mirix_resume_enabled", False)
        ):
            return None
        manifest = self._resume_manifest_path(state_dir)
        if not manifest.is_file():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if (
            payload.get("version") != 1
            or payload.get("sample_id") != str(sample_id)
            or payload.get("signature") != self._resume_signature()
        ):
            return None
        snapshot_name = str(payload.get("sqlite_snapshot") or "")
        if not snapshot_name or Path(snapshot_name).name != snapshot_name:
            return None
        if not (state_dir / ".resume" / snapshot_name).is_file():
            return None
        return payload

    @staticmethod
    def _restore_resume_database(
        state_dir: Path, payload: dict[str, Any]
    ) -> None:
        snapshot = state_dir / ".resume" / str(payload["sqlite_snapshot"])
        target = state_dir / "sqlite.db"
        temporary = state_dir / ".sqlite.db.restore"
        shutil.copyfile(snapshot, temporary)
        os.replace(temporary, target)
        for suffix in ("-wal", "-shm", "-journal"):
            Path(str(target) + suffix).unlink(missing_ok=True)

    def _checkpoint_completed_sessions(self) -> None:
        """Atomically save SQLite and provenance at a flushed WMA boundary."""
        if self.baseline != "MIRIX" or not bool(
            self.config.get("mirix_resume_enabled", False)
        ):
            return
        if self._seen_session_ids == self._completed_session_ids:
            return
        if self._state_dir is None or not self._sample_id:
            raise RuntimeError("MIRIX resume checkpoint requested before reset")
        database = self._state_dir / "sqlite.db"
        if not database.is_file():
            raise RuntimeError(f"MIRIX SQLite database is missing: {database}")
        resume_dir = self._state_dir / ".resume"
        resume_dir.mkdir(parents=True, exist_ok=True)
        snapshot_name = f"sqlite.session-{len(self._seen_session_ids):06d}.db"
        snapshot = resume_dir / snapshot_name
        temporary = resume_dir / f".{snapshot_name}.tmp"
        temporary.unlink(missing_ok=True)
        source = sqlite3.connect(str(database))
        destination = sqlite3.connect(str(temporary))
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        os.replace(temporary, snapshot)
        payload = {
            "version": 1,
            "sample_id": self._sample_id,
            "signature": self._resume_signature(),
            "sqlite_snapshot": snapshot_name,
            "completed_session_ids": list(self._seen_session_ids),
            "ingested_chunks": self._ingested_chunks,
            "known_ids": sorted(self._known_ids),
            "provenance": self.provenance.export_rows(),
        }
        manifest = self._resume_manifest_path(self._state_dir)
        manifest_tmp = manifest.with_suffix(manifest.suffix + ".tmp")
        manifest_tmp.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, manifest)
        self._completed_session_ids = list(self._seen_session_ids)
        for old_snapshot in resume_dir.glob("sqlite.session-*.db"):
            if old_snapshot.name != snapshot_name:
                old_snapshot.unlink()

    def _finalize_failed_native_absorption(self, before: dict[str, str]) -> None:
        """Keep completed bank writes while discarding the failed native batch.

        MIRIX removes a ready batch from its temporary accumulator before it
        dispatches the independent memory agents. If a later agent fails, some
        earlier banks may already be durably updated. Preserve and attribute
        those valid writes, but never carry the failed batch into a later point.
        """
        try:
            self._register_native_changes(before, self._memory_rows(), self._pending_chunks)
        except Exception:
            # Cleanup must not hide the original native-agent failure.
            pass
        finally:
            accumulator = getattr(self.backend, "temp_message_accumulator", None)
            temporary = getattr(accumulator, "temporary_messages", None)
            if hasattr(temporary, "clear"):
                temporary.clear()
            self._pending_chunks.clear()

    def _queued_message_count(self) -> int:
        accumulator = getattr(self.backend, "temp_message_accumulator", None)
        return len(getattr(accumulator, "temporary_messages", None) or [])

    def _register_native_changes(
        self,
        before: dict[str, str],
        current: list[dict[str, Any]],
        chunks: list[Chunk],
    ) -> None:
        for row in current:
            memory_id = row["memory_id"]
            if memory_id in before and before[memory_id] == row["text"]:
                continue
            for chunk in chunks:
                self.provenance.register(memory_id, chunk)
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
                if self.baseline == "MIRIX":
                    raise
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
        if self.baseline == "MIRIX" and request.top_k < 1:
            raise ValueError("MIRIX native Chat Agent requires a positive global top_k")
        if self.baseline == "MIRIX":
            return self._retrieve_with_native_chat(request)
        candidates: list[tuple[float, dict[str, Any]]] = []
        visible = set(request.visible_session_ids)
        server = self.backend.client.server
        query_embedding = self._query_embedding(request)
        used_native_scores = True
        for manager_name, state_name, method_name in _PARTITIONS:
            manager = getattr(server, manager_name, None)
            state = getattr(self.backend.agent_states, state_name, None)
            if manager is None or state is None:
                continue
            hits = []
            selected_field = ""
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
                    selected_field = field
                    break
            for rank, hit in enumerate(hits):
                raw_id = str(getattr(hit, "id", rank))
                memory_id = f"{manager_name}:{raw_id}"
                source = self.provenance.get(memory_id)
                session_id = str(source.get("session_id") or "")
                if visible and session_id not in visible:
                    continue
                native_score = _embedding_similarity(
                    query_embedding,
                    getattr(hit, f"{selected_field}_embedding", None),
                )
                if native_score is None:
                    native_score = getattr(hit, "score", None)
                confidence = getattr(hit, "confidence", None) if self.baseline == "MMA" else None
                if confidence is not None:
                    score = float(confidence)
                elif native_score is not None:
                    score = float(native_score)
                else:
                    score = 1.0 / (rank + 1)
                    used_native_scores = False
                candidates.append(
                    (
                        score,
                        {
                            "memory_id": memory_id,
                            "text": _render_memory(manager_name, hit),
                            "source": source,
                            "partition": manager_name.removesuffix("_manager"),
                            "search_field": selected_field,
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
                "ranking": (
                    "global_embedding_similarity"
                    if used_native_scores
                    else "global_embedding_similarity_with_rank_fallback"
                ),
                "requested_top_k": request.top_k,
                "returned_memories": len(items),
            },
        )

    def _retrieve_with_native_chat(self, request: RetrievalRequest) -> RetrievalResult:
        if request.query_id in self._qa_budgets:
            raise ValueError(f"duplicate MIRIX retrieval query_id: {request.query_id}")
        budget = _MirixRetrievalBudget(self, request.query_id, request.top_k)
        # MIRIX has no independent retrieval endpoint: automatic prefetch and
        # explicit search tools are part of the native Chat Agent lifecycle.
        # Defer that lifecycle to answer_with_memory so the benchmark question
        # is executed exactly once and its selected evidence is returned with
        # the native answer.
        self._qa_budgets[request.query_id] = budget
        return budget.result(stage="deferred_to_native_answer")

    def _retrieved_memory_from_native(
        self, row: Any, raw_id: str
    ) -> RetrievedMemory:
        data = row if isinstance(row, dict) else _model_dump(row) or {}
        memory_type = str(data.get("memory_type") or "unknown")
        memory_id = f"{memory_type}_memory_manager:{raw_id}"
        source = self.provenance.get(memory_id)
        return RetrievedMemory(
            memory_id=memory_id,
            # Native ORM rows also contain one or more 2048-dimensional
            # embedding columns. Serializing the whole row makes the final QA
            # request many times larger than the actual memory and can produce
            # an HTTP-200 error envelope with no ChatCompletion choices. Keep
            # only the human-readable fields MIRIX exposes as memory content.
            text=_render_memory(f"{memory_type}_memory_manager", data),
            score=None,
            session_id=str(source.get("session_id") or ""),
            source_dialogue_ids=list(source.get("source_dialogue_ids") or []),
            image_ids=list(source.get("image_ids") or []),
            image_paths=list(source.get("image_paths") or []),
            metadata={
                "partition": f"{memory_type}_memory",
                "via": "native_chat_agent_tool",
            },
        )

    def _query_embedding(self, request: RetrievalRequest) -> list[float] | None:
        if request.query_vector:
            return [float(value) for value in request.query_vector]
        try:
            module = importlib.import_module(f"{self.package}.embeddings")
            state = self.backend.agent_states.agent_state
            values = module.embedding_model(state.embedding_config).get_text_embedding(
                request.text
            )
            return [float(value) for value in values]
        except Exception:
            if self.config.get("baseline_strict_config", True):
                raise
            return None

    def answer_with_memory(self, request: NativeAnswerRequest) -> NativeAnswerResult:
        if self.baseline != "MIRIX":
            return super().answer_with_memory(request)
        base_budget = self._qa_budgets.pop(request.query_id, None)
        if base_budget is None:
            raise KeyError(
                f"MIRIX answer has no preceding native retrieval: {request.query_id}"
            )
        if request.top_k != base_budget.top_k:
            raise ValueError(
                "MIRIX answer global top_k does not match its retrieval budget: "
                f"answer={request.top_k}, retrieval={base_budget.top_k}"
            )
        client = self.backend.client
        chat_state = self.backend.agent_states.agent_state
        saved = _capture_chat_state(client, chat_state.id)
        budget = base_budget
        token = _ACTIVE_MIRIX_RETRIEVAL.set(budget)
        truncation_state = {
            "accepted": False,
            "finish_reason": "",
            "native_finish_reason": "",
        }
        truncation_token = _ACTIVE_MIRIX_QA_TRUNCATION.set(truncation_state)
        embedding_scope = self._query_multimodal_embedding_scope(
            request.query_image
        )
        try:
            with embedding_scope:
                response = _send_native_benchmark_messages(
                    client,
                    chat_state.id,
                    request.messages,
                    request.query_image,
                )
            text = _ensure_answer_block(_extract_chat_answer(response))
            if not text or text == "ERROR":
                raise RuntimeError("MIRIX Chat Agent did not return a final answer")
            usage = _model_dump(getattr(response, "usage", None))
            retrieval = budget.result(stage="answer_complete")
            selected_ids = [item.memory_id for item in retrieval.items]
            return NativeAnswerResult(
                text=text,
                attempts=1,
                failed_attempts=0,
                image_count=1 if request.query_image else 0,
                usage=usage,
                trace={
                    "via": "mirix_native_chat_agent",
                    "core_memory": "resident",
                    "global_top_k": request.top_k,
                    "retrieved_count": len(selected_ids),
                    "retrieved_memory_ids": selected_ids,
                    "remaining_retrieval_budget": request.top_k - len(selected_ids),
                    "autonomous_retrieval": "enabled_until_final_send_message",
                    "candidate_freezing": False,
                    "single_agent_lifecycle": True,
                    "native_agent_lifecycle_count": 1,
                    "retrieval_deferred_to_answer": True,
                    "provisional_native_answer_ignored": False,
                    "chat_tool_calls": _response_tool_names(response),
                    "qa_answer_attempts": 1,
                    "qa_answer_failed_attempts": 0,
                    "accepted_truncated_qa": bool(truncation_state["accepted"]),
                    "accepted_truncated_qa_finish_reason": str(
                        truncation_state["finish_reason"]
                    ),
                    "accepted_truncated_qa_native_finish_reason": str(
                        truncation_state["native_finish_reason"]
                    ),
                },
                retrieval=retrieval,
            )
        finally:
            _ACTIVE_MIRIX_QA_TRUNCATION.reset(truncation_token)
            _ACTIVE_MIRIX_RETRIEVAL.reset(token)
            _restore_chat_state(client, chat_state.id, saved)

    @contextmanager
    def _query_multimodal_embedding_scope(self, query_image: str | None):
        """Attach the benchmark image to native Chat-Agent query embeddings."""
        global _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING
        if (
            not bool(self.config.get("mirix_multimodal_embedding", False))
            or not query_image
            or not Path(query_image).is_file()
        ):
            yield
            return
        previous = _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING
        _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING = {
            "mode": "query",
            "images": [str(Path(query_image).resolve())],
        }
        try:
            yield
        finally:
            _ACTIVE_MIRIX_MULTIMODAL_EMBEDDING = previous

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
        if self.baseline == "MIRIX":
            chat_state = self.backend.agent_states.agent_state
            for block in chat_state.memory.get_blocks():
                block_id = str(getattr(block, "id", "") or uuid.uuid4().hex[:12])
                label = str(getattr(block, "label", "") or "core")
                value = str(getattr(block, "value", "") or "")
                records.append(
                    MemoryRecord(
                        memory_id=f"core_memory:{block_id}",
                        text=f"[{label}] {value}",
                        backend_type="mirix_core_memory",
                        metadata={"partition": "core_memory", "resident": True},
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
            "native_chat_answer": self.baseline == "MIRIX",
            "fixed_global_top_k": (
                int(self.config.get("top_k", 7))
                if self.baseline == "MIRIX" else None
            ),
            "global_top_k_scope": (
                "automatic_prefetch_plus_explicit_tools"
                if self.baseline == "MIRIX" else None
            ),
            "native_memory_build": self.baseline == "MIRIX",
            "multimodal_embedding": (
                bool(self.config.get("mirix_multimodal_embedding", False))
                if self.baseline == "MIRIX"
                else False
            ),
            "native_absorption_batch": 20 if self.baseline == "MIRIX" else None,
            "retrieval_candidate_freezing": False if self.baseline == "MIRIX" else None,
            "source_commit": (
                "ac0a1f2890df5e7435c66d6c2827f34c5c4ce32d"
                if self.baseline == "MIRIX"
                else None
            ),
            "semantic_fallback_on_error": (
                bool(self.config.get("mirix_semantic_fallback_on_error", False))
                if self.baseline == "MIRIX"
                else True
            ),
        }

    def close(self) -> None:
        self.backend = None


def _request_mirix_multimodal_embedding(
    *,
    endpoint: str,
    model: str,
    dimensions: int,
    text: str,
    image_paths: list[str],
) -> list[float]:
    """Call the local VL embedder with text and concrete source images."""
    content: list[dict[str, Any]] = [{"type": "text", "text": text or " "}]
    for image_path in image_paths:
        path = Path(image_path).resolve()
        if not path.is_file():
            continue
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": str(path)},
            }
        )
    if len(content) == 1:
        raise ValueError("multimodal embedding requested without a readable image")
    payload = {
        "model": str(model),
        "messages": [{"role": "user", "content": content}],
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{str(endpoint).rstrip('/')}/embeddings"
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"multimodal embedding endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    rows = result.get("data") if isinstance(result, dict) else None
    vector = rows[0].get("embedding") if isinstance(rows, list) and rows else None
    if not isinstance(vector, list):
        raise TypeError("multimodal embedding response has no data[0].embedding")
    values = [float(value) for value in vector]
    if len(values) != int(dimensions):
        raise ValueError(
            "multimodal embedding dimension mismatch: "
            f"expected {dimensions}, received {len(values)}"
        )
    return values


def _dedupe_native_rows(rows: list[Any]) -> list[Any]:
    deduped = []
    seen: set[str] = set()
    for row in rows:
        row_id = _native_result_id(row)
        if row_id in seen:
            continue
        seen.add(row_id)
        deduped.append(row)
    return deduped


def _round_robin_native_rows(
    groups: list[tuple[str, list[Any]]], top_k: int
) -> list[tuple[str, Any]]:
    """Interleave native per-bank rankings under one global item cap."""
    selected: list[tuple[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    max_rows = max((len(rows) for _, rows in groups), default=0)
    for rank in range(max_rows):
        for memory_type, rows in groups:
            if rank >= len(rows):
                continue
            row = rows[rank]
            key = (memory_type, _native_result_id(row))
            if key in seen:
                continue
            seen.add(key)
            selected.append((memory_type, row))
            if len(selected) >= top_k:
                return selected
    return selected


def _tree_suffix(row: Any, separator: str = ";") -> str:
    tree_path = list(getattr(row, "tree_path", None) or [])
    return f" {separator} Path: {' > '.join(tree_path)}" if tree_path else ""


def _format_prefetched_episodic(rows: list[Any]) -> str:
    lines = []
    for index, row in enumerate(rows):
        occurred_at = getattr(row, "occurred_at", "")
        if hasattr(occurred_at, "strftime"):
            occurred_at = occurred_at.strftime("%Y-%m-%d %H:%M:%S")
        details = str(getattr(row, "details", "") or "")
        lines.append(
            f"[{index}] Timestamp: {occurred_at} - {getattr(row, 'summary', '')}"
            f"{_tree_suffix(row, separator='-')} (Details: {len(details)} Characters)"
        )
    return "\n".join(lines)


def _format_prefetched_semantic(rows: list[Any]) -> str:
    return "\n".join(
        f"[{index}] Name: {getattr(row, 'name', '')}; Summary: "
        f"{getattr(row, 'summary', '')}{_tree_suffix(row)}"
        for index, row in enumerate(rows)
    )


def _format_prefetched_procedural(rows: list[Any]) -> str:
    return "\n".join(
        f"[{index}] Entry Type: {getattr(row, 'entry_type', '')}; Summary: "
        f"{getattr(row, 'summary', '')}{_tree_suffix(row)}"
        for index, row in enumerate(rows)
    )


def _format_prefetched_resource(rows: list[Any]) -> str:
    return "\n".join(
        f"[{index}] Resource Title: {getattr(row, 'title', '')}; Resource Summary: "
        f"{getattr(row, 'summary', '')} Resource Type: "
        f"{getattr(row, 'resource_type', '')}{_tree_suffix(row)}"
        for index, row in enumerate(rows)
    )


def _format_prefetched_vault(rows: list[Any]) -> str:
    return "\n".join(
        f"[{index}] Knowledge Vault Item ID: {getattr(row, 'id', '')}; Caption: "
        f"{getattr(row, 'caption', '')}"
        for index, row in enumerate(rows)
    )


def _render_memory(manager: str, value: Any) -> str:
    fields = []
    for key in (
        "summary",
        "details",
        "description",
        "steps",
        "content",
        "name",
        "title",
        "caption",
        "secret_value",
        "source",
    ):
        item = value.get(key) if isinstance(value, dict) else getattr(value, key, None)
        if item:
            fields.append(f"{key}: {item}")
    return f"[{manager.removesuffix('_manager')}] " + " | ".join(fields)


def _stage_native_transport_image(
    image_path: str | Path,
    *,
    cache_dir: Path,
    compressor: Any,
) -> Path:
    """Run MIRIX's native upload compression without modifying source images."""
    source = Path(image_path).resolve()
    if source.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return source
    stat = source.stat()
    cache_key = hashlib.sha256(
        f"{source}:{stat.st_size}:{stat.st_mtime_ns}:1920x1080:q85".encode("utf-8")
    ).hexdigest()[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    staged_source = cache_dir / f"{cache_key}{source.suffix.lower()}"
    compressed_path = cache_dir / f"{cache_key}_compressed.jpg"
    if compressed_path.is_file():
        return compressed_path

    shutil.copy2(source, staged_source)
    try:
        result = compressor._compress_image(
            str(staged_source), quality=85, max_size=(1920, 1080)
        )
        candidate = Path(result) if result else None
        if candidate is None or not candidate.is_file():
            return source
        if candidate.stat().st_size >= source.stat().st_size:
            candidate.unlink()
            return source
        return candidate
    finally:
        if staged_source.is_file():
            staged_source.unlink()


def _embedding_similarity(left: Any, right: Any) -> float | None:
    """Return cosine similarity for native MIRIX vectors from any supported DB."""
    left_values = _vector_values(left)
    right_values = _vector_values(right)
    if not left_values or not right_values:
        return None
    size = min(len(left_values), len(right_values))
    left_values = left_values[:size]
    right_values = right_values[:size]
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(a * b for a, b in zip(left_values, right_values)) / (
        left_norm * right_norm
    )


def _vector_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _native_result_id(row: Any) -> str:
    data = row if isinstance(row, dict) else _model_dump(row) or {}
    raw_id = str(data.get("id") or "").strip()
    if raw_id:
        return raw_id
    payload = json.dumps(data, ensure_ascii=False, default=str, sort_keys=True)
    return "anonymous_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _raw_chat_query(text: str, query_image: str | None) -> str | list[dict[str, Any]]:
    if not query_image:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "file_uri", "file_uri": str(Path(query_image).resolve())},
    ]


def _capture_chat_state(client: Any, agent_id: str) -> dict[str, Any]:
    messages = client.get_in_context_messages(agent_id)
    actor = client.server.user_manager.get_user_by_id(client.user.id)
    state = client.server.agent_manager.get_agent_by_id(agent_id=agent_id, actor=actor)
    return {
        "message_ids": [str(row.id) for row in messages],
        "topic": getattr(state, "topic", None),
    }


def _restore_chat_state(client: Any, agent_id: str, saved: dict[str, Any]) -> None:
    """Restore both history and topic so benchmark questions cannot leak."""
    try:
        server = client.server
        actor = server.user_manager.get_user_by_id(client.user.id)
        server.agent_manager.set_in_context_messages(
            agent_id=agent_id,
            message_ids=list(saved["message_ids"]),
            actor=actor,
        )
        server.message_manager.delete_detached_messages_for_agent(
            agent_id=agent_id,
            actor=actor,
        )
        update_module = importlib.import_module("mirix.schemas.agent")
        updated = server.agent_manager.update_agent(
            agent_id=agent_id,
            agent_update=update_module.UpdateAgent(topic=saved.get("topic")),
            actor=actor,
        )
        # AgentWrapper caches the chat state separately from the DB manager.
        # The following native step will otherwise reuse the provisional topic.
        return updated
    except Exception as exc:
        raise RuntimeError("Failed to restore MIRIX Chat Agent history/topic after QA") from exc


def _send_native_benchmark_messages(
    client: Any,
    agent_id: str,
    messages: list[dict[str, Any]],
    query_image: str | None,
) -> Any:
    """Run the benchmark's exact messages through the original Chat Agent."""
    message_module = importlib.import_module("mirix.schemas.message")
    enum_module = importlib.import_module("mirix.schemas.enums")
    content_module = importlib.import_module("mirix.schemas.mirix_message_content")
    response_module = importlib.import_module("mirix.schemas.mirix_response")
    query_image_path: Path | None = None
    if query_image:
        # The native local-file helper copies images verbatim.  Route question
        # images through the same bounded transport copy used during memory
        # ingestion so a high-resolution QA image cannot consume the entire
        # 32k multimodal context before MIRIX has any history to summarize.
        upload_module = importlib.import_module("mirix.agent.upload_manager")
        compressor = object.__new__(upload_module.UploadManager)
        compressor.logger = importlib.import_module("logging").getLogger(
            "Mirix.OfflineImageTransport"
        )
        query_image_path = _stage_native_transport_image(
            query_image,
            cache_dir=(
                Path(client.images_dir).parent / "tmp" / "image_transport"
            ),
            compressor=compressor,
        )
    packed = []
    for index, row in enumerate(messages):
        role = str(row.get("role") or "user")
        content: list[Any] = [
            content_module.TextContent(text=str(row.get("content") or ""))
        ]
        if query_image_path and role == "user" and index == len(messages) - 1:
            metadata = client._save_image_from_file_uri(str(query_image_path))
            content.append(content_module.ImageContent(image_id=metadata.id, detail="auto"))
        packed.append(
            message_module.MessageCreate(
                role=enum_module.MessageRole(role),
                content=content,
            )
        )
    client.interface.clear()
    server = client.server
    usage = server.send_messages(
        actor=server.user_manager.get_user_by_id(client.user.id),
        agent_id=agent_id,
        input_messages=packed,
        interface=client.interface,
        force_response=True,
        # Qwen-family agents commonly search first and call ``send_message``
        # on the following step.  The installed native hook makes the first
        # validated ``send_message`` terminal, so chaining is required here
        # without reintroducing the historical repeated-answer loop.
        chaining=True,
    )
    mirix_messages = []
    for event in client.interface.to_list():
        mirix_messages.extend(event.to_mirix_message())
    return response_module.MirixResponse(messages=mirix_messages, usage=usage)


def _chat_retrieved_memories(
    items: list[RetrievedMemory], *, query: str
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = {
        "episodic": [],
        "semantic": [],
        "procedural": [],
        "resource": [],
        "knowledge_vault": [],
    }
    for item in items:
        partition = str(item.metadata.get("partition") or "")
        key = partition.removesuffix("_memory")
        if key in grouped:
            grouped[key].append(item.text)
        else:
            grouped["semantic"].append(item.text)

    # MIRIX's native prompt builder currently retrieves Knowledge Vault but
    # omits its dedicated section. Preserve those selected rows inside the
    # visible semantic section so all globally selected memories reach Chat.
    semantic_rows = grouped["semantic"] + grouped["knowledge_vault"]
    return {
        "key_words": query,
        # Core is intentionally absent: MIRIX loads it as resident context.
        "episodic": ["", "\n".join(grouped["episodic"])],
        "semantic": "\n".join(semantic_rows),
        "procedural": "\n".join(grouped["procedural"]),
        "resource": "\n".join(grouped["resource"]),
        "knowledge_vault": "\n".join(grouped["knowledge_vault"]),
    }


def _messages_question(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(row.get("content") or "")
        for row in messages
        if str(row.get("role") or "") == "user"
    ).strip()


def _chat_message_content(
    messages: list[dict[str, Any]], query_image: str | None
) -> str | list[dict[str, Any]]:
    sections = []
    for row in messages:
        role = str(row.get("role") or "user").upper()
        content = str(row.get("content") or "").strip()
        if content:
            sections.append(f"[{role} INSTRUCTION]\n{content}")
    text = (
        "Use the resident Core Memory and the globally selected Top-7 memory "
        "items supplied in your system context. Native memory-search tools may "
        "be used when helpful; their searchable scope is frozen to those same "
        "Top-7 items.\n\n" + "\n\n".join(sections)
    )
    if not query_image:
        return text
    path = Path(query_image)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "auto"},
        },
    ]


def _extract_chat_answer(response: Any) -> str:
    messages = list(getattr(response, "messages", None) or [])
    for message in reversed(messages):
        tool_call = getattr(message, "tool_call", None)
        if tool_call is None and isinstance(message, dict):
            tool_call = message.get("tool_call")
        if tool_call is None:
            continue
        name = getattr(tool_call, "name", None)
        arguments = getattr(tool_call, "arguments", None)
        if isinstance(tool_call, dict):
            name = tool_call.get("name", name)
            arguments = tool_call.get("arguments", arguments)
        if name and str(name) != "send_message":
            continue
        payload = _tool_payload(str(arguments or ""))
        if payload and payload.get("message"):
            return str(payload["message"])
    return ""


def _ensure_answer_block(text: str) -> str:
    """Normalize MIRIX's visible Chat response to the benchmark contract."""
    text = str(text or "").strip()
    if not text:
        return ""
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if len(matches) == 1:
        return f"<answer>{matches[0].strip()}</answer>"
    if text.casefold().startswith("<answer>"):
        text = text[len("<answer>") :].strip()
    if text.casefold().endswith("</answer>"):
        text = text[: -len("</answer>")].strip()
    return f"<answer>{text}</answer>"


def _model_dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    return dump() if callable(dump) else None


def _restore_chat_messages(client: Any, agent_id: str, message_ids: list[str]) -> None:
    """Prevent one benchmark question/answer from leaking into the next one."""
    try:
        server = client.server
        actor = server.user_manager.get_user_by_id(client.user.id)
        server.agent_manager.set_in_context_messages(
            agent_id=agent_id,
            message_ids=message_ids,
            actor=actor,
        )
        server.message_manager.delete_detached_messages_for_agent(
            agent_id=agent_id,
            actor=actor,
        )
    except Exception:
        # Restoration is part of correctness. Do not silently continue with
        # leaked QA history when the native backend cannot roll it back.
        raise RuntimeError("Failed to restore MIRIX Chat Agent history after QA")


@contextmanager
def _frozen_native_search_scope(backend: Any, items: list[RetrievedMemory]):
    """Keep MIRIX's autonomous tools, but restrict them to the frozen Top-K set."""
    allowed: dict[str, set[str]] = {}
    for item in items:
        manager_name, _, raw_id = item.memory_id.partition(":")
        allowed.setdefault(manager_name, set()).add(raw_id)

    patched = []
    server = backend.client.server
    try:
        for manager_name, _, method_name in _PARTITIONS:
            manager = getattr(server, manager_name, None)
            if manager is None:
                continue
            manager_class = type(manager)
            original = getattr(manager_class, method_name)
            allowed_ids = allowed.get(manager_name, set())

            def restricted(
                instance: Any,
                *args: Any,
                _original: Any = original,
                _allowed_ids: set[str] = allowed_ids,
                **kwargs: Any,
            ) -> list[Any]:
                rows = _original(instance, *args, **kwargs) or []
                return [row for row in rows if str(getattr(row, "id", "")) in _allowed_ids]

            setattr(manager_class, method_name, restricted)
            patched.append((manager_class, method_name, original))
        yield
    finally:
        for manager_class, method_name, original in reversed(patched):
            setattr(manager_class, method_name, original)


def _response_tool_names(response: Any) -> list[str]:
    names = []
    for message in list(getattr(response, "messages", None) or []):
        tool_call = getattr(message, "tool_call", None)
        if tool_call is None and isinstance(message, dict):
            tool_call = message.get("tool_call")
        name = getattr(tool_call, "name", None)
        if isinstance(tool_call, dict):
            name = tool_call.get("name", name)
        if name:
            names.append(str(name))
    return names


def _cap_legacy_request_tokens(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    max_tokens: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Apply the configured output cap to MIRIX/MMA's legacy request path.

    The legacy streaming fallback passes ``max_tokens=None`` to its request
    model even when ``LLMConfig.max_tokens`` is set.  Without a cap, a local
    vLLM request can generate until the context window and repeatedly outlive
    the benchmark proxy timeout.
    """
    request = kwargs.get("chat_completion_request")
    positional_index = 2
    if request is None and len(args) > positional_index:
        request = args[positional_index]
    if request is None:
        return args, kwargs

    if isinstance(request, dict):
        if (
            request.get("max_tokens") is None
            and request.get("max_completion_tokens") is None
        ):
            request["max_tokens"] = max_tokens
        return args, kwargs

    if (
        getattr(request, "max_tokens", None) is not None
        or getattr(request, "max_completion_tokens", None) is not None
    ):
        return args, kwargs
    try:
        request.max_tokens = max_tokens
        return args, kwargs
    except (AttributeError, TypeError, ValueError):
        model_copy = getattr(request, "model_copy", None)
        if not callable(model_copy):
            return args, kwargs
        request = model_copy(update={"max_tokens": max_tokens})
        if "chat_completion_request" in kwargs:
            kwargs = dict(kwargs)
            kwargs["chat_completion_request"] = request
        else:
            mutable_args = list(args)
            mutable_args[positional_index] = request
            args = tuple(mutable_args)
        return args, kwargs


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


def _mirix_model_handle(*, baseline: str, model: str, endpoint: str) -> str | None:
    """Use MIRIX's vLLM compatibility handle only for a local executor."""
    if baseline != "MIRIX":
        return None
    try:
        hostname = (urlsplit(endpoint).hostname or "").casefold()
    except ValueError:
        hostname = ""
    if hostname in {"127.0.0.1", "localhost", "0.0.0.0", "::1"}:
        return f"vllm/{model}"
    # Hosted OpenAI-compatible providers support required native tool calls;
    # labeling them as vLLM incorrectly downgrades tool_choice to auto.
    return None


def _bound_native_vllm_tool_request(
    data: dict[str, Any], *, repetition_penalty: float = 1.10
) -> dict[str, Any]:
    """Apply the native Qwen/MIRIX one-tool response boundary.

    MIRIX already injects every memory bank's current summary into memory-
    update agents via ``build_system_prompt_with_memories``.  Its v0.1.1
    agents nevertheless expose the universal search/list tools as well as
    their write tools.  Qwen3-VL-4B strongly prefers the redundant read tool,
    and v0.1.1 then ends the update turn because memory-update chaining is
    disabled.  Hide only those redundant read tools when a native write tool
    is present.  The Chat Agent has no write tool and therefore keeps its
    original autonomous search tools and the shared Top-7 budget.
    """
    if data.get("tool_choice") != "auto" or not data.get("tools"):
        return data
    data = _restrict_native_memory_update_tools(data)
    stop = data.get("stop")
    if isinstance(stop, str):
        stops = [stop]
    elif isinstance(stop, list):
        stops = list(stop)
    else:
        stops = []
    if "</tool_call>" not in stops:
        stops.append("</tool_call>")
    if "<|im_end|>" not in stops:
        stops.append("<|im_end|>")
    data["stop"] = stops
    extra_body = data.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    else:
        extra_body = dict(extra_body)
    # This is a vLLM extension, so the OpenAI SDK requires it under
    # ``extra_body`` rather than as a top-level create() argument. Excluding
    # the boundary is intentional: otherwise a structured JSON response that
    # stops on ``<|im_end|>`` is rejected by vLLM as trailing characters. The
    # compatibility parser accepts a valid tool object without its closing XML
    # tag.
    extra_body["include_stop_str_in_output"] = False
    # Qwen3-VL-4B can otherwise loop inside a long JSON string (most often a
    # Resource Memory image description) without ever closing the tool-call
    # envelope.  A small penalty prevents that decoding failure while leaving
    # temperature, prompts, tool choice, and MIRIX execution unchanged.
    extra_body["repetition_penalty"] = repetition_penalty
    data["extra_body"] = extra_body
    return data


_MIRIX_NATIVE_MEMORY_WRITE_TOOLS = {
    "trigger_memory_update",
    "core_memory_append",
    "core_memory_rewrite",
    "episodic_memory_insert",
    "episodic_memory_merge",
    "episodic_memory_replace",
    "check_episodic_memory",
    "procedural_memory_insert",
    "procedural_memory_update",
    "resource_memory_insert",
    "resource_memory_update",
    "knowledge_vault_insert",
    "knowledge_vault_update",
    "semantic_memory_insert",
    "semantic_memory_update",
    "check_semantic_memory",
}
_MIRIX_REDUNDANT_MEMORY_UPDATE_READ_TOOLS = {
    "search_in_memory",
    "list_memory_within_timerange",
}
_MIRIX_DELTA_UPDATE_TOOLS = {
    "resource_memory_update": ("content", "content_delta"),
    "procedural_memory_update": ("steps", "steps_delta"),
}
_MIRIX_BOUNDED_EXECUTION_TOOLS = set(_MIRIX_DELTA_UPDATE_TOOLS) | {
    "semantic_memory_insert",
    "episodic_memory_insert",
    "episodic_memory_merge",
}
_MIRIX_TRUNCATED_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
}
_MIRIX_MAX_TRAILING_TOOL_WHITESPACE = 128
_MIRIX_QA_REPAIRABLE_TOOLS = {
    "search_in_memory",
    "list_memory_within_timerange",
    "send_message",
}


def _native_tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "")


def _response_message_tool_names(response_message: Any) -> list[str]:
    tool_calls = getattr(response_message, "tool_calls", None)
    if isinstance(response_message, dict):
        tool_calls = response_message.get("tool_calls", tool_calls)
    names: list[str] = []
    for tool_call in tool_calls or []:
        function = getattr(tool_call, "function", None)
        if isinstance(tool_call, dict):
            function = tool_call.get("function", function)
        name = getattr(function, "name", None)
        if isinstance(function, dict):
            name = function.get("name", name)
        if name:
            names.append(str(name))
    return names


def _prepare_mirix_delta_tool_request(data: dict[str, Any]) -> dict[str, Any]:
    """Expose bounded delta fields without changing MIRIX's stored tool definitions."""
    tools = data.get("tools")
    if not isinstance(tools, list):
        return data
    changed = False
    rewritten_tools: list[Any] = []
    for raw_tool in tools:
        tool = copy.deepcopy(raw_tool)
        function = tool.get("function") if isinstance(tool, dict) else None
        name = str(function.get("name") or "") if isinstance(function, dict) else ""
        parameters = function.get("parameters") if isinstance(function, dict) else None
        properties = parameters.get("properties") if isinstance(parameters, dict) else None
        if not isinstance(properties, dict):
            rewritten_tools.append(tool)
            continue

        if name in _MIRIX_DELTA_UPDATE_TOOLS:
            old_field, delta_field = _MIRIX_DELTA_UPDATE_TOOLS[name]
            new_items = properties.get("new_items")
            item_schema = (
                new_items.get("items") if isinstance(new_items, dict) else None
            )
            item_properties = (
                item_schema.get("properties")
                if isinstance(item_schema, dict)
                else None
            )
            if isinstance(item_properties, dict) and old_field in item_properties:
                rewritten: dict[str, Any] = {}
                for field, schema in item_properties.items():
                    if name == "resource_memory_update" and field == "summary":
                        delta_schema = dict(schema) if isinstance(schema, dict) else {}
                        delta_schema["maxLength"] = 300
                        delta_schema["description"] = (
                            "Only new summary information absent from the stored resource; "
                            "do not repeat its existing summary or content; maximum 300 "
                            "characters."
                        )
                        rewritten["summary_delta"] = delta_schema
                        continue
                    if name == "resource_memory_update" and field == old_field:
                        # Qwen3-VL copies the complete stored resource whenever
                        # an update content field is exposed, even when that
                        # field is described as a short delta.  The compact
                        # summary_delta already captures the new information;
                        # the runtime appends it to both summary and content.
                        continue
                    if field != old_field:
                        rewritten[field] = schema
                        continue
                    delta_schema = dict(schema) if isinstance(schema, dict) else {}
                    if delta_field == "content_delta":
                        delta_schema["maxLength"] = 900
                        delta_schema["description"] = (
                            "Only new information absent from the old resource; do not "
                            "repeat old content or transcribe sessions; maximum 900 characters."
                        )
                    else:
                        delta_schema["description"] = (
                            "Only new or changed steps absent from the old procedure; "
                            "do not repeat old steps; at most six concise strings."
                        )
                    rewritten[delta_field] = delta_schema
                item_schema["properties"] = rewritten
                required = item_schema.get("required")
                if isinstance(required, list):
                    rewritten_required = []
                    for field in required:
                        if name == "resource_memory_update" and field == old_field:
                            continue
                        rewritten_required.append(
                            "summary_delta"
                            if name == "resource_memory_update" and field == "summary"
                            else delta_field if field == old_field else field
                        )
                    if (
                        name == "resource_memory_update"
                        and "summary_delta" not in rewritten_required
                    ):
                        rewritten_required.append("summary_delta")
                    item_schema["required"] = rewritten_required
                if name == "resource_memory_update":
                    delta_description = (
                        "For updates, provide only summary_delta with the new facts. "
                        "The runtime merges it into both the referenced resource's "
                        "summary and content; never repeat stored resource text."
                    )
                else:
                    delta_description = (
                        f"For updates, {delta_field} is a bounded delta that the runtime "
                        "merges with the referenced old item."
                    )
                function["description"] = (
                    f"{str(function.get('description') or '').rstrip()} "
                    f"{delta_description}"
                ).strip()
                changed = True
        elif name == "episodic_memory_merge":
            details_schema = properties.pop("combined_details", None)
            if details_schema is not None:
                delta_schema = (
                    dict(details_schema) if isinstance(details_schema, dict) else {}
                )
                delta_schema["description"] = (
                    "Only new event details absent from the stored event; do not "
                    "repeat old details; maximum 900 characters."
                )
                properties["details_delta"] = delta_schema
                required = parameters.get("required")
                if isinstance(required, list):
                    parameters["required"] = [
                        "details_delta" if field == "combined_details" else field
                        for field in required
                    ]
                summary_schema = properties.get("combined_summary")
                if isinstance(summary_schema, dict):
                    summary_schema["description"] = (
                        "Complete standalone event summary, maximum 300 characters."
                    )
                function["description"] = (
                    f"{str(function.get('description') or '').rstrip()} "
                    "details_delta is a bounded delta that the runtime appends "
                    "to the referenced event."
                ).strip()
                changed = True
        elif name == "semantic_memory_insert":
            items_schema = properties.get("items")
            item_schema = (
                items_schema.get("items")
                if isinstance(items_schema, dict)
                else None
            )
            item_properties = (
                item_schema.get("properties")
                if isinstance(item_schema, dict)
                else None
            )
            if isinstance(items_schema, dict) and isinstance(item_properties, dict):
                items_schema["maxItems"] = 3
                limits = {
                    "name": 120,
                    "summary": 240,
                    "details": 500,
                    "source": 160,
                }
                for field, max_length in limits.items():
                    schema = item_properties.get(field)
                    if isinstance(schema, dict):
                        schema["maxLength"] = max_length
                tree_path = item_properties.get("tree_path")
                if isinstance(tree_path, dict):
                    tree_path["maxItems"] = 4
                    tree_item = tree_path.get("items")
                    if isinstance(tree_item, dict):
                        tree_item["maxLength"] = 64
                inner_schema = properties.get("inner_thoughts")
                if isinstance(inner_schema, dict):
                    inner_schema["maxLength"] = 200
                function["description"] = (
                    f"{str(function.get('description') or '').rstrip()} "
                    "Consolidate related facts and insert at most three of the "
                    "most important bounded concepts per call."
                ).strip()
                changed = True
        elif name == "episodic_memory_insert":
            items_schema = properties.get("items")
            item_schema = (
                items_schema.get("items")
                if isinstance(items_schema, dict)
                else None
            )
            item_properties = (
                item_schema.get("properties")
                if isinstance(item_schema, dict)
                else None
            )
            if isinstance(items_schema, dict) and isinstance(item_properties, dict):
                items_schema["maxItems"] = 1
                details_schema = item_properties.get("details")
                if isinstance(details_schema, dict):
                    details_schema["maxLength"] = 900
                    details_schema["description"] = (
                        "Compact details for the single most-significant event; "
                        "maximum 900 characters; never transcribe all messages."
                    )
                summary_schema = item_properties.get("summary")
                if isinstance(summary_schema, dict):
                    summary_schema["maxLength"] = 300
                    summary_schema["description"] = (
                        "Concise standalone event summary, maximum 300 characters."
                    )
                inner_schema = properties.get("inner_thoughts")
                if isinstance(inner_schema, dict):
                    inner_schema["maxLength"] = 300
                function["description"] = (
                    f"{str(function.get('description') or '').rstrip()} Insert "
                    "exactly one most-significant bounded event per call."
                ).strip()
                changed = True
        elif name == "resource_memory_insert":
            items = properties.get("items")
            item_schema = items.get("items") if isinstance(items, dict) else None
            item_properties = (
                item_schema.get("properties")
                if isinstance(item_schema, dict)
                else None
            )
            content_schema = (
                item_properties.get("content")
                if isinstance(item_properties, dict)
                else None
            )
            if isinstance(content_schema, dict):
                content_schema["description"] = (
                    "Compact summarized resource content; never transcribe whole sessions; "
                    "maximum 900 characters."
                )
                changed = True
        elif name == "procedural_memory_insert":
            items = properties.get("items")
            item_schema = items.get("items") if isinstance(items, dict) else None
            item_properties = (
                item_schema.get("properties")
                if isinstance(item_schema, dict)
                else None
            )
            steps_schema = (
                item_properties.get("steps")
                if isinstance(item_properties, dict)
                else None
            )
            if isinstance(steps_schema, dict):
                steps_schema["description"] = (
                    "At most eight concise step strings containing the essential procedure."
                )
                changed = True
        rewritten_tools.append(tool)
    if not changed:
        return data
    bounded = dict(data)
    bounded["tools"] = rewritten_tools
    return bounded


def _has_native_tool_calls(response_data: dict[str, Any]) -> bool:
    return any(
        (choice.get("message") or {}).get("tool_calls")
        for choice in response_data.get("choices") or []
        if isinstance(choice, dict)
    )


def _has_unparsed_native_tool_envelope(response_data: dict[str, Any]) -> bool:
    return any(
        isinstance((choice.get("message") or {}).get("content"), str)
        and "<tool_call>" in (choice.get("message") or {}).get("content", "")
        and not (choice.get("message") or {}).get("tool_calls")
        for choice in response_data.get("choices") or []
        if isinstance(choice, dict)
    )


def _mirix_native_memory_truncation_retry_required(
    request_data: dict[str, Any], response_data: dict[str, Any]
) -> bool:
    """Retry one memory-tool turn only when the provider reports truncation."""
    return _mirix_request_requires_native_memory_tool(request_data) and any(
        {
            str(choice.get("finish_reason") or "").casefold(),
            str(choice.get("native_finish_reason") or "").casefold(),
        }.intersection(_MIRIX_TRUNCATED_FINISH_REASONS)
        for choice in response_data.get("choices") or []
        if isinstance(choice, dict)
    )


def _request_output_token_cap(request_data: dict[str, Any]) -> int:
    """Return the configured output cap for one provider request."""
    return max(
        1,
        int(
            request_data.get("max_completion_tokens")
            or request_data.get("max_tokens")
            or 2048
        ),
    )


def _mirix_native_memory_corrective_retry_reason(
    request_data: dict[str, Any], response_data: dict[str, Any]
) -> str | None:
    """Select one bounded retry for a malformed native memory-agent turn.

    The retry happens inside the failing memory agent request, before MIRIX can
    execute a tool. Other memory agents in the same update are therefore not
    replayed and their successful writes cannot be duplicated.
    """
    if not _mirix_request_requires_native_memory_tool(request_data):
        return None
    if _mirix_native_memory_truncation_retry_required(
        request_data, response_data
    ):
        return "truncation"
    try:
        _reject_unparsed_native_tool_response(response_data)
    except ValueError:
        return "malformed_tool_call"
    if not _has_native_tool_calls(response_data):
        return "missing_tool_call"
    return None


def _mirix_native_memory_bad_request_retry_required(
    request_data: dict[str, Any], exc: Exception
) -> bool:
    """Retry only vLLM's generated invalid-JSON tool-call 400 response."""
    if not _mirix_request_requires_native_memory_tool(request_data):
        return False
    status_code = getattr(exc, "status_code", None)
    if status_code != 400:
        return False
    body = getattr(exc, "body", None)
    try:
        body_text = json.dumps(body, ensure_ascii=False)
    except (TypeError, ValueError):
        body_text = str(body)
    text = f"{exc}\n{body_text}".casefold()
    return (
        "invalid json" in text
        and ("control character" in text or "json_invalid" in text)
    )


def _prepare_mirix_native_memory_retry(
    request_data: dict[str, Any], *, max_output_tokens: int = 4096
) -> dict[str, Any]:
    """Request one bounded native tool for a single corrective retry.

    vLLM's ``tool_choice=required`` enables grammar-constrained decoding over
    MIRIX's large union of tool schemas.  On Qwen3-VL-4B that path can take
    longer than the request timeout for a few hundred tokens.  Keep native
    auto-tool parsing and change only the format instruction; the response is
    still rejected before execution unless it contains a valid native tool
    call with schema-valid arguments.
    """
    retry = copy.deepcopy(request_data)
    retry["tool_choice"] = "auto"
    retry["parallel_tool_calls"] = False
    retry["max_tokens"] = int(max_output_tokens)
    retry.pop("max_completion_tokens", None)
    retry["stop"] = ["</tool_call>", "<|im_end|>"]
    extra_body = dict(retry.get("extra_body") or {})
    extra_body["include_stop_str_in_output"] = False
    retry["extra_body"] = extra_body

    correction = (
        "The previous response was rejected only because its tool-call JSON "
        "was missing or malformed. Return exactly one of the provided tools "
        "now, with complete valid JSON arguments. Keep string values concise "
        "and do not answer in plain text."
    )
    messages = retry.get("messages")
    if isinstance(messages, list):
        messages = copy.deepcopy(messages)
        for message in messages:
            if (
                isinstance(message, dict)
                and message.get("role") == "system"
                and isinstance(message.get("content"), str)
            ):
                message["content"] = f"{message['content']}\n\n{correction}"
                break
        else:
            messages.insert(0, {"role": "system", "content": correction})
        retry["messages"] = messages

    def visit(value: Any, *, field_name: str = "") -> None:
        if isinstance(value, dict):
            if value.get("type") == "string":
                limit = 300 if field_name == "inner_thoughts" else 1024
                configured = value.get("maxLength")
                value["maxLength"] = min(
                    int(configured) if configured is not None else limit,
                    limit,
                )
            elif value.get("type") == "array":
                configured = value.get("maxItems")
                value["maxItems"] = min(
                    int(configured) if configured is not None else 8,
                    8,
                )
            for key, child in value.items():
                visit(child, field_name=key)
        elif isinstance(value, list):
            for child in value:
                visit(child, field_name=field_name)

    visit(retry.get("tools") or [])
    return retry


def _mirix_request_requires_native_memory_tool(data: dict[str, Any]) -> bool:
    """Return whether this native MIRIX turn is a memory-update agent turn."""
    return any(
        _native_tool_name(tool) in _MIRIX_NATIVE_MEMORY_WRITE_TOOLS
        for tool in data.get("tools") or []
    )


def _reject_missing_native_memory_tool_response(
    response_data: dict[str, Any],
    *,
    required: bool,
    max_output_tokens: int = 0,
    fail_fast: bool = False,
) -> None:
    """Reject capped or plain-text memory-agent turns before they can write."""
    if not required:
        return
    usage = response_data.get("usage") or {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if (
        max_output_tokens > 0
        and completion_tokens >= max_output_tokens
        and not response_data.get("_offline_complete_native_tool_prefix")
    ):
        error_type = RuntimeError if fail_fast else ValueError
        raise error_type(
            "MIRIX memory-update response reached the configured token limit "
            f"({completion_tokens}/{max_output_tokens}); refusing to execute it"
        )
    if not _has_native_tool_calls(response_data):
        error_type = RuntimeError if fail_fast else ValueError
        raise error_type(
            "MIRIX memory-update response contained no native tool call; "
            "refusing to count plain text as a successful build point"
        )


def _promote_first_complete_native_memory_tool_call(
    response_data: dict[str, Any],
) -> bool:
    """Accept one complete write call before a truncated extra-call suffix.

    This is deliberately not JSON repair: ``raw_decode`` must consume a
    complete JSON object, its function must be a known memory-write tool, and
    any remaining non-whitespace text must begin at a tool-envelope boundary.
    The incomplete suffix is never parsed or executed.
    """
    recovered = False
    decoder = json.JSONDecoder()
    for choice in response_data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message") or {}
        content = message.get("content")
        if message.get("tool_calls") or not isinstance(content, str):
            continue
        if "<tool_call>" not in content:
            continue
        candidate = content.split("<tool_call>", 1)[1].lstrip()
        try:
            payload, end = decoder.raw_decode(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict) or set(payload) - {
            "name",
            "arguments",
            "args",
        }:
            continue
        function_name = str(payload.get("name") or "")
        arguments = payload.get("arguments", payload.get("args"))
        if (
            function_name not in _MIRIX_NATIVE_MEMORY_WRITE_TOOLS
            or not isinstance(arguments, dict)
        ):
            continue
        remainder = candidate[end:].lstrip()
        if remainder.startswith("</tool_call>"):
            remainder = remainder[len("</tool_call>") :].lstrip()
        if remainder and not remainder.startswith("<tool_call>"):
            continue
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
        choice["finish_reason"] = "tool_calls"
        if "native_finish_reason" in choice:
            choice["native_finish_reason"] = "completed"
        recovered = True
    if recovered:
        response_data["_offline_complete_native_tool_prefix"] = True
    return recovered


def _reject_native_tool_response_integrity(
    response_data: dict[str, Any], *, fail_fast: bool = False
) -> None:
    """Reject provider truncation and whitespace-degenerate native tool output."""
    error_type = RuntimeError if fail_fast else ValueError
    for choice in response_data.get("choices") or []:
        finish_reasons = {
            str(choice.get("finish_reason") or "").lower(),
            str(choice.get("native_finish_reason") or "").lower(),
        }
        capped = finish_reasons.intersection(_MIRIX_TRUNCATED_FINISH_REASONS)
        if capped:
            raise error_type(
                "MIRIX provider response was truncated before tool execution: "
                + ", ".join(sorted(capped))
            )
        message = choice.get("message") or {}
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str) or not arguments.strip():
                raise error_type("MIRIX returned empty native tool-call arguments")
            trailing = len(arguments) - len(arguments.rstrip())
            if trailing >= _MIRIX_MAX_TRAILING_TOOL_WHITESPACE:
                raise error_type(
                    "MIRIX returned whitespace-degenerate native tool-call arguments "
                    f"({trailing} trailing whitespace characters)"
                )
            try:
                payload = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise error_type(
                    "MIRIX returned incomplete native tool-call JSON; refusing repair"
                ) from exc
            if not isinstance(payload, dict):
                raise error_type(
                    "MIRIX returned non-object native tool-call arguments"
                )


def _accept_truncated_native_qa_text(response_data: dict[str, Any]) -> bool:
    """Accept only non-empty plain QA text when the provider reaches 512 tokens."""
    state = _ACTIVE_MIRIX_QA_TRUNCATION.get()
    if state is None:
        return False
    accepted = False
    for choice in response_data.get("choices") or []:
        finish_reason = str(choice.get("finish_reason") or "").lower()
        native_finish_reason = str(
            choice.get("native_finish_reason") or ""
        ).lower()
        if not {
            finish_reason,
            native_finish_reason,
        }.intersection(_MIRIX_TRUNCATED_FINISH_REASONS):
            continue
        message = choice.get("message") or {}
        content = message.get("content")
        # A truncated tool-call JSON is never safe to repair or execute. This
        # exception is only for already-visible final answer text.
        if (
            message.get("tool_calls")
            or not isinstance(content, str)
            or not content.strip()
        ):
            continue
        state["accepted"] = True
        state["finish_reason"] = finish_reason
        state["native_finish_reason"] = native_finish_reason
        choice["finish_reason"] = "stop"
        if "native_finish_reason" in choice:
            choice["native_finish_reason"] = "completed"
        accepted = True
    return accepted


def _unique_nonempty_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = re.sub(r"\s+", " ", text).casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _bounded_memory_text(value: Any, max_chars: int) -> str:
    text = compact_text(value)
    if len(text) <= max_chars:
        return text
    boundary = max(text.rfind(mark, 0, max_chars + 1) for mark in (". ", "; ", ": "))
    if boundary >= max_chars // 2:
        return text[: boundary + 1].rstrip()
    return text[: max_chars - 3].rstrip() + "..."


def _merge_mirix_delta_update(
    function_name: str,
    function_args: dict[str, Any],
    old_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Expand bounded model deltas into complete validated replacement items."""
    if function_name not in _MIRIX_DELTA_UPDATE_TOOLS:
        return function_args
    old_field, delta_field = _MIRIX_DELTA_UPDATE_TOOLS[function_name]
    new_items = function_args.get("new_items")
    if not isinstance(new_items, list) or not new_items:
        return function_args
    required_delta_field = (
        "summary_delta" if function_name == "resource_memory_update" else delta_field
    )
    if not any(
        isinstance(item, dict) and required_delta_field in item
        for item in new_items
    ):
        return function_args
    if not old_items:
        raise ValueError(f"{function_name} delta update references no existing item")
    if len(old_items) not in {1, len(new_items)} and len(new_items) != 1:
        raise ValueError(
            f"ambiguous {function_name} delta mapping: "
            f"{len(old_items)} old items and {len(new_items)} new items"
        )

    expanded: list[dict[str, Any]] = []
    for index, raw_delta in enumerate(new_items):
        if not isinstance(raw_delta, dict) or required_delta_field not in raw_delta:
            raise ValueError(
                f"{function_name} requires {required_delta_field} on every item"
            )
        bases = (
            old_items
            if len(new_items) == 1
            else [old_items[index] if len(old_items) > 1 else old_items[0]]
        )
        base = dict(bases[0])
        delta = dict(raw_delta)
        if function_name == "resource_memory_update":
            delta_summary = _bounded_memory_text(
                delta.pop("summary_delta", ""), 300
            )
            base_content = _unique_nonempty_strings(
                [item.get("content") for item in bases]
            )
            delta_content = _bounded_memory_text(
                delta.pop(delta_field, "") or delta_summary,
                900,
            )
            if delta_content and not any(
                delta_content in content for content in base_content
            ):
                base_content.append(delta_content)
            old_summaries = _unique_nonempty_strings(
                [item.get("summary") for item in bases]
            )
            summary = _bounded_memory_text(
                "\n\n".join(
                    _unique_nonempty_strings([delta_summary, *old_summaries])
                ),
                600,
            )
            item = {
                "title": str(delta.get("title") or base.get("title") or "").strip(),
                "summary": summary,
                "resource_type": str(
                    delta.get("resource_type") or base.get("resource_type") or ""
                ).strip(),
                "content": "\n\n".join(base_content),
                "tree_path": delta.get("tree_path") or base.get("tree_path") or [],
            }
            schema_module = importlib.import_module("mirix.schemas.resource_memory")
            validated = schema_module.ResourceMemoryItemBase(**item)
        else:
            base_steps: list[Any] = []
            for old_item in bases:
                steps = old_item.get("steps") or []
                if isinstance(steps, str):
                    try:
                        steps = json.loads(steps)
                    except json.JSONDecodeError:
                        steps = [steps]
                if isinstance(steps, list):
                    base_steps.extend(steps)
            delta_steps = delta.pop(delta_field, [])
            if not isinstance(delta_steps, list):
                raise ValueError("procedural_memory_update steps_delta must be a list")
            delta_steps = [
                _bounded_memory_text(step, 240) for step in delta_steps[:6]
            ]
            item = {
                "entry_type": str(
                    delta.get("entry_type") or base.get("entry_type") or ""
                ).strip(),
                "summary": str(
                    delta.get("summary") or base.get("summary") or ""
                ).strip()[:600],
                "steps": _unique_nonempty_strings(base_steps + delta_steps),
                "tree_path": delta.get("tree_path") or base.get("tree_path") or [],
            }
            schema_module = importlib.import_module("mirix.schemas.procedural_memory")
            validated = schema_module.ProceduralMemoryItemBase(**item)
        model_dump = getattr(validated, "model_dump", None)
        expanded.append(model_dump() if callable(model_dump) else dict(validated))

    result = dict(function_args)
    result["new_items"] = expanded
    return result


def _expand_mirix_delta_update(
    agent: Any, function_name: str, function_args: dict[str, Any]
) -> dict[str, Any]:
    """Load referenced native rows and expand a delta immediately before execution."""
    if function_name == "semantic_memory_insert":
        items = function_args.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError("semantic_memory_insert requires bounded items")
        bounded_items: list[dict[str, Any]] = []
        schema_module = importlib.import_module("mirix.schemas.semantic_memory")
        for raw_item in items[:3]:
            if not isinstance(raw_item, dict):
                raise ValueError("semantic_memory_insert item must be an object")
            item = dict(raw_item)
            item["name"] = _bounded_memory_text(item.get("name"), 120)
            item["summary"] = _bounded_memory_text(item.get("summary"), 240)
            item["details"] = _bounded_memory_text(item.get("details"), 500)
            item["source"] = _bounded_memory_text(item.get("source"), 160)
            tree_path = item.get("tree_path") or []
            if not isinstance(tree_path, list):
                raise ValueError("semantic_memory_insert tree_path must be a list")
            item["tree_path"] = [
                _bounded_memory_text(value, 64) for value in tree_path[:4]
            ]
            validated = schema_module.SemanticMemoryItemBase(**item)
            model_dump = getattr(validated, "model_dump", None)
            bounded_items.append(
                model_dump() if callable(model_dump) else dict(validated)
            )
        result = dict(function_args)
        result["inner_thoughts"] = _bounded_memory_text(
            result.get("inner_thoughts"), 200
        )
        result["items"] = bounded_items
        return result

    if function_name == "episodic_memory_insert":
        items = function_args.get("items")
        if not isinstance(items, list) or len(items) != 1:
            raise ValueError(
                "episodic_memory_insert requires exactly one bounded event"
            )
        item = dict(items[0]) if isinstance(items[0], dict) else None
        if item is None:
            raise ValueError("episodic_memory_insert item must be an object")
        item["summary"] = _bounded_memory_text(item.get("summary"), 300)
        item["details"] = _bounded_memory_text(item.get("details"), 900)
        schema_module = importlib.import_module("mirix.schemas.episodic_memory")
        validated = schema_module.EpisodicEventForLLM(**item)
        model_dump = getattr(validated, "model_dump", None)
        result = dict(function_args)
        result["items"] = [
            model_dump() if callable(model_dump) else dict(validated)
        ]
        return result

    if function_name == "episodic_memory_merge":
        if "details_delta" not in function_args:
            return function_args
        event_id = str(function_args.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("episodic_memory_merge delta requires event_id")
        old_event = agent.episodic_memory_manager.get_episodic_memory_by_id(event_id)
        if old_event is None:
            raise ValueError(
                f"episodic_memory_merge references missing event_id: {event_id}"
            )
        old_data = _model_dump(old_event) or {}
        details_delta = _bounded_memory_text(
            function_args.get("details_delta"), 900
        )
        if details_delta and details_delta in str(old_data.get("details") or ""):
            details_delta = ""
        result = dict(function_args)
        result.pop("details_delta", None)
        result["combined_details"] = details_delta
        result["combined_summary"] = _bounded_memory_text(
            result.get("combined_summary") or old_data.get("summary"), 300
        )
        schema_module = importlib.import_module("mirix.schemas.episodic_memory")
        schema_module.EpisodicEventUpdate(
            id=event_id,
            summary=result["combined_summary"],
            details=result["combined_details"],
        )
        return result

    old_ids = [str(value) for value in function_args.get("old_ids") or []]
    if not old_ids or not function_args.get("new_items"):
        return function_args
    if function_name == "resource_memory_update":
        manager = agent.resource_memory_manager
        rows = manager.list_resources(agent.agent_state, limit=None) or []
    elif function_name == "procedural_memory_update":
        manager = agent.procedural_memory_manager
        rows = manager.list_procedures(agent.agent_state, limit=None) or []
    else:
        return function_args
    by_id = {
        str(getattr(row, "id", "")): (_model_dump(row) or {}) for row in rows
    }
    missing = [old_id for old_id in old_ids if old_id not in by_id]
    if missing:
        raise ValueError(
            f"{function_name} references missing old_ids: {', '.join(missing)}"
        )
    return _merge_mirix_delta_update(
        function_name, function_args, [by_id[old_id] for old_id in old_ids]
    )


def _restrict_native_memory_update_tools(data: dict[str, Any]) -> dict[str, Any]:
    """Remove redundant read tools only from native memory-update agents."""
    tools = data.get("tools")
    if not isinstance(tools, list):
        return data
    names = {_native_tool_name(tool) for tool in tools}
    if not names.intersection(_MIRIX_NATIVE_MEMORY_WRITE_TOOLS):
        return data
    filtered = [
        tool
        for tool in tools
        if _native_tool_name(tool) not in _MIRIX_REDUNDANT_MEMORY_UPDATE_READ_TOOLS
    ]
    if len(filtered) == len(tools):
        return data
    bounded = dict(data)
    bounded["tools"] = filtered
    return bounded


def _reject_unparsed_native_tool_response(response_data: dict[str, Any]) -> None:
    """Reject a malformed native envelope that compatibility recovery missed."""
    for choice in response_data.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            try:
                parsed_arguments = json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                raise ValueError(
                    "vLLM returned malformed native tool-call arguments"
                )
            if not isinstance(parsed_arguments, dict):
                raise ValueError(
                    "vLLM returned non-object native tool-call arguments"
                )
        if (
            isinstance(content, str)
            and "<tool_call>" in content
            and not message.get("tool_calls")
        ):
            raise ValueError(
                "vLLM returned an unparsed native <tool_call> envelope"
            )


def _accept_recovered_native_tool_finish(response_data: dict[str, Any]) -> None:
    """Treat a capped response as a tool response only after JSON validation."""
    for choice in response_data.get("choices") or []:
        if choice.get("finish_reason") != "length":
            continue
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            continue
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            arguments = json.loads(function.get("arguments") or "")
            if not isinstance(arguments, dict) or not function.get("name"):
                break
        else:
            choice["finish_reason"] = "tool_calls"


def _promote_native_chat_text_response(response_data: dict[str, Any]) -> None:
    """Map Qwen's plain Chat-Agent answer to MIRIX's native send tool."""
    if _ACTIVE_MIRIX_RETRIEVAL.get() is None:
        return
    for choice in response_data.get("choices") or []:
        message = choice.get("message") or {}
        content = message.get("content")
        if message.get("tool_calls") or not isinstance(content, str):
            continue
        content = content.removesuffix("<|im_end|>").strip()
        if not content:
            continue
        message["content"] = None
        message["tool_calls"] = [
            {
                "id": f"call_{uuid.uuid4().hex}",
                "type": "function",
                "function": {
                    "name": "send_message",
                    "arguments": json.dumps({"message": content}, ensure_ascii=False),
                },
            }
        ]
        choice["finish_reason"] = "tool_calls"


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
        payload = _tool_payload(
            str(message.get("content") or ""),
            allow_qa_repair=_ACTIVE_MIRIX_RETRIEVAL.get() is not None,
        )
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


def _normalize_openai_tool_response(response: Any) -> Any:
    """Normalize either a raw dict or MIRIX's Pydantic response model."""
    if isinstance(response, dict):
        return _normalize_openai_tool_tags(response)
    model_dump = getattr(response, "model_dump", None)
    if not callable(model_dump):
        return response
    normalized = _normalize_openai_tool_tags(model_dump())
    return type(response)(**normalized)


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


def _tool_payload(
    text: str, *, allow_qa_repair: bool = False
) -> dict[str, Any] | None:
    # Qwen occasionally finishes a valid JSON object at its generation limit
    # without emitting the optional closing tag.  Treat the opening tag as the
    # authoritative boundary so a valid native memory write is not discarded.
    if "<tool_call>" in text:
        candidate = text.split("<tool_call>", 1)[1]
        candidate = candidate.split("</tool_call>", 1)[0].strip()
    else:
        candidate = text
    candidate = _strip_safe_json_wrappers(candidate)
    # OpenRouter Qwen can honor the tool schema semantically while rendering
    # the call as Python-like text instead of an OpenAI ``tool_calls`` object:
    # ``trigger_memory_update(memory_types=['core', 'episodic'])``. Promote
    # that representation at the same compatibility boundary; the original
    # MIRIX tool executor remains responsible for validation and execution.
    python_payload = _python_style_tool_payload(candidate)
    if python_payload is not None:
        return python_payload
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        repaired = _close_unique_json_suffix(candidate)
        if repaired is not None:
            try:
                value = json.loads(repaired)
            except (json.JSONDecodeError, TypeError):
                value = None
        else:
            value = None
        if not isinstance(value, dict) and allow_qa_repair:
            value = _repair_qa_tool_payload(candidate)
    return value if isinstance(value, dict) else None


def _repair_qa_tool_payload(candidate: str) -> dict[str, Any] | None:
    """Repair malformed JSON only for validated, non-writing Chat-Agent tools."""
    try:
        payload = repair_json(
            candidate,
            return_objects=True,
            skip_json_loads=True,
        )
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) - {"name", "arguments", "args"}:
        return None
    name = str(payload.get("name") or "")
    if name not in _MIRIX_QA_REPAIRABLE_TOOLS:
        return None
    arguments = payload.get("arguments", payload.get("args"))
    if not isinstance(arguments, dict):
        return None
    if name == "send_message":
        if not isinstance(arguments.get("message"), str) or not str(
            arguments["message"]
        ).strip():
            return None
    elif name == "search_in_memory":
        if not isinstance(arguments.get("query"), str) or not str(
            arguments["query"]
        ).strip():
            return None
    elif not all(
        isinstance(arguments.get(key), str) and str(arguments[key]).strip()
        for key in ("start_time", "end_time")
    ):
        return None
    return {"name": name, "arguments": arguments}


def _strip_safe_json_wrappers(text: str) -> str:
    """Remove only deterministic transport noise around one JSON value."""
    cleaned = str(text or "").strip()
    while cleaned.endswith("<|im_end|>"):
        cleaned = cleaned[: -len("<|im_end|>")].rstrip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline >= 0:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return cleaned.strip()


def _close_unique_json_suffix(text: str) -> str | None:
    """Close a JSON value only when its missing suffix is unambiguous."""
    source = str(text or "").strip()
    if not source:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in source:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            stack.append(char)
        elif char in "]}":
            expected = "[" if char == "]" else "{"
            if not stack or stack.pop() != expected:
                return None
    if escaped:
        return None
    suffix = '"' if in_string else ""
    suffix += "".join("]" if char == "[" else "}" for char in reversed(stack))
    return source + suffix if suffix else None


def _python_style_tool_payload(text: str) -> dict[str, Any] | None:
    cleaned = "\n".join(
        line
        for line in str(text or "").splitlines()
        if not line.strip().startswith("```")
    ).strip()
    if cleaned in {"finish_memory_update", "finish_memory_update()"}:
        return {"name": "finish_memory_update", "arguments": {}}

    for match in re.finditer(r"(?m)^\s*([A-Za-z_]\w*)\s*\(", cleaned):
        start = match.start(1)
        suffix = cleaned[start:]
        # Try complete line boundaries in order. ``ast`` determines the real
        # closing parenthesis and safely handles multiline calls, nested lists,
        # and triple-quoted strings without executing model-produced text.
        boundaries = [offset + 1 for offset, char in enumerate(suffix) if char == "\n"]
        boundaries.append(len(suffix))
        for end in boundaries:
            candidate = suffix[:end].strip()
            try:
                expression = ast.parse(candidate, mode="eval").body
            except (SyntaxError, ValueError):
                continue
            if not isinstance(expression, ast.Call) or not isinstance(
                expression.func, ast.Name
            ):
                break
            if expression.args or any(
                keyword.arg is None for keyword in expression.keywords
            ):
                break
            try:
                arguments = {
                    str(keyword.arg): ast.literal_eval(keyword.value)
                    for keyword in expression.keywords
                }
            except (ValueError, TypeError, SyntaxError):
                break
            remainder = (cleaned[:start] + suffix[end:]).strip()
            if remainder:
                arguments.setdefault("inner_thoughts", remainder)
            return {"name": expression.func.id, "arguments": arguments}
    return None
