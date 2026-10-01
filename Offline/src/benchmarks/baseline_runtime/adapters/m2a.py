from __future__ import annotations

import ast
import hashlib
import json
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
from embedding.chunk_builder import Chunk


M2A_UPSTREAM_URL = "https://github.com/Little-Fridge/M2A"
M2A_UPSTREAM_COMMIT = "edd8c3b75bae8b2c9c1a0ac8ed67e38c2c2723f8"
M2A_INTERNAL_PROMPT_SHA256 = {
    "chat_agent_query_tool": "49c5ba9ce0f547183e796749ee77418d8dc988f4f12ece67b74bfd93d0b31682",
    "chat_agent_update_tool": "505394d7b2aaa1434bd73df305c76caa57a65e2f47325d9779986aca200f7ade",
    "memory_manager_query": "d0926cbc75edad8dd0c996d984c4be628c38a61dc0360e06ce6fb5b9e4554c7f",
    "memory_manager_update": "9dca4a43041972b483f752f4d376a290f439f7a0a21c46a0ab17eb4c59d5047b",
    "evaluation_ingest": "4fd91819b6f12846777a68c26c795cbeccc039d38e71760cbe3fb857f821e85a",
    "evaluation_query_chat_agent": "8361aaabc9b9410320656bfe850119711b6f73b94986d4a2fa92c991c5440f5c",
}
M2A_DEVIATIONS = [
    "The original M2A Gemini backbone is replaced by the configured Qwen-VL model.",
    "The original agent-selected searches remain unchanged; only the memories handed to the benchmark answer prompt are capped to the configured experiment Top-K budget.",
    "The final answer uses a fresh two-message benchmark QA-only conversation instead of M2A's answer text; no internal ChatAgent prompt, tool history, or agent response is exposed to the answer model.",
    "Qwen textual <tool_call> output is normalized into the structured tool-call object expected by the unchanged M2A agent graphs; accompanying assistant content is preserved and hash-audited.",
    "QA retrieval uses a disposable raw-message store so checkpoint questions cannot contaminate the persistent conversation memory bank.",
    "Remote VLM requests deduplicate identical image attachments and use size-controlled JPEG transport copies; original image paths, memory records, evidence, and embedding inputs remain unchanged.",
    "Tool-loop budgets preserve their original limits; if a provider returns multiple calls at the boundary only calls within the remaining budget are retained, the final legal call is executed, and an unbound no-tools completion produces a traced final response.",
    "Malformed evidence tuples containing more than two explicit raw-message IDs are losslessly canonicalized into sorted ranges before storage, and the original and normalized values are retained in the native semantic log.",
    "When explicitly enabled, a length-truncated MemoryManager update salvages complete rendered CREATE/DELETE JSON operations, stores an unfinished CREATE text prefix with a [TRUNCATED] marker, and falls back to storing the raw truncated response when no memory text can be recovered.",
    "WorldMemArena may batch up to the configured number of complete rounds from one session into one labelled ChatAgent update; batches close before a second image-bearing source turn, retain every source turn in provenance, and parse WMA's fixed timestamp format locally before falling back to the original time-conversion LLM.",
]


class M2AAdapter(BaselineAdapter):
    """Run the official M2A Agent -> MemoryManager pipeline without direct-store fallbacks."""

    def __init__(self, *, baseline: str, source_root: Path, config: dict[str, Any]) -> None:
        self.baseline = baseline
        self.source_root = source_root
        self.config = dict(config)
        self.backend: Any = None
        self.state_dir: Path | None = None
        self._eval_wrapper: Any = None
        self._ingest_agent: Any = None
        self._conversation_info: dict[str, str] = {}
        self._cur_time: datetime | None = None
        self._raw_sources: dict[int, dict[str, Any]] = {}
        self._execution_trace_path: Path | None = None
        self._initialized = False
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
            max_tokens=int(self.config.get("executor_max_tokens") or 2048),
            timeout=int(self.config.get("request_timeout") or 180),
            max_retries=int(self.config.get("retries", 2)),
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
            memory_manager=MemoryManagerConfig(
                salvage_truncated_updates=bool(
                    self.config.get("m2a_salvage_truncated_updates", False)
                )
            ),
        )

    def reset(self, sample_id: str, state_dir: Path) -> None:
        self.close()
        resolved = state_dir.resolve()
        if resolved == resolved.parent or resolved == Path("/"):
            raise ValueError(f"refusing unsafe M2A state directory: {resolved}")
        self.state_dir = resolved
        if resolved.exists():
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True)

        from agent.m2a import M2ASystem
        from langchain_openai import ChatOpenAI
        from eval_wrapper import M2AEvaluationWrapper

        self.backend = M2ASystem(config=self._build_config(resolved))
        # Reuse the official wrapper methods without creating its second set of stores.
        self._eval_wrapper = M2AEvaluationWrapper.__new__(M2AEvaluationWrapper)
        self._eval_wrapper.m2a = self.backend
        self._eval_wrapper.config = self.backend.config
        self._eval_wrapper.db_dir = resolved.parent
        self._eval_wrapper.cur_time = None
        self._eval_wrapper.time_llm = ChatOpenAI(
            model=self.backend.config.llm.model,
            base_url=self.backend.config.llm.base_url,
            api_key=self.backend.config.llm.api_key,
            temperature=0.0,
            max_tokens=50,
            timeout=self.backend.config.llm.timeout,
            max_retries=self.backend.config.llm.max_retries,
        )
        self._eval_wrapper.chat_idx = sample_id
        self._raw_sources = {}
        self._cur_time = None
        self._initialized = False
        self._execution_trace_path = resolved / "m2a_execution_trace.jsonl"
        self._execution_trace_path.unlink(missing_ok=True)
        self._trace_event(
            component="M2AAdapter",
            action="reset",
            sample_id=sample_id,
            upstream_url=M2A_UPSTREAM_URL,
            upstream_commit=M2A_UPSTREAM_COMMIT,
        )

    @staticmethod
    def _conversation_speakers(chunk: Chunk) -> list[str]:
        values = [
            str(value).strip()
            for value in chunk.metadata.get("m2a_speakers", [])
            if str(value).strip()
        ]
        if not values:
            values = [
                str(turn.get("speaker") or turn.get("role") or "").strip()
                for turn in chunk.metadata.get("m2a_turns", [])
                if isinstance(turn, dict)
            ]
        return list(dict.fromkeys(value for value in values if value))

    def _initialize_ingest_agent(self, chunk: Chunk) -> None:
        if self._initialized:
            return
        from agent.agents.chat_agent import ChatAgent

        speakers = self._conversation_speakers(chunk)
        speaker_0 = speakers[0] if speakers else "Speaker 0"
        # The official wrapper has exactly two prompt slots. For a multiparty
        # benchmark, retain every remaining source name in the second slot.
        speaker_1 = ", ".join(speakers[1:]) if len(speakers) > 1 else "Speaker 1"
        self._conversation_info = {
            "conv_idx": str(chunk.metadata.get("dataset") or "sample"),
            "speaker_0": speaker_0,
            "speaker_1": speaker_1,
        }
        self._eval_wrapper.conv_info = dict(self._conversation_info)
        self._ingest_agent = ChatAgent(
            memory_manager=self.backend.memory_manager,
            raw_store=self.backend.raw_store,
            llm=self.backend.llm,
            image_manager=self.backend.image_manager,
            update_memory=True,
            config=self.backend.config.chat_agent,
            update_only=True,
        )
        ingest_prompt = self._eval_wrapper._get_eval_system_prompt(self._conversation_info)
        self._ingest_agent.init_conversation(system_prompt=ingest_prompt)
        self.backend.chat_agent = self._ingest_agent
        self._initialized = True
        self._trace_event(
            component="ChatAgent",
            action="initialize_ingest",
            speakers=speakers,
            system_prompt_sha256=hashlib.sha256(ingest_prompt.encode("utf-8")).hexdigest(),
        )

    def ingest(self, chunk: Chunk) -> None:
        if self.backend is None:
            raise RuntimeError("M2A adapter has not been reset")
        turns = chunk.metadata.get("m2a_turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(
                f"M2A strict ingest requires metadata.m2a_turns: {chunk.chunk_id}"
            )
        self._initialize_ingest_agent(chunk)
        if chunk.metadata.get("m2a_ingest_mode") == "batched_rounds":
            self._ingest_batched_rounds(chunk, turns)
            return
        image_id_by_path = {
            str(path): str(image_id)
            for path, image_id in zip(
                chunk.images, chunk.metadata.get("image_ids", [])
            )
            if path and image_id
        }
        for turn in turns:
            if not isinstance(turn, dict):
                raise TypeError(f"invalid M2A turn in {chunk.chunk_id}: {turn!r}")
            turn_id = str(turn.get("turn_id") or "")
            speaker = str(turn.get("speaker") or turn.get("role") or "").strip()
            text = str(turn.get("text") or "")
            if not turn_id or not speaker or not text.strip():
                raise ValueError(f"incomplete M2A source turn in {chunk.chunk_id}: {turn!r}")
            formatted_time = self._eval_wrapper._format_time(str(turn.get("timestamp") or ""))
            images = [str(value) for value in turn.get("images", []) if value]
            image_path = images[0] if images else None  # Official wrapper uses the first image.
            input_text = f"({speaker}, {formatted_time}) {text}"
            before = self._memory_fingerprints()
            try:
                self._ingest_agent.chat(
                    user_text=input_text,
                    user_image_path_or_url=image_path,
                    timestamp=formatted_time,
                    role=speaker,
                )
            finally:
                self._trace_tool_budget_events(self._ingest_agent, turn_id=turn_id)
            raw_message = self._ingest_agent.raw_messages[-1]
            raw_id = int(raw_message.msg_id)
            source_dialogue_id = str(
                turn.get("source_dialogue_id")
                or chunk.metadata.get("dialogue_id")
                or chunk.chunk_id
            )
            self._raw_sources[raw_id] = {
                "session_id": str(chunk.metadata.get("session_id") or ""),
                "source_dialogue_ids": [source_dialogue_id],
                "image_ids": [image_id_by_path[image_path]] if image_path in image_id_by_path else [],
                "image_paths": [image_path] if image_path else [],
                "turn_id": turn_id,
            }
            self._trace_event(
                component="ChatAgent",
                action="ingest",
                turn_id=turn_id,
                raw_ids=[raw_id],
                session_id=str(chunk.metadata.get("session_id") or ""),
                source_dialogue_id=source_dialogue_id,
            )
            changed = self._changed_memory_ids(before, self._memory_fingerprints())
            if changed:
                self._trace_event(
                    component="MemoryManager",
                    action="semantic_delta",
                    turn_id=turn_id,
                    raw_ids=[raw_id],
                    memory_ids=changed,
                )
            self._cur_time = formatted_time
            self._eval_wrapper.cur_time = formatted_time

    def _ingest_batched_rounds(
        self, chunk: Chunk, turns: list[dict[str, Any]]
    ) -> None:
        """Ingest several labelled WMA rounds through one native ChatAgent call."""
        image_id_by_path = {
            str(path): str(image_id)
            for path, image_id in zip(
                chunk.images, chunk.metadata.get("image_ids", [])
            )
            if path and image_id
        }
        blocks: list[str] = []
        source_turn_ids: list[str] = []
        source_dialogue_ids: list[str] = []
        formatted_times: list[datetime] = []
        image_turns: list[list[str]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                raise TypeError(f"invalid M2A turn in {chunk.chunk_id}: {turn!r}")
            turn_id = str(turn.get("turn_id") or "")
            source_dialogue_id = str(
                turn.get("source_dialogue_id")
                or chunk.metadata.get("dialogue_id")
                or chunk.chunk_id
            )
            speaker = str(turn.get("speaker") or turn.get("role") or "").strip()
            text = str(turn.get("text") or "")
            if not turn_id or not speaker or not text.strip():
                raise ValueError(f"incomplete M2A source turn in {chunk.chunk_id}: {turn!r}")
            formatted_time = self._format_batched_timestamp(
                str(turn.get("timestamp") or "")
            )
            images = [str(value) for value in turn.get("images", []) if value]
            if images:
                image_turns.append(images)
            blocks.append(
                f"[Source round {source_dialogue_id}]\n"
                f"({speaker}, {formatted_time}) {text}"
            )
            source_turn_ids.append(turn_id)
            if source_dialogue_id not in source_dialogue_ids:
                source_dialogue_ids.append(source_dialogue_id)
            formatted_times.append(formatted_time)
        if len(image_turns) > 1:
            raise ValueError(
                f"M2A batched chunk contains multiple image-bearing turns: {chunk.chunk_id}"
            )

        image_path = image_turns[0][0] if image_turns else None
        input_text = "\n\n".join(blocks)
        before = self._memory_fingerprints()
        try:
            self._ingest_agent.chat(
                user_text=input_text,
                user_image_path_or_url=image_path,
                timestamp=formatted_times[-1],
                role="batched_dialogue",
            )
        finally:
            self._trace_tool_budget_events(
                self._ingest_agent,
                turn_id=f"{chunk.chunk_id}:batch",
                source_turn_ids=source_turn_ids,
            )
        raw_message = self._ingest_agent.raw_messages[-1]
        raw_id = int(raw_message.msg_id)
        session_id = str(chunk.metadata.get("session_id") or "")
        self._raw_sources[raw_id] = {
            "session_id": session_id,
            "source_dialogue_ids": source_dialogue_ids,
            "image_ids": (
                [image_id_by_path[image_path]]
                if image_path in image_id_by_path
                else []
            ),
            "image_paths": [image_path] if image_path else [],
            "turn_id": f"{chunk.chunk_id}:batch",
            "source_turn_ids": source_turn_ids,
        }
        self._trace_event(
            component="ChatAgent",
            action="ingest",
            turn_id=f"{chunk.chunk_id}:batch",
            source_turn_ids=source_turn_ids,
            source_turn_count=len(source_turn_ids),
            source_dialogue_id=source_dialogue_ids[0],
            source_dialogue_ids=source_dialogue_ids,
            raw_ids=[raw_id],
            session_id=session_id,
            ingest_mode="batched_rounds",
            round_count=int(chunk.metadata.get("round_count") or 1),
        )
        changed = self._changed_memory_ids(before, self._memory_fingerprints())
        if changed:
            self._trace_event(
                component="MemoryManager",
                action="semantic_delta",
                turn_id=f"{chunk.chunk_id}:batch",
                source_turn_ids=source_turn_ids,
                raw_ids=[raw_id],
                memory_ids=changed,
            )
        self._cur_time = formatted_times[-1]
        self._eval_wrapper.cur_time = formatted_times[-1]

    def _format_batched_timestamp(self, value: str) -> datetime:
        """Parse benchmark-native WMA timestamps before using M2A's LLM fallback."""
        normalized = str(value or "").strip()
        for time_format in (
            "%b %d, %Y, %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(normalized, time_format)
            except ValueError:
                continue
        return self._eval_wrapper._format_time(normalized)

    def end_session(self, session_id: str) -> None:
        self._trace_event(component="ChatAgent", action="end_session", session_id=session_id)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        if not self._initialized or self.backend is None or self.state_dir is None:
            raise RuntimeError("M2A memory must be built before retrieval")
        from agent.stores import RawMessageStore

        manager = self.backend.memory_manager
        manager.set_handoff_cap(request.top_k)
        manager.reset_retrieval_trace()
        original_raw_store = self.backend.raw_store
        original_agent = self.backend.chat_agent
        scratch_dir = self.state_dir / ".qa_scratch"
        scratch_dir.mkdir(exist_ok=True)
        scratch_path = scratch_dir / f"{hashlib.sha256(request.query_id.encode()).hexdigest()[:20]}.db"
        scratch_path.unlink(missing_ok=True)
        scratch_store = RawMessageStore(db_path=str(scratch_path), reuse=False)
        agent_response = ""
        query_agent: Any = None
        retrieval_error = ""
        # This experiment treats retrieval-agent faults as point-local: keep
        # any completed semantic hits (or an empty handoff) and continue QA.
        # Callers that need the original strict behavior can still opt out.
        fail_open = bool(self.config.get("m2a_fail_open_retrieval", True))
        try:
            self.backend.raw_store = scratch_store
            self._eval_wrapper.m2a = self.backend
            self._eval_wrapper.conv_info = dict(self._conversation_info)
            self._eval_wrapper.cur_time = self._cur_time
            try:
                agent_response = self._eval_wrapper.question(
                    text=request.text,
                    image=[request.query_image] if request.query_image else None,
                )
            except Exception as exc:
                if not fail_open:
                    raise
                retrieval_error = f"{type(exc).__name__}: {exc}"
                self._trace_event(
                    component="M2AAdapter",
                    action="retrieval_fail_open",
                    query_id=request.query_id,
                    error=retrieval_error,
                )
            query_agent = self._eval_wrapper.m2a.chat_agent
        finally:
            if query_agent is None:
                query_agent = getattr(self._eval_wrapper.m2a, "chat_agent", None)
            self._trace_tool_budget_events(query_agent, query_id=request.query_id)
            self.backend.raw_store = original_raw_store
            self.backend.chat_agent = original_agent
            self._eval_wrapper.m2a = self.backend
            scratch_path.unlink(missing_ok=True)

        native_events = manager.get_retrieval_trace()
        search_events = [
            event
            for event in native_events
            if event.get("operation") == "search_semantic_memories"
        ]
        query_events = [
            event for event in native_events if event.get("operation") == "memory_manager_query"
        ]
        if not query_events or not search_events:
            missing_search_error = (
                "M2A ChatAgent retrieval completed without the required "
                "MemoryManager semantic search"
            )
            if not fail_open and not retrieval_error:
                raise RuntimeError(missing_search_error)
            if not retrieval_error:
                retrieval_error = f"RuntimeError: {missing_search_error}"
                self._trace_event(
                    component="M2AAdapter",
                    action="retrieval_fail_open",
                    query_id=request.query_id,
                    error=retrieval_error,
                )
        all_candidates = manager.get_handoff_memories(cap=1_000_000)
        candidate_items: list[RetrievedMemory] = []
        rejected_candidates: list[dict[str, str]] = []
        for record in all_candidates:
            try:
                candidate_items.append(self._retrieved_memory(record))
            except Exception as exc:
                if not fail_open:
                    raise
                rejected_candidates.append(
                    {
                        "memory_id": f"m2a:{record.get('memory_id', '')}",
                        "reason": f"conversion_error: {type(exc).__name__}: {exc}",
                    }
                )
        if request.visible_session_ids:
            if fail_open:
                candidate_items, scope_rejections = self._filter_visible_session_scope(
                    candidate_items, request.visible_session_ids
                )
                rejected_candidates.extend(scope_rejections)
            else:
                self._validate_visible_session_scope(
                    candidate_items[: request.top_k], request.visible_session_ids
                )
            # WMA's agent can return several semantic memories containing the
            # same evidence.  Deduplicate the full ranked candidate stream
            # before applying Top-K so lower-ranked distinct evidence refills
            # the handoff instead of returning repeated slots.
            candidate_items, duplicate_rejections = self._deduplicate_wma_handoff(
                candidate_items
            )
            rejected_candidates.extend(duplicate_rejections)
        items = candidate_items[: request.top_k]
        final_ids = [item.memory_id for item in items]
        semantic_searches = [
            {
                "query_text": event.get("query_text"),
                "query_image": event.get("query_image"),
                "requested_top_k": int(event.get("requested_top_k") or 0),
                "returned_memory_ids": [
                    f"m2a:{value}" for value in event.get("semantic_ids", [])
                ],
                "candidate_count": len(event.get("semantic_ids", [])),
            }
            for event in search_events
        ]
        m2a_trace = {
            "query_id": request.query_id,
            "chat_agent_query_calls": len(query_events),
            "memory_manager_query_calls": len(query_events),
            "semantic_searches": semantic_searches,
            "raw_fetches": [
                event
                for event in native_events
                if str(event.get("operation") or "").startswith("fetch_raw_messages")
            ],
            "candidate_memory_ids": [
                f"m2a:{record['memory_id']}" for record in all_candidates
            ],
            "candidate_count": len(all_candidates),
            "final_memory_ids": final_ids,
            "final_memory_count": len(final_ids),
            "handoff_cap": request.top_k,
            "agent_retrieval_response": agent_response,
            "fail_open_enabled": fail_open,
            "retrieval_error": retrieval_error,
            "rejected_candidates": rejected_candidates,
            "rejected_candidate_count": len(rejected_candidates),
        }
        for event in native_events:
            operation = str(event.get("operation") or "")
            if operation == "memory_manager_query":
                action = "query"
            elif operation == "search_semantic_memories":
                action = "semantic_search"
            else:
                action = operation
            self._trace_event(
                component="MemoryManager",
                action=action,
                query_id=request.query_id,
                memory_ids=[f"m2a:{value}" for value in event.get("semantic_ids", [])],
                raw_ids=[
                    row.get("msg_id")
                    for row in event.get("results", [])
                    if isinstance(row, dict) and row.get("msg_id") is not None
                ],
            )
        self._trace_event(
            component="M2AAdapter",
            action="final_handoff",
            query_id=request.query_id,
            memory_ids=final_ids,
            candidate_count=len(all_candidates),
            handoff_cap=request.top_k,
            fail_open=bool(retrieval_error or rejected_candidates),
            retrieval_error=retrieval_error,
            rejected_candidates=rejected_candidates,
        )
        return RetrievalResult(
            items=items,
            trace={"baseline": self.baseline, "via": "m2a_agent_manager", "m2a_trace": m2a_trace},
        )

    @staticmethod
    def _validate_visible_session_scope(
        items: list[RetrievedMemory], visible_session_ids: tuple[str, ...]
    ) -> None:
        """Reject unscoped or future-backed WMA memories instead of filtering silently."""
        visible = set(visible_session_ids)
        for item in items:
            sessions = {
                str(value)
                for value in item.metadata.get("session_ids", [])
                if str(value)
            }
            if not sessions:
                raise RuntimeError(
                    f"M2A WMA memory lacks session provenance: {item.memory_id}"
                )
            future = sorted(sessions - visible)
            if future:
                raise RuntimeError(
                    f"M2A WMA memory {item.memory_id} references non-visible "
                    f"session(s): {future}"
                )

    @staticmethod
    def _filter_visible_session_scope(
        items: list[RetrievedMemory], visible_session_ids: tuple[str, ...]
    ) -> tuple[list[RetrievedMemory], list[dict[str, str]]]:
        """Drop unsafe WMA candidates so one malformed memory cannot abort QA."""
        visible = set(visible_session_ids)
        accepted: list[RetrievedMemory] = []
        rejected: list[dict[str, str]] = []
        for item in items:
            sessions = {
                str(value)
                for value in item.metadata.get("session_ids", [])
                if str(value)
            }
            if not sessions:
                rejected.append(
                    {"memory_id": item.memory_id, "reason": "missing_session_provenance"}
                )
                continue
            future = sorted(sessions - visible)
            if future:
                rejected.append(
                    {
                        "memory_id": item.memory_id,
                        "reason": f"non_visible_sessions: {future}",
                    }
                )
                continue
            accepted.append(item)
        return accepted, rejected

    @staticmethod
    def _deduplicate_wma_handoff(
        items: list[RetrievedMemory],
    ) -> tuple[list[RetrievedMemory], list[dict[str, str]]]:
        accepted: list[RetrievedMemory] = []
        rejected: list[dict[str, str]] = []
        seen_text: set[str] = set()
        seen_provenance: set[tuple[str, ...]] = set()
        for item in items:
            text_key = " ".join(str(item.text or "").casefold().split())
            provenance_key = tuple(
                sorted({str(value) for value in item.source_dialogue_ids if str(value)})
            )
            if text_key and text_key in seen_text:
                rejected.append(
                    {"memory_id": item.memory_id, "reason": "duplicate_memory_text"}
                )
                continue
            if provenance_key and provenance_key in seen_provenance:
                rejected.append(
                    {
                        "memory_id": item.memory_id,
                        "reason": f"duplicate_provenance: {list(provenance_key)}",
                    }
                )
                continue
            accepted.append(item)
            if text_key:
                seen_text.add(text_key)
            if provenance_key:
                seen_provenance.add(provenance_key)
        return accepted, rejected

    def _retrieved_memory(self, record: dict[str, Any]) -> RetrievedMemory:
        memory_id = str(record["memory_id"])
        evidence_ids = self._evidence_ranges(record.get("evidence_ids"))
        raw_messages = self.backend.raw_store.fetch_by_ids(evidence_ids) if evidence_ids else []
        session_ids: list[str] = []
        dialogue_ids: list[str] = []
        image_ids: list[str] = []
        image_paths: list[str] = []
        for raw in raw_messages:
            source = self._raw_sources.get(int(raw.msg_id), {})
            _append_unique(session_ids, [source.get("session_id")])
            _append_unique(dialogue_ids, source.get("source_dialogue_ids", []))
            _append_unique(image_ids, source.get("image_ids", []))
            _append_unique(image_paths, source.get("image_paths", []))
        if record.get("image_path"):
            _append_unique(image_paths, [record["image_path"]])
        text = str(record.get("text") or "")
        caption = str(record.get("image_caption") or "")
        if caption:
            text = f"{text}\nimage_caption: {caption}" if text else f"image_caption: {caption}"
        return RetrievedMemory(
            memory_id=f"m2a:{memory_id}",
            text=text,
            score=None,
            session_id=session_ids[0] if session_ids else "",
            source_dialogue_ids=dialogue_ids,
            image_ids=image_ids,
            image_paths=image_paths,
            metadata={
                "via": "m2a_agent_manager",
                "session_ids": session_ids,
                "evidence_ids": evidence_ids,
                "evidence_raw_messages": [
                    {
                        "msg_id": int(raw.msg_id),
                        "timestamp": raw.timestamp.isoformat(),
                        "role": raw.role,
                        "text": raw.text,
                        "image_path": raw.image_path or "",
                    }
                    for raw in raw_messages
                ],
                "semantic_image_path": str(record.get("image_path") or ""),
            },
        )

    @staticmethod
    def _evidence_ranges(value: Any) -> list[list[int]]:
        from agent.utils.evidence import normalize_evidence_ranges

        return normalize_evidence_ranges(value)

    def _memory_rows(self) -> list[dict[str, Any]]:
        if self.backend is None:
            return []
        return list(
            self.backend.semantic_store.db.query(
                collection_name="memory",
                filter="id >= 0",
                output_fields=["*"],
                limit=16_384,
            )
            or []
        )

    def _memory_fingerprints(self) -> dict[str, str]:
        return {
            str(row["id"]): hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            for row in self._memory_rows()
        }

    @staticmethod
    def _changed_memory_ids(before: dict[str, str], after: dict[str, str]) -> list[str]:
        return sorted(
            (set(before) | set(after))
            - {memory_id for memory_id in set(before) & set(after) if before[memory_id] == after[memory_id]},
            key=lambda value: int(value),
        )

    def snapshot(self) -> list[MemoryRecord]:
        self._export_native_artifacts()
        records: list[MemoryRecord] = []
        for row in self._memory_rows():
            semantic = {
                "memory_id": str(row.get("id") or ""),
                "text": str(row.get("text") or ""),
                "image_caption": str(row.get("image_caption") or ""),
                "image_path": str(row.get("image_path") or ""),
                "evidence_ids": self._evidence_ranges(row.get("evidence_ids")),
            }
            item = self._retrieved_memory(semantic)
            records.append(
                MemoryRecord(
                    memory_id=item.memory_id,
                    text=item.text,
                    session_id=item.session_id,
                    source_dialogue_ids=item.source_dialogue_ids,
                    image_ids=item.image_ids,
                    image_paths=item.image_paths,
                    backend_type="m2a_semantic",
                    metadata=item.metadata,
                )
            )
        return records

    def _export_native_artifacts(self) -> None:
        if self.backend is None or self.state_dir is None:
            return
        self.backend.raw_store.dump_db(str(self.state_dir / "raw.json"))
        self.backend.semantic_store.dump_db(str(self.state_dir / "semantic.json"))
        self.backend.image_manager.save(str(self.state_dir / "image_manager.json"))
        payload = m2a_conformance_manifest(answer_prompt_sha256="")
        (self.state_dir / "m2a_conformance.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _trace_event(self, *, component: str, action: str, **fields: Any) -> None:
        if self._execution_trace_path is None:
            return
        payload = {
            "event": action,
            "component": component,
            "action": action,
            "recorded_at": datetime.now().isoformat(),
            **fields,
        }
        with self._execution_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def _trace_tool_budget_events(self, agent: Any, **fields: Any) -> None:
        if agent is None:
            return
        owners = (
            ("ChatAgent", agent),
            ("MemoryManager", getattr(agent, "memory_manager", None)),
        )
        for component, owner in owners:
            pop_events = getattr(owner, "pop_tool_budget_events", None)
            if not callable(pop_events):
                continue
            for event in pop_events():
                action = str(event.get("op") or "tool_budget_exhausted")
                self._trace_event(
                    component=component,
                    action=action,
                    **fields,
                    **event,
                )

    def close(self) -> None:
        if self.backend is not None:
            self._export_native_artifacts()
            for name in ("semantic_store", "raw_store"):
                try:
                    getattr(self.backend, name).close()
                except Exception:
                    pass
        self.backend = None
        self._ingest_agent = None
        self._eval_wrapper = None
        self._initialized = False

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "m2a",
            "baseline": self.baseline,
            "available": True,
            "supports_images": True,
            "supports_session_filter": True,
            "requires_python": ">=3.10",
            "upstream_url": M2A_UPSTREAM_URL,
            "upstream_commit": M2A_UPSTREAM_COMMIT,
            "compatibility_mode": "native_agent_manager_handoff_topk",
        }


def m2a_conformance_manifest(*, answer_prompt_sha256: str) -> dict[str, Any]:
    runtime_hashes = _m2a_runtime_prompt_hashes()
    return {
        "upstream_url": M2A_UPSTREAM_URL,
        "upstream_commit": M2A_UPSTREAM_COMMIT,
        "deviations": list(M2A_DEVIATIONS),
        "internal_prompt_sha256": {
            name: {"expected": expected, "actual": runtime_hashes.get(name, "")}
            for name, expected in M2A_INTERNAL_PROMPT_SHA256.items()
        },
        "internal_prompt_hash_method": (
            "SHA256 of exact nested tool docstrings/system-prompt literals; "
            "f-string prompts use ast.dump(include_attributes=False)"
        ),
        "answer_prompt_sha256": answer_prompt_sha256,
    }


def _m2a_runtime_prompt_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[4] / "baselines" / "M2A"
    chat_module = ast.parse((root / "agent/agents/chat_agent.py").read_text(encoding="utf-8"))
    manager_module = ast.parse(
        (root / "agent/agents/memory_manager.py").read_text(encoding="utf-8")
    )
    eval_module = ast.parse((root / "eval_wrapper.py").read_text(encoding="utf-8"))

    hashes: dict[str, str] = {}
    for prompt_name, function_name in (
        ("chat_agent_query_tool", "query_memory"),
        ("chat_agent_update_tool", "update_memory"),
    ):
        node = next(
            node
            for node in ast.walk(chat_module)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        )
        value = ast.get_docstring(node, clean=False) or ""
        hashes[prompt_name] = hashlib.sha256(value.encode("utf-8")).hexdigest()

    for prompt_name, function_name in (
        ("memory_manager_query", "_fill_query_sys_prompt"),
        ("memory_manager_update", "_fill_update_sys_prompt"),
    ):
        function = next(
            node
            for node in ast.walk(manager_module)
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        assignment = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "sys_prompt"
                for target in node.targets
            )
        )
        if not isinstance(assignment.value, ast.Constant):
            raise RuntimeError(f"M2A {function_name} prompt is no longer a literal")
        value = str(assignment.value.value)
        hashes[prompt_name] = hashlib.sha256(value.encode("utf-8")).hexdigest()

    for prompt_name, function_name in (
        ("evaluation_ingest", "_get_eval_system_prompt"),
        ("evaluation_query_chat_agent", "question"),
    ):
        function = next(
            node
            for node in ast.walk(eval_module)
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )
        if function_name == "_get_eval_system_prompt":
            expression = next(
                node.value for node in ast.walk(function) if isinstance(node, ast.Return)
            )
        else:
            expression = next(
                node.args[0]
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "init_conversation"
                and node.args
            )
        serialized = ast.dump(expression, include_attributes=False)
        hashes[prompt_name] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return hashes


def _append_unique(target: list[str], values: Any) -> None:
    known = set(target)
    for value in values:
        text = str(value or "")
        if text and text not in known:
            target.append(text)
            known.add(text)
