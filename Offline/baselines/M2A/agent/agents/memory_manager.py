from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy
import hashlib
import json
import re
from langgraph.graph import END
from typing import Any, Callable, Literal, Optional
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from ..stores import RawMessage, RawMessageStore, SemanticStore, SemanticMemory, ImageManager
from ..config import TIME_FMT, MemoryManagerConfig
from ..utils.evidence import normalize_evidence_ranges
from ..utils.message import (
    deduplicate_message_images,
    is_truncated_completion,
    raise_for_truncated_completion,
)
from .tool_call_normalizer import (
    cap_tool_calls_to_budget,
    normalize_qwen_tool_calls,
)

class MemoryManagerTools:
    """Tools for MemoryManager to query storage layers"""
    raw: RawMessageStore
    semantic_store: SemanticStore
    image_manager: ImageManager
    tool_call_id: str
    
    max_raw_msg: int = 20
    
    def __init__(
        self,
        raw_store: RawMessageStore,
        semantic_store: SemanticStore,
        image_manager: ImageManager,
        max_raw_msg: int = 20,
        retrieval_trace_hook: Optional[Callable[[dict[str, Any]], None]] = None,
        handoff_cap: Optional[int] = None,
    ):
        self.raw = raw_store
        self.semantic = semantic_store
        self.image_manager = image_manager
        self.max_raw_msg = max_raw_msg
        self.retrieval_trace_hook = retrieval_trace_hook
        self._retrieval_trace: list[dict[str, Any]] = []
        self._handoff_memories: list[dict[str, Any]] = []
        self._handoff_memory_ids: set[str] = set()
        self.set_handoff_cap(handoff_cap)

    def set_retrieval_trace_hook(
        self, hook: Optional[Callable[[dict[str, Any]], None]]
    ) -> None:
        self.retrieval_trace_hook = hook

    def set_handoff_cap(self, cap: Optional[int]) -> None:
        """Configure the external handoff view without changing agent retrieval."""
        if cap is not None and cap <= 0:
            raise ValueError("handoff_cap must be a positive integer or None")
        self.handoff_cap = cap

    def reset_retrieval_trace(self) -> None:
        self._retrieval_trace = []
        self._handoff_memories = []
        self._handoff_memory_ids = set()

    def get_retrieval_trace(self) -> list[dict[str, Any]]:
        return deepcopy(self._retrieval_trace)

    def get_handoff_memories(
        self, cap: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Return first-seen, de-duplicated search hits for an external answerer."""
        effective_cap = self.handoff_cap if cap is None else cap
        if effective_cap is not None and effective_cap <= 0:
            raise ValueError("handoff cap must be a positive integer or None")
        memories = self._handoff_memories
        if effective_cap is not None:
            memories = memories[:effective_cap]
        return deepcopy(memories)

    def _record_retrieval(self, event: dict[str, Any]) -> None:
        snapshot = deepcopy(event)
        self._retrieval_trace.append(snapshot)
        if self.retrieval_trace_hook is not None:
            self.retrieval_trace_hook(deepcopy(snapshot))

    def record_memory_manager_query(
        self, *, query_text: Optional[str], query_image: Optional[str]
    ) -> None:
        self._record_retrieval(
            {
                "operation": "memory_manager_query",
                "query_text": query_text,
                "query_image": query_image,
            }
        )

    @staticmethod
    def _semantic_memory_record(mem: SemanticMemory, rank: int) -> dict[str, Any]:
        return {
            "memory_id": str(mem.memory_id),
            "rank": rank,
            "text": mem.text or "",
            "image_caption": mem.image_caption or "",
            "image_path": mem.image_path or "",
            "evidence_ids": deepcopy(mem.evidence_ids),
        }

    def _record_semantic_search(
        self,
        *,
        query_text: Optional[str],
        query_image: Optional[str],
        top_k: int,
        results: list[SemanticMemory],
    ) -> None:
        records = [
            self._semantic_memory_record(memory, rank)
            for rank, memory in enumerate(results, start=1)
        ]
        self._record_retrieval(
            {
                "operation": "search_semantic_memories",
                "query_text": query_text,
                "query_image": query_image,
                "requested_top_k": top_k,
                "semantic_ids": [record["memory_id"] for record in records],
                "results": records,
            }
        )
        for record in records:
            memory_id = record["memory_id"]
            if memory_id not in self._handoff_memory_ids:
                self._handoff_memory_ids.add(memory_id)
                self._handoff_memories.append(record)
    
    def get_search_semantic_memories(self):
        @tool
        def search_semantic_memories(
            query_text: Optional[str] = None, 
            query_image: Optional[str] = None, 
            top_k: int = 10
        ):
            """
            Search high-level semantic memories using text or image or both.
            Returns memory items, each consists of id, content and evidence raw message ID ranges.
            
            Args:
                query_text: Query text
                query_image: Query image. Use image token to refer to image(e.g. <image23>). IMPORTANT: Image tokens are ONLY allowed in this field and MUST NOT appear in others. Set query_image=None if this memory contains no image.
                top_k: Number of results to return
            """
            requested_query_image = query_image
            if query_image and query_image != 'N/A':
                query_image = self.image_manager.image_token_to_image(query_image)
            
            results = self.semantic.hybrid_search(
                query_text=query_text,
                query_image_path=query_image,  # For prototype, text-only
                top_k=top_k
            )

            self._record_semantic_search(
                query_text=query_text,
                query_image=requested_query_image,
                top_k=top_k,
                results=results,
            )
            
            if not results:
                return "No relevant semantic memories found."
            
            output = []
            images = []
            for i, mem in enumerate(results):
                json_msg = {
                    "id": mem.memory_id,
                    "text": mem.text or 'N/A',
                    "image": "<image>" if mem.image_path else 'N/A',
                    "Evidence IDs": mem.evidence_ids
                }
                output.append(json_msg)
                if mem.image_path:
                    images.append(mem.image_path)
            
            return self.image_manager.format_obj_to_content(output, images)
        
        return search_semantic_memories
    
    def get_fetch_raw_messages(self):
        @tool
        def fetch_raw_messages(id_ranges: str) -> list[dict]:
            """
            Fetch raw messages by ID ranges. 
            
            WARNING: Use with caution - this tool can return LARGE amounts of data
            that may exceed context limits and impact performance.
            
            Best practices:
            1. Only call this tool when you DO need to examine raw messages
            2. Always estimate result size before calling
            3. Use narrow, specific ranges when possible
            4. Consider iterative fetching for large ranges when necessary
            5. Only request data actually needed for the current task
            6. Give ranges in order.
            
            Args:
                id_ranges: JSON string of ranges, e.g. "[[1,5], [12,12]]"
            """
            try:
                ranges = json.loads(id_ranges)
                messages = self.raw.fetch_by_ids(ranges)
                
                if not messages:
                    self._record_retrieval(
                        {
                            "operation": "fetch_raw_messages",
                            "requested_id_ranges": ranges,
                            "returned_count": 0,
                            "truncated": False,
                            "results": [],
                        }
                    )
                    return f"No messages found in ranges {id_ranges}"
                
                output = [{
                    "type": "text", "text": f"{min(len(messages), self.max_raw_msg)} messages fetched" + (
                        f", truncated to {self.max_raw_msg}:\n\n" if len(messages) > self.max_raw_msg else ":\n\n"
                    )
                }]
                images = []
                for msg in messages[:self.max_raw_msg]:
                    output.append({
                        "id": msg.msg_id,
                        "timestamp": msg.timestamp.strftime(TIME_FMT),
                        "speaker": msg.role,
                        "text": msg.text or 'N/A',
                        "image": "<image>" if msg.image_path else 'N/A'
                    })
                    if msg.image_path:
                        images.append(msg.image_path)

                self._record_retrieval(
                    {
                        "operation": "fetch_raw_messages",
                        "requested_id_ranges": ranges,
                        "returned_count": min(len(messages), self.max_raw_msg),
                        "truncated": len(messages) > self.max_raw_msg,
                        "results": [
                            {
                                "msg_id": msg.msg_id,
                                "timestamp": msg.timestamp.strftime(TIME_FMT),
                                "role": msg.role,
                                "text": msg.text or "",
                                "image_path": msg.image_path or "",
                            }
                            for msg in messages[:self.max_raw_msg]
                        ],
                    }
                )
                              
                return self.image_manager.format_obj_to_content(output, images)

            except Exception as e:
                # print(e)
                # print(f"Error parsing ID ranges: {id_ranges}")
                return f"Error parsing ID ranges: {id_ranges}"
    
        return fetch_raw_messages
    
    def get_fetch_raw_messages_by_time(self):
        @tool
        def fetch_raw_messages_by_time(start_date: str, end_date: str) -> str:
            """
            Fetch raw messages within a time range.
            
            Args:
                start_date: ISO format {(YYYY-MM-DD)}
                end_date: ISO format (YYYY-MM-DD)
            """
            try:
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
                messages = self.raw.fetch_by_timerange(start, end)
                
                if not messages:
                    self._record_retrieval(
                        {
                            "operation": "fetch_raw_messages_by_time",
                            "start_date": start_date,
                            "end_date": end_date,
                            "returned_count": 0,
                            "truncated": False,
                            "results": [],
                        }
                    )
                    return f"No messages between {start_date} and {end_date}"
                
                output = [f"Found {len(messages)} messages:\n"]
                for msg in messages[:20]:  # Limit output
                    output.append(f"[{msg.msg_id}] {msg.timestamp.strftime(TIME_FMT)} - {msg.role}: {msg.text[:50]}...")

                self._record_retrieval(
                    {
                        "operation": "fetch_raw_messages_by_time",
                        "start_date": start_date,
                        "end_date": end_date,
                        "returned_count": min(len(messages), 20),
                        "truncated": len(messages) > 20,
                        "results": [
                            {
                                "msg_id": msg.msg_id,
                                "timestamp": msg.timestamp.strftime(TIME_FMT),
                                "role": msg.role,
                                "text": msg.text or "",
                                "image_path": msg.image_path or "",
                            }
                            for msg in messages[:20]
                        ],
                    }
                )
                
                return "\n".join(output)
            except Exception as e:
                # print(e)
                # print("Error parsing dates")
                return f"Error parsing dates"
        
        return fetch_raw_messages_by_time

    def get_add_memory(self):
        @tool
        def add_memory(
            text: str,
            image: Optional[str] = None,
            image_caption: Optional[str] = None,
            evidence_ids: str = "[]",
        ) -> str:
            """
            Create a new semantic memory entry.
            For optional args, set to None if you don't need them.
            
            Args:
                text: Memory text content
                image: Memory image content. Use image token to refer to image(e.g. <image23>). IMPORTANT: Image tokens are ONLY allowed in this field and MUST NOT appear in others. Set image=None if this memory contains no image.
                image_caption: Caption for associated images
                evidence_ids: JSON string of ID ranges, e.g. "[[1,3], [5,5]]"
            """
            try:
                raw_ev_ids = json.loads(evidence_ids)
                ev_ids = normalize_evidence_ranges(raw_ev_ids)
                if ev_ids != raw_ev_ids:
                    self.semantic.log.append(
                        {
                            "op": "normalize_evidence_ids",
                            "original": raw_ev_ids,
                            "normalized": ev_ids,
                        }
                    )
                if image == 'N/A':
                    image = None
                if image_caption == 'N/A':
                    image_caption = None
                memory = SemanticMemory(
                    # memory_id=str(uuid.uuid4()),
                    text=text,
                    image_caption=image_caption if image_caption else None,
                    image_path=self.image_manager.image_token_to_image(image),
                    evidence_ids=ev_ids,
                )
                mem_id = self.semantic.add(memory)
                return f"Created memory, id: {mem_id}"
            except Exception as e:
                # print(e)
                # print(f"Error creating memory: {e}")
                return f"Error creating memory: {e}"

        return add_memory
        
    def get_delete_memory(self):
        @tool
        def delete_memory(memory_id: str) -> str:
            """
            Delete a semantic memory.
            
            Args:
                memory_id: ID of memory to delete
            """
            success = self.semantic.delete(memory_id)
            if success:
                return f"Deleted memory {memory_id}"
            else:
                return f"Memory {memory_id} not found"

        return delete_memory
   
@dataclass
class MemoryManagerState:
    """State for MemoryManager graph"""
    messages: list[BaseMessage]
    
    # Input from ChatAgent
    operation: Literal["query", "update"] = "query"
    query_text: Optional[str] = None
    query_image_path: Optional[str] = None
    
    context: list[RawMessage] = field(default_factory=list)
    
    # Iterative retrieval state
    messages: list[BaseMessage] = field(default_factory=list)
    iteration_count: int = 0
    
    response: str | list[dict] = ""


class MemoryManager:
    max_iteration: int = 15
    tools: dict[str, BaseTool]
    
    def __init__(
        self,
        tools: MemoryManagerTools,
        raw_store: RawMessageStore,
        semantic_store: SemanticStore,
        llm: ChatOpenAI,
        image_manager: ImageManager,
        config: MemoryManagerConfig
    ):
        # self.tools = tools
        self.raw_store = raw_store
        self.semantic_store = semantic_store
        self.tools = {
            "search_semantic_memories": tools.get_search_semantic_memories(),
            "fetch_raw_messages": tools.get_fetch_raw_messages(),
            "fetch_raw_messages_by_time": tools.get_fetch_raw_messages_by_time(),
            
            "add_memory": tools.get_add_memory(),
            "delete_memory": tools.get_delete_memory(),
        }
        self.tool_cls = tools
        self.config = config
        self._tool_budget_events: list[dict[str, Any]] = []
        
        self.image_manager = image_manager
        self.llm = llm
        self.graph = self._build_graph()

    def pop_tool_budget_events(self) -> list[dict[str, Any]]:
        events = list(self._tool_budget_events)
        self._tool_budget_events.clear()
        return events

    def _record_forced_finalize(self, *, operation: str, iterations: int) -> None:
        self._tool_budget_events.append(
            {
                "operation": operation,
                "tool_iterations": iterations,
                "max_tool_iterations": self.max_iteration,
                "budget_exhausted": True,
                "forced_finalize": True,
            }
        )

    def _cap_response_tool_calls(
        self, response: AIMessage, *, operation: str, iterations: int
    ) -> AIMessage:
        remaining = max(0, self.max_iteration - iterations)
        offered = len(response.tool_calls)
        if offered > remaining:
            self._tool_budget_events.append(
                {
                    "operation": operation,
                    "tool_iterations": iterations,
                    "max_tool_iterations": self.max_iteration,
                    "budget_exhausted": True,
                    "forced_finalize": False,
                    "tool_calls_offered": offered,
                    "tool_calls_accepted": remaining,
                    "tool_calls_omitted": offered - remaining,
                }
            )
        return cap_tool_calls_to_budget(response, remaining)

    @staticmethod
    def _has_text_content(response: AIMessage) -> bool:
        content = response.content
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(
                isinstance(block, str) and bool(block.strip())
                or isinstance(block, dict)
                and bool(str(block.get("text") or block.get("content") or "").strip())
                for block in content
            )
        return False

    @staticmethod
    def _response_text(response: AIMessage) -> str:
        content = response.content
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content or "")
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if value:
                    parts.append(str(value))
        return "\n".join(parts)

    @staticmethod
    def _complete_json_objects(text: str) -> list[tuple[dict[str, Any], int, int]]:
        """Find complete top-level JSON objects embedded in rendered prose."""
        decoder = json.JSONDecoder()
        objects: list[tuple[dict[str, Any], int, int]] = []
        cursor = 0
        while cursor < len(text):
            start = text.find("{", cursor)
            if start < 0:
                break
            try:
                value, consumed = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            end = start + consumed
            if isinstance(value, dict):
                objects.append((value, start, end))
            cursor = max(end, start + 1)
        return objects

    @staticmethod
    def _decode_json_string_prefix(value: str) -> str:
        """Decode the recoverable prefix of an unterminated JSON string."""
        output: list[str] = []
        index = 0
        escapes = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        while index < len(value):
            char = value[index]
            if char != "\\":
                output.append(char)
                index += 1
                continue
            if index + 1 >= len(value):
                break
            escaped = value[index + 1]
            if escaped == "u" and index + 5 < len(value):
                codepoint = value[index + 2 : index + 6]
                if re.fullmatch(r"[0-9a-fA-F]{4}", codepoint):
                    output.append(chr(int(codepoint, 16)))
                    index += 6
                    continue
            output.append(escapes.get(escaped, escaped))
            index += 2
        return "".join(output).strip()

    @classmethod
    def _unfinished_create_text(
        cls,
        text: str,
        complete_objects: list[tuple[dict[str, Any], int, int]],
    ) -> str:
        """Recover only an unterminated text field from a trailing CREATE."""
        tail_start = max((end for _value, _start, end in complete_objects), default=0)
        tail = text[tail_start:]
        markers = list(re.finditer(r'"text"\s*:\s*"', tail, re.IGNORECASE))
        for marker in reversed(markers):
            preceding = tail[: marker.start()]
            create_positions = [
                match.start()
                for pattern in (
                    r'"operation"\s*:\s*"(?:CREATE|ADD)"',
                    r'"name"\s*:\s*"add_memory"',
                )
                for match in re.finditer(pattern, preceding, re.IGNORECASE)
            ]
            delete_positions = [
                match.start()
                for pattern in (
                    r'"operation"\s*:\s*"DELETE"',
                    r'"name"\s*:\s*"delete_memory"',
                )
                for match in re.finditer(pattern, preceding, re.IGNORECASE)
            ]
            if not create_positions or (
                delete_positions and max(delete_positions) > max(create_positions)
            ):
                continue

            raw_start = marker.end()
            escaped = False
            raw_end = len(tail)
            for offset, char in enumerate(tail[raw_start:]):
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    raw_end = raw_start + offset
                    break
            return cls._decode_json_string_prefix(tail[raw_start:raw_end])
        return ""

    @staticmethod
    def _fallback_evidence(context: list[RawMessage]) -> list[list[int]]:
        ids = sorted(
            {
                int(message.msg_id)
                for message in context
                if isinstance(message.msg_id, int) and message.msg_id > 0
            }
        )
        return normalize_evidence_ranges(ids)

    @staticmethod
    def _salvaged_create_args(
        memory: dict[str, Any], fallback_evidence: list[list[int]]
    ) -> dict[str, Any] | None:
        text = memory.get("text")
        if not isinstance(text, str) or not text.strip():
            return None

        raw_evidence = memory.get("evidence_ids")
        if raw_evidence is None:
            raw_evidence = memory.get("Evidence IDs")
        try:
            evidence = normalize_evidence_ranges(raw_evidence)
            if not evidence:
                evidence = fallback_evidence
        except (TypeError, ValueError, json.JSONDecodeError):
            evidence = fallback_evidence

        image = memory.get("image")
        image_caption = memory.get("image_caption")
        if isinstance(image, dict):
            image_caption = image_caption or image.get("caption") or image.get(
                "image_caption"
            )
            image = image.get("image_token") or image.get("token")
        if not isinstance(image, str) or not re.fullmatch(r"<image\d+>", image):
            image = None
        if not isinstance(image_caption, str) or not image_caption.strip():
            image_caption = None

        return {
            "text": text.strip(),
            "image": image,
            "image_caption": image_caption,
            "evidence_ids": json.dumps(evidence),
        }

    @classmethod
    def _salvaged_json_actions(
        cls,
        payload: dict[str, Any],
        fallback_evidence: list[list[int]],
    ) -> list[dict[str, Any]]:
        name = str(payload.get("name") or "").strip()
        arguments = payload.get("arguments")
        if name and isinstance(arguments, dict):
            if name == "add_memory":
                args = cls._salvaged_create_args(arguments, fallback_evidence)
                return [{"name": "add_memory", "args": args}] if args else []
            if name == "delete_memory" and arguments.get("memory_id") is not None:
                return [
                    {
                        "name": "delete_memory",
                        "args": {"memory_id": str(arguments["memory_id"])},
                    }
                ]
            return []

        operation = str(payload.get("operation") or payload.get("op") or "").upper()
        if operation in {"CREATE", "ADD"}:
            memory = payload.get("memory")
            if not isinstance(memory, dict):
                memory = payload
            args = cls._salvaged_create_args(memory, fallback_evidence)
            return [{"name": "add_memory", "args": args}] if args else []
        if operation == "DELETE":
            memory_ids = payload.get("memory_ids")
            if memory_ids is None:
                memory_ids = [payload.get("memory_id")]
            elif not isinstance(memory_ids, list):
                memory_ids = [memory_ids]
            return [
                {
                    "name": "delete_memory",
                    "args": {"memory_id": str(memory_id)},
                }
                for memory_id in memory_ids
                if memory_id is not None and str(memory_id).strip()
            ]
        return []

    def _salvage_truncated_update(
        self, response: AIMessage, state: MemoryManagerState
    ) -> str:
        """Execute the explicitly enabled best-effort truncated-update policy."""
        normalized = response
        try:
            normalized = normalize_qwen_tool_calls(response)
        except ValueError:
            # The trailing textual tool envelope may itself be the truncated part.
            normalized = response

        content = self._response_text(normalized)
        fallback_evidence = self._fallback_evidence(state.context)
        actions: list[dict[str, Any]] = []
        recovered_memory_text = False

        for tool_call in normalized.tool_calls:
            name = str(tool_call.get("name") or "")
            args = tool_call.get("args")
            if name == "add_memory" and isinstance(args, dict):
                create_args = self._salvaged_create_args(args, fallback_evidence)
                if create_args:
                    actions.append({"name": "add_memory", "args": create_args})
                    recovered_memory_text = True
            elif (
                name == "delete_memory"
                and isinstance(args, dict)
                and args.get("memory_id") is not None
            ):
                actions.append(
                    {
                        "name": "delete_memory",
                        "args": {"memory_id": str(args["memory_id"])},
                    }
                )

        complete_objects = self._complete_json_objects(content)
        for payload, _start, _end in complete_objects:
            recovered = self._salvaged_json_actions(payload, fallback_evidence)
            actions.extend(recovered)
            recovered_memory_text = recovered_memory_text or any(
                action["name"] == "add_memory" for action in recovered
            )

        partial_text = self._unfinished_create_text(content, complete_objects)
        if partial_text:
            suffix = "" if partial_text.endswith((" ", "\n")) else " "
            actions.append(
                {
                    "name": "add_memory",
                    "args": {
                        "text": f"{partial_text}{suffix}[TRUNCATED]",
                        "image": None,
                        "image_caption": None,
                        "evidence_ids": json.dumps(fallback_evidence),
                    },
                }
            )
            recovered_memory_text = True

        raw_fallback = False
        if not recovered_memory_text and content.strip():
            actions.append(
                {
                    "name": "add_memory",
                    "args": {
                        "text": f"[TRUNCATED MODEL OUTPUT]\n{content.strip()}",
                        "image": None,
                        "image_caption": None,
                        "evidence_ids": json.dumps(fallback_evidence),
                    },
                }
            )
            raw_fallback = True

        deduplicated: list[dict[str, Any]] = []
        seen: set[str] = set()
        for action in actions:
            signature = json.dumps(action, ensure_ascii=False, sort_keys=True)
            if signature not in seen:
                seen.add(signature)
                deduplicated.append(action)

        results: list[dict[str, str]] = []
        for action in deduplicated:
            result = self.tools[action["name"]].invoke(action["args"])
            results.append(
                {
                    "name": action["name"],
                    "result": str(result),
                }
            )

        audit = {
            "op": "salvage_truncated_update",
            "raw_response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "complete_json_objects": len(complete_objects),
            "actions_executed": len(deduplicated),
            "creates_executed": sum(
                action["name"] == "add_memory" for action in deduplicated
            ),
            "deletes_executed": sum(
                action["name"] == "delete_memory" for action in deduplicated
            ),
            "partial_text_saved": bool(partial_text),
            "raw_fallback_saved": raw_fallback,
            "fallback_evidence_ids": fallback_evidence,
            "results": results,
        }
        log = getattr(self.semantic_store, "log", None)
        if isinstance(log, list):
            log.append(audit)
        return json.dumps(audit, ensure_ascii=False)

    def reset_retrieval_trace(self) -> None:
        self.tool_cls.reset_retrieval_trace()

    def set_retrieval_trace_hook(
        self, hook: Optional[Callable[[dict[str, Any]], None]]
    ) -> None:
        self.tool_cls.set_retrieval_trace_hook(hook)

    def set_handoff_cap(self, cap: Optional[int]) -> None:
        self.tool_cls.set_handoff_cap(cap)

    def get_retrieval_trace(self) -> list[dict[str, Any]]:
        return self.tool_cls.get_retrieval_trace()

    def get_handoff_memories(
        self, cap: Optional[int] = None
    ) -> list[dict[str, Any]]:
        return self.tool_cls.get_handoff_memories(cap)
        
    def _prepair_context(self, context: list[RawMessage]) -> list[dict]:
        content = [{
            "type": "text", "text": "[\n"
        }]
        for msg in context:
            chunks = self.image_manager.format_to_msg_content(
                text=msg.text,
                image=msg.image_path,
                speaker=msg.role,
                timestamp=msg.timestamp.strftime(TIME_FMT),
                message_id=msg.msg_id
            )
            content += chunks + [{
                "type": "text", "text": ",\n"
            }]
        
        content.append({
            "type": "text", "text": "]\n"
        })
        return content
            
    def _fill_query_sys_prompt(self, state: MemoryManagerState) -> MemoryManagerState:
        sys_prompt = """You are MemoryManager handling a QUERY request from ChatAgent.

=== GOAL ===
Retrieve relevant information and return a focused answer to the query.
Use image tokens (e.g. <image23>) when including images in your response.


=== EXECUTION FLOW ===
1. SEARCH semantic memories (always start here)
2. ASSESS if results answer the query:
   - Sufficient? → Proceed to step 4
   - Need details? → Fetch raw messages using evidence_ids
3. ITERATE if needed (additional searches/fetches)
4. RESPOND with a concise, query-focused answer and provide neccessary context.


IMPORTANT:
1. If there are time references (e.g. "last year", "2 months ago"), 
    fitst reason and determine what's the reference time, then calculate the actual date based on the timestamp.
    e.g. (message on July 5th, send by Mary) I went for a trip yesterday -> reference time: July 5th -> actual trip day: July 4th.
    Assume current time is the last messege's time.
2. Always convert relative time references to specific dates, months or years.
3. For optional arguments when calling tools, DO NOT give them in tool call args dict when you don't need them. (i.e. don't fill with 'N/A')


=== EXAMPLES ===
Query: "When did Jack visit Tokyo?"
→ search_semantic_memories("Jack Tokyo visit")
→ Respond: "Jack visited Tokyo in March 2024"

Query: "What did Jack say about the restaurant yesterday?"(current time: Jun 9th)
→ search_semantic_memories("Jack restaurant Jun 8th")
→ fetch_raw_messages using evidence_ids (need exact words)
→ Respond: [Jack's specific comments]

Query: "What's Sarah's cat's name and can you show me?"
→ search_semantic_memories("Sarah cat")
→ Respond: "Sarah's cat is named Whiskers, shown here: <image12>"


Most recent chat messages: 
---
<context>
---
Current query: <query>
"""
        
        # 1. prepair messages
        sys_prompt_content = []
        chunks = re.split(r'(<context>|<query>)', sys_prompt)
        for chunk in chunks:
            if chunk == "<context>":
                sys_prompt_content += self._prepair_context(state.context)
            elif chunk == "<query>":
                sys_prompt_content += self.image_manager.format_to_msg_content(
                    text=state.query_text,
                    image=state.query_image_path
                )
            else:
                sys_prompt_content.append({
                "type": "text", "text": chunk
            })
        
        state.messages = [HumanMessage(sys_prompt_content)]
        return state
        
    def _handle_query(
        self, 
        state: MemoryManagerState
    ) -> Command:
        messages = state.messages
        request_messages = deduplicate_message_images(messages)
        query_tools = [
            self.tools["search_semantic_memories"],
            self.tools["fetch_raw_messages"],
            self.tools["fetch_raw_messages_by_time"],      
        ]
        forced_finalize = state.iteration_count >= self.max_iteration
        if forced_finalize:
            self._record_forced_finalize(
                operation="query", iterations=state.iteration_count
            )
            # Match the original M2A terminal transition: once the tool budget
            # is exhausted, make a plain completion request with no tool schema.
            # Returning here also guarantees that no additional tool call can be
            # routed to exec_tool, even if a provider populates tool_calls.
            response = self.llm.invoke(request_messages)
            raise_for_truncated_completion(response)
            if not self._has_text_content(response):
                raise RuntimeError(
                    "M2A MemoryManager returned an empty query response during "
                    "forced finalization"
                )
            messages.append(response)
            try:
                formatted_response = self.image_manager.format_msg_to_content(
                    response.content
                )
            except Exception as e:
                messages.append(HumanMessage(content=str(e)))
                return Command(
                    update={"messages": messages},
                    goto="handle_query"
                )
            return Command(
                update={"messages": messages, "response": formatted_response},
                goto=END
            )

        response = self.llm.bind_tools(
            query_tools,
            parallel_tool_calls=False,
        ).invoke(request_messages)
        raise_for_truncated_completion(response)
        response = normalize_qwen_tool_calls(response)
        response = self._cap_response_tool_calls(
            response, operation="query", iterations=state.iteration_count
        )
        messages.append(response)
        if not response.tool_calls:
            try:
                response = self.image_manager.format_msg_to_content(
                    response.content
                )
            except Exception as e:
                messages.append(HumanMessage(content=str(e)))
                return Command(
                    update={"messages": messages},
                    goto="handle_query"
                )
            return Command(
                update={"messages": messages, "response": response},
                goto=END
            )
        
        return Command(
            update={"messages": messages},
            goto="exec_tool"
        )
    
    def _exec_tool(
        self, 
        state: MemoryManagerState
    ) -> Command[Literal['handle_query', 'handle_update']]:
        last_message: AIMessage = state.messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state

        for tool_call in last_message.tool_calls:
            if state.iteration_count >= self.max_iteration:
                raise RuntimeError(
                    "M2A MemoryManager received a tool call after its tool budget "
                    "was exhausted"
                )
            state.iteration_count += 1
            
            try:
                query_image = tool_call["args"].get("query_image")
                if query_image == 'N/A':
                    query_image = None
                query_image = self.image_manager.image_token_to_image(query_image)
            except Exception as e:
                state.messages.append(ToolMessage(
                    content=f"Tool error: {e}",
                    tool_call_id=tool_call["id"]
                ))
                return state
                
            # self.tools_cls.tool_call_id = tool_call["id"]
            try:
                tool_result = self.tools[tool_call["name"]].invoke(input=tool_call["args"])
            except Exception as e:
                tool_result = f"Tool error: Please check your input and try again. ({str(e)})"
            
            # memory_result = self._prepair_tool_result(memory_result)
            
            state.messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call['id'])
            )
        
        update = {
            "messages": state.messages,
            "iteration_count": state.iteration_count
        }
        if state.operation == 'query':
            return Command(
                update=update,
                goto="handle_query"
            )
        return Command(update=update,goto="handle_update")
    
    def _fill_update_sys_prompt(self, state: MemoryManagerState) -> MemoryManagerState:
        sys_prompt = """You are MemoryManager processing an UPDATE request from ChatAgent.

=== YOUR ROLE ===
Analyze ChatAgent's update suggestion and maintain the semantic memory database.
You have access to:
- Semantic memory DB: High-level, structured memories you maintain
- Raw message DB: Original conversation messages (read-only)

=== EXECUTION FLOW ===

Step 1: UNDERSTAND THE REQUEST
- Parse ChatAgent's suggestion
- Identify key entities, events, timeframes, and relationships
- Determine if this is: new information / update / correction / summarization

Step 2: QUERY EXISTING MEMORY
- Search semantic memories for related content
- If needed, fetch raw messages for context (use sparingly - see tool warnings)
- Identify: duplicates, contradictions, gaps, related memories. If there are contradictions,
    prioritize the most recent one.

Step 3: PLAN OPERATIONS
Decide on operation type(s):
- CREATE: Novel information not present in memory
- DELETE: Outdated, contradicted, or now-subsumed information
- BOTH: When updating (delete old + add refined version)
- NONE: If information already adequately captured

Step 4: EXECUTE & VERIFY
- Perform planned operations
- For complex updates, you may need multiple add_memory calls to capture different granularities
- Verify no critical information was lost

Step 5: COMPLETE
- When done, respond ONLY with: ""

IMPORTANT:
[Temporal Processing]
- ALWAYS resolve relative time to absolute dates
  ✓ Message: "went hiking yesterday" (Jan 7, 2022) → Store: "Jan 6, 2022"
  ✓ "last summer" (Nov 2023) → "Summer 2023" or "Jun-Aug 2023"
  ✓ "two months ago" (Dec 15, 2024) → "mid-October 2024" or "Oct 15, 2024"
- Include time context in memories unless the information is truly timeless
- Preserve original timestamp precision when available

[Entity References]
- Use SPECIFIC NAMES, never pronouns (he/she/they) or generic terms (user/person)
  ✓ "Jack prefers morning coffee"
  ✗ "He prefers morning coffee"
  ✗ "User prefers morning coffee"
- When speaker name is ambiguous, use context or evidence_ids to clarify

[Granularity Strategy]
- Break complex information into atomic + summary memories:
  Example: "Jack's Tokyo trip (Mar 2025): cherry blossoms, temples, budget travel"
  → Add multiple memories:
    1. "Jack planning Tokyo trip in March 2025" (summary)
    2. "Jack interested in cherry blossom viewing during Tokyo trip"
    3. "Jack wants to visit traditional temples in Tokyo"
    4. "Jack is budget-conscious for Tokyo trip"

[Evidence Tracking]
- ALWAYS provide evidence_ids linking memories to source messages
- Combine contiguous ranges

Most recent chat messages: <context>

ChatAgent suggests: <query>
"""
        sys_prompt_content = []
        chunks = re.split(r'(<context>|<query>)', sys_prompt)
        for chunk in chunks:
            if chunk == "<context>":
                sys_prompt_content += self._prepair_context(state.context)
            elif chunk == "<query>":
                sys_prompt_content += self.image_manager.format_to_msg_content(
                    text=state.query_text,
                    image=state.query_image_path
                )
            else:
                sys_prompt_content.append({
                "type": "text", "text": chunk
            })
        
        state.messages = [HumanMessage(sys_prompt_content)]
        return state
    
    def _handle_update(self, state: MemoryManagerState) -> Command:
        messages = state.messages
        request_messages = deduplicate_message_images(messages)
        update_tools = [
            self.tools["search_semantic_memories"],
            self.tools["fetch_raw_messages"],
            self.tools["fetch_raw_messages_by_time"],   
            
            self.tools["add_memory"],
            self.tools["delete_memory"],
        ]
        forced_finalize = state.iteration_count >= self.max_iteration
        if forced_finalize:
            self._record_forced_finalize(
                operation="update", iterations=state.iteration_count
            )
            # Do not expose tools in the terminal request.  The hard tool budget
            # remains unchanged and the response cannot enter exec_tool.
            response = self.llm.invoke(request_messages)
            if is_truncated_completion(response) and bool(
                getattr(self.config, "salvage_truncated_updates", False)
            ):
                messages.append(response)
                salvaged = self._salvage_truncated_update(response, state)
                return Command(
                    update={"messages": messages, "response": salvaged},
                    goto=END,
                )
            raise_for_truncated_completion(response)
            messages.append(response)
            return Command(
                update={"messages": messages, "response": response.content},
                goto=END
            )

        response = self.llm.bind_tools(
            update_tools,
            parallel_tool_calls=False,
        ).invoke(request_messages)
        if is_truncated_completion(response) and bool(
            getattr(self.config, "salvage_truncated_updates", False)
        ):
            messages.append(response)
            salvaged = self._salvage_truncated_update(response, state)
            return Command(
                update={"messages": messages, "response": salvaged},
                goto=END,
            )
        raise_for_truncated_completion(response)
        response = normalize_qwen_tool_calls(response)
        response = self._cap_response_tool_calls(
            response, operation="update", iterations=state.iteration_count
        )
        messages.append(response)
        if not response.tool_calls:
            return Command(
                update={"messages": messages, "response": response.content},
                goto=END
            )
        
        return Command(
            update={"messages": messages},
            goto="exec_tool"
        )
    
    def _build_graph(
        self
    ) -> CompiledStateGraph[MemoryManagerState, None, MemoryManagerState, MemoryManagerState]:
        """Build LangGraph workflow for MemoryManager"""
        workflow = StateGraph(MemoryManagerState)
        
        workflow.add_node("route", self._route_operation)
        workflow.add_node("fill_query_sys_prompt", self._fill_query_sys_prompt)
        workflow.add_node("handle_query", self._handle_query)
        workflow.add_node("exec_tool", self._exec_tool)
        workflow.add_node("fill_update_sys_prompt", self._fill_update_sys_prompt)
        workflow.add_node("handle_update", self._handle_update)
        
        
        workflow.set_entry_point("route")
        
        workflow.add_conditional_edges(
            "route",
            lambda s: s.operation,
            {
                "query": "fill_query_sys_prompt",
                "update": "fill_update_sys_prompt"
            }
        )
        
        workflow.add_edge("fill_query_sys_prompt", "handle_query")
        workflow.add_edge("fill_update_sys_prompt", "handle_update")
        
        return workflow.compile()
    
    def query(
        self, 
        context: list[RawMessage],
        query_text: Optional[str] = None, 
        query_image: Optional[str] = None,
    ) -> list[dict]:
        """Handle memory query from ChatAgent"""
        self.tool_cls.record_memory_manager_query(
            query_text=query_text,
            query_image=query_image,
        )
        state = MemoryManagerState(
            operation="query",
            query_text=query_text,
            query_image_path=query_image,
            context=context
        )
        result = self.graph.invoke(state,config={
                # "callbacks": [handler],
                "timeout": 20,
                # "callbacks": [langfuse_handler] if DEBUG else [],
                "configurable": {"thread_id": "1"}
            },)
        return result["response"]
    
    def update(
        self, 
        context: list[RawMessage],
        query_text: Optional[str] = None, 
        query_image: Optional[str] = None,
    ) -> str:
        """Handle memory update from ChatAgent"""
        state = MemoryManagerState(
            operation="update",
            query_text=query_text,
            query_image_path=query_image,
            context=context
        )
        result = self.graph.invoke(
            state,
            config={
                # "callbacks": [handler],
                # "callbacks": [langfuse_handler] if DEBUG else [],
                "configurable": {"thread_id": "1"}
            },)
        return result["response"]
    
    def _route_operation(self, state: MemoryManagerState) -> MemoryManagerState:
        """Route to query or update branch"""
        return state
