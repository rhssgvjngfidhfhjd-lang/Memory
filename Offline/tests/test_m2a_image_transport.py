from __future__ import annotations

import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image


M2A_ROOT = Path(__file__).resolve().parents[1] / "baselines" / "M2A"
if str(M2A_ROOT) not in sys.path:
    sys.path.insert(0, str(M2A_ROOT))

from agent.utils.message import (  # noqa: E402
    deduplicate_message_images,
    encode_image_to_base64,
    raise_for_truncated_completion,
)
from agent.utils.evidence import normalize_evidence_ranges  # noqa: E402
from agent.agents.chat_agent import ChatAgent, ChatAgentState  # noqa: E402
from agent.agents.memory_manager import (  # noqa: E402
    MemoryManager,
    MemoryManagerState,
    MemoryManagerTools,
)


def _image_blocks(messages: list[object]) -> list[dict]:
    return [
        block
        for message in messages
        if isinstance(getattr(message, "content", None), list)
        for block in message.content
        if isinstance(block, dict) and block.get("type") == "image"
    ]


class M2AImageTransportTest(unittest.TestCase):
    class _FakeBoundLLM:
        def __init__(self, parent):
            self.parent = parent

        def invoke(self, _messages):
            return self.parent.response

    class _FakeLLM:
        def __init__(self, response):
            self.response = response
            self.bind_kwargs = []
            self.invoke_messages = []

        def bind_tools(self, _tools, **kwargs):
            self.bind_kwargs.append(kwargs)
            return M2AImageTransportTest._FakeBoundLLM(self)

        def invoke(self, messages):
            self.invoke_messages.append(messages)
            return self.response

    def test_evidence_ranges_preserve_and_canonicalize_explicit_ids(self) -> None:
        self.assertEqual(
            normalize_evidence_ranges([[53, 54, 51]]),
            [[51, 51], [53, 54]],
        )
        self.assertEqual(normalize_evidence_ranges([53, 54, 51]), [[51, 51], [53, 54]])
        self.assertEqual(normalize_evidence_ranges([[48, 51]]), [[48, 51]])

    def test_evidence_ranges_reject_invalid_or_reversed_ids(self) -> None:
        for value in ([[0, 1]], [[5, 2]], [[1, "2"]], [[]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_evidence_ranges(value)

    def test_add_memory_normalizes_evidence_before_storage_and_audits_it(self) -> None:
        class FakeSemanticStore:
            def __init__(self):
                self.log = []
                self.added = []

            def add(self, memory):
                self.added.append(memory)
                return "7"

        class FakeImageManager:
            @staticmethod
            def image_token_to_image(_value):
                return None

        semantic = FakeSemanticStore()
        tools = MemoryManagerTools(
            raw_store=object(),
            semantic_store=semantic,
            image_manager=FakeImageManager(),
        )
        result = tools.get_add_memory().invoke(
            {
                "text": "memory",
                "image": None,
                "image_caption": None,
                "evidence_ids": "[[53, 54, 51]]",
            }
        )

        self.assertEqual(result, "Created memory, id: 7")
        self.assertEqual(semantic.added[0].evidence_ids, [[51, 51], [53, 54]])
        self.assertEqual(
            semantic.log,
            [
                {
                    "op": "normalize_evidence_ids",
                    "original": [[53, 54, 51]],
                    "normalized": [[51, 51], [53, 54]],
                }
            ],
        )

    def test_truncated_completion_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            raise_for_truncated_completion(
                SimpleNamespace(response_metadata={"finish_reason": "length"})
            )

        with self.assertRaisesRegex(RuntimeError, "truncated"):
            raise_for_truncated_completion(
                SimpleNamespace(
                    response_metadata={
                        "finish_reason": "tool_calls",
                        "native_finish_reason": "max_output_tokens",
                    }
                )
            )

        raise_for_truncated_completion(
            SimpleNamespace(response_metadata={"finish_reason": "stop"})
        )

    def test_transport_encoder_passes_through_data_and_http_urls(self) -> None:
        data_url = "data:image/jpeg;base64," + base64.b64encode(b"image").decode()
        self.assertEqual(encode_image_to_base64(data_url, compress=True), data_url)
        for url in ("http://example.test/image.jpg", "https://example.test/image.jpg"):
            self.assertEqual(encode_image_to_base64(url, compress=True), url)

    def test_transport_compression_preserves_source_and_limits_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            Image.new("RGB", (3000, 2000), (127, 63, 31)).save(path, format="PNG")
            original = path.read_bytes()

            data_url = encode_image_to_base64(str(path), compress=True)
            encoded = base64.b64decode(data_url.split(",", 1)[1])

            self.assertEqual(path.read_bytes(), original)
            self.assertLess(len(encoded), len(original))
            with Image.open(io.BytesIO(encoded)) as transported:
                self.assertLessEqual(max(transported.size), 1536)

    def test_request_deduplication_spans_multiple_messages(self) -> None:
        first = "data:image/jpeg;base64," + base64.b64encode(b"same").decode()
        second = "data:image/jpeg;base64," + base64.b64encode(b"different").decode()
        messages = [
            HumanMessage(
                content=[
                    {"type": "image", "url": first},
                    {"type": "text", "text": "context"},
                ]
            ),
            ToolMessage(
                tool_call_id="call-1",
                content=[
                    {"type": "image", "url": first},
                    {"type": "image", "url": second},
                    {"type": "image", "url": first},
                ],
            ),
        ]

        request_messages = deduplicate_message_images(messages)

        self.assertEqual(len(_image_blocks(request_messages)), 2)
        self.assertEqual(len(_image_blocks(messages)), 4)
        markers = [
            block
            for message in request_messages
            for block in message.content
            if isinstance(block, dict)
            and "Duplicate image bytes omitted" in block.get("text", "")
        ]
        self.assertEqual(len(markers), 2)

    def test_memory_manager_query_result_is_not_prepared_twice(self) -> None:
        data_url = "data:image/jpeg;base64," + base64.b64encode(b"image").decode()
        prepared = [{"type": "image", "url": data_url}]

        class FakeImageManager:
            @staticmethod
            def image_token_to_image(_value):
                return None

        class FakeMemoryManager:
            config = SimpleNamespace(context_window=5)

            @staticmethod
            def query(**_kwargs):
                return prepared

        agent = object.__new__(ChatAgent)
        agent.config = SimpleNamespace(max_update_iteration=3)
        agent.image_manager = FakeImageManager()
        agent.memory_manager = FakeMemoryManager()
        agent.raw_messages = []
        agent._prepair_message_content = lambda _content: self.fail(
            "MemoryManager result must not be prepared twice"
        )
        state = ChatAgentState(
            messages=[
                SimpleNamespace(
                    tool_calls=[
                        {
                            "name": "query_memory",
                            "id": "call-1",
                            "args": {"text": "query", "image": None},
                        }
                    ]
                )
            ]
        )

        result = agent._exec_update_state_tools(state)

        self.assertIs(result, state)
        self.assertEqual(state.messages[-1].content, prepared)

    def test_chat_agent_executes_last_legal_tool_call(self) -> None:
        calls = []

        class FakeImageManager:
            @staticmethod
            def image_token_to_image(_value):
                return None

        class FakeMemoryManager:
            config = SimpleNamespace(context_window=5)

            @staticmethod
            def query(**kwargs):
                calls.append(kwargs)
                return "memory result"

        agent = object.__new__(ChatAgent)
        agent.config = SimpleNamespace(max_update_iteration=1)
        agent.image_manager = FakeImageManager()
        agent.memory_manager = FakeMemoryManager()
        agent.raw_messages = []
        state = ChatAgentState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_memory",
                            "id": "call-last",
                            "args": {"text": "query"},
                        }
                    ],
                )
            ]
        )

        agent._exec_update_state_tools(state)

        self.assertEqual(state.update_iteration, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(state.messages[-1].content, "memory result")

    def test_chat_agent_forces_query_finalization_without_tools(self) -> None:
        llm = self._FakeLLM(
            AIMessage(content="final answer", response_metadata={"finish_reason": "stop"})
        )
        agent = object.__new__(ChatAgent)
        agent.llm = llm
        agent.tools = {"query": object()}
        agent.config = SimpleNamespace(max_query_iteration=2)
        agent._tool_budget_events = []
        state = ChatAgentState(
            messages=[HumanMessage(content="question")],
            query_iteration=2,
        )

        agent._generate_response(state)

        self.assertEqual(llm.bind_kwargs, [])
        self.assertEqual(len(llm.invoke_messages), 1)
        self.assertEqual(
            agent.pop_tool_budget_events(),
            [
                {
                    "operation": "query",
                    "tool_iterations": 2,
                    "max_tool_iterations": 2,
                    "budget_exhausted": True,
                    "forced_finalize": True,
                }
            ],
        )

    def test_chat_agent_rejects_empty_forced_query_response(self) -> None:
        llm = self._FakeLLM(
            AIMessage(content="", response_metadata={"finish_reason": "stop"})
        )
        agent = object.__new__(ChatAgent)
        agent.llm = llm
        agent.tools = {"query": object()}
        agent.config = SimpleNamespace(max_query_iteration=1)
        agent._tool_budget_events = []
        state = ChatAgentState(
            messages=[HumanMessage(content="question")],
            query_iteration=1,
        )

        with self.assertRaisesRegex(RuntimeError, "empty query response"):
            agent._generate_response(state)

        self.assertEqual(llm.bind_kwargs, [])
        self.assertEqual(len(llm.invoke_messages), 1)

    def test_chat_agent_forced_update_cannot_execute_an_extra_tool(self) -> None:
        llm = self._FakeLLM(
            AIMessage(
                content="<tool_call>{\"name\":\"update_memory\",\"arguments\":{}}</tool_call>",
                response_metadata={"finish_reason": "stop"},
            )
        )
        agent = object.__new__(ChatAgent)
        agent.llm = llm
        agent.tools = {"query": object(), "update": object()}
        agent.config = SimpleNamespace(max_update_iteration=3)
        agent._tool_budget_events = []
        state = ChatAgentState(
            messages=[HumanMessage(content="remember this")],
            update_iteration=3,
        )

        result = agent._update_stage(state)

        self.assertEqual(llm.bind_kwargs, [])
        self.assertEqual(len(llm.invoke_messages), 1)
        self.assertEqual(result.goto, "__end__")

    def test_chat_agent_salvages_truncated_update_text_through_memory_manager(self) -> None:
        updates = []
        llm = self._FakeLLM(
            AIMessage(
                content=(
                    '<tool_call>{"name":"update_memory","arguments":'
                    '{"text":"Evelyn admires springer spaniels'
                ),
                response_metadata={"finish_reason": "length"},
            )
        )

        class FakeMemoryManager:
            config = SimpleNamespace(context_window=5, salvage_truncated_updates=True)
            semantic_store = SimpleNamespace(log=[])

            @staticmethod
            def update(**kwargs):
                updates.append(kwargs)
                return "Created memory, id: 7"

        class FakeImageManager:
            @staticmethod
            def image_token_to_image(_value):
                return None

        agent = object.__new__(ChatAgent)
        agent.llm = llm
        agent.tools = {"query": object(), "update": object()}
        agent.config = SimpleNamespace(max_update_iteration=5)
        agent.memory_manager = FakeMemoryManager()
        agent.image_manager = FakeImageManager()
        agent.raw_messages = [SimpleNamespace(msg_id=41)]
        agent._tool_budget_events = []
        state = ChatAgentState(messages=[HumanMessage(content="remember this")])

        result = agent._update_stage(state)

        self.assertEqual(result.goto, "__end__")
        self.assertEqual(len(updates), 1)
        self.assertEqual(
            updates[0]["query_text"],
            "Evelyn admires springer spaniels [TRUNCATED]",
        )
        audit = agent.memory_manager.semantic_store.log[-1]
        self.assertTrue(audit["chat_agent_truncation_salvaged"])
        self.assertTrue(audit["partial_text_forwarded"])

    def test_chat_agent_salvage_is_opt_in(self) -> None:
        llm = self._FakeLLM(
            AIMessage(content="partial", response_metadata={"finish_reason": "length"})
        )
        agent = object.__new__(ChatAgent)
        agent.llm = llm
        agent.tools = {"query": object(), "update": object()}
        agent.config = SimpleNamespace(max_update_iteration=5)
        agent.memory_manager = SimpleNamespace(
            config=SimpleNamespace(context_window=5, salvage_truncated_updates=False)
        )
        agent._tool_budget_events = []

        with self.assertRaisesRegex(RuntimeError, "truncated"):
            agent._update_stage(
                ChatAgentState(messages=[HumanMessage(content="remember this")])
            )

    def test_memory_manager_forces_query_finalization_without_tools(self) -> None:
        llm = self._FakeLLM(
            AIMessage(content="memory answer", response_metadata={"finish_reason": "stop"})
        )

        class FakeImageManager:
            @staticmethod
            def format_msg_to_content(content):
                return content

        manager = object.__new__(MemoryManager)
        manager.llm = llm
        manager.image_manager = FakeImageManager()
        manager.tools = {
            "search_semantic_memories": object(),
            "fetch_raw_messages": object(),
            "fetch_raw_messages_by_time": object(),
        }
        manager._tool_budget_events = []
        state = MemoryManagerState(
            messages=[HumanMessage(content="memory query")],
            iteration_count=manager.max_iteration,
        )

        manager._handle_query(state)

        self.assertEqual(llm.bind_kwargs, [])
        self.assertEqual(len(llm.invoke_messages), 1)
        self.assertEqual(manager.pop_tool_budget_events()[0]["operation"], "query")

    def test_memory_manager_forced_update_cannot_execute_a_sixteenth_tool(self) -> None:
        llm = self._FakeLLM(
            AIMessage(
                content="<tool_call>{\"name\":\"add_memory\",\"arguments\":{}}</tool_call>",
                response_metadata={"finish_reason": "stop"},
            )
        )
        manager = object.__new__(MemoryManager)
        manager.llm = llm
        manager.tools = {
            "search_semantic_memories": object(),
            "fetch_raw_messages": object(),
            "fetch_raw_messages_by_time": object(),
            "add_memory": object(),
            "delete_memory": object(),
        }
        manager._tool_budget_events = []
        state = MemoryManagerState(
            messages=[HumanMessage(content="update memory")],
            operation="update",
            iteration_count=manager.max_iteration,
        )

        result = manager._handle_update(state)

        self.assertEqual(llm.bind_kwargs, [])
        self.assertEqual(len(llm.invoke_messages), 1)
        self.assertEqual(result.goto, "__end__")
        self.assertEqual(state.iteration_count, manager.max_iteration)

    def test_memory_manager_caps_boundary_batch_before_execution(self) -> None:
        llm = self._FakeLLM(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "add_memory", "id": "call-15", "args": {}},
                    {"name": "add_memory", "id": "call-16", "args": {}},
                ],
                response_metadata={"finish_reason": "stop"},
            )
        )
        manager = object.__new__(MemoryManager)
        manager.llm = llm
        manager.tools = {
            "search_semantic_memories": object(),
            "fetch_raw_messages": object(),
            "fetch_raw_messages_by_time": object(),
            "add_memory": object(),
            "delete_memory": object(),
        }
        manager._tool_budget_events = []
        state = MemoryManagerState(
            messages=[HumanMessage(content="update memory")],
            operation="update",
            iteration_count=manager.max_iteration - 1,
        )

        result = manager._handle_update(state)

        self.assertEqual(result.goto, "exec_tool")
        self.assertEqual(
            [call["id"] for call in state.messages[-1].tool_calls], ["call-15"]
        )
        event = manager.pop_tool_budget_events()[0]
        self.assertEqual(event["tool_calls_accepted"], 1)
        self.assertEqual(event["tool_calls_omitted"], 1)

    def test_memory_manager_executes_and_preserves_fifteenth_update(self) -> None:
        calls = []

        class FakeImageManager:
            @staticmethod
            def image_token_to_image(value):
                return value

        class FakeTool:
            @staticmethod
            def invoke(input):
                calls.append(input)
                return "Created memory, id: 15"

        manager = object.__new__(MemoryManager)
        manager.image_manager = FakeImageManager()
        manager.tools = {"add_memory": FakeTool()}
        state = MemoryManagerState(
            messages=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "add_memory",
                            "id": "call-15",
                            "args": {"text": "preserve update 15"},
                        }
                    ],
                )
            ],
            operation="update",
            iteration_count=manager.max_iteration - 1,
        )

        result = manager._exec_tool(state)

        self.assertEqual(calls, [{"text": "preserve update 15"}])
        self.assertEqual(state.iteration_count, manager.max_iteration)
        self.assertEqual(result.goto, "handle_update")
        self.assertEqual(state.messages[-1].content, "Created memory, id: 15")

    def test_memory_manager_salvages_complete_and_partial_truncated_updates(self) -> None:
        calls = []

        class FakeTool:
            def __init__(self, name):
                self.name = name

            def invoke(self, args):
                calls.append((self.name, args))
                if self.name == "add_memory":
                    return "Created memory, id: 99"
                return f"Deleted memory {args['memory_id']}"

        content = """I will update memory.
```json
{"operation":"DELETE","memory_ids":[17,18]}
```
```json
{"operation":"CREATE","memory":{"text":"complete memory","evidence_ids":[[7,9]]}}
```
```json
{"operation":"CREATE","memory":{"text":"unfinished memory prefix
"""
        manager = object.__new__(MemoryManager)
        manager.llm = self._FakeLLM(
            AIMessage(
                content=content,
                response_metadata={"finish_reason": "length"},
            )
        )
        manager.config = SimpleNamespace(salvage_truncated_updates=True)
        manager.semantic_store = SimpleNamespace(log=[])
        manager.tools = {
            "search_semantic_memories": object(),
            "fetch_raw_messages": object(),
            "fetch_raw_messages_by_time": object(),
            "add_memory": FakeTool("add_memory"),
            "delete_memory": FakeTool("delete_memory"),
        }
        state = MemoryManagerState(
            messages=[HumanMessage(content="update memory")],
            operation="update",
            context=[SimpleNamespace(msg_id=41), SimpleNamespace(msg_id=42)],
        )

        result = manager._handle_update(state)

        self.assertEqual(result.goto, "__end__")
        self.assertEqual(
            calls[:2],
            [
                ("delete_memory", {"memory_id": "17"}),
                ("delete_memory", {"memory_id": "18"}),
            ],
        )
        self.assertEqual(calls[2][0], "add_memory")
        self.assertEqual(calls[2][1]["text"], "complete memory")
        self.assertEqual(calls[2][1]["evidence_ids"], "[[7, 9]]")
        self.assertEqual(calls[3][0], "add_memory")
        self.assertEqual(
            calls[3][1]["text"], "unfinished memory prefix [TRUNCATED]"
        )
        self.assertEqual(calls[3][1]["evidence_ids"], "[[41, 42]]")
        audit = manager.semantic_store.log[-1]
        self.assertEqual(audit["op"], "salvage_truncated_update")
        self.assertEqual(audit["creates_executed"], 2)
        self.assertEqual(audit["deletes_executed"], 2)
        self.assertTrue(audit["partial_text_saved"])
        self.assertFalse(audit["raw_fallback_saved"])

    def test_memory_manager_saves_raw_truncation_when_no_text_is_recoverable(self) -> None:
        calls = []

        class FakeAddTool:
            @staticmethod
            def invoke(args):
                calls.append(args)
                return "Created memory, id: 100"

        manager = object.__new__(MemoryManager)
        manager.llm = self._FakeLLM(
            AIMessage(
                content='{"operation":"DELETE","memory_ids":[17,',
                response_metadata={"finish_reason": "length"},
            )
        )
        manager.config = SimpleNamespace(salvage_truncated_updates=True)
        manager.semantic_store = SimpleNamespace(log=[])
        manager.tools = {
            "search_semantic_memories": object(),
            "fetch_raw_messages": object(),
            "fetch_raw_messages_by_time": object(),
            "add_memory": FakeAddTool(),
            "delete_memory": object(),
        }
        state = MemoryManagerState(
            messages=[HumanMessage(content="update memory")],
            operation="update",
            context=[SimpleNamespace(msg_id=51), SimpleNamespace(msg_id=53)],
        )

        result = manager._handle_update(state)

        self.assertEqual(result.goto, "__end__")
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["text"].startswith("[TRUNCATED MODEL OUTPUT]"))
        self.assertEqual(calls[0]["evidence_ids"], "[[51, 51], [53, 53]]")
        self.assertTrue(manager.semantic_store.log[-1]["raw_fallback_saved"])

    def test_memory_manager_still_rejects_truncation_when_salvage_is_disabled(self) -> None:
        manager = object.__new__(MemoryManager)
        manager.llm = self._FakeLLM(
            AIMessage(
                content='{"operation":"CREATE","memory":{"text":"partial',
                response_metadata={"finish_reason": "length"},
            )
        )
        manager.config = SimpleNamespace(salvage_truncated_updates=False)
        manager.tools = {
            "search_semantic_memories": object(),
            "fetch_raw_messages": object(),
            "fetch_raw_messages_by_time": object(),
            "add_memory": object(),
            "delete_memory": object(),
        }
        state = MemoryManagerState(
            messages=[HumanMessage(content="update memory")],
            operation="update",
        )

        with self.assertRaisesRegex(RuntimeError, "truncated"):
            manager._handle_update(state)


if __name__ == "__main__":
    unittest.main()
