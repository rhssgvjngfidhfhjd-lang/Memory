from __future__ import annotations

import json
import asyncio
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    NativeAnswerRequest,
    NativeAnswerResult,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
    result_context_items,
)
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.call_trace import CallRecorder
from benchmarks.baseline_runtime.parallel_runner import (
    load_sample_artifact,
    parallel_map_ordered,
)
from benchmarks.baseline_runtime.registry import (
    BASELINE_NAMES,
    baseline_metadata,
    canonical_name,
)
from benchmarks.baseline_runtime.adapters import mirix_family as mirix_family_module
from benchmarks.baseline_runtime.adapters.mirix_family import (
    MirixFamilyAdapter,
    _MirixRetrievalBudget,
    _accept_recovered_native_tool_finish,
    _accept_truncated_native_qa_text,
    _bound_native_vllm_tool_request,
    _embedding_similarity,
    _ensure_answer_block,
    _expand_mirix_delta_update,
    _merge_mirix_delta_update,
    _mirix_model_handle,
    _normalize_openai_tool_request,
    _normalize_openai_tool_response,
    _normalize_openai_tool_tags,
    _prepare_mirix_delta_tool_request,
    _python_style_tool_payload,
    _promote_native_chat_text_response,
    _reject_native_tool_response_integrity,
    _reject_unparsed_native_tool_response,
    _round_robin_native_rows,
    _send_native_benchmark_messages,
    _stage_native_transport_image,
)
from benchmarks.baseline_runtime.adapters.omni_simplemem import OmniSimpleMemAdapter
from benchmarks.baseline_runtime.adapters.memverse import (
    MemVerseAdapter,
    _CappedOpenAIClientProxy,
    _apply_executor_max_tokens,
    _bounded_history_messages,
    _bounded_prompt,
    _repair_corrupt_lightrag_caches,
)
from benchmarks.baseline_runtime.adapters.m2a import M2AAdapter
from benchmarks.baseline_runtime.provenance import ProvenanceIndex
from benchmarks.memgallery_harness.runner.answer_client import AnswerResponse
from benchmarks.memgallery_harness.eval_memgallery import prepare_dataset_jobs, run_dataset
from benchmarks.h2hmem_harness.eval_h2hmem import (
    prepare_conversation_jobs as prepare_h2h_conversation_jobs,
)
from benchmarks.wma_harness.eval_wma import (
    prepare_native_sample_jobs,
    run_sample_retry_queue,
)
from embedding.chunk_builder import Chunk, compact_text


class FakeBaseline(BaselineAdapter):
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.ended_sessions: list[str] = []
        self.closed = False

    def reset(self, sample_id: str, state_dir: Path) -> None:
        self.sample_id = sample_id
        self.state_dir = state_dir
        self.chunks = []

    def ingest(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    def end_session(self, session_id: str) -> None:
        self.ended_sessions.append(session_id)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        visible = set(request.visible_session_ids)
        chunks = [
            chunk
            for chunk in self.chunks
            if not visible or str(chunk.metadata.get("session_id") or "") in visible
        ]
        return RetrievalResult(
            items=[
                RetrievedMemory(
                    memory_id=f"fake:{chunk.chunk_id}",
                    text=chunk.text,
                    score=1.0,
                    session_id=str(chunk.metadata.get("session_id") or ""),
                    source_dialogue_ids=[
                        str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
                    ],
                    image_ids=list(chunk.metadata.get("image_ids") or []),
                    image_paths=list(chunk.images),
                )
                for chunk in chunks[-request.top_k :]
            ],
            trace={"via": "fake"},
        )

    def snapshot(self) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                memory_id=f"fake:{chunk.chunk_id}",
                text=chunk.text,
                session_id=str(chunk.metadata.get("session_id") or ""),
                source_dialogue_ids=[
                    str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
                ],
                backend_type="fake",
            )
            for chunk in self.chunks
        ]

    def close(self) -> None:
        self.closed = True


class FakeAnswerClient:
    retries = 0

    def answer_messages_with_usage(self, **kwargs):
        return AnswerResponse(
            text="<answer>answer from memory</answer>",
            usage=None,
            attempts=1,
            failed_attempts=0,
        )

    def answer_with_usage(self, **kwargs):
        return AnswerResponse(
            text="answer from memory",
            usage=None,
            attempts=1,
            failed_attempts=0,
        )


class BaselineProtocolTest(unittest.TestCase):
    def test_mirix_wma_checkpoint_backs_up_sqlite_and_provenance(self):
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.config = {
            "mirix_resume_enabled": True,
            "mirix_resume_signature": "sig",
        }
        adapter._sample_id = "sample"
        adapter._seen_session_ids = ["S1", "S2"]
        adapter._completed_session_ids = []
        adapter._ingested_chunks = 4
        adapter._known_ids = {"semantic:1"}
        adapter.provenance = ProvenanceIndex()
        adapter.provenance.restore_rows(
            {
                "semantic:1": {
                    "session_id": "S1",
                    "session_ids": ["S1"],
                    "source_dialogue_ids": ["D1"],
                    "image_ids": [],
                    "image_paths": [],
                }
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            adapter._state_dir = state
            database = state / "sqlite.db"
            connection = sqlite3.connect(database)
            connection.execute("CREATE TABLE memory (value TEXT)")
            connection.execute("INSERT INTO memory VALUES ('checkpointed')")
            connection.commit()
            connection.close()

            adapter._checkpoint_completed_sessions()
            payload = adapter._load_resume_checkpoint("sample", state)

            self.assertIsNotNone(payload)
            self.assertEqual(payload["completed_session_ids"], ["S1", "S2"])
            self.assertEqual(payload["provenance"]["semantic:1"]["session_id"], "S1")
            connection = sqlite3.connect(database)
            connection.execute("DELETE FROM memory")
            connection.commit()
            connection.close()
            adapter._restore_resume_database(state, payload)
            connection = sqlite3.connect(database)
            value = connection.execute("SELECT value FROM memory").fetchone()[0]
            connection.close()
            self.assertEqual(value, "checkpointed")

    def test_mirix_uses_vllm_handle_only_for_local_executor(self):
        self.assertEqual(
            _mirix_model_handle(
                baseline="MIRIX",
                model="Qwen/Qwen3-VL-4B-Instruct",
                endpoint="http://127.0.0.1:18000/v1",
            ),
            "vllm/Qwen/Qwen3-VL-4B-Instruct",
        )
        self.assertIsNone(
            _mirix_model_handle(
                baseline="MIRIX",
                model="openai/gpt-5-mini",
                endpoint="https://openrouter.ai/api/v1",
            )
        )

    def test_m2a_wma_rejects_missing_or_future_session_provenance(self):
        valid = RetrievedMemory(
            memory_id="m2a:1",
            text="visible",
            metadata={"session_ids": ["S00"]},
        )
        M2AAdapter._validate_visible_session_scope([valid], ("S00",))

        missing = RetrievedMemory(memory_id="m2a:2", text="unscoped")
        with self.assertRaisesRegex(RuntimeError, "lacks session provenance"):
            M2AAdapter._validate_visible_session_scope([missing], ("S00",))

        future = RetrievedMemory(
            memory_id="m2a:3",
            text="future",
            metadata={"session_ids": ["S00", "S01"]},
        )
        with self.assertRaisesRegex(RuntimeError, "non-visible session"):
            M2AAdapter._validate_visible_session_scope([future], ("S00",))

    def test_mirix_native_vllm_stops_after_one_complete_tool_envelope(self):
        request = {
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "tool_choice": "auto",
            "stop": ["existing-stop"],
        }

        bounded = _bound_native_vllm_tool_request(request)

        self.assertEqual(bounded["stop"], ["existing-stop", "</tool_call>"])
        self.assertTrue(bounded["extra_body"]["include_stop_str_in_output"])
        self.assertEqual(bounded["extra_body"]["repetition_penalty"], 1.10)

    def test_mirix_delta_request_replaces_only_update_payload_fields(self):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "resource_memory_update",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "old_ids": {"type": "array"},
                                "new_items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "title": {"type": "string"},
                                            "content": {"type": "string"},
                                            "tree_path": {"type": "array"},
                                        },
                                        "required": ["title", "content", "tree_path"],
                                    },
                                },
                            },
                        },
                    },
                }
            ]
        }

        bounded = _prepare_mirix_delta_tool_request(request)
        item = bounded["tools"][0]["function"]["parameters"]["properties"][
            "new_items"
        ]["items"]

        self.assertIn("content_delta", item["properties"])
        self.assertNotIn("content", item["properties"])
        self.assertEqual(
            item["required"], ["title", "content_delta", "tree_path"]
        )
        self.assertIn("content", request["tools"][0]["function"]["parameters"]["properties"]["new_items"]["items"]["properties"])

    def test_mirix_episodic_merge_request_uses_details_delta(self):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "episodic_memory_merge",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "event_id": {"type": "string"},
                                "combined_summary": {"type": "string"},
                                "combined_details": {"type": "string"},
                            },
                            "required": [
                                "event_id",
                                "combined_summary",
                                "combined_details",
                            ],
                        },
                    },
                }
            ]
        }

        bounded = _prepare_mirix_delta_tool_request(request)
        parameters = bounded["tools"][0]["function"]["parameters"]

        self.assertIn("details_delta", parameters["properties"])
        self.assertNotIn("combined_details", parameters["properties"])
        self.assertEqual(
            parameters["required"],
            ["event_id", "combined_summary", "details_delta"],
        )
        self.assertIn(
            "combined_details",
            request["tools"][0]["function"]["parameters"]["properties"],
        )

    def test_mirix_episodic_insert_request_is_bounded_to_one_item(self):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "episodic_memory_insert",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "inner_thoughts": {"type": "string"},
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "summary": {"type": "string"},
                                            "details": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                }
            ]
        }

        bounded = _prepare_mirix_delta_tool_request(request)
        properties = bounded["tools"][0]["function"]["parameters"]["properties"]

        self.assertEqual(properties["items"]["maxItems"], 1)
        self.assertEqual(
            properties["items"]["items"]["properties"]["details"]["maxLength"],
            900,
        )
        self.assertEqual(
            properties["items"]["items"]["properties"]["summary"]["maxLength"],
            300,
        )
        self.assertEqual(properties["inner_thoughts"]["maxLength"], 300)
        self.assertNotIn(
            "maxItems",
            request["tools"][0]["function"]["parameters"]["properties"]["items"],
        )

    def test_mirix_semantic_insert_request_is_bounded_to_three_items(self):
        request = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "semantic_memory_insert",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "inner_thoughts": {"type": "string"},
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "summary": {"type": "string"},
                                            "details": {"type": "string"},
                                            "source": {"type": "string"},
                                            "tree_path": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                }
            ]
        }

        bounded = _prepare_mirix_delta_tool_request(request)
        properties = bounded["tools"][0]["function"]["parameters"]["properties"]
        item = properties["items"]["items"]["properties"]

        self.assertEqual(properties["items"]["maxItems"], 3)
        self.assertEqual(item["name"]["maxLength"], 120)
        self.assertEqual(item["summary"]["maxLength"], 240)
        self.assertEqual(item["details"]["maxLength"], 500)
        self.assertEqual(item["source"]["maxLength"], 160)
        self.assertEqual(item["tree_path"]["maxItems"], 4)
        self.assertEqual(item["tree_path"]["items"]["maxLength"], 64)
        self.assertEqual(properties["inner_thoughts"]["maxLength"], 200)
        self.assertNotIn(
            "maxItems",
            request["tools"][0]["function"]["parameters"]["properties"]["items"],
        )

    def test_mirix_integrity_gate_rejects_provider_cap_and_whitespace_loop(self):
        capped = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "native_finish_reason": "max_output_tokens",
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "update", "arguments": "{}"}}
                        ]
                    },
                }
            ]
        }
        whitespace_loop = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "native_finish_reason": "completed",
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "update",
                                    "arguments": "{}" + ("\n" * 128),
                                }
                            }
                        ]
                    },
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "truncated"):
            _reject_native_tool_response_integrity(capped)
        with self.assertRaisesRegex(ValueError, "whitespace-degenerate"):
            _reject_native_tool_response_integrity(whitespace_loop)

    def test_mirix_resource_delta_is_merged_and_schema_validated(self):
        class FakeSchema:
            def __init__(self, **values):
                self.values = values

            def model_dump(self):
                return dict(self.values)

        fake_module = SimpleNamespace(ResourceMemoryItemBase=FakeSchema)
        arguments = {
            "old_ids": ["res_1"],
            "new_items": [
                {
                    "title": "Updated notes",
                    "summary": "Updated compact summary",
                    "resource_type": "markdown",
                    "content_delta": "New fact C.",
                    "tree_path": ["research", "notes"],
                }
            ],
        }
        old_items = [
            {
                "title": "Old notes",
                "summary": "Old summary",
                "resource_type": "markdown",
                "content": "Old facts A and B.",
                "tree_path": ["research"],
            }
        ]

        with patch(
            "benchmarks.baseline_runtime.adapters.mirix_family.importlib.import_module",
            return_value=fake_module,
        ):
            merged = _merge_mirix_delta_update(
                "resource_memory_update", arguments, old_items
            )

        item = merged["new_items"][0]
        self.assertEqual(item["content"], "Old facts A and B.\n\nNew fact C.")
        self.assertNotIn("content_delta", item)
        self.assertEqual(item["summary"], "Updated compact summary")

    def test_mirix_procedural_delta_deduplicates_old_steps(self):
        class FakeSchema:
            def __init__(self, **values):
                self.values = values

            def model_dump(self):
                return dict(self.values)

        fake_module = SimpleNamespace(ProceduralMemoryItemBase=FakeSchema)
        arguments = {
            "old_ids": ["proc_1"],
            "new_items": [
                {
                    "entry_type": "workflow",
                    "summary": "Updated workflow",
                    "steps_delta": ["Check labels", "Record outcome"],
                    "tree_path": ["health", "nutrition"],
                }
            ],
        }
        old_items = [
            {
                "entry_type": "workflow",
                "summary": "Old workflow",
                "steps": ["Check labels"],
                "tree_path": ["health"],
            }
        ]

        with patch(
            "benchmarks.baseline_runtime.adapters.mirix_family.importlib.import_module",
            return_value=fake_module,
        ):
            merged = _merge_mirix_delta_update(
                "procedural_memory_update", arguments, old_items
            )

        self.assertEqual(
            merged["new_items"][0]["steps"], ["Check labels", "Record outcome"]
        )
        self.assertNotIn("steps_delta", merged["new_items"][0])

    def test_mirix_episodic_details_delta_is_bounded_and_validated(self):
        class FakeSchema:
            def __init__(self, **values):
                self.values = values

        old_event = SimpleNamespace(
            model_dump=lambda: {
                "summary": "Old event summary",
                "details": "Previously stored details.",
            }
        )
        manager = SimpleNamespace(
            get_episodic_memory_by_id=lambda event_id: old_event
        )
        agent = SimpleNamespace(episodic_memory_manager=manager)
        fake_module = SimpleNamespace(EpisodicEventUpdate=FakeSchema)

        with patch(
            "benchmarks.baseline_runtime.adapters.mirix_family.importlib.import_module",
            return_value=fake_module,
        ):
            merged = _expand_mirix_delta_update(
                agent,
                "episodic_memory_merge",
                {
                    "event_id": "ep_1",
                    "combined_summary": "Updated event summary",
                    "details_delta": "New event details. " * 100,
                },
            )

        self.assertNotIn("details_delta", merged)
        self.assertLessEqual(len(merged["combined_details"]), 900)
        self.assertEqual(merged["combined_summary"], "Updated event summary")

    def test_mirix_episodic_insert_is_bounded_and_schema_validated(self):
        class FakeSchema:
            def __init__(self, **values):
                self.values = values

            def model_dump(self):
                return dict(self.values)

        fake_module = SimpleNamespace(EpisodicEventForLLM=FakeSchema)
        with patch(
            "benchmarks.baseline_runtime.adapters.mirix_family.importlib.import_module",
            return_value=fake_module,
        ):
            bounded = _expand_mirix_delta_update(
                object(),
                "episodic_memory_insert",
                {
                    "inner_thoughts": "keep",
                    "items": [
                        {
                            "summary": "Long summary. " * 100,
                            "details": "Long details. " * 200,
                        }
                    ],
                },
            )

        self.assertEqual(len(bounded["items"]), 1)
        self.assertLessEqual(len(bounded["items"][0]["summary"]), 300)
        self.assertLessEqual(len(bounded["items"][0]["details"]), 900)
        self.assertEqual(bounded["inner_thoughts"], "keep")

        with self.assertRaisesRegex(ValueError, "exactly one"):
            _expand_mirix_delta_update(
                object(), "episodic_memory_insert", {"items": [{}, {}]}
            )

    def test_mirix_semantic_insert_is_bounded_and_schema_validated(self):
        class FakeSchema:
            def __init__(self, **values):
                self.values = values

            def model_dump(self):
                return dict(self.values)

        fake_module = SimpleNamespace(SemanticMemoryItemBase=FakeSchema)
        items = [
            {
                "name": "name " * 100,
                "summary": "summary " * 100,
                "details": "details " * 200,
                "source": "source " * 100,
                "tree_path": ["path " * 50] * 8,
            }
            for _ in range(4)
        ]
        with patch(
            "benchmarks.baseline_runtime.adapters.mirix_family.importlib.import_module",
            return_value=fake_module,
        ):
            bounded = _expand_mirix_delta_update(
                object(),
                "semantic_memory_insert",
                {"inner_thoughts": "thought " * 100, "items": items},
            )

        self.assertEqual(len(bounded["items"]), 3)
        self.assertLessEqual(len(bounded["inner_thoughts"]), 200)
        for item in bounded["items"]:
            self.assertLessEqual(len(item["name"]), 120)
            self.assertLessEqual(len(item["summary"]), 240)
            self.assertLessEqual(len(item["details"]), 500)
            self.assertLessEqual(len(item["source"]), 160)
            self.assertLessEqual(len(item["tree_path"]), 4)
            self.assertTrue(all(len(value) <= 64 for value in item["tree_path"]))

    def test_mirix_native_vllm_leaves_non_auto_requests_unchanged(self):
        request = {
            "tools": [{"type": "function", "function": {"name": "search"}}],
            "tool_choice": "required",
        }

        self.assertEqual(_bound_native_vllm_tool_request(request), request)

    def test_mirix_native_memory_update_hides_redundant_read_tools(self):
        request = {
            "tools": [
                {"type": "function", "function": {"name": "search_in_memory"}},
                {
                    "type": "function",
                    "function": {"name": "list_memory_within_timerange"},
                },
                {
                    "type": "function",
                    "function": {"name": "trigger_memory_update"},
                },
                {
                    "type": "function",
                    "function": {"name": "finish_memory_update"},
                },
            ],
            "tool_choice": "auto",
        }

        bounded = _bound_native_vllm_tool_request(request)
        names = [tool["function"]["name"] for tool in bounded["tools"]]

        self.assertEqual(names, ["trigger_memory_update", "finish_memory_update"])

    def test_mirix_native_chat_keeps_autonomous_read_tools(self):
        request = {
            "tools": [
                {"type": "function", "function": {"name": "search_in_memory"}},
                {
                    "type": "function",
                    "function": {"name": "list_memory_within_timerange"},
                },
                {
                    "type": "function",
                    "function": {"name": "trigger_memory_update_with_instruction"},
                },
                {"type": "function", "function": {"name": "send_message"}},
            ],
            "tool_choice": "auto",
        }

        bounded = _bound_native_vllm_tool_request(request)
        names = [tool["function"]["name"] for tool in bounded["tools"]]

        self.assertIn("search_in_memory", names)
        self.assertIn("list_memory_within_timerange", names)

    def test_mirix_retries_unparsed_native_tool_envelope(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": '<tool_call>{"name":"broken"}</tool_call>',
                        "tool_calls": [],
                    }
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "unparsed native"):
            _reject_unparsed_native_tool_response(response)

    def test_mirix_accepts_server_parsed_native_tool_envelope(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "search_in_memory",
                                    "arguments": "{}",
                                }
                            }
                        ],
                    }
                }
            ]
        }

        _reject_unparsed_native_tool_response(response)

    def test_mirix_retries_server_parsed_malformed_arguments(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "semantic_memory_insert",
                                    "arguments": '{"items": []}{"extra": true}',
                                }
                            }
                        ],
                    }
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "malformed native"):
            _reject_unparsed_native_tool_response(response)

    def test_mirix_accepts_length_finish_only_with_valid_tool_call(self):
        valid = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "episodic_memory_insert",
                                    "arguments": '{"items": []}',
                                }
                            }
                        ]
                    },
                }
            ]
        }
        empty = {
            "choices": [
                {"finish_reason": "length", "message": {"tool_calls": []}}
            ]
        }

        _accept_recovered_native_tool_finish(valid)
        _accept_recovered_native_tool_finish(empty)

        self.assertEqual(valid["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(empty["choices"][0]["finish_reason"], "length")

    def test_mirix_promotes_plain_native_chat_answer_to_send_message(self):
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "The answer is reef A.<|im_end|>",
                        "tool_calls": [],
                    },
                }
            ]
        }
        token = mirix_family_module._ACTIVE_MIRIX_RETRIEVAL.set(object())
        try:
            _promote_native_chat_text_response(response)
        finally:
            mirix_family_module._ACTIVE_MIRIX_RETRIEVAL.reset(token)

        choice = response["choices"][0]
        function = choice["message"]["tool_calls"][0]["function"]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(function["name"], "send_message")
        self.assertEqual(json.loads(function["arguments"])["message"], "The answer is reef A.")

    def test_mirix_accepts_truncated_plain_qa_text_for_scoring(self):
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "native_finish_reason": "max_output_tokens",
                    "message": {"content": "<answer>partial final answer", "tool_calls": []},
                }
            ]
        }
        state = {"accepted": False, "finish_reason": "", "native_finish_reason": ""}
        token = mirix_family_module._ACTIVE_MIRIX_QA_TRUNCATION.set(state)
        retrieval_token = mirix_family_module._ACTIVE_MIRIX_RETRIEVAL.set(object())
        try:
            self.assertTrue(_accept_truncated_native_qa_text(response))
            _reject_native_tool_response_integrity(response)
            _promote_native_chat_text_response(response)
        finally:
            mirix_family_module._ACTIVE_MIRIX_RETRIEVAL.reset(retrieval_token)
            mirix_family_module._ACTIVE_MIRIX_QA_TRUNCATION.reset(token)

        self.assertTrue(state["accepted"])
        self.assertEqual(state["native_finish_reason"], "max_output_tokens")
        arguments = response["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"]
        partial = json.loads(arguments)["message"]
        self.assertEqual(_ensure_answer_block(partial), "<answer>partial final answer</answer>")

    def test_mirix_does_not_accept_truncated_qa_tool_json(self):
        response = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "native_finish_reason": "max_output_tokens",
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "send_message",
                                    "arguments": '{"message":"unfinished',
                                }
                            }
                        ],
                    },
                }
            ]
        }
        state = {"accepted": False, "finish_reason": "", "native_finish_reason": ""}
        token = mirix_family_module._ACTIVE_MIRIX_QA_TRUNCATION.set(state)
        try:
            self.assertFalse(_accept_truncated_native_qa_text(response))
            with self.assertRaisesRegex(ValueError, "truncated"):
                _reject_native_tool_response_integrity(response)
        finally:
            mirix_family_module._ACTIVE_MIRIX_QA_TRUNCATION.reset(token)
        self.assertFalse(state["accepted"])

    def test_mirix_does_not_accept_truncated_plain_text_outside_qa(self):
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "partial build output", "tool_calls": []},
                }
            ]
        }

        self.assertFalse(_accept_truncated_native_qa_text(response))
        with self.assertRaisesRegex(ValueError, "truncated"):
            _reject_native_tool_response_integrity(response)

    def test_mirix_model_config_identifies_native_vllm_provider(self):
        class FakeConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        fake_package = ModuleType("fake_mirix")
        fake_package.LLMConfig = FakeConfig
        fake_package.EmbeddingConfig = FakeConfig
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.package = "fake_mirix"
        adapter.config = {
            "executor_model": "Qwen/Qwen3-VL-4B-Instruct",
            "executor_base_url": "http://127.0.0.1:8014/v1",
            "executor_temperature": 0.0,
            "executor_max_tokens": 16384,
            "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
            "embedding_base_url": "http://127.0.0.1:8001/v1",
            "embedding_dim": 2048,
        }

        with patch.dict(sys.modules, {"fake_mirix": fake_package}):
            llm, _embedding = adapter._model_configs()

        self.assertEqual(llm.handle, "vllm/Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(llm.model_endpoint_type, "openai")

    def test_mirix_global_similarity_uses_cosine_score(self):
        self.assertAlmostEqual(_embedding_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(_embedding_similarity([1, 0], [0, 1]), 0.0)

    def test_mirix_native_chat_shares_seven_item_budget_without_freezing(self):
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.provenance = ProvenanceIndex()
        budget = _MirixRetrievalBudget(adapter, "q", 7)
        first = [
            {"memory_type": "semantic", "id": f"m{i}", "summary": str(i)}
            for i in range(5)
        ]
        second = [
            {"memory_type": "semantic", "id": f"m{i}", "summary": str(i)}
            for i in range(4, 12)
        ]
        first_return = budget.limit_tool_result("search_in_memory", (first, 5))
        second_return = budget.limit_tool_result("search_in_memory", (second, 8))
        self.assertEqual(first_return[1], 5)
        self.assertEqual(second_return[1], 2)
        self.assertEqual(len(budget.result(stage="answer_complete").items), 7)
        self.assertEqual(budget.result(stage="answer_complete").trace["remaining_budget"], 0)

    def test_mirix_automatic_prefetch_reserves_shared_top_seven_budget(self):
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.provenance = ProvenanceIndex()
        budget = _MirixRetrievalBudget(adapter, "q", 7)
        selected = [
            ("episodic", {"id": "e1", "summary": "episode"}),
            ("semantic", {"id": "s1", "summary": "fact"}),
        ]

        budget.install_prefetch(
            {"key_words": "dogs"},
            selected,
            {"selection": "native_per_bank_rank_round_robin"},
        )
        explicit = [
            {"memory_type": "episodic", "id": "e1", "summary": "duplicate"},
            *[
                {"memory_type": "semantic", "id": f"s{i}", "summary": str(i)}
                for i in range(2, 10)
            ],
        ]
        returned = budget.limit_tool_result("search_in_memory", (explicit, len(explicit)))
        result = budget.result(stage="answer_complete")

        self.assertEqual(returned[1], 5)
        self.assertEqual(len(result.items), 7)
        self.assertEqual(result.trace["remaining_budget"], 0)
        self.assertTrue(result.trace["automatic_prefetch_counts_toward_top_k"])
        self.assertEqual(
            result.trace["automatic_system_prefetch"][0]["selected_count"], 2
        )
        self.assertEqual(
            result.items[0].metadata["via"],
            "native_chat_agent_automatic_prefetch",
        )

    def test_mirix_prefetch_evidence_excludes_native_embedding_columns(self):
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.provenance = ProvenanceIndex()
        budget = _MirixRetrievalBudget(adapter, "q", 7)

        budget.install_prefetch(
            {"key_words": "Milo"},
            [
                (
                    "episodic",
                    {
                        "id": "e1",
                        "summary": "Met Milo",
                        "details": "Milo is a Cocker Spaniel.",
                        "details_embedding": [0.125] * 2048,
                    },
                )
            ],
            {"selection": "native_per_bank_rank_round_robin"},
        )

        evidence = budget.result(stage="retrieval").items[0].text
        self.assertIn("Milo is a Cocker Spaniel.", evidence)
        self.assertNotIn("embedding", evidence)
        self.assertLess(len(evidence), 200)

    def test_mirix_prefetch_interleaves_native_bank_rankings(self):
        groups = [
            ("episodic", [{"id": "e1"}, {"id": "e2"}, {"id": "e3"}]),
            ("semantic", [{"id": "s1"}, {"id": "s2"}]),
            ("resource", [{"id": "r1"}]),
        ]

        selected = _round_robin_native_rows(groups, 5)

        self.assertEqual(
            [(kind, row["id"]) for kind, row in selected],
            [
                ("episodic", "e1"),
                ("semantic", "s1"),
                ("resource", "r1"),
                ("episodic", "e2"),
                ("semantic", "s2"),
            ],
        )

    def test_mirix_retrieval_probe_disables_chaining(self):
        client = SimpleNamespace(send_message=Mock(return_value=SimpleNamespace(messages=[])))
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.backend = SimpleNamespace(
            client=client,
            agent_states=SimpleNamespace(agent_state=SimpleNamespace(id="chat")),
        )
        adapter._qa_budgets = {}
        with (
            patch.object(
                mirix_family_module,
                "_capture_chat_state",
                return_value={"message_ids": [], "topic": None},
            ),
            patch.object(mirix_family_module, "_restore_chat_state"),
        ):
            adapter._retrieve_with_native_chat(
                RetrievalRequest(query_id="q", text="question", top_k=7)
            )

        self.assertFalse(client.send_message.call_args.kwargs["chaining"])

    def test_mirix_single_send_message_clears_native_failure_and_stops(self):
        class FakeAgent:
            def execute_tool_and_persist_state(self, *args, **kwargs):
                return None

            def build_system_prompt_with_memories(self, *args, **kwargs):
                return "prompt"

            def _handle_ai_response(self, input_message, response_message):
                # MIRIX v0.1.1 can report the promoted plain-text send as a
                # bookkeeping failure, which would otherwise force 11 retries.
                return ["persisted"], True, True

        adapter = object.__new__(MirixFamilyAdapter)
        fake_module = SimpleNamespace(Agent=FakeAgent)
        response_message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(name="send_message")
                )
            ]
        )
        agent = SimpleNamespace(
            agent_state=SimpleNamespace(name="chat_agent")
        )

        with patch.object(
            mirix_family_module.importlib,
            "import_module",
            return_value=fake_module,
        ):
            adapter._install_native_retrieval_hooks()
        token = mirix_family_module._ACTIVE_MIRIX_RETRIEVAL.set(object())
        try:
            result = FakeAgent._handle_ai_response(
                agent, object(), response_message
            )
        finally:
            mirix_family_module._ACTIVE_MIRIX_RETRIEVAL.reset(token)

        self.assertEqual(result, (["persisted"], False, False))

    def test_mirix_native_answer_call_disables_chaining(self):
        message_module = SimpleNamespace(
            MessageCreate=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        enum_module = SimpleNamespace(MessageRole=lambda value: value)
        content_module = SimpleNamespace(
            TextContent=lambda text: {"type": "text", "text": text},
            ImageContent=lambda **kwargs: kwargs,
        )
        response_module = SimpleNamespace(
            MirixResponse=lambda **kwargs: SimpleNamespace(**kwargs)
        )
        server = SimpleNamespace(
            user_manager=SimpleNamespace(get_user_by_id=lambda _id: object()),
            send_messages=Mock(return_value={"total_tokens": 1}),
        )
        client = SimpleNamespace(
            user=SimpleNamespace(id="user"),
            server=server,
            interface=SimpleNamespace(clear=Mock(), to_list=lambda: []),
        )
        modules = {
            "mirix.schemas.message": message_module,
            "mirix.schemas.enums": enum_module,
            "mirix.schemas.mirix_message_content": content_module,
            "mirix.schemas.mirix_response": response_module,
        }

        with patch.object(
            mirix_family_module.importlib,
            "import_module",
            side_effect=lambda name: modules[name],
        ):
            _send_native_benchmark_messages(
                client,
                "chat",
                [{"role": "user", "content": "question"}],
                None,
            )

        self.assertFalse(server.send_messages.call_args.kwargs["chaining"])

    def test_mirix_reuses_native_upload_compression_without_modifying_source(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "large.jpg"
            Image.new("RGB", (6000, 4000), (127, 63, 31)).save(
                source, format="JPEG", quality=100
            )
            original = source.read_bytes()

            def compress(path, *, quality, max_size):
                with Image.open(path) as image:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                    output = Path(path).with_suffix("").with_name(
                        Path(path).stem + "_compressed"
                    ).with_suffix(".jpg")
                    image.save(output, format="JPEG", quality=quality, optimize=True)
                    return str(output)

            compressor = SimpleNamespace(_compress_image=Mock(side_effect=compress))
            transported = _stage_native_transport_image(
                source,
                cache_dir=root / "state" / "tmp" / "image_transport",
                compressor=compressor,
            )

            self.assertEqual(source.read_bytes(), original)
            self.assertNotEqual(transported, source)
            self.assertLess(transported.stat().st_size, source.stat().st_size)
            with Image.open(transported) as image:
                self.assertLessEqual(image.width, 1920)
                self.assertLessEqual(image.height, 1080)
            self.assertEqual(
                compressor._compress_image.call_args.kwargs,
                {"quality": 85, "max_size": (1920, 1080)},
            )

    def test_mirix_rejects_non_seven_chat_budget(self):
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        with self.assertRaisesRegex(ValueError, "top_k must be 7"):
            adapter.answer_with_memory(
                NativeAnswerRequest(
                    query_id="q",
                    messages=[],
                    retrieval=RetrievalResult(),
                    top_k=5,
                )
            )

    def test_mirix_required_tool_choice_uses_textual_qwen_selection(self):
        request = {
            "tools": [
                {"type": "function", "function": {"name": "search_in_memory"}},
                {"type": "function", "function": {"name": "insert_memory"}},
            ],
            "tool_choice": "required",
        }
        normalized = _normalize_openai_tool_request(request)
        self.assertEqual(normalized["tool_choice"], "none")

    def test_mirix_explicit_named_tool_choice_is_preserved(self):
        choice = {"type": "function", "function": {"name": "insert_memory"}}
        request = {"tools": [{"function": {"name": "insert_memory"}}], "tool_choice": choice}
        normalized = _normalize_openai_tool_request(request)
        self.assertEqual(normalized["tool_choice"], choice)

    def test_mirix_tool_compat_wraps_sync_and_async_requests(self):
        textual = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<tool_call>{"name":"insert_memory",'
                            '"arguments":{"title":"Almond"}}</tool_call>'
                        ),
                        "tool_calls": [],
                    }
                }
            ]
        }

        class FakeOpenAIClient:
            def build_request_data(self):
                return {
                    "tools": [{"function": {"name": "insert_memory"}}],
                    "tool_choice": "required",
                }

            def request(self, _request_data):
                return json.loads(json.dumps(textual))

            async def request_async(self, _request_data):
                return json.loads(json.dumps(textual))

            def convert_response_to_chat_completion(self, response_data, marker):
                return response_data, marker

        class FakeCompletion:
            def __init__(self, **payload):
                self.payload = payload

            def model_dump(self):
                return self.payload

        legacy_seen = {}

        def fake_legacy_request(*_args, **kwargs):
            legacy_seen["request"] = kwargs.get("chat_completion_request")
            return FakeCompletion(**json.loads(json.dumps(textual)))

        legacy_module = SimpleNamespace(
            openai_chat_completions_request=fake_legacy_request
        )

        module = SimpleNamespace(OpenAIClient=FakeOpenAIClient)
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.package = "mirix"
        adapter.config = {"num_predict": 512}

        def fake_import(name):
            if name.endswith(".openai_client"):
                return module
            if name.endswith(".llm_api_tools"):
                return legacy_module
            raise AssertionError(name)

        with patch(
            "benchmarks.baseline_runtime.adapters.mirix_family.importlib.import_module",
            side_effect=fake_import,
        ):
            adapter._patch_openai_tool_compat()

        client = FakeOpenAIClient()
        self.assertEqual(client.build_request_data()["tool_choice"], "none")
        sync_message = client.request({})["choices"][0]["message"]
        async_message = asyncio.run(client.request_async({}))["choices"][0]["message"]
        converted, marker = client.convert_response_to_chat_completion(
            json.loads(json.dumps(textual)), "converted"
        )
        converted_message = converted["choices"][0]["message"]
        legacy_request = SimpleNamespace(max_tokens=None)
        legacy_message = legacy_module.openai_chat_completions_request(
            chat_completion_request=legacy_request
        ).model_dump()["choices"][0]["message"]
        self.assertEqual(
            sync_message["tool_calls"][0]["function"]["name"], "insert_memory"
        )
        self.assertEqual(
            async_message["tool_calls"][0]["function"]["name"], "insert_memory"
        )
        self.assertEqual(
            converted_message["tool_calls"][0]["function"]["name"],
            "insert_memory",
        )
        self.assertEqual(marker, "converted")
        self.assertEqual(
            legacy_message["tool_calls"][0]["function"]["name"], "insert_memory"
        )
        self.assertEqual(legacy_seen["request"].max_tokens, 512)

    def test_mirix_normalizes_pydantic_style_tool_response(self):
        class FakeCompletion:
            def __init__(self, **payload):
                self.payload = payload

            def model_dump(self):
                return self.payload

        response = FakeCompletion(
            choices=[
                {
                    "message": {
                        "content": (
                            '<tool_call>{"name":"finish",'
                            '"arguments":{}}</tool_call>'
                        ),
                        "tool_calls": [],
                    }
                }
            ]
        )
        normalized = _normalize_openai_tool_response(response)
        message = normalized.model_dump()["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "finish")

    def test_mirix_converts_qwen_textual_tool_selection(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<tool_call>\n{"name":"insert_memory",'
                            '"arguments":{"title":"Almond"}}\n</tool_call>'
                        ),
                        "tool_calls": [],
                    }
                }
            ]
        }
        message = _normalize_openai_tool_tags(response)["choices"][0]["message"]
        self.assertIsNone(message["content"])
        function = message["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "insert_memory")
        self.assertEqual(json.loads(function["arguments"]), {"title": "Almond"})

    def test_mirix_converts_qwen_python_style_tool_selection(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "trigger_memory_update("
                            "memory_types=['core', 'episodic'])\n\n"
                            "The profile and event should be retained.\n\n"
                            "finish_memory_update()"
                        ),
                        "tool_calls": [],
                    }
                }
            ]
        }
        message = _normalize_openai_tool_tags(response)["choices"][0]["message"]
        function = message["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "trigger_memory_update")
        arguments = json.loads(function["arguments"])
        self.assertEqual(arguments["memory_types"], ["core", "episodic"])
        self.assertIn("profile and event", arguments["inner_thoughts"])

    def test_mirix_python_style_parser_rejects_positional_calls(self):
        self.assertIsNone(_python_style_tool_payload("unsafe_tool('value')"))

    def test_mirix_converts_multiline_python_style_tool_selection(self):
        payload = _python_style_tool_payload(
            "```python\n"
            "episodic_memory_insert(\n"
            "    items=[{'summary': 'Milo', 'details': '''line one\nline two'''}]\n"
            ")\n"
            "```"
        )
        self.assertEqual(payload["name"], "episodic_memory_insert")
        self.assertEqual(payload["arguments"]["items"][0]["summary"], "Milo")

    def test_mirix_converts_bare_finish_memory_update(self):
        self.assertEqual(
            _python_style_tool_payload("finish_memory_update"),
            {"name": "finish_memory_update", "arguments": {}},
        )

    def test_mirix_converts_tool_call_without_closing_tag(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<tool_call>\n{"name":"insert_memory",'
                            '"arguments":{"title":"Almond"}}'
                        ),
                        "tool_calls": [],
                    }
                }
            ]
        }
        message = _normalize_openai_tool_tags(response)["choices"][0]["message"]
        self.assertIsNone(message["content"])
        function = message["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "insert_memory")
        self.assertEqual(json.loads(function["arguments"]), {"title": "Almond"})

    def test_mirix_repairs_truncated_textual_tool_json(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<tool_call>{"name":"insert_memory",'
                            '"arguments":{"title":"Almond"}'
                        ),
                        "tool_calls": [],
                    }
                }
            ]
        }
        message = _normalize_openai_tool_tags(response)["choices"][0]["message"]
        function = message["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "insert_memory")
        self.assertEqual(json.loads(function["arguments"]), {"title": "Almond"})

    def test_mirix_normalizes_empty_tool_arguments_to_json_object(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_memory_update",
                                    "arguments": "",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        normalized = _normalize_openai_tool_tags(response)
        arguments = normalized["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"]
        self.assertEqual(json.loads(arguments), {})

    def test_mirix_normalizes_invalid_tool_arguments_to_json_object(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_memory_update",
                                    "arguments": "<invalid arguments>",
                                }
                            }
                        ]
                    }
                }
            ]
        }
        normalized = _normalize_openai_tool_tags(response)
        arguments = normalized["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"]
        self.assertEqual(json.loads(arguments), {})

    def test_mirix_preserves_standard_structured_tool_arguments(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "update_memory",
                                    "arguments": '{"memory_id": "m1", "value": 7}',
                                }
                            }
                        ]
                    }
                }
            ]
        }
        normalized = _normalize_openai_tool_tags(response)
        function = normalized["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]
        self.assertEqual(function["name"], "update_memory")
        self.assertEqual(
            json.loads(function["arguments"]),
            {"memory_id": "m1", "value": 7},
        )

    def test_mirix_repairs_unsupported_resource_embedding_search(self):
        response = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<tool_call>{"name":"search_in_memory",'
                            '"arguments":{"memory_type":"resource",'
                            '"query":"repair manual","search_field":"content",'
                            '"search_method":"embedding"}}</tool_call>'
                        ),
                        "tool_calls": [],
                    }
                }
            ]
        }
        message = _normalize_openai_tool_tags(response)["choices"][0]["message"]
        arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["search_field"], "summary")
        self.assertEqual(arguments["search_method"], "embedding")

    def test_mirix_does_not_fabricate_semantic_memory_for_native_noop(self):
        existing = {
            "memory_id": "semantic_memory_manager:sem_old",
            "text": "old memory",
        }
        accumulator = SimpleNamespace(temporary_messages=[])

        def native_send(**_kwargs):
            accumulator.temporary_messages.append(("timestamp", {}))

        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.config = {"mirix_semantic_fallback_on_error": False}
        adapter.backend = SimpleNamespace(
            send_message=Mock(side_effect=native_send),
            temp_message_accumulator=accumulator,
        )
        adapter._memory_rows = Mock(return_value=[existing])
        adapter._insert_fallback_memory = Mock()
        adapter.provenance = ProvenanceIndex()
        adapter._known_ids = {existing["memory_id"]}
        adapter._last_chunk = None
        adapter._pending_chunks = []
        adapter._ingested_chunks = 0
        chunk = Chunk(
            chunk_id="S1:R1",
            text="new memory",
            metadata={"session_id": "S1"},
        )

        adapter.ingest(chunk)

        adapter._insert_fallback_memory.assert_not_called()
        self.assertEqual(adapter._pending_chunks, [chunk])
        self.assertEqual(adapter._ingested_chunks, 1)

    def test_mirix_native_error_is_not_silently_replaced_by_semantic_memory(self):
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.config = {"mirix_semantic_fallback_on_error": False}
        adapter.backend = SimpleNamespace(
            send_message=Mock(side_effect=RuntimeError("native failure")),
            temp_message_accumulator=SimpleNamespace(temporary_messages=[]),
        )
        adapter._memory_rows = Mock(return_value=[])
        adapter._insert_fallback_memory = Mock()
        adapter.provenance = ProvenanceIndex()
        adapter._known_ids = set()
        adapter._last_chunk = None
        adapter._pending_chunks = []
        adapter._ingested_chunks = 0
        chunk = Chunk(chunk_id="S1:R1", text="new", metadata={})

        with self.assertRaisesRegex(RuntimeError, "native failure"):
            adapter.ingest(chunk)
        adapter._insert_fallback_memory.assert_not_called()

    def test_mirix_never_direct_inserts_after_wma_chunk_40(self):
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.config = {"native_ingest_chunk_limit": 40}
        adapter._ingested_chunks = 400
        chunk = Chunk(
            chunk_id="S41:R1",
            text="late lifelong memory",
            metadata={
                "benchmark": "WorldMemArena",
                "dataset": "academic_03",
                "dialogue_id": "S41",
                "round_id": "R1",
                "date": "2026-01-01",
            },
        )

        self.assertFalse(adapter._should_direct_insert(chunk))

    def test_mirix_native_batch_size_is_configurable_for_qwen(self):
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.baseline = "MIRIX"
        adapter.config = {"mirix_native_batch_size": 5}
        accumulator = SimpleNamespace(temporary_message_limit=20)
        adapter.backend = SimpleNamespace(temp_message_accumulator=accumulator)

        adapter._configure_native_absorption_batch()

        self.assertEqual(accumulator.temporary_message_limit, 20)

    def test_output_layout_keeps_memory_under_baseline_root(self):
        layout = BaselineOutputLayout(Path("outputs/Mem-Gallery/M2A"))
        self.assertEqual(
            layout.datasets_dir,
            Path("outputs/Mem-Gallery/M2A/memory/datasets"),
        )
        self.assertEqual(
            layout.snapshot,
            Path("outputs/Mem-Gallery/M2A/memory/memory_snapshot.jsonl"),
        )
        self.assertEqual(layout.state_root("custom/state"), Path("custom/state"))

    def test_chunk_text_replaces_lone_unicode_surrogates(self):
        self.assertEqual(compact_text("before\udc94after"), "before?after")

    def test_registry_exposes_every_supported_baseline_and_aliases(self):
        self.assertEqual(
            set(BASELINE_NAMES),
            {
                "HiveMem",
                "AUGUSTUSMemory",
                "OmniSimpleMem",
                "M2A",
                "MIRIX",
                "MMA",
                "MemVerse",
                "M3-Agent-caption",
            },
        )
        self.assertEqual(canonical_name("m3-agent"), "M3-Agent-caption")
        self.assertEqual(canonical_name("omni-simplemem"), "OmniSimpleMem")
        with self.assertRaises(KeyError):
            canonical_name("MGMemory")
        self.assertEqual(
            baseline_metadata("m3-agent")["compatibility_mode"],
            "official-memory-graph-control-dialogue-observation",
        )

    def test_omni_fixed_top_k_disables_lexical_modality_hard_filters(self):
        for modality in ("visual", "audio", "video", "text"):
            with self.subTest(modality=modality):
                adapter = object.__new__(OmniSimpleMemAdapter)
                processor = SimpleNamespace(
                    determine_retrieval_strategy=lambda _parsed, value=modality: {
                        "top_k": 20,
                        "use_hybrid": True,
                        "time_filter": (1, 2),
                        "modality_filter": SimpleNamespace(value=value),
                    }
                )
                adapter.backend = SimpleNamespace(query_processor=processor)
                adapter._last_native_top_k = None
                adapter._last_native_modality_filter = None
                adapter._install_fixed_top_k_policy()
                strategy = processor.determine_retrieval_strategy(object())
                self.assertEqual(
                    strategy,
                    {
                        "top_k": 7,
                        "use_hybrid": True,
                        "time_filter": (1, 2),
                        "modality_filter": None,
                    },
                )
                self.assertEqual(adapter._last_native_top_k, 20)
                self.assertEqual(adapter._last_native_modality_filter, modality)

    def test_omni_fixed_top_k_preserves_absent_modality_filter(self):
        adapter = object.__new__(OmniSimpleMemAdapter)
        processor = SimpleNamespace(
            determine_retrieval_strategy=lambda _parsed: {
                "top_k": 10,
                "modality_filter": None,
            }
        )
        adapter.backend = SimpleNamespace(query_processor=processor)
        adapter._last_native_top_k = None
        adapter._last_native_modality_filter = "stale"
        adapter._install_fixed_top_k_policy()
        strategy = processor.determine_retrieval_strategy(object())
        self.assertEqual(strategy["top_k"], 7)
        self.assertIsNone(strategy["modality_filter"])
        self.assertIsNone(adapter._last_native_modality_filter)

    def test_omni_uses_verified_official_source_and_keeps_graph_enabled(self):
        source_root = (
            Path(__file__).resolve().parents[1]
            / ".upstream"
            / "simplemem-836ce97"
            / "OmniSimpleMem"
        )
        base_config = {
            "embedding_model": "embedding",
            "embedding_dim": 2048,
            "embedding_base_url": "http://127.0.0.1:8001/v1",
            "executor_model": "executor",
            "executor_base_url": "http://127.0.0.1:8015/v1",
            "executor_api_key": "test-key",
            "executor_temperature": 0,
            "retries": 2,
            "top_k": 7,
        }
        adapter = OmniSimpleMemAdapter(
            baseline="OmniSimpleMem",
            source_root=source_root,
            config=base_config,
        )
        self.assertTrue(adapter._build_config().retrieval.enable_graph_traversal)
        self.assertEqual(adapter._build_config().retrieval.default_top_k, 7)

    def test_memverse_restores_confirmed_graph_boundaries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "adapter_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "completed_chunk_ids": ["S1:R1"],
                        "completed_session_ids": ["S1"],
                        "graph_rows": {"core": 1, "episodic": 0, "semantic": 1},
                    }
                ),
                encoding="utf-8",
            )
            adapter = object.__new__(MemVerseAdapter)
            adapter._memory_rows = {
                memory_type: {"S1:R1": {"id": "S1:R1"}}
                for memory_type in adapter._memory_types()
            }
            adapter._reuse_existing_state = True
            adapter._state_path = state_path
            adapter._restore_state()
            self.assertEqual(adapter._completed_chunk_ids, {"S1:R1"})
            self.assertEqual(adapter._completed_session_ids, {"S1"})
            self.assertEqual(
                adapter._graph_rows,
                {"core": 1, "episodic": 0, "semantic": 1},
            )

    def test_memverse_graph_resume_skips_confirmed_stores(self):
        calls: list[tuple[str, int]] = []

        async def insert(_store, path, *, start_row):
            calls.append((path, start_row))

        adapter = object.__new__(MemVerseAdapter)
        adapter.module = SimpleNamespace(
            count_jsonl_rows=lambda _path: 2,
            insert_chunks_from_json=insert,
        )
        adapter._current_session = "S1"
        adapter._graph_rows = {"core": 2, "episodic": 0, "semantic": 0}
        adapter._graph_stores = lambda: (
            ("core", object(), "core"),
            ("episodic", object(), "episodic"),
            ("semantic", object(), "semantic"),
        )
        adapter._run = asyncio.run
        adapter._persist_state = Mock()
        adapter._flush_graph()
        self.assertEqual(calls, [("episodic", 0), ("semantic", 0)])
        self.assertEqual(adapter._graph_rows, {"core": 2, "episodic": 2, "semantic": 2})
        self.assertEqual(adapter._persist_state.call_count, 2)

    def test_memverse_queries_three_stores_concurrently(self):
        active = 0
        max_active = 0

        class Store:
            async def aquery(self, _text, *, param):
                nonlocal active, max_active
                del param
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.01)
                active -= 1
                return "context"

        fake_lightrag = ModuleType(
            "MemoryKB.Long_Term_Memory.Graph_Construction.lightrag"
        )
        fake_lightrag.QueryParam = lambda **kwargs: kwargs
        adapter = object.__new__(MemVerseAdapter)
        adapter.module = SimpleNamespace(
            mem_core=Store(), mem_epi=Store(), mem_sem=Store()
        )
        adapter.baseline = "MemVerse"
        adapter._records = {}
        adapter._loop = asyncio.new_event_loop()
        try:
            with patch.dict(
                sys.modules,
                {
                    "MemoryKB.Long_Term_Memory.Graph_Construction.lightrag": fake_lightrag
                },
            ):
                result = adapter.retrieve(
                    RetrievalRequest(query_id="q", text="question", top_k=5)
                )
        finally:
            adapter._loop.close()
        self.assertEqual(max_active, 3)
        self.assertEqual(len(result.items), 3)

    def test_memverse_bounds_verbose_gleaning_history(self):
        class CharacterTokenizer:
            @staticmethod
            def encode(text):
                return list(text)

            @staticmethod
            def decode(tokens):
                return "".join(tokens)

        history = [
            {"role": "user", "content": "old-prompt"},
            {"role": "assistant", "content": "x" * 1000},
        ]
        bounded = _bounded_history_messages(
            history,
            prompt="continue",
            system_prompt="system",
            tokenizer=CharacterTokenizer(),
            max_input_tokens=400,
        )
        self.assertEqual(len(bounded), 1)
        self.assertEqual(bounded[0]["role"], "assistant")
        self.assertGreater(len(bounded[0]["content"]), 0)
        self.assertLess(len(bounded[0]["content"]), 1000)

    def test_memverse_bounds_oversized_prompt_and_preserves_ends(self):
        class CharacterTokenizer:
            @staticmethod
            def encode(text):
                return list(text)

            @staticmethod
            def decode(tokens):
                return "".join(tokens)

        prompt = "INSTRUCTIONS:" + "x" * 1000 + ":CURRENT_SOURCE"
        bounded = _bounded_prompt(
            prompt,
            system_prompt="system",
            tokenizer=CharacterTokenizer(),
            max_input_tokens=600,
            history_message_count=2,
        )
        self.assertTrue(bounded.startswith("INSTRUCTIONS:"))
        self.assertTrue(bounded.endswith(":CURRENT_SOURCE"))
        self.assertIn("middle truncated", bounded)
        self.assertLessEqual(
            len(CharacterTokenizer.encode(bounded)),
            600 - (256 + 16 * 4) - len("system"),
        )

        short = "short prompt"
        self.assertEqual(
            _bounded_prompt(
                short,
                system_prompt="system",
                tokenizer=CharacterTokenizer(),
                max_input_tokens=400,
            ),
            short,
        )

    def test_memverse_repairs_only_corrupt_lightrag_response_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            core = state_dir / "graph" / "core"
            core.mkdir(parents=True)
            cache = core / "kv_store_llm_response_cache.json"
            cache.write_text('{"unfinished":', encoding="utf-8")
            authoritative = core / "kv_store_full_docs.json"
            authoritative.write_text('{"keep": true}', encoding="utf-8")

            repaired = _repair_corrupt_lightrag_caches(state_dir)

            self.assertEqual(len(repaired), 1)
            self.assertEqual(repaired[0][0], cache)
            self.assertEqual(json.loads(cache.read_text(encoding="utf-8")), {})
            self.assertEqual(
                repaired[0][1].read_text(encoding="utf-8"),
                '{"unfinished":',
            )
            self.assertEqual(
                json.loads(authoritative.read_text(encoding="utf-8")),
                {"keep": True},
            )

    def test_memverse_replaces_explicit_none_max_tokens(self):
        kwargs = {"max_tokens": None, "temperature": 0}
        _apply_executor_max_tokens(kwargs, configured_max_tokens=512)
        self.assertEqual(kwargs["max_tokens"], 512)
        self.assertEqual(kwargs["temperature"], 0)

        explicit = {"max_tokens": 128}
        _apply_executor_max_tokens(explicit, configured_max_tokens=512)
        self.assertEqual(explicit["max_tokens"], 128)

        oversized = {"max_tokens": 1200}
        _apply_executor_max_tokens(oversized, configured_max_tokens=512)
        self.assertEqual(oversized["max_tokens"], 512)

    def test_memverse_caps_native_summary_client(self):
        seen: list[dict[str, object]] = []

        class Completions:
            def create(self, **kwargs):
                seen.append(kwargs)
                return "ok"

        delegate = SimpleNamespace(
            chat=SimpleNamespace(completions=Completions())
        )
        client = _CappedOpenAIClientProxy(delegate, 512)
        self.assertEqual(
            client.chat.completions.create(
                model="executor", max_tokens=1200, temperature=0
            ),
            "ok",
        )
        self.assertEqual(seen[0]["max_tokens"], 512)
        self.assertEqual(seen[0]["temperature"], 0)

    def test_parallel_map_waits_for_other_samples_before_reporting_failure(self):
        visited: list[int] = []
        errors: list[str] = []

        def worker(value: int) -> int:
            visited.append(value)
            if value == 2:
                raise ValueError("broken")
            return value

        with self.assertRaisesRegex(RuntimeError, "2: ValueError: broken"):
            parallel_map_ordered(
                [1, 2, 3],
                worker,
                max_workers=2,
                on_error=lambda key, _exc: errors.append(key),
            )
        self.assertEqual(set(visited), {1, 2, 3})
        self.assertEqual(errors, ["2"])

    def test_stale_sample_checkpoint_requires_explicit_recovery_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            from benchmarks.baseline_runtime.parallel_runner import sample_artifact_path

            path = sample_artifact_path(root, "sample")
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "signature": "old",
                        "artifact": {"value": 1},
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(
                load_sample_artifact(root, "sample", signature="new")
            )
            with patch.dict(
                "os.environ",
                {"BASELINE_ALLOW_STALE_SAMPLE_CHECKPOINT": "1"},
            ):
                self.assertEqual(
                    load_sample_artifact(root, "sample", signature="new"),
                    {"value": 1},
                )

    def test_protocol_keeps_provenance_in_answer_context(self):
        result = RetrievalResult(
            items=[
                RetrievedMemory(
                    memory_id="m1",
                    text="fact",
                    session_id="S1",
                    source_dialogue_ids=["D1"],
                    image_ids=["I1"],
                    image_paths=["image.png"],
                )
            ]
        )
        item = result_context_items(result)[0]
        self.assertEqual(item["metadata"]["session_id"], "S1")
        self.assertEqual(item["metadata"]["source_dialogue_ids"], ["D1"])
        self.assertEqual(item["image"], {"path": "image.png", "img_id": "I1"})


class BaselineHarnessTest(unittest.TestCase):
    def test_wma_sample_retry_queue_isolates_then_retries(self):
        calls: list[str] = []

        def worker(path: Path) -> dict[str, Any]:
            calls.append(path.stem)
            if path.stem == "bad" and calls.count("bad") == 1:
                raise RuntimeError("empty native answer")
            return {"sample_id": path.stem}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "bad.json", root / "good.json"]
            artifacts = run_sample_retry_queue(
                paths,
                worker,
                max_attempts=3,
                status_path=root / "sample_status.json",
            )
            status = json.loads((root / "sample_status.json").read_text())

        self.assertEqual(calls, ["bad", "good", "bad"])
        self.assertEqual([row["sample_id"] for row in artifacts], ["bad", "good"])
        self.assertEqual(status["samples"]["bad"]["state"], "completed")
        self.assertEqual(status["samples"]["bad"]["attempts"], 2)

    def test_wma_mirix_checkpoints_each_qa_and_resumes_only_missing_question(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [{"_v2_session_id": "S1", "dialogue": []}],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S1"],
                    "questions": [
                        {"question": "first", "answer": "one", "question_type_abbrev": "FR"},
                        {"question": "second", "answer": "two", "question_type_abbrev": "FR"},
                    ],
                }
            ],
        }

        class RetryableMirix(FakeBaseline):
            def __init__(self, fail_second: bool):
                super().__init__()
                self.fail_second = fail_second
                self.answer_calls = 0

            def completed_session_ids(self):
                return ()

            def filter_completed_session_chunks(self, chunks):
                return chunks

            def answer_with_memory(self, request: NativeAnswerRequest):
                self.answer_calls += 1
                if self.fail_second and self.answer_calls == 2:
                    raise RuntimeError("no final send_message")
                return NativeAnswerResult(
                    text="<answer>ok</answer>", retrieval=request.retrieval
                )

        completed: dict[str, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []
        fixed = [
            Chunk(
                chunk_id="S1:R1",
                text="fact",
                metadata={"session_id": "S1", "dialogue_id": "D1"},
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")

            def checkpoint(job, result, _trace):
                completed[job["manifest_question_id"]] = job
                results.append(result)

            first = RetryableMirix(fail_second=True)
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter", return_value=first
            ), patch(
                "benchmarks.wma_harness.eval_wma.wma_chunks", return_value=fixed
            ):
                with self.assertRaisesRegex(RuntimeError, "no final send_message"):
                    prepare_native_sample_jobs(
                        sample_path,
                        None,
                        baseline="MIRIX",
                        state_root=root / "state",
                        top_k=7,
                        config_overrides={},
                        checkpoint_answer_client=FakeAnswerClient(),
                        completed_jobs={},
                        on_qa_completed=checkpoint,
                    )

            second = RetryableMirix(fail_second=False)
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter", return_value=second
            ), patch(
                "benchmarks.wma_harness.eval_wma.wma_chunks", return_value=fixed
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="MIRIX",
                    state_root=root / "state",
                    top_k=7,
                    config_overrides={},
                    checkpoint_answer_client=FakeAnswerClient(),
                    completed_jobs=completed,
                    on_qa_completed=checkpoint,
                )

        self.assertEqual(len(jobs), 2)
        self.assertEqual(first.answer_calls, 2)
        self.assertEqual(second.answer_calls, 1)
        self.assertEqual(len(results), 2)

    def test_memgallery_m2a_skips_nine_consecutive_build_faults_and_resets(self):
        payload = {"character_profile": {}, "human-annotated QAs": []}

        class FailingBaseline(FakeBaseline):
            def ingest(self, chunk: Chunk) -> None:
                if chunk.chunk_id.startswith("bad"):
                    raise RuntimeError("Qwen emitted malformed tool arguments")
                super().ingest(chunk)

        fake = FailingBaseline()
        chunks = [
            *[Chunk(chunk_id=f"bad-a-{index}", text="bad") for index in range(9)],
            Chunk(chunk_id="good", text="good"),
            *[Chunk(chunk_id=f"bad-b-{index}", text="bad") for index in range(9)],
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.build_chunks_from_data",
                return_value=chunks,
            ):
                artifact = prepare_dataset_jobs(
                    dataset_path,
                    root,
                    root,
                    None,
                    baseline="M2A",
                    state_root=root / "state",
                    config_overrides={
                        "m2a_skip_failed_build_points": True,
                        "m2a_max_consecutive_failed_build_points": 10,
                    },
                )

        self.assertEqual([chunk.chunk_id for chunk in fake.chunks], ["good"])
        self.assertEqual(len(artifact["build_failures"]), 18)
        self.assertEqual(
            [row["consecutive_failed_build_points"] for row in artifact["build_failures"]],
            [*range(1, 10), *range(1, 10)],
        )
        self.assertTrue(artifact["build_failure_policy"]["enabled"])
        self.assertTrue(fake.closed)

    def test_wma_mma_checkpoints_empty_native_answer_and_skips_it_on_resume(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [{"_v2_session_id": "S1", "dialogue": []}],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S1"],
                    "questions": [
                        {
                            "question": "first",
                            "answer": "one",
                            "question_type_abbrev": "FR",
                        }
                    ],
                }
            ],
        }

        class EmptyAnswerMMA(FakeBaseline):
            def __init__(self, *, fail_if_called: bool = False):
                super().__init__()
                self.answer_calls = 0
                self.fail_if_called = fail_if_called

            def completed_session_ids(self):
                return ()

            def filter_completed_session_chunks(self, chunks):
                return chunks

            def answer_with_memory(self, request: NativeAnswerRequest):
                self.answer_calls += 1
                if self.fail_if_called:
                    raise AssertionError("checkpointed MMA QA was repeated")
                return NativeAnswerResult(
                    text="",
                    retrieval=request.retrieval,
                    trace={"bad_qa_point": True},
                )

        fixed = [
            Chunk(
                chunk_id="S1:R1",
                text="fact",
                metadata={"session_id": "S1", "dialogue_id": "D1"},
            )
        ]
        completed: dict[str, dict[str, Any]] = {}
        checkpoint_results: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")

            def checkpoint(job, result, _trace):
                completed[job["manifest_question_id"]] = job
                checkpoint_results.append(result)

            first = EmptyAnswerMMA()
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter", return_value=first
            ), patch(
                "benchmarks.wma_harness.eval_wma.build_omni_wma_chunks_from_data",
                return_value=fixed,
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="MMA",
                    state_root=root / "state",
                    top_k=7,
                    config_overrides={},
                    checkpoint_answer_client=FakeAnswerClient(),
                    completed_jobs={},
                    on_qa_completed=checkpoint,
                    allow_native_qa_errors=True,
                )

            resumed = EmptyAnswerMMA(fail_if_called=True)
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter", return_value=resumed
            ), patch(
                "benchmarks.wma_harness.eval_wma.build_omni_wma_chunks_from_data",
                return_value=fixed,
            ):
                resumed_jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="MMA",
                    state_root=root / "state",
                    top_k=7,
                    config_overrides={},
                    checkpoint_answer_client=FakeAnswerClient(),
                    completed_jobs=completed,
                    on_qa_completed=checkpoint,
                    allow_native_qa_errors=True,
                )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(len(resumed_jobs), 1)
        self.assertEqual(first.answer_calls, 1)
        self.assertEqual(resumed.answer_calls, 0)
        self.assertEqual(len(checkpoint_results), 1)
        self.assertTrue(checkpoint_results[0]["error"])

    def test_wma_mma_checkpoints_tenth_bad_qa_before_stopping_sample(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [{"_v2_session_id": "S1", "dialogue": []}],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S1"],
                    "questions": [
                        {
                            "question": "failed question",
                            "answer": "gold",
                            "question_type_abbrev": "FR",
                        }
                    ],
                }
            ],
        }

        class ThresholdMMA(FakeBaseline):
            def completed_session_ids(self):
                return ()

            def filter_completed_session_chunks(self, chunks):
                return chunks

            def answer_with_memory(self, _request: NativeAnswerRequest):
                raise RuntimeError(
                    "MMA answer_with_memory failed: MMAConsecutiveBadPointError: "
                    "MMA produced 10 consecutive failed QA points"
                )

        fixed = [
            Chunk(
                chunk_id="S1:R1",
                text="fact",
                metadata={"session_id": "S1", "dialogue_id": "D1"},
            )
        ]
        checkpointed: list[tuple[dict[str, Any], dict[str, Any]]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter",
                return_value=ThresholdMMA(),
            ), patch(
                "benchmarks.wma_harness.eval_wma.build_omni_wma_chunks_from_data",
                return_value=fixed,
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "10 consecutive failed QA points"
                ):
                    prepare_native_sample_jobs(
                        sample_path,
                        None,
                        baseline="MMA",
                        state_root=root / "state",
                        top_k=7,
                        config_overrides={},
                        checkpoint_answer_client=FakeAnswerClient(),
                        on_qa_completed=lambda job, result, _trace: checkpointed.append(
                            (job, result)
                        ),
                        allow_native_qa_errors=True,
                    )

        self.assertEqual(len(checkpointed), 1)
        self.assertEqual(checkpointed[0][0]["native_answer"]["text"], "")
        self.assertTrue(checkpointed[0][1]["error"])

    def test_memgallery_mma_skips_durable_native_qa_on_resume(self):
        payload = {
            "character_profile": {"name": "Sample"},
            "human-annotated QAs": [
                {"question": "first", "answer": "one", "point": "FR"}
            ],
        }

        class CheckpointMMA(FakeBaseline):
            def __init__(self, *, fail_if_called: bool = False):
                super().__init__()
                self.answer_calls = 0
                self.retrieve_calls = 0
                self.fail_if_called = fail_if_called

            def completed_session_ids(self):
                return ()

            def filter_completed_session_chunks(self, chunks):
                return chunks

            def retrieve(self, request: RetrievalRequest):
                self.retrieve_calls += 1
                if self.fail_if_called:
                    raise AssertionError("checkpointed MMA retrieval was repeated")
                return super().retrieve(request)

            def answer_with_memory(self, request: NativeAnswerRequest):
                self.answer_calls += 1
                return NativeAnswerResult(
                    text="<answer>ok</answer>", retrieval=request.retrieval
                )

        completed: dict[str, dict[str, Any]] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "sample.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            first = CheckpointMMA()
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=first,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.build_omni_memgallery_chunks",
                return_value=[],
            ):
                artifact = prepare_dataset_jobs(
                    dataset_path,
                    root,
                    root,
                    None,
                    baseline="MMA",
                    state_root=root / "state",
                    on_qa_completed=lambda job: completed.setdefault(
                        job["manifest_question_id"], job
                    ),
                )

            resumed = CheckpointMMA(fail_if_called=True)
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=resumed,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.build_omni_memgallery_chunks",
                return_value=[],
            ):
                resumed_artifact = prepare_dataset_jobs(
                    dataset_path,
                    root,
                    root,
                    None,
                    baseline="MMA",
                    state_root=root / "state",
                    completed_jobs=completed,
                )

        self.assertEqual(len(artifact["jobs"]), 1)
        self.assertEqual(len(resumed_artifact["jobs"]), 1)
        self.assertEqual(first.retrieve_calls, 1)
        self.assertEqual(first.answer_calls, 1)
        self.assertEqual(resumed.retrieve_calls, 0)
        self.assertEqual(resumed.answer_calls, 0)

    def test_h2hmem_mma_skips_durable_native_qa_on_resume(self):
        qa = {
            "question_id": "q1",
            "question": {"text": "first"},
            "question_type": {"sub_type": "FR"},
            "original_answer": "one",
        }

        class CheckpointMMA(FakeBaseline):
            def __init__(self, *, fail_if_called: bool = False):
                super().__init__()
                self.answer_calls = 0
                self.retrieve_calls = 0
                self.fail_if_called = fail_if_called

            def completed_session_ids(self):
                return ()

            def filter_completed_session_chunks(self, chunks):
                return chunks

            def retrieve(self, request: RetrievalRequest):
                self.retrieve_calls += 1
                if self.fail_if_called:
                    raise AssertionError("checkpointed MMA retrieval was repeated")
                return super().retrieve(request)

            def answer_with_memory(self, request: NativeAnswerRequest):
                self.answer_calls += 1
                return NativeAnswerResult(
                    text="<answer>ok</answer>", retrieval=request.retrieval
                )

        completed: dict[str, dict[str, Any]] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conversation = root / "dyadic" / "dialogue1" / "S1"
            conversation.mkdir(parents=True)
            question_file = conversation / "questions.json"
            first = CheckpointMMA()
            with patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.create_adapter",
                return_value=first,
            ), patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.build_omni_h2h_chunks_from_directory",
                return_value=[],
            ), patch(
                "benchmarks.h2hmem_harness.eval_h2hmem._question_rows",
                return_value=[(question_file, 1, qa)],
            ):
                artifact = prepare_h2h_conversation_jobs(
                    data_dir=root,
                    variant="dyadic",
                    conversation_id="dialogue1",
                    baseline="MMA",
                    state_root=root / "state",
                    config={"top_k": 7},
                    on_qa_completed=lambda job: completed.setdefault(
                        job["manifest_question_id"], job
                    ),
                )

            resumed = CheckpointMMA(fail_if_called=True)
            with patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.create_adapter",
                return_value=resumed,
            ), patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.build_omni_h2h_chunks_from_directory",
                return_value=[],
            ), patch(
                "benchmarks.h2hmem_harness.eval_h2hmem._question_rows",
                return_value=[(question_file, 1, qa)],
            ):
                resumed_artifact = prepare_h2h_conversation_jobs(
                    data_dir=root,
                    variant="dyadic",
                    conversation_id="dialogue1",
                    baseline="MMA",
                    state_root=root / "state",
                    config={"top_k": 7},
                    completed_jobs=completed,
                )

        self.assertEqual(len(artifact["jobs"]), 1)
        self.assertEqual(len(resumed_artifact["jobs"]), 1)
        self.assertEqual(first.retrieve_calls, 1)
        self.assertEqual(first.answer_calls, 1)
        self.assertEqual(resumed.retrieve_calls, 0)
        self.assertEqual(resumed.answer_calls, 0)

    def test_memgallery_m2a_stops_at_ten_consecutive_build_faults(self):
        payload = {"character_profile": {}, "human-annotated QAs": []}

        class AlwaysFailingBaseline(FakeBaseline):
            def ingest(self, chunk: Chunk) -> None:
                raise RuntimeError("Qwen emitted malformed tool arguments")

        fake = AlwaysFailingBaseline()
        chunks = [Chunk(chunk_id=f"bad-{index}", text="bad") for index in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.build_chunks_from_data",
                return_value=chunks,
            ):
                with self.assertRaisesRegex(RuntimeError, "10 consecutive failed M2A"):
                    prepare_dataset_jobs(
                        dataset_path,
                        root,
                        root,
                        None,
                        baseline="M2A",
                        state_root=root / "state",
                        config_overrides={
                            "m2a_skip_failed_build_points": True,
                            "m2a_max_consecutive_failed_build_points": 10,
                        },
                    )
        self.assertTrue(fake.closed)

    def test_memgallery_mirix_skips_isolated_build_faults_and_resets_counter(self):
        payload = {"character_profile": {}, "human-annotated QAs": []}

        class FailingBaseline(FakeBaseline):
            def ingest(self, chunk: Chunk) -> None:
                if chunk.chunk_id.startswith("bad"):
                    raise RuntimeError("native tool arguments truncated")
                super().ingest(chunk)

        fake = FailingBaseline()
        chunks = [
            Chunk(chunk_id="bad-1", text="bad", metadata={"session_id": "S1"}),
            Chunk(chunk_id="good", text="good", metadata={"session_id": "S1"}),
            Chunk(chunk_id="bad-2", text="bad", metadata={"session_id": "S1"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            recorder = CallRecorder(
                trace_path=root / "trace.jsonl",
                baseline="MIRIX",
                benchmark="Mem-Gallery",
                sample_id="demo",
                reset=True,
            )
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.memgallery_chunks",
                return_value=chunks,
            ):
                artifact = prepare_dataset_jobs(
                    dataset_path,
                    root,
                    root,
                    None,
                    baseline="MIRIX",
                    state_root=root / "state",
                    config_overrides={
                        "mirix_skip_failed_build_points": True,
                        "mirix_max_consecutive_failed_build_points": 10,
                    },
                    call_recorder=recorder,
                )

            trace = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]

        self.assertEqual([chunk.chunk_id for chunk in fake.chunks], ["good"])
        self.assertEqual(len(artifact["build_failures"]), 2)
        self.assertEqual(
            [row["consecutive_failed_build_points"] for row in artifact["build_failures"]],
            [1, 1],
        )
        self.assertEqual([row["phase"] for row in trace], ["build_fault", "build_fault"])
        self.assertTrue(fake.closed)

    def test_memgallery_mma_skips_isolated_build_fault_without_flag(self):
        payload = {"character_profile": {}, "human-annotated QAs": []}

        class FailingMMA(FakeBaseline):
            def filter_completed_session_chunks(self, chunks):
                return chunks

            def ingest(self, chunk: Chunk) -> None:
                if chunk.chunk_id.startswith("bad"):
                    raise RuntimeError(
                        "MMANativeAgentFailure: response JSON: Unterminated string"
                    )
                super().ingest(chunk)

        fake = FailingMMA()
        chunks = [
            Chunk(chunk_id="bad-1", text="bad", metadata={"session_id": "S1"}),
            Chunk(chunk_id="good", text="good", metadata={"session_id": "S1"}),
            Chunk(chunk_id="bad-2", text="bad", metadata={"session_id": "S1"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.build_omni_memgallery_chunks",
                return_value=chunks,
            ):
                artifact = prepare_dataset_jobs(
                    dataset_path,
                    root,
                    root,
                    None,
                    baseline="MMA",
                    state_root=root / "state",
                    config_overrides={},
                )

        self.assertEqual([chunk.chunk_id for chunk in fake.chunks], ["good"])
        self.assertEqual(
            [row["consecutive_failed_build_points"] for row in artifact["build_failures"]],
            [1, 1],
        )
        self.assertTrue(artifact["build_failure_policy"]["enabled"])
        self.assertTrue(fake.closed)

    def test_memgallery_mirix_stops_at_ten_consecutive_build_faults(self):
        payload = {"character_profile": {}, "human-annotated QAs": []}

        class AlwaysFailingBaseline(FakeBaseline):
            def ingest(self, chunk: Chunk) -> None:
                raise RuntimeError("native tool arguments truncated")

        fake = AlwaysFailingBaseline()
        chunks = [Chunk(chunk_id=f"bad-{index}", text="bad") for index in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.memgallery_chunks",
                return_value=chunks,
            ):
                with self.assertRaisesRegex(RuntimeError, "10 consecutive"):
                    prepare_dataset_jobs(
                        dataset_path,
                        root,
                        root,
                        None,
                        baseline="MIRIX",
                        state_root=root / "state",
                        config_overrides={
                            "mirix_skip_failed_build_points": True,
                            "mirix_max_consecutive_failed_build_points": 10,
                        },
                    )
        self.assertTrue(fake.closed)

    def test_h2hmem_mirix_skips_isolated_build_faults_and_resets_counter(self):
        class FailingBaseline(FakeBaseline):
            def ingest(self, chunk: Chunk) -> None:
                if chunk.chunk_id.startswith("bad"):
                    raise RuntimeError("native tool arguments truncated")
                super().ingest(chunk)

        fake = FailingBaseline()
        chunks = [
            Chunk(chunk_id="bad-1", text="bad", metadata={"session_id": "S1"}),
            Chunk(chunk_id="good", text="good", metadata={"session_id": "S1"}),
            Chunk(chunk_id="bad-2", text="bad", metadata={"session_id": "S1"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dyadic" / "dialogue1").mkdir(parents=True)
            with patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.h2hmem_chunks",
                return_value=chunks,
            ):
                artifact = prepare_h2h_conversation_jobs(
                    data_dir=root,
                    variant="dyadic",
                    conversation_id="dialogue1",
                    baseline="MIRIX",
                    state_root=root / "state",
                    config={
                        "mirix_skip_failed_build_points": True,
                        "mirix_max_consecutive_failed_build_points": 10,
                    },
                )

        self.assertEqual([chunk.chunk_id for chunk in fake.chunks], ["good"])
        self.assertEqual(len(artifact["build_failures"]), 2)
        self.assertEqual(
            [row["consecutive_failed_build_points"] for row in artifact["build_failures"]],
            [1, 1],
        )
        self.assertTrue(fake.closed)

    def test_h2hmem_mma_skips_isolated_build_fault_without_flag(self):
        class FailingMMA(FakeBaseline):
            def filter_completed_session_chunks(self, chunks):
                return chunks

            def ingest(self, chunk: Chunk) -> None:
                if chunk.chunk_id.startswith("bad"):
                    raise RuntimeError(
                        "MMANativeAgentFailure: response JSON: Unterminated string"
                    )
                super().ingest(chunk)

        fake = FailingMMA()
        chunks = [
            Chunk(chunk_id="bad-1", text="bad", metadata={"session_id": "S1"}),
            Chunk(chunk_id="good", text="good", metadata={"session_id": "S1"}),
            Chunk(chunk_id="bad-2", text="bad", metadata={"session_id": "S1"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dyadic" / "dialogue1").mkdir(parents=True)
            with patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.build_omni_h2h_chunks_from_directory",
                return_value=chunks,
            ):
                artifact = prepare_h2h_conversation_jobs(
                    data_dir=root,
                    variant="dyadic",
                    conversation_id="dialogue1",
                    baseline="MMA",
                    state_root=root / "state",
                    config={},
                )

        self.assertEqual([chunk.chunk_id for chunk in fake.chunks], ["good"])
        self.assertEqual(
            [row["consecutive_failed_build_points"] for row in artifact["build_failures"]],
            [1, 1],
        )
        self.assertTrue(fake.closed)

    def test_wma_m2a_answers_before_ingesting_the_next_checkpoint(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [
                {"_v2_session_id": "S00", "dialogue": []},
                {"_v2_session_id": "S01", "dialogue": []},
            ],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S00"],
                    "questions": [
                        {
                            "question": "early",
                            "answer": "visible",
                            "question_type_abbrev": "FR",
                        }
                    ],
                },
                {
                    "checkpoint_id": "QA01",
                    "covered_sessions": ["S01"],
                    "questions": [
                        {
                            "question": "late",
                            "answer": "future",
                            "question_type_abbrev": "FR",
                        }
                    ],
                },
            ],
        }
        events: list[str] = []

        class OrderedBaseline(FakeBaseline):
            def ingest(self, chunk: Chunk) -> None:
                super().ingest(chunk)
                events.append(f"ingest:{chunk.metadata['session_id']}")

            def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
                events.append(f"retrieve:{request.text}")
                return super().retrieve(request)

        class OrderedAnswerClient(FakeAnswerClient):
            def answer_messages_with_usage(self, **kwargs):
                events.append("answer")
                return super().answer_messages_with_usage(**kwargs)

        chunks = [
            Chunk(
                chunk_id="S00:R1",
                text="Visible fact.",
                metadata={"session_id": "S00", "dialogue_id": "S00"},
            ),
            Chunk(
                chunk_id="S01:R1",
                text="Future fact.",
                metadata={"session_id": "S01", "dialogue_id": "S01"},
            ),
        ]
        results: list[dict] = []
        traces: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter",
                return_value=OrderedBaseline(),
            ), patch(
                "benchmarks.wma_harness.eval_wma.build_wma_chunks_from_data",
                return_value=chunks,
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="M2A",
                    state_root=root / "state",
                    top_k=7,
                    config_overrides={},
                    checkpoint_answer_client=OrderedAnswerClient(),
                    checkpoint_results=results,
                    checkpoint_traces=traces,
                )

        self.assertEqual(
            events,
            [
                "ingest:S00",
                "retrieve:early",
                "answer",
                "ingest:S01",
                "retrieve:late",
                "answer",
            ],
        )
        self.assertEqual(len(jobs), 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(traces), 2)
        self.assertTrue(
            all(
                row["checkpoint_protocol"]["mode"]
                == "answer_before_future_ingest"
                for row in results
            )
        )

    def test_wma_mirix_skips_isolated_build_faults_and_resets_counter(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [{"_v2_session_id": "S00", "dialogue": []}],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S00"],
                    "questions": [],
                }
            ],
        }

        class FailingBaseline(FakeBaseline):
            def completed_session_ids(self) -> tuple[str, ...]:
                return ()

            def filter_completed_session_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
                return chunks

            def ingest(self, chunk: Chunk) -> None:
                if chunk.chunk_id.startswith("bad"):
                    raise RuntimeError("native tool arguments truncated")
                super().ingest(chunk)

        fake = FailingBaseline()
        chunks = [
            Chunk(chunk_id="bad-1", text="bad", metadata={"session_id": "S00"}),
            Chunk(chunk_id="good", text="good", metadata={"session_id": "S00"}),
            Chunk(chunk_id="bad-2", text="bad", metadata={"session_id": "S00"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.wma_harness.eval_wma.wma_chunks",
                return_value=chunks,
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="MIRIX",
                    state_root=root / "state",
                    top_k=7,
                    config_overrides={
                        "mirix_skip_failed_build_points": True,
                        "mirix_max_consecutive_failed_build_points": 10,
                    },
                )

        self.assertEqual(jobs, [])
        self.assertEqual([chunk.chunk_id for chunk in fake.chunks], ["good"])
        self.assertTrue(fake.closed)

    def test_wma_mma_skips_isolated_build_fault_without_flag(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [{"_v2_session_id": "S00", "dialogue": []}],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S00"],
                    "questions": [],
                }
            ],
        }

        class FailingMMA(FakeBaseline):
            def completed_session_ids(self):
                return ()

            def filter_completed_session_chunks(self, chunks):
                return chunks

            def ingest(self, chunk: Chunk) -> None:
                if chunk.chunk_id.startswith("bad"):
                    raise RuntimeError(
                        "MMANativeAgentFailure: response JSON: Unterminated string"
                    )
                super().ingest(chunk)

        fake = FailingMMA()
        chunks = [
            Chunk(chunk_id="bad-1", text="bad", metadata={"session_id": "S00"}),
            Chunk(chunk_id="good", text="good", metadata={"session_id": "S00"}),
            Chunk(chunk_id="bad-2", text="bad", metadata={"session_id": "S00"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.wma_harness.eval_wma.build_omni_wma_chunks_from_data",
                return_value=chunks,
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="MMA",
                    state_root=root / "state",
                    top_k=7,
                    config_overrides={},
                )

        self.assertEqual(jobs, [])
        self.assertEqual([chunk.chunk_id for chunk in fake.chunks], ["good"])
        self.assertTrue(fake.closed)

    def test_memgallery_native_path_ingests_then_answers(self):
        payload = {
            "character_profile": {"name": "Ava"},
            "multi_session_dialogues": [
                {
                    "session_id": "S1",
                    "date": "2025-01-01",
                    "dialogues": [
                        {"round": "D1", "user": "My mug is blue.", "assistant": "Noted."}
                    ],
                }
            ],
            "human-annotated QAs": [
                {"question": "What color is my mug?", "answer": "blue", "point": "AR"}
            ],
        }
        fake = FakeBaseline()
        snapshots: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            fixed = [
                Chunk(
                    chunk_id="D1",
                    text="My mug is blue.",
                    metadata={"session_id": "S1", "dialogue_id": "D1"},
                )
            ]
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.memgallery_chunks",
                return_value=fixed,
            ):
                rows, traces = run_dataset(
                    dataset_path,
                    root,
                    Path(),
                    FakeAnswerClient(),
                    None,
                    baseline="m3-agent",
                    state_root=root / "state",
                    memory_snapshots=snapshots,
                )
        self.assertEqual(rows[0]["system_answer"], "answer from memory")
        self.assertEqual(traces[0]["top_k"][0]["source_dialogue_ids"], ["D1"])
        self.assertEqual(fake.ended_sessions, ["S1"])
        self.assertTrue(fake.closed)
        self.assertEqual(len(snapshots), 1)

    def test_memgallery_excluded_category_never_retrieves_or_answers(self):
        payload = {
            "character_profile": {"name": "Ava"},
            "multi_session_dialogues": [],
            "human-annotated QAs": [
                {"question": "Unknown detail?", "answer": "Not mentioned.", "point": "AR"}
            ],
        }
        fake = FakeBaseline()
        stats: dict[str, int] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.memgallery_chunks",
                return_value=[
                    Chunk(
                        chunk_id="D1",
                        text="fact",
                        metadata={"session_id": "S1", "dialogue_id": "D1"},
                    )
                ],
            ):
                rows, traces = run_dataset(
                    dataset_path,
                    root,
                    Path(),
                    FakeAnswerClient(),
                    None,
                    baseline="m3-agent",
                    state_root=root / "state",
                    excluded_categories=frozenset({"ar"}),
                    qa_stats=stats,
                )
        self.assertEqual(rows, [])
        self.assertEqual(traces, [])
        self.assertEqual(stats, {"eligible_questions": 0, "excluded_questions": 1})
        self.assertTrue(fake.closed)

    def test_wma_native_path_never_ingests_future_sessions(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [
                {
                    "_v2_session_id": "S00",
                    "dialogue": [
                        {"role": "user", "content": "Visible fact."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                },
                {
                    "_v2_session_id": "S01",
                    "dialogue": [
                        {"role": "user", "content": "Future fact."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                },
            ],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S00"],
                    "questions": [
                        {
                            "question": "What is visible?",
                            "answer": "Visible fact.",
                            "question_type_abbrev": "FR",
                        }
                    ],
                },
                {
                    "checkpoint_id": "QA01",
                    "covered_sessions": ["S01"],
                    "questions": [
                        {
                            "question": "What is future?",
                            "answer": "Future fact.",
                            "question_type_abbrev": "FR",
                        }
                    ],
                },
            ],
        }
        fake = FakeBaseline()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")
            fixed = [
                Chunk(
                    chunk_id="S00:R1",
                    text="Visible fact.",
                    metadata={"session_id": "S00", "dialogue_id": "S00"},
                ),
                Chunk(
                    chunk_id="S01:R1",
                    text="Future fact.",
                    metadata={"session_id": "S01", "dialogue_id": "S01"},
                ),
            ]
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter", return_value=fake
            ), patch(
                "benchmarks.wma_harness.eval_wma.wma_chunks", return_value=fixed
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="MemVerse",
                    state_root=root / "state",
                    top_k=5,
                    config_overrides={},
                    ordered_question_ids=["sample_01:QA00:Q001"],
                )
        self.assertEqual([chunk.metadata["session_id"] for chunk in fake.chunks], ["S00"])
        self.assertEqual(jobs[0]["visible_sessions"], ["S00"])
        self.assertEqual(jobs[0]["retrieval_top_k"][0]["session_id"], "S00")
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
