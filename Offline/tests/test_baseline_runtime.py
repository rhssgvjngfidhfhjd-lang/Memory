from __future__ import annotations

import json
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
    result_context_items,
)
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.parallel_runner import (
    load_sample_artifact,
    parallel_map_ordered,
)
from benchmarks.baseline_runtime.registry import (
    BASELINE_NAMES,
    baseline_metadata,
    canonical_name,
)
from benchmarks.baseline_runtime.adapters.mirix_family import (
    MirixFamilyAdapter,
    _normalize_openai_tool_request,
    _normalize_openai_tool_response,
    _normalize_openai_tool_tags,
)
from benchmarks.baseline_runtime.adapters.omni_simplemem import OmniSimpleMemAdapter
from benchmarks.baseline_runtime.adapters.memverse import MemVerseAdapter
from benchmarks.baseline_runtime.provenance import ProvenanceIndex
from benchmarks.memgallery_harness.runner.answer_client import AnswerResponse
from benchmarks.memgallery_harness.eval_memgallery import run_dataset
from benchmarks.wma_harness.eval_wma import prepare_native_sample_jobs
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

    def answer_with_usage(self, **kwargs):
        return AnswerResponse(
            text="answer from memory",
            usage=None,
            attempts=1,
            failed_attempts=0,
        )


class BaselineProtocolTest(unittest.TestCase):
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

        legacy_module = SimpleNamespace(
            openai_chat_completions_request=lambda *_args, **_kwargs: FakeCompletion(
                **json.loads(json.dumps(textual))
            )
        )

        module = SimpleNamespace(OpenAIClient=FakeOpenAIClient)
        adapter = object.__new__(MirixFamilyAdapter)
        adapter.package = "mirix"

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
        legacy_message = legacy_module.openai_chat_completions_request().model_dump()[
            "choices"
        ][0]["message"]
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
            "dialogue_round_as_clip",
        )

    def test_omni_resume_restores_persisted_provenance(self):
        mau = SimpleNamespace(
            id="mau-1",
            summary="remembered fact",
            raw_pointer="/tmp/image.jpg",
            metadata=SimpleNamespace(
                to_dict=lambda: {
                    "session_id": "session2",
                    "tags": [
                        "dialogue_id:session2:R0003",
                        "session_id:session2",
                    ],
                }
            ),
        )
        adapter = object.__new__(OmniSimpleMemAdapter)
        adapter.backend = SimpleNamespace(
            mau_store=SimpleNamespace(get_active=lambda limit: [mau])
        )
        adapter.provenance = ProvenanceIndex()

        adapter._restore_persisted_provenance()

        self.assertEqual(
            adapter.provenance.get("mau-1"),
            {
                "session_id": "session2",
                "session_ids": ["session2"],
                "source_dialogue_ids": ["session2:R0003"],
                "image_ids": [],
                "image_paths": ["/tmp/image.jpg"],
            },
        )

    def test_omni_benchmark_config_disables_unused_graph_traversal(self):
        source_root = Path(__file__).resolve().parents[1] / "baselines" / "OmniSimpleMem"
        base_config = {
            "embedding_model": "embedding",
            "embedding_dim": 2048,
            "embedding_base_url": "http://127.0.0.1:8001/v1",
            "executor_model": "executor",
            "executor_base_url": "http://127.0.0.1:8015/v1",
            "executor_temperature": 0,
            "retries": 2,
        }
        adapter = OmniSimpleMemAdapter(
            baseline="OmniSimpleMem",
            source_root=source_root,
            config=base_config,
        )
        self.assertFalse(adapter._build_config().retrieval.enable_graph_traversal)
        adapter.config["omni_enable_graph_traversal"] = True
        self.assertTrue(adapter._build_config().retrieval.enable_graph_traversal)

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
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
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
                }
            ],
        }
        fake = FakeBaseline()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter", return_value=fake
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="MemVerse",
                    state_root=root / "state",
                    top_k=5,
                    config_overrides={},
                )
        self.assertEqual([chunk.metadata["session_id"] for chunk in fake.chunks], ["S00"])
        self.assertEqual(jobs[0]["visible_sessions"], ["S00"])
        self.assertEqual(jobs[0]["retrieval_top_k"][0]["session_id"], "S00")
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
