from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmarks.baseline_runtime.adapters.mma_original import (
    MAX_CONSECUTIVE_BAD_MEMORY_POINTS,
    MAX_CONSECUTIVE_BAD_QA_POINTS,
    MMANativeAgentFailure,
    MMAOriginalAdapter,
    UPSTREAM_COMMIT,
    UPSTREAM_TREE,
    _is_unsafe_native_tool_error,
    _record_native_agent_failure,
    _require_answer_block,
    _reject_unstructured_native_tool_calls,
    _is_output_length_exhaustion,
    _provider_name,
    _prepare_embeddings_from_config_compat,
    quarantine_pristine_mma_state,
    _mma_observation_timestamp,
    mma_conformance_manifest,
    _selected_memory_context,
    _structured_memory,
    _sum_usage,
)
from benchmarks.baseline_runtime.protocol import (
    NativeAnswerRequest,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
)
from benchmarks.baseline_runtime.provenance import ProvenanceIndex
from benchmarks.baseline_runtime.registry import baseline_metadata
from embedding.chunk_builder import Chunk


def _adapter() -> MMAOriginalAdapter:
    adapter = object.__new__(MMAOriginalAdapter)
    adapter.baseline = "MMA"
    adapter.config = {}
    adapter.provenance = ProvenanceIndex()
    adapter._retrievals = {}
    adapter._ingested_chunks = 0
    adapter._pending_chunks = []
    adapter._pending_memory_fingerprints = None
    adapter._absorption_batches = 0
    adapter._absorption_batch_size = 20
    adapter._source_commit = UPSTREAM_COMMIT
    adapter._source_tree = UPSTREAM_TREE
    adapter._sample_id = "sample"
    adapter._bad_memory_points = {}
    adapter._bad_retrieval_points = {}
    adapter._consecutive_bad_memory_points = 0
    adapter._bad_qa_points = {}
    adapter._consecutive_bad_qa_points = 0
    return adapter


class MMAOriginalReproductionTest(unittest.TestCase):
    @staticmethod
    def _write_pristine_initialization_database(path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            for table in (
                "episodic_memory",
                "semantic_memory",
                "procedural_memory",
                "resource_memory",
                "knowledge_vault",
                "agents",
                "messages",
                "steps",
            ):
                connection.execute(f'CREATE TABLE "{table}" (id INTEGER)')
            connection.executemany(
                "INSERT INTO agents (id) VALUES (?)", [(value,) for value in range(8)]
            )
            connection.executemany(
                "INSERT INTO messages (id) VALUES (?)", [(value,) for value in range(8)]
            )
            connection.commit()
        finally:
            connection.close()

    def test_pristine_precheckpoint_state_is_preserved_by_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sample"
            state.mkdir()
            self._write_pristine_initialization_database(state / "sqlite.db")

            quarantined = quarantine_pristine_mma_state(state)

            self.assertIsNotNone(quarantined)
            assert quarantined is not None
            self.assertFalse(state.exists())
            self.assertTrue((quarantined / "sqlite.db").is_file())

    def test_uncheckpointed_state_with_memory_is_never_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "sample"
            state.mkdir()
            database = state / "sqlite.db"
            self._write_pristine_initialization_database(database)
            connection = sqlite3.connect(database)
            try:
                connection.execute("INSERT INTO semantic_memory (id) VALUES (1)")
                connection.commit()
            finally:
                connection.close()

            self.assertIsNone(quarantine_pristine_mma_state(state))
            self.assertTrue(state.is_dir())

    def test_registry_locks_the_official_source(self) -> None:
        metadata = baseline_metadata("MMA")
        self.assertEqual(metadata["adapter"], "mma_original")
        self.assertEqual(metadata["upstream_commit"], UPSTREAM_COMMIT)
        self.assertEqual(metadata["upstream_tree"], UPSTREAM_TREE)
        self.assertNotIn("fallback", metadata["compatibility_mode"])

    def test_provider_handle_only_marks_loopback_as_vllm(self) -> None:
        self.assertEqual(_provider_name("http://127.0.0.1:8014/v1"), "vllm")
        self.assertEqual(_provider_name("http://localhost:8014/v1"), "vllm")
        self.assertEqual(_provider_name("https://openrouter.ai/api/v1"), "openai")

    def test_conformance_manifest_locks_prompts_and_forbidden_paths(self) -> None:
        manifest = mma_conformance_manifest(answer_prompt_sha256="qa-hash")
        self.assertEqual(manifest["answer_prompt_sha256"], "qa-hash")
        self.assertEqual(len(manifest["internal_prompt_sha256"]), 8)
        self.assertTrue(
            all(
                row["expected"] == row["actual"]
                for row in manifest["internal_prompt_sha256"].values()
            )
        )
        self.assertFalse(any(manifest["forbidden_paths"].values()))

    def test_chunk_is_queued_without_forced_absorption(self) -> None:
        adapter = _adapter()
        adapter._memory_rows = Mock(return_value=[])
        accumulator = SimpleNamespace(temporary_messages=[], temporary_message_limit=20)

        def send_message(**kwargs):
            accumulator.temporary_messages.append(("timestamp", kwargs))

        adapter.backend = SimpleNamespace(
            send_message=send_message,
            temp_message_accumulator=accumulator,
        )
        chunk = Chunk(
            chunk_id="dataset:D1:1",
            text="user: hello\nassistant: hi",
            metadata={
                "session_id": "D1",
                "dialogue_id": "D1:1",
                "timestamp": "2026-01-02",
                "image_ids": ["image-1"],
            },
        )

        adapter.ingest(chunk)

        kwargs = accumulator.temporary_messages[0][1]
        self.assertTrue(kwargs["memorizing"])
        self.assertFalse(kwargs["force_absorb_content"])
        self.assertFalse(kwargs["async_upload"])
        self.assertFalse(kwargs["delete_after_upload"])
        self.assertEqual(kwargs["specific_timestamps"], ["2026-01-02 00:00:00"])
        self.assertEqual(adapter._pending_chunks, [chunk])
        self.assertEqual(adapter.provenance._rows, {})

    def test_original_limit_absorbs_and_registers_batch_provenance(self) -> None:
        adapter = _adapter()
        memory_id = "semantic_memory_manager:memory-1"
        adapter._memory_rows = Mock(
            side_effect=[[], [{"memory_id": memory_id, "fingerprint": "new"}]]
        )
        accumulator = SimpleNamespace(temporary_messages=[], temporary_message_limit=20)

        def send_message(**kwargs):
            accumulator.temporary_messages.append(("timestamp", kwargs))
            if len(accumulator.temporary_messages) >= accumulator.temporary_message_limit:
                accumulator.temporary_messages.clear()

        adapter.backend = SimpleNamespace(
            send_message=send_message,
            temp_message_accumulator=accumulator,
        )
        for index in range(20):
            adapter.ingest(
                Chunk(
                    chunk_id=f"dataset:D1:{index + 1}",
                    text=f"round {index + 1}",
                    metadata={
                        "session_id": "D1",
                        "dialogue_id": f"D1:{index + 1}",
                    },
                )
            )

        self.assertEqual(accumulator.temporary_messages, [])
        self.assertEqual(adapter._pending_chunks, [])
        self.assertEqual(adapter._absorption_batches, 1)
        self.assertEqual(
            adapter.provenance.get(memory_id)["source_dialogue_ids"],
            [f"D1:{index + 1}" for index in range(20)],
        )

    def test_session_end_absorbs_and_registers_tail(self) -> None:
        adapter = _adapter()
        memory_id = "resource_memory_manager:memory-1"
        adapter._memory_rows = Mock(
            side_effect=[[], [{"memory_id": memory_id, "fingerprint": "new"}]]
        )
        accumulator = SimpleNamespace(temporary_messages=[], temporary_message_limit=20)

        def send_message(**kwargs):
            accumulator.temporary_messages.append(("timestamp", kwargs))

        def absorb_content_into_memory(_agent_states):
            accumulator.temporary_messages.clear()

        accumulator.absorb_content_into_memory = Mock(
            side_effect=absorb_content_into_memory
        )
        adapter.backend = SimpleNamespace(
            send_message=send_message,
            temp_message_accumulator=accumulator,
            agent_states=SimpleNamespace(),
        )
        for index in range(3):
            adapter.ingest(
                Chunk(
                    chunk_id=f"dataset:D1:{index + 1}",
                    text=f"round {index + 1}",
                    metadata={
                        "session_id": "D1",
                        "dialogue_id": f"D1:{index + 1}",
                    },
                )
            )

        adapter.end_session("D1")

        accumulator.absorb_content_into_memory.assert_called_once_with(
            adapter.backend.agent_states
        )
        self.assertEqual(adapter._pending_chunks, [])
        self.assertEqual(adapter._absorption_batches, 1)
        self.assertEqual(
            adapter.provenance.get(memory_id)["source_dialogue_ids"],
            ["D1:1", "D1:2", "D1:3"],
        )

    def test_session_checkpoint_restores_database_and_provenance(self) -> None:
        adapter = _adapter()
        adapter.config.update(
            {"mma_resume_enabled": True, "mma_resume_signature": "signature-1"}
        )
        adapter._sample_id = "sample-1"
        adapter._completed_session_ids = []
        adapter._ingested_chunks = 5
        adapter._state_dir = None
        adapter.provenance.register(
            "semantic_memory_manager:memory-1",
            Chunk(
                chunk_id="sample-1:D1:1",
                text="first session",
                metadata={"session_id": "D1", "dialogue_id": "D1:1"},
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory)
            adapter._state_dir = state_dir
            database = state_dir / "sqlite.db"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE marker (value TEXT)")
                connection.execute("INSERT INTO marker VALUES ('complete-D1')")
                connection.commit()

            adapter._checkpoint_completed_session("D1")
            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE marker SET value = 'partial-D2'")
                connection.commit()

            payload = adapter._load_resume_checkpoint("sample-1", state_dir)
            self.assertIsNotNone(payload)
            adapter._restore_resume_database(state_dir, payload)
            with sqlite3.connect(database) as connection:
                value = connection.execute("SELECT value FROM marker").fetchone()[0]

            self.assertEqual(value, "complete-D1")
            self.assertEqual(payload["completed_session_ids"], ["D1"])
            self.assertEqual(
                payload["provenance"]["semantic_memory_manager:memory-1"][
                    "source_dialogue_ids"
                ],
                ["D1:1"],
            )
            manifest = json.loads(
                (state_dir / ".offline_mma_resume.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["sqlite_snapshot"], "sqlite.session-000001.db")

    def test_session_checkpoint_accepts_explicit_compatible_signature(self) -> None:
        adapter = _adapter()
        adapter.config.update(
            {"mma_resume_enabled": True, "mma_resume_signature": "legacy-signature"}
        )
        adapter._sample_id = "sample-1"
        adapter._completed_session_ids = []
        adapter._ingested_chunks = 0
        adapter._state_dir = None
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory)
            adapter._state_dir = state_dir
            with sqlite3.connect(state_dir / "sqlite.db") as connection:
                connection.execute("CREATE TABLE marker (value TEXT)")
                connection.commit()
            adapter._checkpoint_completed_session("D1")
            adapter.config.update(
                mma_resume_signature="normalized-signature",
                mma_resume_compatible_signatures=["legacy-signature"],
            )

            payload = adapter._load_resume_checkpoint("sample-1", state_dir)

            self.assertIsNotNone(payload)
            self.assertEqual(payload["signature"], "legacy-signature")

    def test_resume_refuses_to_delete_incompatible_existing_state(self) -> None:
        adapter = _adapter()
        adapter.config = {"mma_resume_enabled": True}
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_dir = Path(temporary_directory)
            marker = state_dir / "sqlite.db"
            marker.write_bytes(b"existing-state")
            with patch.object(adapter, "_verify_official_source"), patch.object(
                adapter, "_load_resume_checkpoint", return_value=None
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "refusing to delete existing state"
                ):
                    adapter.reset("sample-1", state_dir)

            self.assertEqual(marker.read_bytes(), b"existing-state")

    def test_resume_filters_only_a_contiguous_completed_session_prefix(self) -> None:
        adapter = _adapter()
        adapter._completed_session_ids = ["D1", "D2"]
        chunks = [
            Chunk(
                chunk_id=f"sample:{session_id}:1",
                text=session_id,
                metadata={"session_id": session_id},
            )
            for session_id in ("D1", "D2", "D3")
        ]

        pending = adapter.filter_completed_session_chunks(chunks)

        self.assertEqual([chunk.metadata["session_id"] for chunk in pending], ["D3"])
        adapter._completed_session_ids = ["D2"]
        with self.assertRaisesRegex(RuntimeError, "contiguous source-session prefix"):
            adapter.filter_completed_session_chunks(chunks)

    def test_native_noop_is_not_replaced_with_a_memory(self) -> None:
        adapter = _adapter()
        adapter._memory_rows = Mock(side_effect=[[], []])
        accumulator = SimpleNamespace(temporary_messages=[], temporary_message_limit=20)

        def send_message(**kwargs):
            accumulator.temporary_messages.append(("timestamp", kwargs))

        def absorb_content_into_memory(_agent_states):
            accumulator.temporary_messages.clear()

        accumulator.absorb_content_into_memory = absorb_content_into_memory
        adapter.backend = SimpleNamespace(
            send_message=send_message,
            temp_message_accumulator=accumulator,
            agent_states=SimpleNamespace(),
        )

        adapter.ingest(Chunk(chunk_id="D1:1", text="ordinary turn"))
        adapter.end_session("")

        self.assertEqual(adapter.provenance._rows, {})
        self.assertFalse(hasattr(adapter, "_insert_fallback_memory"))

    def test_native_build_error_is_not_silenced(self) -> None:
        adapter = _adapter()
        adapter._memory_rows = Mock(return_value=[])
        adapter.backend = SimpleNamespace(
            send_message=Mock(side_effect=RuntimeError("native build failed")),
            temp_message_accumulator=SimpleNamespace(temporary_messages=[]),
        )

        with self.assertRaisesRegex(RuntimeError, "native build failed"):
            adapter.ingest(Chunk(chunk_id="D1:1", text="important turn"))

    def test_output_length_exhaustion_is_distinct_from_context_overflow(self) -> None:
        self.assertTrue(
            _is_output_length_exhaustion(
                RuntimeError(
                    "Retries exhausted and no valid response received. Final error: "
                    "maximum context length exceeded or generated content is too long"
                )
            )
        )
        self.assertFalse(
            _is_output_length_exhaustion(RuntimeError("maximum context length is 4096"))
        )

    def test_swallowed_native_agent_failure_is_raised_by_adapter(self) -> None:
        adapter = _adapter()
        lock = __import__("threading").Lock()
        queue = SimpleNamespace(
            _message_queue_lock=lock,
            _offline_mma_native_failures=[
                {"agent_type": "resource_memory_agent", "error": "truncated"}
            ],
        )
        adapter.backend = SimpleNamespace(message_queue=queue)

        with self.assertRaisesRegex(RuntimeError, "resource_memory_agent"):
            adapter._raise_native_agent_failures()

        self.assertEqual(queue._offline_mma_native_failures, [])

    def test_nested_failure_stops_meta_agent_before_another_chain_step(self) -> None:
        adapter = _adapter()

        class FakeMessageQueue:
            pass

        class FakeAgent:
            def execute_tool_and_persist_state(self, *args, **kwargs):
                return "ok"

            def inner_step(self, *args, **kwargs):
                return "successful-looking upstream step"

        modules = {
            "mma.agent.message_queue": SimpleNamespace(MessageQueue=FakeMessageQueue),
            "mma.agent.agent": SimpleNamespace(Agent=FakeAgent),
        }
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            adapter._install_strict_message_queue()

        queue = SimpleNamespace(
            _message_queue_lock=__import__("threading").Lock(),
            _offline_mma_native_failures=[
                {"agent_type": "resource_memory_agent", "error": "truncated"}
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "resource_memory_agent"):
            FakeAgent().inner_step(message_queue=queue)

        # The outer adapter owns final reporting/clearing of the failure ledger.
        self.assertEqual(len(queue._offline_mma_native_failures), 1)

    def test_recoverable_tool_validation_error_remains_in_native_agent_loop(self) -> None:
        adapter = _adapter()

        class FakeMessageQueue:
            pass

        class FakeAgent:
            agent_state = SimpleNamespace(name="core_memory_agent")

            def execute_tool_and_persist_state(self, *args, **kwargs):
                return (
                    "Error executing function core_memory_append: ValueError: "
                    "You should not include 'Line n:' in the content."
                )

            def inner_step(self, *args, **kwargs):
                self.tool_response = self.execute_tool_and_persist_state(
                    "core_memory_append"
                )
                return "native agent may retry this validation failure"

        modules = {
            "mma.agent.message_queue": SimpleNamespace(MessageQueue=FakeMessageQueue),
            "mma.agent.agent": SimpleNamespace(Agent=FakeAgent),
        }
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            adapter._install_strict_message_queue()

        queue = SimpleNamespace(
            _message_queue_lock=__import__("threading").Lock(),
            _offline_mma_native_failures=[],
        )
        response = FakeAgent().inner_step(message_queue=queue)

        self.assertIn("may retry", response)
        self.assertEqual(queue._offline_mma_native_failures, [])
        self.assertFalse(
            _is_unsafe_native_tool_error(
                "Error executing function core_memory_append: ValueError: invalid"
            )
        )

    def test_unsafe_database_tool_error_stops_memory_agent(self) -> None:
        adapter = _adapter()

        class FakeMessageQueue:
            pass

        class FakeAgent:
            agent_state = SimpleNamespace(name="episodic_memory_agent")

            def __init__(self):
                self.executed_after_failure = False

            def execute_tool_and_persist_state(self, *args, **kwargs):
                return (
                    "Error executing function episodic_memory_merge: "
                    "DetachedInstanceError: database session closed"
                )

            def inner_step(self, *args, **kwargs):
                self.execute_tool_and_persist_state("episodic_memory_merge")
                self.executed_after_failure = True
                return "successful-looking upstream step"

        modules = {
            "mma.agent.message_queue": SimpleNamespace(MessageQueue=FakeMessageQueue),
            "mma.agent.agent": SimpleNamespace(Agent=FakeAgent),
        }
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            adapter._install_strict_message_queue()

        queue = SimpleNamespace(
            _message_queue_lock=__import__("threading").Lock(),
            _offline_mma_native_failures=[],
        )
        agent = FakeAgent()
        with self.assertRaisesRegex(RuntimeError, "database session closed"):
            agent.inner_step(message_queue=queue)

        self.assertFalse(agent.executed_after_failure)
        self.assertEqual(
            queue._offline_mma_native_failures[0]["agent_type"],
            "episodic_memory_agent",
        )
        self.assertEqual(
            queue._offline_mma_native_failures[0]["tool_name"],
            "episodic_memory_merge",
        )
        self.assertEqual(len(queue._offline_mma_native_failures), 1)
        self.assertTrue(
            _is_unsafe_native_tool_error(
                "Error executing function episodic_memory_merge: "
                "DetachedInstanceError: database session closed"
            )
        )

    def test_only_first_concurrent_native_failure_is_recorded(self) -> None:
        queue = SimpleNamespace(
            _message_queue_lock=__import__("threading").Lock(),
            _offline_mma_native_failures=[],
        )

        _record_native_agent_failure(
            queue,
            "episodic_memory_agent",
            RuntimeError("first root cause"),
            tool_name="episodic_memory_merge",
        )
        _record_native_agent_failure(
            queue,
            "semantic_memory_agent",
            MMANativeAgentFailure(queue._offline_mma_native_failures),
        )

        self.assertEqual(len(queue._offline_mma_native_failures), 1)
        self.assertIn(
            "first root cause", queue._offline_mma_native_failures[0]["error"]
        )
        self.assertEqual(
            queue._offline_mma_native_failures[0]["tool_name"],
            "episodic_memory_merge",
        )

    def test_recorded_failure_propagation_does_not_amplify_ledger(self) -> None:
        adapter = _adapter()

        class FakeMessageQueue:
            def __init__(self):
                self._message_queue_lock = __import__("threading").Lock()
                self.message_queue = {}
                self._offline_mma_native_failures = [
                    {
                        "agent_type": "episodic_memory_agent",
                        "tool_name": "episodic_memory_merge",
                        "error": "RuntimeError: first root cause",
                    }
                ]

            def _check_if_earlier_requests_are_finished(self, _message_id):
                return True

        class FakeAgent:
            def execute_tool_and_persist_state(self, *args, **kwargs):
                return "ok"

            def inner_step(self, *args, **kwargs):
                return "ok"

        modules = {
            "mma.agent.message_queue": SimpleNamespace(MessageQueue=FakeMessageQueue),
            "mma.agent.agent": SimpleNamespace(Agent=FakeAgent),
        }
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            adapter._install_strict_message_queue()

        queue = FakeMessageQueue()
        propagated = MMANativeAgentFailure(queue._offline_mma_native_failures)
        client = SimpleNamespace(send_message=Mock(side_effect=propagated))

        with self.assertRaises(MMANativeAgentFailure):
            queue.send_message_in_queue(client, "agent-id", {})

        self.assertEqual(len(queue._offline_mma_native_failures), 1)
        self.assertIn(
            "first root cause", queue._offline_mma_native_failures[0]["error"]
        )
        self.assertEqual(queue.message_queue, {})

    def test_database_error_classifier_is_narrow_and_fail_closed(self) -> None:
        unsafe_names = (
            "DetachedInstanceError",
            "PendingRollbackError",
            "OperationalError",
            "IntegrityError",
        )
        for exception_name in unsafe_names:
            with self.subTest(exception_name=exception_name):
                self.assertTrue(
                    _is_unsafe_native_tool_error(
                        "Error executing function episodic_memory_merge: "
                        f"{exception_name}: database failure"
                    )
                )

        recoverable = (
            "ValueError",
            "TypeError",
            "JSONDecodeError",
            "ValidationError",
        )
        for exception_name in recoverable:
            with self.subTest(exception_name=exception_name):
                self.assertFalse(
                    _is_unsafe_native_tool_error(
                        "Error executing function core_memory_append: "
                        f"{exception_name}: correct and retry"
                    )
                )

        self.assertFalse(
            _is_unsafe_native_tool_error(
                "ordinary assistant response mentioning DetachedInstanceError"
            )
        )

    def test_session_context_is_reentrant_and_only_outer_owner_closes(self) -> None:
        adapter = _adapter()
        from contextlib import contextmanager
        from sqlalchemy.orm import Session, sessionmaker

        class TrackingSession(Session):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                return super().close()

        session_factory = sessionmaker(
            class_=TrackingSession,
            expire_on_commit=False,
        )

        @contextmanager
        def upstream_db_context():
            session = session_factory()
            try:
                yield session
            finally:
                session.close()

        module = SimpleNamespace(
            SessionLocal=session_factory,
            db_context=upstream_db_context,
        )
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            return_value=module,
        ):
            adapter._install_sqlalchemy_session_compatibility()
            installed_class = session_factory.class_
            adapter._install_sqlalchemy_session_compatibility()

        self.assertIs(session_factory.class_, installed_class)
        self.assertTrue(session_factory.kw["expire_on_commit"])
        self.assertTrue(module.db_context._offline_mma_manager_owned_context)
        with module.db_context() as session:
            outer = session
            self.assertIs(outer, session)
            self.assertEqual(session._offline_mma_context_depth, 1)
            for _ in range(3):
                with session as nested:
                    self.assertIs(nested, session)
                    self.assertEqual(session._offline_mma_context_depth, 2)
                self.assertEqual(session._offline_mma_context_depth, 1)
                self.assertEqual(session.close_calls, 0)
        self.assertEqual(session._offline_mma_context_depth, 0)
        self.assertEqual(session.close_calls, 1)

    def test_embedding_storage_dimension_uses_configured_native_width(self) -> None:
        adapter = _adapter()
        adapter.config["embedding_dim"] = 2048
        constants = SimpleNamespace(MAX_EMBEDDING_DIM=4096)
        early_module = SimpleNamespace(MAX_EMBEDDING_DIM=4096)
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            return_value=constants,
        ), patch.dict(
            "benchmarks.baseline_runtime.adapters.mma_original.sys.modules",
            {"mma.schemas.episodic_memory": early_module},
        ):
            adapter._install_embedding_dimension_compatibility()
            adapter._install_embedding_dimension_compatibility()

        self.assertEqual(constants.MAX_EMBEDDING_DIM, 2048)
        self.assertEqual(constants._offline_mma_upstream_max_embedding_dim, 4096)
        self.assertEqual(constants._offline_mma_storage_embedding_dim, 2048)
        self.assertEqual(early_module.MAX_EMBEDDING_DIM, 2048)

        adapter.config["embedding_dim"] = 1024
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            return_value=constants,
        ), self.assertRaisesRegex(RuntimeError, "cannot change"):
            adapter._install_embedding_dimension_compatibility()

    def test_embedding_storage_dimension_rejects_width_above_upstream_max(self) -> None:
        adapter = _adapter()
        adapter.config["embedding_dim"] = 8192
        constants = SimpleNamespace(MAX_EMBEDDING_DIM=4096)
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            return_value=constants,
        ), self.assertRaisesRegex(ValueError, "upstream maximum 4096"):
            adapter._install_embedding_dimension_compatibility()

    def test_nested_session_exception_propagates_and_outer_owner_closes(self) -> None:
        adapter = _adapter()
        from contextlib import contextmanager
        from sqlalchemy.orm import Session, sessionmaker

        class TrackingSession(Session):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                return super().close()

        class ExpectedFailure(RuntimeError):
            pass

        session_factory = sessionmaker(class_=TrackingSession)
        @contextmanager
        def upstream_db_context():
            session = session_factory()
            try:
                yield session
            finally:
                session.close()

        module = SimpleNamespace(
            SessionLocal=session_factory,
            db_context=upstream_db_context,
        )
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            return_value=module,
        ):
            adapter._install_sqlalchemy_session_compatibility()

        with self.assertRaisesRegex(ExpectedFailure, "database operation failed"):
            with module.db_context() as session:
                with session:
                    self.assertEqual(session.close_calls, 0)
                    raise ExpectedFailure("database operation failed")

        self.assertEqual(session._offline_mma_context_depth, 0)
        self.assertEqual(session.close_calls, 1)

    def test_real_orm_object_stays_attached_across_three_nested_updates(self) -> None:
        adapter = _adapter()
        from contextlib import contextmanager
        from sqlalchemy import Column, Integer, String, create_engine, inspect, select
        from sqlalchemy.orm import Session, declarative_base, sessionmaker

        base = declarative_base()

        class MemoryRow(base):
            __tablename__ = "memory_row"
            id = Column(Integer, primary_key=True)
            details = Column(String)

        engine = create_engine("sqlite:///:memory:")

        class TrackingSession(Session):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.close_calls = 0

            def close(self):
                self.close_calls += 1
                return super().close()

        session_factory = sessionmaker(
            bind=engine,
            class_=TrackingSession,
            expire_on_commit=True,
        )
        base.metadata.create_all(engine)
        @contextmanager
        def upstream_db_context():
            session = session_factory()
            try:
                yield session
            finally:
                session.close()

        module = SimpleNamespace(
            SessionLocal=session_factory,
            db_context=upstream_db_context,
        )
        with patch(
            "benchmarks.baseline_runtime.adapters.mma_original.importlib.import_module",
            return_value=module,
        ):
            adapter._install_sqlalchemy_session_compatibility()

        row = MemoryRow(details="initial")
        with module.db_context() as session:
            session.add(row)
            session.commit()
            session.refresh(row)
            for index in range(3):
                with session as helper_session:
                    row.details = f"update-{index}"
                    helper_session.add(row)
                    helper_session.commit()
                    helper_session.refresh(row)
                self.assertFalse(inspect(row).detached)
                self.assertEqual(session.close_calls, 0)
                self.assertEqual(
                    session.scalar(select(MemoryRow.details).where(MemoryRow.id == row.id)),
                    f"update-{index}",
                )

        self.assertTrue(inspect(row).detached)
        self.assertEqual(session.close_calls, 1)
        engine.dispose()

    def test_textual_tool_call_markup_is_rejected_not_repaired(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='<tool_call>{"name":"core_memory_append"}</tool_call>',
                        tool_calls=[],
                    )
                )
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "textual parsing and repair"):
            _reject_unstructured_native_tool_calls(response)

    def test_structured_native_tool_call_is_accepted(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "send_message"}}],
                    }
                }
            ]
        }

        _reject_unstructured_native_tool_calls(response)

    def test_chat_answer_is_validated_not_rewritten(self) -> None:
        self.assertEqual(_require_answer_block("<answer>yes</answer>"), "<answer>yes</answer>")
        with self.assertRaisesRegex(RuntimeError, "response rewriting is forbidden"):
            _require_answer_block("yes")

    def test_retry_usage_is_accumulated(self) -> None:
        self.assertEqual(
            _sum_usage(
                {"prompt_tokens": 10, "completion_tokens": 2},
                {"prompt_tokens": 12, "completion_tokens": 3},
            ),
            {"prompt_tokens": 22, "completion_tokens": 5},
        )

    def test_embedding_helper_preserves_the_legacy_tuple_contract(self) -> None:
        original = Mock(return_value=({"summary": [1.0]}, "config"))

        result = _prepare_embeddings_from_config_compat(
            original, "config", {"summary": "text"}
        )

        self.assertEqual(result, ({"summary": [1.0]}, "config"))
        original.assert_called_once_with("config", {"summary": "text"})

    def test_source_date_is_represented_at_midnight_for_mma(self) -> None:
        self.assertEqual(
            _mma_observation_timestamp("2024-07-27"),
            "2024-07-27 00:00:00",
        )
        self.assertEqual(
            _mma_observation_timestamp("2024-07-27 12:34:56"),
            "2024-07-27 12:34:56",
        )

    def test_embedding_helper_bridges_the_episodic_keyword_contract(self) -> None:
        original = Mock(
            return_value=({"summary": [1.0], "details": [2.0]}, "config")
        )

        result = _prepare_embeddings_from_config_compat(
            original,
            embedding_config="config",
            texts={"summary": "short", "details": "long"},
            existing_embeddings={
                "summary_embedding": [9.0],
                "details_embedding": None,
            },
        )

        self.assertEqual(result["summary_embedding"], [9.0])
        self.assertEqual(result["details_embedding"], [2.0])
        original.assert_called_once_with(
            "config", {"summary": "short", "details": "long"}
        )

    def test_end_session_never_sends_an_empty_flush(self) -> None:
        adapter = _adapter()
        adapter.backend = SimpleNamespace(
            send_message=Mock(),
            temp_message_accumulator=SimpleNamespace(temporary_messages=[]),
        )

        adapter.end_session("D1")

        adapter.backend.send_message.assert_not_called()

    def test_retrieval_returns_global_top7_with_structured_provenance(self) -> None:
        adapter = _adapter()
        hits = [
            SimpleNamespace(
                id=f"sem-{index}",
                name=f"memory {index}",
                summary=f"summary {index}",
                details=f"details {index}",
                details_embedding=[1.0, index / 100.0],
            )
            for index in range(9)
        ]
        semantic_manager = SimpleNamespace(
            list_semantic_items=Mock(return_value=hits)
        )
        empty = Mock(return_value=[])
        server = SimpleNamespace(
            episodic_memory_manager=SimpleNamespace(list_episodic_memory=empty),
            semantic_memory_manager=semantic_manager,
            procedural_memory_manager=SimpleNamespace(list_procedures=empty),
            resource_memory_manager=SimpleNamespace(list_resources=empty),
            knowledge_vault_manager=SimpleNamespace(list_knowledge=empty),
            user_manager=SimpleNamespace(
                get_user_by_id=lambda _user_id: SimpleNamespace(timezone="UTC")
            ),
        )
        states = SimpleNamespace(
            agent_state=SimpleNamespace(),
            episodic_memory_agent_state=SimpleNamespace(),
            semantic_memory_agent_state=SimpleNamespace(),
            procedural_memory_agent_state=SimpleNamespace(),
            resource_memory_agent_state=SimpleNamespace(),
            knowledge_vault_agent_state=SimpleNamespace(),
        )
        adapter.backend = SimpleNamespace(
            client=SimpleNamespace(
                server=server,
                user=SimpleNamespace(id="user"),
            ),
            agent_states=states,
        )
        for index in range(9):
            adapter.provenance.register(
                f"semantic_memory_manager:sem-{index}",
                Chunk(
                    chunk_id=f"dataset:D1:{index}",
                    text="source",
                    metadata={
                        "session_id": "D1",
                        "dialogue_id": f"D1:{index}",
                        "image_ids": [f"image-{index}"],
                    },
                ),
            )

        result = adapter.retrieve(
            RetrievalRequest(
                query_id="q1",
                text="question",
                top_k=7,
                query_vector=[1.0, 0.0],
            )
        )

        self.assertEqual(len(result.items), 7)
        self.assertEqual(result.items[0].source_dialogue_ids, ["D1:0"])
        self.assertEqual(result.items[0].image_ids, ["image-0"])
        self.assertTrue(result.trace["structured_evidence"])
        self.assertFalse(result.trace["qa_prompt_applied"])

    def test_retrieval_skips_native_bad_point_without_rank_fallback(self) -> None:
        adapter = _adapter()
        hits = [
            SimpleNamespace(
                id="bad",
                name="bad memory",
                summary="missing native vector",
                details="missing native vector",
                details_embedding=None,
            ),
            SimpleNamespace(
                id="good",
                name="good memory",
                summary="valid native memory",
                details="valid native memory",
                details_embedding=[1.0, 0.0],
            ),
        ]
        empty = Mock(return_value=[])
        server = SimpleNamespace(
            episodic_memory_manager=SimpleNamespace(list_episodic_memory=empty),
            semantic_memory_manager=SimpleNamespace(
                list_semantic_items=Mock(return_value=hits)
            ),
            procedural_memory_manager=SimpleNamespace(list_procedures=empty),
            resource_memory_manager=SimpleNamespace(list_resources=empty),
            knowledge_vault_manager=SimpleNamespace(list_knowledge=empty),
            user_manager=SimpleNamespace(
                get_user_by_id=lambda _user_id: SimpleNamespace(timezone="UTC")
            ),
        )
        states = SimpleNamespace(
            agent_state=SimpleNamespace(),
            episodic_memory_agent_state=SimpleNamespace(),
            semantic_memory_agent_state=SimpleNamespace(),
            procedural_memory_agent_state=SimpleNamespace(),
            resource_memory_agent_state=SimpleNamespace(),
            knowledge_vault_agent_state=SimpleNamespace(),
        )
        adapter.backend = SimpleNamespace(
            client=SimpleNamespace(
                server=server,
                user=SimpleNamespace(id="user"),
            ),
            agent_states=states,
        )
        for memory_id in ("bad", "good"):
            adapter.provenance.register(
                f"semantic_memory_manager:{memory_id}",
                Chunk(
                    chunk_id=f"dataset:D1:{memory_id}",
                    text="source",
                    metadata={"session_id": "D1", "dialogue_id": memory_id},
                ),
            )

        result = adapter.retrieve(
            RetrievalRequest(
                query_id="q-bad-point",
                text="question",
                top_k=7,
                query_vector=[1.0, 0.0],
            )
        )

        self.assertEqual(
            [item.memory_id for item in result.items],
            ["semantic_memory_manager:good"],
        )
        self.assertEqual(result.trace["skipped_bad_memory_point_count"], 1)
        self.assertEqual(
            result.trace["skipped_bad_memory_points"][0]["memory_id"],
            "semantic_memory_manager:bad",
        )
        self.assertEqual(result.trace["unique_bad_memory_point_count"], 1)
        self.assertEqual(result.trace["unique_bad_retrieval_point_count"], 1)
        self.assertEqual(result.trace["consecutive_bad_retrieval_points"], 1)

    def test_ten_bad_rows_in_one_retrieval_count_as_one_bad_point(self) -> None:
        adapter = _adapter()
        hits = [
            SimpleNamespace(id=f"bad-{index}", details_embedding=None)
            for index in range(MAX_CONSECUTIVE_BAD_MEMORY_POINTS)
        ]
        empty = Mock(return_value=[])
        server = SimpleNamespace(
            episodic_memory_manager=SimpleNamespace(list_episodic_memory=empty),
            semantic_memory_manager=SimpleNamespace(
                list_semantic_items=Mock(return_value=hits)
            ),
            procedural_memory_manager=SimpleNamespace(list_procedures=empty),
            resource_memory_manager=SimpleNamespace(list_resources=empty),
            knowledge_vault_manager=SimpleNamespace(list_knowledge=empty),
            user_manager=SimpleNamespace(
                get_user_by_id=lambda _user_id: SimpleNamespace(timezone="UTC")
            ),
        )
        adapter.backend = SimpleNamespace(
            client=SimpleNamespace(server=server, user=SimpleNamespace(id="user")),
            agent_states=SimpleNamespace(
                agent_state=SimpleNamespace(),
                episodic_memory_agent_state=SimpleNamespace(),
                semantic_memory_agent_state=SimpleNamespace(),
                procedural_memory_agent_state=SimpleNamespace(),
                resource_memory_agent_state=SimpleNamespace(),
                knowledge_vault_agent_state=SimpleNamespace(),
            ),
        )
        for hit in hits:
            adapter.provenance.register(
                f"semantic_memory_manager:{hit.id}",
                Chunk(
                    chunk_id=f"dataset:D1:{hit.id}",
                    text="source",
                    metadata={"session_id": "D1", "dialogue_id": hit.id},
                ),
            )

        result = adapter.retrieve(
            RetrievalRequest(
                query_id="q-ten-bad-points",
                text="question",
                top_k=7,
                query_vector=[1.0, 0.0],
            )
        )

        self.assertEqual(result.items, [])
        self.assertEqual(result.trace["skipped_bad_memory_point_count"], 10)
        self.assertEqual(result.trace["consecutive_bad_retrieval_points"], 1)

    def test_ten_consecutive_malformed_retrieval_requests_stop_sample(self) -> None:
        adapter = _adapter()
        hit = SimpleNamespace(id="bad", details_embedding=None)
        empty = Mock(return_value=[])
        server = SimpleNamespace(
            episodic_memory_manager=SimpleNamespace(list_episodic_memory=empty),
            semantic_memory_manager=SimpleNamespace(
                list_semantic_items=Mock(return_value=[hit])
            ),
            procedural_memory_manager=SimpleNamespace(list_procedures=empty),
            resource_memory_manager=SimpleNamespace(list_resources=empty),
            knowledge_vault_manager=SimpleNamespace(list_knowledge=empty),
            user_manager=SimpleNamespace(
                get_user_by_id=lambda _user_id: SimpleNamespace(timezone="UTC")
            ),
        )
        adapter.backend = SimpleNamespace(
            client=SimpleNamespace(server=server, user=SimpleNamespace(id="user")),
            agent_states=SimpleNamespace(
                agent_state=SimpleNamespace(),
                episodic_memory_agent_state=SimpleNamespace(),
                semantic_memory_agent_state=SimpleNamespace(),
                procedural_memory_agent_state=SimpleNamespace(),
                resource_memory_agent_state=SimpleNamespace(),
                knowledge_vault_agent_state=SimpleNamespace(),
            ),
        )
        adapter.provenance.register(
            "semantic_memory_manager:bad",
            Chunk(
                chunk_id="dataset:D1:bad",
                text="source",
                metadata={"session_id": "D1", "dialogue_id": "bad"},
            ),
        )

        for index in range(MAX_CONSECUTIVE_BAD_MEMORY_POINTS - 1):
            result = adapter.retrieve(
                RetrievalRequest(
                    query_id=f"q-bad-{index}",
                    text="question",
                    top_k=7,
                    query_vector=[1.0, 0.0],
                )
            )
            self.assertEqual(
                result.trace["consecutive_bad_retrieval_points"], index + 1
            )
        with self.assertRaisesRegex(RuntimeError, "10 consecutive malformed retrieval"):
            adapter.retrieve(
                RetrievalRequest(
                    query_id="q-bad-final",
                    text="question",
                    top_k=7,
                    query_vector=[1.0, 0.0],
                )
            )

    def test_failed_qa_attempts_count_as_one_bad_point_and_continue(self) -> None:
        adapter = _adapter()
        adapter.config = {"retries": 2}
        retrieval = RetrievalResult(items=[])
        adapter._retrievals["q-bad-answer"] = retrieval
        adapter.backend = SimpleNamespace(
            client=SimpleNamespace(),
            agent_states=SimpleNamespace(agent_state=SimpleNamespace(id="chat")),
        )
        request = NativeAnswerRequest(
            query_id="q-bad-answer",
            messages=[{"role": "user", "content": "question"}],
            retrieval=retrieval,
            top_k=7,
        )
        response = SimpleNamespace(usage={})
        with (
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._capture_chat_state",
                return_value={},
            ),
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._restore_chat_state"
            ),
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._fixed_chat_memory_prompt",
                return_value=__import__("contextlib").nullcontext(),
            ),
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._send_native_benchmark_messages",
                return_value=(response, 0),
            ) as send,
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._extract_chat_answer",
                return_value="",
            ),
        ):
            result = adapter.answer_with_memory(request)

        self.assertEqual(send.call_count, 3)
        self.assertEqual(result.text, "")
        self.assertEqual(result.failed_attempts, 3)
        self.assertTrue(result.trace["bad_qa_point"])
        self.assertEqual(result.trace["consecutive_bad_qa_points"], 1)

    def test_ten_consecutive_failed_qa_points_stop_sample(self) -> None:
        adapter = _adapter()
        for index in range(MAX_CONSECUTIVE_BAD_QA_POINTS - 1):
            self.assertEqual(
                adapter._record_bad_qa_point(
                    f"q-{index}", RuntimeError("truncated")
                ),
                index + 1,
            )
        self.assertEqual(
            adapter._record_bad_qa_point(
                "q-final", RuntimeError("truncated")
            ),
            MAX_CONSECUTIVE_BAD_QA_POINTS,
        )
        self.assertEqual(adapter._consecutive_bad_qa_points, 10)

    def test_output_length_retry_exhaustion_is_one_bad_qa_point(self) -> None:
        adapter = _adapter()
        adapter.config = {"retries": 2}
        retrieval = RetrievalResult(items=[])
        adapter._retrievals["q-output-length"] = retrieval
        adapter.backend = SimpleNamespace(
            client=SimpleNamespace(),
            agent_states=SimpleNamespace(agent_state=SimpleNamespace(id="chat")),
        )
        request = NativeAnswerRequest(
            query_id="q-output-length",
            messages=[{"role": "user", "content": "question"}],
            retrieval=retrieval,
            top_k=7,
        )
        exhausted = RuntimeError(
            "Retries exhausted and no valid response received. Final error: "
            "maximum context length exceeded or generated content is too long"
        )
        with (
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._capture_chat_state",
                return_value={},
            ),
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._restore_chat_state"
            ),
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._fixed_chat_memory_prompt",
                return_value=__import__("contextlib").nullcontext(),
            ),
            patch(
                "benchmarks.baseline_runtime.adapters.mma_original._send_native_benchmark_messages",
                side_effect=exhausted,
            ) as send,
        ):
            result = adapter.answer_with_memory(request)

        self.assertEqual(send.call_count, 1)
        self.assertEqual(result.text, "")
        self.assertEqual(result.failed_attempts, 3)
        self.assertEqual(result.trace["consecutive_bad_qa_points"], 1)

    def test_selected_memory_context_keeps_individual_structured_rows(self) -> None:
        items = [
            RetrievedMemory(
                memory_id="semantic_memory_manager:s1",
                text='{"memory_type":"semantic_memory","id":"s1"}',
                metadata={"partition": "semantic_memory"},
            ),
            RetrievedMemory(
                memory_id="episodic_memory_manager:e1",
                text='{"memory_type":"episodic_memory","id":"e1"}',
                metadata={"partition": "episodic_memory"},
            ),
        ]
        messages = [
            {"role": "system", "content": "benchmark QA system"},
            {"role": "user", "content": "benchmark question"},
        ]

        context = _selected_memory_context(items, messages)

        self.assertIn('"id":"s1"', context["semantic"])
        self.assertIn('"id":"e1"', context["episodic"][1])
        self.assertEqual(context["key_words"], "benchmark question")

    def test_snapshot_schema_records_embedding_presence_without_vectors(self) -> None:
        structured = _structured_memory(
            SimpleNamespace(
                id="s1",
                name="fact",
                details="a structured fact",
                details_embedding=[0.1, 0.2, 0.3],
            ),
            "semantic_memory",
        )
        self.assertEqual(structured["memory_type"], "semantic_memory")
        self.assertEqual(structured["embedding_fields"], {"details_embedding": 3})
        self.assertNotIn("details_embedding", structured)


if __name__ == "__main__":
    unittest.main()
