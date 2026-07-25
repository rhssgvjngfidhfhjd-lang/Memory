import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from agentmem.executor import ExecutionResult, MemoryExecutor
from agentmem.memory_bank import MemoryBank
from agentmem.memgallery.builder import MemGalleryMemoryBuilder
from agentmem.memgallery.input_loader import MemoryEvent
from agentmem.memgallery.metrics import provenance_hit
from agentmem.operations import get_operations
from memgallery_harness.runner.ollama_client import OllamaVLMClient


class FakeEmbedder:
    def embed_texts(self, texts, mode="context"):
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        result = np.asarray([[float(len(text)), 1.0] for text in values], dtype=np.float32)
        return result[0] if single else result


class ProvenanceMemoryBankTest(unittest.TestCase):
    def test_update_merges_provenance_and_persists(self):
        bank = MemoryBank()
        bank.add_memory(
            "old",
            np.asarray([1.0, 0.0]),
            metadata={"source_dialogue_ids": ["D1:1"], "source_chunk_ids": ["x:D1:1"]},
        )
        executor = MemoryExecutor(backend=None, embedder=FakeEmbedder())
        executor.apply_to_memory_bank(
            [ExecutionResult("UPDATE", True, memory_index=0, memory_content="new")],
            bank,
            [0],
            event_metadata={
                "source_dialogue_ids": ["D1:2"],
                "source_chunk_ids": ["x:D1:2"],
                "image_ids": ["D1:IMG_1"],
            },
        )
        self.assertEqual(bank.memories[0].metadata["source_dialogue_ids"], ["D1:1", "D1:2"])
        with tempfile.TemporaryDirectory() as directory:
            bank.save(directory)
            loaded = MemoryBank.load(directory)
        self.assertEqual(loaded.memories[0].memory_id, bank.memories[0].memory_id)
        self.assertEqual(loaded.memories[0].metadata, bank.memories[0].metadata)

    def test_insert_inherits_event_metadata(self):
        bank = MemoryBank()
        executor = MemoryExecutor(backend=None, embedder=FakeEmbedder())
        executor.apply_to_memory_bank(
            [ExecutionResult("INSERT", True, memory_content="fact")],
            bank,
            [],
            event_metadata={"source_dialogue_ids": ["D2:3"]},
        )
        self.assertEqual(bank.memories[0].metadata["source_dialogue_ids"], ["D2:3"])

    def test_hit_uses_top_k_memory_groups(self):
        groups = [["D1:1", "D1:2"], ["D2:1"]]
        self.assertEqual(provenance_hit(groups, ["D1:2"], k=1), 1.0)
        self.assertEqual(provenance_hit(groups, ["D2:1"], k=1), 0.0)

    def test_duplicate_insert_merges_provenance_instead_of_dropping_it(self):
        bank = MemoryBank()
        executor = MemoryExecutor(backend=None, embedder=FakeEmbedder())
        executor.apply_to_memory_bank(
            [ExecutionResult("INSERT", True, memory_content="Alice likes coffee")],
            bank,
            [],
            event_metadata={"source_dialogue_ids": ["D1:1"], "source_chunk_ids": ["x:D1:1"]},
        )
        executor.apply_to_memory_bank(
            [ExecutionResult("INSERT", True, memory_content="Alice likes coffee")],
            bank,
            [],
            event_metadata={"source_dialogue_ids": ["D1:5"], "source_chunk_ids": ["x:D1:5"]},
        )
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank.memories[0].metadata["source_dialogue_ids"], ["D1:1", "D1:5"])


class CrashingBackend:
    def __init__(self, crash_at):
        self.n = 0
        self.crash_at = crash_at

    def generate(self, prompt):
        self.n += 1
        if self.n - 1 == self.crash_at:
            raise RuntimeError("simulated crash")
        return f"ACTION: INSERT\nMEMORY_ITEM: fact number {self.n}"


class BuilderResumeTest(unittest.TestCase):
    def test_resume_after_mid_run_crash_does_not_duplicate_trace_entries(self):
        events = [
            MemoryEvent(
                text=f"event {i}",
                dataset="d",
                dialogue_id=f"D1:{i}",
                session_id="D1",
                round_id=i,
                source_chunk_id=f"c{i}",
            )
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            builder = MemGalleryMemoryBuilder(CrashingBackend(crash_at=4), FakeEmbedder(), top_k=5)
            with self.assertRaises(RuntimeError):
                builder.build(events, out, resume=True, checkpoint_every=3)

            builder_resumed = MemGalleryMemoryBuilder(CrashingBackend(crash_at=999), FakeEmbedder(), top_k=5)
            builder_resumed.build(events, out, resume=True, checkpoint_every=3)

            with (out / "build_trace.jsonl").open() as handle:
                indices = [json.loads(line)["event_index"] for line in handle]
        self.assertEqual(indices, [0, 1, 2, 3, 4])


class ScriptedBackend:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt):
        self.last_prompt = prompt
        return self.response


class OperationTogglesTest(unittest.TestCase):
    def test_insert_only_prompt_omits_disabled_blocks_and_mandates_insert(self):
        executor = MemoryExecutor(backend=None, embedder=FakeEmbedder())
        prompt = executor._build_prompt(get_operations(["insert"]), "chunk", ["m0"])
        self.assertIn("INSERT block:", prompt)
        self.assertNotIn("UPDATE block:", prompt)
        self.assertNotIn("DELETE block:", prompt)
        self.assertNotIn("NOOP block:", prompt)
        self.assertIn("MUST output at least one INSERT block", prompt)

    def test_disabled_action_from_model_is_rejected(self):
        backend = ScriptedBackend("ACTION: UPDATE\nMEMORY_INDEX: 0\nUPDATED_MEMORY: changed")
        executor = MemoryExecutor(backend=backend, embedder=FakeEmbedder())
        _, actions = executor.execute(
            operations=get_operations(["insert"]),
            chunk_text="chunk",
            retrieved_memories=["m0"],
        )
        self.assertEqual(len(actions), 1)
        self.assertFalse(actions[0].success)
        self.assertIn("not enabled", actions[0].reasoning)

    def test_insert_only_builder_falls_back_to_raw_chunk_on_unusable_response(self):
        events = [
            MemoryEvent(
                text="event zero",
                dataset="d",
                dialogue_id="D1:0",
                session_id="D1",
                round_id=0,
                source_chunk_id="c0",
            )
        ]
        builder = MemGalleryMemoryBuilder(
            ScriptedBackend("gibberish with no action"),
            FakeEmbedder(),
            top_k=5,
            operations=get_operations(["insert"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            stats = builder.build(events, Path(directory) / "out", resume=False)
            bank = MemoryBank.load(Path(directory) / "out")
        self.assertEqual(stats["fallback_inserts_this_run"], 1)
        self.assertEqual(len(bank), 1)
        self.assertEqual(bank.memories[0].content, "event zero")
        self.assertEqual(bank.memories[0].metadata["source_dialogue_ids"], ["D1:0"])
        self.assertEqual(bank.memories[0].metadata["source"], "fallback_insert")


class MemoryEventMetadataTest(unittest.TestCase):
    def test_metadata_includes_singular_keys_for_answer_prompt_formatter(self):
        event = MemoryEvent(
            text="hello",
            dataset="d",
            dialogue_id="D1:3",
            session_id="D1",
            round_id=3,
            source_chunk_id="c3",
            image_ids=["D1:IMG_001"],
        )
        metadata = event.metadata
        self.assertEqual(metadata["dialogue_id"], "D1:3")
        self.assertEqual(metadata["image_id"], "D1:IMG_001")


class AnswerPromptImageIdLeakTest(unittest.TestCase):
    def setUp(self):
        self.client = OllamaVLMClient()
        self.memory = {
            "text": "A robot painting at an easel (image_id: D6:IMG_001).",
            "image": None,
            "metadata": {
                "session_id": "D6",
                "dialogue_id": "D6:1",
                "image_id": "D6:IMG_001",
            },
        }

    def test_unattached_memory_id_is_redacted_for_visual_question(self):
        text, paths = self.client._build_text_and_image_paths(
            [self.memory], "question", {"path": "/tmp/query.png"}, "VS"
        )
        self.assertNotIn("D6:IMG_001", text)
        self.assertIn("[IMAGE_ID_REDACTED]", text)
        self.assertEqual(paths, ["/tmp/query.png"])

    def test_attached_memory_image_keeps_its_candidate_id(self):
        memory = {**self.memory, "image": {"path": "/tmp/memory.png", "img_id": "D6:IMG_001"}}
        text, paths = self.client._build_text_and_image_paths(
            [memory], "question", {"path": "/tmp/query.png"}, "VS"
        )
        self.assertIn("IMG:D6:IMG_001", text)
        self.assertIn("Attached memory image 1: D6:IMG_001", text)
        self.assertEqual(paths, ["/tmp/memory.png", "/tmp/query.png"])

    def test_nonvisual_question_does_not_expose_unattached_candidate_id(self):
        memory = {**self.memory, "image": {"path": "/tmp/memory.png", "img_id": "D6:IMG_001"}}
        text, paths = self.client._build_text_and_image_paths([memory], "question", None, "TTL")
        self.assertNotIn("D6:IMG_001", text)
        self.assertEqual(paths, [])


if __name__ == "__main__":
    unittest.main()
