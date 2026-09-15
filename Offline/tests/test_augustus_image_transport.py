from __future__ import annotations

from types import SimpleNamespace
import unittest

from benchmarks.baseline_runtime.adapters.memengine import MemEngineAdapter
from benchmarks.baseline_runtime.protocol import RetrievalRequest, result_context_items
from benchmarks.memgallery_harness.runner.answer_client import (
    build_retrieved_memory_context,
)
from embedding.chunk_builder import Chunk


class AugustusImageTransportTest(unittest.TestCase):
    def test_common_top_k_caps_augustus_final_node_selection(self):
        adapter = object.__new__(MemEngineAdapter)
        adapter.config = {"top_k": 7}
        native_config = {
            "recall": {"max_nodes": 10},
            "concept_retrieval": {"topk": 10},
        }

        adapter._apply_common_config(native_config)

        self.assertEqual(native_config["recall"]["max_nodes"], 7)
        self.assertEqual(native_config["concept_retrieval"]["topk"], 7)

    def test_native_recall_keeps_each_nodes_image_for_answer_client(self):
        adapter = object.__new__(MemEngineAdapter)
        adapter.baseline = "AUGUSTUSMemory"
        adapter._chunks = [
            Chunk(
                chunk_id="D1:1",
                text="first round",
                images=["/tmp/first.jpg"],
                metadata={
                    "dialogue_id": "D1:1",
                    "session_id": "D1",
                    "image_ids": ["D1:IMG_001"],
                },
            ),
            Chunk(
                chunk_id="D1:2",
                text="second round",
                images=["/tmp/second.jpg"],
                metadata={
                    "dialogue_id": "D1:2",
                    "session_id": "D1",
                    "image_ids": ["D1:IMG_002"],
                },
            ),
        ]
        adapter.memory = SimpleNamespace(
            recall=lambda _query: [
                {
                    "text": "[Memory 0] first round",
                    "image": {
                        "path": "/tmp/first.jpg",
                        "img_id": "D1:IMG_001",
                    },
                    "timestamp": "2025-01-01",
                    "dialogue_id": "D1:1",
                },
                {
                    "text": "[Memory 1] second round",
                    "image": "/tmp/second.jpg",
                    "timestamp": "2025-01-02",
                    "dialogue_id": "D1:2",
                },
            ]
        )

        result = adapter.retrieve(
            RetrievalRequest(query_id="q1", text="question", category="VS")
        )

        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].source_dialogue_ids, ["D1:1"])
        self.assertEqual(result.items[0].image_paths, ["/tmp/first.jpg"])
        self.assertEqual(result.items[1].image_paths, ["/tmp/second.jpg"])
        memory_items = result_context_items(result)
        _, answer_image_paths = build_retrieved_memory_context(memory_items, "VS")
        self.assertEqual(
            answer_image_paths,
            ["/tmp/first.jpg", "/tmp/second.jpg"],
        )


if __name__ == "__main__":
    unittest.main()
