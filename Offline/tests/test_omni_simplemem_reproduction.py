from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from benchmarks.baseline_runtime.adapters.omni_simplemem import OmniSimpleMemAdapter
from benchmarks.baseline_runtime.omni_inputs import (
    build_omni_memgallery_chunks,
    build_omni_wma_chunks_from_data,
)
from benchmarks.baseline_runtime.provenance import ProvenanceIndex
from embedding.chunk_builder import Chunk


class OmniSimpleMemReproductionTest(unittest.TestCase):
    def test_memgallery_mapping_is_one_source_grounded_round(self):
        dataset = {
            "character_profile": {"name": "Chloe"},
            "multi_session_dialogues": [
                {
                    "session_id": "D1",
                    "date": "2024-06-26",
                    "dialogues": [
                        {
                            "round": "D1:1",
                            "user": "question",
                            "assistant": "answer",
                            "image_id": ["D1:IMG_001"],
                            "input_image": ["../image/example.jpg"],
                            "image_caption": ["caption"],
                        }
                    ],
                }
            ],
        }
        chunks = build_omni_memgallery_chunks(dataset, "/dataset", "sample")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].text,
            "user (Chloe): question\nassistant: answer\nimage_caption: caption",
        )
        self.assertNotIn("previous_round_summary", chunks[0].text)
        self.assertEqual(
            chunks[0].metadata["omni_input_mode"],
            "multimodal_source_round",
        )

    def test_wma_mapping_emits_source_turns_instead_of_shared_chunks(self):
        sample = {
            "sample_id": "personal_01",
            "sessions": [
                {
                    "_v2_session_id": "S00",
                    "dialogue": [
                        {
                            "role": "user",
                            "content": "remember this",
                            "timestamp": "now",
                            "attachments": [],
                        },
                        {
                            "role": "assistant",
                            "content": "remembered",
                            "timestamp": "later",
                            "attachments": [],
                        },
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            sample_path = Path(temporary) / "personal_01.json"
            chunks = build_omni_wma_chunks_from_data(
                sample, temporary, sample_path=sample_path
            )
        self.assertEqual([row.text for row in chunks], ["user: remember this", "assistant: remembered"])
        self.assertTrue(
            all(
                row.metadata["omni_input_mode"] == "multimodal_source_turn"
                for row in chunks
            )
        )

    def test_ingest_does_not_force_official_text_filter(self):
        calls = []
        result = SimpleNamespace(
            success=True,
            skipped=False,
            error=None,
            mau=SimpleNamespace(id="mau-1"),
        )

        def add_text(*args, **kwargs):
            calls.append((args, kwargs))
            return result

        adapter = object.__new__(OmniSimpleMemAdapter)
        adapter.backend = SimpleNamespace(
            start_session=lambda _session: None,
            end_session=lambda: None,
            add_text=add_text,
        )
        adapter.provenance = ProvenanceIndex()
        adapter._current_session = ""
        adapter._ingest_stats = {"stored": 0, "skipped": 0, "failed": 0}
        adapter.ingest(
            Chunk(
                chunk_id="D1:1",
                text="a sufficiently long observation",
                images=[],
                metadata={
                    "session_id": "D1",
                    "dialogue_id": "D1:1",
                    "omni_input_mode": "multimodal_source_round",
                },
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertNotIn("force", calls[0][1])
        self.assertEqual(adapter._ingest_stats, {"stored": 1, "skipped": 0, "failed": 0})


if __name__ == "__main__":
    unittest.main()
