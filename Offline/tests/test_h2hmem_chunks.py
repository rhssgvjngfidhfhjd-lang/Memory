from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from embedding.chunk_builder import (
    build_h2h_chunks_from_data,
    build_h2h_chunks_from_directory,
    iter_h2h_session_files,
)


def _session_payload() -> dict:
    return {
        "session_id": "native-pet-1",
        "session_title": "A shared photo",
        "theme": "pets",
        "timeline_date": "2024-01-15",
        "dialogue": [
            {
                "role": "Alice",
                "content": {"text": "Look at my cat.", "image": "1.png"},
            },
            {
                "role": "Alice",
                "content": {"text": "Her name is Almond.", "image": ""},
            },
            {
                "role": "Bob",
                "content": {"text": "She is lovely.", "image": ""},
            },
            {
                "role": "Carol",
                "content": {"text": "I agree.", "image": "2.png"},
            },
        ],
    }


class H2HMemChunkTest(unittest.TestCase):
    def test_consecutive_speaker_messages_merge_before_pairing(self):
        with tempfile.TemporaryDirectory() as directory:
            session_path = (
                Path(directory)
                / "dyadic"
                / "dialogue1"
                / "scenes"
                / "session1"
                / "session.json"
            )
            image_dir = session_path.parent / "image"
            image_dir.mkdir(parents=True)
            (image_dir / "1.png").touch()
            (image_dir / "2.png").touch()
            chunks = build_h2h_chunks_from_data(
                _session_payload(),
                session_path=session_path,
                variant="dyadic",
                conversation_id="dialogue1",
            )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["speaker_a"], "Alice")
        self.assertEqual(chunks[0].metadata["speaker_b"], "Bob")
        self.assertIn("speaker_a: Look at my cat.", chunks[0].text)
        self.assertIn("speaker_a: Her name is Almond.", chunks[0].text)
        self.assertEqual(chunks[0].metadata["session_id"], "session1")
        self.assertEqual(chunks[0].metadata["native_session_id"], "native-pet-1")
        self.assertEqual(chunks[1].metadata["speaker_a"], "Carol")
        self.assertEqual(chunks[1].metadata["speaker_b"], "")
        self.assertEqual(Path(chunks[0].images[0]).name, "1.png")
        self.assertEqual(Path(chunks[0].images[0]).parent.name, "image")
        self.assertNotIn("original_answer", json.dumps([row.to_dict() for row in chunks]))

    def test_directory_builder_orders_sessions_numerically_and_filters_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant_dir, session_name in (
                ("dyadic", "session10"),
                ("dyadic", "session2"),
                ("multi-party", "session1"),
            ):
                path = (
                    root
                    / variant_dir
                    / "dialogue1"
                    / "scenes"
                    / session_name
                    / "session.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(_session_payload()), encoding="utf-8")

            paths = iter_h2h_session_files(root, variant="dyadic")
            chunks = build_h2h_chunks_from_directory(root, variant="dyadic")

        self.assertEqual([path.parent.name for path in paths], ["session2", "session10"])
        self.assertEqual(
            [row.metadata["session_id"] for row in chunks],
            ["session2", "session2", "session10", "session10"],
        )
        self.assertTrue(all(row.metadata["variant"] == "dyadic" for row in chunks))


if __name__ == "__main__":
    unittest.main()
