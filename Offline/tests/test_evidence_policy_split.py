from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from evidence_policy.episode_sources import iter_source_questions
from evidence_policy.split_manifest import SplitManifestIndex, normalize_split_name


def write_manifest(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "split_unit": "conversation",
        "datasets": [
            {
                "data_source": "toy",
                "splits": {
                    "train": {
                        "conversation_count": 1,
                        "question_count": 2,
                        "conversations": [
                            {
                                "conversation_id": "toy:c1",
                                "source_id": "c1",
                                "variant": "toy",
                                "question_ids": ["toy:c1:q1", "toy:c1:q2"],
                            }
                        ],
                    },
                    "val": {
                        "conversation_count": 1,
                        "question_count": 1,
                        "conversations": [
                            {
                                "conversation_id": "toy:c2",
                                "source_id": "c2",
                                "variant": "toy",
                                "question_ids": ["toy:c2:q1"],
                            }
                        ],
                    },
                    "test": {
                        "conversation_count": 1,
                        "question_count": 1,
                        "conversations": [
                            {
                                "conversation_id": "toy:c3",
                                "source_id": "c3",
                                "variant": "toy",
                                "question_ids": ["toy:c3:q1"],
                            }
                        ],
                    },
                },
            }
        ],
        "sha256": "test-only",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class SplitManifestIndexTest(unittest.TestCase):
    def test_validation_alias_maps_to_val(self):
        self.assertEqual(normalize_split_name("validation"), "val")
        self.assertEqual(normalize_split_name("valid"), "val")

    def test_indexes_conversations_and_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path)
            index = SplitManifestIndex(path)
            self.assertEqual(index.source_ids("train", "toy"), ("c1",))
            self.assertEqual(index.source_ids("validation", "toy"), ("c2",))
            self.assertTrue(index.contains_question("validation", "toy", "toy:c2:q1"))
            self.assertFalse(index.contains_question("test", "toy", "toy:c2:q1"))
            self.assertEqual(index.split_for_question("toy:c3:q1"), "test")
            summary = index.summary()
            self.assertEqual(summary["splits"]["train"]["question_count"], 2)
            self.assertEqual(summary["splits"]["val"]["conversation_count"], 1)

    def test_rejects_cross_split_conversation_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_manifest(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["datasets"][0]["splits"]["test"]["conversations"][0][
                "conversation_id"
            ] = "toy:c1"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "appears in both"):
                SplitManifestIndex(path)


@unittest.skipUnless(
    os.environ.get("MULTIMODAL_SPLIT_MANIFEST"),
    "Set MULTIMODAL_SPLIT_MANIFEST for the full local-source integration test",
)
class FullLocalSplitIntegrationTest(unittest.TestCase):
    def test_manifest_matches_all_three_source_repositories(self):
        manifest = SplitManifestIndex(os.environ["MULTIMODAL_SPLIT_MANIFEST"])
        workspace_root = Path(__file__).resolve().parents[2]
        rows = list(iter_source_questions(manifest, workspace_root))
        self.assertEqual(len(rows), 5403)
        self.assertEqual(len({row.question_id for row in rows}), 5403)
        counts = {
            split: sum(row.split == split for row in rows)
            for split in ("train", "val", "test")
        }
        self.assertEqual(counts, {"train": 3268, "val": 1060, "test": 1075})


if __name__ == "__main__":
    unittest.main()
