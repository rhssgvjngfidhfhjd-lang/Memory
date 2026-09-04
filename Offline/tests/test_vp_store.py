from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from evidence_policy.vp_store import VPArtifactIndex


class VPArtifactIndexTest(unittest.TestCase):
    def test_resolves_by_relative_path_and_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "dataset" / "topic" / "image.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"source")
            run = root / "run"
            crop = run / "items" / "img_1" / "vp_0001.jpg"
            crop.parent.mkdir(parents=True)
            crop.write_bytes(b"crop")
            (run / "exports").mkdir()
            (run / "run.json").write_text(
                json.dumps({"schema_version": "1.0", "run_id": "test"}),
                encoding="utf-8",
            )
            record = {
                "schema_version": "1.0",
                "run_id": "test",
                "image_id": "img_1",
                "source": {
                    "dataset": "Demo",
                    "relative_path": "topic/image.jpg",
                    "sha256": hashlib.sha256(b"source").hexdigest(),
                },
                "status": "success",
                "primitives": [
                    {
                        "vp_id": "img_1_vp_0001",
                        "label": "subject",
                        "bbox_norm": [1, 2, 3, 4],
                        "crop_path": "items/img_1/vp_0001.jpg",
                    }
                ],
            }
            (run / "exports" / "images.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            index = VPArtifactIndex(run)

            by_path = index.primitives_for(image)
            copy = root / "renamed.jpg"
            copy.write_bytes(b"source")
            by_hash = index.primitives_for(copy)
            by_blob_sha = index.primitives_for(
                "/home/user/.cache/huggingface/hub/datasets--demo/blobs/"
                + hashlib.sha256(b"source").hexdigest()
            )

        self.assertEqual(by_path[0].vp_id, "img_1_vp_0001")
        self.assertEqual(by_hash, by_path)
        self.assertEqual(by_blob_sha, by_path)

    def test_audit_reports_missing_crop_without_eager_global_stat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run / "exports").mkdir(parents=True)
            (run / "run.json").write_text(
                json.dumps({"schema_version": "1.0", "run_id": "test"}),
                encoding="utf-8",
            )
            record = {
                "schema_version": "1.0",
                "run_id": "test",
                "image_id": "img_1",
                "source": {
                    "dataset": "Demo",
                    "relative_path": "topic/image.jpg",
                    "sha256": "",
                },
                "status": "success",
                "primitives": [
                    {
                        "vp_id": "img_1_vp_0001",
                        "label": "subject",
                        "bbox_norm": [1, 2, 3, 4],
                        "crop_path": "items/img_1/missing.jpg",
                    }
                ],
            }
            (run / "exports" / "images.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )

            index = VPArtifactIndex(run)
            report = index.audit(["topic/image.jpg"])

        self.assertEqual(report["matched_records"], 1)
        self.assertEqual(report["missing_crop_files"], 1)


if __name__ == "__main__":
    unittest.main()
