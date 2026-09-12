import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vp_extractor.io import (
    ArtifactStore,
    extraction_signatures_compatible,
    load_caption_map,
    scan_datasets,
    scan_path,
)
from vp_extractor.models import ExtractionResult, ImageRecord, PrimitiveRecord


class StorageTests(unittest.TestCase):
    def test_signature_allows_only_loopback_forwarding_port_move(self):
        existing = {
            "settings": {
                "vlm": {
                    "model": "model",
                    "base_url": "http://127.0.0.1:18001/v1",
                },
                "crop": {"quality": 95},
            }
        }
        moved = json.loads(json.dumps(existing))
        moved["settings"]["vlm"]["base_url"] = "http://localhost:18003/v1/"
        self.assertTrue(extraction_signatures_compatible(existing, moved))

        changed_model = json.loads(json.dumps(moved))
        changed_model["settings"]["vlm"]["model"] = "other"
        self.assertFalse(extraction_signatures_compatible(existing, changed_model))

        remote = json.loads(json.dumps(existing))
        remote["settings"]["vlm"]["base_url"] = "http://example.com:18003/v1"
        self.assertFalse(extraction_signatures_compatible(existing, remote))

    def test_scan_datasets_combines_configured_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for dataset, color in (("one", "blue"), ("two", "red")):
                image_dir = root / dataset / "images" / "sample"
                image_dir.mkdir(parents=True)
                Image.new("RGB", (4, 4), color).save(image_dir / f"{dataset}.png")
            specs = {
                "One": {
                    "root": "one",
                    "include": ["images/**/*", "**/images/**/*"],
                },
                "Two": {
                    "root": "two",
                    "include": ["images/**/*", "**/images/**/*"],
                },
            }
            sources = scan_datasets(("One", "Two"), specs, root)

        self.assertEqual([row.dataset for row in sources], ["One", "Two"])

    def test_loads_mem_gallery_dialog_and_attaches_caption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "images" / "D1_IMG_001.jpg"
            image_path.parent.mkdir()
            Image.new("RGB", (20, 10), "blue").save(image_path)
            dialog_path = root / "dialog.json"
            dialog_path.write_text(
                json.dumps(
                    {
                        "multi_session_dialogues": [
                            {
                                "dialogues": [
                                    {
                                        "input_image": ["../image/topic/D1_IMG_001.jpg"],
                                        "image_caption": ["A blue subject."],
                                    }
                                ]
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            captions = load_caption_map(dialog_path)
            source = scan_path(image_path.parent, "Mem-Gallery", captions)[0]

            self.assertEqual(source.caption, "A blue subject.")

    def test_scan_save_and_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "input" / "photo.jpg"
            source_path.parent.mkdir()
            Image.new("RGB", (20, 10), "blue").save(source_path)
            source = scan_path(source_path, "demo")[0]

            primitive = PrimitiveRecord(
                vp_id=f"{source.image_id}_vp_0001",
                label="blue rectangle",
                bbox_norm=(0, 0, 500, 1000),
                bbox_px=(0, 0, 10, 10),
                crop_path=f"items/{source.image_id}/vp_0001.jpg",
            )
            record = ImageRecord(
                schema_version="1.0",
                run_id="test_run",
                image_id=source.image_id,
                source={
                    "dataset": "demo",
                    "relative_path": "photo.jpg",
                    "sha256": "test",
                    "width": 20,
                    "height": 10,
                },
                status="success",
                primitives=(primitive,),
            )
            result = ExtractionResult(
                record=record,
                source_image=Image.new("RGB", (20, 10), "blue"),
                crops={primitive.vp_id: Image.new("RGB", (10, 10), "blue")},
            )
            store = ArtifactStore(root / "outputs", "test_run", create_preview=True)
            store.initialize({"settings": "test"})
            store.save(result)
            images_path, primitives_path = store.export()

            self.assertTrue(store.is_complete(source.image_id))
            self.assertTrue((store.root / primitive.crop_path).is_file())
            self.assertTrue((store.items_dir / source.image_id / "preview.jpg").is_file())
            image_rows = images_path.read_text(encoding="utf-8").splitlines()
            primitive_rows = primitives_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(image_rows), 1)
            self.assertEqual(len(primitive_rows), 1)
            self.assertEqual(json.loads(primitive_rows[0])["vp_id"], primitive.vp_id)


if __name__ == "__main__":
    unittest.main()
