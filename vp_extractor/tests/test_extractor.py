import tempfile
import unittest
from pathlib import Path

from PIL import Image

from vp_extractor.extractor import (
    VPExtractor,
    bbox_iou,
    deduplicate,
    norm_to_pixels,
    normalize_bbox,
    suppress_full_frame_parents,
)
from vp_extractor.models import ImageInput, PrimitiveCandidate, Settings


class FakeDiscoverer:
    def __init__(self):
        self.focus_context = None

    def discover(self, image, focus_context=None):
        self.focus_context = focus_context
        return [
            PrimitiveCandidate("green object", (100, 100, 600, 900)),
            PrimitiveCandidate("duplicate object", (110, 110, 590, 890)),
            PrimitiveCandidate("small cup", (900, 0, 900, 10)),
        ]

    def relocalize(self, image, candidate):
        if candidate.label == "small cup":
            return PrimitiveCandidate("small cup", (700, 100, 900, 300))
        return None


class ExtractorTests(unittest.TestCase):
    def test_bbox_geometry(self):
        self.assertEqual(normalize_bbox((-5, 10, 1005, 900)), (0, 10, 1000, 900))
        self.assertIsNone(normalize_bbox((10, 10, 10, 20)))
        self.assertEqual(norm_to_pixels((100, 100, 600, 900), 100, 50), (10, 5, 60, 45))
        self.assertGreater(bbox_iou((100, 100, 600, 900), (110, 110, 590, 890)), 0.85)

    def test_extracts_multiple_crops_and_relocalizes_invalid_box(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.jpg"
            Image.new("RGB", (100, 50), "white").save(path)
            source = ImageInput(path, "test", "source.jpg", "img_test")
            settings = Settings(model="fake", base_url="http://fake")

            discoverer = FakeDiscoverer()
            result = VPExtractor(discoverer, settings).extract_image(source)

            self.assertEqual(result.record.status, "success")
            self.assertEqual(len(result.record.primitives), 2)
            self.assertEqual(result.record.rejected_candidates, 1)
            self.assertEqual(result.record.primitives[0].bbox_px, (10, 5, 60, 45))
            self.assertEqual(result.crops["img_test_vp_0001"].size, (50, 40))
            self.assertEqual(result.crops["img_test_vp_0002"].size, (20, 10))
            self.assertEqual(result.record.source["extraction_mode"], "generic")

    def test_passes_caption_to_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.jpg"
            Image.new("RGB", (100, 50), "white").save(path)
            source = ImageInput(
                path,
                "test",
                "source.jpg",
                "img_test",
                "A woman modeling a red dress.",
            )
            discoverer = FakeDiscoverer()
            settings = Settings(model="fake", base_url="http://fake")

            result = VPExtractor(discoverer, settings).extract_image(source)

            self.assertEqual(discoverer.focus_context, source.caption)
            self.assertEqual(result.record.source["extraction_mode"], "caption_guided")
            self.assertEqual(result.record.source["caption"], source.caption)

    def test_suppresses_full_frame_parent_with_specific_child(self):
        full_frame = PrimitiveCandidate("open cardboard box", (0, 0, 1000, 1000))
        child = PrimitiveCandidate("dog in newspapers", (283, 338, 768, 857))
        result = suppress_full_frame_parents([full_frame, child], 0.9)
        self.assertEqual(result, [child])

    def test_keeps_full_frame_candidate_without_child(self):
        full_frame = PrimitiveCandidate("close-up document", (0, 0, 1000, 1000))
        self.assertEqual(
            suppress_full_frame_parents([full_frame], 0.9), [full_frame]
        )

    def test_keeps_large_non_full_frame_candidate(self):
        large = PrimitiveCandidate("large poster", (0, 0, 900, 900))
        child = PrimitiveCandidate("small logo", (100, 100, 300, 300))
        self.assertEqual(
            suppress_full_frame_parents([large, child], 0.9), [large, child]
        )

    def test_nms_keeps_larger_box_regardless_of_input_order(self):
        smaller = PrimitiveCandidate("white fluffy dog", (325, 354, 679, 809))
        larger = PrimitiveCandidate("stack of newspapers", (283, 338, 768, 854))
        self.assertEqual(deduplicate([smaller, larger], 0.6), [larger])

    def test_nms_keeps_distinct_boxes_below_threshold(self):
        first = PrimitiveCandidate("banana", (100, 100, 600, 600))
        second = PrimitiveCandidate("plate", (250, 100, 750, 600))
        self.assertEqual(len(deduplicate([first, second], 0.6)), 2)


if __name__ == "__main__":
    unittest.main()
