import unittest

from PIL import Image

from vp_extractor.vlm import ObjectDiscoverer, parse_candidates


class FakeVLM:
    def __init__(self):
        self.prompt = ""

    def generate(self, prompt, image):
        self.prompt = prompt
        return "[]"


class CandidateParsingTests(unittest.TestCase):
    def test_discovery_injects_caption_context(self):
        vlm = FakeVLM()
        discoverer = ObjectDiscoverer(
            vlm,
            "__FOCUS_CONTEXT__\nLimit __MAX_PRIMITIVES__",
            "",
            5,
        )

        discoverer.discover(Image.new("RGB", (10, 10)), "A red dress.")

        self.assertIn("Memory caption:\nA red dress.", vlm.prompt)
        self.assertIn("Limit 5", vlm.prompt)

    def test_build_prompt_matches_discovery_prompt(self):
        vlm = FakeVLM()
        discoverer = ObjectDiscoverer(vlm, "__FOCUS_CONTEXT__", "", 5)

        expected = discoverer.build_prompt("A red dress.")
        discoverer.discover(Image.new("RGB", (10, 10)), "A red dress.")

        self.assertEqual(vlm.prompt, expected)

    def test_caption_prompt_marks_incidental_objects_as_context(self):
        guided = "Caption: __CAPTION__\nincluding details are context only"
        discoverer = ObjectDiscoverer(FakeVLM(), "generic", "", 5, guided)

        result = discoverer.build_prompt(
            "A dog in a box, including a flyer with visible text."
        )

        self.assertIn("Caption: A dog in a box", result)
        self.assertIn("including details are context only", result)

    def test_discovery_uses_generic_mode_without_caption(self):
        vlm = FakeVLM()
        discoverer = ObjectDiscoverer(vlm, "__FOCUS_CONTEXT__", "", 5)

        discoverer.discover(Image.new("RGB", (10, 10)))

        self.assertIn("generic image-only discovery", vlm.prompt)

    def test_parses_fenced_json_and_qwen_bbox_alias(self):
        result = parse_candidates(
            '```json\n[{"label":"green can","bbox_2d":[10,20,300,400]}]\n```'
        )
        self.assertEqual(result[0].label, "green can")
        self.assertEqual(result[0].bbox_norm, (10.0, 20.0, 300.0, 400.0))

    def test_ignores_malformed_items(self):
        result = parse_candidates(
            '[{"label":"valid","bbox_norm":[1,2,3,4]},'
            '{"label":"missing box"},42]'
        )
        self.assertEqual([item.label for item in result], ["valid"])

    def test_accepts_single_relocalization_object(self):
        result = parse_candidates('{"bbox_norm":[10,20,30,40]}', default_label="cup")
        self.assertEqual(result[0].label, "cup")


if __name__ == "__main__":
    unittest.main()
