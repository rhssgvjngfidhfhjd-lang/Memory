from __future__ import annotations

import unittest

from benchmarks.question_filter import is_excluded_category, parse_excluded_categories


class QuestionFilterTest(unittest.TestCase):
    def test_categories_are_normalized(self):
        excluded = parse_excluded_categories(" AR, mb ,Answer Refusal ")
        self.assertTrue(is_excluded_category("ar", excluded))
        self.assertTrue(is_excluded_category("MB", excluded))
        self.assertTrue(is_excluded_category("answer refusal", excluded))
        self.assertFalse(is_excluded_category("FR", excluded))

    def test_empty_configuration_disables_filter(self):
        self.assertFalse(is_excluded_category("AR", parse_excluded_categories("")))


if __name__ == "__main__":
    unittest.main()
