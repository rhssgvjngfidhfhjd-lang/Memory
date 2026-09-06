from __future__ import annotations

import unittest

from scripts.judge_results_llm_parallel import normalize_judge_row, summarize


class JudgeEvidencePolicyRolloutTest(unittest.TestCase):
    def test_accepts_evidence_policy_answer_field(self) -> None:
        normalized = normalize_judge_row(
            "memgallery",
            {
                "query_id": "dataset::question-1",
                "dataset": "dataset",
                "answer": "predicted answer",
                "original_answer": "reference answer",
            },
            1,
        )

        self.assertEqual(normalized["prediction"], "predicted answer")
        self.assertEqual(normalized["references"], ["reference answer"])

    def test_summarizes_judge_scores_by_category(self) -> None:
        rows = [
            {"category": "FR", "label": "correct", "score": 1.0},
            {"category": "FR", "label": "partial", "score": 0.5},
            {"category": "VR", "label": "incorrect", "score": 0.0},
        ]

        metrics = summarize(rows, "judge", expected_count=3)

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["by_category"]["FR"]["accuracy"], 0.75)
        self.assertEqual(metrics["by_category"]["VR"]["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
