from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

from scripts.judge_results_llm_parallel import (
    normalize_judge_row,
    render_judge_prompt,
    summarize,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


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

    def test_wma_prompt_matches_official_prompt_with_all_context(self) -> None:
        prompt_path = (
            WORKSPACE_ROOT
            / "WorldMemArena"
            / "eval_framework"
            / "judges"
            / "prompts.py"
        )
        spec = importlib.util.spec_from_file_location("wma_official_judge_prompts", prompt_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        normalized = normalize_judge_row(
            "worldmemarena",
            {
                "query_id": "sample::question-1",
                "sample_id": "sample",
                "question": "What happened?",
                "system_answer": "A concise prediction.",
                "original_answer": "The reference answer.",
                "gold_evidence_contents": ["First memory point.", "Second memory point."],
            },
            1,
        )
        actual = render_judge_prompt(
            normalized["protocol_id"],
            prediction=normalized["prediction"],
            references=normalized["references"],
            question=normalized["question"],
            key_memory_points=normalized["key_memory_points"],
        )
        expected = module.QA_EVALUATION_PROMPT.format(
            question="What happened?",
            reference_answer="The reference answer.",
            key_memory_points="First memory point.\nSecond memory point.",
            response="A concise prediction.",
        )
        self.assertEqual(actual, expected)

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
