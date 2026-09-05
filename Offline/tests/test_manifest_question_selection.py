from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.baseline_runtime.protocol import RetrievalResult
from benchmarks.h2hmem_harness.eval_h2hmem import (
    h2hmem_manifest_question_id,
    prepare_conversation_jobs,
)
from benchmarks.memgallery_harness.eval_memgallery import (
    memgallery_manifest_question_id,
    prepare_dataset_jobs,
)
from benchmarks.wma_harness.eval_wma import (
    _with_manifest_question_id,
    prepare_native_sample_jobs,
    wma_manifest_question_id,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.requests = []

    def reset(self, sample_id, state_dir) -> None:
        return None

    def ingest(self, chunk) -> None:
        return None

    def end_session(self, session_id) -> None:
        return None

    def retrieve(self, request):
        self.requests.append(request)
        return RetrievalResult()

    def snapshot(self):
        return []

    def close(self) -> None:
        return None


class CanonicalManifestQuestionIdTest(unittest.TestCase):
    def test_memgallery_uses_zero_based_manifest_suffix(self):
        self.assertEqual(
            memgallery_manifest_question_id("sample", 1), "sample_q0000"
        )

    def test_h2hmem_uses_original_csq_id(self):
        self.assertEqual(
            h2hmem_manifest_question_id(
                "dyadic",
                "dialogue10",
                "session0",
                1,
                {"original_question_id": "CSQ001"},
            ),
            "h2hmem:dyadic:dialogue10:session0:CSQ001",
        )

    def test_h2hmem_numbers_regular_session_questions(self):
        self.assertEqual(
            h2hmem_manifest_question_id(
                "multiparty", "dialogue3", "session2", 4, {}
            ),
            "h2hmem:multiparty:dialogue3:session2:Q004",
        )

    def test_wma_uses_checkpoint_and_one_based_question_number(self):
        self.assertEqual(
            wma_manifest_question_id("academic_03", "QA00", 1),
            "academic_03:QA00:Q001",
        )

    def test_wma_backfills_manifest_id_in_legacy_checkpoint_job(self):
        legacy = {
            "query_id": "academic_03::QA00::1::FR::hash",
            "sample_id": "academic_03",
            "checkpoint_id": "QA00",
            "qa_index": 1,
        }
        normalized = _with_manifest_question_id(legacy)
        self.assertEqual(
            normalized["manifest_question_id"],
            "academic_03:QA00:Q001",
        )
        self.assertNotIn("manifest_question_id", legacy)


class StrictManifestSelectionTest(unittest.TestCase):
    def test_memgallery_retrieves_only_manifest_questions_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "dialog" / "sample.json"
            dataset_path.parent.mkdir(parents=True)
            dataset_path.write_text(
                json.dumps(
                    {
                        "character_profile": {"name": "Sample"},
                        "human-annotated QAs": [
                            {"question": "first", "answer": "1", "point": "AR"},
                            {"question": "second", "answer": "2", "point": "FR"},
                            {"question": "third", "answer": "3", "point": "FR"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            adapter = FakeAdapter()
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=adapter,
            ), patch(
                "benchmarks.memgallery_harness.eval_memgallery.build_chunks_from_data",
                return_value=[],
            ):
                artifact = prepare_dataset_jobs(
                    dataset_path,
                    root,
                    root,
                    None,
                    baseline="M2A",
                    ordered_question_ids=("sample_q0002", "sample_q0000"),
                )
        self.assertEqual(
            [row["manifest_question_id"] for row in artifact["jobs"]],
            ["sample_q0002", "sample_q0000"],
        )
        self.assertEqual([request.text for request in adapter.requests], ["[FR] third", "[AR] first"])

    def test_h2hmem_retrieves_only_manifest_questions_in_manifest_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question_path = (
                root / "dyadic" / "dialogue1" / "scenes" / "session1" / "questions.json"
            )
            question_path.parent.mkdir(parents=True)
            question_path.write_text(
                json.dumps(
                    {
                        "questions": [
                            {"question": {"text": "first"}},
                            {"question": {"text": "second"}},
                            {"question": {"text": "third"}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            adapter = FakeAdapter()
            with patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.create_adapter",
                return_value=adapter,
            ), patch(
                "benchmarks.h2hmem_harness.eval_h2hmem.build_h2h_chunks_from_directory",
                return_value=[],
            ):
                artifact = prepare_conversation_jobs(
                    data_dir=root,
                    variant="dyadic",
                    conversation_id="dialogue1",
                    baseline="M2A",
                    state_root=root / "state",
                    config={"top_k": 7},
                    ordered_question_ids=(
                        "h2hmem:dyadic:dialogue1:session1:Q003",
                        "h2hmem:dyadic:dialogue1:session1:Q001",
                    ),
                )
        self.assertEqual(
            [row["manifest_question_id"] for row in artifact["jobs"]],
            [
                "h2hmem:dyadic:dialogue1:session1:Q003",
                "h2hmem:dyadic:dialogue1:session1:Q001",
            ],
        )
        self.assertEqual([request.text for request in adapter.requests], ["third", "first"])

    def test_wma_retrieves_chronologically_then_returns_manifest_order(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [
                {"_v2_session_id": "S00", "dialogue": []},
                {"_v2_session_id": "S01", "dialogue": []},
            ],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S00"],
                    "questions": [{"question": "early", "answer": "e"}],
                },
                {
                    "checkpoint_id": "QA01",
                    "covered_sessions": ["S01"],
                    "questions": [{"question": "late", "answer": "l"}],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")
            adapter = FakeAdapter()
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter",
                return_value=adapter,
            ), patch(
                "benchmarks.wma_harness.eval_wma.build_wma_chunks_from_data",
                return_value=[],
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="M2A",
                    state_root=root / "state",
                    top_k=7,
                    config_overrides={},
                    ordered_question_ids=(
                        "sample_01:QA01:Q001",
                        "sample_01:QA00:Q001",
                    ),
                )
        self.assertEqual([request.text for request in adapter.requests], ["early", "late"])
        self.assertEqual(
            [row["manifest_question_id"] for row in jobs],
            ["sample_01:QA01:Q001", "sample_01:QA00:Q001"],
        )


if __name__ == "__main__":
    unittest.main()
