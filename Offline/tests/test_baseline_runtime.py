from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.baseline_runtime.protocol import (
    BaselineAdapter,
    MemoryRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedMemory,
    result_context_items,
)
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.registry import (
    BASELINE_NAMES,
    baseline_metadata,
    canonical_name,
)
from benchmarks.memgallery_harness.runner.answer_client import AnswerResponse
from benchmarks.memgallery_harness.eval_memgallery import run_dataset
from benchmarks.wma_harness.eval_wma import prepare_native_sample_jobs
from embedding.chunk_builder import Chunk, compact_text


class FakeBaseline(BaselineAdapter):
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.ended_sessions: list[str] = []
        self.closed = False

    def reset(self, sample_id: str, state_dir: Path) -> None:
        self.sample_id = sample_id
        self.state_dir = state_dir
        self.chunks = []

    def ingest(self, chunk: Chunk) -> None:
        self.chunks.append(chunk)

    def end_session(self, session_id: str) -> None:
        self.ended_sessions.append(session_id)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        visible = set(request.visible_session_ids)
        chunks = [
            chunk
            for chunk in self.chunks
            if not visible or str(chunk.metadata.get("session_id") or "") in visible
        ]
        return RetrievalResult(
            items=[
                RetrievedMemory(
                    memory_id=f"fake:{chunk.chunk_id}",
                    text=chunk.text,
                    score=1.0,
                    session_id=str(chunk.metadata.get("session_id") or ""),
                    source_dialogue_ids=[
                        str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
                    ],
                    image_ids=list(chunk.metadata.get("image_ids") or []),
                    image_paths=list(chunk.images),
                )
                for chunk in chunks[-request.top_k :]
            ],
            trace={"via": "fake"},
        )

    def snapshot(self) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                memory_id=f"fake:{chunk.chunk_id}",
                text=chunk.text,
                session_id=str(chunk.metadata.get("session_id") or ""),
                source_dialogue_ids=[
                    str(chunk.metadata.get("dialogue_id") or chunk.chunk_id)
                ],
                backend_type="fake",
            )
            for chunk in self.chunks
        ]

    def close(self) -> None:
        self.closed = True


class FakeAnswerClient:
    retries = 0

    def answer_with_usage(self, **kwargs):
        return AnswerResponse(
            text="answer from memory",
            usage=None,
            attempts=1,
            failed_attempts=0,
        )


class BaselineProtocolTest(unittest.TestCase):
    def test_output_layout_keeps_memory_under_baseline_root(self):
        layout = BaselineOutputLayout(Path("outputs/Mem-Gallery/M2A"))
        self.assertEqual(
            layout.datasets_dir,
            Path("outputs/Mem-Gallery/M2A/memory/datasets"),
        )
        self.assertEqual(
            layout.snapshot,
            Path("outputs/Mem-Gallery/M2A/memory/memory_snapshot.jsonl"),
        )
        self.assertEqual(layout.state_root("custom/state"), Path("custom/state"))

    def test_chunk_text_replaces_lone_unicode_surrogates(self):
        self.assertEqual(compact_text("before\udc94after"), "before?after")

    def test_registry_exposes_every_supported_baseline_and_aliases(self):
        self.assertEqual(
            set(BASELINE_NAMES),
            {
                "HiveMem",
                "AUGUSTUSMemory",
                "OmniSimpleMem",
                "M2A",
                "MIRIX",
                "MMA",
                "MemVerse",
                "M3-Agent-caption",
            },
        )
        self.assertEqual(canonical_name("m3-agent"), "M3-Agent-caption")
        self.assertEqual(canonical_name("omni-simplemem"), "OmniSimpleMem")
        with self.assertRaises(KeyError):
            canonical_name("MGMemory")
        self.assertEqual(
            baseline_metadata("m3-agent")["compatibility_mode"],
            "dialogue_round_as_clip",
        )

    def test_protocol_keeps_provenance_in_answer_context(self):
        result = RetrievalResult(
            items=[
                RetrievedMemory(
                    memory_id="m1",
                    text="fact",
                    session_id="S1",
                    source_dialogue_ids=["D1"],
                    image_ids=["I1"],
                    image_paths=["image.png"],
                )
            ]
        )
        item = result_context_items(result)[0]
        self.assertEqual(item["metadata"]["session_id"], "S1")
        self.assertEqual(item["metadata"]["source_dialogue_ids"], ["D1"])
        self.assertEqual(item["image"], {"path": "image.png", "img_id": "I1"})


class BaselineHarnessTest(unittest.TestCase):
    def test_memgallery_native_path_ingests_then_answers(self):
        payload = {
            "character_profile": {"name": "Ava"},
            "multi_session_dialogues": [
                {
                    "session_id": "S1",
                    "date": "2025-01-01",
                    "dialogues": [
                        {"round": "D1", "user": "My mug is blue.", "assistant": "Noted."}
                    ],
                }
            ],
            "human-annotated QAs": [
                {"question": "What color is my mug?", "answer": "blue", "point": "AR"}
            ],
        }
        fake = FakeBaseline()
        snapshots: list[dict] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ):
                rows, traces = run_dataset(
                    dataset_path,
                    root,
                    Path(),
                    FakeAnswerClient(),
                    None,
                    baseline="m3-agent",
                    state_root=root / "state",
                    memory_snapshots=snapshots,
                )
        self.assertEqual(rows[0]["system_answer"], "answer from memory")
        self.assertEqual(traces[0]["top_k"][0]["source_dialogue_ids"], ["D1"])
        self.assertEqual(fake.ended_sessions, ["S1"])
        self.assertTrue(fake.closed)
        self.assertEqual(len(snapshots), 1)

    def test_memgallery_excluded_category_never_retrieves_or_answers(self):
        payload = {
            "character_profile": {"name": "Ava"},
            "multi_session_dialogues": [],
            "human-annotated QAs": [
                {"question": "Unknown detail?", "answer": "Not mentioned.", "point": "AR"}
            ],
        }
        fake = FakeBaseline()
        stats: dict[str, int] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "demo.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.memgallery_harness.eval_memgallery.create_adapter",
                return_value=fake,
            ):
                rows, traces = run_dataset(
                    dataset_path,
                    root,
                    Path(),
                    FakeAnswerClient(),
                    None,
                    baseline="m3-agent",
                    state_root=root / "state",
                    excluded_categories=frozenset({"ar"}),
                    qa_stats=stats,
                )
        self.assertEqual(rows, [])
        self.assertEqual(traces, [])
        self.assertEqual(stats, {"eligible_questions": 0, "excluded_questions": 1})
        self.assertTrue(fake.closed)

    def test_wma_native_path_never_ingests_future_sessions(self):
        payload = {
            "sample_id": "sample_01",
            "sessions": [
                {
                    "_v2_session_id": "S00",
                    "dialogue": [
                        {"role": "user", "content": "Visible fact."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                },
                {
                    "_v2_session_id": "S01",
                    "dialogue": [
                        {"role": "user", "content": "Future fact."},
                        {"role": "assistant", "content": "Noted."},
                    ],
                },
            ],
            "qa_checkpoints": [
                {
                    "checkpoint_id": "QA00",
                    "covered_sessions": ["S00"],
                    "questions": [
                        {
                            "question": "What is visible?",
                            "answer": "Visible fact.",
                            "question_type_abbrev": "FR",
                        }
                    ],
                }
            ],
        }
        fake = FakeBaseline()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample_01.json"
            sample_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "benchmarks.wma_harness.eval_wma.create_adapter", return_value=fake
            ):
                jobs = prepare_native_sample_jobs(
                    sample_path,
                    None,
                    baseline="MemVerse",
                    state_root=root / "state",
                    top_k=5,
                    config_overrides={},
                )
        self.assertEqual([chunk.metadata["session_id"] for chunk in fake.chunks], ["S00"])
        self.assertEqual(jobs[0]["visible_sessions"], ["S00"])
        self.assertEqual(jobs[0]["retrieval_top_k"][0]["session_id"], "S00")
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main()
