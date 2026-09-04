from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.wma_harness.retrieval.query_embedding_cache import (
    build_gold_evidence_map,
    iter_qa_items,
    make_query_id,
    visible_sessions_for_checkpoint,
)
from benchmarks.wma_harness.eval_wma import prepare_sample_jobs
from benchmarks.wma_harness.runner.answer_client import build_retrieved_memory_context
from benchmarks.wma_harness.runner.metrics import (
    answer_span_exact_match,
    summarize_results,
)
from embedding.chunk_builder import build_wma_chunks_from_data
from embedding.chunk_builder import iter_wma_sample_files
from evidence_policy.evidence import WMADialogueStore, make_policy_observation
from hive_mem.mau import MAU, MAUBank
from hive_mem.retriever import GraphExpandedIndex, MemoryHit, SimpleMemoryIndex


def sample_payload() -> dict:
    return {
        "sample_id": "sample_01",
        "sessions": [
            {
                "_v2_session_id": "S00",
                "dialogue": [
                    {
                        "role": "user",
                        "content": "I bought the blue cup.",
                        "timestamp": "Jan 01, 2025, 09:00:00",
                        "attachments": [
                            {
                                "image_id": "sample_01_img_001",
                                "caption": "A blue cup.",
                                "file_path": "images/sample_01/sample_01_img_001.png",
                            }
                        ],
                    },
                    {"role": "assistant", "content": "Noted.", "attachments": []},
                ],
                "memory_points": [
                    {"memory_id": "mp_S00_1", "_session_id": "S00"}
                ],
            },
            {
                "_v2_session_id": "S01",
                "dialogue": [
                    {"role": "user", "content": "Future secret.", "attachments": []},
                    {"role": "assistant", "content": "Okay.", "attachments": []},
                ],
            },
            {
                "_v2_session_id": "S02",
                "dialogue": [
                    {"role": "user", "content": "Later future.", "attachments": []},
                    {"role": "assistant", "content": "Okay.", "attachments": []},
                ],
            },
        ],
        "memory_points": [
            {
                "session_id": "S01",
                "memory_points": [
                    {
                        "memory_id": "mp_S01_1",
                        "_session_id": "S01",
                        "memory_content": "The future secret.",
                    }
                ],
            }
        ],
        "qa_checkpoints": [
            {
                "checkpoint_id": "QA00",
                "covered_sessions": ["S00"],
                "questions": [
                    {
                        "question": "What color was the cup?",
                        "answer": "Blue.",
                        "question_type": "Fact Recall",
                        "question_type_abbrev": "FR",
                        "difficulty": "easy",
                        "evidence": [{"memory_id": "mp_S00_1"}],
                    }
                ],
            }
        ],
    }


class WMAChunkTest(unittest.TestCase):
    def test_chunk_schema_matches_builder_and_excludes_gold(self):
        payload = sample_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "lifelong" / "personal" / "sample_01.json"
            image_path = sample_path.parent / "images" / "sample_01" / "sample_01_img_001.png"
            image_path.parent.mkdir(parents=True)
            image_path.touch()
            chunks = build_wma_chunks_from_data(
                payload, root, sample_path=sample_path
            )
        self.assertEqual(len(chunks), 3)
        first = chunks[0]
        self.assertEqual(first.chunk_id, "sample_01:S00:R0001")
        self.assertEqual(first.metadata["dataset"], "sample_01")
        self.assertEqual(first.metadata["dialogue_id"], "S00:R0001")
        self.assertIn("I bought the blue cup.", first.text)
        serialized = json.dumps([row.to_dict() for row in chunks])
        self.assertNotIn("What color was the cup?", serialized)
        self.assertNotIn("mp_S00_1", serialized)
        self.assertEqual(first.images, [str(image_path.resolve())])

    def test_checkpoint_covered_sessions_are_trigger_not_whitelist(self):
        visible = visible_sessions_for_checkpoint(
            ["S00", "S01", "S02"], ["S01"]
        )
        self.assertEqual(visible, ["S00", "S01"])

    def test_gold_map_includes_top_level_memory_point_blocks(self):
        points = build_gold_evidence_map(sample_payload())
        self.assertEqual(points["mp_S01_1"]["session_id"], "S01")
        self.assertEqual(points["mp_S01_1"]["content"], "The future secret.")

    def test_query_cache_items_are_checkpoint_qualified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "lifelong" / "personal" / "sample_01.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(sample_payload()), encoding="utf-8")
            rows = iter_qa_items(root)
        self.assertEqual(len(rows), 1)
        self.assertIn("sample_01::QA00::1::FR::", rows[0]["query_id"])
        self.assertEqual(rows[0]["covered_sessions"], ["S00"])
        self.assertEqual(rows[0]["visible_sessions"], ["S00"])
        self.assertEqual(
            rows[0]["query_id"],
            make_query_id(
                sample_id="sample_01", checkpoint_id="QA00", qa_index=1,
                category="FR", question="What color was the cup?",
            ),
        )

    def test_query_cache_omits_excluded_memory_boundary_without_renumbering(self):
        payload = sample_payload()
        payload["qa_checkpoints"][0]["questions"].insert(
            0,
            {
                "question": "What was never stated?",
                "answer": "Unknown.",
                "question_type": "Memory Boundary",
                "question_type_abbrev": "MB",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "lifelong" / "personal" / "sample_01.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = iter_qa_items(root, excluded_categories=frozenset({"mb"}))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "FR")
        self.assertIn("sample_01::QA00::2::FR::", rows[0]["query_id"])

    def test_sample_loader_accepts_wma_subtree(self):
        with tempfile.TemporaryDirectory() as directory:
            subtree = Path(directory) / "lifelong"
            sample_path = subtree / "personal" / "sample_01.json"
            sample_path.parent.mkdir(parents=True)
            sample_path.write_text(json.dumps(sample_payload()), encoding="utf-8")
            paths = iter_wma_sample_files(subtree)
        self.assertEqual(paths, [sample_path])

    def test_sample_loader_uses_only_lifelong_from_wma_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lifelong = root / "lifelong" / "personal" / "personal_01.json"
            agent = root / "agent" / "gui" / "mobile_01.json"
            lifelong.parent.mkdir(parents=True)
            agent.parent.mkdir(parents=True)
            lifelong.write_text(json.dumps(sample_payload()), encoding="utf-8")
            agent.write_text(json.dumps(sample_payload()), encoding="utf-8")
            paths = iter_wma_sample_files(root)
        self.assertEqual(paths, [lifelong])


class WMARetrievalTest(unittest.TestCase):
    def _index(self, root: Path, graph: bool = False):
        bank = MAUBank()
        bank.add_memory(
            "visible old fact", np.asarray([1.0, 0.0], dtype=np.float32),
            metadata={"session_id": "S00", "source_dialogue_ids": ["S00:R0001"]},
        )
        bank.add_memory(
            "future fact", np.asarray([1.0, 0.0], dtype=np.float32),
            metadata={"session_id": "S01", "source_dialogue_ids": ["S01:R0001"]},
        )
        bank.memories[0].links["next"] = bank.memories[1].id
        bank.memories[1].links["prev"] = bank.memories[0].id
        bank.save(root)
        if graph:
            return GraphExpandedIndex(
                root, mode="append", append_k=2,
                expand_entity=False, expand_attribute=False,
            )
        return SimpleMemoryIndex(root)

    def test_checkpoint_mask_blocks_future_vector_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            index = self._index(Path(directory))
            hits = index.search(
                [1.0, 0.0], top_k=5, allowed_session_ids={"S00"}
            )
        self.assertEqual([hit.item.metadata["session_id"] for hit in hits], ["S00"])

    def test_checkpoint_mask_blocks_future_graph_neighbour(self):
        with tempfile.TemporaryDirectory() as directory:
            index = self._index(Path(directory), graph=True)
            hits = index.search(
                [1.0, 0.0], top_k=1, allowed_session_ids={"S00"}
            )
        self.assertEqual([hit.item.metadata["session_id"] for hit in hits], ["S00"])

    def test_wma_runner_rejects_non_prefix_safe_graph(self):
        with self.assertRaisesRegex(ValueError, "not prefix-safe"):
            prepare_sample_jobs(
                Path("unused.json"),
                Path("unused"),
                None,
                top_k=5,
                graph_options={"mode": "rerank"},
            )


class WMAEvidenceAndMetricsTest(unittest.TestCase):
    def test_dialogue_store_uses_same_round_ids_as_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "agent" / "demo" / "sample_01.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(sample_payload()), encoding="utf-8")
            store = WMADialogueStore(root)
            row = store.get("sample_01", "S00:R0001")
        self.assertEqual(row.user, "I bought the blue cup.")
        self.assertEqual(row.assistant, "Noted.")

    def test_dialogue_store_resolves_image_relative_to_sample_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "agent" / "demo" / "sample_01.json"
            sample_path.parent.mkdir(parents=True)
            sample_path.write_text(json.dumps(sample_payload()), encoding="utf-8")
            image_path = sample_path.parent / "images" / "sample_01" / "sample_01_img_001.png"
            image_path.parent.mkdir(parents=True)
            image_path.touch()
            resolved = WMADialogueStore(root).resolve_image_path(
                "sample_01", "images/sample_01/sample_01_img_001.png"
            )
        self.assertEqual(resolved, image_path)

    def test_nonvisual_context_redacts_wma_image_id(self):
        text, images = build_retrieved_memory_context(
            [
                {
                    "text": "The image is sample_01_img_001.",
                    "image": {"path": "/tmp/image.png", "img_id": "sample_01_img_001"},
                    "metadata": {
                        "session_id": "S00", "dialogue_id": "S00:R0001",
                        "image_id": "sample_01_img_001", "image_ids": ["sample_01_img_001"],
                    },
                }
            ],
            "FR",
        )
        self.assertEqual(images, [])
        self.assertNotIn("sample_01_img_001", text)

    def test_wma_visual_categories_enable_policy_image_action(self):
        item = MAU(
            id="m1", summary="visual memory",
            embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            metadata={
                "image_paths": ["image.png"],
                "image_captions": ["caption"],
            },
        )
        observation = make_policy_observation(
            [1.0, 0.0], [MemoryHit(item=item, score=1.0, rank=1)], "VFR",
            visual_categories={"VFR", "VS", "VU", "CMR"},
        )
        self.assertTrue(bool(observation.visual_action_mask[0]))

    def test_metrics_group_by_category_and_difficulty(self):
        metrics = summarize_results(
            [
                {
                    "system_answer": "blue", "original_answer": "blue",
                    "category": "FR", "difficulty": "easy",
                    "gold_sessions": ["S00"], "retrieved_sessions": ["S00"],
                }
            ]
        )
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["em"], 1.0)
        self.assertEqual(metrics["strict_em"], 1.0)
        self.assertNotIn("exact_match", metrics)
        self.assertEqual(metrics["retrieval_hitrate@5"], 1.0)
        self.assertEqual(metrics["by_category"]["FR"]["count"], 1)
        self.assertEqual(metrics["future_gold_evidence_questions"], 0)

    def test_wma_em_matches_concise_answer_inside_explanatory_reference(self):
        self.assertEqual(
            answer_span_exact_match(
                "Laura K. Simmons",
                "Laura K. Simmons gave the XRD training.",
            ),
            1.0,
        )
        self.assertEqual(
            answer_span_exact_match("Laura Simmons", "Laura K. Simmons gave it."),
            0.0,
        )

        metrics = summarize_results(
            [
                {
                    "system_answer": "Unknown",
                    "original_answer": "Unknown; the detail was not provided.",
                }
            ]
        )
        self.assertEqual(metrics["em"], 1.0)
        self.assertEqual(metrics["strict_em"], 0.0)

    def test_metrics_do_not_treat_future_gold_as_retrievable(self):
        metrics = summarize_results(
            [
                {
                    "system_answer": "unknown",
                    "original_answer": "blue",
                    "category": "FR",
                    "difficulty": "easy",
                    "gold_sessions": ["S00", "S02"],
                    "gold_visible_sessions": ["S00"],
                    "gold_future_evidence_ids": ["future_image"],
                    "retrieved_sessions": ["S02"],
                }
            ]
        )
        self.assertEqual(metrics["retrieval_hitrate@5"], 0.0)
        self.assertEqual(metrics["future_gold_evidence_questions"], 1)


if __name__ == "__main__":
    unittest.main()
