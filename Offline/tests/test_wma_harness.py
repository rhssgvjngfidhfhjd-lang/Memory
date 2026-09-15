from __future__ import annotations

from argparse import Namespace
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
from benchmarks.wma_harness.eval_wma import (
    _mma_resume_signature_digests,
    _run_signature,
    m2a_wma_input_manifest,
    prepare_sample_jobs,
)
from benchmarks.wma_harness.runner.answer_client import build_retrieved_memory_context
from benchmarks.wma_harness.runner.metrics import (
    answer_span_exact_match,
    summarize_results,
)
from embedding.chunk_builder import (
    Chunk,
    balance_wma_chunks,
    build_wma_chunks_from_data,
)
from embedding.chunk_builder import iter_wma_sample_files
from evidence_policy.evidence import WMADialogueStore, make_policy_observation
from evidence_policy.retrieval import build_wma_prefix_graph_index
from hive_mem.mau import MAU, MAUBank
from hive_mem.prefix_graph import materialize_prefix_graph
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
    def test_mma_signature_ignores_retry_and_m2a_only_switches(self):
        with tempfile.TemporaryDirectory() as directory:
            sample_path = Path(directory) / "sample.json"
            sample_path.write_text("{}", encoding="utf-8")
            common = {
                "baseline": "MMA",
                "split_manifest": "",
                "answer_api_key": "secret",
                "answer_concurrency": 16,
                "sample_concurrency": 2,
                "allow_answer_errors": True,
                "checkpoint_every": 10,
                "result_dir": "results",
                "resume": True,
                "skip_model_check": False,
                "sample_attempts": 3,
                "mirix_skip_failed_build_points": False,
                "mirix_max_consecutive_failed_build_points": 10,
                "m2a_skip_failed_build_points": False,
                "m2a_max_consecutive_failed_build_points": 10,
                "top_k": 7,
            }
            first = _run_signature(Namespace(**common), [sample_path])
            changed = dict(common)
            changed.update(
                sample_attempts=99,
                m2a_skip_failed_build_points=True,
                m2a_max_consecutive_failed_build_points=2,
            )
            second = _run_signature(Namespace(**changed), [sample_path])

            self.assertEqual(first, second)
            digests = _mma_resume_signature_digests(
                Namespace(**changed), second
            )
            self.assertEqual(len(digests), 2)
            self.assertEqual(digests[0], _mma_resume_signature_digests(
                Namespace(**common), first
            )[0])

    def test_m2a_input_manifest_records_native_builder_and_raw_files(self):
        payload = sample_payload()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "personal" / "sample_01.json"
            image_path = (
                sample_path.parent
                / "images"
                / "sample_01"
                / "sample_01_img_001.png"
            )
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"image")
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            sample_path.write_text(json.dumps(payload), encoding="utf-8")

            manifest = m2a_wma_input_manifest([sample_path])

        self.assertEqual(
            manifest["builder"],
            "embedding.chunk_builder.build_wma_chunks_from_data",
        )
        self.assertFalse(manifest["shared_fixed_chunks"])
        self.assertEqual(manifest["input_file_count"], 2)
        self.assertIn(str(sample_path.resolve()), manifest["input_files"])
        self.assertIn(str(image_path.resolve()), manifest["input_files"])

    def test_balanced_chunks_keep_rounds_whole_and_sessions_separate(self):
        rounds = []
        for session_id, count in (("S00", 3), ("S01", 1)):
            for number in range(1, count + 1):
                dialogue_id = f"{session_id}:R{number:04d}"
                rounds.append(
                    Chunk(
                        chunk_id=f"sample_01:{dialogue_id}",
                        text=(
                            f"profile_summary: \nsession: {session_id}\ndate: 2025-01-01\n"
                            f"round: {dialogue_id}\nuser: {'x' * 360}\nassistant: y"
                        ),
                        metadata={
                            "dataset": "sample_01",
                            "session_id": session_id,
                            "dialogue_id": dialogue_id,
                            "round_id": number,
                            "date": "2025-01-01",
                            "image_ids": [],
                            "image_captions": [],
                        },
                    )
                )
        chunks = balance_wma_chunks(rounds, target_tokens=210)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].metadata["round_count"], 2)
        self.assertEqual(
            chunks[0].metadata["source_dialogue_ids"],
            ["S00:R0001", "S00:R0002"],
        )
        self.assertEqual(chunks[1].metadata["source_dialogue_ids"], ["S00:R0003"])
        self.assertEqual(chunks[2].metadata["session_id"], "S01")

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

    def test_prefix_graph_excludes_future_rows_and_slices_image_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "datasets" / "sample_01"
            bank = MAUBank()
            for session_id, value in (("S00", 1.0), ("S01", 0.8), ("S02", 0.6)):
                bank.add_memory(
                    f"memory {session_id}",
                    np.asarray([value, 1.0 - value], dtype=np.float32),
                    metadata={
                        "session_id": session_id,
                        "source_dialogue_ids": [f"{session_id}:R0001"],
                    },
                )
            bank.memories[0].links["related"] = [
                {"target": bank.memories[2].id, "type": "SAME_EPISODE"}
            ]
            bank.save(source)
            vectors_dir = source / "vectors"
            np.save(
                vectors_dir / "image.npy",
                np.asarray([[1.0, 0.0], [0.8, 0.2], [0.6, 0.4]], dtype=np.float32),
            )
            np.save(vectors_dir / "image_mask.npy", np.asarray([True, False, True]))

            checkpoint_root = root / "prefix" / "sample_01" / "QA01"
            materialize_prefix_graph(
                source,
                checkpoint_root,
                sample_id="sample_01",
                checkpoint_id="QA01",
                visible_session_ids=("S00", "S01"),
                graph_options={"mode": "append", "append_k": 2},
            )

            prefix_dir = checkpoint_root / "datasets" / "sample_01"
            prefix = MAUBank.load(prefix_dir)
            self.assertEqual(
                [row.metadata["session_id"] for row in prefix.memories],
                ["S00", "S01"],
            )
            self.assertEqual(prefix.memories[0].links["next"], prefix.memories[1].id)
            self.assertEqual(prefix.memories[1].links["prev"], prefix.memories[0].id)
            self.assertTrue(all(not row.links["related"] for row in prefix.memories))
            self.assertEqual(np.load(prefix_dir / "vectors" / "image.npy").shape, (2, 2))
            self.assertEqual(
                np.load(prefix_dir / "vectors" / "image_mask.npy").tolist(),
                [True, False],
            )
            manifest = json.loads(
                (checkpoint_root / "prefix_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["visible_session_ids"], ["S00", "S01"])
            self.assertEqual(manifest["memory_count"], 2)

    def test_evidence_policy_prefix_index_never_returns_future_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source" / "datasets" / "sample_01"
            bank = MAUBank()
            for session_id, vector in (
                ("S00", [1.0, 0.0]),
                ("S01", [0.9, 0.1]),
                ("S02", [0.8, 0.2]),
            ):
                bank.add_memory(
                    f"memory {session_id}",
                    np.asarray(vector, dtype=np.float32),
                    metadata={"session_id": session_id},
                )
            bank.memories[0].links["related"] = [
                {"target": bank.memories[2].id, "type": "SAME_EPISODE"}
            ]
            bank.save(source)
            options = {
                "seed_k": 0,
                "mode": "append",
                "append_k": 2,
                "expansion_bonus": 0.2,
                "expand_entity": False,
                "expand_attribute": False,
            }

            index, signature = build_wma_prefix_graph_index(
                source,
                root / "prefix",
                sample_id="sample_01",
                checkpoint_id="QA01",
                visible_session_ids=("S00", "S01"),
                options=options,
            )
            hits = index.search(
                [1.0, 0.0],
                top_k=5,
                allowed_session_ids={"S00", "S01"},
            )

        self.assertTrue(signature)
        self.assertEqual(
            {hit.item.metadata["session_id"] for hit in hits},
            {"S00", "S01"},
        )

    def test_wma_runner_uses_checkpoint_prefix_graph(self):
        class QueryCache:
            @staticmethod
            def get_by_id(_query_id):
                return np.asarray([1.0, 0.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "lifelong" / "personal" / "sample_01.json"
            sample_path.parent.mkdir(parents=True)
            sample_path.write_text(json.dumps(sample_payload()), encoding="utf-8")
            source = root / "index" / "datasets" / "sample_01"
            bank = MAUBank()
            for session_id, vector in (
                ("S00", [1.0, 0.0]),
                ("S01", [0.9, 0.1]),
                ("S02", [0.8, 0.2]),
            ):
                bank.add_memory(
                    f"memory {session_id}",
                    np.asarray(vector, dtype=np.float32),
                    metadata={
                        "session_id": session_id,
                        "source_dialogue_ids": [f"{session_id}:R0001"],
                    },
                )
            bank.save(source)
            jobs = prepare_sample_jobs(
                sample_path,
                root / "index",
                QueryCache(),
                top_k=5,
                graph_options={"mode": "append", "append_k": 2},
                prefix_graph_root=root / "prefix_graphs",
            )

            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["visible_sessions"], ["S00"])
            self.assertTrue(Path(jobs[0]["graph_prefix_manifest"]).is_file())
            self.assertEqual(
                {row["session_id"] for row in jobs[0]["retrieval_top_k"]},
                {"S00"},
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

    def test_duplicate_image_paths_across_memory_items_are_attached_once(self):
        image = {"path": "/tmp/shared.png", "img_id": "sample_01_img_001"}
        memory_items = [
            {
                "text": "First visual memory.",
                "images": [image],
                "metadata": {"session_id": "S00", "dialogue_id": "S00:R0001"},
            },
            {
                "text": "Second visual memory.",
                "images": [dict(image)],
                "metadata": {"session_id": "S01", "dialogue_id": "S01:R0001"},
            },
        ]
        text, images = build_retrieved_memory_context(memory_items, "VFR")
        self.assertEqual(images, ["/tmp/shared.png"])
        self.assertEqual(text.count("Attached memory image"), 1)

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
