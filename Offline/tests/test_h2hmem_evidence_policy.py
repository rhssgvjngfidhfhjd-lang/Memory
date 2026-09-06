from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from evidence_policy.evidence import (
    DialogueStore,
    EvidenceChainBuilder,
    EvidenceStrategy,
    H2HMemDialogueStore,
)
from evidence_policy.rollout import EvidenceEpisode, EvidenceSelectionEnv
from hive_mem.mau import MAU, MAUBank
from hive_mem.retriever import MemoryHit
from scripts.evidence_policy import evidence_data_sources, iter_h2hmem_episodes


class H2HMemDialogueStoreTest(unittest.TestCase):
    def test_loads_rounds_and_maps_remote_image_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            session = root / "dyadic" / "dialogue1" / "scenes" / "session1"
            (session / "image").mkdir(parents=True)
            (session / "image" / "1.png").write_bytes(b"image")
            (session / "session.json").write_text(
                json.dumps(
                    {
                        "dialogue": [
                            {"role": "Alice", "content": {"text": "hello", "image": "1.png"}},
                            {"role": "Bob", "content": {"text": "hi", "image": ""}},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            store = H2HMemDialogueStore(root)
            rendered = store.get("dyadic_dialogue1", "session1:R0001").render()
            self.assertIn("Alice: hello", rendered)
            self.assertIn("Bob: hi", rendered)
            resolved = store.resolve_image_path(
                "dyadic_dialogue1",
                "/server/h2hmem_raw/dyadic/dialogue1/scenes/session1/image/1.png",
            )
            self.assertEqual(resolved, session / "image" / "1.png")


class H2HMemPolicyPlumbingTest(unittest.TestCase):
    def test_combined_h2h_data_sources_are_inferred(self):
        self.assertEqual(
            evidence_data_sources({"benchmark": "h2hmem"}),
            ("h2hmem_dyadic", "h2hmem_multiparty"),
        )

    def test_answer_renderer_uses_visual_category_override(self):
        item = MAU(
            id="m1",
            summary="fact",
            embedding=np.ones(4, dtype=np.float32),
            metadata={"source_dialogue_ids": ["D1"]},
        )
        hit = MemoryHit(item=item, score=1.0, rank=1)
        client = MagicMock()
        client.answer.return_value = "answer"
        client.model = "model"
        client.base_url = "http://example/v1"
        client.num_predict = 32
        client.temperature = 0.0
        client.think = False
        client.backend = "openai"
        builder = EvidenceChainBuilder(DialogueStore("."))
        env = EvidenceSelectionEnv(client, builder)
        episode = EvidenceEpisode(
            query_id="h2hmem:dyadic:dialogue1:session1:Q001",
            dataset="dyadic_dialogue1",
            category="Cross-modal Related Retrieval",
            question_prompt="question",
            system_prompt="system",
            ground_truth="answer",
            query_embedding=np.ones(4, dtype=np.float32),
            memory_hits=(hit,),
            metadata={"answer_category": "VR"},
        )

        env.rollout(episode, EvidenceStrategy.SUMMARY)

        self.assertEqual(client.answer.call_args.kwargs["category"], "VR")

    def test_h2hmem_ppo_retrieval_returns_vector_five_plus_graph_two(self):
        class QueryCache:
            @staticmethod
            def get_by_id(_query_id):
                return np.asarray([1.0, 0.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "memory" / "datasets" / "dyadic_dialogue1"
            bank = MAUBank()
            vectors = (
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.1, 0.9],
                [0.0, 1.0],
            )
            for index, vector in enumerate(vectors):
                bank.add_memory(
                    f"memory {index}",
                    np.asarray(vector, dtype=np.float32),
                    metadata={"session_id": f"session{index}"},
                )
            bank.memories[0].links["related"] = [
                {"target": bank.memories[5].id, "type": "SAME_EPISODE"},
                {"target": bank.memories[6].id, "type": "SAME_EPISODE"},
            ]
            bank.save(dataset_dir)
            row = SimpleNamespace(
                metadata={
                    "variant": "dyadic",
                    "question_image": "",
                    "answer_session": ["session0"],
                    "session_id": "session6",
                    "difficulty": "easy",
                },
                source_id="dialogue1",
                question_id="h2hmem:dyadic:dialogue1:q1",
                source_path=str(root / "question.json"),
                category="Unimodal Precise Recall",
                question="What happened?",
                answer="memory 0",
            )
            config = {
                "memory_bank": str(root / "memory"),
                "data_dir": str(root),
                "workspace_root": str(root),
                "split_manifest": str(root / "manifest.json"),
                "data_sources": ["h2hmem_dyadic"],
                "top_k": 5,
                "visual_categories": [],
                "graph_options": {
                    "mode": "append",
                    "append_k": 2,
                    "expand_temporal": False,
                    "expand_entity": False,
                    "expand_attribute": False,
                },
            }
            with patch(
                "scripts.evidence_policy.configured_split_manifest",
                return_value=MagicMock(),
            ), patch(
                "scripts.evidence_policy.iter_source_questions",
                return_value=iter([row]),
            ):
                episode = next(iter_h2hmem_episodes(config, "train", QueryCache()))

        self.assertEqual(len(episode.memory_hits), 7)
        self.assertEqual(
            [hit.via for hit in episode.memory_hits],
            ["vector"] * 5 + ["graph"] * 2,
        )
        self.assertEqual(episode.metadata["graph_append_k"], 2)


if __name__ == "__main__":
    unittest.main()
