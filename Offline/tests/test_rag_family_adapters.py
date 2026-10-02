from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.baseline_runtime.adapters.rag_family import RAGFamilyAdapter
from benchmarks.baseline_runtime.protocol import RetrievalRequest
from embedding.chunk_builder import Chunk
from embedding.openai_memory_embedder import OpenAIMemoryEmbedder


class FakeEmbedder:
    def embed_texts(self, texts, mode="context"):
        del mode
        values = [texts] if isinstance(texts, str) else list(texts)
        result = np.asarray([self._vector(value, []) for value in values])
        return result[0] if isinstance(texts, str) else result

    def embed_multimodal(self, text, image_paths=(), *, mode="context"):
        del mode
        return self._vector(text, list(image_paths))

    @staticmethod
    def _vector(text, images):
        value = str(text).casefold()
        if images or "visual" in value or "image" in value:
            return np.asarray([0.0, 1.0], dtype=np.float32)
        return np.asarray([1.0, 0.0], dtype=np.float32)


def config() -> dict:
    return {
        "top_k": 7,
        "embedding_dim": 2,
        "embedding_model": "fake-vl",
        "embedding_base_url": "http://127.0.0.1:1/v1",
        "embedding_api_key_env": "EMBEDDING_API_KEY",
        "answer_model": "answer",
        "answer_base_url": "http://127.0.0.1:2/v1",
        "executor_model": "answer",
        "executor_base_url": "http://127.0.0.1:2/v1",
        "request_timeout": 1,
    }


def chunk(index: int, *, session: str, visual: bool = False) -> Chunk:
    return Chunk(
        chunk_id=f"c{index}",
        text="visual memory" if visual else "text memory",
        images=[__file__] if visual else [],
        metadata={
            "session_id": session,
            "dialogue_id": f"d{index}",
            "image_ids": [f"img{index}"] if visual else [],
        },
    )


class RAGFamilyAdapterTest(unittest.TestCase):
    def make_adapter(self, baseline: str) -> RAGFamilyAdapter:
        adapter = RAGFamilyAdapter(
            baseline=baseline, source_root=Path(__file__).parent, config=config()
        )
        adapter.embedder = FakeEmbedder()
        adapter.reset("sample", Path("state"))
        return adapter

    def test_naive_rag_uses_fixed_text_chunks_and_never_attaches_images(self):
        adapter = self.make_adapter("NaiveRAG")
        adapter.ingest(chunk(1, session="s1", visual=False))
        adapter.ingest(chunk(2, session="s1", visual=True))
        adapter.end_session("s1")
        result = adapter.retrieve(
            RetrievalRequest(query_id="q", text="text question", top_k=1)
        )
        self.assertEqual([row.memory_id for row in result.items], ["NaiveRAG:sample:c1"])
        self.assertEqual(result.items[0].image_paths, [])
        self.assertEqual(len(adapter.snapshot()), 2)
        self.assertTrue(all(row.metadata["fixed_chunk"] for row in adapter.snapshot()))

    def test_murag_fuses_query_image_and_honours_visible_sessions(self):
        adapter = self.make_adapter("MuRAG")
        adapter.ingest(chunk(1, session="past", visual=False))
        adapter.end_session("past")
        adapter.ingest(chunk(2, session="future", visual=True))
        adapter.end_session("future")
        result = adapter.retrieve(
            RetrievalRequest(
                query_id="q",
                text="question",
                query_image=__file__,
                visible_session_ids=("past",),
                top_k=7,
            )
        )
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].session_id, "past")
        self.assertEqual(result.trace["visible_session_filter"], ["past"])

    def test_universalrag_routes_to_separate_document_and_image_corpora(self):
        adapter = self.make_adapter("UniversalRAG")
        adapter.ingest(chunk(1, session="s1", visual=False))
        adapter.ingest(chunk(2, session="s1", visual=True))
        adapter.end_session("s1")
        with patch.object(adapter, "_route", return_value="image"):
            image_result = adapter.retrieve(
                RetrievalRequest(query_id="qi", text="visual question", top_k=7)
            )
        self.assertEqual([row.memory_id for row in image_result.items], ["UniversalRAG:sample:c2"])
        self.assertEqual(image_result.trace["route"], "image")
        with patch.object(adapter, "_route", return_value="no"):
            no_result = adapter.retrieve(
                RetrievalRequest(query_id="qn", text="common knowledge", top_k=7)
            )
        self.assertEqual(no_result.items, [])
        self.assertEqual(no_result.trace["route"], "no")

    def test_top_k_is_capped_at_protocol_budget(self):
        adapter = self.make_adapter("NaiveRAG")
        for index in range(10):
            adapter.ingest(chunk(index, session="s"))
        adapter.end_session("s")
        result = adapter.retrieve(
            RetrievalRequest(query_id="q", text="text", top_k=100)
        )
        self.assertEqual(len(result.items), 7)


class OpenAIMultimodalEmbedderTest(unittest.TestCase):
    def test_multimodal_payload_preserves_mode_text_and_image(self):
        embedder = OpenAIMemoryEmbedder(
            base_url="http://127.0.0.1:1/v1",
            model_name="fake",
            expected_dim=2,
        )
        captured = {}

        def request(payload):
            captured.update(payload)
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

        embedder._request = request  # type: ignore[method-assign]
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"test")
            value = embedder.embed_multimodal(
                "question", [str(image)], mode="query"
            )
        self.assertEqual(value.tolist(), [1.0, 0.0])
        self.assertEqual(captured["mode"], "query")
        content = captured["messages"][0]["content"]
        self.assertEqual(content[-1], {"type": "text", "text": "question"})
        self.assertEqual(content[0]["type"], "image_url")


if __name__ == "__main__":
    unittest.main()
