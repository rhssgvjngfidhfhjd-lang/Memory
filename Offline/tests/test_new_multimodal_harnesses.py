from __future__ import annotations

import json
import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.memeye_harness.dataset import load_memeye_samples
from benchmarks.memeye_harness import prompts as memeye_prompts
from benchmarks.baseline_runtime.m3_inputs import build_m3_chunks_from_round_chunks
from benchmarks.baseline_runtime.output_layout import BaselineOutputLayout
from benchmarks.baseline_runtime.parallel_runner import save_sample_artifact
from benchmarks.baseline_runtime.protocol import (
    NativeAnswerResult,
    RetrievalResult,
    RetrievedMemory,
)
from benchmarks.memlens_harness.dataset import load_memlens_samples
from benchmarks.memlens_harness import prompts as memlens_prompts
from benchmarks.multimodal_dataset_harness.runner import (
    HarnessQuestion,
    HarnessSample,
    _answer_job,
    _prepare_sample,
    _runtime_config,
    _sample_signature,
    add_common_arguments,
    validate_common_arguments,
    validate_samples,
)
from benchmarks.zero_hit import ZERO_HIT_PROMPT_MARKER
from embedding.chunk_builder import Chunk
from scripts.judge_results_llm_parallel import (
    normalize_judge_row,
    validate_protocol_snapshot,
)


class NewMultimodalHarnessTest(unittest.TestCase):

    def test_zero_hit_answer_marker_does_not_create_memory(self):
        class FakeClient:
            retries = 2

            def answer_messages_with_usage(self, **kwargs):
                self.request = kwargs
                return type(
                    "Response",
                    (),
                    {
                        "text": "<answer>Insufficient information</answer>",
                        "usage": {},
                        "attempts": 1,
                        "failed_attempts": 0,
                        "image_count": 0,
                    },
                )()

        client = FakeClient()
        job = {
            "query_id": "memlens:q1",
            "manifest_question_id": "memlens:q1",
            "sample_id": "q1",
            "source_name": "dataset_32k.json",
            "question_id": "q1",
            "question": "When?",
            "category": "information_extraction",
            "clue": [],
            "memory_items": [],
            "retrieval_top_k": [],
            "retrieval_method_trace": {},
            "query_image": None,
            "question_metadata": {},
            "native_answer": None,
        }
        result, trace = _answer_job(
            client,
            job,
            benchmark="MEMLENS",
            allow_answer_errors=False,
            attach_retrieved_images=True,
        )
        self.assertTrue(result["zero_hit_prompt_marker_used"])
        self.assertTrue(trace["zero_hit_prompt_marker_used"])
        self.assertEqual(client.request["memory_items"], [])
        self.assertIn(ZERO_HIT_PROMPT_MARKER, str(client.request["messages"]))
    @staticmethod
    def _mirix_args(result_dir: Path, *extra: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, default_data_dir=Path("."))
        args = parser.parse_args(
            [
                "--baseline",
                "MIRIX",
                "--result-dir",
                str(result_dir),
                "--skip-model-check",
                *extra,
            ]
        )
        validate_common_arguments(parser, args)
        return args

    def test_memlens_maps_one_question_to_an_isolated_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "release_images" / "needle_images" / "a.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
            payload = [
                {
                    "question_id": "q1",
                    "question_type": "information_extraction",
                    "question_subtype": "entity",
                    "question": "What was shown?",
                    "answer": "A cup",
                    "question_date": "2026-01-02",
                    "haystack_dates": ["2026-01-01"],
                    "haystack_session_ids": ["s1"],
                    "haystack_sessions": [[
                        {
                            "role": "user",
                            "content": "Remember this.",
                            "has_answer": True,
                            "images": [{
                                "file": "needle_images/a.jpg",
                                "blip_caption": "a blue cup",
                            }],
                        },
                        {"role": "assistant", "content": "Okay.", "images": []},
                    ]],
                    "answer_session_ids": ["s1"],
                }
            ]
            (root / "dataset_32k.json").write_text(json.dumps(payload))
            samples = load_memlens_samples(root)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "q1")
            self.assertEqual(samples[0].questions[0].clue_ids, ["s1:R0001"])
            self.assertIn("image_caption: a blue cup", samples[0].chunks[0].text)
            self.assertNotIn("has_answer", samples[0].chunks[0].text)
            m3_chunks = build_m3_chunks_from_round_chunks(
                samples[0].chunks, benchmark="memlens"
            )
            observation = m3_chunks[0].metadata["m3_observation"]
            self.assertEqual(observation["input_mode"], "dialogue_round_as_clip")
            self.assertEqual(
                [turn["text"] for turn in observation["turns"]],
                ["Remember this.", "Okay."],
            )
            self.assertEqual(observation["images"][0]["image_id"], "a.jpg")
            self.assertEqual(
                samples[0].chunks[0].metadata["m2a_turns"][0]["images"],
                [str(image.resolve())],
            )
            self.assertEqual(validate_samples(samples)["questions"], 1)

    def test_memeye_resolves_memory_and_query_images_from_image_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dialog_root = root / "dialog"
            image_root = root / "image" / "Task"
            dialog_root.mkdir()
            image_root.mkdir(parents=True)
            (image_root / "memory.jpg").write_bytes(b"memory")
            (image_root / "query.jpg").write_bytes(b"query")
            payload = {
                "character_profile": {},
                "multi_session_dialogues": [{
                    "session_id": "D1",
                    "date": "2026-01-01",
                    "dialogues": [{
                        "round": "D1:R1",
                        "user": "Look",
                        "assistant": "Noted",
                        "input_image": ["Task/memory.jpg"],
                        "image_caption": ["a room"],
                    }],
                }],
                "human-annotated QAs": [{
                    "question_id": "Q1",
                    "question": "What room?",
                    "answer": "Kitchen",
                    "point": [["X2"], ["Y1"]],
                    "session_id": ["D1"],
                    "clue": ["D1:R1"],
                    "question_image": "Task/query.jpg",
                }],
            }
            (dialog_root / "Task_Open.json").write_text(json.dumps(payload))
            samples = load_memeye_samples(root)

            self.assertEqual(samples[0].sample_id, "Task")
            self.assertEqual(samples[0].questions[0].category, "X2/Y1")
            self.assertEqual(samples[0].questions[0].session_ids, ["D1"])
            self.assertEqual(samples[0].questions[0].visible_session_ids, [])
            self.assertEqual(samples[0].chunks[0].images, [str((image_root / 'memory.jpg').resolve())])
            self.assertEqual(
                samples[0].questions[0].query_image["path"],
                str((image_root / "query.jpg").resolve()),
            )
            self.assertEqual(validate_samples(samples)["query_images"], 1)
            m3_chunks = build_m3_chunks_from_round_chunks(
                samples[0].chunks, benchmark="memeye"
            )
            observation = m3_chunks[0].metadata["m3_observation"]
            self.assertEqual(observation["session_id"], "D1")
            self.assertEqual(observation["dialogue_id"], "D1:R1")
            self.assertEqual([turn["text"] for turn in observation["turns"]], ["Look", "Noted"])
            self.assertEqual(observation["images"][0]["caption"], "a room")

            repository_root = root / "repository"
            (repository_root / "data").mkdir(parents=True)
            (repository_root / "data" / "dialog").symlink_to(dialog_root)
            (repository_root / "data" / "image").symlink_to(root / "image")
            self.assertEqual(
                load_memeye_samples(repository_root)[0].sample_id,
                "Task",
            )

    def test_m3_alias_is_accepted_and_enables_final_retrieved_images(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, default_data_dir=Path("."))
        args = parser.parse_args(["--baseline", "m3-agent", "--validate-only"])
        validate_common_arguments(parser, args)
        self.assertEqual(args.baseline, "M3-Agent-caption")
        self.assertTrue(args.attach_retrieved_images)

    def test_m2a_is_accepted_without_final_retrieved_images(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, default_data_dir=Path("."))
        args = parser.parse_args(["--baseline", "M2A", "--validate-only"])
        validate_common_arguments(parser, args)
        self.assertEqual(args.baseline, "M2A")
        self.assertFalse(args.attach_retrieved_images)

    def test_m3_control_ablation_arguments_reach_runtime_config(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, default_data_dir=Path("."))
        args = parser.parse_args(
            [
                "--baseline",
                "M3-Agent-caption",
                "--m3-control-rounds",
                "1",
                "--m3-search-top-k",
                "1",
                "--m3-retrieval-threshold",
                "0.8",
                "--validate-only",
            ]
        )
        validate_common_arguments(parser, args)
        config = _runtime_config(args)
        self.assertEqual(config["m3_control_rounds"], 1)
        self.assertEqual(config["m3_search_top_k"], 1)
        self.assertEqual(config["m3_retrieval_threshold"], 0.8)

    def test_m3_reuse_root_is_forwarded_as_sample_state(self):
        class FakeAdapter:
            def reset(self, sample_id, state_dir):
                raise RuntimeError("stop after adapter construction")

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parser = argparse.ArgumentParser()
            add_common_arguments(parser, default_data_dir=Path("."))
            args = parser.parse_args(
                [
                    "--baseline",
                    "M3-Agent-caption",
                    "--result-dir",
                    str(root / "results"),
                    "--m3-reuse-state-root",
                    str(root / "source"),
                ]
            )
            validate_common_arguments(parser, args)
            sample = HarnessSample(
                sample_id="sample-a",
                source_name="source.json",
                source_path=root / "source.json",
                chunks=[],
                questions=[],
            )
            sample.source_path.write_text("{}", encoding="utf-8")
            output_layout = BaselineOutputLayout(Path(args.result_dir))
            state_root = output_layout.state_root("")
            state_root.mkdir(parents=True)
            with patch(
                "benchmarks.multimodal_dataset_harness.runner.create_adapter",
                return_value=FakeAdapter(),
            ) as create:
                with self.assertRaisesRegex(RuntimeError, "stop after adapter construction"):
                    _prepare_sample(
                        args=args,
                        benchmark="MemEye",
                        sample=sample,
                        output_layout=output_layout,
                        state_root=state_root,
                    )
            config = create.call_args.kwargs["config_overrides"]
            self.assertEqual(
                config["m3_reuse_sample_state"],
                str((root / "source" / "sample-a").resolve()),
            )

    def test_mirix_is_accepted_with_native_top7_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._mirix_args(Path(directory))
        self.assertEqual(args.baseline, "MIRIX")
        self.assertFalse(args.attach_retrieved_images)
        config = _runtime_config(args)
        self.assertTrue(config["executor_native_tool_calls"])
        self.assertEqual(config["top_k"], 7)

    def test_mirix_accepts_top5_configuration(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, default_data_dir=Path("."))
        args = parser.parse_args(
            ["--baseline", "MIRIX", "--top-k", "5", "--validate-only"]
        )
        validate_common_arguments(parser, args)
        self.assertEqual(args.top_k, 5)

    def test_mirix_preparation_uses_native_answer_and_skips_one_bad_build_point(self):
        class FakeAdapter:
            def __init__(self):
                self.ingested: list[str] = []
                self.retrieval_request = None
                self.answer_request = None
                self.closed = False

            def reset(self, sample_id, state_dir):
                self.reset_args = (sample_id, state_dir)

            def ingest(self, chunk):
                self.ingested.append(chunk.chunk_id)
                if chunk.chunk_id == "bad":
                    raise ValueError("incomplete native tool-call JSON")

            def end_session(self, session_id):
                self.ended = session_id

            def retrieve(self, request):
                self.retrieval_request = request
                return RetrievalResult(
                    items=[
                        RetrievedMemory(
                            memory_id="episodic_memory_manager:m1",
                            text="The cup is blue.",
                            session_id="S1",
                            source_dialogue_ids=["S1:R2"],
                        )
                    ],
                    trace={"via": "native_chat_agent_tools"},
                )

            def answer_with_memory(self, request):
                self.answer_request = request
                return NativeAnswerResult(
                    text="<answer>Blue</answer>",
                    usage={"total_tokens": 12},
                    trace={"via": "mirix_native_chat_agent"},
                    retrieval=request.retrieval,
                )

            def snapshot(self):
                return []

            def close(self):
                self.closed = True

        class NoFallbackClient:
            retries = 2

            def answer_messages_with_usage(self, **kwargs):
                raise AssertionError("generic answer client must not be called for MIRIX")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("{}")
            result_dir = root / "result"
            args = self._mirix_args(
                result_dir,
                "--mirix-skip-failed-build-points",
            )
            sample = HarnessSample(
                sample_id="sample",
                source_name="source",
                source_path=source,
                chunks=[
                    Chunk("bad", "bad input", metadata={"session_id": "S1"}),
                    Chunk(
                        "good",
                        "The cup is blue.",
                        metadata={"session_id": "S1", "dialogue_id": "S1:R2"},
                    ),
                ],
                questions=[
                    HarnessQuestion(
                        question_id="Q1",
                        question="What color is the cup?",
                        answer="Blue",
                        category="information_extraction",
                        clue_ids=["S1:R2"],
                    )
                ],
            )
            adapter = FakeAdapter()
            with patch(
                "benchmarks.multimodal_dataset_harness.runner.create_adapter",
                return_value=adapter,
            ):
                artifact = _prepare_sample(
                    args=args,
                    benchmark="MEMLENS",
                    sample=sample,
                    output_layout=BaselineOutputLayout(result_dir),
                    state_root=result_dir / "memory" / "datasets",
                )
            self.assertEqual(adapter.ingested, ["bad", "good"])
            self.assertEqual(len(artifact["build_failures"]), 1)
            self.assertEqual(
                artifact["build_failures"][0]["event"], "skipped_build_point"
            )
            self.assertEqual(
                adapter.retrieval_request.text, "What color is the cup?"
            )
            self.assertEqual(adapter.answer_request.top_k, 7)
            self.assertEqual(
                artifact["jobs"][0]["native_answer"]["trace"]["via"],
                "mirix_native_chat_agent",
            )
            result, trace = _answer_job(
                NoFallbackClient(),
                artifact["jobs"][0],
                benchmark="MEMLENS",
                allow_answer_errors=False,
                attach_retrieved_images=False,
            )
            self.assertEqual(result["system_answer"], "Blue")
            self.assertEqual(
                result["native_answer_trace"]["via"], "mirix_native_chat_agent"
            )
            self.assertEqual(trace["top_k"][0]["memory_id"], "episodic_memory_manager:m1")
            self.assertTrue(adapter.closed)

    def test_mirix_resume_keeps_completed_native_qa(self):
        class FakeAdapter:
            def __init__(self):
                self.retrieved_query_ids: list[str] = []

            def reset(self, sample_id, state_dir):
                pass

            def filter_completed_session_chunks(self, chunks):
                return chunks

            def retrieve(self, request):
                self.retrieved_query_ids.append(request.query_id)
                return RetrievalResult(trace={"via": "native_chat_agent_tools"})

            def answer_with_memory(self, request):
                return NativeAnswerResult(
                    text="<answer>second</answer>",
                    trace={"via": "mirix_native_chat_agent"},
                    retrieval=request.retrieval,
                )

            def snapshot(self):
                return []

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("{}")
            result_dir = root / "result"
            args = self._mirix_args(result_dir, "--resume")
            questions = [
                HarnessQuestion("Q1", "First?", "first", "test", []),
                HarnessQuestion("Q2", "Second?", "second", "test", []),
            ]
            sample = HarnessSample("sample", "source", source, [], questions)
            signature = _sample_signature(args, "MemEye", sample, [])
            first_query_id = "memeye:sample:Q1"
            completed_job = {
                "query_id": first_query_id,
                "native_answer": {"text": "<answer>first</answer>"},
            }
            layout = BaselineOutputLayout(result_dir)
            save_sample_artifact(
                layout.sample_checkpoint_dir,
                sample.sample_id,
                signature=signature,
                artifact={
                    "sample_id": sample.sample_id,
                    "jobs": [completed_job],
                    "build_failures": [],
                    "complete": False,
                },
            )
            adapter = FakeAdapter()
            with patch(
                "benchmarks.multimodal_dataset_harness.runner.create_adapter",
                return_value=adapter,
            ):
                artifact = _prepare_sample(
                    args=args,
                    benchmark="MemEye",
                    sample=sample,
                    output_layout=layout,
                    state_root=result_dir / "memory" / "datasets",
                )
            self.assertEqual(
                adapter.retrieved_query_ids,
                ["memeye:sample:Q2"],
            )
            self.assertEqual(
                [job["query_id"] for job in artifact["jobs"]],
                [first_query_id, "memeye:sample:Q2"],
            )
            self.assertTrue(artifact["complete"])

    def test_m3_accepts_top_five_for_the_final_handoff(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, default_data_dir=Path("."))
        args = parser.parse_args(
            ["--baseline", "m3-agent", "--top-k", "5", "--validate-only"]
        )
        validate_common_arguments(parser, args)
        self.assertEqual(args.top_k, 5)

    def test_prompt_uses_answer_tag_contract(self):
        for prompt_module in (memeye_prompts, memlens_prompts):
            with self.subTest(prompt_module=prompt_module.__name__):
                messages = prompt_module.build_answer_messages(
                    question="Question?",
                    question_type="X3/Y2",
                    memory_evidence=["Evidence"],
                    question_date="2024/05/31 (Fri) 07:58",
                )
                self.assertEqual(
                    [row["role"] for row in messages], ["system", "user"]
                )
                self.assertIn("<answer>", messages[0]["content"])
                self.assertEqual(
                    len(prompt_module.prompt_manifest()["prompt_sha256"]), 64
                )

    def test_judge_accepts_both_new_result_schemas(self):
        validate_protocol_snapshot()
        row = {
            "sample_id": "sample",
            "question_id": "Q1",
            "system_answer": "prediction",
            "original_answer": "reference",
        }
        self.assertEqual(
            normalize_judge_row("memlens", row, 1)["protocol_id"],
            "memlens_answer_v1",
        )
        self.assertEqual(
            normalize_judge_row("memeye", row, 1)["protocol_id"],
            "memeye_answer_v1",
        )


if __name__ == "__main__":
    unittest.main()
