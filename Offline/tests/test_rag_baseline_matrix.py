from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = ROOT / "scripts" / "run_rag_baseline_matrix.py"
    spec = importlib.util.spec_from_file_location("rag_matrix_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RAGBaselineMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()
        cls.path = ROOT / "configs" / "rag_baseline_matrix.json"
        cls.config = cls.runner.load_config(cls.path)

    def test_matrix_expands_to_eighteen_formal_jobs(self):
        count = (
            len(self.config["baselines"])
            * len(self.config["benchmarks"])
            * len(self.config["answer_profiles"])
        )
        self.assertEqual(count, 18)
        self.assertEqual(self.config["top_k"], 7)
        self.assertEqual(self.config["embedding"]["dimension"], 2048)

    def test_profiles_use_required_models_and_services(self):
        local = self.config["answer_profiles"]["qwen3vl4b_local"]
        api = self.config["answer_profiles"]["gpt5mini_api"]
        self.assertEqual(local["model"], "Qwen/Qwen3-VL-4B-Instruct")
        self.assertEqual(len(local["endpoints"]), 3)
        self.assertEqual(api["model"], "openai/gpt-5-mini")
        self.assertEqual(api["workers"], 3)
        self.assertEqual(
            self.config["embedding"]["model"], "Qwen/Qwen3-VL-Embedding-2B"
        )

    def test_runner_command_selects_only_new_baselines_and_all_benchmarks(self):
        command, env, run_id = self.runner.profile_command(
            self.config,
            config_path=self.path,
            profile_name="qwen3vl4b_local",
            run_id_prefix="test_rag",
            smoke_only=True,
            skip_smoke=False,
        )
        selected_baselines = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--baseline"
        ]
        selected_benchmarks = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--benchmark"
        ]
        self.assertEqual(selected_baselines, self.config["baselines"])
        self.assertEqual(selected_benchmarks, self.config["benchmarks"])
        self.assertIn("--smoke-only", command)
        self.assertEqual(run_id, "test_rag_qwen3vl4b_local")
        self.assertEqual(env["OPENAI_API_KEY"], "EMPTY")
        manifest = command[command.index("--split-manifest") + 1]
        self.assertEqual(
            Path(manifest), ROOT / "configs" / "multimodal_split_manifest.json"
        )

    def test_shared_runner_disables_graph_expansion_for_rag_baselines(self):
        module_path = ROOT / "scripts" / "run_test_baseline_matrix.py"
        source = module_path.read_text(encoding="utf-8")
        self.assertIn(
            'job.method in {"NaiveRAG", "MuRAG", "UniversalRAG"}', source
        )
        self.assertIn('arguments.append("--no-graph-retrieval")', source)

    def test_call_contracts_cover_all_three_baselines(self):
        payload = json.loads(
            (ROOT / "configs" / "baseline_call_contracts.json").read_text()
        )
        contracts = payload["baselines"]
        for baseline in self.config["baselines"]:
            with self.subTest(baseline=baseline):
                self.assertEqual(contracts[baseline]["top_k"], 7)
                self.assertEqual(
                    contracts[baseline]["expected_qa_counts"],
                    self.config["expected_qa_counts"],
                )


if __name__ == "__main__":
    unittest.main()
