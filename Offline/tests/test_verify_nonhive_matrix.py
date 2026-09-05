import json
import hashlib

import pytest

from scripts.verify_nonhive_matrix import (
    validate_answer_metrics,
    validate_integrated_call_metrics,
    validate_judge_metrics,
    validate_mb_call_metrics,
    validate_native_memory_artifacts,
    validate_unified_metrics,
    validate_wma_prefix_safety,
)


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_answer_and_judge_metric_validation(tmp_path):
    answer = tmp_path / "metrics.json"
    _write(answer, {"count": 3, "f1": 0.25, "em": 0.5})
    validate_answer_metrics(answer, 3)

    judge = tmp_path / "judge.json"
    _write(
        judge,
        {
            "count": 3,
            "valid_count": 3,
            "judge_errors": 0,
            "provisional": False,
            "accuracy": 2 / 3,
        },
    )
    validate_judge_metrics(judge, 3)

    _write(answer, {"count": 2, "f1": 0.25, "em": 0.5})
    with pytest.raises(ValueError, match="expected 3"):
        validate_answer_metrics(answer, 3)


def test_answer_metric_recalculation_detects_stale_em(tmp_path):
    results = [
        {
            "system_answer": "Laura K. Simmons",
            "original_answer": "Laura K. Simmons",
            "category": "FR",
            "retrieved_source_groups": [["S01:R0001"]],
            "clue": ["S01:R0001"],
        }
    ]
    _write(tmp_path / "results.json", results)
    metrics = {
        "count": 1,
        "f1": 1.0,
        "em": 1.0,
        "retrieval_hitrate@5": 1.0,
        "by_category": {
            "FR": {
                "count": 1,
                "f1": 1.0,
                "em": 1.0,
                "retrieval_hitrate@5": 1.0,
            }
        },
    }
    path = tmp_path / "metrics.json"
    _write(path, metrics)
    validate_answer_metrics(path, 1, benchmark="Mem-Gallery")
    metrics["em"] = 0.0
    _write(path, metrics)
    with pytest.raises(ValueError, match="recalculated"):
        validate_answer_metrics(path, 1, benchmark="Mem-Gallery")


def test_deep_judge_validation_ties_raw_scores_to_source(tmp_path):
    source = [
        {"question": "q1", "category": "FR", "system_answer": "a1"},
        {"question": "q2", "category": "TR", "system_answer": "a2"},
    ]
    source_path = tmp_path / "results.json"
    _write(source_path, source)
    raw = [
        {
            "index": index,
            "question": row["question"],
            "category": row["category"],
            "prediction": row["system_answer"],
            "score": score,
            "label": "Correct" if score == 1 else "Partial",
            "judge": {
                "status": "complete",
                "model": "openai/gpt-4o-mini",
            },
        }
        for index, (row, score) in enumerate(zip(source, (1.0, 0.5)), start=1)
    ]
    _write(tmp_path / "llm_judge_results.json", raw)
    _write(
        tmp_path / "llm_judge_checkpoint.json",
        {
            "completed": 2,
            "expected": 2,
            "signature": {
                "results_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "judge": {
                    "model": "openai/gpt-4o-mini",
                    "base_url": "https://openrouter.ai/api/v1",
                    "temperature": 0.0,
                },
            },
        },
    )
    metrics_path = tmp_path / "llm_judge_metrics.json"
    _write(
        metrics_path,
        {
            "model": "openai/gpt-4o-mini",
            "count": 2,
            "valid_count": 2,
            "judge_errors": 0,
            "provisional": False,
            "correct": 1,
            "score_sum": 1.5,
            "average_score": 0.75,
            "accuracy": 0.75,
        },
    )
    validate_judge_metrics(metrics_path, 2, deep=True)
    raw[1]["prediction"] = "wrong source row"
    _write(tmp_path / "llm_judge_results.json", raw)
    with pytest.raises(ValueError, match="source result"):
        validate_judge_metrics(metrics_path, 2, deep=True)


def test_mb_call_metric_validation(tmp_path):
    path = tmp_path / "metrics.json"
    payload = {
        "available": True,
        "num_samples": 2,
        "completed_samples": 2,
        "failed_samples": 0,
        "total_calls": 5,
        "failed_calls": 1,
        "successful_calls": 4,
        "mean_per_sample": 2.5,
    }
    _write(path, payload)
    validate_mb_call_metrics(path, 2)

    payload["successful_calls"] = 5
    _write(path, payload)
    with pytest.raises(ValueError, match="invalid MB-call totals"):
        validate_mb_call_metrics(path, 2)


def test_mb_call_deep_validation_checks_config_samples_and_trace(tmp_path):
    path = tmp_path / "metrics.json"
    trace = tmp_path / "traces" / "sample.jsonl"
    trace.parent.mkdir()
    trace.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "request_id": 1,
                "method": "POST",
                "path": "/v1/chat/completions",
                "status": 200,
                "failed": False,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "available": True,
        "benchmark": "Mem-Gallery",
        "baseline": "MIRIX",
        "num_samples": 1,
        "completed_samples": 1,
        "failed_samples": 0,
        "total_calls": 1,
        "failed_calls": 0,
        "successful_calls": 1,
        "mean_per_sample": 1.0,
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "usage_missing_calls": 0,
        "samples": [
            {
                "benchmark": "Mem-Gallery",
                "baseline": "MIRIX",
                "sample_id": "sample",
                "status": "completed",
                "total_calls": 1,
                "failed_calls": 0,
                "successful_calls": 1,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "usage_available_calls": 1,
                "usage_missing_calls": 0,
                "trace_path": "/moved/repository/sample.jsonl",
            }
        ],
    }
    _write(path, payload)
    _write(
        tmp_path / "run_config.json",
        {
            "benchmark": "Mem-Gallery",
            "baseline": "MIRIX",
            "executor_model": "Qwen/Qwen3-VL-4B-Instruct",
            "executor_base_url": "http://127.0.0.1:8013/v1",
            "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
            "embedding_base_url": "http://127.0.0.1:8001/v1",
            "embedding_dim": 2048,
            "request_timeout": 180,
            "retries": 2,
            "max_samples": 0,
            "max_chunks": 0,
            "samples": ["sample"],
        },
    )
    validate_mb_call_metrics(
        path,
        1,
        benchmark="Mem-Gallery",
        method="MIRIX",
    )
    payload["samples"][0]["total_calls"] = 2
    _write(path, payload)
    with pytest.raises(ValueError, match="raw trace totals differ"):
        validate_mb_call_metrics(
            path,
            1,
            benchmark="Mem-Gallery",
            method="MIRIX",
        )

    trace_rows = [
        {
            "sample_id": "sample",
            "request_id": request_id,
            "method": "POST",
            "path": "/v1/chat/completions",
            "status": 200,
            "failed": False,
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }
        for request_id in (2, 1)
    ]
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows), encoding="utf-8"
    )
    payload.update(
        {
            "total_calls": 2,
            "successful_calls": 2,
            "mean_per_sample": 2.0,
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
        }
    )
    payload["samples"][0].update(
        {
            "total_calls": 2,
            "successful_calls": 2,
            "prompt_tokens": 20,
            "completion_tokens": 4,
            "total_tokens": 24,
            "usage_available_calls": 2,
        }
    )
    _write(path, payload)
    validate_mb_call_metrics(path, 1, benchmark="Mem-Gallery", method="MIRIX")

    trace_rows[0]["request_id"] = 1
    trace.write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicated or incomplete"):
        validate_mb_call_metrics(path, 1, benchmark="Mem-Gallery", method="MIRIX")


def test_integrated_call_metric_validation(tmp_path):
    measured = tmp_path / "measured.json"
    formal = tmp_path / "formal.json"
    _write(
        measured,
        {
            "total_calls": 5,
            "failed_calls": 1,
            "successful_calls": 4,
            "num_samples": 2,
        },
    )
    _write(
        formal,
        {
            "calls": {
                "memory_bank": {
                    "available": True,
                    "total_calls": 5,
                    "failed_calls": 1,
                    "successful_calls": 4,
                    "num_samples": 2,
                },
                "qa": {"available": True, "total_calls": 3, "num_samples": 2},
                "total": {"available": True, "total_calls": 8, "num_samples": 2},
            }
        },
    )
    validate_integrated_call_metrics(formal, measured, 2)
    payload = json.loads(formal.read_text())
    payload["calls"]["memory_bank"]["total_calls"] = 6
    _write(formal, payload)
    with pytest.raises(ValueError, match="measured=5"):
        validate_integrated_call_metrics(formal, measured, 2)


def test_native_memory_validation_for_persistent_and_in_memory_methods(tmp_path):
    persistent = tmp_path / "persistent"
    for sample in ("one", "two"):
        path = persistent / "memory" / "datasets" / sample / "raw.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"sqlite")
    validate_native_memory_artifacts(persistent, "M2A", 2)
    with pytest.raises(ValueError, match="expected at least 3"):
        validate_native_memory_artifacts(persistent, "M2A", 3)

    in_memory = tmp_path / "in_memory"
    snapshot = in_memory / "memory" / "memory_snapshot.jsonl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text('{"memory_id":"one"}\n', encoding="utf-8")
    validate_native_memory_artifacts(in_memory, "M3-Agent-caption", 2)


def test_memverse_native_validation_requires_all_three_graphs_and_memories(tmp_path):
    result_dir = tmp_path / "memverse"
    for memory_type in ("core", "episodic", "semantic"):
        chunk = (
            result_dir
            / "memory"
            / "datasets"
            / "sample"
            / "memory_chunks"
            / f"{memory_type}_memory.json"
        )
        graph = (
            result_dir
            / "memory"
            / "datasets"
            / "sample"
            / "graph"
            / memory_type
            / "vdb_entities.json"
        )
        chunk.parent.mkdir(parents=True, exist_ok=True)
        graph.parent.mkdir(parents=True, exist_ok=True)
        chunk.write_text("[]", encoding="utf-8")
        graph.write_text("{}", encoding="utf-8")
    validate_native_memory_artifacts(result_dir, "MemVerse", 1)
    graph.unlink()
    with pytest.raises(ValueError, match="vdb_entities"):
        validate_native_memory_artifacts(result_dir, "MemVerse", 1)


def test_unified_metrics_match_judge_and_summary(tmp_path):
    metrics = {
        "f1": 0.25,
        "em": 0.5,
        "llm_judge": 0.75,
        "count": 2,
        "calls": {"total": {"total_calls": 8}},
    }
    _write(tmp_path / "metrics.json", metrics)
    _write(tmp_path / "summary.json", metrics)
    _write(tmp_path / "llm_judge_metrics.json", {"accuracy": 0.75})
    validate_unified_metrics(tmp_path, 2)
    metrics["llm_judge"] = 0.5
    _write(tmp_path / "summary.json", metrics)
    with pytest.raises(ValueError, match="summary"):
        validate_unified_metrics(tmp_path, 2)


def test_wma_prefix_safety_rejects_future_retrieval(tmp_path):
    trace = tmp_path / "retrieval_trace.jsonl"
    rows = [
        {
            "visible_sessions": ["S00", "S01"],
            "covered_sessions": ["S01"],
            "top_k": [
                {
                    "session_id": "S01",
                    "source_dialogue_ids": ["S01:R0001"],
                }
            ],
        },
        {
            "visible_sessions": ["S00", "S01", "S02"],
            "covered_sessions": ["S02"],
            "top_k": [],
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    validate_wma_prefix_safety(tmp_path, 2)

    rows[1]["top_k"] = [
        {"session_id": "S03", "source_dialogue_ids": ["S03:R0001"]}
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="future"):
        validate_wma_prefix_safety(tmp_path, 2)
