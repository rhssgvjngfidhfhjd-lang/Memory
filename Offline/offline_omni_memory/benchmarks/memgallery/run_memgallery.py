from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from offline_omni_memory.benchmarks.memgallery.adapter import MemGalleryOfflineAdapter
from offline_omni_memory.benchmarks.memgallery.metrics import summarize_results
from offline_omni_memory.benchmarks.memgallery.ollama_client import OllamaVLMClient
from offline_omni_memory.core.config import OfflineOmniConfig
from offline_omni_memory.retrieval.query_embedding_cache import QueryEmbeddingCache, make_query_id


PROMPT_DIR = Path("/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/prompt")


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


SYSTEM_PROMPT = load_prompt("sys_prompt.txt")


def resolve_question_image(data_dir: Path, qa: dict) -> dict | None:
    raw = qa.get("question_image")
    if not raw:
        return None
    if os.path.isabs(raw):
        path = raw
    elif raw.startswith("../image/"):
        path = str((data_dir / "image" / raw.replace("../image/", "")).resolve())
    else:
        path = str((data_dir / "image" / raw).resolve())
    out = {"path": path}
    if qa.get("image_caption"):
        out["caption"] = qa["image_caption"]
    return out


def format_question_prompt(question: str, category: str, speaker_a: str, speaker_b: str) -> str:
    constraint = ""
    if category == "AR":
        constraint = load_prompt("ar_prompt.txt")
    elif category == "CD":
        constraint = load_prompt("cd_prompt.txt")
    elif category == "VS":
        constraint = load_prompt("vs_prompt.txt")
    if constraint:
        constraint = "\n\n" + constraint
    return (
        f"Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} "
        "in a concise manner with the help of memory content.\n"
        "Please only provide the content of the answer, without including introductory phrases like 'answer:'.\n"
        "For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.\n\n"
        f"The current question is as follows:\n{question}{constraint}"
    )


def run_dataset(
    dataset_path: Path,
    cfg: OfflineOmniConfig,
    client: OllamaVLMClient,
    max_qa: int = 0,
    qa_start: int = 1,
    qa_end: int = 0,
    query_cache: QueryEmbeddingCache | None = None,
    allow_query_embed_fallback: bool = False,
) -> list[dict]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_name = dataset_path.stem
    profile = dataset.get("character_profile") or {}
    speaker_a = f"user ({profile.get('name')})" if profile.get("name") else "user"
    speaker_b = "assistant"
    adapter = MemGalleryOfflineAdapter(cfg, dataset_name=dataset_name)
    results = []
    qa_pairs = dataset.get("human-annotated QAs", [])
    total_qa = len(qa_pairs)
    if qa_end <= 0:
        qa_end = total_qa
    processed = 0
    for idx, qa in enumerate(qa_pairs, start=1):
        if idx < qa_start or idx > qa_end:
            continue
        if max_qa and processed >= max_qa:
            break
        processed += 1
        question = qa.get("question", "")
        category = str(qa.get("point", ""))
        query_image = resolve_question_image(Path(cfg.data_dir), qa)
        query_vector = None
        if query_cache is not None:
            query_id = make_query_id(
                dataset_name=dataset_name,
                qa_index=idx,
                category=category,
                question=question,
                query_image=query_image,
            )
            query_vector = query_cache.get_by_id(query_id)
            if query_vector is None and not allow_query_embed_fallback:
                raise KeyError(f"Missing cached query embedding: {query_id}")
        query = {"text": f"[{category}] {question}", "image": query_image, "vector": query_vector}
        memory_items = adapter.recall(query)
        prompt = format_question_prompt(question, category, speaker_a, speaker_b)
        try:
            answer = client.answer(
                system_prompt=SYSTEM_PROMPT,
                memory_items=memory_items,
                question_prompt=prompt,
                query_image=query_image,
                category=category,
            )
            error = ""
        except Exception as exc:
            answer = ""
            error = str(exc)
        session_id = qa.get("session_id", "")
        if isinstance(session_id, list):
            session_id = session_id[0] if session_id else ""
        result = {
            "sample_id": profile.get("name", dataset_name),
            "dataset": dataset_name,
            "session_id": session_id,
            "question": question,
            "category": category,
            "system_answer": answer,
            "original_answer": qa.get("answer", ""),
            "retrieved_ids": list(adapter.last_retrieved_ids),
            "clue": qa.get("clue", []) if isinstance(qa.get("clue", []), list) else [],
            "error": error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        print(f"[{dataset_name} {idx}/{total_qa}] {category} answer={answer[:80]!r} error={error[:80]!r}")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Mem-Gallery with offline Qwen3-VL FAISS memory and a VLM answer model.")
    parser.add_argument("--data-name", default="AI_Robotics_Automation_Future_Tech")
    parser.add_argument("--all-datasets", action="store_true")
    parser.add_argument("--data-dir", default="/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data")
    parser.add_argument("--index-dir", default="artifacts/faiss_index")
    parser.add_argument("--query-embedding-dir", default="")
    parser.add_argument("--embedding-model", default=None, help="Override OfflineOmniConfig.embedding_model (e.g. Qwen/Qwen3-VL-Embedding-8B).")
    parser.add_argument("--embedding-dim", type=int, default=None, help="Override OfflineOmniConfig.embedding_dim (e.g. 4096 for the 8B embedder).")
    parser.add_argument("--result-dir", default="artifacts/results/offline_qwen3vl_vllm_qwen3vl4b")
    parser.add_argument("--max-qa", type=int, default=5)
    parser.add_argument("--qa-start", type=int, default=1)
    parser.add_argument("--qa-end", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--answer-base-url", default=os.getenv("OPENAI_BASE_URL") or os.getenv("ANSWER_BASE_URL"))
    parser.add_argument("--answer-model", default=os.getenv("OPENAI_MODEL") or os.getenv("ANSWER_MODEL"))
    parser.add_argument("--answer-api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("ANSWER_API_KEY"))
    parser.add_argument("--answer-backend", choices=["openai", "ollama"], default=os.getenv("ANSWER_BACKEND"))
    parser.add_argument("--ollama-url", default=None, help="Deprecated alias for --answer-base-url with --answer-backend ollama.")
    parser.add_argument("--ollama-model", default=None, help="Deprecated alias for --answer-model with --answer-backend ollama.")
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--allow-query-embed-fallback", action="store_true")
    args = parser.parse_args()

    answer_backend = args.answer_backend
    answer_base_url = args.answer_base_url
    answer_model = args.answer_model
    if args.ollama_url:
        answer_base_url = args.ollama_url
        answer_backend = "ollama"
    if args.ollama_model:
        answer_model = args.ollama_model
        answer_backend = "ollama"

    cfg = OfflineOmniConfig.from_env(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        device=args.device,
        dtype=args.dtype,
        answer_base_url=answer_base_url,
        answer_model=answer_model,
        answer_api_key=args.answer_api_key,
        answer_backend=answer_backend,
        embedding_model=args.embedding_model,
        embedding_dim=args.embedding_dim,
    )
    client = OllamaVLMClient(
        model=cfg.answer_model,
        base_url=cfg.answer_base_url,
        api_key=cfg.answer_api_key,
        num_predict=args.num_predict,
        timeout=args.request_timeout,
        retries=args.retries,
        think=args.think,
        backend=cfg.answer_backend,
    )
    if not args.skip_model_check:
        client.assert_model_available()

    data_dir = Path(cfg.data_dir)
    query_cache = (
        QueryEmbeddingCache(args.query_embedding_dir, expected_dim=cfg.embedding_dim)
        if args.query_embedding_dir
        else None
    )
    if args.all_datasets:
        paths = sorted((data_dir / "dialog").glob("*.json"))
    else:
        paths = [data_dir / "dialog" / f"{args.data_name}.json"]

    all_results = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        all_results.extend(
            run_dataset(
                path,
                cfg,
                client,
                max_qa=args.max_qa,
                qa_start=args.qa_start,
                qa_end=args.qa_end,
                query_cache=query_cache,
                allow_query_embed_fallback=args.allow_query_embed_fallback,
            )
        )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    results_path = result_dir / "results.json"
    metrics_path = result_dir / "metrics.json"
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = summarize_results(all_results, k=5)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved results: {results_path}")
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
