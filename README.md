# Memory Baselines Workspace

Multi-project workspace for evaluating memory-augmented agent baselines
(`AgentMem`, A-Mem, M2A, SimpleMem, Omni-SimpleMem, ...) on the Mem-Gallery
and WorldMemArena (WMA) benchmarks.

## Layout

- `Offline/` — `agentmem` (the AgentMem pipeline), `memgallery_harness`
  (Mem-Gallery offline RAG + shared scoring utilities), `qwenvl_embedding_2b_rag`
  (Qwen3-VL embedding / FAISS core), build/eval scripts, `artifacts/` (run outputs).
- `WorldMemArena/` — WMA eval framework, baseline adapters, vendored baseline
  code (A-Mem, M2A, SimpleMem, Omni-SimpleMem). Originally forked from
  `UCSB-AI/WorldMemArena`; tracked directly in this repo (not a submodule).
- `Mem-Gallery/` — Mem-Gallery benchmark harness (memengine, prompts, run
  scripts). Originally from `YuanchenBei/Mem-Gallery`; tracked directly.
- `MemEye/` — MemEye visual-memory benchmark (separate project, own vendored
  baselines; not part of the AgentMem/Mem-Gallery/WMA workflow above).
  Originally from `MinghoKwok/MemEye`; tracked directly.
- `Nvida_api/` — NVIDIA API key pool helpers for LLM-judge calls.

## One-time setup after cloning

```sh
./setup_data.sh
```

This downloads the WorldMemArena and MemEye datasets from their official
public sources (`huggingface-cli`, ~11 GB total). Mem-Gallery's
`benchmark/data/` has no documented public download — the script prints
where to place it manually.

You also need to provide your own secrets locally (both are gitignored,
never commit real values):

- `WorldMemArena/.env` — copy from `WorldMemArena/.env.example` and fill in
  API keys / endpoints (see that file for the full list of vars).
- `Nvida_api/apikey` — one NVIDIA API key per line, used by the LLM-judge
  key pool (`Nvida_api/key_pool.py`).

## Notes

- `WorldMemArena`, `Mem-Gallery`, and `MemEye` are plain tracked directories
  in this repo (their original `.git` histories were dropped), so all local
  customizations (e.g. `agentmem_adapter.py` under
  `WorldMemArena/eval_framework/memory_adapters/`) are captured directly in
  this repo's history — no forking or submodule sync needed. Upstream
  updates from the original projects have to be merged in manually.
- Rebuildable caches (`Offline/artifacts/chunks.jsonl`, `query_embeddings/`,
  `embeddings*/`, `faiss_index*/`, any `*.npy`) are gitignored — regenerate
  with `build_chunks.py` / `embed_chunks.py` / `build_faiss.py` /
  `build_query_embeddings.py`.
