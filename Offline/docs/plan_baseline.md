# Manifest Test Baseline Matrix Plan

## Scope

Run six baselines on the manifest-defined test split for three benchmarks,
excluding MemVerse. The split manifest is the sole authority for conversation
membership, question membership, and question order.

Baselines:

1. M3-Agent-caption
2. M2A
3. AUGUSTUSMemory
4. MMA
5. MIRIX
6. OmniSimpleMem

Benchmarks and required QA counts:

- Mem-Gallery: four complete conversations, 275 questions.
- WorldMemArena lifelong: eight complete samples, 440 questions including 40 MB questions.
- H2HMEM: four dyadic dialogues with 327 questions plus one multiparty dialogue with 33 questions.

## Fixed configuration

- Answer model: `Qwen/Qwen3-VL-4B-Instruct`
- Answer endpoints on GPU3/4/5: ports 8013/8014/8015
- Embedding model: `Qwen/Qwen3-VL-Embedding-2B`
- Embedding endpoint: port 8001, dimension 2048
- `top_k=7`, temperature 0
- Request timeout 180 seconds, two retries
- Judge: OpenRouter `openai/gpt-4o-mini`

## Strict data contract

- Build memory from every chunk in each selected test conversation/sample.
- Never build memory from train or validation conversations.
- Retrieve and answer only question IDs listed by the manifest.
- Preserve manifest conversation order and per-conversation question order.
- For WMA, ingest sessions chronologically and restrict every checkpoint to its
  visible session prefix; output remains in manifest question order.
- `results.json`, `retrieval_trace.jsonl`, and `pipeline_qa.jsonl` must contain
  identical ordered `manifest_question_id` sequences.
- Missing, extra, duplicated, or reordered IDs are fatal before Judge starts.

## Priority order

The shared three-worker queue must use this exact order. GPU assignment is
dynamic: each free GPU takes the earliest remaining job.

1. M3-Agent-caption x H2HMEM
2. M3-Agent-caption x Mem-Gallery
3. M3-Agent-caption x WorldMemArena
4. M2A x Mem-Gallery
5. M2A x WorldMemArena
6. M2A x H2HMEM
7. AUGUSTUSMemory x H2HMEM
8. AUGUSTUSMemory x WorldMemArena
9. AUGUSTUSMemory x Mem-Gallery
10. MMA x Mem-Gallery
11. MIRIX x Mem-Gallery
12. MMA x H2HMEM
13. OmniSimpleMem x Mem-Gallery
14. OmniSimpleMem x H2HMEM
15. MIRIX x H2HMEM
16. MMA x WorldMemArena
17. MIRIX x WorldMemArena
18. OmniSimpleMem x WorldMemArena

## Start gates

1. Validate the full manifest against source data: 275/440/327/33.
2. Generate a derived smoke manifest, exercise all three benchmark harnesses
   and all six baseline adapters through the strict manifest path.
3. Verify all three answer endpoints, the embedding endpoint, and a minimal
   OpenRouter Judge completion.
4. Use a fresh output root so invalid legacy QA checkpoints cannot be reused.

## Outputs and completion

Every job must contain `results.json`, `retrieval_trace.jsonl`,
`pipeline_qa.jsonl`, a memory snapshot and native state/index,
`run_manifest.json`, `metrics.json`, and `llm_judge_metrics.json`. Calls and
available costs must be included in metrics.

The run is complete only when all 18 jobs and all 18 Judge runs pass with no
answer/Judge errors and exact manifest order. Final reporting includes F1, EM,
Judge, Cost-MB, Cost-QA, Calls, runtime, and output path.

## Monitoring and recovery

- Scheduler and watchdog run in tmux.
- Watchdog samples GPU, service, process, and status state every 20 minutes.
- Normal work is not polled continuously.
- Only `complete` stops the watchdog; `incomplete` remains visible.
- A supervisor retries a failed scheduler safely with resume checkpoints, up
  to three launches. After that it leaves an alert for manual diagnosis rather
  than looping destructively.
