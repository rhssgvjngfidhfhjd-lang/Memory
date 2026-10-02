# MemVerse, M3-Agent, and MIRIX on MEMLENS and MemEye

The two harnesses use the same adapter protocol and artifact layout as
`memgallery_harness`. They currently expose the `MemVerse`,
`M3-Agent-caption`, and `MIRIX` adapters through the same entrypoints.

## Entrypoints

- `python -m benchmarks.memlens_harness.eval_memlens`
- `python -m benchmarks.memeye_harness.eval_memeye`

Run from `Offline/` with `PYTHONPATH=src` and the MemVerse environment:

```bash
export PYTHONPATH=src
export MEMVERSE_PYTHON=/data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python
export MIRIX_PYTHON=/data/haozhen/Memory-clean/Offline/.venvs/mirix/bin/python
```

Always validate the selected source before model calls:

```bash
"$MEMVERSE_PYTHON" -m benchmarks.memlens_harness.eval_memlens --validate-only
"$MEMVERSE_PYTHON" -m benchmarks.memeye_harness.eval_memeye --validate-only
```

## Dataset mapping

MEMLENS uses one independent MemVerse state directory per `question_id`. Each
question has its own 32K haystack, so combining questions into one memory bank
would leak unrelated contexts. User/assistant turns are paired into round-level
chunks; `has_answer` is used only to derive retrieval provenance and is never
written into chunk text or metadata.

MemEye uses one independent state directory per task JSON. Its eight task files
share the Mem-Gallery-style dialogue/QA schema. Memory and question images are
resolved relative to `MemEye/data/image`; X/Y coordinates are retained as the
QA category (for example `X3/Y2`).

Both harnesses retain released captions and image paths in the common chunk
protocol. MemVerse uses caption memory by default. MIRIX sends the chunk through
the existing official v0.1.1 six-agent absorption adapter, performs native Chat
Agent retrieval with one shared global Top-7 budget, and obtains the final
answer from that same native Chat Agent. It never falls back to the generic
answer client.

For MIRIX, `--top-k` must remain exactly `7`. The optional
`--mirix-skip-failed-build-points` policy rejects and audits incomplete native
tool output, skips isolated bad build points, resets the counter after a success,
and stops the sample at `--mirix-max-consecutive-failed-build-points` (default
10). `--resume` reuses only signature-compatible sample and SQLite checkpoints.

## Example smoke runs

These commands call the configured executor, embedding, and answer services:

```bash
"$MEMVERSE_PYTHON" -m benchmarks.memlens_harness.eval_memlens \
  --sample-id q_4106e113 \
  --result-dir outputs/MEMLENS/MemVerse/smoke \
  --sample-concurrency 1 \
  --answer-concurrency 1

"$MEMVERSE_PYTHON" -m benchmarks.memeye_harness.eval_memeye \
  --sample-id Brand_Memory_Test \
  --max-qa 1 \
  --result-dir outputs/MemEye/MemVerse/smoke \
  --sample-concurrency 1 \
  --answer-concurrency 1
```

For API execution, override `--executor-base-url`, `--answer-base-url`, models,
and embedding settings exactly as in the existing MemVerse tutorials. A new run
must not set `MEMVERSE_REUSE_STATE`. For an interrupted run with the same source,
prompt, and configuration, set `MEMVERSE_REUSE_STATE=1` and pass `--resume`.

MIRIX smoke examples use the same dataset entrypoints and result layout:

```bash
"$MIRIX_PYTHON" -m benchmarks.memlens_harness.eval_memlens \
  --baseline MIRIX \
  --sample-id q_600a79aa \
  --result-dir outputs/MEMLENS/MIRIX/smoke \
  --sample-concurrency 1 \
  --answer-concurrency 1 \
  --top-k 7 \
  --mirix-skip-failed-build-points

"$MIRIX_PYTHON" -m benchmarks.memeye_harness.eval_memeye \
  --baseline MIRIX \
  --sample-id Brand_Memory_Test \
  --max-qa 1 \
  --result-dir outputs/MemEye/MIRIX/smoke \
  --sample-concurrency 1 \
  --answer-concurrency 1 \
  --top-k 7 \
  --mirix-skip-failed-build-points
```

Set executor, answer, and embedding endpoint/model flags explicitly for the
target environment. These commands are examples only and are not run by
`--validate-only`.

## Artifacts

Each run writes:

- `results.json`
- `retrieval_trace.jsonl`
- `pipeline_qa.jsonl`
- `run_manifest.json`
- `metrics.json`
- `call_trace.jsonl` and `call_metrics.json`
- `efficiency_metrics.json` and `memory_metrics.json`
- `memory/memory_snapshot.jsonl`
- `.checkpoint/` sample and answer checkpoints

After generation, the shared Judge runner accepts both new result schemas:

```bash
"$MEMVERSE_PYTHON" scripts/judge_results_llm_parallel.py \
  --benchmark memlens \
  --results outputs/MEMLENS/MemVerse/<run-id>/results.json

"$MEMVERSE_PYTHON" scripts/judge_results_llm_parallel.py \
  --benchmark memeye \
  --results outputs/MemEye/MemVerse/<run-id>/results.json
```

## Current data preflight

The current local data passes full read-only validation:

- MemEye: 8 samples, 371 questions, 848 chunks, 438 memory-image references,
  and 8 query-image references.
- MEMLENS: 173 isolated samples/questions, 11,115 chunks, and 2,301
  memory-image references.
