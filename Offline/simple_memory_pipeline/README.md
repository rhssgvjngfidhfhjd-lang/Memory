# Simple Memory Pipeline

This directory contains a minimal inference-only memory extraction pipeline for raw text, document collections, and dialogue transcripts.

## What It Does

1. Normalize raw input into chunks.
2. Retrieve the most relevant existing memories with a semantic retriever.
3. Send every chunk plus the four fixed operations (`insert`, `update`, `delete`, `noop`) to the executor.
4. Parse the executor output into memory actions.
5. Apply the actions to a simple memory bank.
6. Return both the final memories and a chunk-by-chunk trace.

## Python Usage

```python
from simple_memory_pipeline import MemoryExtractionPipeline

pipeline = MemoryExtractionPipeline(
    model="qwen/qwen3-next-80b-a3b-instruct",
    api=True,
    api_base="https://your-api-base/v1",
    api_key=["YOUR_API_KEY"],
)

result = pipeline.run(
    input_data="Alice moved to Boston in 2024. She now works at a biotech startup.",
    input_type="text",
)

print(result["final_memories"])
```

## CLI Usage

### Text

```bash
python -m simple_memory_pipeline \
  --input-file ./examples/text.txt \
  --input-type text \
  --model qwen/qwen3-next-80b-a3b-instruct \
  --api \
  --api-base https://your-api-base/v1 \
  --api-key YOUR_API_KEY \
  --out-file ./tmp/text_memories.json
```

### Documents

The input file should be a JSON list of strings.

```bash
python -m simple_memory_pipeline \
  --input-file ./examples/documents.json \
  --input-type documents \
  --chunk-mode fixed-length \
  --model qwen/qwen3-next-80b-a3b-instruct \
  --api \
  --api-base https://your-api-base/v1 \
  --api-key YOUR_API_KEY
```

### Dialogue

The input file should be a JSON list of turns using either:

- `{"speaker": "...", "text": "..."}`
- `{"role": "...", "content": "..."}`

```bash
python -m simple_memory_pipeline \
  --input-file ./examples/dialogue.json \
  --input-type dialogue \
  --chunk-mode turn-pair \
  --model qwen/qwen3-next-80b-a3b-instruct \
  --api \
  --api-base https://your-api-base/v1 \
  --api-key YOUR_API_KEY
```

## Output Shape

The pipeline returns:

- `input_type`
- `chunk_mode`
- `chunks`
- `trace`
- `final_memories`

## Mem-Gallery Baseline

The Mem-Gallery adapter builds a new dynamic memory bank from the existing a/b/c
chunk files. It does not reuse their final FAISS indexes.

Audit the three inputs:

```bash
cd /data1/haozhen/Visual_Primitives/Offline/Offline
PYTHONPATH=. python -m simple_memory_pipeline.audit_memgallery
```

Build one dataset in mode a:

```bash
PYTHONPATH=. python -m simple_memory_pipeline.build_memgallery_memories \
  --mode a \
  --dataset AI_Robotics_Automation_Future_Tech \
  --executor-model Qwen/Qwen3-VL-4B-Instruct \
  --executor-base-url http://127.0.0.1:8000/v1 \
  --executor-api-key EMPTY \
  --device cuda:2
```

Run its QA evaluation:

```bash
PYTHONPATH=. python -m simple_memory_pipeline.run_memgallery_baseline \
  --index-root artifacts/simple_memory_pipeline/a_text_only \
  --query-embedding-dir artifacts/query_embeddings \
  --result-dir artifacts/simple_memory_pipeline/results/a_qwen3vl4b \
  --answer-base-url http://127.0.0.1:8000/v1 \
  --answer-model Qwen/Qwen3-VL-4B-Instruct
```

Use `--all-datasets` on both commands for the complete 20-dataset run. Builds
checkpoint after every event by default and resume automatically. Modes map to:

- `a`: text only
- `b`: text and caption
- `c`: text, caption, and a separate image-vector index

The QA output includes `retrieved_source_groups`. HIT@5 is computed over the
top five memory items and matches a clue against every source dialogue retained
by each memory.
