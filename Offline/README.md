cd /data1/haozhen/Visual_Primitives/Offline/Offline

PYTHONPATH=$PWD python -u offline_omni_memory/benchmarks/memgallery/run_memgallery.py \
  --all-datasets \
  --max-qa 0 \
  --index-dir artifacts/faiss_index \
  --query-embedding-dir artifacts/query_embeddings \
  --result-dir artifacts/results/memgallery20_text_caption_image_qwen3vl4b_manual_$(date +%Y%m%d_%H%M) \
  --answer-backend openai \
  --answer-base-url http://localhost:8000/v1 \
  --answer-model Qwen/Qwen3-VL-4B-Instruct \
  --answer-api-key EMPTY \
  --num-predict 8000 \
  --request-timeout 180 \
  --retries 2 \
  --no-think





# Offline Mem-Gallery Qwen3-VL Chunk Index

This directory implements `plan1_chunk.md` for Mem-Gallery:

- one dialogue round -> one chunk
- text-only and image-text chunks share one embedding space
- image rounds pass real image files to `Qwen/Qwen3-VL-Embedding-2B`
- one normalized `faiss.IndexFlatIP(2048)` index
- metadata is stored outside FAISS

## Data

Input data:

```text
/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data
```

Generated chunk file:

```text
artifacts/chunks.jsonl
```

Current chunk build result:

```text
chunks: 3962
chunks with images: 1003
missing image paths: 0
```

## Dependencies

The current lightweight shell only has `numpy`. The embedding/index steps need:

```bash
pip install torch transformers pillow qwen-vl-utils faiss-cpu
```

For GPU FAISS, install the CUDA-compatible FAISS package in your environment instead of `faiss-cpu`.

## Step 1: Build Chunks

Already run once:

```bash
python build_chunks.py \
  --data-dir /data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data \
  --output artifacts/chunks.jsonl \
  --max-tokens 800
```

This creates:

```text
artifacts/chunks.jsonl
artifacts/chunks.stats.json
```

## Step 2: Embed Chunks on Four GPUs

Run in an environment with torch/transformers:

```bash
python embed_chunks.py \
  --chunks artifacts/chunks.jsonl \
  --out-dir artifacts/embeddings \
  --model-name Qwen/Qwen3-VL-Embedding-2B \
  --dim 2048 \
  --devices 0,1,2,3 \
  --dtype bfloat16
```

For an offline cache-only run:

```bash
python embed_chunks.py \
  --chunks artifacts/chunks.jsonl \
  --out-dir artifacts/embeddings \
  --model-name Qwen/Qwen3-VL-Embedding-2B \
  --dim 2048 \
  --devices 0,1,2,3 \
  --dtype bfloat16 \
  --local-files-only
```

Outputs:

```text
artifacts/embeddings/manifest.json
artifacts/embeddings/embeddings_worker0.npy
artifacts/embeddings/ids_worker0.json
...
```

## Step 3: Build FAISS Index

```bash
python build_faiss.py \
  --chunks artifacts/chunks.jsonl \
  --embedding-dir artifacts/embeddings \
  --index-dir artifacts/faiss_index \
  --dim 2048
```

Outputs:

```text
artifacts/faiss_index/vectors.index
artifacts/faiss_index/id_mapping.json
artifacts/faiss_index/chunks.jsonl
artifacts/faiss_index/metadata.jsonl
```

## Step 4: Query Test

```bash
python query_index.py \
  "[VS] Which image shown in 2024-06-17 is an application of artificial intelligence in the field of education?" \
  --index-dir artifacts/faiss_index \
  --top-k 5 \
  --device cuda:0 \
  --dtype bfloat16
```

## Step 5: Run Mem-Gallery QA

Optional but recommended: precompute QA query embeddings so benchmark runs do not
load the Qwen3-VL embedding model at answer time.

```bash
python build_query_embeddings.py \
  --data-dir /data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data \
  --out-dir artifacts/query_embeddings \
  --model-name Qwen/Qwen3-VL-Embedding-2B \
  --dim 2048 \
  --devices 0,1,2,3 \
  --dtype bfloat16
```

The default answer model is a vLLM/OpenAI-compatible Qwen3-VL endpoint:

```text
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=Qwen/Qwen3-VL-4B-Instruct
```

Example:

```bash
python -m offline_omni_memory.benchmarks.memgallery.run_memgallery \
  --data-name AI_Robotics_Automation_Future_Tech \
  --index-dir artifacts/faiss_index \
  --result-dir artifacts/results/offline_qwen3vl_vllm_qwen3vl4b \
  --max-qa 5 \
  --device cuda:0 \
  --dtype bfloat16 \
  --query-embedding-dir artifacts/query_embeddings \
  --answer-base-url http://localhost:8000/v1 \
  --answer-model Qwen/Qwen3-VL-4B-Instruct
```

## Code Map

- `offline_memgallery_qwen/chunk_builder.py`: parses Mem-Gallery JSON and builds dialogue-round chunks
- `offline_memgallery_qwen/qwen3vl_embedding.py`: Qwen3-VL text/image embedding wrapper
- `offline_memgallery_qwen/faiss_store.py`: one-index FAISS store and metadata reranking
- `offline_memgallery_qwen/adapter.py`: recall adapter returning Mem-Gallery multimodal memory items
- `offline_omni_memory/benchmarks/memgallery/run_memgallery.py`: Mem-Gallery QA runner using the FAISS memory index and a VLM answer model
- `build_chunks.py`: data -> chunks
- `embed_chunks.py`: chunks -> embeddings, with multi-GPU sharding
- `build_faiss.py`: embeddings -> FAISS index
- `build_query_embeddings.py`: QA questions -> cached query embeddings
- `query_index.py`: manual retrieval test

## Notes

- Images are passed as real image inputs to the embedding model.
- Captions remain in chunk text.
- All vectors are normalized before FAISS insertion/search.
- The code fails fast if embeddings are not 2048-dimensional.
