# HiveMem

Attribute-graph memory system for multimodal conversational agents, evaluated on the
Mem-Gallery benchmark (20 personas / 1,711 QA).

Each memory unit (MAU) is produced by a single LLM call over a dialogue chunk and
carries `summary + entities + per-entity attributes` (closed 6-type / 28-key ontology,
see `src/hive_mem/entity_schema.py`). Edges are built deterministically from these
fields — temporal chain, shared-entity and shared-attribute commonality — plus optional
LLM-confirmed event relations. Retrieval is vector top-k with optional one-hop graph
expansion (best config: append mode, +1.6pp JudgeAcc, McNemar p=0.032).

## Layout

```
src/hive_mem/            core package: build / entity schema / edges / graph retrieval
src/memgallery_harness/  Mem-Gallery eval runner (answer client, metrics, query cache)
src/embedding/           Qwen3-VL-Embedding-2B service wrapper
scripts/                 standalone tools (chunking, judge, graph viz export)
tests/                   unit tests
configs/profiles.json    dataset -> persona map (injected into build prompts)
data/qwen3_vl_embedding_2b/   default VL chunks and query embeddings
data/qwen3_embedding_0_6b/    optional text-only 0.6B chunks and query embeddings
outputs/                 experiment products: memory banks + eval results (git-ignored)
```

Each run uses one flat, predictable output root:

```text
outputs/<run>/
├── build_manifest.json
├── datasets/
│   └── <dataset>/
│       ├── memories.jsonl
│       ├── vectors/
│       │   ├── text.npy
│       │   ├── image.npy
│       │   └── image_mask.npy
│       ├── reports/
│       │   ├── build.json
│       │   ├── edges.json
│       │   └── conflicts.json
│       └── traces/
│           ├── build.jsonl
│           └── edges.jsonl
├── results/<experiment>/
│   ├── metrics.json
│   ├── llm_judge_metrics.json
│   └── summary.json
└── logs/
```

Incomplete builds temporarily use `outputs/<run>/.checkpoints/`; successful
full builds remove their checkpoint automatically.

## Setup

```bash
pip install -e .          # src-layout install; removes any need for PYTHONPATH
```

Serve the executor/answer model (Qwen3-VL-4B-Instruct) with vLLM on `:8000`.

## Pipeline

```bash
# 1. Build memory banks (executor LLM + embedder GPU)
python -m hive_mem.build_memories --mode c \
  --chunks data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl --all-datasets \
  --profiles-file configs/profiles.json \
  --output-root outputs/<run> \
  --executor-model Qwen/Qwen3-VL-4B-Instruct --executor-base-url http://127.0.0.1:8000/v1 \
  --device cuda:0

# 2. Deterministic edges (temporal chain; entity/attribute pairs derived at load time)
python -m hive_mem.build_memory_edges outputs/<run>/datasets/*

# 3. QA eval (baseline = pure vector; add --graph-retrieval --graph-mode append for graph)
python -m benchmarks.memgallery_harness.eval_memgallery --all-datasets \
  --data-dir /data/haozhen/Memory/Mem-Gallery/benchmark/data \
  --index-root outputs/<run> \
  --result-dir outputs/<run>/results/<name>

# 4. LLM judge
python scripts/judge_results_llm_parallel.py --results <result-dir>/results.json \
  --key-file <nvidia-api-keyfile> --key-count 12
```

Run tests: `python -m unittest discover -s tests -t .`
