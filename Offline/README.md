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

Baseline benchmark runs use the benchmark/baseline hierarchy directly:

```text
outputs/
├── Mem-Gallery/<baseline>/
├── WorldMemArena/<baseline>/
└── H2HMEM-main/<baseline>/
```

Each baseline directory keeps comparable evaluation files at its root. Native
memory state and the normalized snapshot are isolated below `memory/`:

```text
outputs/<benchmark>/<baseline>/
├── results.json
├── metrics.json
├── retrieval_trace.jsonl
├── run_manifest.json
└── memory/
    ├── memory_snapshot.jsonl
    └── datasets/<sample>/
```

For HiveMem, pass `<baseline>/memory` as `--index-root`; its existing
`datasets/<sample>/memories.jsonl` and vector layout is retained. For native
baselines, `--baseline-state-dir` defaults to `<baseline>/memory/datasets`.

H2HMem A/B exchange chunks are generated separately for dyadic and multi-party
data:

```bash
python scripts/build_chunks.py --benchmark h2hmem --variant dyadic \
  --output data/h2hmem/chunks_dyadic.jsonl
python scripts/build_chunks.py --benchmark h2hmem --variant multiparty \
  --output data/h2hmem/chunks_multiparty.jsonl
```

Consecutive messages from the same speaker are merged before adjacent speaker
blocks are paired. `questions.json` and answer metadata are never loaded by the
chunk builder.

## Setup

```bash
pip install -e .          # src-layout install; removes any need for PYTHONPATH
```

Serve the executor/answer model (Qwen3-VL-4B-Instruct) with vLLM on `:8000`.

The executor uses the chunk's original image as its visual evidence by default
(`executor_visual_input: image` in `configs/defaults.json`) and removes the
`image_caption` text. Use `--executor-visual-input caption` to reproduce the
legacy caption-only build, or `--executor-visual-input image_caption` to send
both forms of visual evidence for an ablation experiment.

## Pipeline

```bash
# 1. Build memory banks (executor LLM + embedder GPU)
python -m hive_mem.build_memories --mode c \
  --chunks data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl --all-datasets \
  --profiles-file configs/profiles.json \
  --output-root outputs/<run> \
  --executor-model Qwen/Qwen3-VL-4B-Instruct --executor-base-url http://127.0.0.1:18000/v1 \
  --executor-visual-input image \
  --device cuda:0

# 2. Deterministic edges (temporal chain; entity/attribute pairs derived at load time)
python -m hive_mem.build_memory_edges outputs/<run>/datasets/*

# 3. QA eval (baseline = pure vector; add --graph-retrieval --graph-mode append for graph)
python -m benchmarks.memgallery_harness.eval_memgallery --all-datasets \
  --data-dir ../Mem-Gallery/benchmark/data \
  --index-root outputs/<run> \
  --result-dir outputs/<run>/results/<name>

# 4. LLM judge
python scripts/judge_results_llm_parallel.py --results <result-dir>/results.json \
  --key-file <nvidia-api-keyfile> --key-count 12
```

The query stage excludes evidence-refusal questions by default: Mem-Gallery
skips `AR`, and the WorldMemArena runner skips `MB`. The raw datasets and
original QA indices remain unchanged. Override the policy with
`--exclude-categories` (pass an empty value to include every category).

## Evidence-policy train/validation/test manifest

The PPO evidence policy can use a conversation-level multimodal split manifest
instead of the legacy per-config `split` lists. The checked manifest contains
57/8/17 conversations and 3,766/562/1,075 questions for train/validation/test
(approximately 70%/10%/20%). `validation` in the training CLI maps to `val` in
the manifest. A whole conversation always remains in one split.

Validate an existing Mem-Gallery or WorldMemArena evidence-policy setup without
rewriting its config:

```bash
python scripts/evidence_policy.py \
  --config configs/evidence_policy.json \
  --split-manifest /path/to/multimodal_split_manifest.json \
  prepare-split
```

Train or evaluate with the same manifest:

```bash
python scripts/evidence_policy.py --config configs/evidence_policy.json \
  --split-manifest /path/to/multimodal_split_manifest.json train
python scripts/evidence_policy.py --config configs/evidence_policy.json \
  --split-manifest /path/to/multimodal_split_manifest.json eval \
  --strategy ppo --split validation --checkpoint <checkpoint.pt>
```

Materialize read-only JSONL indexes directly from all three source repositories:

```bash
python -m evidence_policy.episode_sources \
  --manifest /path/to/multimodal_split_manifest.json \
  --workspace-root .. \
  --output outputs/evidence_policy_splits
```

The source adapters cover H2HMem dyadic/multi-party, Mem-Gallery, and
WorldMemArena lifelong data. H2HMem can be indexed and validated now, but PPO
training on it additionally requires H2HMem memory banks and query-embedding
caches in the same 2,048-dimensional space. WorldMemArena `MB` remains a
second-stage evidence-policy category filter: retaining the current `MB`
exclusion yields 3,631/547/1,035 effective episodes while preserving the same
conversation boundaries.

Run tests: `python -m unittest discover -s tests -t .`
