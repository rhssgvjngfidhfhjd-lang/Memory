# HiVe_mem

HiVe_mem builds multimodal memory episodes from dialogue rounds and extracts
textual anchors and visual anchors to construct a Cross-Episode Affinity Graph.
Horizontal Memory Expansion augments semantically retrieved seeds with related
graph neighbours. Vertical Evidence Composition uses a PPO-trained evidence
selector to compose Compact Summary, Raw Interaction Text, Image Caption,
Raw Image, and Grounded Visual View (GVV) evidence for each retrieved memory.
This release contains the method and its Mem-Gallery, H2HMEM, and WorldMemArena
training and evaluation interfaces.

## Repository

```text
src/                     Core memory construction, storage, affinity graphs, horizontal expansion, and shared utilities
configs/                 Method defaults, evaluation/PPO protocols, and frozen source hashes
evidence_policy/         Vertical Evidence Composition and PPO training/inference
embedding/               Data preparation, dependency checks, and query/memory embeddings
benchmarks/              Evaluation, fixed conversation splits, and dataset-specific metadata
gvv_extractor/           GVV discovery and cropping, with packaged prompts/configurations
scripts/run.sh           Single launcher for every preparation, training, and evaluation stage
outputs/audit/           Generated source snapshots and audit reports (not packaged)
```

The core `src` package has six files:

- `__init__.py`: package marker.
- `build.py`: memory construction, its model client, checkpoints, and the build command.
- `memory.py`: memory episodes, bank storage, anchors, and executor prompts/response parsing.
- `graph.py`: Cross-Episode Affinity Graph construction and checkpoint-visible prefix graphs.
- `retriever.py`: semantic retrieval and Horizontal Memory Expansion.
- `utils.py`: portable configuration, resource/output paths, and shared file utilities.

Benchmark evaluation is organized into shared code and three flat dataset harnesses:

```text
benchmarks/
├── run_hivemem.py
├── judge.py
├── multimodal_split_manifest.json
├── common/
│   ├── answer_client.py
│   ├── prompts.py
│   ├── metrics.py
│   ├── query_cache.py
│   └── utils.py
├── runtime/
│   ├── adapter.py
│   ├── evaluation.py
│   └── call_trace.py
├── memgallery_harness/
│   ├── eval_memgallery.py
│   └── profiles.json
├── h2hmem_harness/
│   └── eval_h2hmem.py
└── wma_harness/
    ├── eval_wma.py
    ├── questions.py
    └── metrics.py
```

The tree omits package `__init__.py` files. Shared prompts keep separate
Mem-Gallery, H2HMEM, and WMA builders and prompt signatures. The runtime handles
retrieval adaptation, concurrent evaluation, checkpoints, and result outputs.
WMA-specific question visibility, image rules, and scoring stay in its harness.

The embedding package contains `cli.py` for data download, fixed-split auditing,
and dependency checks; `chunks.py` for dataset readers and chunk preparation;
`backends.py` for local/API embedders and HTTP serving; and
`build_embeddings.py` for query and chunk vector caches.
`embedding.cli` loads preparation and embedding modules only for the selected
command, so auditing and dependency checks do not initialize models.

The evidence-policy package has five files:

- `__init__.py`: lightweight package marker.
- `evidence.py`: source questions, fixed splits, GVV indexes, Horizontal Memory Expansion, and EvidenceComposer.
- `ppo.py`: the evidence-selection network, PPO buffer, and policy updates.
- `rollout.py`: answer calls, rollout caching, quality/cost rewards, and the evidence-selection environment.
- `cli.py`: configuration, training/evaluation orchestration, checkpoints, and result reporting.

Dialogue chunk preparation, fixed-split auditing, and GVV-index reading work without
PyTorch. Neural evidence selection and PPO require the training dependencies.

Only this README is maintained. Raw datasets and generated artifacts are separate
from the release. Commands create their output directories when needed:

```text
data/raw/<benchmark>/                      Downloaded or manually prepared raw sources
data/<benchmark>/chunks/                   Prepared dialogue rounds
data/<benchmark>/query_embeddings/         Query vectors and metadata
outputs/memory/<benchmark>/                Memory banks, graphs, and construction checkpoints
outputs/gvv/default/                       GVV records, crops, and exported indexes
outputs/evidence_policy/<benchmark>/       PPO checkpoints, rollout cache, and metrics
outputs/evaluation/                       Graph evaluation answers, traces, logs, and checkpoints
```

Directories are created automatically; preparing a directory does not download
models or generate its inputs. Follow the stages below before training or evaluation.

## Install and configure

Run the following from the cloned repository. Use Python 3.10 or newer;
install a PyTorch build appropriate for your CPU or CUDA device:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[eval,train,embedding,data]"
cp .env.example .env
./scripts/run.sh check --group eval --group train --group embedding --group data
```

A core-only installation is `python -m pip install -e .`. GVV extraction is included
in the main package and uses the configured multimodal executor service.
Configure models, endpoints, dataset roots, and API keys in `.env`.
Answering and memory/GVV extraction require an OpenAI-compatible multimodal
chat service. They may share a service or use separate executor and answer
models. The judge uses a separately configured chat service. Embedding requires
the multimodal embedding API below or explicit local model loading.

For an NVIDIA GPU server, one answer/executor deployment option is vLLM in a
separate environment. Qwen documents Qwen3-VL support in vLLM 0.11.0 or newer;
the following is a deployment example, not a tested GPU configuration:

```bash
python -m venv .venv-vllm
source .venv-vllm/bin/activate
python -m pip install "vllm>=0.11.0"
vllm serve Qwen/Qwen3-VL-4B-Instruct --host 0.0.0.0 --port 8000 \
  --limit-mm-per-prompt '{"image":128}'
```

Use `--revision <model-commit>` to pin its weights. Set the image limit to cover
the raw images and GVV crops in your prompts, and tune context length and
concurrency for your available memory. See the
[official Qwen deployment guide](https://github.com/QwenLM/Qwen3-VL#deployment)
and [vLLM serving arguments](https://docs.vllm.ai/en/v0.11.0/cli/serve.html).
In another terminal, activate the project environment and start embedding:

```bash
source .venv/bin/activate
./scripts/run.sh serve --host 0.0.0.0 --port 8001 --device cuda:0 \
  --no-local-files-only
```

For services on the same machine, example `.env` addresses are
`HIVE_ANSWER_BASE_URL=http://127.0.0.1:8000/v1`, the same value for
`HIVE_EXECUTOR_BASE_URL`, and
`HIVE_EMBEDDING_BASE_URL=http://127.0.0.1:8001/v1`.
For remote services, use their reachable addresses. Images are sent as encoded
bytes, so the client and embedding service do not need a shared filesystem.
Set the judge URL and model to the identifiers accepted by your provider.
If a service requires authentication, fill in the corresponding role key;
the bundled embedding server does not require one.

Embedding loads weights on its first request. Verify model loading with an
actual request, for example:

```bash
curl http://127.0.0.1:8001/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":"connection check"}'
```

The default `--local-files-only` requires cached weights; use
`--no-local-files-only` for the initial download. A CPU-only embedding service
can use `--device cpu --dtype float32`; query preparation accepts `--devices cpu
--dtype float32`, and PPO accepts `--device cpu`. Actual multimodal model
inference can be slow and requires sufficient RAM/VRAM. A local CPU client can
also use remote model services.

Fill in `HIVE_ANSWER_BASE_URL`, `HIVE_EXECUTOR_BASE_URL`,
`HIVE_EMBEDDING_BASE_URL`, and `HIVE_JUDGE_BASE_URL` with your service API roots.
The release supplies no default service addresses. The build, GVV, answer, and
judge commands validate their required addresses before starting work; help,
dependency checks, GVV export, and data preparation without API calls work without
them. To embed memories directly with a local model, pass
`./scripts/run.sh build --local-embedding` together with the benchmark arguments;
this explicitly selects local model loading instead of an embedding API.

Set `HIVE_EMBEDDING_MODEL` and `HIVE_EMBEDDING_DIM` for embedding, memory
construction, and evaluation. PPO uses the same dimension for its policy inputs.
Set `HIVE_JUDGE_MODEL` when running the judge or the graph evaluation launcher.
These settings have no runtime defaults; `.env.example` supplies the example
values Qwen3-VL-Embedding-2B, 2,048 dimensions, and `openai/gpt-4o-mini`.
Explicit CLI arguments can override them; each command lists its flags in
`--help`.

The single `scripts/run.sh` launcher loads `.env` and runs from the repository root.
Python modules and installed `hivemem-*` commands require exported environment
variables. Settings
follow defaults, an optional `HIVE_CONFIG`/`HIVE_POLICY_CONFIG` JSON, environment
variables, and explicit CLI arguments. Use `HIVE_OUTPUT_ROOT` for a common generated
output root, and `HIVE_MEMORY_BANK`, `HIVE_QUERY_CACHE`, or `HIVE_GVV_RUN_DIR` for
existing artifacts. API roles use `ANSWER_API_KEY`, `EXECUTOR_API_KEY`,
`EMBEDDING_API_KEY`, and `JUDGE_API_KEY`, with optional common key fallbacks.

Default configurations live in `configs/` and are included in the installed
package. `HIVE_CONFIG` overrides runtime fields used by build/embedding/graph
evaluation, and the executor model/URL for GVV's default configuration.
It accepts either a `runtime` object or a flat runtime JSON.
PPO uses `HIVE_POLICY_CONFIG` or `train/eval policy --config`; its fields are a
policy configuration (`model`, `reward`, `ppo`, and retrieval settings), or the
`protocols` structure in `configs/experiments.json`. A runtime override does not
automatically replace PPO reward or optimizer settings. Environment variables
for the answer model, tokenizer, and embedding dimension apply to PPO as well.

Cost rewards require the answer model's complete tokenizer/processor files
locally, even when answering uses a remote API. Download them before offline
training, and set `HIVE_TOKENIZER` to that snapshot directory; processor-only
loading does not need the answer model's weight files. For example, with the
project environment and exported Hugging Face credentials when needed:

```bash
python -c 'from transformers import AutoProcessor; AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct").save_pretrained("data/models/answer-processor")'
# Then set HIVE_TOKENIZER=data/models/answer-processor in .env.
```

Pin the same processor revision as the answering service when downloading it.
When using a custom model name or service alias, add an entry under
`model_efficiency.models` in `configs/defaults.json` for that exact name, with
the pricing and latency fields shown by the existing entry. Configure the
corresponding local processor separately with `HIVE_TOKENIZER`. PPO reads prices
from its `efficiency_config` when supplied, otherwise from the packaged defaults;
graph evaluation accepts `--efficiency-config`. The included latency coefficients
are modeled estimates, and the prices are a preserved experiment profile, not
current provider quotations. Cost-enabled training also requires token usage
from the answer API. Changing prices or the tokenization protocol requires a
new training run; incompatible cost windows are rejected on resume.

Use the launcher help to find commands and their arguments:

```bash
./scripts/run.sh --help
./scripts/run.sh prepare --help
```

Commands are `check`, `prepare`, `serve`, `build`, `graph`, `gvv`, `train`, `eval`,
and `judge`. Use `eval policy` for PPO/full-evidence evaluation. Arguments after
the command are forwarded to its Python entrypoint.

## Prepare datasets

Provide one common `HIVE_DATA_ROOT` containing `memgallery/`, `h2hmem/`, and `wma/`,
or set the individual dataset roots shown in `.env.example`.

| Dataset | Required layout under its resolved root | Individual variable |
| --- | --- | --- |
| Mem-Gallery | `dialog/` and `image/` | `HIVE_MEMGALLERY_ROOT` |
| H2HMEM | `dyadic/` and `multi-party/`, with conversation scenes and images | `HIVE_H2HMEM_ROOT` |
| WorldMemArena | `project/`, `personal/`, and their image trees | `HIVE_WMA_ROOT` |

Datasets are distributed by their upstream projects: [Mem-Gallery](https://huggingface.co/datasets/Ethan-Bei/Mem-Gallery),
[H2HMEM](https://huggingface.co/datasets/varib/H2HMEM), and
[WorldMemArena](https://huggingface.co/datasets/LCZZZZ/WorldMemArena).
The downloader records the resolved upstream revision and file checksums. Omit
`--revision` to download current HEAD, or provide a full upstream commit:

```bash
./scripts/run.sh prepare download memgallery --revision <upstream-commit>
./scripts/run.sh prepare download h2hmem --revision <upstream-commit>
./scripts/run.sh prepare download wma --revision <upstream-commit>
./scripts/run.sh prepare audit --manifest-only
# Check the historical selected source files shipped with this release:
./scripts/run.sh prepare audit --reference configs/source_reference.json \
  --output outputs/audit/current.json
# Optionally record a separate reference for your own validated snapshot:
./scripts/run.sh prepare audit --output outputs/audit/reference.json
# Compare later sources against that reference and save the current report:
./scripts/run.sh prepare audit --reference outputs/audit/reference.json \
  --output outputs/audit/current.json
```

The current upstream snapshots contain directories: Mem-Gallery uses
`data/dialog/` and `data/image/`; H2HMEM uses `dyadic/` and `multi-party/`;
WMA uses `lifelong/project/` and `lifelong/personal/`. The downloader writes to
`data/raw/<benchmark>/` and the root resolver detects these nesting layouts.
Set an individual root to the directory containing the required markers when
preparing data manually. An alternative archive distribution must be extracted
to that layout before chunk preparation.

`configs/source_reference.json` contains the historical SHA-256 values of 375
selected QA/dialogue JSON files and the matching split-manifest hash. It contains
no raw data and does not cover every image. The original upstream commit IDs
were not recorded; arbitrary current HEAD downloads may differ and must pass
this comparison before claiming the same source snapshot. If a comparison
fails, inspect the report and obtain the matching historical source files;
generating a new reference does not establish historical equivalence.

Runtime audit reports are excluded from source distributions and wheels; the
small checked-in reference is packaged. Without `--output`, an audit writes
`audit/current.json` beneath the
configured output root (`HIVE_OUTPUT_ROOT` overrides it). Explicit `--output` and
`--reference` paths are used as supplied. Reference comparisons report modified,
added, and missing selected source files; `--source` limits both snapshots to the
same dataset scope. The current report cannot overwrite its reference.
An existing reference report should be retained for historical comparisons: generating a
new one records the current sources and does not reproduce an earlier snapshot.
A manifest-only report does not validate source hashes and cannot serve as a
reference report.
The checked-in `benchmarks/multimodal_split_manifest.json` keeps whole conversations in one train/validation/test
split and fixes the question allowlists: 49/16/17 conversations and
3,268/1,060/1,075 questions. This fixed manifest and
`benchmarks/memgallery_harness/profiles.json` are included in wheels.

Generate complete multimodal dialogue-round chunks and matching query vectors:

```bash
./scripts/run.sh prepare chunks --benchmark memgallery
./scripts/run.sh prepare chunks --benchmark h2hmem --variant dyadic
./scripts/run.sh prepare chunks --benchmark h2hmem --variant multiparty
./scripts/run.sh prepare chunks --benchmark wma
./scripts/run.sh prepare queries --benchmark memgallery
./scripts/run.sh prepare queries --benchmark h2hmem
./scripts/run.sh prepare queries --benchmark wma
```

The data and embedding commands can also be invoked directly:

```bash
python -m embedding.cli check --group data
python -m embedding.cli data audit --manifest-only
python -m embedding.chunks --benchmark memgallery
python -m embedding.build_embeddings queries --benchmark memgallery
python -m embedding.build_embeddings chunks --input <chunks.jsonl> --output-dir <vector-cache>
python -m embedding.backends serve
```

Query preparation uses the configured local embedding model, overridable with
`--model-name`, `--model-revision`, and `--dim`; model, weight revision, and
dimensions must match memory construction. Set `HIVE_EMBEDDING_REVISION` to a
full model commit for query preparation, the bundled embedding server, and
local memory embedding. External embedding services must use the same weights.
Explicit revision pins require caches that record the same revision; rebuild
historical query caches that do not record one. Unpinned runs retain support for
legacy caches.
H2HMEM uses the packaged split manifest for stable query IDs that include the
session. All three benchmarks use prepared multimodal query vectors in both
graph and PPO evaluation. Chunk paths are
derived from the benchmark and variant. Query caches use a directory named after
the selected embedding model, with a `lifelong/` subdirectory for WMA; default
PPO and graph evaluation commands locate the same caches automatically.
The cost configuration is located in `configs/defaults.json`; Mem-Gallery
profiles are located in `benchmarks/memgallery_harness/profiles.json`.
Use `--profiles-file` to override the default profile JSON for memory construction.
These standard paths do not need entries in
`configs/defaults.json`.

## Build memories, graphs, and grounded visual views

Start the configured model services before memory construction. Build a memory
bank and then its Cross-Episode Affinity Graph for each benchmark:

```bash
./scripts/run.sh build --benchmark memgallery --all-datasets
./scripts/run.sh graph --degree-cap 4 outputs/memory/memgallery/datasets/*
./scripts/run.sh build --benchmark h2hmem --all-datasets
./scripts/run.sh graph --degree-cap 4 outputs/memory/h2hmem/datasets/*
./scripts/run.sh build --benchmark wma --all-datasets
./scripts/run.sh graph --degree-cap 4 outputs/memory/wma/datasets/*
./scripts/run.sh gvv extract --dataset all
```

The graph degree cap is 4 in construction, graph evaluation, and PPO.
The commands use the default output root; adjust their explicit paths when
using `HIVE_OUTPUT_ROOT`. Retrieval rejects a graph built with a different cap.
Memory construction commits a checkpoint generation with its vectors and state
together, retains the previous generation for recovery, and ignores uncommitted
trace tails on resume. Finished-bank checks include text, image, and attribute
vectors. Changed build inputs or model revisions require a separate output or a
deliberate rebuild. GVV extraction writes `outputs/gvv/default/`, validates records
and crops before skipping completed images, repairs missing crops, and rebuilds
its exports. Dataset scans include optional dialog captions. The same GVV directory
is used by the default PPO configuration. `./scripts/run.sh gvv export` rebuilds exports
from existing records.

## Train and evaluate

Training requires prepared raw data, matching memory/query vectors, affinity
graphs, and complete GVV coverage. Select the benchmark with `--benchmark`:

```bash
./scripts/run.sh train --benchmark memgallery
./scripts/run.sh train --benchmark h2hmem
./scripts/run.sh train --benchmark wma
./scripts/run.sh eval policy --benchmark memgallery --strategy ppo --split test \
  --checkpoint outputs/evidence_policy/memgallery/checkpoints/<checkpoint>
./scripts/run.sh eval policy --benchmark memgallery --strategy full-evidence --split test
```

PPO chooses CUDA when available and otherwise uses CPU. It selects checkpoints
using validation; test questions remain separate. Missing questions from the
fixed manifest are errors, including when a run uses `--limit`.
Rollout cache keys include reasoning/retry settings and the content hashes of
query images and selected image/GVV evidence. Replacing an image at the same
path invalidates its old answer cache.
The preserved PPO retrieval defaults are vector 7 plus up to 2 graph neighbours
for Mem-Gallery/WMA and vector 5 plus up to 2 for H2HMEM. The graph evaluation
protocol uses vector 5 plus up to 2 graph neighbours for all three benchmarks.
The configuration files define these distinct protocols explicitly.

The retrieval settings `top_k` (or `vector_top_k`), `append_k` (or
`graph_append_k`), and `degree_cap` correspond to the paper's semantic seed count
K_ret, horizontal expansion count K_exp, and maximum graph degree K_edge.
`MemoryEpisode` represents one memory; `MemoryBank` stores the collection;
`MemoryEpisodeBuilder` constructs episodes. The Python fields `textual_anchors`
and `visual_anchors` are serialized as the paper's `Ti` and `Vi`.

The graph evaluation launcher builds a test-selected memory bank, evaluates the
fixed question allowlist, verifies result/trace counts, and runs the judge:

```bash
./scripts/run.sh eval --benchmark Mem-Gallery
./scripts/run.sh eval --benchmark H2HMEM
./scripts/run.sh eval --benchmark WorldMemArena
./scripts/run.sh judge --benchmark memgallery --results <result-directory>/results.json
```

For direct evaluation of an existing bank, use the corresponding
`python -m benchmarks.<benchmark>_harness.eval_<benchmark>` command with
`--index-root`, `--result-dir`, and the dataset/service arguments shown by `--help`.
WMA uses checkpoint-visible session prefixes for both vector and graph retrieval.
Sample/QA checkpoints and run signatures support resuming evaluation. Direct
H2HMEM evaluation requires `--query-embedding-dir`; the launcher supplies the
default prepared cache. Judge results are reused only when their source-answer
hash matches. Call counts include actual transport retries and context-recovery
requests, and image counts reflect images sent across those requests.

To score previously saved WMA results, including optional official evaluator modes:

```bash
python -m benchmarks.wma_harness.eval_wma --summarize-only \
  --results <result-directory>/results.json --output <result-directory>/metrics.json
python -m benchmarks.wma_harness.eval_wma --summarize-only --help
```

## Reproducibility and validation

Keep the recorded dataset commits/checksums, model commits, service version,
configuration, and generated run/checkpoint manifests with each experiment.
The original model-serving versions and historical model commits are not
available in this release, so exact historical numerical equivalence cannot be
asserted from model names alone.

The release fixes were checked offline with Python 3.10.12, PyTorch 2.5.1+cu121
(CPU execution), Transformers 4.57.1, OpenAI SDK 1.109.1, NumPy 2.2.6, Pillow
12.3.0, and huggingface-hub 0.36.2. Regression checks use temporary fixtures and
simulated model APIs: interrupted checkpoint recovery, cache/input rejection,
remote image serialization, all three benchmark evaluation flows, CPU PPO
updates and checkpoint evaluation, and source/wheel/editable installations from
an external working directory. This does not substitute for a full benchmark
run with real model services or a GPU deployment test.

## License

The method code is distributed under the [MIT License](LICENSE). Obtain datasets
and model weights from their upstream distributions.
