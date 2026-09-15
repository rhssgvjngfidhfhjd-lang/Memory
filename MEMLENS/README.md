---
license: cc-by-4.0
language:
  - en
task_categories:
  - question-answering
  - visual-question-answering
pretty_name: MemLens
tags:
  - multimodal
  - long-context
  - conversational-memory
  - vision-language-models
  - benchmark
  - VLM-evaluation
configs:
  - config_name: 32k
    data_files: dataset_32k.parquet
  - config_name: 64k
    data_files: dataset_64k.parquet
  - config_name: 128k
    data_files: dataset_128k.parquet
  - config_name: 256k
    data_files: dataset_256k.parquet
    default: true
---

# MemLens: Benchmarking Multimodal Long-Context Conversational Memory in Vision-Language Models

<p align="center">
    <a href="https://github.com/xrenaf/MEMLENS" target="_blank">
        <img alt="Code" src="https://img.shields.io/badge/Code-GitHub-181717?logo=github">
    </a>
    <a href="https://huggingface.co/datasets/xiyuRenBill/MEMLENS" target="_blank">
        <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-blue">
    </a>
    <a href="#" target="_blank">
        <img alt="Paper" src="https://img.shields.io/badge/paper-paper?logo=arxiv&logoColor=%23B31B1B&labelColor=white&color=%23B31B1B">
    </a>
</p>

> **This repository hosts the MemLens dataset only.** Evaluation code, model wrappers, and scoring scripts live at **[github.com/xrenaf/MEMLENS](https://github.com/xrenaf/MEMLENS)**.

## Overview

MemLens is a benchmark for evaluating long-horizon conversational memory in vision-language models. It tests whether models can retrieve, recall, update, and reason over visual and textual information embedded across multi-session dialogues at 32K / 64K / 128K / 256K context windows.

**789 questions** across 5 types: Information Extraction, Knowledge Update, Temporal Reasoning, Multi-Session Reasoning, and Answer Refusal (Abstention).

## Repository Structure

```
xiyuRenBill/MEMLENS/
  dataset_32k.json        # 789 items, 32K context  (~98 MB)
  dataset_64k.json        # 789 items, 64K context  (~191 MB)
  dataset_128k.json       # 789 items, 128K context (~369 MB)
  dataset_256k.json       # 789 items, 256K context (~732 MB)
  dataset_32k.parquet     # Parquet equivalent of dataset_32k.json
  dataset_64k.parquet
  dataset_128k.parquet
  dataset_256k.parquet
  agent_subset_195.json   # Indexing file: 195 question_ids used for memory-agent evaluation
  release_images/
    haystack_images/      # images referenced by haystack sessions
    needle_images/        # images referenced by needle (evidence) sessions
  metadata/
    croissant.json        # Croissant 1.0 + RAI metadata
  DATASHEET.md            # RAI datasheet
  CITATION.cff
  LICENSE-DATA            # CC-BY-4.0
```

The same 789 question_ids appear in all four `dataset_*` splits — only the surrounding haystack length differs. The Parquet files are byte-equivalent records to the JSON files, provided for the HuggingFace Dataset Viewer / Data Studio and for fast columnar loading.

## Splits

| Config | Context | Records | JSON | Parquet |
|---|---|---|---|---|
| `32k`  | 32 768 tokens  | 789 | `dataset_32k.json`  (~98 MB)  | `dataset_32k.parquet`  (~52 MB)  |
| `64k`  | 65 536 tokens  | 789 | `dataset_64k.json`  (~191 MB) | `dataset_64k.parquet`  (~101 MB) |
| `128k` | 131 072 tokens | 789 | `dataset_128k.json` (~369 MB) | `dataset_128k.parquet` (~195 MB) |
| `256k` | 262 144 tokens | 789 | `dataset_256k.json` (~732 MB) | `dataset_256k.parquet` (~387 MB) |

## Per-Question Schema

Each record in `dataset_*k.json` / `dataset_*k.parquet` has these top-level fields:

| Field | Type | Description |
|---|---|---|
| `question_id` | string | Stable across splits; e.g. `q_4106e113` |
| `question_type` | string | One of `information_extraction`, `knowledge_update`, `temporal_reasoning`, `multi_session_reasoning`, `answer_refusal` |
| `question` | string | Natural-language question + answer-format hint |
| `answer` | string | Gold answer |
| `question_date` | string | Timestamp of the question turn (e.g. `2024/05/31 (Fri) 07:58`) |
| `haystack_dates` | list[string] | Per-session date strings |
| `haystack_session_ids` | list[string] | Per-session id (e.g. `sess_6a60a229`) |
| `haystack_sessions` | list[list[turn]] | Conversation turns per session — see below |
| `answer_session_ids` | list[string] | Subset of `haystack_session_ids` containing evidence for the answer |

A single **turn** has:

| Field | Type | Description |
|---|---|---|
| `role` | string | `user` or `assistant` |
| `content` | string | Text content |
| `images` | list[image_ref] | Possibly empty |
| `has_answer` | bool | Whether this turn contains gold-answer evidence |

A single **image_ref** has:

| Field | Type | Description |
|---|---|---|
| `file` | string | Repo-relative path under `release_images/` (e.g. `needle_images/a3b2c891f04e.jpg`) |
| `image_url` | string | Original source URL where the image was retrieved |
| `blip_caption` | string | Auto-generated caption for indexing |

## Loading

### Via the `datasets` library (uses Parquet, viewer-aligned)

```python
from datasets import load_dataset

ds = load_dataset("xiyuRenBill/MEMLENS", "256k")   # also: "32k", "64k", "128k"
print(ds)
print(ds["train"][0]["question_id"])
```

### Direct `json.load` (legacy)

```python
import json
data = json.load(open("dataset_256k.json"))
print(len(data), data[0]["question_id"])
```

### Resolving image paths

```python
from pathlib import Path

REPO = Path("/path/to/local/MEMLENS-dataset")     # where you downloaded the repo
img = data[0]["haystack_sessions"][0][0]["images"][0]
local_path = REPO / "release_images" / img["file"]   # e.g. release_images/needle_images/a3b2c891f04e.jpg
```

Image filenames are 12-character random hex (e.g. `a3b2c891f04e.jpg`), globally unique across both `haystack_images/` and `needle_images/`.

## Agent Subset (n = 195)

Memory-augmented agent pipelines (M3-Agent, M2A, M3C, Memory-T1, Mem0, MemOS, MemAgent-7B) are evaluated on a fixed stratified 195-question subset of the full 789-question benchmark, because per-question agent inference is roughly 60× slower than direct VLM inference. The exact `question_id` list lives in `agent_subset_195.json` (an *indexing* file with no QA payload), together with the per-type breakdown (61 IE / 35 MSR / 48 TR / 29 KU / 22 AR), stratification details (seed = 42, derived from a 200-sample then intersected with available agent runs to drop 5 incomplete questions), and a Python snippet for filtering each `dataset_*.json` to the subset. See paper Appendix G.2 for full derivation.

## Supported Models (via the GitHub eval code)

The evaluation code at [github.com/xrenaf/MEMLENS](https://github.com/xrenaf/MEMLENS) supports:

**Closed-source API models**: GPT-4o, GPT-4.1, o3, o4-mini, Seed-1.8, Claude Sonnet 4 / Opus 4, Gemini 2.5/3 Pro/Flash, Kimi K2.5.

**Open-source local models**: Qwen3-VL (2B / 4B / 8B / MoE 30B / MoE 235B), Qwen2.5-VL (7B / 72B), Qwen2-VL, Gemma 3 (4B / 12B / 27B), Gemma 4, GLM-4.5V / GLM-4.6V, Phi-4, Cosmos-Reason2-8B, Nemotron-Nano-12B VL.

Both HuggingFace Transformers and vLLM backends are supported.

## Datasheet

A full datasheet (motivation, composition, collection, preprocessing, uses, distribution, maintenance) is in [`DATASHEET.md`](DATASHEET.md), and machine-readable [Croissant 1.0 + RAI metadata](metadata/croissant.json) is provided.

## Licenses

- The MemLens dataset (question metadata, conversation sessions, prompt templates, judge artefacts) is released under **CC-BY-4.0** (see [`LICENSE-DATA`](LICENSE-DATA)).
- Images in `release_images/` are sourced from the web; each image retains its original source-site license. A takedown contact is provided in the project repository; any flagged image will be removed within seven days.
- Evaluation code (at the GitHub repo) is released separately under the MIT License.

## Citation

```bibtex
@inproceedings{ren2026memlens,
    title={{MemLens}: Benchmarking Multimodal Long-Context Conversational Memory in Vision-Language Models},
    author={Ren, Xiyu and Wang, Zhaowei and Du, Yiming and Xie, Zhongwei and Liu, Chi and Yang, Xinlin and Feng, Haoyue and Pan, Wenjun and Zheng, Tianshi and Xu, Baixuan and Li, Zhengnan and Song, Yangqiu and Wong, Ginny and See, Simon},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
    year={2026}
}
```
