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

> **Local experimental subset.** This directory contains only the experiment-required MEMLENS-32K-Agent answerable subset. Evaluation code, model wrappers, and scoring scripts live separately at **[github.com/xrenaf/MEMLENS](https://github.com/xrenaf/MEMLENS)**.

## Overview

MemLens is a benchmark for evaluating long-horizon conversational memory in vision-language models. This local experiment uses only the **32K** version and only the answerable portion of the canonical Agent subset.

The retained test set has **173 instances / 173 questions** across 4 types: Information Extraction (61), Multi-Session Reasoning (35), Temporal Reasoning (48), and Knowledge Update (29). It references **1,851 unique images**.

The upstream canonical Agent subset contains 195 questions. The 22 Answer Refusal questions were removed for this experiment, so `195 - 22 = 173`. The remaining upstream questions, context versions, serializations, and unreferenced images were also deleted from this local copy.

## Repository Structure

```
MEMLENS/
  dataset_32k.json        # 173 answerable Agent-subset items, 32K context
  agent_subset_173.json   # Exact question_id index for this experiment
  release_images/
    haystack_images/      # retained images referenced by haystack sessions
    needle_images/        # retained images referenced by needle sessions
  metadata/
    croissant.json        # unmodified upstream provenance metadata
  DATASHEET.md            # local scope plus upstream provenance datasheet
  CITATION.cff
  LICENSE-DATA            # CC-BY-4.0
```

Only files required by the local experiment are retained. The Croissant metadata is preserved unchanged for upstream provenance and therefore describes the complete upstream release rather than this pruned directory.

## Splits

| Config | Setting | Context | Records | QA | Images | File |
|---|---|---:|---:|---:|---:|---|
| `32k-agent-answerable` | OOD test only | 32,768 tokens | 173 | 173 | 1,851 | `dataset_32k.json` |

## Per-Question Schema

Each record in `dataset_32k.json` has these top-level fields:

| Field | Type | Description |
|---|---|---|
| `question_id` | string | Stable across splits; e.g. `q_4106e113` |
| `question_type` | string | One of `information_extraction`, `knowledge_update`, `temporal_reasoning`, `multi_session_reasoning`; `answer_refusal` was excluded by the experiment protocol |
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

### Direct `json.load`

```python
import json
data = json.load(open("dataset_32k.json"))
assert len(data) == 173
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

## Experiment Agent Subset (n = 173)

The experiment starts from the upstream canonical 195-question Agent subset and removes its 22 `answer_refusal` records. The resulting OOD test set contains 173 answerable questions: 61 IE / 35 MSR / 48 TR / 29 KU. The exact retained IDs are recorded in `agent_subset_173.json`.

This local subset is an experiment-specific derivative and must not be described as the complete upstream MEMLENS release.

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
