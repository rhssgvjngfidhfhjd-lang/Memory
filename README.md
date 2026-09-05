---
license: mit
task_categories:
- question-answering
language:
- en
tags:
- llm
- agent
- agent-memory
---

# 📊 H2HMem: A Multimodal Memory Benchmark for Agents in Human-Human Interactions

## 📄 Paper

This dataset is introduced in the following research work:

> **H2HMem: A Multimodal Memory Benchmark for Agents in Human-Human Interactions**

-📄 arXiv: https://arxiv.org/abs/2606.09461v1   
-💻 Code: https://github.com/varib1/H2HMEM  
-📊 Dataset: https://huggingface.co/datasets/varib/H2HMEM  
-🌐 Project Page: https://h2hmemprojectpage.vercel.app/  
-🏆 Leaderboard: https://h2hmemleaderboard1.vercel.app/  
-📤 Leaderboard Submission: https://huggingface.co/spaces/varib/H2HMEM-Submit  

---

**H2HMem** is a benchmark designed to evaluate multimodal memory and reasoning capabilities in LLM-based agents across dyadic and multi-party human-human conversations.

---

## 📊 Dataset Overview

| Aspect | Dyadic | Multi-party | Total |
|:--|--:|--:|--:|
| Dialogues | 20 | 5 | **25** |
| Sessions | 284 | 25 | **309** |
| Dialogue Rounds | 5,316 | 1,762 | **7,078** |
| Images | 951 | 349 | **1,300** |
| QA Pairs | 2,046 | 190 | **2,236** |

---

## 🏷️ Task Taxonomy

The benchmark includes three major categories of memory evaluation tasks:

### 🧠 Memory Recall
| Sub-task | Abbreviation |
|----------|-------------|
| Unimodal Precise Recall | UPR |
| Cross-modal Related Retrieval | CRR |
| Knowledge Resolution | KR |

### 🧩 Memory Reasoning
| Sub-task | Abbreviation |
|----------|-------------|
| Temporal Reasoning | TR |
| Multimodal Causal Reasoning | MCR |
| Reference & Evolution Tracking | RET |

### ⚙️ Memory Application
| Sub-task | Abbreviation |
|----------|-------------|
| Test-Time Learning | TTL |
| Conflict Detection | CD |
| Answer Refusal | AR |