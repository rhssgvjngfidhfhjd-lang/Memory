# Mem-Gallery × SimpleMem / A-Mem / M2A baseline runbook

目标：让 WorldMemArena 里的三个记忆系统（SimpleMem、A-Mem、M2A，另有可选的
Omni-SimpleMem）在 Mem-Gallery（20 数据集 / 1711 QA）上出 **F1 / HIT@5 /
LLM-Judge Acc**，与 AgentMem a/b/c 结果同轨可比。

## 固定评测配置（与 AgentMem 基线一致）

| 项 | 值 |
|---|---|
| 执行/答题模型 | `Qwen/Qwen3-VL-4B-Instruct`（vLLM OpenAI 兼容, 默认端口 8000） |
| 答题参数 | max_tokens 8000, temperature 0, timeout 180s, retries 2, think=False |
| Judge | `openai/gpt-oss-120b`（NVIDIA API, timeout 60s, 前 12 个 key, max_tokens 1024） |
| Embedding | `Qwen/Qwen3-VL-Embedding-2B`, dim 2048（构建索引在 cuda:3） |
| Query 向量缓存 | `Offline/Offline/artifacts/query_embeddings`（1711 条全量已缓存） |
| 检索 | 余弦 top-5（有图 memory 走 image-vector max-fusion, 同 mode c） |
| 输入 | `artifacts/chunks.jsonl`（caption + 图片路径完整版, 与 AgentMem 输入一致） |

设计：三个 baseline **只做记忆构建**（各自的原生 LLM 记忆管理栈），检索/答题/
评测统一走 `run_memgallery_baseline` + `judge_results_llm_parallel.py`。
SimpleMem 的 BM25 混合检索、M2A 的 Milvus 检索均不使用（保证检索公平可比）。

## 产物布局

```
Offline/Offline/artifacts/baseline_memories/
  <baseline>/datasets/<dataset>/memories.jsonl      # Stage 1（WMA venv 产出）
  <baseline>/datasets/<dataset>/vectors.npy         # Stage 2（+image_vectors/mask）
  <baseline>/build_manifest.json, embed_manifest.json
  results/<baseline>_qwen3vl4b/{results.json, metrics.json, llm_judge_metrics.json}
  logs/
```

`<baseline>` ∈ `simplemem`（原版 text 模式）/ `omnisimplemem` / `amem` / `m2a`。

## Stage 1 — 记忆构建（WorldMemArena repo，必须用它的 .venv）

```sh
cd /data1/haozhen/Visual_Primitives/Offline/WorldMemArena
# 长跑一律放 screen（后台任务会随 Claude 会话终止, 见 2026-07-16 教训）
CUDA_VISIBLE_DEVICES=3 .venv/bin/python eval_framework/scripts/build_memgallery_baselines.py \
  --baseline amem --llm-base-url http://127.0.0.1:8000/v1 --resume
```

- `--baseline simplemem|omnisimplemem|amem|m2a`；`--datasets A,B` 跑子集；
  `--max-rounds N` 冒烟；`--resume` 跳过已完成数据集（断点续跑按数据集粒度）。
- `CUDA_VISIBLE_DEVICES=3`：A-Mem 的 sentence-transformer、M2A 的本地 embed
  server、Omni-SimpleMem 的本地 Qwen3-VL-Embedding 都落在 GPU3（GPU0/1/2 被
  vLLM 占用）。LLM 调用走 HTTP 不受影响。
- M2A 默认 `--m2a-mode agent`（上游 ChatAgent/MemoryManager 决定存什么，
  ~2 次 LLM 调用/轮，全量约数小时）；`--m2a-mode direct` 为逐轮强制入库的快速路径。
- 溯源（HIT@5 用）：Omni-SimpleMem 按 MAU 的 dialogue_id tag 精确归属；
  A-Mem / M2A 逐轮 diff 精确归属；SimpleMem text 按窗口/日期归属（其压缩以
  session 为 finalize 边界, 粒度为 session 内窗口, 属系统固有行为）。

## Stage 2+3 — 嵌入、QA、Judge（Offline repo）

```sh
# 全链路（embed → qa → judge）：
sh /data1/haozhen/Visual_Primitives/Offline/Offline/scripts/run_membaselines_memgallery.sh amem all 8000
# 或分步：... amem embed / ... amem qa 8000 / ... amem judge
```

- 指标：`results/<b>_qwen3vl4b/metrics.json`（F1、EM、retrieval_hitrate@5，
  含分类目），`llm_judge_metrics.json`（JudgeAcc）。
- QA 前确认 vLLM 答题服务在目标端口存活：`curl http://127.0.0.1:8000/v1/models`。
- Judge 只用前 12 个 NVIDIA key（后面的已过期, 2026-07-16 验证）。

## 冒烟状态与已知事项（2026-07-18）

- 冒烟已全链路通过（AI_Robotics 6 轮 × 3 baseline → embed → QA×3 → judge）。
  产物在 `artifacts/baseline_memories_smoke/`（与正式目录隔离）。
- **冒烟用的 LLM 是 8000 端口的 qwen3-coder-30b**（临时借用，非 spec 模型）：当时
  4 张 GPU 各只剩 ~12GiB（derrick 的 30B tp4 服务占用），VL-4B 起不来。**正式跑前必须
  自起 Qwen3-VL-4B 服务**，且要加：
  `--enable-auto-tool-choice --tool-call-parser hermes`（M2A agent 模式的 LangChain
  tool-call 必需，否则 400）；上下文 ≥32k（QA 的 max_tokens=8000）。
- M2A 上游已打的补丁（均在 `eval_framework/baselines/M2A/`）：
  1. `agents/memory_manager.py`、`agents/chat_agent.py`：`from tkinter import END`
     → `from langgraph.graph import END`，并删除 `workflow.add_node(END, ...)`
     （langgraph 保留节点）；
  2. `stores/semantic.py`：memory collection 加 2 维占位向量字段 `pad_vec`
     （milvus-lite 3.0 不允许纯标量 collection）；VARCHAR 1000→8192、
     image_path 100→512，插入时截断到 8000；
  3. `memory_adapters/_m2a_embed_server.py`：非 `text-embedding-3*` 模型名改为
     本地 sentence-transformers 推理（原实现代理到 chat server 的
     /v1/embeddings，必 500）。
- 溯源公平性规则：只有溯源精确到单轮的 memory 才携带 image_ids/image_paths
  （SimpleMem 的 session 级 memory 不继承整个 session 的图片）。
- image-ID 泄漏防护已复验：simplemem VS 冒烟中模型答错 ID 是瞎猜（memory 文本无
  ID，redaction 正则 `\bD\d+:IMG_\d+\b` 覆盖本数据格式）。

## 冒烟（单数据集、少量轮次）

```sh
# Stage 1（约 6 轮）：
cd WorldMemArena && CUDA_VISIBLE_DEVICES=3 .venv/bin/python \
  eval_framework/scripts/build_memgallery_baselines.py --baseline amem \
  --datasets AI_Robotics_Automation_Future_Tech --max-rounds 6 \
  --out-root /data1/haozhen/Visual_Primitives/Offline/Offline/artifacts/baseline_memories_smoke
# Stage 2+3 手动指向 smoke 目录, QA 加 --max-qa 3（冒烟结果与正式目录隔离）
```
