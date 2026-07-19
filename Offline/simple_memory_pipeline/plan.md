# Simple Memory Pipeline x Mem-Gallery 实施计划

## 1. 实验目标

把 `simple_memory_pipeline` 接入 Mem-Gallery，形成一个最简 AgentMem baseline：按对话时间顺序读取原始 round，由 LLM 对记忆执行 `INSERT / UPDATE / DELETE / NOOP`，再从最终记忆库检索 top-5，调用与现有实验相同的回答模型并计算 F1、EM、HIT@5 和 LLM Judge Accuracy。

本实验复用现有 a/b/c chunk 作为输入事件，但不复用原 FAISS index 作为最终检索库。AgentMem 操作会压缩、合并或删除记忆，因此每种方法都要生成自己的 memory bank 和 index。

## 2. 固定实验设置

为保证与已有结果可比，先固定以下设置：

- Mem-Gallery：20 datasets，共 1711 QA。
- 输入 chunk：复用当前 a/b/c 的 JSONL，保持每个 dataset 内按 session、round 的时间顺序处理。
- Memory embedding：`Qwen/Qwen3-VL-Embedding-2B`，2048 维。
- Recall：top-k = 5。
- Answer model：沿用已有 Qwen3-VL vLLM OpenAI-compatible 配置。
- Answer 参数：`max_tokens=8000`、`temperature=0`、`timeout=180`、`retries=2`、`think=False`。
- Prompt：继续使用 Mem-Gallery 原生 prompt。
- Judge：`openai/gpt-oss-120b`，并单独统计 `judge_error` 和剔除 error 后的 accuracy。
- 数据隔离：每个 dataset 建立独立 memory bank，禁止不同人物或 dataset 之间互相检索。

## 3. a/b/c 输入定义

| 方法 | 输入 chunk | Memory executor 看到的内容 | Memory embedding | 回答阶段 memory image |
|---|---|---|---|---|
| a. text only | `artifacts/chunks_text_only.jsonl` | 对话文本，不含 caption | memory 文本 | 不传 |
| b. text + caption | `artifacts/chunks_text_caption.jsonl` | 对话文本 + caption | memory 文本 | 不传 |
| c. text + caption + image | `artifacts/chunks.jsonl` | 对话文本 + caption，并保留图片引用 | 第一版采用文本/图片分离向量 | VS/VR 可传召回记忆对应原图 |

说明：c 不再把文本和图片压进同一个 fused vector。已有消融已证明这种融合会伤害非视觉题检索。计划采用 text vector 作为通用检索基础，image vector 作为视觉查询的补充，并在 dialogue 级别合并排名。

## 4. 需要新增或修改的模块

### 4.1 Mem-Gallery 输入加载器

新增 `memgallery/input_loader.py`：

- 读取 a/b/c JSONL。
- 按 `metadata.dataset` 分组。
- 按 session/date/round 稳定排序。
- 输出结构化 `MemoryEvent`，包含：
  - `text`
  - `dataset`
  - `dialogue_id`
  - `session_id`
  - `round_id`
  - `image_ids`
  - `image_paths`
  - `source_chunk_id`
- 启动时校验 a/b/c chunk 数量、caption/image 规则和文件存在性，防止实验模式串库。

### 4.2 带来源追踪的动态 MemoryBank

扩展 `memory_bank.py`：

- 每条 memory 保存稳定 `memory_id`、content、embedding、metadata。
- metadata 必须保存 `source_dialogue_ids`、`source_chunk_ids`、image ids/paths。
- INSERT：继承当前 event 的来源。
- UPDATE：合并旧 memory 与当前 event 的来源集合。
- DELETE：删除 memory，同时在 trace 中保留删除记录。
- 支持 JSONL/NPY/FAISS 持久化和重新加载。
- 检索返回 memory 内容、score、memory_id 及完整 provenance。

这是 HIT@5 正确计算的关键：如果 `D1:3` 和 `D1:4` 被合并为一条 memory，该 memory 的 retrieved id 必须同时包含两个 source dialogue ids。

### 4.3 Executor 接口改造

扩展 `pipeline.py` 和 `executor.py`：

- `run()` 接受结构化 events，而不只接受字符串列表。
- 每处理一个 event，将 event metadata 传给 action 应用阶段。
- 保留现有四种 memory 操作和输出格式。2026-07-14 修订：INSERT 收录标准从"用户持久事实"扩展为也包括每轮对话的关键讨论内容（话题、问答、观点、结论），NOOP 限定为寒暄/填充/已覆盖内容；原标准在知识型对话上产生 80% NOOP，会系统性丢弃 QA 考点轮次（冒烟测试中 10 轮仅存 2 轮、考点覆盖 1/4；修订后 8/10 轮、考点覆盖 3/4）。
- 记录 parse failure、API failure、重试次数和每类 action 数量。
- 支持 checkpoint/resume，避免约 3962 次 memory-management LLM 调用中断后从头开始。
- 默认串行处理同一个 dataset，确保 UPDATE/DELETE 的时序确定性；不同 dataset 可并行。

### 4.4 Qwen embedding 适配器

新增 `memgallery/qwen_embedder.py`，封装现有 Offline 项目的 Qwen3-VL embedding service：

- 固定 2B、2048 维和归一化规则。
- 区分 query/context embedding。
- 支持 batch embedding。
- a/b 只生成 text vectors。
- c 生成 text vectors 和 image vectors，分别保存。
- 查询优先复用现有 1711 条 query embedding cache；若 c 的视觉查询需要独立 image query vector，则生成独立 cache，不能误用 fused cache。

### 4.5 AgentMem Mem-Gallery Adapter

新增 `memgallery/adapter.py`，实现与现有 runner 一致的接口：

- `reset(dataset_name)`：加载该 dataset 的最终 memory bank/index。
- `store()`：构建阶段调用；QA 阶段不再修改记忆。
- `recall(query)`：返回 top-5 memory items。
- `last_retrieved_ids`：由 top-5 memories 的 `source_dialogue_ids` 展开并去重，用于 HIT@5。
- 保存详细 retrieval trace：query、memory_id、score、content、source ids、image ids/paths。

HIT@5 的计算单位仍是 top-5 memory，而不是展开后取前五个 dialogue id。只要某个 top-5 memory 的 provenance 与 QA clue 相交，就记为命中。

### 4.6 构建与评测 runner

新增两个独立入口，避免影响已有 Offline RAG 代码：

- `build_memgallery_memories.py`
  - 参数：`--mode a|b|c`、chunk 路径、输出目录、executor model/API、embedding 配置、resume。
  - 输出：memory bank、vectors/index、action trace、构建统计和失败记录。
- `run_memgallery_baseline.py`
  - 加载已构建的 memory index。
  - 复用现有原生 prompt、回答 client、QA 字段和 metrics。
  - 输出 `results.json`、`metrics.json`、`retrieval_trace.jsonl`。

建议 artifacts 目录：

```text
artifacts/simple_memory_pipeline/
  a_text_only/
    datasets/<dataset>/memories.jsonl
    datasets/<dataset>/vectors.npy
    datasets/<dataset>/index.faiss
    build_trace.jsonl
    build_stats.json
  b_text_caption/
  c_text_caption_image/
  results/
```

## 5. 执行步骤

### Stage 0：数据审计

1. 校验三个 chunk 文件均为 3962 条，并覆盖同一组 chunk ids。
2. 校验 a 无 caption/图片，b 有 caption 但不带图片输入，c 有 caption 和有效图片路径。
3. 校验每个 dataset 的 round 排序及 dialogue_id 唯一性。
4. 输出审计 JSON，任何模式不一致时停止构建。

### Stage 1：最小代码适配和单元测试

1. 实现 `MemoryEvent`、provenance 合并和持久化。
2. 实现 Qwen embedder 与 adapter。
3. 测试 INSERT/UPDATE/DELETE 后来源 id 是否正确。
4. 测试一个合并 memory 同时命中多个 clue 的 HIT@5 逻辑。
5. 测试 checkpoint 恢复不会重复处理 event。

### Stage 2：单 dataset 冒烟测试

使用 `AI_Robotics_Automation_Future_Tech`：

1. 分别构建 a/b/c memory bank。
2. 检查前 10 个 memory-management actions 是否可解析、来源是否保留。
3. 每种模式跑前 10 个 QA。
4. 人工核查 top-5 内容、retrieved ids、图片传递和 clue 命中。
5. 对比现有 RAG 前 10 题，确认差异来自 memory extraction，而不是 prompt/answer 参数变化。

### Stage 3：20 datasets 全量构建

1. 每个 dataset 独立构建并即时 checkpoint。
2. 记录总耗时、每 dataset 耗时、LLM 调用数、token 使用、action 分布、最终 memory 数和压缩率。
3. 构建完成后统一生成 Qwen embedding 和 FAISS index。
4. 对所有 artifacts 做可加载性及向量维度检查。

执行顺序建议先 a，再 b，最后 c。a 最简单，可先暴露 pipeline 与 provenance 问题；c 在 a/b 稳定后再加入视觉向量路由。

### Stage 4：1711 QA 正式评测

1. a/b/c 使用相同 answer model、prompt 和生成参数。
2. 跑 F1、EM、HIT@5，并按 AR/CD/FR/KR/MR/TR/TTL/VR/VS 分类汇总。
3. 保存每题 top-5 的 memory 文本、score 和 provenance，便于解释回退案例。
4. QA 完成后启动 LLM Judge；Judge 可独立断点续跑。
5. 同时报告 Judge Acc、judge_error 数和剔除 error 后 Acc。

### Stage 5：结果验收与对比

最终表至少包含：

| Method | Final memories | Compression ratio | F1 | EM | HIT@5 | Judge Acc | Judge Acc excl. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimpleMem-a | | | | | | | |
| SimpleMem-b | | | | | | | |
| SimpleMem-c | | | | | | | |

并与已有 Offline RAG a/b/c 结果放在同一张表中。额外分析：

- memory 操作分布和 parse/error rate。
- 压缩率与 HIT@5/F1 的关系。
- 按题型的增益或回退。
- HIT@5 回退是否由 memory 合并、删除、摘要信息丢失或视觉路由造成。

## 6. 验收标准

- 三种模式没有跨 dataset 记忆污染。
- a 的 executor、memory 和 answer prompt 中均不存在 caption 或图片内容。
- b 不使用原图或 image embedding。
- c 的 text/image vectors 分开存储，图片仅按既定视觉策略传给回答模型。
- 每条最终 memory 都有非空且可追溯的 `source_dialogue_ids`。
- `retrieval_trace.jsonl` 可以还原每个 QA 的 top-5 和 HIT@5 判定。
- 1711 QA 无 answer error；Judge error 独立统计且可续跑。
- 所有命令、配置、耗时和随机性设置写入 run manifest，结果可复现。

## 7. 预期产物

完成后会得到两组可直接比较的系统：

1. 现有 Offline a/b/c：原始 round 直接建立索引，属于 memory-augmented RAG。
2. SimpleMem a/b/c：先由 LLM 对 round 做动态记忆管理，再对最终 memories 建索引，属于最简 AgentMem baseline。

因此最终差异能够回答：AgentMem 的记忆压缩、更新与删除，相比直接索引全部对话，是否提升检索和回答质量，以及这种变化在文本、caption 和图片三种信息条件下是否一致。
