# 计划 2：使用 Offline Qwen3-VL Chunk Memory 跑 Mem-Gallery F1

## 目标

模仿下面这个项目的代码组织方式：

```text
/data1/haozhen/Visual_Primitives/Offline/SimpleMem/OmniSimpleMem
```

但所有新代码都放在：

```text
/data1/haozhen/Visual_Primitives/Offline/Offline
```

使用我们已经构建好的 Qwen3-VL embedding 结果和 FAISS index 来跑 Mem-Gallery，并计算 F1 分数。

当前已经完成的产物：

```text
artifacts/chunks.jsonl
artifacts/embeddings/
artifacts/faiss_index/
```

当前索引状态：

```text
chunks: 3962
vectors: 3962
dimension: 2048
dialogue images embedded: 1003
FAISS index: artifacts/faiss_index/vectors.index
```

## 设计方向

新增一个小型 Offline-Omni 包，结构上模仿 OmniSimpleMem 的 benchmark adapter 风格：

```text
offline_omni_memory/
  __init__.py
  core/
    config.py
  storage/
    chunk_store.py
  retrieval/
    retriever.py
    formatter.py
  embeddings/
    qwen3vl.py
  benchmarks/
    memgallery/
      adapter.py
      run_memgallery.py
      evaluate_results.py
```

对应 OmniSimpleMem 的概念：

- `config.py` 对应 `omni_memory/core/config.py`
- `chunk_store.py` 对应 `omni_memory/storage/vector_store.py` + `mau_store.py`
- `retriever.py` 对应 `omni_memory/retrieval/pyramid_retriever.py`
- `adapter.py` 对应 `OmniSimpleMem/benchmarks/memgallery/adapter.py`
- `run_memgallery.py` 是本地版 Mem-Gallery runner

## 为什么不重新 embedding

benchmark 运行时不要重新构建 memory embedding。

memory 侧已经完成：

```text
artifacts/embeddings/
artifacts/faiss_index/
```

benchmark 运行时只需要：

1. 加载 FAISS 和 metadata
2. 对每个 QA query 动态使用 Qwen3-VL 做 embedding
3. 检索 top chunks
4. 格式化 context
5. 调用 answer LLM/VLM
6. 和 ground truth 计算 F1

## 运行时组件

### 1. OfflineOmniConfig

文件：

```text
offline_omni_memory/core/config.py
```

字段：

```python
index_dir = "artifacts/faiss_index"
data_dir = "/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data"
embedding_model = "Qwen/Qwen3-VL-Embedding-2B"
embedding_dim = 2048
device = "cuda:0"
dtype = "bfloat16"
top_k_default = 5
top_k_visual = 5
top_k_broad = 8
answer_model = ...
answer_api_key = ...
answer_base_url = ...
```

### 2. ChunkStore

文件：

```text
offline_omni_memory/storage/chunk_store.py
```

职责：

- 加载 `vectors.index`
- 加载 `id_mapping.json`
- 加载 `chunks.jsonl`
- 加载 `metadata.jsonl`
- 提供 `search(query_embedding, top_k, category, query_text)`

实现上可以复用已有代码：

```text
offline_memgallery_qwen/faiss_store.py
```

### 3. Qwen3VLEmbedder

文件：

```text
offline_omni_memory/embeddings/qwen3vl.py
```

职责：

- 加载 `Qwen/Qwen3-VL-Embedding-2B`
- 对纯文本 query 做 embedding
- 如果 QA 有 `question_image`，对图文 query 做 embedding
- normalize 输出
- 检查输出必须是 2048 维

可以复用已有代码：

```text
offline_memgallery_qwen/qwen3vl_embedding.py
```

### 4. MemGalleryRetriever

文件：

```text
offline_omni_memory/retrieval/retriever.py
```

职责：

- 解析任务类别：`AR`、`CD`、`VS`、`VR`、`TR`、`FR`、`KR`、`MR`、`TTL`
- 选择 top-k
- 调用 Qwen3-VL query embedding
- 搜索一个 FAISS index
- 做类别感知 rerank
- 保存 `last_retrieved_ids`，用于 retrieval metrics

类别行为：

| 类别 | 检索策略 |
|---|---|
| `VS` | 优先 `has_image=true` 的 chunk，保留精确 `image_id` |
| `VR` | 优先图片 chunk，同时保留对话上下文 |
| `TTL` | 优先图片 chunk 和 caption |
| `TR` | 先扩大候选，再按 `(session_id, round_id)` 排序 |
| `CD` | 略微扩大 top-k，保留可能冲突或最近的候选 |
| `KR` | 略微扩大 top-k，偏向最新信息 |
| `FR` | list/count 问题略微扩大 top-k |
| `MR` | 默认 top-k，必要时增加 session 多样性 |
| `AR` | 如果没有强相关证据，允许返回低上下文或空上下文 |

### 5. Context Formatter

文件：

```text
offline_omni_memory/retrieval/formatter.py
```

需要两种输出模式：

1. 普通文本 context：

```text
[1] SESSION:D2 | ROUND:D2:6 | DATE:2024-07-07 | IMG:D2:IMG_003
user: ...
assistant: ...
image_caption: ...
```

2. 兼容 Mem-Gallery `VLMAgent.fast_run_with_mm_memory` 的多模态 memory item：

```python
[
    {
        "text": "...",
        "image": {
            "path": "...",
            "img_id": "D2:IMG_003",
            "caption": "..."
        }
    }
]
```

优先使用第二种，因为它可以把检索到的真实图片传给回答 VLM。

### 6. MemGalleryOfflineAdapter

文件：

```text
offline_omni_memory/benchmarks/memgallery/adapter.py
```

模仿：

```text
SimpleMem/OmniSimpleMem/benchmarks/memgallery/adapter.py
```

接口：

```python
class MemGalleryOfflineAdapter:
    def reset(self): ...
    def store(self, observation): ...
    def recall(self, query): ...
    def display(self): ...
    def manage(self, operation, **kwargs): ...
    def optimize(self, **kwargs): ...
```

重要区别：

- `store()` 是 no-op，因为 memory 已经提前构建好了
- `recall()` 使用当前 FAISS index，并动态 embedding query

adapter 需要暴露：

```python
self.last_retrieved_ids
```

这样可以和 `qa["clue"]` 计算 retrieval metrics。

## Runner 计划

新增：

```text
offline_omni_memory/benchmarks/memgallery/run_memgallery.py
```

这个 runner 模仿 Mem-Gallery 官方：

```text
Mem-Gallery/benchmark/run/run_bench.py
```

但不修改原始 benchmark 仓库。

### Runner 步骤

1. 从下面目录加载一个或全部 dataset JSON：

```text
/data1/haozhen/Visual_Primitives/Offline/Mem-Gallery/benchmark/data/dialog
```

2. 对每个 QA：

- 读取 `question`
- 如果存在，读取 `question_image`
- 读取 `point` 作为 category
- 构造类别约束 prompt
- 调用 adapter.recall：

```python
memory_items = adapter.recall({
    "text": f"[{category}] {question}",
    "image": question_image_dict_or_none
})
```

3. 调用 answer LLM/VLM。

可选方式：

- 复用 Mem-Gallery 的 `VLMAgent` 逻辑
- 或本地实现一个 OpenAI-compatible answer client

第一版建议：

- 复制 `run_bench.py` 里的关键 prompt 构造逻辑
- 用 `openai.OpenAI` 调 OpenAI-compatible API
- 支持检索 memory image 和 query image

4. 保存每道题结果：

```json
{
  "sample_id": "...",
  "session_id": "...",
  "question": "...",
  "category": "VS",
  "system_answer": "...",
  "original_answer": "...",
  "retrieved_ids": ["D2:6", "..."],
  "clue": ["D2:6", "..."]
}
```

5. 保存 dataset-level 和 overall metrics。

## 评估计划

新增：

```text
offline_omni_memory/benchmarks/memgallery/evaluate_results.py
```

复用下面文件里的 normalization/F1 思路：

```text
Mem-Gallery/benchmark/memengine/evaluate/evaluation.py
```

指标：

- overall F1
- overall exact match
- per-category F1
- per-category exact match
- 如果有 `retrieved_ids` 和 `clue`，计算 retrieval Recall@K / HitRate@K

输出：

```text
artifacts/results/offline_qwen3vl/results.json
artifacts/results/offline_qwen3vl/metrics.json
```

## API / 模型配置

benchmark 需要一个 answer model，和 embedding model 分开。

支持环境变量：

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
export OPENAI_MODEL=...
```

如果使用本地 VLLM：

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_MODEL=/path/or/name/of/vlm
```

如果使用 OpenAI/OpenRouter-compatible API：

```bash
export OPENAI_BASE_URL=https://...
export OPENAI_MODEL=...
```

## 命令计划

### 单数据集 smoke run

```bash
python -m offline_omni_memory.benchmarks.memgallery.run_memgallery \
  --data-name AI_Robotics_Automation_Future_Tech \
  --index-dir artifacts/faiss_index \
  --result-dir artifacts/results/offline_qwen3vl_smoke \
  --max-qa 5 \
  --device cuda:0 \
  --dtype bfloat16
```

### 全量运行

```bash
python -m offline_omni_memory.benchmarks.memgallery.run_memgallery \
  --all-datasets \
  --index-dir artifacts/faiss_index \
  --result-dir artifacts/results/offline_qwen3vl \
  --device cuda:0 \
  --dtype bfloat16
```

### 评估已有结果

```bash
python -m offline_omni_memory.benchmarks.memgallery.evaluate_results \
  --results artifacts/results/offline_qwen3vl/results.json \
  --output artifacts/results/offline_qwen3vl/metrics.json
```

## 实现顺序

1. 创建 `offline_omni_memory/` 目录结构。
2. 封装现有 `FaissChunkStore` 和 `Qwen3VLEmbeddingService`。
3. 实现 `MemGalleryOfflineAdapter`。
4. 实现本地 answer prompt formatter。
5. 实现 `run_memgallery.py`。
6. 实现 `evaluate_results.py`。
7. 跑 5-QA smoke test。
8. 跑一个完整 dataset。
9. 跑全部 datasets。
10. 汇报 overall F1 和 per-category F1。

## 关键检查点

- `VS` 的 recall context 里必须有精确 `image_id`
- 检索到图片 chunk 时必须包含真实 image path
- 如果 QA 有 `question_image`，query 阶段要动态传给 Qwen3-VL embedding
- `last_retrieved_ids` 应该用 `D2:6` 这样的 dialogue id，而不是内部 chunk id
- 结果文件必须保留 `qa["point"]` 作为 category
- F1 评估必须保护 `D2:IMG_003` 这类 ID，不要被 normalization 破坏

## 预期风险

1. Answer model API 没配置。
   解决：runner 在缺少 `OPENAI_MODEL` 或 API 配置时清晰报错。

2. Query embedding 重复加载模型导致慢。
   解决：整个 run 只保留一个 Qwen3-VL embedder 实例。

3. VS 输出格式错误。
   解决：加入 category prompt constraint，并在 context 中显式暴露 image ID。

4. 检索过多影响回答质量。
   解决：默认 top-k 固定为 5；只有 list/count/conflict 类问题可以使用 top-k 8。

5. 全量运行成本较高。
   解决：支持 `--max-datasets`、`--max-qa` 和从结果文件续跑。

## 成功标准

- 可以跑通 5-QA smoke test。
- 可以跑通一个完整 dataset 并保存结果。
- 可以评估保存结果并计算 F1。
- 全量 Mem-Gallery 运行后产出：

```text
overall F1
per-category F1
retrieval metrics
```

- 不修改原始 SimpleMem 或 Mem-Gallery 文件。
