# 计划 1：使用 Qwen3-VL Embedding 重构 Omni-SimpleMem 的 Mem-Gallery Chunk

## 目标

把 Mem-Gallery 的记忆写入流程改成“一个 dialogue round 一个 chunk”，并用 `Qwen/Qwen3-VL-Embedding-2B` 把文本、图文、图片相关 round 都 embedding 到同一个多模态向量空间里。

新的设计目标：

- chunk 单位：一个 dialogue round
- embedding model：`Qwen/Qwen3-VL-Embedding-2B`
- embedding 维度：2048
- vector index：一个 FAISS index
- metadata 保留：`session_id`、`round_id`、`image_id`、`timestamp`、`category`

## 核心原则

1. 图片必须作为真实 image input 传给 embedding model。
   不要把图片写成文本里的 `"image"` 字符串。

2. `image_caption` 必须保留在文本 chunk 里。
   即使模型能看图，caption 对 Mem-Gallery 的检索、image_id 对齐、视觉问答 grounding 仍然非常重要。

3. 同一个 chunk 可以同时包含文本和图片。
   有图片的 dialogue round 应该用 `chunk_text + real image input` 一起做 embedding。

4. 索引结构可以简化成一个 VectorStore。
   原来 Omni-SimpleMem 是 text vector store + visual vector store。改成 Qwen3-VL 后，所有 chunk 都可以进入同一个 FAISS index。

5. chunk size 上限建议 500-800 tokens。
   chunk 单位是一个 dialogue round，默认不需要 overlap。如果跨 round 依赖比较强，可以额外加入一个很短的 `previous_round_summary`。

## Chunk 数据结构

每个 chunk 在 embedding 前建议整理成如下结构：

```python
{
    "chunk_id": "D2:R17",
    "text": "...",
    "images": ["/path/to/image.jpg"],
    "embedding": [0.0, ...],
    "metadata": {
        "session_id": "D2",
        "round_id": 17,
        "dialogue_id": "D2:17",
        "image_id": "D2:IMG_003",
        "timestamp": "2024-01-01T10:00:00",
        "category": "VS",
        "has_image": true
    }
}
```

如果是无图片 round：

```python
"images": []
"metadata.has_image": false
```

## Chunk 文本模板

为了让 embedding model 看到稳定的字段，建议固定 chunk text 模板。

### 无图片 Dialogue Round

```text
profile_summary: {compact_profile_summary}
session: {session_id}
date: {date_or_timestamp}
round: {round_id}
user: {user_text}
assistant: {assistant_text}
previous_round_summary: {optional_previous_round_summary}
```

### 有图片 Dialogue Round

```text
profile_summary: {compact_profile_summary}
session: {session_id}
date: {date_or_timestamp}
round: {round_id}
user: {user_text}
assistant: {assistant_text}
image_id: {image_id}
image_caption: {image_caption}
previous_round_summary: {optional_previous_round_summary}
```

真实图片单独传给 embedding model：

```python
embedding = embed_multimodal(
    text=chunk_text,
    images=[image_path],
)
```

不要这样做：

```python
chunk_text = "... image ... image_caption ..."
```

如果 `image` 只是字符串，模型并没有看到图片像素，只是看到一个弱文本标记。

## Token 预算

每个 chunk 目标控制在 500-800 tokens。

如果一个 round 太长，建议按下面顺序压缩：

1. 保留 `session`、`date`、`round`、`image_id`
2. 完整保留 `image_caption`
3. 保留 `user` 的关键内容
4. 保留 `assistant` 的关键内容
5. 压缩 `profile_summary`
6. 压缩或删除 `previous_round_summary`

`profile_summary` 不要太长。每个 chunk 都重复很长的 profile 会稀释当前 round 的局部信息。

## Embedding 方案

新增一个 Qwen3-VL embedding 封装。

目标接口：

```python
class Qwen3VLEmbeddingService:
    dim = 2048

    def embed_chunk(self, text: str, images: list[str] | None = None) -> list[float]:
        ...

    def embed_query(self, query: str, images: list[str] | None = None) -> list[float]:
        ...
```

规则：

- 无图片 round：`embed_chunk(text, [])`
- 有图片 round：`embed_chunk(text, [image_path])`
- 普通 QA query：`embed_query(question, [])`
- 如果 benchmark 提供 query image：`embed_query(question, [query_image])`
- 写入 FAISS 前对 embedding 做 normalize
- 写入前检查 embedding 长度必须是 2048

## VectorStore 方案

所有 chunk 使用一个 FAISS index。

推荐：

```python
faiss.IndexFlatIP(2048)
```

因为向量已经 normalize，inner product 就等价于 cosine similarity。

## FAISS 配置确认项

第一版实现建议使用下面这些默认设置：

1. Index 类型：

```python
faiss.IndexFlatIP(2048)
```

这是 exact search，Mem-Gallery 规模下足够稳定。因为向量会 normalize，所以 inner product 等价于 cosine similarity。

2. 所有向量必须 normalize。

写入的 chunk embedding 和查询 embedding 都要在进入 FAISS 前做 L2 normalize。

```python
embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
```

3. 强制检查维度。

所有 embedding 必须是 2048 维：

- text-only chunk：2048
- image-text chunk：2048
- text query：2048
- 如果使用 image-text query：2048

插入 FAISS 前如果维度不对，直接报错，不要静默写入。

4. metadata 必须存在 FAISS 外部。

FAISS 只存向量，不存文本、图片路径和 metadata。

建议持久化文件：

```text
vectors.index
id_mapping.json
metadata.jsonl
chunks.jsonl
```

5. 更新策略：重建优先，不做复杂 delete/update。

`IndexFlatIP` 不适合频繁删除。Mem-Gallery 是 benchmark 风格，一般一次性 ingest，append-only 就够。如果 chunk 需要更新，建议从 `chunks.jsonl` 和 `metadata.jsonl` 重建 index。

6. top-k 默认值。

建议第一版：

```text
普通问题 top_k: 20
list/count/conflict 类问题 top_k: 40
visual search top_k: 30
```

如果未来数据规模到几十万或百万 chunk，再考虑 `IndexIVFFlat` 或 `IndexHNSWFlat`。Mem-Gallery 规模先用 `IndexFlatIP`。

需要持久化：

```text
vectors.index
id_mapping.json
metadata.jsonl
chunks.jsonl
```

含义：

- `vectors.index`：FAISS 向量索引
- `id_mapping.json`：FAISS row 到 `chunk_id` 的映射
- `metadata.jsonl`：存 `chunk_id`、`session_id`、`round_id`、`image_id`、`timestamp`、`category`、`has_image`
- `chunks.jsonl`：存 chunk text 和 image path 指针

## Mem-Gallery 写入流程

改造或扩展 `OmniMemAdapter.store(observation)`。

当前逻辑：

- 提取 observation text
- 提取 tags
- 调用 `orchestrator.add_text(...)`

新逻辑：

1. 从 observation 解析出一个 dialogue round。
2. 构造 `chunk_text`。
3. 如果有图片，根据 `image_id` 找到真实 image path。
4. 保留 `image_id` 和 `image_caption`。
5. 调用 Qwen3-VL embedding，输入是 text + real image。
6. 把 embedding 写入同一个 FAISS index。
7. 保存 chunk metadata。

伪代码：

```python
def store(observation):
    chunk = build_dialogue_round_chunk(observation)
    embedding = embedder.embed_chunk(
        text=chunk.text,
        images=chunk.images,
    )
    vector_store.add(
        chunk_id=chunk.chunk_id,
        embedding=embedding,
        metadata=chunk.metadata,
        text=chunk.text,
        images=chunk.images,
    )
```

## Mem-Gallery 召回流程

改造或扩展 `OmniMemAdapter.recall(query)`。

新默认流程：

1. 解析问题类别标记：`[AR]`、`[CD]`、`[VS]`、`[VR]`、`[TR]`、`[FR]`、`[KR]`、`[MR]`、`[TTL]`
2. 用 Qwen3-VL 对 query 做 embedding
3. 搜一个 FAISS index
4. 根据 metadata 做轻量 rerank
5. 把 top chunks 格式化成 answer LLM 的 context

类别策略：

- `VS`：优先召回 `has_image=true` 的 chunk；最终必须保留并输出 `image_id`
- `VR`：优先召回图片 chunk，但也保留对话上下文
- `TR`：候选召回后按 timestamp 或 `(session_id, round_id)` 排序
- `CD` / `KR`：扩大 top-k，并按 recency 处理冲突，最新信息优先
- `FR`：list/count 问题扩大 top-k
- `TTL`：强保留图片 chunk 和 caption，因为 test-time visual knowledge 通常依赖它们
- `AR`：如果相似度低且没有支撑 chunk，返回“不足以回答”的信号

## Metadata Rerank

FAISS 召回后做轻量 rerank。

建议分数：

```text
final_score =
    faiss_score
    + image_bonus
    + category_bonus
    + exact_image_id_bonus
    + session_match_bonus
    + recency_bonus
```

例子：

- `[VS]` / `[VR]`：`has_image=true` 加分
- query 中出现 `D2:IMG_003`：对应 `image_id` 大幅加分
- `[TR]`：不要只按相似度排序；候选选出来后要保留时间顺序
- `[KR]`：加 recency bonus

## Context 格式

召回结果格式化时必须带 metadata header。

```text
[1] SESSION:D2 | ROUND:17 | DATE:2024-01-01 | IMG:D2:IMG_003
profile_summary: ...
user: ...
assistant: ...
image_caption: ...
```

视觉搜索任务必须保留精确 image id：

```text
Candidate image_id: D2:IMG_003
Caption: ...
Conversation context: ...
```

## 可能新增或修改的文件

建议新增：

- `OmniSimpleMem/benchmarks/memgallery/qwen3vl_embedding.py`
- `OmniSimpleMem/benchmarks/memgallery/chunk_builder.py`
- `OmniSimpleMem/benchmarks/memgallery/faiss_chunk_store.py`

建议修改：

- `OmniSimpleMem/benchmarks/memgallery/adapter.py`

如果要合入统一 `simplemem` 包，也可以同步改：

- `simplemem/multimodal/utils/embedding.py`
- `simplemem/multimodal/storage/vector_store.py`
- `simplemem/multimodal/retrieval/pyramid_retriever.py`

## 验证计划

1. 测试 chunk 构造。
   - 无图片 round 的 `images=[]`
   - 有图片 round 保留 image path、image id、caption、session、round
   - chunk text 尽量控制在 800 tokens 以内

2. 测试 embedding wrapper。
   - text-only 返回 2048 维
   - image-text 返回 2048 维
   - embedding 已 normalize

3. 测试 FAISS store。
   - add chunks
   - search query
   - FAISS row 能映射回 chunk_id 和 metadata

4. 小规模 Mem-Gallery dry run。
   - ingest 一个 dialogue/session
   - 分别跑一个 `[FR]`、`[VS]`、`[VR]`、`[TTL]` query
   - 检查 context 是否包含正确 `image_id` 和 caption

5. 全量 benchmark。
   - 和当前 `OmniMemAdapter` 对比
   - 看 overall F1 和 per-category F1
   - 重点检查 VS、VR、TTL、KR、CD

## 主要风险

1. 一个 dialogue round 信息过多。
   如果召回变噪，可以只对特别长的 round 拆成 subchunks，但保留同一个 `round_id`。

2. `profile_summary` 太长。
   每个 chunk 重复长 profile 会降低当前 round 的可检索性，应保持紧凑。

3. image path 解析失败。
   Mem-Gallery observation 可能只有 image id，没有本地路径。需要增加 `image_id -> local image path` resolver。

4. 维度不一致。
   写入前强制检查 2048 维。如果模型配置返回其他维度，要 fail fast。

5. VS 输出格式。
   Visual Search 通常要求输出准确 image id。必须把 `image_id` 保存在 metadata，并在 context 中显式暴露。

## 成功标准

- 所有 dialogue round 都变成 text-only 或 image-text chunk
- 有图片 round 会把真实 image 文件传给 Qwen3-VL embedding
- 所有 chunk 都进入同一个 2048 维向量空间
- 一个 FAISS index 负责全部检索
- Mem-Gallery recall 返回带精确 `image_id` 的 metadata-rich context
- VS、VR、TTL 等视觉相关类别不明显退化
