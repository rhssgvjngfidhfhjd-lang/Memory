# OmniSimpleMem 跑 Mem-Gallery 的改造记录

本次改造对象是 `/data1/haozhen/Visual_Primitives/Offline/SimpleMem/OmniSimpleMem`，目标是让它用
`omni_memory/orchestrator.py` 自带的机制跑 Mem-Gallery（1711 道题，20 个数据集），并在此基础上
修复三个实际影响效果的问题。embedding 全程用 `Qwen/Qwen3-VL-Embedding-2B`（2048 维），跟
Offline 项目保持一致，方便对比。

## 改动 1：top_k 从固定 5 改回 category-aware 的 dynamic

**Step**：不同类别问题需要的候选数量不一样（比如 FR 的"列举/计数"类问题需要更大的召回池，
AR 判断类问题不需要太多），原来这套 dynamic 逻辑写好了但从来没被调用过，一直被外部传入的
`--fixed-top-k 5` 短路掉。

**对应代码位置**：
- `benchmarks/memgallery/adapter.py: OmniMemAdapter._get_dynamic_top_k()` —— 已有的类别 top_k 表
  （AR:10, CD/VS/VR/TTL:15, TR:20, KR:20, MR:25, FR:30，外加 list/count/专有名词的加成逻辑）
- `benchmarks/memgallery/adapter.py: OmniMemAdapter.recall()` —— 把原来写死的
  `effective_top_k = self.fixed_top_k or 5` 改成：`fixed_top_k` 为 `None` 时才调
  `self._get_dynamic_top_k(query_text, category)`
- `benchmarks/memgallery/run_memgallery.py` —— CLI 参数 `--fixed-top-k` 默认值从 `5` 改成 `None`
  （不传这个参数就是 dynamic；传了具体数字才是固定 top_k）

## 改动 2：修复约 1.7% QA 直接返回"生成失败"的问题

**Step**：查日志发现真实原因不是图片编码坏了，而是 vLLM 报
`"decoder prompt is longer than the maximum model length of 32768"`——检索到的文本 + 最多 4 张
memory 图片的 token 总数超过了模型上限。原来的重试逻辑是原样重试同一个请求，必然还是超，等于
白白重试。现在改成：检测到这个报错后，**丢弃本次请求里所有的图片、只用文本重试**，而不是原样
重试同一个超长请求。

**对应代码位置**：
- `omni_memory/orchestrator.py: OmniMemoryOrchestrator.generate_answer_from_context()` —— 新增的
  重试逻辑：`err_text` 里检测到 `"maximum model length"` 或 `"decoder prompt"` 字样时，把
  `remaining_parts`（图片 content parts）清空后 `continue` 重试，而不是原样重试
- 顺带修复三处 `httpx.Client()` 默认 5 秒超时导致长请求被误判为"连接失败"的隐藏 bug（统一改成
  180 秒）：
  - `omni_memory/orchestrator.py: _get_llm_client()`
  - `omni_memory/processors/base.py: BaseProcessor._get_llm_client()`
  - `omni_memory/knowledge/entity_extractor.py: EntityExtractor._get_llm_client()`

## 改动 3：把 text MAU 和 visual MAU 合并成一个 chunk

**Step**：原来一个带图片的对话轮次会生成两个独立 MAU（`add_text()` 存对话文本一个、
`add_visual_with_caption_embedding()` 存图片 caption 另一个），两个向量在检索时互相竞争同一个
top-k 名额。现在改成：把对话文本 + `image_id` + `image_caption` 拼成一段文本，只调一次
`add_text()`，生成单一 MAU、单一 embedding 向量；真实图片路径挂在这个 MAU 的 `raw_pointer` 上，
retrieval 阶段照样能把真实图片传给回答模型。这是 Offline 项目"一个 dialogue round = 一个
chunk"设计的做法。

**对应代码位置**：
- `benchmarks/memgallery/adapter.py: OmniMemAdapter.store()` —— 不再分别调
  `add_text()` + `add_visual_with_caption_embedding()`，改成拼好 `combined_text`
  （文本 + `image_id:` + `image_caption:`）后只调一次 `add_text(combined_text, tags)`，
  再把返回的 `mau.raw_pointer` 设成真实图片路径、打上 `has_image` 标签，
  `self.orchestrator.mau_store.update(mau)` 落盘
- `benchmarks/memgallery/adapter.py: OmniMemAdapter._collect_image_paths()` —— 原来靠
  `modality_type == "visual"` 判断是否有图，现在合并后的 MAU 是 `modality_type == "text"`
  但带 `raw_pointer`，所以改成不看 modality_type，只要 `raw_pointer` 能解析成真实存在的文件
  就当作有图

**影响**：这个改动会改变 MAU 的存储结构，**不能复用旧的 `eval_data/` 记忆**，需要用一个新的
`--memory-run-name` 重新走一遍存储流程。

## 启动命令

前提：vLLM 已经在本机跑起来 `Qwen/Qwen3-VL-4B-Instruct`（`curl http://localhost:8000/v1/models`
能看到）。如果 GPU 显存被 vLLM 占满，建议把 embedding 用的 GPU 单独指定出来，避免 fallback 到
CPU（见下面 `QWEN3VL_EMBEDDING_DEVICE` 环境变量）。

### 第一步：只建 MAU 记忆索引（不跑 QA）

```bash
cd /data1/haozhen/Visual_Primitives/Offline/SimpleMem/OmniSimpleMem

QWEN3VL_EMBEDDING_DEVICE=cuda:2 python3 -u benchmarks/memgallery/run_memgallery.py \
  --port 8000 \
  --model "Qwen/Qwen3-VL-4B-Instruct" \
  --all-datasets \
  --run-name merged_chunk_dynamic_topk_full20 \
  --memory-run-name merged_chunk_qwen3emb2048_qwen3vl4b_full20 \
  --reuse-memory \
  --build-only \
  --qa-max-tokens 192 \
  --qa-timeout 180 \
  --qa-retries 2 \
  --query-embedding-dir /data1/haozhen/Visual_Primitives/Offline/Offline/artifacts/query_embeddings \
  --max-memory-images 4
```

`--reuse-memory` 配合 `--build-only`：已经建好的数据集会被跳过，只重新建还没建过（或者被打断、
不完整）的数据集。**如果某个数据集是被打断的（存到一半），需要先删掉它对应的目录再重新跑**，
否则 `--reuse-memory` 会误判"已经建好"直接跳过，留一个不完整的记忆：

```bash
rm -rf /data1/haozhen/Visual_Primitives/Offline/SimpleMem/OmniSimpleMem/eval_data/merged_chunk_qwen3emb2048_qwen3vl4b_full20/<被打断的数据集名>
```

### 第二步：MAU 索引建完之后，跑全量 1711 题 QA

```bash
cd /data1/haozhen/Visual_Primitives/Offline/SimpleMem/OmniSimpleMem

python3 -u benchmarks/memgallery/run_memgallery.py \
  --port 8000 \
  --model "Qwen/Qwen3-VL-4B-Instruct" \
  --all-datasets \
  --run-name merged_chunk_dynamic_topk_full20 \
  --memory-run-name merged_chunk_qwen3emb2048_qwen3vl4b_full20 \
  --reuse-memory \
  --qa-max-tokens 8000 \
  --qa-timeout 180 \
  --qa-retries 2 \
  --query-embedding-dir /data1/haozhen/Visual_Primitives/Offline/Offline/artifacts/query_embeddings \
  --max-memory-images 4
```

不加 `--build-only` 就会在存储/复用记忆之后接着跑 QA；不加 `--fixed-top-k` 就是 dynamic top_k
（对应改动 1）。

### 跑完之后算 F1/EM/hit@5

```bash
python3 benchmarks/memgallery/summarize_metrics.py \
  --run-root results/merged_chunk_dynamic_topk_full20 \
  --output /tmp/merged_chunk_dynamic_topk_metrics.json
```

## query 索引说明

QA 问题的 embedding 缓存（`/data1/haozhen/Visual_Primitives/Offline/Offline/artifacts/query_embeddings`，
覆盖全部 1711 题）只依赖题目文本本身，跟 MAU 怎么切、怎么合并无关，本次三处改动都不需要重建它，
`--query-embedding-dir` 直接指过去复用即可。
