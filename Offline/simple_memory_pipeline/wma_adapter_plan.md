# Plan: AgentMem (simple_memory_pipeline) × WorldMemArena adapter

目标：让 `simple_memory_pipeline`（INSERT/UPDATE/DELETE/NOOP 的 AgentMem baseline）作为一个
baseline 跑通 WorldMemArena (WMA) 的 eval_framework，复用 WMA 已有的 runner、QA、judge、
指标体系，不改动 simple_memory_pipeline 的核心语义。

## 0. 方向选择（已定）

两条路线：

- **A（选定）：在 WMA 侧写 online adapter** —— 新建
  `WorldMemArena/eval_framework/memory_adapters/agentmem_adapter.py`，实现
  `MemoryAdapter` 抽象接口（`base.py`），内部 import 并驱动 simple_memory_pipeline 的
  `MemoryExecutor` / `MemoryBank` / `operations` / `QwenMemoryEmbedder`。
- B（否决）：像 memgallery 那样离线预构建索引再挂只读 adapter。不可行：WMA 的
  `pipeline/runner.py` 协议是 **逐 session ingest → 每 session 导出 snapshot/delta →
  checkpoint QA 穿插在 session 流中间**，且 gold_state 按 session 对齐评 memory 质量。
  离线预构建无法产出 per-session snapshot/delta，会丢掉 WMA 一半的评测面。

选 A 的另一个原因：AgentMem 本来就是流式逐 round 处理，与 runner 的 ingest_turn 协议天然吻合。

## 1. WMA 侧协议（对接面，均已读码确认）

- 数据：`load_worldmemarena(data_dir)` 直接读 `WorldMemArena/WorldMemArena/` 原生目录
  （agent/gui、agent/embodied、lifelong/project、lifelong/personal，461 个 sample；
  `--split small` 用 `small_ids.json` 取 150 个）。图片路径已在 loader 里解析成绝对路径。
- runner 驱动（`pipeline/runner.py::run_eval_sample`）：
  1. `adapter.reset()`（每个 sample 一个新 adapter 实例，cli.py L1236，线程池并发跑 sample）
  2. 逐 turn `adapter.ingest_turn(NormalizedTurn)`；turn 有
     `session_id / turn_index / role / text / attachments(caption, image_id, file_path) / timestamp`
  3. session 末尾 `adapter.end_session(session_id)`
  4. `adapter.snapshot_memories() -> list[MemorySnapshotRecord]`、
     `adapter.export_memory_delta(session_id) -> list[MemoryDeltaRecord]`
     （op ∈ add/update/keep/suppress/archive）
  5. checkpoint 命中时 `adapter.retrieve(question, top_k, category=<FR/TR/VS/...>)
     -> RetrievalRecord`，答案由框架的 answer_fn（Base Model VLM）生成，
     `RetrievalItem.image_path` 若给出会被 answer 阶段作为图片证据（受
     `baselines.<name>.mm_mode` 与 `base_model.mode` 门控）。
- `get_capabilities()` 必须返回 `{"available": True, ...}` 否则 runner 直接拒跑。

## 2. simple_memory_pipeline 侧可复用组件（零改动或微改动）

- `MemoryExecutor`（executor.py）：`execute(operations, chunk_text, retrieved_memories)` +
  `apply_to_memory_bank(results, bank, retrieved_indices, event_metadata)`。原样复用。
- `MemoryBank`（memory_bank.py）：add/update/delete/retrieve（余弦 top-k），
  metadata 归并逻辑已处理 list 类 key。原样复用。
- `operations.get_default_operations()`：INSERT/UPDATE/DELETE/NOOP 四操作 prompt
  （2026-07-14 修订版，含 salient-content INSERT）。原样复用——**不要**为 WMA 单独改 prompt，
  否则两个 benchmark 的 baseline 定义分叉。
- `backends.OpenAICompatibleBackend`：executor LLM 走 vLLM Qwen3-VL-4B server
  （http://127.0.0.1:8000/v1，screen `qwen_vl_answer_4b`）。原样复用。
- `memgallery/qwen_embedder.QwenMemoryEmbedder`：Qwen3-VL-Embedding-2B，cuda:0，2048 维。
  原样复用，但在 adapter 侧包一层**进程级单例 + threading.Lock**（见 §4 风险）。

唯一微改动需求：无。adapter 全部新代码放 WMA 侧。

## 3. 新文件：`eval_framework/memory_adapters/agentmem_adapter.py`

### 3.1 导入方式

simple_memory_pipeline 不在 WMA 的包树里。adapter 顶部做 path 注入：

```python
_PIPELINE_ROOT = Path(os.getenv(
    "SIMPLE_MEMORY_PIPELINE_ROOT",
    "/data1/haozhen/Visual_Primitives/Offline/Offline",
))  # 亦可从 config.yaml baselines.AgentMem.pipeline_root 读
if str(_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_ROOT))
```

注意 `offline_memgallery_qwen`（embedder 依赖）也在同一根目录下，一次注入两个包都可用。
import 失败不抛异常，记录到 `self._integration_error`，由 `get_capabilities()` 返回
`available: False` + 错误详情（框架会打印原因，模式与 amem/m2a 的 adapter 一致）。

### 3.2 turn → chunk（事件粒度）

AgentMem 的处理单元是 "round"（对应 memgallery 实验的 dialogue round），不是单 turn。
映射规则（`chunk_mode: "round"`，默认）：

- `ingest_turn` 只把 turn 追加进 `self._round_buffer`；
- 当收到 **user turn 且 buffer 里已有 assistant turn** 时，先 flush 旧 round 再开新 round
  （即一个 round = 连续 user turns + 其后的连续 assistant turns）；
- `end_session()` 强制 flush 残余 buffer。

chunk 文本格式（与 memgallery 事件文本风格保持一致，含时间戳供 TR 题用）：

```
[S03] [2024-05-12 10:32] user: <text>
image_id: S03_img_1
image_caption: <caption>          # 仅 mm_mode ∈ {caption, image} 时保留
[S03] [2024-05-12 10:33] assistant: <text>
```

备选粒度作为可配参数留出：`chunk_mode: "turn"`（逐 turn 调 executor，成本 ×2，不默认）、
`"session"`（太粗，丢 update 时序，仅调试用）。

### 3.3 flush 一个 round 的处理流程（即 builder.py 主循环的移植）

```
query_vec  = embedder.embed_texts(chunk_text, mode="query")
retrieved, indices = bank.retrieve(query_vec, top_k=ops_top_k)      # ops_top_k=5，与 memgallery build 一致
raw, actions = executor.execute(operations, chunk_text, retrieved)
executor.apply_to_memory_bank(actions, bank, indices, event_metadata)
bank.step()
```

`event_metadata` 按 input_loader.MemoryEvent.metadata 的 key 约定构造：
`dataset=sample_id, session_id, round_id, dialogue_id=f"{session_id}_r{round_id}",
source_dialogue_ids, image_ids, image_paths, image_captions, timestamp`。
这样 `MemoryBank` 现成的 metadata 归并（UPDATE 合并 provenance）直接生效。

每 round 追加一行 build trace（raw_response / actions / memory_count）到
adapter 内部 list，最终塞进 `retrieve()` 的 `raw_trace` 与 capabilities，便于事后审计
NOOP 率——与 memgallery 的 build_trace.jsonl 对齐口径。

### 3.4 snapshot / delta

- `snapshot_memories()`：遍历 `bank.get_items()`，
  `MemorySnapshotRecord(memory_id=item.memory_id, text=item.content,
  session_id=item.metadata.get("session_id",""), status="active",
  source="AgentMem", metadata=item.metadata)`。
- `export_memory_delta(session_id)`：维护 `self._last_snapshot: dict[memory_id, content]`，
  每次调用后刷新。diff 规则：
  - 新 id → `op="add"`
  - 同 id 内容变化 → `op="update"`（UPDATE 或 INSERT-dedup 触发的 metadata 变化只在
    content 变化时才报 update；纯 provenance 合并报 `keep` 或不报，取**不报**，噪声小）
  - id 消失（DELETE）→ `op="suppress"`

### 3.5 retrieve

```
query_vec = embedder.embed_texts(query, mode="query")
contents, indices = bank.retrieve(query_vec, top_k)
items = [RetrievalItem(rank=i, memory_id=..., text=content, score=cos,
                       image_path=metadata["image_paths"][0] or None), ...]
```

- `bank.retrieve` 现在不返回分数；adapter 里自己算一遍余弦（或给 MemoryBank 加一个
  `retrieve_scored`，二选一，倾向 adapter 内自算避免动 pipeline 代码）。
- `category` kwarg（VS/FR/TR…）v1 忽略——基线行为就是纯向量检索；category-aware 是
  Omni-SimpleMem 的特性，不属于本 baseline。留 TODO。
- `mm_mode="image"` 时 `image_path` 给绝对路径，answer 阶段 VLM 能看到原图；
  text/caption 模式置 None。

### 3.6 与 a/b/c 消融的对应

| WMA config `mm_mode` | 等价 memgallery 模式 | chunk 内容 | RetrievalItem.image_path |
|---|---|---|---|
| `text` | a | 纯对话文本，剥掉 caption | None |
| `caption` | b | 文本 + image_caption 行 | None |
| `image` | c | 文本 + caption + metadata 存 image_paths | 绝对路径 |

检索通道 v1 一律是文本向量（与 memgallery 模式 c 的主通道一致）；
image_vectors 双通道检索（SimpleMemoryIndex 的 VS 路径）列为 v2 可选项，不阻塞。

## 4. 注册与配置

1. `registry.py`：
   - `EXTERNAL_ADAPTER_KEYS` 加 `"AgentMem"`；
   - `create_agentmem_adapter()` 工厂（lazy import adapter 模块）；
   - `EXTERNAL_ADAPTER_REGISTRY["AgentMem"] = create_agentmem_adapter`。
2. `config.yaml` 增加：

```yaml
  AgentMem:
    mm_mode: "caption"                 # text | caption | image，对应 a/b/c
    chunk_mode: "round"
    ops_top_k: 5                       # executor 决策时检索的旧记忆条数
    pipeline_root: "/data1/haozhen/Visual_Primitives/Offline/Offline"
    executor_model: "Qwen/Qwen3-VL-4B-Instruct"   # 以 vLLM 实际 served name 为准，跑前用 /v1/models 确认
    executor_base_url: "http://127.0.0.1:8000/v1"
    executor_api_key_env: "AGENTMEM_EXECUTOR_API_KEY"   # 本地 vLLM 用 "EMPTY"
    executor_max_new_tokens: 1024
    embedder_model: "Qwen/Qwen3-VL-Embedding-2B"
    embedder_device: "cuda:0"
```

   读取用现成的 `resolve_baseline_param(name, key, default)`。
3. **embedder 单例**：cli 的 `per_baseline_workers=3` 会在 3 个线程各建一个 adapter；
   embedder 若随 adapter 实例化会加载 3 份 2B 模型 → cuda:0 直接 OOM
   （该卡还跑着 4B vLLM，只剩 ~7GB headroom，单份 embedder 已占 ~5GB）。
   方案：模块级 `_get_shared_embedder()`（double-checked lock），embed 调用外再套一把
   `threading.Lock` 串行化 GPU 前向。executor 走 HTTP，天然线程安全，每 adapter 各建
   `OpenAICompatibleBackend` 即可。

## 5. 测试与验收

1. **单元测试**（新文件 `eval_framework/` 侧或 simple_memory_pipeline/tests 侧均可，
   倾向放 WMA 侧 `eval_framework/tests/test_agentmem_adapter.py`）：
   - fake backend（固定返回 INSERT/UPDATE/DELETE 脚本）+ fake embedder（hash 向量），
     不碰 GPU；
   - 覆盖：round 切分（user/assistant 交替、连续同角色、end_session 残余 flush）、
     snapshot/delta 三种 op、retrieve 的 image_path 门控、`available=False` 路径
     （pipeline_root 不存在时不炸整个 cli）。
2. **冒烟**（真模型，先起 vLLM server 与确认 GPU 余量）：

```bash
cd /data1/haozhen/Visual_Primitives/Offline/WorldMemArena
python -m eval_framework.cli \
  --dataset WorldMemArena --dataset-type worldmemarena \
  --baseline AgentMem --sample-index 0 --max-sessions 3 \
  --per-baseline-workers 1 --output-dir eval_framework/results/agentmem_smoke
```

   验收点：session record 里 snapshot 非空、delta 的 add/update/suppress 与 trace 中的
   action 计数对得上、checkpoint QA 有 retrieval items、无 parse-failure 风暴
   （NOOP 率应与 memgallery 冒烟量级一致，~20%；残余 NOOP 是基线行为，不修）。
3. **规模评估前先算账**：全量 59,239 turns ≈ 3 万 round；executor ~1–2 s/round →
   全量单模式 8–16 h。建议顺序：单 sample 冒烟 → 一个子目录（如 lifelong/personal，
   20 sample）→ `--split small`。`per-baseline-workers` 提到 2–3 时瓶颈在共享 embedder
   锁 + vLLM 吞吐，先压测再放并发。

## 6. 明确不做 / 风险

- 不改 operations.py prompt、不为特定题型调 executor 行为（基线纯净性，见 memory 记录）。
- category-aware 检索、image_vectors 双通道、BM25 增强：全部 v2，避免和 SimpleMem/
  Omni-SimpleMem 的特性混淆。
- 风险：lifelong sample 单条 ~810 turns/30 sessions，MemoryBank 是 O(N) 全量余弦，
  N 在几百量级没问题；executor 的检索窗口只有 ops_top_k=5，长程 UPDATE 命中率天然受限
  ——这是 baseline 特性，记录进结果分析而非提前修补。
- 风险：WMA answer 阶段用 OPENAI_MODEL（GPT 系）作答，与 memgallery 实验里本地 Qwen 作答
  不同源；对比两边结果时注明 answer model 差异。
