# M2A 三 Benchmark Test Split：本地 vLLM 运行教程


## 1. 目标和固定协议

只运行 test split：

| Benchmark | Test sample | Test QA |
|---|---:|---:|
| Mem-Gallery | 4 | 275 |
| H2HMEM dyadic | 4 | 327 |
| H2HMEM multiparty | 1 | 33 |
| H2HMEM 合计 | 5 | 360 |
| WorldMemArena lifelong | 8 | 440 |

固定配置：

- 协议：`configs/test_baseline_matrix.json`
- Split manifest：`configs/multimodal_split_manifest.json`
- Baseline：`M2A`
- M2A 官方仓库：`https://github.com/Little-Fridge/M2A`
- 锁定上游 commit：`edd8c3b75bae8b2c9c1a0ac8ed67e38c2c2723f8`
- 建库与回答模型：`Qwen/Qwen3-VL-4B-Instruct`
- Embedding：`Qwen/Qwen3-VL-Embedding-2B`，`http://127.0.0.1:8001/v1`，维度 2048
- 回答端口：8013、8014、8015
- `temperature=0.0`
- 最终 benchmark QA `max_tokens=512`
- M2A 内部建库与检索 Agent `max_tokens=4096`；该上限只用于本地 Qwen3-VL 完成原版工具调用，不改变最终 benchmark QA 的 512-token 上限
- HTTP timeout：180 秒
- retries：2（当前 `openai==1.109.1` 的 SDK 默认值为 2；运行前必须核验）
- `top_k=7`
- LLM Judge：OpenRouter `openai/gpt-4o-mini`，temperature 0，max tokens 512
- Judge 调用不计入 MB/QA calls、cost 或 latency

除本教程明确记录的兼容修改外，不得改写 M2A 内部 Agent prompt、benchmark QA prompt、原版工具语义或检索顺序。

## 2. Benchmark-native 输入与 test 筛选

M2A 不读取固定 chunk JSONL；三个 harness 使用各自的原始 benchmark 数据和专用 builder，把原始轮次转换为带 `metadata.m2a_turns` 的 M2A 输入：

| Benchmark | M2A 输入构造入口 |
|---|---|
| Mem-Gallery | `build_chunks_from_data(dataset, data_dir, dataset_name)` |
| H2HMEM | `build_h2h_chunks_from_directory(data_dir, variant, conversation_ids)` |
| WMA lifelong | `build_wma_chunks_from_data(payload, sample_path.parent, sample_path=...)` |

`configs/multimodal_split_manifest.json` 只负责精确限定 test sample 和 question ID；Split manifest SHA256 为 `590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。不得重新划分 split，也不得纳入 Adversarial Query。

```bash
cd /data/haozhen/Memory-clean/Offline
sha256sum configs/multimodal_split_manifest.json
test -d /data/haozhen/Memory-clean/Mem-Gallery/benchmark/data
test -d /data/haozhen/Memory-clean/H2HMEM-main/dataset
test -d /data/haozhen/Memory-clean/WorldMemArena/WorldMemArena/lifelong
```

每次正式实验必须使用全新 Run-ID，从 test manifest 选中的原始数据重新构建 M2A 的 `raw.db`、`semantic.db` 和 image index；不得复用旧 memory、snapshot 或 retrieval trace。`run_manifest.json` 必须记录 builder 身份/版本、实际原始输入文件 manifest/hash 和 split manifest；固定 chunk JSONL 及其 hash 不属于 M2A 的输入协议或运行签名。

## 3. M2A 原版调用链

### 3.1 建库

M2A 不是“chunk 原文直接写 Semantic Store”。锁定版本的原版调用链是：

```text
M2AEvaluationWrapper.start_conversation / M2AAdapter.reset
  -> M2ASystem
  -> RawMessageStore + SemanticStore + ImageManager
  -> ChatAgent(update_memory=True, update_only=True)

每条原生消息
  -> ChatAgent.chat()
  -> 先追加 RawMessageStore
  -> ChatAgent 原版 update prompt 自主选择 update_memory/query_memory
  -> MemoryManager.update()
  -> 原版 MemoryManager update prompt
  -> search_semantic_memories / fetch_raw_messages / fetch_raw_messages_by_time
  -> add_memory / delete_memory
  -> SemanticStore + evidence_ids
  -> 文本 dense、Milvus BM25 sparse、图文 embedding 与 image index
```

`M2AAdapter.ingest()` 会读取上述 builder 生成的 `metadata.m2a_turns`，保留 source speaker、timestamp、source dialogue、session 和 image provenance，并逐 turn 调用原生 ChatAgent。这就是 M2A 的正式输入路径，不需要 fixed-Chunk → turn 桥接。

### 3.2 原版检索

```text
benchmark question/query image
  -> M2AEvaluationWrapper.question()
  -> 新建 ChatAgent(update_memory=False)，使用原版 retrieval system prompt
  -> ChatAgent 自主调用 query_memory
  -> MemoryManager.query()
  -> 原版 MemoryManager query prompt 自主多轮调用：
       search_semantic_memories
       fetch_raw_messages
       fetch_raw_messages_by_time
  -> SemanticStore.hybrid_search()
       text dense + Milvus BM25 sparse
       + text-to-image/image-to-image（题目或 memory 有图时）
       -> RRF
  -> MemoryManager 生成检索回复给 ChatAgent
```

原版 `search_semantic_memories` 的工具默认 `top_k=10`，Agent 可以自行指定；`SemanticStore.hybrid_search()` 内部各路径候选规模也有自己的原版值。实验协议不得把每个内部搜索强制改成 7，否则会改变 M2A 的检索决策空间。

### 3.3 回答

M2A ChatAgent 的原版 retrieval prompt只负责发起检索并得到 MemoryManager 的结果；正式 benchmark 答案不采用该 Agent 自带的一句式回答文本。Adapter 应把合规的最终 memory handoff 转为统一 `RetrievedMemory`，随后由对应 harness 的现有 QA prompt 和 `VLMAnswerClient` 生成答案：

- Mem-Gallery：`src/benchmarks/memgallery_harness/runner/prompts.py`
- H2HMEM：`src/benchmarks/h2hmem_harness/prompts.py`
- WMA：`src/benchmarks/wma_harness/runner/prompts.py`

当前 prompt SHA256：

| Benchmark | SHA256 |
|---|---|
| Mem-Gallery | `ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff` |
| H2HMEM | `4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f` |
| WMA | `6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250` |

任一 prompt hash 改变都必须使用新 Run-ID。M2A 内部 Agent prompt 必须保持锁定版本；当前 Adapter 用 AST/hash 对六个内部 prompt 进行核对。

### 3.4 Top-7 的准确施加层

M2A 的 Top-7 是**最终显式 memory handoff budget**，不是把内部所有 search tool 的 `top_k` 改成 7：

1. ChatAgent 与 MemoryManager 保留原版多轮 query/search/fetch 行为及 Agent 自选 `top_k`。
2. 对本题全部原版 semantic search 命中按原生返回顺序合并，以 `memory_id` 去重，并显式 handoff 最多 7 条给 benchmark QA prompt。
3. MemoryManager 返回 ChatAgent 的内部自然语言结果可以综合超过 7 条候选；这些属于 M2A 原版内部推理，不计入最终显式 handoff 的 Top-7。
4. Raw evidence 回查和内部 fetch 不另计入 Top-7，但必须保留 trace。
5. WMA 的最终 handoff memory 必须具有非空 session provenance，且完全属于当前 `visible_sessions`。

### 3.5 WMA checkpoint

严格顺序必须是：

```text
写入截至 checkpoint 可见的 session
  -> 完成 M2A update/build
  -> 当前 checkpoint 每道 QA：原生 Agent/Manager 检索 -> Top-7 -> harness prompt 回答完成并记录
  -> 当前 checkpoint 全部 QA 完成
  -> 才写入下一批未来 session
```

当前实现通过共享 answer pool 保留 QA 并行，但在 checkpoint 边界等待该 checkpoint 的全部答案完成；结果 trace 写入 `checkpoint_protocol.mode=answer_before_future_ingest`。问题自身及 QA scratch raw store 不得进入持久 conversation memory。任何 retrieved memory 的 provenance 都必须非空并完全属于该题的 `visible_sessions`，否则立即失败。

## 4. 允许与禁止的兼容行为

允许且必须声明：

- 用指定 Qwen3-VL-4B 替换原版模型；不改 M2A Agent prompt。
- 将 Qwen 文本形式 `<tool_call>` 规范化为 LangChain 需要的 structured tool call；若响应同时含普通 assistant content，则原样保留该 content 并记录其 SHA256。若仅有缺失逗号、冒号、引号或尾随分隔符等 JSON 表层语法错误，可在该兼容边界修复后继续，但仍须严格校验 `name + arguments` 完整 schema，并在 `AIMessage.additional_kwargs.qwen_tool_call_repairs` 记录原始 payload SHA256；不得改写工具名或参数语义。
- 仅在本地 loopback vLLM + Qwen3-VL 的 retrieval 阶段，将协议已要求必经的首个 `ChatAgent -> query_memory` 和首个 `MemoryManager -> search_semantic_memories` 请求改为 named tool choice，使 vLLM 对参数执行 schema-constrained decoding；后续 MemoryManager 工具轮次仍保持原版 `auto`，可以自主继续或停止。完整的裸 `name(key=value)` 输出只在工具名属于当前请求 schema、参数可安全解析且响应未截断时提升为 structured tool call；不得补全或接受截断参数。
- 远程 VLM 请求可对相同图片去重并生成受控 JPEG transport copy；原图片、embedding 输入、memory/provenance 不变。
- ChatAgent/MemoryManager 保持原迭代上限；若兼容服务在预算边界违反 `parallel_tool_calls=False` 一次返回多个调用，只保留剩余预算内的调用并审计丢弃数量。最后一个合法工具调用执行完毕后，使用完全不绑定 tools 的普通模型请求收尾，并记录 `tool_budget_exhausted`。硬上限之后不会执行额外工具调用；查询收尾响应为空时必须失败。
- QA 使用 disposable raw-message store，避免问题写入持久 memory。
- 最终答案使用 benchmark harness 的现有 QA prompt。

禁止：

- direct insert 到 Semantic Store、chunk 原文直写 semantic memory、绕过 ChatAgent/MemoryManager。
- 绕过上述 benchmark-native builder、使用另一套未记录的输入转换，或在不同 Run-ID 中静默改变 builder 行为。
- 无检索结果或工具失败时换 baseline、换模型/provider、复用旧 memory/trace 或注入 gold evidence。
- 图片缺失时把路径字符串当视觉输入，或用 caption-only 静默代替原图。
- 修改 M2A 内部 prompt、benchmark QA prompt、检索排序或原版工具循环来“提高成功率”。
- WMA 提前写入未来 session，或只冻结 evidence 但延后回答。

原版工具返回的 `No relevant semantic memories found`、错误字符串和迭代上限可以被记录，但正式运行若最终无 memory、未执行 `query_memory -> MemoryManager -> search_semantic_memories`、存在未处理 tool error，必须失败并报告，不能继续计分。

## 5. 运行前环境与代码检查

以下命令均已与当前 CLI 核对：

```bash
cd /data/haozhen/Memory-clean/Offline

/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/run_test_baseline_matrix.py --help
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/validate_m2a_original_run.py --help

git status --short -- \
  baselines/M2A \
  src/benchmarks/baseline_runtime/adapters/m2a.py \
  src/benchmarks/memgallery_harness/eval_memgallery.py \
  src/benchmarks/h2hmem_harness/eval_h2hmem.py \
  src/benchmarks/wma_harness/eval_wma.py

rg -n 'baseline == "M2A"|build_.*chunks_from_(data|directory)|memgallery_chunks|h2hmem_chunks|wma_chunks' \
  src/benchmarks/memgallery_harness/eval_memgallery.py \
  src/benchmarks/h2hmem_harness/eval_h2hmem.py \
  src/benchmarks/wma_harness/eval_wma.py
```

M2A 分支应明确命中这三个 benchmark-native builder；若改为固定 chunk loader、直接写 Semantic Store，或 builder 产物缺少可审计的 `m2a_turns`/provenance，则停止并报告。

检查配置和 SDK retries：

```bash
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  configs/defaults.json >/dev/null
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  configs/test_baseline_matrix.json >/dev/null
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python - <<'PY'
import openai
from openai._constants import DEFAULT_MAX_RETRIES
assert openai.__version__ == "1.109.1"
assert DEFAULT_MAX_RETRIES == 2
print("M2A runtime SDK/retries OK")
PY
```

检查 test manifest 数量：

```bash
PYTHONPATH=src /data/haozhen/miniconda3/envs/pipeline_repro/bin/python - <<'PY'
from pathlib import Path
from evidence_policy.split_manifest import SplitManifestIndex

x = SplitManifestIndex(Path("configs/multimodal_split_manifest.json"))
expected = {
    "mem_gallery": (4, 275),
    "h2hmem_dyadic": (4, 327),
    "h2hmem_multiparty": (1, 33),
    "worldmemarena_lifelong": (8, 440),
}
for source, target in expected.items():
    rows = x.conversations("test", data_source=source)
    actual = (len(rows), sum(len(row.question_ids) for row in rows))
    assert actual == target, (source, actual, target)
    print(source, actual)
PY
```

## 6. 启动本地服务

先确认 GPU3/4/5 和端口没有承载其他任务；不得杀掉不属于本实验的进程：

```bash
nvidia-smi -i 3,4,5
ss -ltnp | rg ':(8001|8013|8014|8015)\b' || true
curl -fsS http://127.0.0.1:8001/v1/models
```

如 8013/8014/8015 尚未启动，使用仓库现有 `scripts/serve_vllm.sh`：

Qwen3-VL-4B 当前 bundled chat template 使用 Hermes JSON `<tool_call>{"name": ..., "arguments": ...}</tool_call>` 包络，因此必须保留脚本默认的 `TOOL_CALL_PARSER=hermes`；不得覆盖成格式不匹配的 `qwen3_xml`。

```bash
cd /data/haozhen/Memory-clean/Offline
mkdir -p outputs/_services

tmux new-session -d -s m2a_vllm_gpu3_8013 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=3 PORT=8013 sh scripts/serve_vllm.sh \
   > outputs/_services/m2a_vllm_gpu3_8013.log 2>&1"

tmux new-session -d -s m2a_vllm_gpu4_8014 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=4 PORT=8014 sh scripts/serve_vllm.sh \
   > outputs/_services/m2a_vllm_gpu4_8014.log 2>&1"

tmux new-session -d -s m2a_vllm_gpu5_8015 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=5 PORT=8015 sh scripts/serve_vllm.sh \
   > outputs/_services/m2a_vllm_gpu5_8015.log 2>&1"
```

服务就绪后逐一确认模型名：

```bash
for port in 8013 8014 8015; do
  curl -fsS "http://127.0.0.1:${port}/v1/models" \
    | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
      'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="Qwen/Qwen3-VL-4B-Instruct" for x in d.get("data", []))'
done
```

Embedding 必须实测维度 2048：

```bash
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["M2A preflight"]}' \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048'
```


## 7. 正式 tmux 启动

生成全新 Run-ID；一个 matrix 进程接收三个 endpoint，三个 M2A benchmark 使用独立 worker/result/state/checkpoint：

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID="m2a_local_$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s "m2a_local_$RUN_ID" \
  "cd /data/haozhen/Memory-clean/Offline && \
   M2A_PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --baseline M2A \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint http://127.0.0.1:8013/v1 \
   --endpoint http://127.0.0.1:8014/v1 \
   --endpoint http://127.0.0.1:8015/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --defaults /data/haozhen/Memory-clean/Offline/configs/defaults.json \
   --efficiency-config /data/haozhen/Memory-clean/Offline/configs/model_efficiency.json \
   --split-manifest /data/haozhen/Memory-clean/Offline/configs/multimodal_split_manifest.json \
   --output-root /data/haozhen/Memory-clean/Offline/outputs \
   --skip-smoke \
   --run-id '$RUN_ID' \
   > 'outputs/_runs/$RUN_ID/launcher.log' 2>&1"

echo "$RUN_ID"
```

命令显式使用 `--skip-smoke`，直接进入正式任务。

## 8. 监控与断点恢复

```bash
tmux ls | rg 'm2a_local'
tmux attach -t "m2a_local_<Run-ID>"
cat "outputs/_runs/<Run-ID>/status.json"

tail -n 80 "outputs/_runs/<Run-ID>/_logs/baseline/m2a__mem_gallery.log"
tail -n 80 "outputs/_runs/<Run-ID>/_logs/baseline/m2a__h2hmem.log"
tail -n 80 "outputs/_runs/<Run-ID>/_logs/baseline/m2a__worldmemarena.log"
tail -n 80 "outputs/_runs/<Run-ID>/_logs/judge/m2a__mem_gallery.log"
```

恢复时使用第 7 节完全相同的 Run-ID、配置、split manifest、原始输入、builder、endpoint 和参数重新启动。Runner 会复用签名匹配的 `.checkpoint/samples`；未完成 sample 会重建其独立 M2A state。配置或输入 hash 改变时不得恢复，必须换新 Run-ID。

不得为了“从头重跑”删除旧结果；使用新的 Run-ID。

## 9. Outputs 与验收

正式结果目录：

```text
/data/haozhen/Memory-clean/Offline/outputs/Mem-Gallery/M2A/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/H2HMEM/M2A/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/WorldMemArena/M2A/<Run-ID>/
```

当前代码实际生成或要求的主要产物：

- `results.json`
- `retrieval_trace.jsonl`
- `pipeline_qa.jsonl`
- `memory/memory_snapshot.jsonl`
- `memory/datasets/<sample>/raw.db`、`semantic.db`、`raw.json`、`semantic.json`、`image_manager.json`
- `memory/datasets/<sample>/m2a_execution_trace.jsonl`、`m2a_conformance.json`
- `run_manifest.json`
- `metrics.json`、`efficiency_metrics.json`、`call_metrics.json`
- `call_trace.jsonl` 与 `call_traces/<sample>.jsonl`
- `.checkpoint/`
- `llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json`、`llm_judge_metrics.json`
- Mem-Gallery/WMA 的 `memory_metrics.json`
- H2HMEM 的 `prediction_dyadic.json`、`prediction_multi_party.json`

验收必须同时满足：

1. 三组 QA 数分别为 275、360、440，顺序、question ID 和 sample 集与 test manifest 完全一致；无 Adversarial Query 混入。
2. `run_manifest.json` 的 split、manifest SHA、原始输入文件 manifest/hash、builder 身份/版本、模型、prompt SHA、Top-7 和 M2A commit正确；execution trace 能证明实际使用了声明的 benchmark-native 输入转换。
3. 每个 sample 使用独立全新 Raw/Semantic/Image state，所有 semantic memory 有有效 evidence range；图片 memory 有 image embedding。
4. 建库确实经过 `ChatAgent -> MemoryManager -> SemanticStore`，无 direct insert/fallback。
5. 每题确实经过 `ChatAgent query_memory -> MemoryManager.query -> search_semantic_memories`；内部候选数允许超过 7，但最终交给 benchmark QA prompt 的 distinct semantic memory 不超过 7。
6. WMA 所有 retrieval provenance 是 `visible_sessions` 子集，并且回答落盘时间早于下一批未来 session ingest。
7. 结果无 answer/tool error、重复问题、未完成 Judge；Judge count 与 QA count一致，Judge 不计入效率指标。

指标按 `docs/baselines.md` 固定字段输出：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- F1、EM、Judge：test QA 的 qa-wise average。
- Cost-MB/Lat-MB：memory build 的 USD/sample 与 seconds/sample。
- Cost-QA/Lat-QA：retrieval + answer 的 USD/sample 与 seconds/sample。
- `#Calls (MB+QA)=(MB calls+answer calls)/sample 数`；retrieval calls单独保存在 trace，Judge 完全排除。

## 10. 必须停止并报告

遇到以下任一情况立即停止，不得静默折中：split manifest 或原始输入 hash 不符；M2A builder/`m2a_turns` 与 manifest 声明不一致；内部 prompt hash 不符；最终显式 handoff 超过 7；ChatAgent/MemoryManager 被绕过；图片、embedding 或 tool call失败；WMA 先写未来 session；QA/Judge 数量不符；需要更换模型、prompt、provider、检索排序或旧 memory。

## 11. 给 Codex 的快捷指令

> 严格遵循本教程。M2A 使用 test manifest 选中的原始 benchmark 数据和三个专用 builder，不使用固定 chunk JSONL；核对 builder/原始输入 manifest、最终显式 handoff ≤7 与 WMA 时序。使用全新 Run-ID、GPU3/4/5、8013/8014/8015 和 embedding 8001，在 tmux 直接运行 275/360/440 QA，并完成 M2A 专用验收、Judge、调用追踪和结果验收。
