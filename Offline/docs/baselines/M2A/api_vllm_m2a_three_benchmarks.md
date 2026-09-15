# M2A 三 Benchmark Test Split：OpenRouter API 运行教程


## 1. 目标与固定配置

| Benchmark | Test sample | Test QA |
|---|---:|---:|
| Mem-Gallery | 4 | 275 |
| H2HMEM dyadic | 4 | 327 |
| H2HMEM multiparty | 1 | 33 |
| H2HMEM 合计 | 5 | 360 |
| WorldMemArena lifelong | 8 | 440 |

固定配置：

- Baseline：`M2A`
- M2A 官方仓库：`https://github.com/Little-Fridge/M2A`
- 锁定 commit：`edd8c3b75bae8b2c9c1a0ac8ed67e38c2c2723f8`
- 协议：`/data/haozhen/Memory-clean/Offline/configs/test_baseline_matrix.json`
- Split manifest：`/data/haozhen/Memory-clean/Offline/configs/multimodal_split_manifest.json`
- API defaults：`/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json`
- Efficiency config：`/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini`
- API key 文件：`/data/haozhen/Memory-clean/Nvida_api/Openrouter_api`
- 建库与回答：OpenRouter `openai/gpt-5-mini`
- API endpoint：`https://openrouter.ai/api/v1`
- `reasoning_effort=minimal`
- temperature 0；QA max tokens 512；M2A 建库/内部检索 max tokens 2048；WMA 每次最多合并 4 个完整 round；timeout 180 秒，retries 2
- Embedding：本地 `Qwen/Qwen3-VL-Embedding-2B`，端口 8001，维度 2048
- `top_k=7`
- Judge：OpenRouter `openai/gpt-4o-mini`，temperature 0，max tokens 512
- Judge 请求不计入 MB/QA calls、cost 或 latency

`config_gpt-5-mini` 中当前成本参数为输入 0.25 USD/M tokens、输出 2.00 USD/M tokens；latency 参数为 base 3.47 秒、输入 `0.000003` 秒/token、输出 `0.01754386` 秒/token、图片 `0.30` 秒/image。指标计算必须读取该文件，不能把 key 或价格写死进 runner。

API key 只能从文件加载到进程环境，不得粘贴进文档、命令参数、日志或终端输出。

## 2. Benchmark-native 输入与 test 筛选

M2A 不读取固定 chunk JSONL；三个 harness 使用各自的原始 benchmark 数据和专用 builder，把原始轮次转换为带 `metadata.m2a_turns` 的 M2A 输入：

| Benchmark | M2A 输入构造入口 |
|---|---|
| Mem-Gallery | `build_chunks_from_data(dataset, data_dir, dataset_name)` |
| H2HMEM | `build_h2h_chunks_from_directory(data_dir, variant, conversation_ids)` |
| WMA lifelong | `build_wma_chunks_from_data(...)` 后由 `batch_wma_rounds_for_m2a(..., rounds_per_batch=4)` 合并 |

`configs/multimodal_split_manifest.json` 只负责精确限定 test sample 和 question ID；Split manifest SHA256 为 `590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。不得重新划分 split，也不得纳入 Adversarial Query。

```bash
cd /data/haozhen/Memory-clean/Offline
sha256sum configs/multimodal_split_manifest.json
test -d /data/haozhen/Memory-clean/Mem-Gallery/benchmark/data
test -d /data/haozhen/Memory-clean/H2HMEM-main/dataset
test -d /data/haozhen/Memory-clean/WorldMemArena/WorldMemArena/lifelong
```

每次正式任务使用全新 Run-ID，从 test manifest 选中的原始数据重新构建独立的 M2A Raw/Semantic/Image state；三个 benchmark 不得共享进程内 memory、数据库、sample checkpoint 或输出目录。`run_manifest.json` 必须记录 builder 身份/版本、实际原始输入文件 manifest/hash 和 split manifest；固定 chunk JSONL 及其 hash 不属于 M2A 的输入协议或运行签名。

## 3. M2A 原版流程与实验 handoff

### 3.1 建库

```text
test manifest 选中的原始 benchmark 数据
  -> benchmark-native builder 生成 `metadata.m2a_turns`
  -> M2ASystem
  -> RawMessageStore + SemanticStore + ImageManager
  -> ChatAgent(update_only=True) 原版 update prompt
  -> update_memory/query_memory
  -> MemoryManager.update() 原版 prompt
  -> search/fetch/add/delete 原版工具
  -> Semantic Memory + evidence_ids
  -> text dense + Milvus BM25 sparse + multimodal/image index
```

上述原始数据与专用 builder 是正式输入路径。严禁 chunk 原文 direct insert 到 Semantic Store，或因 API tool call失败而绕过 ChatAgent/MemoryManager。

WMA 的 API 适配版本在同一 session 内把最多 4 个完整 user/assistant round 标注后合并为一次 ChatAgent update；不跨 session，遇到第二个含图 source turn 时提前切块。原始 `m2a_turns`、speaker、timestamp、source dialogue 和图片 provenance 全部保留。WMA 固定时间格式优先本地确定性解析，无法识别时才回退原版时间转换 LLM。该策略是为控制 API 建库成本而加入的实验适配，并非 M2A 官方默认 chunk 规则。

### 3.2 检索

```text
benchmark query/query image
  -> M2AEvaluationWrapper.question()
  -> ChatAgent(update_memory=False)，使用原版 retrieval prompt
  -> query_memory
  -> MemoryManager.query()，使用原版 query prompt
  -> Agent 自主多轮 search_semantic_memories / fetch_raw_messages / fetch_raw_messages_by_time
  -> SemanticStore.hybrid_search：
       text dense + BM25 sparse + text-to-image/image-to-image
       -> RRF
  -> MemoryManager 返回检索结果给 ChatAgent
```

原版 search tool 默认 `top_k=10`，Agent 可以自行设置；不得为了实验 Top-7 改写每次内部搜索为 7。

### 3.3 Top-7 定义

Top-7 施加在**原生多轮检索完成后的最终 semantic-memory handoff**：

1. 保留 ChatAgent/MemoryManager 原版多轮工具决策和内部候选规模。
2. 汇总本题全部 `search_semantic_memories` 命中，按原生 first-seen 顺序以 `memory_id` 去重。
3. 最终交给 benchmark QA prompt 的 distinct semantic memory 最多 7 条；MemoryManager 返回 ChatAgent 的内部自然语言结果可以综合超过 7 条候选，这属于原版内部推理。
4. `fetch_raw_messages*` 和内部 evidence 回查不另计入 Top-7，但必须可追溯。
5. WMA 的最终 handoff memory 必须具有非空 session provenance，且完全属于当前 `visible_sessions`。

### 3.4 回答 Prompt

M2A 内部 Agent prompt 保持锁定版本，仅用于 update/query。最终 benchmark 答案由 harness 当前 QA prompt + `VLMAnswerClient` 生成：

| Benchmark | Prompt 文件 | 当前 SHA256 |
|---|---|---|
| Mem-Gallery | `src/benchmarks/memgallery_harness/runner/prompts.py` | `ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff` |
| H2HMEM | `src/benchmarks/h2hmem_harness/prompts.py` | `4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f` |
| WMA | `src/benchmarks/wma_harness/runner/prompts.py` | `6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250` |

不采用 M2A ChatAgent 自带的一句式答案作为最终 benchmark 答案。任一 Agent/QA prompt hash 变化都必须重新完成运行前检查并使用新 Run-ID。

### 3.5 WMA checkpoint

每个 checkpoint 严格执行：写入当前可见 session → M2A 完成 update → 当前 QA 的原生检索、Top-7、harness 回答 → 等待该 checkpoint 全部答案完成 → 才写未来 session。共享 answer pool 只并行同一可见状态下的 QA，checkpoint barrier 不允许未来 ingest 提前；结果 trace 写入 `checkpoint_protocol.mode=answer_before_future_ingest`。QA scratch raw store 不得污染持久 memory，retrieval provenance 必须非空且完全属于 `visible_sessions`。

## 4. 允许与禁止的兼容行为

允许并必须记录：

- 原版模型替换为 OpenRouter `openai/gpt-5-mini`，reasoning effort minimal；Agent prompt不变。
- WMA 同一 session 内最多 4 个完整 round 合并为一次带来源标签的 ChatAgent update；第二个含图 source turn 前强制切块，且 run manifest 必须记录批大小和图片边界规则。
- WMA 原生固定 timestamp 使用确定性本地解析；未知格式回退原版时间转换 LLM。
- Qwen 文本 `<tool_call>` 只做结构正规化，不改变工具名、参数和循环；若响应同时含普通 assistant content，则原样保留并记录其 SHA256。
- 相同图片在单次远程请求中去重，并使用受控 JPEG transport copy；原始图片、embedding 和 provenance不变。
- 若兼容 API 模型把三个以上的显式 raw-message ID 放入一个 evidence 内层列表（例如 `[[53,54,51]]`），入库前无损规范化为排序、合并后的合法范围（例：`[[51,51],[53,54]]`）；不得删除任何 ID，且 `semantic_log.json` 必须保留原值和规范值。
- M2A 建库/内部检索响应如返回 `finish_reason=length|max_tokens` 必须立即失败，不得消费被截断的自然语言或工具调用。
- ChatAgent/MemoryManager 保持原迭代上限；若兼容服务在预算边界违反 `parallel_tool_calls=False` 一次返回多个调用，只保留剩余预算内的调用并审计丢弃数量。执行最后一个合法工具调用后，使用完全不绑定 tools 的普通模型请求生成收尾响应，并在 execution trace 记录 `tool_budget_exhausted`。硬上限之后不会执行额外工具调用；查询收尾响应为空时必须失败。
- QA 使用 disposable raw-message store，最终答案改用 benchmark QA prompt。

禁止：direct semantic insert；绕过已声明的 benchmark-native builder 或静默改变输入转换；caption/path 字符串替代失败图片；切换模型/provider；复用旧 memory/snapshot/retrieval；注入 gold；修改内部 prompt、hybrid/RRF 排序或工具循环；吞掉 tool/API error；WMA 延迟回答；三个 worker 共用可变 state。

原版工具的空结果/错误字符串必须落盘；若最终没有 memory、缺少 `query_memory -> MemoryManager -> search_semantic_memories`，或存在未处理 tool error，则本题和正式任务必须失败，不能计分。

## 5. 环境、配置和 CLI 核验

不输出 key 地检查文件和 CLI：

```bash
cd /data/haozhen/Memory-clean/Offline

/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json >/dev/null
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini >/dev/null
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api

/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/run_test_baseline_matrix.py --help
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/validate_m2a_original_run.py --help

rg -n 'baseline == "M2A"|build_.*chunks_from_(data|directory)|memgallery_chunks|h2hmem_chunks|wma_chunks' \
  src/benchmarks/memgallery_harness/eval_memgallery.py \
  src/benchmarks/h2hmem_harness/eval_h2hmem.py \
  src/benchmarks/wma_harness/eval_wma.py
```

M2A 分支应明确命中这三个 benchmark-native builder；若改为固定 chunk loader、直接写 Semantic Store，或 builder 产物缺少可审计的 `m2a_turns`/provenance，则停止并报告。当前 matrix runner 正式运行所需参数包括：可重复 `--endpoint`、`--baseline`、`--benchmark`，以及 `--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id` 和 `--skip-smoke`。

确认 API defaults 的关键值：

```bash
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python - <<'PY'
import json
from pathlib import Path

d = json.loads(Path("/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json").read_text())
assert d["answer_model"] == d["executor_model"] == "openai/gpt-5-mini"
assert d["answer_base_url"] == d["executor_base_url"] == "https://openrouter.ai/api/v1"
assert d["reasoning_effort"] == "minimal"
assert d["answer_temperature"] == d["executor_temperature"] == 0
assert d["num_predict"] == d["executor_max_tokens"] == 512
assert d["m2a_executor_max_tokens"] == 2048
assert d["m2a_wma_rounds_per_ingest"] == 4
assert d["request_timeout"] == 180 and d["retries"] == 2 and d["top_k"] == 7
assert d["embedding_model"] == "Qwen/Qwen3-VL-Embedding-2B"
assert d["embedding_dim"] == 2048
assert d["judge_model"] == "openai/gpt-4o-mini"
print("API defaults OK")
PY
```

检查本地 embedding：

```bash
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["M2A API preflight"]}' \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048'
```


## 6. 三个 API worker 正式启动

三次相同 endpoint 表示三个独立 matrix worker。它们可并行请求同一 OpenRouter endpoint，但每个 benchmark 的进程、M2A state、checkpoint 和结果目录必须独立。

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID="m2a_gpt5mini_api_$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s "m2a_api_$RUN_ID" \
  "cd /data/haozhen/Memory-clean/Offline && \
   set +x && \
   export OPENAI_API_KEY=\$(tr -d '\\r\\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api) && \
   M2A_PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json \
   --efficiency-config /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
   --split-manifest /data/haozhen/Memory-clean/Offline/configs/multimodal_split_manifest.json \
   --output-root /data/haozhen/Memory-clean/Offline/outputs \
   --baseline M2A \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --skip-smoke \
   --run-id '$RUN_ID' \
   > 'outputs/_runs/$RUN_ID/launcher.log' 2>&1"

echo "$RUN_ID"
```

不要使用 `echo $OPENAI_API_KEY`、`set -x`、把 key 放入命令参数或将含 key 的环境 dump 到日志。命令显式使用 `--skip-smoke`，直接进入正式任务。

## 7. 监控与断点恢复

```bash
tmux ls | rg 'm2a_api'
tmux attach -t "m2a_api_<Run-ID>"
cat "outputs/_runs/<Run-ID>/status.json"

tail -n 80 "outputs/_runs/<Run-ID>/_logs/baseline/m2a__mem_gallery.log"
tail -n 80 "outputs/_runs/<Run-ID>/_logs/baseline/m2a__h2hmem.log"
tail -n 80 "outputs/_runs/<Run-ID>/_logs/baseline/m2a__worldmemarena.log"
tail -n 80 "outputs/_runs/<Run-ID>/_logs/judge/m2a__mem_gallery.log"
tail -n 80 "outputs/_runs/<Run-ID>/_logs/judge/m2a__h2hmem.log"
tail -n 80 "outputs/_runs/<Run-ID>/_logs/judge/m2a__worldmemarena.log"
```

检查三个 job 在 `status.json` 中有不同 child PID、result_dir 和 state。API 并发出现持续 429或 latency 明显恶化时停止并报告并发数、错误率、累计调用和已产生费用；不得静默改单 worker、换模型或跳过失败样本。

断点恢复时使用相同 Run-ID、defaults、efficiency config、split manifest、原始输入、builder、endpoint 数量和 embedding。Runner 复用签名匹配的 sample checkpoint；未完成 sample重建自己的 M2A state。若输入、builder 或配置 hash 改变，使用新 Run-ID。

## 8. Outputs、指标与验收

```text
/data/haozhen/Memory-clean/Offline/outputs/Mem-Gallery/M2A/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/H2HMEM/M2A/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/WorldMemArena/M2A/<Run-ID>/
```

每个目录的当前产物结构包括：

- `results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`
- `memory/memory_snapshot.jsonl`
- `memory/datasets/<sample>/raw.db`、`semantic.db`、`raw.json`、`semantic.json`、`image_manager.json`
- `memory/datasets/<sample>/m2a_execution_trace.jsonl`、`m2a_conformance.json`
- `run_manifest.json`
- `metrics.json`、`efficiency_metrics.json`、`call_metrics.json`
- `call_trace.jsonl`、`call_traces/<sample>.jsonl`、`.checkpoint/`
- `llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json`、`llm_judge_metrics.json`
- Mem-Gallery/WMA 的 `memory_metrics.json`
- H2HMEM 的 `prediction_dyadic.json`、`prediction_multi_party.json`

验收要求：

1. QA 数严格 275/360/440；question ID、顺序和 sample 与 test manifest 完全相同，无 Adversarial Query。
2. 原始输入文件 manifest/hash、builder 身份/版本、split manifest SHA、API model/provider、reasoning、prompt SHA、Top-7 和 M2A commit正确；execution trace 能证明使用了声明的 benchmark-native 输入转换。
3. 三个任务的进程、Raw/Semantic/Image state、checkpoint、snapshot、trace和输出完全独立。
4. 每个 sample 走 `ChatAgent -> MemoryManager -> SemanticStore`；每题走原版 agent query/search/fetch，内部候选可超过 7，最终显式 handoff 不超过 7。
5. WMA 先答后写，retrieval provenance 是当前 `visible_sessions` 子集。
6. 无 answer/tool/API error、重复题和未完成 Judge；Judge count 等于 QA count，Judge 完全排除效率统计。

指标字段固定为：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- F1、EM、Judge 为 test QA 的 qa-wise average。
- Cost-MB/Lat-MB 与 Cost-QA/Lat-QA 均为 sample-wise average。
- `#Calls (MB+QA)=(MB calls+answer calls)/sample 数`；retrieval calls另存 trace，Judge 不计入。
- Cost/latency 必须读取 `/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini`。

## 9. 必须停止并报告

split manifest 或原始输入 hash异常、builder/`m2a_turns` 与声明不一致、内部 prompt不符、最终显式 handoff 超过 7、Agent/Manager 被绕过、图片或 embedding失败、WMA未来泄漏、QA/Judge 数量不符、401/402/429/timeout/provider error、需要更换模型/provider/prompt/检索策略或复用旧结果时，必须停止并先报告证据与建议。

## 10. 给 Codex 的快捷指令

> 严格遵循本教程。M2A 使用 test manifest 选中的原始 benchmark 数据和三个专用 builder，不使用固定 chunk JSONL；核对 builder/原始输入 manifest、最终显式 handoff ≤7 和 WMA 时序。使用全新 Run-ID、三个独立 worker、同一 OpenRouter endpoint 及本地 embedding 8001，直接并行运行 275/360/440 QA；保护 API key 并完成专用验收、Judge、调用追踪和结果验收。
