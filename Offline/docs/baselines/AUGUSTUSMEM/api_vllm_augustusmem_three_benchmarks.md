# AUGUSTUSMEM 三 Benchmark Test Split OpenRouter API 运行教程

## 1. 范围与名称

通过 OpenRouter `openai/gpt-5-mini` 同时承担 AUGUSTUSMemory 的 concept-extraction 建库调用和 benchmark 最终回答，完成：

- Mem-Gallery test：严格 275 QA；
- H2HMEM test：dyadic + multiparty，严格 360 QA；
- WorldMemArena：只运行 lifelong test，严格 440 checkpoint QA。

文档/目录名使用 `AUGUSTUSMEM`，但当前代码注册名、CLI 值和输出目录名是 **`AUGUSTUSMemory`**。所有命令必须传 `--baseline AUGUSTUSMemory`。

三个 benchmark 可并行调用同一个 API endpoint，但必须由三个独立 harness 子进程执行，并使用独立的 memory state、sample checkpoint、result directory 和调用追踪。每次正式实验使用全新 Run-ID，禁止复用旧 Run-ID 的 memory、snapshot、prepared QA 或 retrieval trace；相同签名的同 Run-ID 中断恢复除外。

当前 `MemEngineAdapter.reset()` 会忽略 `state_dir`，AUGUSTUS 的 TagGraphStorage 只活在各 baseline worker 进程内；真正持久化的是统一 memory snapshot 和 harness sample checkpoint。三任务的内存对象由独立子进程隔离。恢复时可复用同 Run-ID 已完成 sample 的 prepared artifact，未完成 sample 要从固定 chunk 重建，不能宣称恢复了一个不存在的原生持久数据库。

## 2. 固定配置

- 工作目录：`/data/haozhen/Memory-clean/Offline`
- 协议：`configs/test_baseline_matrix.json`
- Split manifest：`configs/multimodal_split_manifest.json`
- API defaults：`/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json`
- API efficiency config：`/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini`
- API key 文件：`/data/haozhen/Memory-clean/Nvida_api/Openrouter_api`
- API base URL：`https://openrouter.ai/api/v1`
- 建库与回答模型：`openai/gpt-5-mini`
- reasoning effort：`minimal`
- temperature：0
- QA max tokens：512
- concept-extraction max tokens：512，由 counting proxy 强制写入请求
- timeout：180 秒
- retries：2
- `top_k=7`
- Embedding：本地 `Qwen/Qwen3-VL-Embedding-2B`
- Embedding endpoint：`http://127.0.0.1:8001/v1`
- Embedding 维度：2048
- Judge：OpenRouter `openai/gpt-4o-mini`
- Judge max tokens：512

当前 efficiency config 中 `openai/gpt-5-mini` 参数为：input 0.25 USD/million tokens、output 2.00 USD/million tokens、base latency 3.47 s、input 0.000003 s/token、output 0.01754386 s/token、image 0.30 s/image。运行时必须从指定文件读取，不得手填另一套价格或延迟。

`configs/baselines.json` 没有记录 AUGUSTUSMemory 的独立 upstream URL/commit。每个 正式运行签名必须记录工作区 Git commit、relevant dirty diff hash 和 AUGUSTUS 源文件 hash，避免未提交 adapter 修改被 commit 字段掩盖。

## 3. 固定输入与 test split

三个 harness 对 AUGUSTUSMemory 只能经 `src/benchmarks/fixed_chunks.py` 读取：

| 数据源 | 固定文件 | 当前行数 | 当前 SHA256 |
|---|---|---:|---|
| Mem-Gallery | `data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl` | 3962 | `93e1d0b778addfbf4d8df8c387c588a1a5333f19864c41b6436e54e9dd0e189e` |
| H2HMEM dyadic | `data/h2hmem/chunks_dyadic.jsonl` | 2645 | `e046039299b40b0458a7b6e88f482c4081664a41aaf10f3861d2aff511e821b9` |
| H2HMEM multiparty | `data/h2hmem/chunks_multiparty.jsonl` | 866 | `75ebf39c87ce60ea98313270754c0fdb46bdf5458491f8138e69313999113e51` |
| WMA lifelong | `data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl` | 15437 | `75e89446f6d991b4a2e5b471ab754d2d7f8942615badc73930c4385d153fe748` |

当前 manifest SHA256：`590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。运行时重新计算；有任何变化就重新完成运行前检查并使用新 Run-ID。禁止重新切分、现场生成、改写或静默重建 chunk。

Manifest 必须精确得到 Mem-Gallery 4 conversations/275 QA、H2HMEM 5 conversations/360 QA、WMA lifelong 8 samples/440 QA，并保持 manifest question ID 顺序。

## 4. AUGUSTUSMemory 特有流程

### 4.1 建库

```text
固定 Chunk JSONL
→ MemEngineAdapter.ingest()
→ observation{text, image=首张原图路径, images, metadata, source_dialogue_ids}
→ AUGUSTUSMemory.store()
→ AUGUSTUSMemoryStore.__call__()
→ openai/gpt-5-mini 使用原版 LLMConceptExtractor prompt 生成 concepts
→ 本地 Qwen3-VL-Embedding-2B 生成 text/image 联合 embedding
→ TagGraphStorage 写入一个 Context Node
→ temporal_succession + semantic_similarity + concept_association 边
```

Context Node 是一个完整 observation 对应的图节点，不是 MIRIX 的 memory bank。节点保存 text、image(s)、timestamp、dialogue/provenance 和 concepts。当前固定输入是每 chunk 一个节点。GPT-5-mini 只替换原 config 中 concept extractor 的模型/base URL/temperature/reasoning 请求设置；`LLMConceptExtractor._default_prompt()` 必须保持原样。

### 4.2 检索与 Top-7

```text
query（可含 query image）
→ AUGUSTUSMemory.recall()
→ multimodal embedding 最多取 3 个起点
→ GPT-5-mini 提取 query concepts
→ CoPe Stage 1 查 concept-tag 命中节点
→ 合并起点并执行 max_depth=3 的 CoPe Stage 2 图遍历
→ embedding/concept 0.5/0.5 融合排序
→ recall.max_nodes 截断
→ MultiModalUtilization 返回逐节点 dict
→ adapter 保留逐节点 text/provenance/image_paths
```

原版 `DEFAULT_GRAPH_MAX_NODES=10`，当前 adapter 将 `recall.max_nodes` 绑定到实验 `top_k=7`。所以 **Top-7 只定义为融合排序后最终返回给回答链的 Context Node 上限**；embedding 起点仍是最多 3 个，concept/遍历候选可先多于 7，再在最终节点层截断。不能照搬 MIRIX 的六层 memory、Core、automatic prefetch 或 Chat Agent Top-K。

### 4.3 回答与图片

vendored AUGUSTUSMemory 只定义 `store()/recall()`，没有原生 Chat Agent/answer 实现。当前正式回答链是：

```text
最多 7 个 Context Node
→ RetrievedMemory{text, source_dialogue_ids, image_ids, image_paths}
→ benchmark 当前 build_answer_messages() QA prompt
→ VLMAnswerClient
→ OpenRouter openai/gpt-5-mini（reasoning=minimal）
→ benchmark parse_answer_response()
```

视觉 QA 会附上召回节点原图；问题自己的 query image 独立附加。非视觉类别是否附召回图由各 harness 当前 category policy 决定。禁止把 list 整体字符串化、合并掉节点 provenance 或只保留 query image。

当前 prompt SHA256：

- Mem-Gallery：`ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff`
- H2HMEM：`4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f`
- WMA：`6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250`

## 5. Fallback 规则

允许：

- 相同 OpenRouter 模型/provider、输入和参数的瞬态失败重试，最多 2 次；每次 attempt 必须留 trace；
- 同一 Run-ID、完全相同签名的 checkpoint 恢复；
- 原生 recall 明确返回空列表后，仍用原 benchmark QA prompt 回答。

禁止：

- concept LLM 失败后接受 keyword extraction；
- embedding HTTP 400 后接受截断、text-only 或 blank embedding；
- 自动换 provider、model、reasoning effort、caption-only 或本地回答模型；
- 429/timeout 后跳题、写空 answer、允许 answer errors 或静默降并发后继续登记同一正式配置；
- 旧 memory/snapshot/retrieval trace、重切 chunk、统一 graph 参数替代 CoPe；
- 召回图片或 provenance 丢失。

## 6. 实际核对过的 runner 与 CLI

已用 `pipeline_repro` Python 实际执行四个 `--help`：

- 外层：`scripts/run_test_baseline_matrix.py`
- 真实 baseline：`--baseline AUGUSTUSMemory`
- 外层参数：重复 `--endpoint`、`--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id`、重复 `--baseline`、重复 `--benchmark`、`--skip-smoke`
- Mem-Gallery：`python -m benchmarks.memgallery_harness.eval_memgallery`，外层增加 `--all-datasets`
- H2HMEM：`python -m benchmarks.h2hmem_harness.eval_h2hmem`，默认 `--variant all`
- WMA：`python -m benchmarks.wma_harness.eval_wma`
- 外层实际向 harness 传 `--baseline-state-dir --result-dir --sample-concurrency --answer-concurrency --checkpoint-every --answer-* --executor-* --embedding-* --top-k --request-timeout --retries --reasoning-effort --efficiency-config --resume --split-manifest --split test`

`CountingProxy` 把 concept-extraction 请求的 max tokens 限为 512、temperature 设为 0，并加入 `reasoning: {"effort":"minimal"}`。QA client 同样使用 max 512、temperature 0、reasoning minimal。


## 7. 环境、输入和密钥预检

```bash
cd /data/haozhen/Memory-clean/Offline
PY=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
export MEMENGINE_PYTHON="$PY"

"$PY" scripts/run_test_baseline_matrix.py --help >/dev/null
"$PY" -m json.tool /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json >/dev/null
"$PY" -m json.tool /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini >/dev/null
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api

git rev-parse HEAD
git diff --binary -- \
  baselines/AUGUSTUSMemory \
  src/benchmarks/baseline_runtime/adapters/memengine.py \
  src/benchmarks/wma_harness/eval_wma.py | sha256sum

sha256sum \
  configs/multimodal_split_manifest.json \
  data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunks_multiparty.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl
```

安全加载 key；严禁 `set -x`、`echo`、打印环境、把 key 写进教程/日志/命令行参数：

```bash
set +x
IFS= read -r OPENAI_API_KEY < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api
export OPENAI_API_KEY
test "${#OPENAI_API_KEY}" -gt 20
```

检查 OpenRouter 模型但不输出 Authorization header 或响应内容：

```bash
curl -fsS https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | \
  "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="openai/gpt-5-mini" for x in d.get("data", []))'
```

## 8. 启动/检查本地 Embedding

API 版不启动本地回答 vLLM，但 embedding 仍必须在 8001 提供 2048 维向量。先检查现有服务；若没有，在确认 GPU3 空闲后启动：

```bash
curl -fsS http://127.0.0.1:8001/v1/models >/dev/null || true
nvidia-smi -i 3

SERVICE_ID=augustus_api_embed
mkdir -p "outputs/_runs/$SERVICE_ID"
tmux new-session -d -s augustus_api_embed_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=3 \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/serve_embeddings.py \
   --port 8001 --model Qwen/Qwen3-VL-Embedding-2B \
   --dim 2048 --device cuda:0 --dtype bfloat16 --local-files-only \
   > outputs/_runs/$SERVICE_ID/embedding_8001.log 2>&1"
```

服务启动后做真实维度检查：

```bash
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["preflight"]}' | \
  "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048'
```

## 9. 三个 benchmark 并行正式启动

以下命令显式使用 `--skip-smoke`，直接进入三个 benchmark 的正式任务：

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID="$(date +%m%d)_augustusmem_gpt5mini_api_test_v1"
test ! -e "outputs/_runs/$RUN_ID"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s augustusmem_gpt5mini_api_test \
  "cd /data/haozhen/Memory-clean/Offline && \
   set +x && \
   IFS= read -r OPENAI_API_KEY < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api && \
   export OPENAI_API_KEY && \
   export MEMENGINE_PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json \
   --efficiency-config /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
   --split-manifest configs/multimodal_split_manifest.json \
   --baseline AUGUSTUSMemory \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --output-root /data/haozhen/Memory-clean/Offline/outputs \
   --skip-smoke \
   --run-id $RUN_ID \
   > outputs/_runs/$RUN_ID/launcher.log 2>&1"
```

三个相同 `--endpoint` 会建立三个 matrix worker，并各自启动一个 benchmark harness 子进程；不是三个 API 账号。正式目录分别是 `outputs/<Benchmark>/AUGUSTUSMemory/<Run-ID>/`，每个 harness 的 `--baseline-state-dir` 是自己目录下的 `memory/datasets`，checkpoint 也各自在自己的 `.checkpoint`。命令中的 `--skip-smoke` 使 runner 直接进入正式任务。

## 10. 监控、降速判断与恢复

```bash
tmux attach -t augustusmem_gpt5mini_api_test
tail -f "/data/haozhen/Memory-clean/Offline/outputs/_runs/$RUN_ID/launcher.log"
cat "/data/haozhen/Memory-clean/Offline/outputs/_runs/$RUN_ID/status.json"
find "/data/haozhen/Memory-clean/Offline/outputs/_runs/$RUN_ID/_logs" \
  -type f -name '*.log' -print
```

持续核对三个 job 各有独立 child PID/result_dir，比较每 5–10 分钟完成题数、成功/失败调用、实际 duration 和 429/timeout 比例。API 并行明显变慢或持续 401/402/429/provider error 时，先停止并汇报已发生的调用、token、USD 成本与错误率；不得静默换模型/provider、跳题或把降低并发后的任务继续登记为同一固定配置。

断点恢复必须复用完全相同的 Run-ID、命令、模型/provider、reasoning、manifest、chunk 和配置。runner 会传 `--resume`；签名不一致必须拒绝。全新重跑只能用新 Run-ID，不删除、不覆盖旧目录。

## 11. 输出与验收

```text
/data/haozhen/Memory-clean/Offline/outputs/Mem-Gallery/AUGUSTUSMemory/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/H2HMEM/AUGUSTUSMemory/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/WorldMemArena/AUGUSTUSMemory/<Run-ID>/
```

每个目录至少应有：`results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_metrics.json`、`call_traces/*.jsonl`、`llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json`、`llm_judge_metrics.json`。根 `call_trace.jsonl` 是 Judge trace，baseline 的 MB/retrieval/QA trace 位于 `call_traces/`。

必须验收：

1. QA 数和 Judge count 分别为 275/360/440，`judge_errors=0`；
2. question IDs/order、sample set、split、manifest SHA、prompt SHA 完全一致；
3. H2 有 dyadic + multiparty，WMA 只有 lifelong；
4. 三个目录的 memory、checkpoint、snapshot 和 call trace 完全独立，均由本 Run-ID 固定 chunk 重建；
5. `answer_model=executor_model=openai/gpt-5-mini`、reasoning minimal、temperature 0、QA/executor max tokens 512、Top-7、embedding 模型/维度正确；
6. 每题最终 `top_k` 最多 7 个 Context Node，`retrieval_method_trace.via=native_recall`；
7. 每个节点 provenance 有效，视觉类召回图确实进入 answer request；
8. 无 keyword/text-only/blank/truncation fallback，无替代模型、answer error、未完成调用或缺失 token usage；
9. WMA retrieved sessions 属于 `visible_sessions`，并能从实现和 trace 证明 answer 完成早于未来 session ingest；冻结 evidence 本身不够；
10. efficiency 指标使用指定 GPT-5-mini config，Judge 调用没有混入 MB/QA 成本。

统一汇总：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

其中 Cost 为 USD/sample，Latency 为 seconds/sample；`#Calls=(MB calls + answer calls)/sample 数`。Retrieval calls 单独保留，Judge calls/cost/latency 不计入该表。人民币换算如另有需要，必须记录采用的汇率和日期，不能改写原始 USD 指标。

## 12. 必须停止并报告

出现以下任一情况立即停止：API key/模型/provider/reasoning 不符；输入、manifest、prompt 或配置 hash 变化；不是新 Run-ID；concept/embedding 自动降级；Top-7 超限；图片/provenance 丢失；WMA 时序或 future-session 检查失败；QA/ID/Judge 不一致；持续 401/402/429/timeout/provider error；token 用满导致 concept 解析失败；缺失 usage 或效率配置不匹配。

## 13. 给 Codex 的快捷指令

> 严格遵循本教程，安全加载指定 key，使用指定 defaults/efficiency config、本地 8001 embedding 和 OpenRouter `openai/gpt-5-mini`（minimal reasoning），以新 Run-ID 和三个独立 worker/process/state/checkpoint/output 并行运行 AUGUSTUSMemory 的 Mem-Gallery 275、H2HMEM 360、WMA lifelong 440 QA。持续监测吞吐、错误和费用，并完成 CoPe 最终 Context Node Top-7、图片、fallback、WMA 时序及统一指标验收。
