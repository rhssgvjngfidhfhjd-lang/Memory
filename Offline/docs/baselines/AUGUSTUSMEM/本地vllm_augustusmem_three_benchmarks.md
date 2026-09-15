# AUGUSTUSMEM 三 Benchmark Test Split 本地 vLLM 运行教程


## 1. 范围与名称

本教程运行当前工作区中的 AUGUSTUSMEM，完成：

- Mem-Gallery test：严格 275 QA；
- H2HMEM test：dyadic + multiparty，严格 360 QA；
- WorldMemArena：只运行 lifelong test，严格 440 checkpoint QA。

文档和目录使用 `AUGUSTUSMEM`。代码注册名、CLI 参数值和结果目录名实际是 **`AUGUSTUSMemory`**，命令必须使用这个真实名称，不能改成不存在的 `AUGUSTUSMEM` CLI 名称。

每次正式实验必须使用全新 Run-ID，从固定 chunk 构建三份彼此独立的 AUGUSTUSMemory 状态；禁止复用其他 Run-ID 的 memory、snapshot、prepared QA 或 retrieval trace。相同 Run-ID 的同签名中断恢复除外。

当前 `MemEngineAdapter.reset()` 会丢弃传入的 `state_dir`，AUGUSTUS 原生图仅存在于对应 baseline worker 的内存中；持久化产物是统一 snapshot 和 harness sample checkpoint。三个任务依靠独立子进程隔离内存对象。恢复时，已完成 sample 可复用同 Run-ID 的 prepared artifact；未完成 sample 必须重新从固定 chunk 构建，不能假称从原生数据库恢复。

## 2. 固定配置

- 工作目录：`/data/haozhen/Memory-clean/Offline`
- 协议：`configs/test_baseline_matrix.json`
- Split manifest：`configs/multimodal_split_manifest.json`
- Defaults：`configs/defaults.json`
- Efficiency：`configs/model_efficiency.json`
- 建库与回答模型：`Qwen/Qwen3-VL-4B-Instruct`
- Embedding：`Qwen/Qwen3-VL-Embedding-2B`
- Embedding endpoint：`http://127.0.0.1:8001/v1`
- Embedding 维度：2048
- 回答 endpoints：`8013`、`8014`、`8015`
- temperature：0
- QA max tokens：512
- AUGUSTUS concept-extraction max tokens：由 runner 的 `executor_max_tokens=512` 在 counting proxy 边界强制执行
- timeout：180 秒
- retries：2
- `top_k=7`
- Judge：OpenRouter `openai/gpt-4o-mini`
- Judge max tokens：512

`configs/baselines.json` 没有为 AUGUSTUSMemory 记录独立 upstream URL/commit；因此正式运行记录至少要同时记录工作区 Git commit、dirty diff hash，以及本教程运行前检查列出的 AUGUSTUS 源文件 hash。不能只写仓库 commit 后忽略未提交的 adapter 修改。

## 3. 固定输入

只有 `src/benchmarks/fixed_chunks.py` 读取的以下 JSONL 可以进入 AUGUSTUSMemory：

| 数据源 | 固定文件 | 当前行数 | 当前 SHA256 |
|---|---|---:|---|
| Mem-Gallery | `data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl` | 3962 | `93e1d0b778addfbf4d8df8c387c588a1a5333f19864c41b6436e54e9dd0e189e` |
| H2HMEM dyadic | `data/h2hmem/chunks_dyadic.jsonl` | 2645 | `e046039299b40b0458a7b6e88f482c4081664a41aaf10f3861d2aff511e821b9` |
| H2HMEM multiparty | `data/h2hmem/chunks_multiparty.jsonl` | 866 | `75ebf39c87ce60ea98313270754c0fdb46bdf5458491f8138e69313999113e51` |
| WMA lifelong | `data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl` | 15437 | `75e89446f6d991b4a2e5b471ab754d2d7f8942615badc73930c4385d153fe748` |

当前 test manifest SHA256 为 `590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。这些值是教程编写时的实测值；运行时必须重新计算并与表中比较。任何变化都要重新完成运行前检查并使用新 Run-ID，不能静默重建或改写 chunk。

Matrix runner 会从 manifest 建只读 staged dataset view，并按 question ID 顺序筛选：Mem-Gallery 4 个 conversation/275 QA，H2HMEM 5 个 conversation/360 QA，WMA lifelong 8 个 sample/440 QA。

## 4. AUGUSTUSMemory 原版与当前调用链

### 4.1 建库

当前 vendored AUGUSTUSMemory 的真实链是：

```text
固定 Chunk JSONL
→ MemEngineAdapter.ingest()
→ observation{text, image=首张图片路径, images, metadata, source_dialogue_ids}
→ AUGUSTUSMemory.store()
→ AUGUSTUSMemoryStore.__call__()
→ 原版 LLMConceptExtractor prompt 提取 concepts
→ Qwen3-VL-Embedding-2B 生成 text/image 联合 embedding
→ TagGraphStorage 写入一个 Context Node
→ 建 temporal_succession、semantic_similarity、concept_association 边
```

“Node”就是一条完整 observation 对应的 Contextual Memory 图节点，不是 token、QA 或 MIRIX memory bank。节点保留文本、图片、timestamp、dialogue/provenance 和 concepts；当前默认每个固定 chunk 写一个节点。建库阶段 Qwen3-VL-4B 参与 concept extraction，Embedding 服务参与节点向量和语义边构建。

原版内部 prompt 指 `DefaultAUGUSTUSMemoryConfig.py` 选择的 `LLMConceptExtractor._default_prompt()`；adapter 只替换 model/base URL/temperature，不得改该 prompt。

### 4.2 检索与 Top-7

真实 recall 链是：

```text
benchmark query（有 query image 时一并传入）
→ AUGUSTUSMemory.recall()
→ embedding 相似度取最多 3 个起点
→ LLM concepts + CoPe Stage 1 找 concept-tag 命中节点
→ 合并起点
→ CoPe Stage 2 以 max_depth=3 图遍历
→ embedding/concept 各 0.5 融合并按分数排序
→ recall.max_nodes 截断
→ MultiModalUtilization 返回逐节点 dict
→ adapter 保留逐节点 text/provenance/image_paths
```

原版 `DEFAULT_GRAPH_MAX_NODES=10`；当前实验适配器 `_apply_common_config()` 将 `recall.max_nodes` 改为全局 `top_k=7`。因此 **Top-7 施加在 CoPe 与 embedding 候选融合排序后的最终 Context Node 集合**。初始 embedding 起点仍最多 3 个；concept 命中和图遍历负责扩候选，最终最多返回 7 个节点。不得再套 MIRIX 的六库、Core memory、automatic prefetch 或 Chat Agent Top-K 规则。

### 4.3 回答与图片

当前 vendored AUGUSTUSMemory 包只实现 `store()/recall()`，没有原生 Chat Agent/answer 方法。当前回答链是：

```text
最多 7 个 native Context Node
→ RetrievedMemory（含 source_dialogue_ids/image_ids/image_paths）
→ 各 benchmark 当前 build_answer_messages() QA prompt
→ VLMAnswerClient
→ Qwen/Qwen3-VL-4B-Instruct
→ parse_answer_response()
```

视觉类 QA 会把召回节点的原图随 evidence 传给回答模型；问题自身图片另行附加。非视觉类别按 harness 的当前 image-category policy 不附召回图片。不得把节点列表整体 `str()` 聚合，也不得丢失逐节点 provenance 或图片。

当前 prompt SHA256：Mem-Gallery `ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff`；H2HMEM `4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f`；WMA `6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250`。运行时必须由当前 prompt 模块重新计算。

## 5. Fallback 规则

允许：

- timeout/网络瞬态错误后，使用相同模型、相同输入和相同参数重试，最多 2 次，并保留失败 attempt trace；
- 完全相同签名的同 Run-ID 断点恢复；
- memory 为空时返回空 retrieval，再由原 benchmark QA prompt 回答；这属于明确的空结果，不是替代检索算法。

禁止：

- concept LLM 失败后接受关键词抽取结果；
- embedding HTTP 400 后接受截断文本、text-only 或 blank embedding；
- 改用其他 embedding、LLM、provider、caption-only 或纯文本路径；
- 旧 memory/snapshot/retrieval trace、重新切 chunk、跳题、允许 answer error；
- 绕过 CoPe/图遍历，或把统一 graph-retrieval CLI 参数误当成 AUGUSTUS 的原生图检索；
- 丢失召回图片后只传问题图片。

## 6. 实际 runner 与 CLI

已实际执行 `--help` 核对：

- 外层入口：`scripts/run_test_baseline_matrix.py`
- 真实 baseline 值：`--baseline AUGUSTUSMemory`
- 外层使用的参数：重复的 `--endpoint`、`--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id`、重复的 `--baseline`、重复的 `--benchmark`、`--skip-smoke`
- Mem-Gallery：`python -m benchmarks.memgallery_harness.eval_memgallery`，runner 加 `--all-datasets --split-manifest ... --split test`
- H2HMEM：`python -m benchmarks.h2hmem_harness.eval_h2hmem`，默认 `--variant all`，即 dyadic + multiparty
- WMA：`python -m benchmarks.wma_harness.eval_wma`
- 三个 harness 都接收 runner 实际传入的 `--baseline-state-dir --result-dir --sample-concurrency --answer-concurrency --checkpoint-every --answer-* --executor-* --embedding-* --top-k --request-timeout --retries --efficiency-config --resume`；API 版还会收到 `--reasoning-effort minimal`


## 7. 运行前环境和签名检查

```bash
cd /data/haozhen/Memory-clean/Offline
PY=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
export MEMENGINE_PYTHON="$PY"

"$PY" scripts/run_test_baseline_matrix.py --help >/dev/null
PYTHONPATH=src "$PY" -c \
  'from benchmarks.baseline_runtime.registry import baseline_metadata; print(baseline_metadata("AUGUSTUSMemory"))'

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

wc -l \
  data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunks_multiparty.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl
```

确认 Judge key 文件存在但不要输出内容：

```bash
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api
```

## 8. 启动本地服务

先用 `nvidia-smi -i 3,4,5` 确认进程归属；不得终止其他任务。以下示例在 GPU3 同时放 embedding 和一个 4B worker，在 GPU4/5 各放一个 4B worker；若资源布局不同，只能调整 GPU/端口映射，不能改模型和请求配置。

```bash
cd /data/haozhen/Memory-clean/Offline
SERVICE_ID=augustus_local_services
mkdir -p "outputs/_runs/$SERVICE_ID"

tmux new-session -d -s augustus_embed_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=3 \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/serve_embeddings.py \
   --port 8001 --model Qwen/Qwen3-VL-Embedding-2B \
   --dim 2048 --device cuda:0 --dtype bfloat16 --local-files-only \
   > outputs/_runs/$SERVICE_ID/embedding_8001.log 2>&1"

tmux new-session -d -s augustus_vllm_8013 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=3 PORT=8013 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 \
   GPU_MEMORY_UTILIZATION=0.65 sh scripts/serve_vllm.sh \
   > outputs/_runs/$SERVICE_ID/vllm_8013.log 2>&1"

tmux new-session -d -s augustus_vllm_8014 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=4 PORT=8014 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 \
   GPU_MEMORY_UTILIZATION=0.65 sh scripts/serve_vllm.sh \
   > outputs/_runs/$SERVICE_ID/vllm_8014.log 2>&1"

tmux new-session -d -s augustus_vllm_8015 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=5 PORT=8015 MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 \
   GPU_MEMORY_UTILIZATION=0.65 sh scripts/serve_vllm.sh \
   > outputs/_runs/$SERVICE_ID/vllm_8015.log 2>&1"
```

等待服务完成加载，再验证模型和 embedding 维度：

```bash
for port in 8013 8014 8015; do
  curl -fsS "http://127.0.0.1:$port/v1/models" | \
    "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="Qwen/Qwen3-VL-4B-Instruct" for x in d.get("data", []))'
done

curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["preflight"]}' | \
  "$PY" -c 'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048'
```

## 9. 正式启动

以下命令显式使用 `--skip-smoke`，直接进入三个 benchmark 的正式任务：

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID="$(date +%m%d)_augustusmem_local_test_v1"
test ! -e "outputs/_runs/$RUN_ID"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s augustusmem_local_test \
  "cd /data/haozhen/Memory-clean/Offline && \
   export MEMENGINE_PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults configs/defaults.json \
   --efficiency-config configs/model_efficiency.json \
   --split-manifest configs/multimodal_split_manifest.json \
   --baseline AUGUSTUSMemory \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint http://127.0.0.1:8013/v1 \
   --endpoint http://127.0.0.1:8014/v1 \
   --endpoint http://127.0.0.1:8015/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --output-root /data/haozhen/Memory-clean/Offline/outputs \
   --skip-smoke \
   --run-id $RUN_ID \
   > outputs/_runs/$RUN_ID/launcher.log 2>&1"
```

三个 endpoint 创建三个 worker，三个 benchmark 由三个独立 harness 子进程领取；它们的 state、checkpoint 和结果目录互不共享。命令中的 `--skip-smoke` 使 runner 直接进入正式任务。

## 10. 监控与断点恢复

```bash
tmux attach -t augustusmem_local_test
tail -f "/data/haozhen/Memory-clean/Offline/outputs/_runs/$RUN_ID/launcher.log"
cat "/data/haozhen/Memory-clean/Offline/outputs/_runs/$RUN_ID/status.json"

find "/data/haozhen/Memory-clean/Offline/outputs/_runs/$RUN_ID/_logs" \
  -type f -name '*.log' -print
```

`status.json` 必须分别显示三个 job 的 PID、endpoint、result_dir 和状态。判断性能时同时观察三个 child 的已完成题数、API/vLLM 日志、GPU 利用率和单位时间调用量，不能只看主进程。

中断恢复：用**完全相同**的 Run-ID、命令、模型、endpoint、manifest、chunk 和配置重新执行；runner 会把 `--resume` 传给 harness，并拒绝不匹配的 checkpoint。要从头重跑必须换新 Run-ID，禁止删除或覆盖旧正式目录。

## 11. 输出与验收

结果目录：

```text
outputs/Mem-Gallery/AUGUSTUSMemory/<Run-ID>/
outputs/H2HMEM/AUGUSTUSMemory/<Run-ID>/
outputs/WorldMemArena/AUGUSTUSMemory/<Run-ID>/
```

每个目录至少必须包含：`results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_metrics.json`、`call_traces/*.jsonl`、`llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json`、`llm_judge_metrics.json`。根目录 `call_trace.jsonl` 是 Judge 调用追踪；baseline 的 MB/retrieval/QA 调用在 `call_traces/*.jsonl`。

Runner 自身会验证 QA/trace/pipeline 数量、question ID 顺序、prompt hash、Top-7、answer errors 和效率字段。还必须人工/脚本补验：

1. 数量严格为 275/360/440；H2 同时包含 dyadic/multiparty，WMA 只含 lifelong；
2. `selection_mode=strict_manifest`，manifest/chunk SHA256 与运行前记录一致；
3. 新 Run-ID 的 snapshot 由本次固定 chunk 重新构建；
4. 每个 retrieval item 有有效 provenance，视觉类召回节点的 `image_paths` 在 answer request 中实际出现；
5. 每题 `top_k` 长度不超过 7，且 `retrieval_method_trace.via=native_recall`；
6. call trace 无 concept/embedding 降级、failed call、替代模型或缺失 usage；
7. WMA retrieved session 是 `visible_sessions` 子集，并且代码/trace 能证明真实回答发生在未来 session ingest 前；仅冻结 evidence 不算通过；
8. Judge count 分别等于 275/360/440 且 `judge_errors=0`。

统一汇总字段及顺序：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

`Cost-MB/Lat-MB` 是 memory build 的 USD/sample、seconds/sample；`Cost-QA/Lat-QA` 是 retrieval + answer；`#Calls` 为 `(MB calls + answer calls) / sample 数`，retrieval calls 单独留 trace，Judge 不计入上述效率指标。

## 12. 必须停止并报告

遇到以下情况不得自行折中：输入/hash/prompt 变化；不是新 Run-ID；服务模型或维度不符；concept/embedding 降级；Top-7 超限；图片/provenance 丢失；WMA 时序或 future-session 检查失败；QA、question ID、Judge 数不符；answer error、持续 timeout/429、token 截断或缺失 usage。

## 13. 给 Codex 的快捷指令

> 严格遵循本教程，核对固定 chunk、manifest、AUGUSTUS 源、dirty diff、三个 prompt hash 和本地服务，以新 Run-ID 在 tmux 中运行 AUGUSTUSMemory 的 Mem-Gallery 275、H2HMEM 360 和 WMA lifelong 440 QA。持续监控三进程性能，并完成 CoPe 最终节点 Top-7、召回图片、fallback 和 WMA 时序验收。
