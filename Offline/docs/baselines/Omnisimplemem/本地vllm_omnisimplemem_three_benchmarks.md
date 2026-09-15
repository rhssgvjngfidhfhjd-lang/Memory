# OmniSimpleMem 三 Benchmark Test Split 本地 vLLM 运行教程

## 1. 目标

使用 OmniSimpleMem 官方 core 和当前 Offline Adapter，运行 test manifest 中的三个 benchmark：

- Mem-Gallery：严格 275 QA。
- H2HMEM：严格 360 QA，包含 dyadic 和 multiparty。
- WorldMemArena：只运行 lifelong，严格 440 checkpoint QA。

每次正式实验使用新的 Run-ID，为三个任务分别重新构建 OmniSimpleMem memory，不得复用其他 Run-ID 的 memory、snapshot 或 retrieval trace。

## 2. 固定配置

- 协议：`configs/test_baseline_matrix.json`
- Split manifest：`configs/multimodal_split_manifest.json`
- Baseline CLI 名：`OmniSimpleMem`
- 上游 commit：`836ce9718f3e9cb7f93c9d7c842b47f62e177a66`
- 上游 `HEAD:OmniSimpleMem` tree：`685109637c4c8b9a2469e695ad3dbed40762c0f2`
- `top_k=7`
- 建库与回答模型：`Qwen/Qwen3-VL-4B-Instruct`
- temperature：`0.0`
- QA max tokens：`512`
- timeout：`180` 秒
- retries：`2`
- Embedding：`Qwen/Qwen3-VL-Embedding-2B`
- Embedding endpoint：`http://127.0.0.1:8001/v1`
- Embedding 维度：`2048`
- 回答端口：`8013`、`8014`、`8015`
- 可用 GPU：GPU3、GPU4、GPU5
- LLM Judge：OpenRouter `openai/gpt-4o-mini`
- Judge 调用不计入 MB/QA calls、cost 或 latency

其余配置读取 `configs/defaults.json`、`configs/baselines.json` 和 `configs/model_efficiency.json`，不得静默修改。

## 3. 固定输入

目标实验协议使用以下固定 chunk JSONL，并按 test manifest 精确筛选 sample 和 question ID：

| Benchmark | 固定 chunk JSONL | 行数 | SHA256 |
|---|---|---:|---|
| Mem-Gallery | `data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl` | 3962 | `93e1d0b778addfbf4d8df8c387c588a1a5333f19864c41b6436e54e9dd0e189e` |
| H2HMEM dyadic | `data/h2hmem/chunks_dyadic.jsonl` | 2645 | `e046039299b40b0458a7b6e88f482c4081664a41aaf10f3861d2aff511e821b9` |
| H2HMEM multiparty | `data/h2hmem/chunks_multiparty.jsonl` | 866 | `75ebf39c87ce60ea98313270754c0fdb46bdf5458491f8138e69313999113e51` |
| WMA lifelong | `data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl` | 15437 | `75e89446f6d991b4a2e5b471ab754d2d7f8942615badc73930c4385d153fe748` |

Split manifest SHA256：`590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。

禁止运行时重新切分、改写或静默重建 chunk。合规结果的 `run_manifest.json` 应记录 chunk 的绝对路径、SHA256、格式和数量。

```bash
cd /data/haozhen/Memory-clean/Offline
sha256sum \
  data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunks_multiparty.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl \
  configs/multimodal_split_manifest.json
```

## 4. OmniSimpleMem 流程和 Top-7

### 4.1 建库

```text
run_test_baseline_matrix.py
  -> benchmark harness
  -> create_adapter("OmniSimpleMem")
  -> OmniSimpleMemAdapter.reset()
  -> 官方 OmniMemoryOrchestrator
  -> start_session()
  -> Chunk.text 调用 add_text()
  -> Chunk.images 逐张调用 add_image()
  -> end_session()
```

Adapter 将官方 text/image embedding transport 接到本地 Qwen3-VL-Embedding-2B 服务。官方 text summary、image caption、entity extraction、event 和 knowledge graph 流程保持开启，内部 prompt 不改写。

当前兼容层使用官方 `add_text()` 和 `add_image()`，不调用存在 `MAULinks.related=null` 兼容问题的 `add_multimodal()`。允许官方 processor 返回 `skipped=True`，以及官方已有的 summary、caption、entity 错误处理；禁止 Adapter 使用 `force=True`、direct insert、旧 memory、其他 baseline、零向量或替代模型进行静默补救。

### 4.2 检索和 Top-7

```text
benchmark query
  -> OmniMemoryOrchestrator.query()
  -> QueryProcessor.determine_retrieval_strategy()
  -> PyramidRetriever / FAISS preview
  -> graph/entity 与 parametric 旁路结果
  -> RetrievalResult.items
```

原版 QueryProcessor 会动态选择 5、10 或 20。当前实验在 QueryProcessor 策略输出层将 `top_k` 固定为 7，再调用：

```python
OmniMemoryOrchestrator.query(
    text,
    top_k=7,
    auto_expand=False,
    benchmark_safe=True,
)
```

Adapter 保留原版排序，WMA 先过滤不可见 session，最后向 benchmark QA prompt 提供最多 7 条 MAU。不足 7 条不补齐。这里的 Top-7 属于 OmniSimpleMem 单一检索管线，不采用 MIRIX 的六层 memory、Chat Agent 或 automatic prefetch 规则。

Adapter 会记录原版 lexical modality hint，但关闭其对 `TEXT/VISUAL/AUDIO/VIDEO` 的后置硬过滤，使当前统一 2048 维多模态 embedding 可以在同一 MAU 池检索；这是因为上游使用子串匹配，可能把 `clipboard/context/sounded` 等普通措辞误判为模态约束。原始动态 top-k、原始/有效 modality filter 和最终返回数必须写入 trace。

### 4.3 回答和 WMA

```text
最多 7 条 retrieval evidence
  -> 当前 benchmark harness 已有 QA prompt
  -> VLMAnswerClient.answer_messages_with_usage()
  -> Qwen/Qwen3-VL-4B-Instruct
```

不调用 `OmniMemoryOrchestrator.answer()`，因为它会使用上游自己的回答 prompt。Benchmark QA prompt 和 OmniSimpleMem 内部 summary/caption/entity prompt 均保持现有版本。

WMA 的目标协议是按 checkpoint 增量建库：只写入当前可见 session，回答当前 QA 后才能继续写入未来 session；检索结果必须受 `visible_session_ids` 限制。

## 5. 环境与服务检查

```bash
cd /data/haozhen/Memory-clean/Offline
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/run_test_baseline_matrix.py --help
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/prepare_omnisimplemem.py
git -C .upstream/simplemem-836ce97 rev-parse HEAD 'HEAD:OmniSimpleMem'
git -C .upstream/simplemem-836ce97 status --short
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api
nvidia-smi -i 3,4,5
```

检查 embedding 维度：

```bash
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["dimension check"]}' \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048; print("embedding_dim=2048")'
```

若 8001 未启动，只在确认模型已在本机 cache 且有空闲 GPU 后启动：

```bash
EMBEDDING_GPU=<空闲GPU编号>
mkdir -p outputs/_runs
tmux new-session -d -s omnisimplemem_embedding_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=$EMBEDDING_GPU \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/serve_embeddings.py \
   --host 127.0.0.1 --port 8001 \
   --model Qwen/Qwen3-VL-Embedding-2B \
   --dim 2048 --device cuda:0 --dtype bfloat16 --local-files-only \
   > outputs/_runs/embedding_8001.log 2>&1"
```

## 6. 启动本地 vLLM

设置一次全新 Run-ID，并在后续命令中保持不变：

```bash
export RUN_ID="omnisimplemem_local_$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/_runs/$RUN_ID"

for spec in 3:8013 4:8014 5:8015; do
  gpu=${spec%%:*}; port=${spec##*:}
  tmux new-session -d -s "omni_vllm_${port}_${RUN_ID}" \
    "cd /data/haozhen/Memory-clean/Offline && \
     PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
     GPUS=${gpu} PORT=${port} \
     MODEL=/data/shared_models/Qwen3-VL-4B-Instruct \
     SERVED_NAME=Qwen/Qwen3-VL-4B-Instruct \
     MAX_MODEL_LEN=131072 MAX_NUM_SEQS=16 \
     GPU_MEMORY_UTILIZATION=0.75 TOOL_CALL_PARSER=hermes \
     sh scripts/serve_vllm.sh \
     > outputs/_runs/$RUN_ID/vllm_${port}.log 2>&1"
done
```

如果相同配置的服务已经运行，可跳过启动，但必须确认三个端口返回正确模型：

```bash
for port in 8013 8014 8015; do
  curl -fsS "http://127.0.0.1:${port}/v1/models"
done
```


## 7. tmux 启动三个 Benchmark

```bash
: "${RUN_ID:?先执行第 6 节设置 RUN_ID}"

tmux new-session -d -s "omni_local_${RUN_ID}" \
  "cd /data/haozhen/Memory-clean/Offline && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults configs/defaults.json \
   --efficiency-config configs/model_efficiency.json \
   --baseline OmniSimpleMem \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint http://127.0.0.1:8013/v1 \
   --endpoint http://127.0.0.1:8014/v1 \
   --endpoint http://127.0.0.1:8015/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --split-manifest configs/multimodal_split_manifest.json \
   --skip-smoke \
   --run-id $RUN_ID \
   > outputs/_runs/$RUN_ID/launcher.log 2>&1"
```

三个 endpoint 对应三个 worker。命令中的 `--skip-smoke` 使 runner 直接进入正式任务。

## 8. 监控与恢复

```bash
: "${RUN_ID:?必须恢复启动时使用的准确 RUN_ID}"
tmux attach -t "omni_local_${RUN_ID}"
tail -f "outputs/_runs/$RUN_ID/launcher.log"
cat "outputs/_runs/$RUN_ID/status.json"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/omnisimplemem__mem_gallery.log"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/omnisimplemem__h2hmem.log"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/omnisimplemem__worldmemarena.log"
nvidia-smi -i 3,4,5
```

中断后使用完全相同的 Run-ID、参数、endpoint、manifest 和 chunk 文件重新执行第 7 节命令。兼容 checkpoint 可以恢复已完成 QA；不得把其他 Run-ID 的 memory、snapshot 或 retrieval trace 搬入本次任务。从头重跑必须使用新 Run-ID。

## 9. 输出与验收

```text
outputs/Mem-Gallery/OmniSimpleMem/<Run-ID>/
outputs/H2HMEM/OmniSimpleMem/<Run-ID>/
outputs/WorldMemArena/OmniSimpleMem/<Run-ID>/
```

核心产物包括：`results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_metrics.json`、`call_trace.jsonl`、`llm_judge_results.json` 和 `llm_judge_metrics.json`。

快速验收：

```bash
: "${RUN_ID:?必须设置待验收的 RUN_ID}"
RUN_ID="$RUN_ID" /data/haozhen/miniconda3/envs/pipeline_repro/bin/python - <<'PY'
import json, os
from pathlib import Path

root = Path("/data/haozhen/Memory-clean/Offline")
rid = os.environ["RUN_ID"]
expected = {"Mem-Gallery": 275, "H2HMEM": 360, "WorldMemArena": 440}
for benchmark, count in expected.items():
    out = root / "outputs" / benchmark / "OmniSimpleMem" / rid
    results = json.loads((out / "results.json").read_text())
    traces = [json.loads(x) for x in (out / "retrieval_trace.jsonl").read_text().splitlines() if x]
    manifest = json.loads((out / "run_manifest.json").read_text())
    judge = json.loads((out / "llm_judge_metrics.json").read_text())
    assert len(results) == len(traces) == count == judge["count"]
    assert manifest["top_k"] == 7 and manifest["split"] == "test"
    assert manifest["answer_model"] == manifest["executor_model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert [x["manifest_question_id"] for x in results] == manifest["ordered_question_ids"]
    assert all(not x.get("error") for x in results)
    assert all(len(x.get("top_k", [])) <= 7 for x in traces)
    if benchmark == "WorldMemArena":
        assert all(set(x.get("retrieved_sessions", [])) <= set(x.get("visible_sessions", [])) for x in results)
    chunk_rows = manifest.get("chunk_inputs") or [manifest.get("chunk_input") or {}]
    fixed = all(x.get("format") == "Chunk JSONL" and x.get("sha256") for x in chunk_rows)
    print(benchmark, "PASS", count, "fixed_chunks=", fixed)
PY
```

指标按以下固定顺序汇总：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- `F1`、`EM`、`Judge`：test QA 的 qa-wise average。
- `Cost-MB`、`Lat-MB`：建库阶段 USD/sample、seconds/sample。
- `Cost-QA`、`Lat-QA`：检索及回答阶段合计 USD/sample、seconds/sample。
- `#Calls (MB+QA)`：`(MB calls + answer calls) / sample 数`；retrieval 和 Judge calls 不计入。

## 10. 当前代码差异

当前命令可以运行，但截至 2026-09-14 有两项与目标协议不一致：

- 三个 OmniSimpleMem harness 仍通过 `baseline_runtime/omni_inputs.py` 从 benchmark 原始数据构造输入，并记录 `shared_fixed_chunks=false`，尚未接入第 3 节固定 chunk。
- WMA 会在当前 checkpoint 完成检索并固化 evidence 后继续 ingest 后续 session，实际回答调用稍后执行；检索 evidence 不含未来 session，但不满足“回答完成后再写未来 session”的严格执行顺序。

因此直接运行得到的是当前 OmniSimpleMem benchmark-native 结果。若用于严格固定-chunk正式对比，应先修复这两项并使用新 Run-ID 重跑；若用户明确接受偏离，可作为 diagnostic/non-compliant 结果保存。`scripts/validate_omnisimplemem_reproduction.py` 仍按旧输入和“恰好 7 条”口径验收，不应代替本教程的“最多 7 条”检查。

## 11. 给 Codex 的快捷指令

> 严格遵循本教程，检查本地服务，使用 GPU3/4/5、端口 8013/8014/8015、embedding 8001 和全新 Run-ID，在 tmux 直接运行 OmniSimpleMem 的 Mem-Gallery 275 QA、H2HMEM 360 QA 和 WMA lifelong 440 QA。持续监控三个任务并完成结果验收。
