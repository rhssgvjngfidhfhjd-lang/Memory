# OmniSimpleMem 三 Benchmark Test Split OpenRouter API 运行教程

## 1. 目标

通过 OpenRouter `openai/gpt-5-mini` 运行 OmniSimpleMem 的三个 test split：

- Mem-Gallery：严格 275 QA。
- H2HMEM：严格 360 QA，包含 dyadic 和 multiparty。
- WorldMemArena：只运行 lifelong，严格 440 checkpoint QA。

三个任务可以通过三个独立 worker 并行调用同一 API endpoint，但必须使用独立进程、memory state、checkpoint、trace 和输出目录。每次正式实验使用新的 Run-ID，从输入重新构建 memory，不复用其他 Run-ID 的 memory、snapshot 或 retrieval trace。

## 2. 固定配置

- Baseline CLI 名：`OmniSimpleMem`
- 协议：`configs/test_baseline_matrix.json`
- Split manifest：`configs/multimodal_split_manifest.json`
- 上游 commit：`836ce9718f3e9cb7f93c9d7c842b47f62e177a66`
- 上游 `HEAD:OmniSimpleMem` tree：`685109637c4c8b9a2469e695ad3dbed40762c0f2`
- Defaults：`/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json`
- Efficiency config：`/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini`
- API key 文件：`/data/haozhen/Memory-clean/Nvida_api/Openrouter_api`
- 建库与回答模型：OpenRouter `openai/gpt-5-mini`
- API endpoint：`https://openrouter.ai/api/v1`
- reasoning effort：`minimal`
- temperature：`0.0`
- QA max tokens：`512`
- timeout：`180` 秒
- retries：`2`
- `top_k=7`
- Embedding：本地 `Qwen/Qwen3-VL-Embedding-2B`
- Embedding endpoint：`http://127.0.0.1:8001/v1`
- Embedding 维度：`2048`
- LLM Judge：OpenRouter `openai/gpt-4o-mini`

API key 只能从 key 文件读入进程环境，不得写入文档、命令参数、日志或终端输出。

## 3. 固定输入

目标实验协议使用：

| Benchmark | 固定 chunk JSONL | 行数 | SHA256 |
|---|---|---:|---|
| Mem-Gallery | `data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl` | 3962 | `93e1d0b778addfbf4d8df8c387c588a1a5333f19864c41b6436e54e9dd0e189e` |
| H2HMEM dyadic | `data/h2hmem/chunks_dyadic.jsonl` | 2645 | `e046039299b40b0458a7b6e88f482c4081664a41aaf10f3861d2aff511e821b9` |
| H2HMEM multiparty | `data/h2hmem/chunks_multiparty.jsonl` | 866 | `75ebf39c87ce60ea98313270754c0fdb46bdf5458491f8138e69313999113e51` |
| WMA lifelong | `data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl` | 15437 | `75e89446f6d991b4a2e5b471ab754d2d7f8942615badc73930c4385d153fe748` |

Split manifest SHA256：`590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。

Harness 应按 test manifest 精确筛选 sample/question ID；禁止运行时重新切分、改写或静默重建 chunk。合规 `run_manifest.json` 应记录 chunk 的绝对路径、SHA256、格式和数量。

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

建库调用链：

```text
run_test_baseline_matrix.py
  -> benchmark harness
  -> create_adapter("OmniSimpleMem")
  -> OmniSimpleMemAdapter.reset()
  -> 官方 OmniMemoryOrchestrator
  -> start_session()
  -> add_text() / add_image()
  -> end_session()
```

Adapter 将官方 text/image embedding transport 接到本地 8001 服务，保持官方 summary、caption、entity、event、graph 和内部 prompt。当前兼容层使用官方 `add_text()` 与 `add_image()`，不使用存在 `MAULinks.related=null` 兼容问题的 `add_multimodal()`。

允许官方 processor 正常 skip，以及官方已有的 summary/caption/entity 错误处理；禁止 Adapter 使用 `force=True`、direct insert、其他 baseline、旧 memory/trace、零向量或替代模型进行静默 fallback。

检索调用链：

```text
OmniMemoryOrchestrator.query()
  -> QueryProcessor.determine_retrieval_strategy()（原版动态 5/10/20）
  -> PyramidRetriever / FAISS preview
  -> graph/entity 与 parametric 旁路结果
  -> RetrievalResult.items
```

Top-7 施加在 QueryProcessor 策略输出层：保留原版 query 解析和其他策略字段，将动态 `top_k` 固定为 7，再调用 `query(text, top_k=7, auto_expand=False, benchmark_safe=True)`。Adapter 保留原版顺序，WMA 先过滤不可见 session，最终交给 QA prompt 的 MAU 最多 7 条；不足 7 条不补齐。这不是 MIRIX 六层 memory、Chat Agent 或 automatic prefetch 的 Top-K。

当前 Adapter 记录原版 lexical modality hint，但关闭其对 `TEXT/VISUAL/AUDIO/VIDEO` 的后置硬过滤，以便统一 2048 维多模态 embedding 检索同一 MAU 池；上游使用子串匹配，可能把 `clipboard/context/sounded` 等普通措辞误判为模态约束。该 override 和原始/有效策略必须进入 trace。

回答调用链：

```text
最多 7 条 retrieval evidence
  -> 当前 benchmark harness 已有 QA prompt
  -> VLMAnswerClient.answer_messages_with_usage()
  -> OpenRouter openai/gpt-5-mini
```

不调用 `OmniMemoryOrchestrator.answer()`。Benchmark QA prompt 与 OmniSimpleMem 内部 summary/caption/entity prompt 均保持现有版本。

WMA 的目标协议是 checkpoint 增量建库：只写入当前可见 session，回答当前 QA 后才写入未来 session；检索结果必须受 `visible_session_ids` 限制。

## 5. 环境与服务检查

```bash
cd /data/haozhen/Memory-clean/Offline
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/run_test_baseline_matrix.py --help
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/prepare_omnisimplemem.py
git -C .upstream/simplemem-836ce97 rev-parse HEAD 'HEAD:OmniSimpleMem'
git -C .upstream/simplemem-836ce97 status --short
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json >/dev/null
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini >/dev/null
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api
```

安全加载 key；不要启用 `set -x` 或输出变量：

```bash
export OPENAI_API_KEY="$(tr -d '\r\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api)"
test -n "$OPENAI_API_KEY"
curl -fsS https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="openai/gpt-5-mini" for x in d.get("data", []))'
```

检查本地 embedding：

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


## 6. 三 worker tmux 启动

```bash
export RUN_ID="omnisimplemem_api_$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s "omni_api_${RUN_ID}" \
  "cd /data/haozhen/Memory-clean/Offline && \
   export OPENAI_API_KEY=\$(tr -d '\\r\\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api) && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json \
   --efficiency-config /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
   --baseline OmniSimpleMem \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --split-manifest configs/multimodal_split_manifest.json \
   --skip-smoke \
   --run-id $RUN_ID \
   > outputs/_runs/$RUN_ID/launcher.log 2>&1"
```

三个相同 endpoint 参数代表三个并行 worker，不代表三个 provider。三个 harness 以独立子进程运行，结果和 state 分别位于各 benchmark 目录。命令中的 `--skip-smoke` 使 runner 直接进入正式任务。

持续出现 401、402、429、timeout 或 provider error 时应停止并记录；不得静默换模型、换 provider、减样本或跳过失败题。

## 7. 监控与恢复

```bash
: "${RUN_ID:?必须恢复启动时使用的准确 RUN_ID}"
tmux attach -t "omni_api_${RUN_ID}"
tail -f "outputs/_runs/$RUN_ID/launcher.log"
cat "outputs/_runs/$RUN_ID/status.json"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/omnisimplemem__mem_gallery.log"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/omnisimplemem__h2hmem.log"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/omnisimplemem__worldmemarena.log"
```

中断后使用相同 Run-ID、命令、API 模型/provider、reasoning effort、manifest 和 chunk 文件重启。兼容 checkpoint 可以恢复已完成 QA；不得搬用其他 Run-ID 的 memory 或 retrieval trace。从头重跑必须使用新 Run-ID。

## 8. 输出与验收

```text
outputs/Mem-Gallery/OmniSimpleMem/<Run-ID>/
outputs/H2HMEM/OmniSimpleMem/<Run-ID>/
outputs/WorldMemArena/OmniSimpleMem/<Run-ID>/
```

核心产物包括：`results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_metrics.json`、`call_trace.jsonl`、`llm_judge_results.json` 和 `llm_judge_metrics.json`。三个任务不得复用或覆盖这些文件。

快速验收，不会输出 API key：

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
    assert manifest["answer_model"] == manifest["executor_model"] == "openai/gpt-5-mini"
    assert manifest["reasoning_effort"] == "minimal"
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

指标统一为：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- `F1`、`EM`、`Judge`：test QA 的 qa-wise average。
- `Cost-MB`、`Lat-MB`：建库阶段 USD/sample、seconds/sample。
- `Cost-QA`、`Lat-QA`：检索及回答阶段合计 USD/sample、seconds/sample。
- `#Calls (MB+QA)`：`(MB calls + answer calls) / sample 数`；retrieval/Judge calls 不计入。
- Cost/latency 使用 `/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini` 中 `openai/gpt-5-mini` 的参数。

## 9. 当前代码差异

当前命令可以执行，但截至 2026-09-14：

- 三个 OmniSimpleMem harness 仍通过 `baseline_runtime/omni_inputs.py` 构造 benchmark-native 输入，并记录 `shared_fixed_chunks=false`，没有读取第 3 节固定 chunk。
- WMA 在当前 checkpoint 完成检索并固化 evidence 后继续 ingest 后续 session，实际回答稍后发生；检索 evidence 不包含未来 session，但不满足“回答完成后再写未来 session”的严格顺序。

直接运行应标记为 diagnostic/non-compliant。若需要严格固定-chunk正式对比，应先修复上述两项并使用新 Run-ID；若用户明确接受偏离，可以保留当前结果，但不能把它描述为固定-chunk结果。`scripts/validate_omnisimplemem_reproduction.py` 仍采用旧输入和“恰好 7 条”口径，不应代替这里的“最多 7 条”验收。

## 10. 给 Codex 的快捷指令

> 严格遵循本教程，检查 OpenRouter 配置、API key 和本地 embedding，使用全新 Run-ID 与三个独立 worker，在 tmux 直接运行 OmniSimpleMem 的 Mem-Gallery 275 QA、H2HMEM 360 QA 和 WMA lifelong 440 QA。保护 API key，持续监控三个任务并完成结果验收。
