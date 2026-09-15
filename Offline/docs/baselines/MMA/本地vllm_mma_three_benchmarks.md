# MMA 三 Benchmark Test Split 本地 vLLM 运行教程


## 1. 目标与固定配置

仅运行 test manifest 规定的：

- Mem-Gallery：严格 275 QA。
- H2HMEM：dyadic + multiparty，严格 360 QA。
- WorldMemArena：只运行 lifelong，严格 440 checkpoint QA。

固定配置：

- 协议：`configs/test_baseline_matrix.json`
- Split manifest：`configs/multimodal_split_manifest.json`
- Baseline CLI：`MMA`
- 官方上游：`https://github.com/AIGeeksGroup/MMA.git`
- 上游 commit：`c0e1a127722edcfa4db5e71d03708cba53363000`
- 上游 tree：`4bb8b33535cb9d14b39277d953bcc80b5aec2e4c`
- 建库与回答模型：`Qwen/Qwen3-VL-4B-Instruct`
- temperature：`0.0`
- QA max tokens：`512`
- MMA executor max tokens：当前 `configs/defaults.json` 为 `512`
- timeout：`180` 秒
- retries：`2`
- `top_k=7`
- Embedding：`Qwen/Qwen3-VL-Embedding-2B`，`http://127.0.0.1:8001/v1`，维度 2048
- 回答端口：`8013`、`8014`、`8015`
- MMA Python：默认 `.venvs/mirix/bin/python`
- vLLM tool parser：`hermes`，必须开启 native tool calls
- LLM Judge：OpenRouter `openai/gpt-4o-mini`
- Judge 不计入 MB/QA calls、cost 或 latency

其余配置读取 `configs/defaults.json`、`configs/baselines.json` 和 `configs/model_efficiency.json`，不得运行时静默改写。

## 2. MMA 原版 Agent、检索与回答流程

### 2.1 建库 Agent

原版 MMA 的普通行为是把 observation 放入 `TemporaryMessageAccumulator`，其 `TEMPORARY_MESSAGE_LIMIT=20`。本实验的统一输入协议要求每个固定 chunk 独立建库，因此 Adapter 使用原版入口：

```text
matrix runner
  -> benchmark harness
  -> create_adapter("MMA")
  -> MMAOriginalAdapter.reset()
  -> 官方 AgentWrapper + 隔离 SQLite
  -> 对每个固定 Chunk：
       AgentWrapper.send_message(
         message=Chunk.text,
         image_uris=Chunk.images,
         memorizing=True,
         force_absorb_content=True
       )
  -> TemporaryMessageAccumulator._build_memory_message()
  -> 原版 Meta Memory Agent
  -> trigger_memory_update
  -> 原版 Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault Agent
  -> 各原版 Manager 决定是否新增、更新或不写 memory
```

`force_absorb_content=True` 只把吸收边界从原版累计 20 条调整为实验要求的一固定 chunk 一 observation；它不能绕过 Meta Agent、Memory Agent 或 Manager。Agent 合理决定不写 memory 是合法结果，禁止补造 Semantic Memory、direct insert 或用其他 baseline 回退。

图片由 Adapter 从本地文件写入 MMA 自己的 image database，再以原版 image message 交给 Agent；禁止把图片变成路径字符串放进正文。每个 sample 使用 `<result-dir>/memory/datasets/<sample>/` 下的独立 SQLite 和图片目录。正式实验必须使用新 Run-ID 从零建库，不得复用旧 SQLite、snapshot 或 retrieval trace。

MMA 的 8 份内部 system prompt 必须保持官方字节不变。当前 Adapter 在 `mma_conformance.internal_prompt_sha256` 中记录 chat、meta、core、episodic、semantic、procedural、resource 和 knowledge-vault prompt 的 expected/actual SHA256。

### 2.2 Manager 检索和 Top-7

官方 `Agent.build_system_prompt_with_memories()` 原本会自动检索，单次 Manager 调用的 `MAX_RETRIEVAL_LIMIT_IN_SYSTEM=10`；其中 Episodic 会分别执行 recent 与 relevant 两次查询。为满足实验的最终全局 Top-7，当前 Adapter 改为每个可检索 partition 一次 embedding 候选查询，其显式检索层为：

```text
benchmark question
  -> Qwen3-VL query embedding（2048 维；按 MMA 上限补零到 4096）
  -> 五个原版 Manager 各以 embedding search 取最多 10 个候选：
       Episodic.details
       Semantic.details
       Procedural.summary
       Resource.summary
       KnowledgeVault.caption（仅 low/medium sensitivity）
  -> WMA 先按 provenance 过滤不可见 session
  -> 使用候选原始 embedding 重新计算 cosine
  -> 按 (-score, memory_id) 全局稳定排序和 memory_id 去重
  -> 最终截断为 0..7 条结构化 memory
  -> 注入原版 Chat Agent memory system prompt
```

因此 Top-7 的准确位置是：**五个 Manager 各自最多 10 条原版候选之后、Chat Agent 最终接收 evidence 之前的全局去重排序层**。Core Memory 由原版 Chat Agent 常驻读取，不参加 Top-7；不足 7 条不得补齐。它不是 MIRIX automatic prefetch 与六库并集的实现，也不是公共 embedding chunk 直接回答。

每个入选项必须独立保留 `memory_id`、partition、结构化 memory JSON、score、session、完整 `source_dialogue_ids`、原始 dialogue 内容、`image_ids` 和 `image_paths`。禁止 `str(list)`、错误解析 dialogue ID、把多条结果聚合成一条或丢弃 provenance。运行前应检查原始 dialogue 内容是否完整。

原版 Chat Agent 仍可调用 `search_in_memory`/`list_memory_within_timerange`，其内部工具结果允许使 Agent 实际检查的候选超过 7；这属于 MMA 原版内部推理，不计入显式 handoff Top-7。工具名仍应进入 trace，但无需把“预注入 + 内部工具结果”的并集限制为 7。

### 2.3 原版 Chat Agent 回答

回答链为：

```text
最多 7 条合规结构化 evidence + provenance + 所需原图
  -> 当前 benchmark 的 build_answer_messages()（仅此时加入 QA prompt）
  -> MMAOriginalAdapter.answer_with_memory()
  -> 原版 Chat Agent system prompt + 原版工具循环
  -> 原版 send_message tool
  -> 要求原生输出包含 <answer>...</answer>
  -> harness 原有 parse_answer_response()
```

Adapter 不调用通用回答模型替代 MMA Chat Agent。它在每题后恢复 Chat Agent 的 message history 和 topic，避免 QA 间泄漏。`retries=2` 表示回答契约最多 3 次尝试；缺失 `<answer>` 时只允许用同一模型和同一 prompt 重试，禁止替模型补标签、解析正文中的伪 tool call 或改写答案。其他 Agent、embedding、图片、Manager 和 tool-call 错误必须硬失败。

### 2.4 WMA checkpoint

WMA 必须逐 checkpoint 执行：仅 ingest 新增且当前可见的固定 chunk，结束当前 session，完成当前所有 QA 的 Manager 检索和 MMA Chat Agent 回答，然后才能 ingest 未来 session。`visible_session_ids` 必须同时约束候选和最终 Top-7。禁止预建完整 lifelong memory；任何未来 session 出现在 memory、retrieval、工具结果或 answer prompt 中都必须失败。

### 2.5 Fallback 边界

允许的行为只有：Memory Agent 根据原版 prompt 合理决定不写 memory；同一回答模型、同一 prompt 对 `<answer>` 契约失败最多重试 2 次；锁定 SDK 的原生传输重试；以及 manifest 明示的一固定 chunk 强制吸收、本地图片入 MMA database、配置指定 embedding transport、provenance sidecar 和全局 Top-7 handoff。

禁止：semantic fallback、direct insert、空消息 flush、旧 SQLite/snapshot/trace、frozen retrieval、换模型/provider、零向量或排名 fallback、将图片改成路径文本、`str(list)`、聚合 evidence、丢失 dialogue/provenance、解析正文伪 tool call、Adapter 补 `<answer>`/改写回答、统一回答 client，以及 WMA future leakage。任何非允许项都必须硬失败。

## 3. 固定输入

| Benchmark | 固定 chunk JSONL | 行数 | SHA256 |
|---|---|---:|---|
| Mem-Gallery | `data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl` | 3962 | `93e1d0b778addfbf4d8df8c387c588a1a5333f19864c41b6436e54e9dd0e189e` |
| H2HMEM dyadic | `data/h2hmem/chunks_dyadic.jsonl` | 2645 | `e046039299b40b0458a7b6e88f482c4081664a41aaf10f3861d2aff511e821b9` |
| H2HMEM multiparty | `data/h2hmem/chunks_multiparty.jsonl` | 866 | `75ebf39c87ce60ea98313270754c0fdb46bdf5458491f8138e69313999113e51` |
| WMA lifelong | `data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl` | 15437 | `75e89446f6d991b4a2e5b471ab754d2d7f8942615badc73930c4385d153fe748` |

Split manifest SHA256：`590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。

```bash
cd /data/haozhen/Memory-clean/Offline
sha256sum \
  data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunks_multiparty.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl \
  configs/multimodal_split_manifest.json
wc -l \
  data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunks_multiparty.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl
```

Harness 只能通过 `fixed_chunks.py` 按 manifest 选中的 sample 读取这些文件；禁止重新切分或从 benchmark source 静默重建。

## 4. 正式运行前检查

```bash
cd /data/haozhen/Memory-clean/Offline
PY=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python

$PY scripts/run_test_baseline_matrix.py --help
$PY scripts/prepare_mma_original.py --help
$PY scripts/validate_mma_original_run.py --help
PYTHONPATH=src $PY -m benchmarks.memgallery_harness.eval_memgallery --help
PYTHONPATH=src $PY -m benchmarks.h2hmem_harness.eval_h2hmem --help
PYTHONPATH=src $PY -m benchmarks.wma_harness.eval_wma --help
$PY scripts/prepare_mma_original.py
git -C .upstream/mma-c0e1a12 rev-parse HEAD 'HEAD^{tree}'
git -C .upstream/mma-c0e1a12 status --short
PYTHONPATH=src $PY -m unittest tests.test_mma_original_reproduction
```

上游必须是上述 commit/tree 且 `status --short` 无输出。实际 harness 入口是 `benchmarks.memgallery_harness.eval_memgallery`、`benchmarks.h2hmem_harness.eval_h2hmem` 和 `benchmarks.wma_harness.eval_wma`。已核对 matrix 的 CLI 只有：可重复的 `--endpoint`、`--baseline`、`--benchmark`，以及 `--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id`、`--skip-smoke`。Matrix 再向 harness 下发 `--split test`、`--resume`、state/result 目录、模型、endpoint、temperature、token、Top-K、timeout、retries 和 checkpoint 参数。

检查当前实现；以下输出用于确认 MMA 的输入路由与 manifest：

```bash
rg -n 'MMA|build_omni_|omni_input_manifest|shared_fixed_chunks' \
  src/benchmarks/memgallery_harness/eval_memgallery.py \
  src/benchmarks/h2hmem_harness/eval_h2hmem.py \
  src/benchmarks/wma_harness/eval_wma.py \
  src/benchmarks/baseline_runtime/omni_inputs.py
rg -n 'source_dialogue_ids|image_paths|chat_tool_calls|retrieved_memory_ids' \
  src/benchmarks/baseline_runtime/provenance.py \
  src/benchmarks/baseline_runtime/adapters/mma_original.py
```

检查 SDK 默认重试和服务；不得终止不属于本任务的进程：

```bash
.venvs/mirix/bin/python - <<'PY'
import openai
from openai._constants import DEFAULT_MAX_RETRIES
assert DEFAULT_MAX_RETRIES == 2
print("openai", openai.__version__, "default_max_retries", DEFAULT_MAX_RETRIES)
PY
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["dimension check"]}' \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; assert len(json.load(sys.stdin)["data"][0]["embedding"])==2048; print("embedding_dim=2048")'
nvidia-smi -i 3,4,5
for port in 8013 8014 8015; do curl -fsS "http://127.0.0.1:${port}/v1/models"; done
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api
```

## 5. 启动本地服务

若 8001 未运行，先用 `nvidia-smi` 选择一张不属于 GPU3/4/5 且确认空闲的 GPU，然后设置 `EMBEDDING_GPU`：

```bash
: "${EMBEDDING_GPU:?先设置空闲的 embedding GPU 编号}"
mkdir -p outputs/_services
tmux new-session -d -s mma_embedding_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=$EMBEDDING_GPU \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/serve_embeddings.py --port 8001 \
   --model Qwen/Qwen3-VL-Embedding-2B --dim 2048 --device cuda:0 \
   > outputs/_services/mma_embedding_8001.log 2>&1"
```

GPU3/4/5 确认空闲后分别启动三个回答/建库 worker：

```bash
mkdir -p outputs/_services
for spec in 3:8013 4:8014 5:8015; do
  gpu=${spec%:*}; port=${spec#*:}
  tmux new-session -d -s "mma_vllm_${port}" \
    "cd /data/haozhen/Memory-clean/Offline && \
     PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
     GPUS=$gpu PORT=$port \
     MODEL=Qwen/Qwen3-VL-4B-Instruct \
     SERVED_NAME=Qwen/Qwen3-VL-4B-Instruct \
     MAX_MODEL_LEN=131072 MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.75 \
     TOOL_CALL_PARSER=hermes sh scripts/serve_vllm.sh \
     > outputs/_services/mma_vllm_${port}.log 2>&1"
done
```

服务就绪后确认三个 `/v1/models` 都只使用指定模型，并检查日志中已启用 native tool choice 和 Hermes parser。


## 6. tmux 正式启动

使用从未出现过的新 Run-ID。三个 endpoint 会创建三个独立 worker；三个 benchmark 各自拥有独立进程、SQLite state、checkpoint、trace 和输出目录。

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID="mma_local_$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s mma_local_three \
  "cd /data/haozhen/Memory-clean/Offline && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults configs/defaults.json \
   --efficiency-config configs/model_efficiency.json \
   --split-manifest configs/multimodal_split_manifest.json \
   --baseline MMA \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint http://127.0.0.1:8013/v1 \
   --endpoint http://127.0.0.1:8014/v1 \
   --endpoint http://127.0.0.1:8015/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --skip-smoke \
   --run-id $RUN_ID \
   > outputs/_runs/$RUN_ID/launcher.log 2>&1"
```

命令中的 `--skip-smoke` 使 runner 直接进入正式任务。

## 7. 监控与断点恢复

```bash
tmux attach -t mma_local_three
tail -f "outputs/_runs/$RUN_ID/launcher.log"
cat "outputs/_runs/$RUN_ID/status.json"
find "outputs/_runs/$RUN_ID/_logs" -maxdepth 3 -type f -print
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/mma__mem_gallery.log"
```

分别确认三个 job 有不同 PID、result directory 和 `memory/datasets`。不能只看 matrix 主进程存在；还要观察 `call_traces/*.jsonl`、`.checkpoint/` 和 `status.json` 的 completed 数增长。

中断恢复时使用完全相同的 Run-ID、命令、模型、endpoint、manifest 和 chunk hash重新执行第 6 节命令；matrix 会下发 `--resume`。任何配置或输入签名变化都必须拒绝恢复。若要从头重跑，使用新的 Run-ID，不覆盖旧目录。

## 8. 输出与验收

```text
outputs/Mem-Gallery/MMA/<Run-ID>/
outputs/H2HMEM/MMA/<Run-ID>/
outputs/WorldMemArena/MMA/<Run-ID>/
```

每个目录必须包含 `results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_metrics.json`、`llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json`、`llm_judge_metrics.json`、根目录 Judge `call_trace.jsonl`、`.checkpoint/` 和逐 sample 的 `call_traces/*.jsonl`。Mem-Gallery/WMA 通常还写 `memory_metrics.json`；H2HMEM 写 `prediction_dyadic.json` 与 `prediction_multi_party.json`。

先运行已有 MMA validator；它是必要但不充分的检查：

```bash
for item in Mem-Gallery:275 H2HMEM:360 WorldMemArena:440; do
  benchmark=${item%:*}; count=${item#*:}
  /data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
    scripts/validate_mma_original_run.py \
    "outputs/$benchmark/MMA/$RUN_ID" --expected-qa "$count" \
    --output "outputs/$benchmark/MMA/$RUN_ID/mma_validation.json"
done
```

再执行当前产物 schema 能支持的跨 benchmark 检查；原始 dialogue 内容和 Chat Agent 工具返回并集仍必须由 validator/trace 额外验收：

```bash
: "${RUN_ID:?请设置正式运行使用的准确 Run-ID}"
RUN_ID="$RUN_ID" /data/haozhen/miniconda3/envs/pipeline_repro/bin/python - <<'PY'
import json, os
from pathlib import Path

root = Path("/data/haozhen/Memory-clean/Offline")
rid = os.environ["RUN_ID"]
expected = {
    "Mem-Gallery": (275, {"memgallery"}, "ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff"),
    "H2HMEM": (360, {"h2hmem_dyadic", "h2hmem_multiparty"}, "4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f"),
    "WorldMemArena": (440, {"wma_lifelong"}, "6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250"),
}
chunks = {
    "memgallery": (3962, "93e1d0b778addfbf4d8df8c387c588a1a5333f19864c41b6436e54e9dd0e189e", root / "data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl"),
    "h2hmem_dyadic": (2645, "e046039299b40b0458a7b6e88f482c4081664a41aaf10f3861d2aff511e821b9", root / "data/h2hmem/chunks_dyadic.jsonl"),
    "h2hmem_multiparty": (866, "75ebf39c87ce60ea98313270754c0fdb46bdf5458491f8138e69313999113e51", root / "data/h2hmem/chunks_multiparty.jsonl"),
    "wma_lifelong": (15437, "75e89446f6d991b4a2e5b471ab754d2d7f8942615badc73930c4385d153fe748", root / "data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl"),
}
for benchmark, (count, sources, prompt_hash) in expected.items():
    out = root / "outputs" / benchmark / "MMA" / rid
    required = (
        "results.json", "retrieval_trace.jsonl", "pipeline_qa.jsonl",
        "memory/memory_snapshot.jsonl", "run_manifest.json", "metrics.json",
        "efficiency_metrics.json", "call_metrics.json", "call_trace.jsonl",
        "llm_judge_results.json", "llm_judge_progress.jsonl",
        "llm_judge_checkpoint.json", "llm_judge_metrics.json",
    )
    assert all((out / name).is_file() for name in required)
    assert (out / ".checkpoint").is_dir()
    assert list((out / "call_traces").glob("*.jsonl"))
    results = json.loads((out / "results.json").read_text())
    traces = [json.loads(x) for x in (out / "retrieval_trace.jsonl").read_text().splitlines() if x]
    pipeline = [json.loads(x) for x in (out / "pipeline_qa.jsonl").read_text().splitlines() if x]
    manifest = json.loads((out / "run_manifest.json").read_text())
    judge = json.loads((out / "llm_judge_metrics.json").read_text())
    config = manifest.get("configuration") or manifest
    assert len(results) == len(traces) == len(pipeline) == count == judge["count"]
    assert judge["judge_errors"] == 0 and all(not row.get("error") for row in results)
    assert manifest["selection_mode"] == "strict_manifest" and manifest["split"] == "test"
    assert manifest["split_manifest_sha256"] == "590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36"
    ids = [row["manifest_question_id"] for row in results]
    assert ids == manifest["ordered_question_ids"]
    assert ids == [row["manifest_question_id"] for row in traces]
    assert ids == [row["manifest_question_id"] for row in pipeline]
    assert manifest["prompt_sha256"] == prompt_hash
    assert config["answer_model"] == config["executor_model"] == "Qwen/Qwen3-VL-4B-Instruct"
    assert int(manifest["top_k"]) == int(config["top_k"]) == 7
    runtime = manifest["baseline_runtime"]
    assert runtime["adapter"] == "mma_original"
    assert runtime["upstream_commit"] == "c0e1a127722edcfa4db5e71d03708cba53363000"
    assert runtime["upstream_tree"] == "4bb8b33535cb9d14b39277d953bcc80b5aec2e4c"
    rows = manifest.get("chunk_inputs") or [manifest.get("chunk_input")]
    assert all(isinstance(row, dict) for row in rows)
    assert {row["source"] for row in rows} == sources
    assert all(row.get("format") == "Chunk JSONL" and row.get("shared_fixed_chunks") is not False for row in rows)
    assert all((row["chunk_count"], row["sha256"]) == chunks[row["source"]][:2] for row in rows)
    assert all(Path(row["path"]).resolve() == chunks[row["source"]][2] for row in rows)
    for trace in traces:
        top = trace.get("top_k") or []
        assert len(top) <= 7
        assert len({row["memory_id"] for row in top}) == len(top)
        assert all(row.get("source_dialogue_ids") for row in top)
        assert all(Path(path).is_file() for row in top for path in row.get("image_paths", []))
        method = trace.get("retrieval_method_trace") or {}
        assert method.get("via") == "mma_original_managers_global_top7_handoff"
        assert method.get("candidate_limit_per_partition") == 10
    assert all((row.get("native_answer_trace") or {}).get("via") == "mma_original_chat_agent" for row in results)
    assert all(len((row.get("native_answer_trace") or {}).get("retrieved_memory_ids") or []) <= 7 for row in results)
    if benchmark == "WorldMemArena":
        assert all(
            {item.get("session_id") for item in trace.get("top_k", []) if item.get("session_id")}
            <= set(trace.get("visible_sessions", []))
            for trace in traces
        )
    print(benchmark, "PASS", count)
PY
```

正式验收还必须确认：

1. QA 数为 275/360/440，question ID、顺序和 sample 集与 test manifest 完全一致；H2HMEM 同时包含 dyadic 和 multiparty，WMA 仅 lifelong。
2. Manifest 记录四个固定 chunk 的绝对 path、SHA256、格式和数量，不得出现 `shared_fixed_chunks=false` 或 `benchmark-native source observations`。
3. `baseline_runtime.adapter=mma_original`，上游 commit/tree 正确，8 个 Agent prompt expected/actual hash 全部一致。
4. 每个 memory/evidence 是独立结构化对象，拥有真实 memory ID、partition、原始 dialogue、dialogue ID、图片和 provenance；所有图片路径存在，未降级成正文路径。
5. Manager 每分区候选数不超过 10；显式预注入 Chat Agent 的结构化 memory 不超过 7。Chat Agent 内部工具候选可超过 7，不作为失败条件。
6. `native_answer_trace.via=mma_original_chat_agent`，benchmark QA prompt 只在 final answer 阶段加入，回答来自原版 `send_message` tool；没有统一回答 client、文本 tool-call 修复、答案补标签或其他 fallback。
7. WMA 每个 checkpoint 的回答完成事件早于未来 session ingest，所有 retrieval/tool evidence 属于 `visible_sessions`。
8. 无 answer error、重复/缺失题或未完成 Judge；Judge 数与 QA 数一致且不计入效率指标。

当前 QA prompt SHA256 为：Mem-Gallery `ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff`，H2HMEM `4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f`，WMA `6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250`。正式运行时必须以代码现场重新计算并与 manifest 比较。

指标固定为：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- `F1`、`EM`、`Judge`：test QA 的 qa-wise average。
- `Cost-MB`、`Lat-MB`：建库阶段 USD/sample、seconds/sample。
- `Cost-QA`、`Lat-QA`：检索与回答阶段合计 USD/sample、seconds/sample。
- `#Calls (MB+QA)`：`(MB calls + answer calls) / sample 数`；retrieval 和 Judge calls 不计入。

## 9. 必须停止并报告

固定 chunk/manifest 缺失或 hash 变化、官方 checkout 不干净、仍走 `omni_inputs.py`、Agent prompt 改变、Meta/Memory Agent 或 Manager 被绕过、图片变成路径字符串、原始 dialogue/provenance 丢失、显式预注入 memory 超过 7、出现 silent fallback、WMA 未来泄漏、QA/Judge 数不匹配或 native tool call 无法解析时，立即停止并报告真实文件、调用链、trace 和日志，不得自行改变算法继续。

## 10. 给 Codex 的快捷指令

> 严格遵循本教程，核对 MMA 三个 harness 的输入、原始 dialogue、图片、provenance 和 validator。Top-7 只约束显式预注入 Chat Agent 的 memory ≤7，内部工具候选可以超过 7。使用 GPU3/4/5、8013/8014/8015、embedding 8001 和全新 Run-ID，在 tmux 直接运行 275/360/440 QA。
