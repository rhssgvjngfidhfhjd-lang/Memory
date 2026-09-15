# M3-Agent 三 Benchmark Test Split 本地 vLLM 运行教程

## 0. 当前实现状态与已批准实验口径

截至 2026-09-14，当前 M3 实现允许直接启动本地 vLLM 实验。

本实验明确接受：CLI/结果目录使用 `M3-Agent-caption`；三个 harness 通过 `m3_inputs.py` 从 benchmark 原始数据构造 M3 observation并记录 `shared_fixed_chunks: false`；WMA 在 checkpoint 时完成 Control retrieval并冻结只含 `visible_sessions` 的 `0..7` 个 clip bundle，最终 QA 可延后并发但只能消费冻结 job；官方 prompt literal 后允许追加已批准的长度约束与 JSON Schema；Control 尚无可交付 clip 时允许追加已批准的空证据搜索规则，要求继续 `[Search]` 且不得用模型先验知识直接 `[Answer]`；旧单数据集 validator 仅作辅助检查；最终 Top-7 单位使用 `m3:clip:<id>` bundle，而不是单个 graph node。

本实验的关键约束是：WMA 冻结 handoff 不含未来 session；最终显式 handoff 不超过 7 个 clip bundle；最终回答使用对应 benchmark 的 `build_answer_messages()` prompt。内部 graph node、Control 中间候选及普通/character 检索规模不计入最终 Top-7。

## 1. 实验目标与固定配置

按上述口径，使用本地 vLLM 完成：

- Mem-Gallery test：4 个 sample，严格 275 QA。
- H2HMEM test：dyadic 327 QA + multiparty 33 QA，合计严格 360 QA。
- WorldMemArena：只运行 lifelong，8 个 sample，严格 440 checkpoint QA。

固定参数：

- Matrix 协议：`configs/test_baseline_matrix.json`。
- Test manifest：`configs/multimodal_split_manifest.json`。
- CLI baseline：`M3-Agent-caption`。
- M3 上游：`https://github.com/ByteDance-Seed/m3-agent`，commit `0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c`，tree `af5aab0f5883ba4bf97209d10cc46b3837a9f94b`。
- 已批准的兼容修改：`mmagent/memory_processing_qwen.py` 把解析键 `video_descriptions` 改为官方 prompt 真实输出键 `video_description`；官方 memory prompt literal 后追加长度约束并请求 JSON Schema，但不得改写 literal 本身。
- 建库、Control 和最终回答模型：`Qwen/Qwen3-VL-4B-Instruct`。
- Embedding：`Qwen/Qwen3-VL-Embedding-2B`，`http://127.0.0.1:8001/v1`，2048 维。
- 回答 endpoint：`8013`、`8014`、`8015`，可分别放在 GPU3/4/5。
- Memory/Control/QA temperature：`0.0`。
- QA max tokens：`512`；M3 memory/Control max tokens：`1024`。两者必须分别记录并纳入运行签名。
- HTTP timeout：`180` 秒；retries：`2`。
- `top_k=7`；M3 普通搜索每轮 Top-2，Control 最多 5 轮。
- LLM Judge：OpenRouter `openai/gpt-4o-mini`，temperature `0.0`，max tokens `512`。Judge 不计入 MB/QA calls、cost 或 latency。

每次正式实验必须使用新 Run-ID，从 benchmark-native builder 重建 M3 自己的 graph/state，不得复用其他 Run-ID 的 memory graph、snapshot 或 retrieval trace。

## 2. Benchmark-native 输入与当前基准 hash

| 数据源 | M3 observation 构造入口 |
|---|---|
| Mem-Gallery | `build_m3_memgallery_chunks(dataset, data_dir, dataset_name)` |
| H2HMEM dyadic/multiparty | `build_m3_h2h_chunks_from_directory(data_dir, variant, conversation_id)` |
| WMA lifelong | `build_m3_wma_chunks_from_data(sample, data_dir, sample_path=...)` |

Test manifest SHA256：`590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。manifest、原始 benchmark 输入或 builder mapping 变化时必须重新完成运行前检查并使用新 Run-ID。

Builder 必须按原始 dialogue round 构造 temporally ordered observation，并保留 speaker、dialogue text、session ID、round/dialogue ID、timestamp、image ID、真实 image path 和 source provenance。run manifest 必须记录 `m3_inputs.py` mapping hash及 `shared_fixed_chunks: false`。

## 3. Prompt 边界

M3 内部 prompt literal 必须保持官方原文：

- `mmagent.prompts.prompt_generate_memory_with_ids_sft` SHA256：`75dd3021b9dc951e6b6f8c158f006eeafbc302045ad4bdce7b10ce44b3b687dc`。
- `m3_agent/control.py:system_prompt` SHA256：`6f334258d44951e39d64add4dc9c8e6fa7b312bbe6b6337df5480a91fde7f6fe`。
- `m3_agent/control.py:instruction` SHA256：`d09f2de7ec4ec0aa8169c3211d231eed958e342857544f9c32b1920073e5c4bf`。

`prompt_generate_memory_with_ids_sft` literal 后允许追加已批准的输出约束：两个结果列表各最多 6 条、每条不超过 40 词，并请求严格 JSON Schema。上述 hash只校验官方 literal；追加约束必须在 conformance manifest 中单独记录。

Benchmark QA prompt 只在 Control 结束、最终 evidence 完成后使用：

- Mem-Gallery：`src/benchmarks/memgallery_harness/runner/prompts.py`，prompt hash `ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff`。
- H2HMEM：`src/benchmarks/h2hmem_harness/prompts.py`，prompt hash `4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f`。
- WMA：`src/benchmarks/wma_harness/runner/prompts.py`，prompt hash `6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250`。

Control Agent 的 `[Answer]` 只是判断内部搜索是否终止，不得代替 benchmark QA 模型的最终回答。不得把 benchmark QA prompt 塞入 memory generation 或 Control 轮次。

## 4. M3-Agent 原版核心与非视频适配

### 4.1 建库调用链

```text
benchmark 原始数据
→ m3_inputs.py 按原始 dialogue round 构造 observation
→ 原始 speaker/dialogue/session/round/timestamp + 原始图片字节
→ 官方 prompt_generate_memory_with_ids_sft
→ video_description = Episodic Memory
→ high_level_conclusions = Semantic Memory
→ 官方 process_memories
→ VideoGraph text nodes / entity edges / clip temporal index
→ 持久化 memory_graph + snapshot + provenance trace
```

图片必须作为真实多模态 `image_url`/image payload 传给建库模型；path 只用于可追溯地定位原图，不得把图片降级成 path 字符串。

这三个 benchmark 不提供视频/音频，因此明确不执行 `process_video_clip`、视频帧抽取、音频处理、Face Detection、Speaker Diarization、Face/Voice 特征建点。不得用文本 embedding 伪造 Face/Voice node 或 Character Equivalence。图中的 Episodic/Semantic Memory、原版 `process_memories`、VideoGraph 和 graph traversal 必须保留。

### 4.2 Control Agent 多轮检索

1. 使用官方 Control `system_prompt` 和 `instruction`，输出 `[Search]` 或 `[Answer]`。
2. 最多 5 轮；第 5 轮按官方逻辑强制 `[Answer]`。
3. 普通 `[Search]` 调用官方 `mmagent.retrieve.search`，每轮 `topk=2`，先做节点相似度/图相关节点计算，再按 clip score 聚合。
4. 官方的特殊“character id”查询分支会用 memory-wise Top-20；它属于内部候选搜索，不受最终显式 handoff Top-7 约束。由于本适配不伪造 Character Equivalence，必须在 trace 中显式标出该分支，不得把它伪装成普通 Top-2。
5. 每轮记录 query、action、官方 search 返回节点/clip、score 和轮次；不得在检索失败时切换成公共扩展 chunk、改写 embedding 逻辑或其他 baseline。

### 4.3 最终 Top-7 的唯一正确定义

Agent 每轮 Top-2 和最终 Top-7 是两个不同层次：

- **每轮 Top-2**：只作用于普通 M3 search 的 clip retrieval，保留原版控制策略。
- **最多 5 轮**：Control 可以改写 query 并多次访问 graph。
- **最终 Top-7**：Control 完成后，将各轮发现的 clip 按原版顺序和 clip ID 去重，最多取 7 个 `RetrievedMemory` clip bundle 交给当前 benchmark QA prompt。每个 bundle 可以包含该 clip 的多个 Episodic/Semantic graph node；Control 内部看到的 node/clip 候选总数也可以超过 7。

最终每个 clip bundle 必须保留：clip memory ID、内部 node ID及其 `episodic|semantic` 类型和原文、source dialogue/round ID、session ID、timestamp、image ID、真实 image path、首次命中的 Agent search round 和 score。不得用无结构的 `str(list)` 代替结构化 bundle，不得返回无 provenance/无图片的聚合条目，不得从 chunk ID 中错误截取 dialogue ID。

### 4.4 最终回答与 WMA 时序

最终回答调用链：

```text
最多 7 个结构化 M3 clip memory bundle + 原始图片/provenance
→ 当前 benchmark harness 的 build_answer_messages
→ Qwen/Qwen3-VL-4B-Instruct
→ <answer>...</answer> 解析
→ results / retrieval trace / metrics / Judge
```

WMA 对每个 sample 按 checkpoint 串行 ingest，并在写入未来 session 前完成当前 checkpoint 的全部 Control retrieval。检索结果、问题、图片和 provenance 随即冻结到 job；最终 QA 可在 sample preparation 后并发执行，但只能消费冻结 job，不能重新查询完整 graph。每题 `retrieved_sessions` 必须是 `visible_sessions` 的子集。

## 5. Fallback 规则

允许且必须可追踪：

- 按固定配置对同一模型 endpoint 重试 2 次；失败后报错。
- 官方 `validate_and_fix_json` 对 memory JSON 的格式归一；如果仍不是严格的 `video_description`/`high_level_conclusions` list，建库失败。
- 已批准的长度约束、JSON Schema、`finish_reason=length` 检测和失败原始响应留档。
- 官方 Control 无法解析 action 时的 `Search` 路径，但必须在 trace 中记录 `parse_fallback=true`；如果造成无有效 evidence 或最终轮仍无法解析，正式验收必须失败。
- memory 可以合法地少于 7 条甚至为 0；但不得伪造或补齐。当前 harness 对空 evidence 可能直接报错，不得通过无 provenance 内容静默规避。

禁止：改写官方 M3 prompt literal 或追加未批准指令；除已批准的空证据搜索规则外改变 Control 决策协议；绕过 `process_memories` direct insert；强行生成 Semantic Memory；伪造 Face/Voice/Character Equivalence；图片降级为 path text；绕过声明的 `m3_inputs.py` builder；静默更换 embedding/回答模型/其他 baseline；使用旧 graph、snapshot 或 trace；补齐 Top-7；把 Control `[Answer]` 当 benchmark 答案；返回无 provenance evidence；让 WMA 冻结 handoff 包含未来 session，或在最终回答阶段重新查询完整 graph。

## 6. 运行前环境和协议检查

先进入唯一工作目录：

```bash
cd /data/haozhen/Memory-clean/Offline
```

确认 runner 的真实 CLI：

```bash
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/run_test_baseline_matrix.py --help
```

输出必须包含 `--baseline ... M3-Agent-caption ...`、三个 `--benchmark`、`--endpoint`、`--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id` 和 `--skip-smoke`。

校验 benchmark-native 原始输入与 split manifest：

```bash
test -d /data/haozhen/Memory-clean/Mem-Gallery/benchmark/data
test -d /data/haozhen/Memory-clean/H2HMEM-main/dataset
test -d /data/haozhen/Memory-clean/WorldMemArena/WorldMemArena/lifelong
echo '590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36  configs/multimodal_split_manifest.json' \
  | sha256sum -c -
```

确认 M3 venv、源码与基础 conformance 测试：

```bash
test -x .venvs/m3_agent/bin/python
test -d baselines/m3-agent-master
PYTHONPATH=src .venvs/m3_agent/bin/python -m unittest \
  tests/test_m3_agent_conformance.py
```

还必须审查 `git diff -- Offline/baselines/m3-agent-master Offline/src/benchmarks/baseline_runtime/adapters/m3_agent.py Offline/src/benchmarks/baseline_runtime/m3_inputs.py Offline/src/benchmarks/{memgallery_harness,h2hmem_harness,wma_harness}`，确认官方 prompt literal 未变，且附加约束和兼容修改均已批准并记录。

## 7. 启动和检查本地服务

先检查 GPU 和端口；不得终止不属于本实验的进程：

```bash
nvidia-smi -i 3,4,5
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8013/v1/models
curl -fsS http://127.0.0.1:8014/v1/models
curl -fsS http://127.0.0.1:8015/v1/models
```

如 8001 未启动，选择一张经 `nvidia-smi` 确认空闲且不与回答 worker 冲突的物理 GPU，再启动：

```bash
EMBED_GPU=<已确认的空闲物理GPU编号>
mkdir -p outputs/_services
tmux new-session -d -s m3_embedding_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=$EMBED_GPU \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/serve_embeddings.py \
   --host 127.0.0.1 --port 8001 \
   --model Qwen/Qwen3-VL-Embedding-2B \
   --dim 2048 --device cuda:0 --dtype bfloat16 --local-files-only \
   > outputs/_services/m3_embedding_8001.log 2>&1"
```

如 8013–8015 未启动，在 GPU3/4/5 各启动一个完全相同的 vLLM：

```bash
mkdir -p outputs/_services

tmux new-session -d -s m3_vllm_8013 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=3 PORT=8013 \
   MODEL=/data/shared_models/Qwen3-VL-4B-Instruct \
   SERVED_NAME=Qwen/Qwen3-VL-4B-Instruct \
   MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.90 \
   sh scripts/serve_vllm.sh > outputs/_services/m3_vllm_8013.log 2>&1"

tmux new-session -d -s m3_vllm_8014 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=4 PORT=8014 \
   MODEL=/data/shared_models/Qwen3-VL-4B-Instruct \
   SERVED_NAME=Qwen/Qwen3-VL-4B-Instruct \
   MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.90 \
   sh scripts/serve_vllm.sh > outputs/_services/m3_vllm_8014.log 2>&1"

tmux new-session -d -s m3_vllm_8015 \
  "cd /data/haozhen/Memory-clean/Offline && \
   PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
   GPUS=5 PORT=8015 \
   MODEL=/data/shared_models/Qwen3-VL-4B-Instruct \
   SERVED_NAME=Qwen/Qwen3-VL-4B-Instruct \
   MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.90 \
   sh scripts/serve_vllm.sh > outputs/_services/m3_vllm_8015.log 2>&1"
```

用真实 embedding 请求验证 2048 维，并校验三个 served model ID：

```bash
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["m3 preflight"]}' \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048'

for port in 8013 8014 8015; do
  curl -fsS "http://127.0.0.1:${port}/v1/models" \
    | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
      'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="Qwen/Qwen3-VL-4B-Instruct" for x in d.get("data", []))'
done

test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api
```


## 8. 正式启动

Runner 会为三个 endpoint 创建 worker 线程，三个 M3 benchmark job 可并行；每个 harness 子进程拥有独立的 `memory/datasets`、`.checkpoint`、trace 和输出目录。

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID="m3_agent_local_$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s m3_agent_local_matrix \
  "cd /data/haozhen/Memory-clean/Offline && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --baseline M3-Agent-caption \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint http://127.0.0.1:8013/v1 \
   --endpoint http://127.0.0.1:8014/v1 \
   --endpoint http://127.0.0.1:8015/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --split-manifest /data/haozhen/Memory-clean/Offline/configs/multimodal_split_manifest.json \
   --output-root /data/haozhen/Memory-clean/Offline/outputs \
   --skip-smoke \
   --run-id $RUN_ID \
   > outputs/_runs/$RUN_ID/launcher.log 2>&1"
```

上述命令通过 `--skip-smoke` 直接进入正式任务。正式新 Run-ID 不能指向已有结果目录。

## 9. 监控与断点恢复

```bash
tmux attach -t m3_agent_local_matrix
tail -f "/data/haozhen/Memory-clean/Offline/outputs/_runs/<Run-ID>/launcher.log"
cat "/data/haozhen/Memory-clean/Offline/outputs/_runs/<Run-ID>/status.json"
```

另开终端检查三个 job 的 endpoint、PID、状态和日志：

```bash
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  "/data/haozhen/Memory-clean/Offline/outputs/_runs/<Run-ID>/status.json"
find "/data/haozhen/Memory-clean/Offline/outputs/_runs/<Run-ID>/_logs" \
  -type f -maxdepth 3 -print
```

中断后只能使用同一 Run-ID、同一 endpoint、同一 chunk/manifest hash 和同一参数重新执行第 8 节命令。三个 harness 均接收 `--resume`，matrix 会自动传入；checkpoint 签名不匹配必须拒绝恢复。某个未完成 sample 需重建自己的 M3 graph 是正常恢复行为，不得借用其他 Run-ID 的 state。如果要从头重跑，创建另一个新 Run-ID，不覆盖旧目录。

## 10. 输出与验收

实际正式路径使用 CLI 注册名 `M3-Agent-caption`：

```text
/data/haozhen/Memory-clean/Offline/outputs/Mem-Gallery/M3-Agent-caption/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/H2HMEM/M3-Agent-caption/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/WorldMemArena/M3-Agent-caption/<Run-ID>/
```

每个目录必须包含 `results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`memory/datasets/<sample>/memory_graph.pkl`、`memory/datasets/<sample>/clip_manifest.jsonl`、`memory/datasets/<sample>/m3_execution_trace.jsonl`、`memory/datasets/<sample>/m3_conformance.json`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_trace.jsonl`、`call_traces/`、`llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json` 和 `llm_judge_metrics.json`。若 memory generation 失败，还必须保留 `memory_generation_failures/` 原始响应。

先检查 matrix 总状态与 QA/Judge 数：

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID=<要验收的Run-ID>
jq -e '.phase == "complete" and (.failed_jobs|length)==0 and (.failed_judges|length)==0' \
  "outputs/_runs/$RUN_ID/status.json"

for spec in 'Mem-Gallery 275' 'H2HMEM 360' 'WorldMemArena 440'; do
  set -- $spec
  result_dir="outputs/$1/M3-Agent-caption/$RUN_ID"
  test "$(jq 'length' "$result_dir/results.json")" -eq "$2"
  test "$(wc -l < "$result_dir/retrieval_trace.jsonl")" -eq "$2"
  test "$(wc -l < "$result_dir/pipeline_qa.jsonl")" -eq "$2"
  jq -e --argjson n "$2" '.count == $n and .judge_errors == 0' \
    "$result_dir/llm_judge_metrics.json"
done
```

再检查 M3 协议、Top-K、provenance 和 WMA 可见性（每条最终 item 是一个结构化 clip memory bundle）：

```bash
PYTHONPATH=src /data/haozhen/miniconda3/envs/pipeline_repro/bin/python - <<'PY'
import json
from pathlib import Path
run_id = "<要验收的Run-ID>"
root = Path("outputs")
for benchmark in ("Mem-Gallery", "H2HMEM", "WorldMemArena"):
    d = root / benchmark / "M3-Agent-caption" / run_id
    manifest = json.loads((d / "run_manifest.json").read_text())
    inputs = manifest.get("chunk_inputs") or [manifest.get("chunk_input")]
    assert inputs and all(x and x.get("shared_fixed_chunks") is False for x in inputs)
    assert all(x.get("path", "").endswith("m3_inputs.py") for x in inputs)
    assert all(x.get("mapping_sha256") for x in inputs)
    conf = manifest["m3_conformance"]
    assert conf["native_search_top_k"] == 2
    assert conf["native_round_limit"] == 5
    assert conf["handoff_top_k"] == 7
    traces = [json.loads(x) for x in (d / "retrieval_trace.jsonl").read_text().splitlines() if x]
    for trace in traces:
        items = trace["top_k"]
        assert 0 <= len(items) <= 7
        assert len({x["memory_id"] for x in items}) == len(items)
        for item in items:
            assert item["memory_id"].startswith("m3:clip:")
            assert item.get("source_dialogue_ids")
            assert item.get("session_id")
            assert "[episodic]" in item.get("content", "") or "[semantic]" in item.get("content", "")
            assert "image_ids" in item and "image_paths" in item
        method = trace["retrieval_method_trace"]
        assert 1 <= len(method["rounds"]) <= 5
        for row in method["rounds"]:
            if row.get("action") == "Search" and row.get("content") and "character id" not in row["content"]:
                assert row["native_search_top_k"] == 2
        if benchmark == "WorldMemArena":
            visible = set(trace["visible_sessions"])
            assert all(item["session_id"] in visible for item in items)
print("M3 protocol acceptance passed")
PY
```

`run_manifest.json` 还必须记录原始输入 manifest/hash、`m3_inputs.py` mapping hash、test manifest hash、三个官方 prompt literal hash、已批准输出约束、当前 QA prompt hash、上游 commit/tree、已批准 parser patch、模型、embedding 维度、Top-2/5-round/最终 clip-bundle Top-7。必须确认无 answer error、重复/缺失 question ID、未完成 Judge、冻结 handoff 中的 future session、silent fallback 或无结构聚合 evidence。

## 11. 指标汇报

严格使用固定列和顺序：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- `F1`、`EM`、`Judge`：按 QA 平均。
- `Cost-MB`、`Lat-MB`：memory build 的 USD/sample 和 seconds/sample。
- `Cost-QA`、`Lat-QA`：Control retrieval + 最终回答的 USD/sample 和 seconds/sample。
- `#Calls (MB+QA)`：`(MB calls + answer calls) / sample 数`；Control/retrieval calls 只保存在 call trace，不计入该列。
- Judge calls 不计入 calls、cost 或 latency。

## 12. 必须停止并报告

任一 split manifest、原始输入或 builder mapping hash 改变；CLI 或输出 schema 与本文不符；官方 M3 prompt literal hash 变化或出现未批准附加指令；建库未同时使用 Episodic/Semantic + VideoGraph；图片未作为视觉输入；出现伪造 Face/Voice/Equivalence；普通 search 不是 Top-2；Control 超过 5 轮；最终显式 `m3:clip:*` bundle handoff 超过 7；provenance 不全；WMA 冻结 handoff 含未来 session，或最终回答重新访问已写入未来 session 的 graph；发生未授权 fallback、answer error、题数/ID 不符或 Judge 未完成时，必须停止并提供代码位置、trace 和建议，不得自行降级。

## 13. 给 Codex 的快捷指令

> 严格遵循本教程，核对 benchmark-native 原始输入、`m3_inputs.py` mapping、split manifest、官方 prompt literal 和已批准输出约束，使用真实 CLI `M3-Agent-caption`、新 Run-ID 和 GPU3/4/5 的 8013/8014/8015 服务，直接运行 275/360/440 QA。内部 Control/graph 候选允许超过 7；最终显式 `m3:clip:*` handoff ≤7。WMA 最终回答只能消费 checkpoint 时冻结且仅含 `visible_sessions` 的 handoff。
