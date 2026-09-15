# M3-Agent 三 Benchmark Test Split OpenRouter API 运行教程

## 0. 当前实现状态与已批准实验口径

截至 2026-09-14，当前 M3 实现允许直接启动 OpenRouter API 实验。

本实验采用以下实现口径：

1. CLI 和结果目录使用 runner 的真实名称 `M3-Agent-caption`。
2. 三个 harness 通过 `src/benchmarks/baseline_runtime/m3_inputs.py` 从 benchmark 原始数据现场构造按原始轮次排序的 M3 observation；run manifest 如实记录 `shared_fixed_chunks: false`。本实验不要求 M3 与其他 baseline 共用固定 chunk JSONL。
3. WMA 在每个 checkpoint 只 ingest 当前可见 session，立即完成 Control retrieval，并将已过滤的 `0..7` 个 memory bundle、问题和 provenance 冻结到 job；最终 benchmark QA API 请求允许在 sample preparation 后并发执行。回答阶段只能读取冻结 job，不得再次访问已包含未来 session 的 graph。`retrieved_sessions` 必须是 `visible_sessions` 的子集。
4. `reasoning_effort=minimal` 由统一 API counting proxy 写入实际发送给 OpenRouter 的 memory-generation、Control 和最终 QA 请求。
5. `scripts/validate_m3_agent_reproduction.py` 只覆盖旧单数据集；正式验收以 matrix 状态、三个 harness 的 run manifest/results/retrieval trace/call trace、QA/Judge 数量和本文验收脚本为准。
6. 保留官方 M3 prompt literal，并允许在其后追加已批准的输出长度约束和 JSON Schema 请求，用于避免不完整 JSON；该兼容层必须在 manifest 中记录。
7. 最终 Top-7 的计数单位是 `m3:clip:<id>` bundle。每个 bundle 可包含同一 clip 的多个 Episodic/Semantic graph node；不要求将每个 node 单独计为一条 handoff memory。
8. Control 的官方 prompt literal 保持不变；经用户批准，adapter 在尚未检索到任何可交付 clip 的每一轮追加一条空证据搜索规则：必须选择 `[Search]`，不得用模型先验知识直接 `[Answer]`。最后一轮只有已经检索到 clip 时才追加官方强制 `[Answer]` 提示。

本实验的关键约束是：WMA 冻结 handoff 不含未来 session；最终显式 handoff 不超过 7 个 clip bundle；最终回答必须使用对应 benchmark 的 `build_answer_messages()` prompt。内部 graph node 数、Control 中间候选数和普通/character 检索规模不计入最终 Top-7。

## 1. 目标与 API 固定配置

按上述口径，使用 OpenRouter API 独立完成：

- Mem-Gallery test：4 个 sample，严格 275 QA。
- H2HMEM test：dyadic 327 QA + multiparty 33 QA，合计严格 360 QA。
- WorldMemArena：只运行 lifelong，8 个 sample，严格 440 checkpoint QA。

固定配置：

- Matrix：`/data/haozhen/Memory-clean/Offline/scripts/run_test_baseline_matrix.py`。
- 协议：`/data/haozhen/Memory-clean/Offline/configs/test_baseline_matrix.json`。
- Test manifest：`/data/haozhen/Memory-clean/Offline/configs/multimodal_split_manifest.json`。
- CLI baseline：`M3-Agent-caption`。
- API defaults：`/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json`。
- Efficiency config：`/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini`。
- API key 文件：`/data/haozhen/Memory-clean/Nvida_api/Openrouter_api`。
- 建库、Control 与最终回答模型：OpenRouter `openai/gpt-5-mini`。
- API base URL：`https://openrouter.ai/api/v1`。
- `reasoning_effort=minimal`；temperature `0.0`；QA max tokens `512`；M3 memory/Control max tokens `1024`。
- timeout `180` 秒；retries `2`。
- Embedding 仍使用本地 `Qwen/Qwen3-VL-Embedding-2B`，`http://127.0.0.1:8001/v1`，2048 维。
- M3 普通 search 每轮 Top-2；Control 最多 5 轮；最终显式 clip memory bundle `0..7`。
- Judge：OpenRouter `openai/gpt-4o-mini`，temperature `0.0`，max tokens `512`。

API key 绝对不得写进文档、日志、CLI 参数或终端输出。下文只从 key file 读取到环境变量，不显示其值；不得启用 `set -x`。

API efficiency config 中当前 `openai/gpt-5-mini` 系数为：输入 0.25 USD/million tokens，输出 2.00 USD/million tokens，latency base 3.47 seconds，input 0.000003 seconds/token，output 0.01754386 seconds/token，image 0.30 seconds/image。

M3 上游固定为 commit `0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c`、tree `af5aab0f5883ba4bf97209d10cc46b3837a9f94b`。已经用户批准的 vendored patch 将 `mmagent/memory_processing_qwen.py` 的 `video_descriptions` 解析键更正为官方 prompt 输出的 `video_description`。API 兼容层还在官方 memory prompt literal 后追加长度约束并请求 JSON Schema；不得改写官方 literal 本身。

## 2. Benchmark-native 输入、split 与 hash

| 数据源 | M3 observation 构造入口 |
|---|---|
| Mem-Gallery | `build_m3_memgallery_chunks(dataset, data_dir, dataset_name)` |
| H2HMEM dyadic/multiparty | `build_m3_h2h_chunks_from_directory(data_dir, variant, conversation_id)` |
| WMA lifelong | `build_m3_wma_chunks_from_data(sample, data_dir, sample_path=...)` |

Test manifest SHA256：`590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。按 test manifest 精确筛选后为 4/275、dyadic 4/327 + multiparty 1/33、8/440。manifest 或原始 benchmark 输入变化时必须重新完成运行前检查并使用新 Run-ID，不得静默复用旧 state。

每个 builder 必须按原始 benchmark 轮次构造一个 temporally ordered observation，并保留 speaker、dialogue text、session、round/dialogue ID、timestamp、image ID、真实 image path 和 source provenance。run manifest 必须记录 builder 路径/映射 hash，并明确写入 `shared_fixed_chunks: false`。

## 3. M3 prompt 和 benchmark QA prompt 边界

官方 M3 内部 prompt literal 必须保持原文：

- `prompt_generate_memory_with_ids_sft`：`75dd3021b9dc951e6b6f8c158f006eeafbc302045ad4bdce7b10ce44b3b687dc`。
- Control `system_prompt`：`6f334258d44951e39d64add4dc9c8e6fa7b312bbe6b6337df5480a91fde7f6fe`。
- Control `instruction`：`d09f2de7ec4ec0aa8169c3211d231eed958e342857544f9c32b1920073e5c4bf`。

`prompt_generate_memory_with_ids_sft` literal 后允许追加已批准的 API 输出约束：`video_description` 和 `high_level_conclusions` 各最多 6 条、每条不超过 40 词，并请求严格 JSON Schema。上述三个 SHA256 只校验官方 literal，不包含该兼容层；兼容层内容必须单独写入 conformance manifest。

Control 还允许追加已批准的空证据搜索规则：只要当前尚无可交付 clip，就必须 `[Search]` 且不得以模型先验知识代替检索证据。该规则不改变上述官方 literal 及其 SHA256，并必须作为 deviation 写入 conformance manifest。

Benchmark QA prompt 只能在最终回答时调用：

- Mem-Gallery `src/benchmarks/memgallery_harness/runner/prompts.py`：`ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff`。
- H2HMEM `src/benchmarks/h2hmem_harness/prompts.py`：`4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f`。
- WMA `src/benchmarks/wma_harness/runner/prompts.py`：`6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250`。

Control `[Answer]` 只终止内部检索，不得作为 benchmark 最终答案。不得在 memory generation/Control 使用 benchmark QA prompt，也不得在最终 QA 使用 Control prompt。

## 4. M3-Agent 建库、检索和回答调用链

### 4.1 Episodic/Semantic Memory Graph

```text
benchmark 原始数据
→ m3_inputs.py 按原始 dialogue round 构造 observation
→ speaker/dialogue/session/round/timestamp + 原始图片字节/provenance
→ GPT-5-mini + 官方 prompt_generate_memory_with_ids_sft
→ video_description (Episodic) + high_level_conclusions (Semantic)
→ 官方 process_memories
→ VideoGraph text nodes / entity edges / clip temporal index
→ 持久化 graph、snapshot、generation 和 execution trace
```

原始图片必须作为真实多模态 payload 进入 GPT-5-mini，path 仅用于 provenance；不能只把 path 拼进文本。非视频 benchmark 不运行帧抽取、音频、Face Detection、Speaker Diarization 和 Face/Voice node，也不能用文本 embedding 伪造 Character Equivalence。Episodic/Semantic、官方 `process_memories`、VideoGraph 和 traversal 不可绕过。

### 4.2 Control 多轮检索

1. GPT-5-mini 使用官方 Control prompt 输出 `[Search]` 或 `[Answer]`。
2. 尚未发现任何可交付 clip 时，每轮追加已批准的空证据搜索规则并要求 `[Search]`；只有已经发现 clip 后才允许 `[Answer]`。
3. 最多 5 轮；第 5 轮仅在已经发现 clip 时追加官方强制 `[Answer]` 指令，否则继续要求 `[Search]`，并将该轮真实检索结果作为 handoff。
4. 每个普通 `[Search]` 调用官方 `mmagent.retrieve.search(..., topk=2, threshold=0.5)`，保留 graph-related node 和 clip-score 聚合逻辑。
5. 官方“character id”特殊分支使用 memory-wise Top-20；它属于内部候选搜索，不受最终显式 handoff Top-7 约束。本适配没有伪造 Character Equivalence，如果触发该分支必须在 trace 如实记录。
6. 所有轮次的 action、query、候选 node/clip、score、模型用量和 parse fallback 都必须记录。

### 4.3 Agent Top-2 / 5 轮与最终 Top-7

- Agent 每轮 Top-2 只限普通 M3 clip search，是原版内部检索上限。
- Control 可以在最多 5 轮中使用不同 query 继续查 graph。
- Control 结束后，将各轮发现的 clip 按原版顺序和 clip ID 去重，最多取 7 个 `RetrievedMemory` clip bundle 交给 benchmark QA prompt。每个 bundle 可以包含该 clip 的多个 Episodic/Semantic graph node；Control 内部看到的 node/clip 候选总数可以超过 7。

每个最终 clip bundle 必须包含 clip memory ID、内部 node ID及其 `episodic|semantic` 类型和内容、source dialogue/round ID、session ID、timestamp、image ID、真实 image path、首次命中的 Agent round 和 score。禁止用无结构的 `str(list)` 代替结构化 bundle，禁止错解 dialogue ID，禁止返回无 provenance/无图片的聚合 evidence。

### 4.4 最终 QA 与 WMA

```text
0..7 条结构化 M3 memory + 原始图片/provenance
→ 当前 benchmark build_answer_messages
→ OpenRouter openai/gpt-5-mini (minimal, temperature=0, max tokens=512)
→ <answer>...</answer>
→ results / retrieval trace / metrics / Judge
```

WMA 每个 sample 内按 checkpoint 串行 ingest，并在写入未来 session 前完成当前 checkpoint 的全部 Control retrieval。检索结果、问题、图片和 provenance 随即冻结到 job；最终 QA API 请求可以在 sample preparation 完成后并发执行，但只能使用冻结 job，不能重新查询完整 graph。每题必须确认 `retrieved_sessions` 是其 `visible_sessions` 子集。

## 5. Fallback 边界

允许且必须显式记录：

- 对同一 OpenRouter 模型/provider 的 2 次重试；最终失败就使整题/任务失败。
- 官方 `validate_and_fix_json` 的 memory JSON 格式归一；仍不符合两个 list 键时不得伪造 memory。
- 已批准的长度约束、严格 JSON Schema、`finish_reason=length` 检测和失败原始响应留档。
- 官方 Control 解析失败后的 `Search` 路径，但必须 trace `parse_fallback=true`；若导致最终轮仍无有效 action/evidence，正式验收失败。
- 经用户批准的空证据搜索规则：尚无可交付 clip 时必须继续 `[Search]`，不得用 Control 的先验答案替代真实检索证据。
- 合法的 `0..7` 条 memory；空或不足 7 不能用无 provenance 文本补齐。

禁止：改写官方 M3 prompt literal 或追加未批准指令；使用其他 API model/provider；丢弃 `minimal`；绕过 `m3_inputs.py` 声明的 benchmark-native builder；direct insert；伪造 Semantic/Face/Voice/Character Equivalence；把图片当 path text；替换 embedding 逻辑；借用其他 baseline、旧 memory/snapshot/trace；Top-7 补齐；把 Control answer 当 benchmark answer；无 provenance evidence；让冻结 handoff 包含 WMA 未来 session；在最终回答阶段重新查询包含未来 session 的 graph；吞掉 429/timeout/answer error。

## 6. 运行前环境和输入检查

```bash
cd /data/haozhen/Memory-clean/Offline

/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/run_test_baseline_matrix.py --help

/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json >/dev/null
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini >/dev/null
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api
test -x .venvs/m3_agent/bin/python
test -d baselines/m3-agent-master

test -d /data/haozhen/Memory-clean/Mem-Gallery/benchmark/data
test -d /data/haozhen/Memory-clean/H2HMEM-main/dataset
test -d /data/haozhen/Memory-clean/WorldMemArena/WorldMemArena/lifelong
echo '590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36  configs/multimodal_split_manifest.json' \
  | sha256sum -c -

PYTHONPATH=src .venvs/m3_agent/bin/python -m unittest \
  tests/test_m3_agent_conformance.py
```

`--help` 必须显示 `M3-Agent-caption`、三个 benchmark choice，以及 `--endpoint`、`--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id` 和 `--skip-smoke`。

安全加载 key，不输出其值：

```bash
export OPENAI_API_KEY="$(tr -d '\r\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api)"
test -n "$OPENAI_API_KEY"
```

检查 OpenRouter 模型，不输出 header 或 key：

```bash
curl -fsS https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="openai/gpt-5-mini" for x in d.get("data", []))'
```

检查本地 embedding 服务：

```bash
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["m3 api preflight"]}' \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048'
```

如 8001 未启动，选择经 `nvidia-smi` 确认空闲的物理 GPU，使用当前真实服务脚本：

```bash
EMBED_GPU=<已确认的空闲物理GPU编号>
mkdir -p outputs/_services
tmux new-session -d -s m3_api_embedding_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=$EMBED_GPU \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/serve_embeddings.py \
   --host 127.0.0.1 --port 8001 \
   --model Qwen/Qwen3-VL-Embedding-2B \
   --dim 2048 --device cuda:0 --dtype bfloat16 --local-files-only \
   > outputs/_services/m3_api_embedding_8001.log 2>&1"
```


## 7. 三个 Benchmark 并行正式启动

三次相同 `--endpoint` 会创建三个 matrix worker，各自领取一个 M3 benchmark job。每个 job 由独立 harness 子进程运行，且结果路径天然分离；禁止手工改成共享 memory state/checkpoint/trace。

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID="m3_agent_api_$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s m3_agent_api_matrix \
  "cd /data/haozhen/Memory-clean/Offline && \
   export OPENAI_API_KEY=\$(tr -d '\\r\\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api) && \
   test -n \"\$OPENAI_API_KEY\" && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json \
   --efficiency-config /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
   --baseline M3-Agent-caption \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --split-manifest /data/haozhen/Memory-clean/Offline/configs/multimodal_split_manifest.json \
   --output-root /data/haozhen/Memory-clean/Offline/outputs \
   --skip-smoke \
   --run-id $RUN_ID \
   > outputs/_runs/$RUN_ID/launcher.log 2>&1"
```

命令行不含 key，runner 日志只记录子进程命令而不记录环境变量。不得在启动命令中使用 `--answer-api-key`，不得打印 `$OPENAI_API_KEY`。当前 defaults 使三个 benchmark 拥有各自的 sample concurrency；API 限流时必须报告真实并发、429 率和 latency，不能静默降并发、换模型或跳过失败样本。

命令显式使用 `--skip-smoke`，直接进入正式任务。

## 8. 监控、断点恢复与安全

```bash
tmux attach -t m3_agent_api_matrix
tail -f "/data/haozhen/Memory-clean/Offline/outputs/_runs/<Run-ID>/launcher.log"
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  "/data/haozhen/Memory-clean/Offline/outputs/_runs/<Run-ID>/status.json"
find "/data/haozhen/Memory-clean/Offline/outputs/_runs/<Run-ID>/_logs" \
  -maxdepth 3 -type f -print
```

`status.json` 中必须有三个不同 M3 job，各自具有 child PID、result directory 和状态。不得只根据 tmux/matrix 主进程存活判定三个任务在推进。日志中如出现 API key，必须立即停止、清理暴露产物并更换 key。

中断后用同一 Run-ID、defaults、efficiency config、provider/model、endpoint、embedding、原始输入/builder/manifest hash 和全部参数重新执行第 7 节命令。Matrix 会向 harness 传 `--resume`，只能复用签名一致的同 Run-ID checkpoint。新正式实验使用新 Run-ID 从 benchmark-native builder 重建 graph，不复用旧 memory/snapshot/retrieval trace。

## 9. 输出与验收

真实正式输出目录：

```text
/data/haozhen/Memory-clean/Offline/outputs/Mem-Gallery/M3-Agent-caption/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/H2HMEM/M3-Agent-caption/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/WorldMemArena/M3-Agent-caption/<Run-ID>/
```

每个目录必须有 `results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`memory/datasets/<sample>/memory_graph.pkl`、`clip_manifest.jsonl`、`m3_execution_trace.jsonl`、`m3_conformance.json`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_trace.jsonl`、`call_traces/`、`llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json` 和 `llm_judge_metrics.json`。若发生 memory-generation 失败，还必须保留 `memory_generation_failures/` 原始响应。

QA/Judge 数量和 matrix 状态：

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

M3 协议、最终 clip-bundle Top-7 和 WMA 可见性：

```bash
PYTHONPATH=src /data/haozhen/miniconda3/envs/pipeline_repro/bin/python - <<'PY'
import json
from pathlib import Path
run_id = "<要验收的Run-ID>"
for benchmark in ("Mem-Gallery", "H2HMEM", "WorldMemArena"):
    d = Path("outputs") / benchmark / "M3-Agent-caption" / run_id
    m = json.loads((d / "run_manifest.json").read_text())
    inputs = m.get("chunk_inputs") or [m.get("chunk_input")]
    assert inputs and all(x and x.get("shared_fixed_chunks") is False for x in inputs)
    assert all(x.get("path", "").endswith("m3_inputs.py") for x in inputs)
    assert all(x.get("mapping_sha256") for x in inputs)
    c = m["m3_conformance"]
    assert (c["native_search_top_k"], c["native_round_limit"], c["handoff_top_k"]) == (2, 5, 7)
    assert m.get("reasoning_effort") == "minimal" or m.get("configuration", {}).get("reasoning_effort") == "minimal"
    traces = [json.loads(x) for x in (d / "retrieval_trace.jsonl").read_text().splitlines() if x]
    for t in traces:
        items = t["top_k"]
        assert len(items) <= 7 and len({x["memory_id"] for x in items}) == len(items)
        for x in items:
            assert x["memory_id"].startswith("m3:clip:")
            assert x.get("source_dialogue_ids") and x.get("session_id")
            assert "[episodic]" in x.get("content", "") or "[semantic]" in x.get("content", "")
            assert all(k in x for k in ("image_ids", "image_paths"))
        method = t["retrieval_method_trace"]
        assert 1 <= len(method["rounds"]) <= 5
        for row in method["rounds"]:
            if row.get("action") == "Search" and row.get("content") and "character id" not in row["content"]:
                assert row["native_search_top_k"] == 2
        if benchmark == "WorldMemArena":
            visible = set(t["visible_sessions"])
            assert all(x["session_id"] in visible for x in items)
print("M3 API protocol acceptance passed")
PY
```

每个 `run_manifest.json` 必须记录原始输入 manifest/hash、`m3_inputs.py` builder mapping hash、split manifest hash、M3 上游 commit/tree/已批准 patch、官方内部 prompt literal hash、已批准输出约束、当前 benchmark QA prompt hash、`openai/gpt-5-mini`、minimal、embedding 2048、Top-2/5-round/Top-7 和无 top-up。还必须确认每个 M3 memory/Control call trace 都是指定 API 配置，三 benchmark 的 PID、state、checkpoint、trace 和目录独立，且无 answer error、429 遗留、重复/缺失 ID、冻结 handoff 中的 future session、silent fallback 或未完成 Judge。

## 10. 指标汇报

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- `F1`、`EM`、`Judge` 是 QA-wise average。
- `Cost-MB`、`Lat-MB` 是 memory build 的 USD/sample 和 seconds/sample。
- `Cost-QA`、`Lat-QA` 是 Control retrieval + 最终 answer 的 USD/sample 和 seconds/sample。
- `#Calls (MB+QA)=(MB calls + answer calls)/sample 数`；Control/retrieval calls 只记录在 call trace，不计入该列。
- Judge calls 不计入 calls/cost/latency；cost 和 latency 使用指定 API efficiency config 系数。

## 11. 必须停止并报告

出现任一情况时不得自行降级：split manifest、原始输入或 builder mapping hash 变化；CLI/schema 与本文不符；官方 M3 prompt literal 改变或出现未批准附加指令；Episodic/Semantic/VideoGraph/traversal 被绕过；图片变 path text；伪造 Face/Voice/Equivalence；普通 search 非 Top-2；Control 超过 5 轮；最终显式 `m3:clip:*` bundle handoff 超过 7；provenance 不全；WMA 冻结 handoff 含未来 session，或最终回答重新访问已写入未来 session 的 graph；任一 memory/Control/QA call 未使用指定 GPT-5-mini/minimal；API key 泄露；未授权 fallback、429/timeout/answer error 被吞掉；QA 数/ID 或 Judge 不完整。应提供代码位置、真实 trace 和建议，等待确认。

## 12. 给 Codex 的快捷指令

> 严格遵循本教程，核对 benchmark-native 原始输入、`m3_inputs.py` mapping、split manifest、官方 prompt literal 与已批准输出约束，安全加载 OpenRouter key，使用真实 CLI `M3-Agent-caption`、GPT-5-mini minimal、本地 8001 embedding 和新 Run-ID，直接并行运行 275/360/440 QA。内部 Control/graph 候选允许超过 7；最终显式 `m3:clip:*` handoff ≤7。WMA 最终回答只能消费 checkpoint 时冻结且仅含 `visible_sessions` 的 handoff。
