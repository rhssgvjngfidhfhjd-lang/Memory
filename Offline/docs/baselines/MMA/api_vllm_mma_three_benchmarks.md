# MMA 三 Benchmark Test Split OpenRouter API 运行教程


## 1. 目标与固定配置

- Mem-Gallery：严格 275 QA。
- H2HMEM：dyadic + multiparty，严格 360 QA。
- WorldMemArena：仅 lifelong，严格 440 checkpoint QA。
- 协议：`configs/test_baseline_matrix.json`
- Split manifest：`configs/multimodal_split_manifest.json`
- Baseline CLI：`MMA`
- MMA 上游 commit：`c0e1a127722edcfa4db5e71d03708cba53363000`
- MMA 上游 tree：`4bb8b33535cb9d14b39277d953bcc80b5aec2e4c`
- API defaults：`/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json`
- Efficiency config：`/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini`
- API key 文件：`/data/haozhen/Memory-clean/Nvida_api/Openrouter_api`
- 建库与回答模型：OpenRouter `openai/gpt-5-mini`
- API endpoint：`https://openrouter.ai/api/v1`
- reasoning effort：`minimal`
- temperature：`0.0`
- QA max tokens：`512`
- MMA 建库 executor max tokens：`4096`
- timeout：`180` 秒
- retries：`2`
- `top_k=7`
- Embedding：本地 `Qwen/Qwen3-VL-Embedding-2B`，端口 8001，维度 2048
- LLM Judge：OpenRouter `openai/gpt-4o-mini`

三个 benchmark 可由三个独立 worker 并行调用同一 endpoint，但必须使用独立进程、MMA SQLite/image state、checkpoint、call trace 和输出目录。API key 只能通过进程环境或 Judge 的 `--key-file` 传递，不得写进文档、命令行参数、日志或终端输出。

## 2. MMA 原版流程与 Top-7

### 2.1 建库 Agent

建库链必须是：

```text
run_test_baseline_matrix.py
  -> benchmark harness 按 test manifest 选择 sample
  -> fixed_chunks.py 读取不可变 Chunk JSONL
  -> create_adapter("MMA") / 独立 MMAOriginalAdapter
  -> 官方 AgentWrapper + sample 独立 SQLite
  -> 每个固定 Chunk 独立 send_message(
       message=Chunk.text,
       image_uris=Chunk.images,
       memorizing=True,
       force_absorb_content=False)
  -> 官方 TemporaryMessageAccumulator；每批最多 B 个 Chunk（B <= 官方默认 20）
  -> 满 B 个自动吸收；session/checkpoint 结束时直接吸收不足 B 个的尾部
  -> 官方 Meta Memory Agent
  -> trigger_memory_update
  -> 官方 Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault Agent
  -> 官方 Manager 决定新增、更新或不写 memory
```

上游默认 `TEMPORARY_MESSAGE_LIMIT=20`，且构造函数原生支持配置该累计门槛。Adapter 使用 `mma_native_batch_size` 配置 B，并在 session/checkpoint 结束时调用原生 `absorb_content_into_memory()` 处理尾部。禁止逐固定 chunk 强制吸收、empty-message flush、direct insert 或跳过原版 Agent/Manager。批次新增或更新 memory 的 provenance 必须覆盖该批次的全部源 Chunk。Agent 选择不写 memory 是合法行为，不能补造 Semantic Memory。

截至 2026-09-14，GPT-5-mini、建库输出上限 2048 的真实 API smoke 中，B=20、B=10 和 B=5 都出现同一建库请求连续三次达到 2048 tokens，因而均按停止规则终止。建库上限恢复为 MMA 官方默认的 4096、保留 B=5，QA 保持 512；4096 的两轮 API 诊断共 44 次调用均未发生输出截断或 HTTP 错误。官方 prompt、工具 schema 和 Manager 路由均不修改。

Pinned MMA 的 Manager 在外层进入生成器式 `db_context()`，它原本只 yield Session，再把同一个 Session 交给会进入 `with db_session` 的 ORM helper；内层退出会过早关闭/expunge Manager 仍需使用的对象，造成 `DetachedInstanceError`。Adapter 让 Manager `db_context` 真正进入当前 `SessionLocal` 的可重入 Session：仅最外层 Manager context 负责关闭，嵌套 helper context 保留原版 commit、refresh、rollback 和异常传播；`expire_on_commit` 恢复 SQLAlchemy 默认 `True`。工具参数校验等普通错误仍由原版 Agent 接收并修正；`DetachedInstanceError`、事务/session 状态错误和 SQL/driver 错误因无法确认是否已部分提交而立即停止，只记录第一个根因及 Agent/工具，禁止自动重试造成重复追加。

MMA 上游还把 `MAX_EMBEDDING_DIM` 固定为 4096，新建 memory 的 schema 会把本实验模型输出的 2048 维向量补零到 4096，但 merge/update 路径直接写入新的 2048 维向量，会导致 `(4096,)` / `(2048,)` 冲突。每个正式 sample 都从全新 SQLite 库开始；Adapter 在 ORM/schema/query 模块初始化前将 MMA 运行时存储维度统一为配置的 `embedding_dim=2048`。不修改 embedding 模型或值，不混用旧库，manifest 同时记录上游上限 4096 和实验存储维度 2048。

额度保护配置固定 `sample_concurrency=1`、`formal_job_attempts=1`。正式 worker 失败后不得由 Matrix 自动启动第二次 attempt；必须检查失败原因并使用全新 Run-ID。

Adapter 把本地图片存入 MMA image database，再作为真正的多模态 image content 发给 Agent；禁止把路径拼进正文。MMA 的 chat/meta/core/episodic/semantic/procedural/resource/knowledge-vault 8 份内部 system prompt 必须保持官方字节和 SHA256 不变。Benchmark QA prompt 不能进入建库过程。

每个正式 Run-ID 必须全新，从固定 chunk 重建 MMA 自己的 memory；三个 worker 不得共享 SQLite、图片目录、snapshot、checkpoint 或 retrieval trace。

### 2.2 Manager 检索和全局 Top-7

MMA 上游 `Agent.build_system_prompt_with_memories()` 原本按 memory component 自动检索，单次 Manager 调用上限 `MAX_RETRIEVAL_LIMIT_IN_SYSTEM=10`；Episodic 会分别执行 recent 和 relevant 两次查询。本实验 Adapter 改为每个可检索 partition 一次 embedding 候选查询，固定 Top-7 施加在以下位置：

```text
question -> 本地 8001 query embedding（2048，按 MMA schema 补到 4096）
  -> 原版 Episodic/Semantic/Procedural/Resource/KnowledgeVault Manager
     各自 embedding search，最多 10 个候选
  -> WMA provenance 可见性过滤
  -> 对候选原始 embedding 计算 cosine
  -> 全局 (-score, memory_id) 排序和 memory_id 去重
  -> 截断为 0..7 条独立结构化 memory
  -> 注入原版 Chat Agent system memory context
```

Core Memory 由 Chat Agent 常驻读取，不进入 Top-7。Top-7 是五个 Manager 候选汇合后的全局 handoff，不是每分区各取 7 条；不足 7 条不补齐。每条必须保留 memory ID、partition、结构化字段、score、原始 dialogue、完整 dialogue ID、图片和 provenance，禁止整体 `str(list)` 或合并为一条 evidence。

Chat Agent 的原版 `search_in_memory`/`list_memory_within_timerange` 工具必须保留，其内部工具结果允许使 Agent 实际检查的候选超过 7；这属于 MMA 原版内部推理，不计入显式 handoff Top-7。工具名仍应进入 trace。

### 2.3 原版 Chat Agent 回答

```text
最多 7 条合规 evidence + 对应图片
  -> 当前 harness 的 build_answer_messages()
  -> MMAOriginalAdapter.answer_with_memory()
  -> 原版 Chat Agent system prompt
  -> 原版 reasoning/tool loop
  -> 原版 send_message tool
  -> <answer>...</answer>
  -> harness 原有 parse_answer_response()
```

QA prompt 只在这一阶段加入；Meta/Memory Agent 与 Chat Agent 原版 prompt 均不改。回答必须来自 MMA Chat Agent，不能改走统一 QA client。每道 QA 后恢复 Chat Agent history/topic。`retries=2` 允许回答契约最多 3 次同模型重试；缺失 `<answer>` 时禁止 Adapter 补标签或重写正文，普通文本中的 `<tool_call>` 也不得被解析成原生 tool call。

Counting proxy 会在 API 转发边界记录全部 MMA chat completion，并强制 `reasoning={"effort":"minimal"}`、`temperature=0`、输出上限不超过 512，upstream timeout 为 180 秒。当前 MMA 内部 OpenAI client 没有显式传 `max_retries`，依赖锁定环境 `openai==1.109.1` 的默认 2 次重试；版本或默认值变化时必须停止。

### 2.4 WMA checkpoint

每个 checkpoint 只 ingest 新增且当前可见的固定 chunk，随后立即完成当前 QA 的 Manager 检索和 MMA Chat Agent 回答，之后才允许写入未来 session。`visible_session_ids` 必须约束 Manager 候选、Chat Agent 工具结果与最终 evidence；禁止未来 session 进入 memory context、图片或 prompt。

### 2.5 Fallback 边界

允许的行为只有：原版 Memory Agent 合理选择不写 memory；同一模型、provider 和 prompt 对 `<answer>` 契约失败最多重试 2 次；锁定 SDK 的原生传输重试；以及 manifest 明示的原生 accumulator 与 session-tail 吸收、本地图片入 MMA database、配置指定 embedding transport、SQLAlchemy session 生命周期兼容、首个不安全数据库根因硬失败、provenance sidecar 与全局 Top-7 handoff。

禁止：semantic fallback、direct insert、空消息 flush、旧 memory/snapshot/trace、frozen retrieval、换模型/provider、零向量或排名 fallback、图片路径字符串化、`str(list)`、聚合 evidence、丢失 dialogue/provenance、正文伪 tool-call 修复、补 `<answer>`/改写答案、统一回答 client、跳过失败题和 WMA future leakage。基础设施或协议错误必须硬失败。

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

禁止重新切分、重写或从 benchmark source 静默重建。Manifest 和 checkpoint signature 必须记录绝对 path、SHA256、格式及 chunk 数量。

## 4. API、环境与实现检查

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

官方 checkout 必须为第 1 节 commit/tree 且干净。实际 harness 入口是 `benchmarks.memgallery_harness.eval_memgallery`、`benchmarks.h2hmem_harness.eval_h2hmem` 和 `benchmarks.wma_harness.eval_wma`。Matrix 实际 CLI 为：可重复的 `--endpoint`、`--baseline`、`--benchmark`，以及 `--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id`、`--skip-smoke`。Matrix 再向 harness 下发 `--split test`、`--resume`、独立 state/result 目录、模型、endpoint、reasoning effort、temperature、token、Top-K、timeout、retries 和 checkpoint 参数。

只要下面仍显示 MMA 走 `build_omni_*`/`omni_input_manifest`，必须停止：

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

验证配置，不输出 key：

```bash
$PY -m json.tool /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json >/dev/null
$PY -m json.tool /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini >/dev/null
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api
.venvs/mirix/bin/python - <<'PY'
import openai
from openai._constants import DEFAULT_MAX_RETRIES
assert openai.__version__ == "1.109.1"
assert DEFAULT_MAX_RETRIES == 2
print("openai SDK retry contract OK")
PY
```

安全加载 key；不得使用 `set -x` 或 `echo "$OPENAI_API_KEY"`：

```bash
export OPENAI_API_KEY="$(tr -d '\r\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api)"
test -n "$OPENAI_API_KEY"
curl -fsS https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  | $PY -c 'import json,sys; assert any(x.get("id")=="openai/gpt-5-mini" for x in json.load(sys.stdin).get("data", [])); print("model OK")'
```

Embedding 仍是本地服务：

```bash
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["dimension check"]}' \
  | $PY -c 'import json,sys; assert len(json.load(sys.stdin)["data"][0]["embedding"])==2048; print("embedding_dim=2048")'
```

如 8001 未运行，先确认一张空闲 GPU 并设置 `EMBEDDING_GPU`：

```bash
: "${EMBEDDING_GPU:?先设置空闲的 embedding GPU 编号}"
mkdir -p outputs/_services
tmux new-session -d -s mma_api_embedding_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=$EMBEDDING_GPU \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/serve_embeddings.py --port 8001 \
   --model Qwen/Qwen3-VL-Embedding-2B --dim 2048 --device cuda:0 \
   > outputs/_services/mma_api_embedding_8001.log 2>&1"
```


## 5. 三 worker tmux 正式启动

使用全新 Run-ID；重复三次相同 endpoint 表示三个独立 worker，不是三个 provider。

```bash
cd /data/haozhen/Memory-clean/Offline
RUN_ID="mma_api_$(date +%Y%m%d_%H%M%S)"
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s mma_gpt5mini_three \
  "cd /data/haozhen/Memory-clean/Offline && \
   export OPENAI_API_KEY=\$(tr -d '\r\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api) && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json \
   --efficiency-config /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
   --split-manifest configs/multimodal_split_manifest.json \
   --baseline MMA \
   --benchmark Mem-Gallery \
   --benchmark H2HMEM \
   --benchmark WorldMemArena \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --endpoint https://openrouter.ai/api/v1 \
   --embedding-base-url http://127.0.0.1:8001/v1 \
   --skip-smoke \
   --run-id $RUN_ID \
   > outputs/_runs/$RUN_ID/launcher.log 2>&1"
```

命令中的 `--skip-smoke` 使 runner 直接进入正式任务。持续出现 401/402/429、timeout 或 provider error 时停止并报告；不得静默降样本、换模型/provider 或跳过失败题。

## 6. 监控与断点恢复

```bash
tmux attach -t mma_gpt5mini_three
tail -f "outputs/_runs/$RUN_ID/launcher.log"
cat "outputs/_runs/$RUN_ID/status.json"
find "outputs/_runs/$RUN_ID/_logs" -maxdepth 3 -type f -print
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/mma__mem_gallery.log"
```

确认三个 benchmark 的 PID、result directory、`memory/datasets`、`.checkpoint/` 和 `call_traces/` 相互独立，并监控请求错误率、token、费用和完成数。

中断后用完全相同的 Run-ID、命令、API 模型/provider、embedding、manifest 和 chunk hash重启，matrix 会下发 `--resume`。配置或输入签名变化必须拒绝恢复。从头重跑使用新的 Run-ID，不覆盖旧目录。

## 7. 输出与验收

```text
outputs/Mem-Gallery/MMA/<Run-ID>/
outputs/H2HMEM/MMA/<Run-ID>/
outputs/WorldMemArena/MMA/<Run-ID>/
```

每个目录必须包含 `results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_metrics.json`、`llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json`、`llm_judge_metrics.json`、Judge `call_trace.jsonl`、`.checkpoint/` 和 baseline `call_traces/*.jsonl`。Mem-Gallery/WMA 通常另有 `memory_metrics.json`；H2HMEM 另有两个 prediction JSON。

先执行现有 validator，并补充核对固定 chunk、原始 dialogue 和 Chat Agent 工具并集：

```bash
for item in Mem-Gallery:275 H2HMEM:360 WorldMemArena:440; do
  benchmark=${item%:*}; count=${item#*:}
  /data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
    scripts/validate_mma_original_run.py \
    "outputs/$benchmark/MMA/$RUN_ID" --expected-qa "$count" \
    --output "outputs/$benchmark/MMA/$RUN_ID/mma_validation.json"
done
```

再执行当前产物 schema 能支持的跨 benchmark 检查；原始 dialogue 内容、Chat Agent 工具返回并集和 checkpoint 事件顺序仍须由 validator/trace 额外验收：

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
    assert config["answer_model"] == config["executor_model"] == "openai/gpt-5-mini"
    assert config["reasoning_effort"] == "minimal"
    assert float(config["answer_temperature"]) == float(config["executor_temperature"]) == 0
    assert int(config["num_predict"]) == 512
    assert int(config["executor_max_tokens"]) == 4096
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

正式结果必须同时满足：

1. QA 数严格为 275/360/440，question ID、顺序和 sample 与 test manifest 一致；H2HMEM 包含 dyadic/multiparty，WMA 仅 lifelong。
2. Manifest 记录固定 chunk 的绝对 path、SHA256、格式和数量，不出现 `shared_fixed_chunks=false`；模型/provider/reasoning/temperature/token/timeout/retry 和 prompt hash 正确。
3. 官方 commit/tree 与 8 个 Agent prompt hash正确；所有 memory 均由 Meta/Memory Agent 和原版 Manager 生成，无 direct insert 或人工 fallback。
4. 每条 evidence 独立保存结构化 memory、memory ID、partition、原始 dialogue、dialogue ID、图片和 provenance；图片文件真实存在且以视觉输入发送。
5. 每分区候选最多 10；显式预注入 Chat Agent 的 memory 为 0..7，内部工具候选允许超过 7且不作为失败条件。
6. 回答 trace 为 `mma_original_chat_agent`，QA prompt 仅 final 使用，最终来自原版 `send_message`；没有正文 tool-call 解析、答案补标签、统一回答 client 或模型/provider fallback。
7. WMA answer 事件早于未来 session ingest，所有预取和工具 evidence 属于 `visible_sessions`。
8. 无 answer error、API 遗留错误、重复/缺失题或未完成 Judge；Judge 数与 QA 数相同且不计入效率。

当前 QA prompt SHA256：Mem-Gallery `ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff`；H2HMEM `4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f`；WMA `6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250`。运行时仍须从代码现场重新计算。

指标严格按以下字段和顺序：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- `F1`、`EM`、`Judge`：test QA 的 qa-wise average。
- `Cost-MB`、`Lat-MB`：建库 USD/sample、seconds/sample。
- `Cost-QA`、`Lat-QA`：检索与回答合计 USD/sample、seconds/sample。
- `#Calls (MB+QA)`：`(MB calls + answer calls) / sample 数`；retrieval/Judge calls 不计入。
- API cost/latency 使用 `/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini` 中 `openai/gpt-5-mini` 的固定系数。

## 8. 必须停止并报告

仍走 `omni_inputs.py`、chunk/manifest hash变化、官方源码不干净、Agent prompt 变化、Meta/Memory Agent 或 Manager 被绕过、原始 dialogue/图片/provenance 丢失、显式预注入 memory 超过 7、出现 silent fallback、WMA 未来泄漏、QA/Judge 数不符、API key/provider/model/config 不一致或持续 API 错误时，立即停止。报告真实日志、trace、调用链和已产生费用，不得自行改变算法继续。

## 9. 给 Codex 的快捷指令

> 严格遵循本教程，核对三个 MMA harness 的输入、原始 dialogue、图片、provenance 和 validator。Top-7 只约束显式预注入 Chat Agent 的 memory ≤7，内部工具候选可以超过 7。使用全新 Run-ID、三个独立 worker、同一 OpenRouter endpoint 和本地 embedding 8001，直接并行运行 275/360/440 QA。
