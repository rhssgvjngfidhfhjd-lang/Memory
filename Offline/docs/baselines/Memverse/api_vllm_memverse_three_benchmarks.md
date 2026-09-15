# MemVerse OpenRouter API 三 Benchmark Test Split 运行教程


## 1. API 固定配置

- baseline：`MemVerse`；
- defaults：`/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json`；
- efficiency config：`/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini`；
- key file：`/data/haozhen/Memory-clean/Nvida_api/Openrouter_api`；
- 建库与回答：OpenRouter `openai/gpt-5-mini`；
- provider/base URL：`https://openrouter.ai/api/v1`；
- `reasoning_effort=minimal`，`temperature=0`，QA `max_tokens=512`；
- timeout 180 秒，retries 2；
- embedding 仍为本地 `Qwen/Qwen3-VL-Embedding-2B`，端口 8001，2048 维；
- `top_k=7`；judge 为 OpenRouter `openai/gpt-4o-mini`。

Test split 固定为 Mem-Gallery 275 QA、H2HMEM dyadic + multiparty 360 QA、WorldMemArena lifelong 440 checkpoint QA。三个 benchmark 可用三个独立 worker 并行调用同一 endpoint，但必须是独立 harness 进程、独立 memory state、checkpoint、call trace 和输出目录。

API efficiency config 当前 SHA256 为 `6c789e87feb8f4737dd221a2da5a9294b25ef250062fd64c5e7602d43855c3b7`；其中 GPT-5-mini 的固定价格是 input 0.25、output 2.00 USD/million tokens，latency 参数为 base 3.47 秒、input 0.000003 秒/token、output 0.01754386 秒/token、image 0.30 秒/image。不得运行时换模型或改系数。

## 2. 固定输入与 prompt 身份

只允许 `src/benchmarks/fixed_chunks.py` 读取以下不可变 JSONL：

| 输入 | 行数 | SHA256 |
|---|---:|---|
| `data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl` | 3962 | `93e1d0b778addfbf4d8df8c387c588a1a5333f19864c41b6436e54e9dd0e189e` |
| `data/h2hmem/chunks_dyadic.jsonl` | 2645 | `e046039299b40b0458a7b6e88f482c4081664a41aaf10f3861d2aff511e821b9` |
| `data/h2hmem/chunks_multiparty.jsonl` | 866 | `75ebf39c87ce60ea98313270754c0fdb46bdf5458491f8138e69313999113e51` |
| `data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl` | 15437 | `75e89446f6d991b4a2e5b471ab754d2d7f8942615badc73930c4385d153fe748` |

Test manifest SHA256：`590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。QA prompt SHA256：Mem-Gallery `ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff`，H2HMEM `4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f`，WMA `6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250`。

按 manifest 精确筛选 sample/question；禁止重新切分或静默重建 chunk。每个正式 Run-ID 必须从固定 chunk 重新生成 MemVerse memory，不能使用任何旧 Run-ID 产物。

## 3. MemVerse 特有调用链与 Top-7

### 3.1 原版链路

原版插入路径是 `app.py:/insert` → `orchestrator.handle_insert()` → 原版 image/video/audio processor 生成 caption → conversation entry → `update_long_term_memory()` → `build_memory.process_memory()` 并发调用未改写的 core/episodic/semantic system prompt → 三个 memory JSONL → 三个 LightRAG 图/向量库。

原版查询路径是 `app.py:/query` → `handle_query()` → 可选 parametric memory 与 relevance check → 不相关时 `rag_retrieve()` → **仅 `mem_core.aquery(mode="hybrid")`** → LightRAG keyword/entity/relation/graph/source-chunk 检索及 RAG response → `generate_final_answer()`。

当前 benchmark QA 必须用三个 harness 现有 prompt；MemVerse 内部三个 Memory Manager prompt 保持原样。兼容实现必须明确外层回答只执行一次，并把真实原生 retrieval context 交给 benchmark prompt；不能把 LightRAG 生成答案当作逐条 memory，再让 benchmark model二次回答而不记录这次调用。

### 3.2 Top-7 定义

`QueryParam(mode="hybrid", top_k=7)` 中的 7 施加在 LightRAG 原生 entity vector seed 和 relationship vector seed 两条检索通道，各自最多 7；随后才图遍历和 source chunk 回溯。它不是三类 memory 的数量，也不是三段聚合文本的外层切片。

图展开可能产生超过 7 个 entity/relation/chunk。若统一实验规范还要求最终交给回答模型的 evidence 总数不超过 7，应在获得确认后增加显式、可追踪的最终 handoff cap，并同时保留原生 seed 与图遍历 trace；不得改变 seed Top-7 含义或静默截断。当前代码把 Top-7 分别用于三个 store，已偏离上游 core-only 路径；正式结果应在运行记录和验收报告中如实标注这一实现差异。

### 3.3 允许和禁止的 fallback

允许：原版明确启用 parametric memory 时，其“不相关 → long-term RAG”分支；memory 不足时原版明确允许的普通回答；同一 API 模型/provider 的最多 2 次请求重试；同签名同 Run-ID 的 checkpoint 恢复。

禁止：API 报错后换模型/provider；embedding 失败后改用其他 embedding 或旧向量；用全库、聚合三段文本、统一 cosine 或旧 snapshot 代替原生检索；无 provenance/image 的 evidence；跳过失败 QA；重建 chunk；修改内部 Agent prompt/QA prompt；吞掉 401/402/429/timeout 后继续计分。

## 4. 已核对入口、CLI 与进程隔离

总入口 `scripts/run_test_baseline_matrix.py` 的真实 CLI 只有：`--endpoint`、`--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id`、`--baseline`、`--benchmark`、`--skip-smoke`；其中 endpoint/baseline/benchmark 可重复。

它分别启动：

- `python -m benchmarks.memgallery_harness.eval_memgallery --all-datasets ...`；
- `python -m benchmarks.h2hmem_harness.eval_h2hmem ...`，默认 variant `all`；
- `python -m benchmarks.wma_harness.eval_wma ...`，staged data 只有 lifelong test。

共同参数由 runner 真实传入：`--baseline/--result-dir/--baseline-state-dir/--sample-concurrency/--answer-concurrency/--checkpoint-every`、answer/executor/embedding 配置、`--top-k/--request-timeout/--retries/--efficiency-config/--resume`、`--split-manifest/--split test`。API defaults 中的 `reasoning_effort=minimal` 会追加为 `--reasoning-effort minimal`。

重复三次同一个 OpenRouter endpoint 会创建三个 matrix worker；每个 worker领取一个 benchmark 后启动独立 subprocess。输出路径天然按 benchmark 分开，state 位于各自 `result-dir/memory/datasets`，不能人工改成共享路径。

## 5. 环境、配置与服务检查

```bash
cd /data/haozhen/Memory-clean/Offline
export MEMVERSE_PYTHON=/data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python

test -x "$MEMVERSE_PYTHON"
"$MEMVERSE_PYTHON" -c 'import openai,numpy,tiktoken,nano_vectordb,networkx,json_repair; print("MemVerse env OK")'
"$MEMVERSE_PYTHON" -m json.tool /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json >/dev/null
"$MEMVERSE_PYTHON" -m json.tool /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini >/dev/null
test -s /data/haozhen/Memory-clean/Nvida_api/Openrouter_api

git diff --exit-code -- baselines/MemVerse-main
git rev-parse HEAD:Offline/baselines/MemVerse-main
sha256sum /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json \
  /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
  configs/multimodal_split_manifest.json \
  data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunks_multiparty.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl

PYTHONPATH=src "$MEMVERSE_PYTHON" - <<'PY'
from pathlib import Path
from scripts.run_test_baseline_matrix import load_selection
s=load_selection(Path('configs/multimodal_split_manifest.json').resolve())
assert [len(s.question_ids_for_benchmark(x)) for x in ('Mem-Gallery','H2HMEM','WorldMemArena')]==[275,360,440]
print('test manifest counts OK')
PY
```

当前源码 tree 输出应记录为审计证据，但 `247d7836c568ffac6cd1832c43a26e089f6a69bd` 尚未在 registry 中绑定到经验证的官方版本；正式运行前必须与采用的 `upstream_tree` 比较，不能把未认证的当前 tree 自动视为官方 pin。

在当前 shell 安全加载 key；不要开启 `set -x`，不要 echo key，不要把 header 写进日志：

```bash
export OPENAI_API_KEY="$(tr -d '\r\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api)"
test -n "$OPENAI_API_KEY"

curl -fsS https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  | "$MEMVERSE_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="openai/gpt-5-mini" for x in d.get("data",[])); print("GPT-5-mini available")'
```

检查本地 embedding 及 2048 维：

```bash
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["dimension check"]}' \
  | "$MEMVERSE_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048; print("embedding dim=2048")'
```

如果 8001 不存在，确认一张空闲 GPU 后使用现有 server：

```bash
: "${EMBED_GPU:?设置一张已确认空闲的 embedding GPU 编号}"
tmux new-session -d -s memverse_api_embed_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=$EMBED_GPU \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u scripts/serve_embeddings.py \
   --host 127.0.0.1 --port 8001 \
   --model Qwen/Qwen3-VL-Embedding-2B --dim 2048 --device cuda:0 \
   --dtype bfloat16 --local-files-only \
   > outputs/embedding_8001.log 2>&1"
```


## 6. 三 worker tmux 正式启动

以下命令使用当前真实 CLI 和全新 Run-ID，直接启动正式实验：

```bash
cd /data/haozhen/Memory-clean/Offline
export RUN_ID="memverse_api_$(date +%Y%m%d_%H%M%S)"
export MEMVERSE_PYTHON=/data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python
unset MEMVERSE_REUSE_STATE
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s "memverse_api_${RUN_ID}" \
  "cd /data/haozhen/Memory-clean/Offline && \
   export MEMVERSE_PYTHON=/data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python && \
   export OPENAI_API_KEY=\$(tr -d '\\r\\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api) && \
   unset MEMVERSE_REUSE_STATE && \
   /data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json \
   --efficiency-config /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
   --split-manifest configs/multimodal_split_manifest.json \
   --baseline MemVerse \
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

三个相同 endpoint 参数表示三个独立 worker，不是三个 provider。命令中的 `--skip-smoke` 使 runner 直接进入正式任务。

持续出现 401、402、429、timeout 或 provider error 时必须停止并报告已发生调用、费用和失败率；不得静默降样本、换模型/provider、跳过问题或修改并发后当作同一 run。

## 7. 监控与恢复

```bash
tmux attach -t "memverse_api_${RUN_ID}"
tail -f "outputs/_runs/$RUN_ID/launcher.log"
cat "outputs/_runs/$RUN_ID/status.json"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/memverse__mem_gallery.log"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/memverse__h2hmem.log"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/memverse__worldmemarena.log"
```

检查 `status.json` 中三个 job 的 child PID、endpoint、result_dir 均独立。恢复必须使用相同 Run-ID、代码、API 模型/provider、reasoning effort、配置、manifest、chunk 和 endpoint。

首次新 Run-ID 必须 unset `MEMVERSE_REUSE_STATE`；同签名中断恢复才把启动命令中的 unset 改为 `export MEMVERSE_REUSE_STATE=1`。Harness 的 `--resume` 可恢复已完成 sample/QA；该环境变量允许 MemVerse adapter 保留未完成 sample 的 graph/state。绝不能从另一个 Run-ID 复制 state 或 checkpoint。

## 8. 输出与验收

三个独立正式目录：

```text
outputs/Mem-Gallery/MemVerse/<Run-ID>/
outputs/H2HMEM/MemVerse/<Run-ID>/
outputs/WorldMemArena/MemVerse/<Run-ID>/
```

核心输出包括 `results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_metrics.json`、`call_trace.jsonl`、`llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json`、`llm_judge_metrics.json`，以及 `.checkpoint/` 和 sample `call_traces/`。H2HMEM 另有 dyadic/multiparty prediction JSON。

```bash
PYTHONPATH=src "$MEMVERSE_PYTHON" - "$RUN_ID" <<'PY'
import json, sys
from pathlib import Path
run_id=sys.argv[1]
expected={'Mem-Gallery':275,'H2HMEM':360,'WorldMemArena':440}
for benchmark,n in expected.items():
    root=Path('outputs')/benchmark/'MemVerse'/run_id
    results=json.loads((root/'results.json').read_text())
    traces=[json.loads(x) for x in (root/'retrieval_trace.jsonl').read_text().splitlines() if x.strip()]
    pipeline=[json.loads(x) for x in (root/'pipeline_qa.jsonl').read_text().splitlines() if x.strip()]
    judge=json.loads((root/'llm_judge_metrics.json').read_text())
    manifest=json.loads((root/'run_manifest.json').read_text())
    assert len(results)==len(traces)==len(pipeline)==n
    assert judge['count']==n and judge['judge_errors']==0
    assert manifest['selection_mode']=='strict_manifest'
    assert manifest['split_manifest_sha256']=='590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36'
    assert not any(row.get('error') for row in results)
    print(benchmark, n, 'basic acceptance OK')
PY
```

此外必须验收：模型精确为 `openai/gpt-5-mini`、provider/OpenRouter、reasoning effort minimal、temperature 0、QA max 512；官方 MemVerse version/tree 落盘；chunk/prompt/config hash 匹配；Top-7 是真实 native seed 且 graph/source provenance 完整；不存在三条全库聚合 evidence；图片对应真实命中；WMA 按 checkpoint 答后再写且无未来信息；三个任务 state/trace/output 完全独立；Judge 计数一致且不计入效率指标。

固定汇总列为 `Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA)`。F1/EM/Judge 按 QA 平均；cost/latency 按 sample；calls 为 `(MB calls + answer calls) / sample`；retrieval trace 单独保存；Judge 排除。API cost/latency 必须使用指定 efficiency config，不能用实时账单反推或换系数。

## 9. 给 Codex 的快捷指令

> 严格遵循本教程，使用指定 GPT-5-mini defaults、efficiency config 和安全读取的 OpenRouter key，以新 Run-ID 和三个独立 worker 并行运行 275/360/440 QA；完成官方版本、原版检索、逐项 provenance/image、WMA 时序、API 调用、费用和 conformance 验收，不要静默 fallback。
