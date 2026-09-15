# MemVerse 本地 vLLM 三 Benchmark Test Split 运行教程


## 1. 实验目标与固定配置

使用新的 Run-ID 分别完成：

- Mem-Gallery test：275 QA；
- H2HMEM test：dyadic + multiparty，共 360 QA；
- WorldMemArena：只跑 lifelong test，共 440 checkpoint QA。

固定配置：

- baseline 名：`MemVerse`；adapter：`memverse`；
- protocol：`/data/haozhen/Memory-clean/Offline/configs/test_baseline_matrix.json`；
- split manifest：`/data/haozhen/Memory-clean/Offline/configs/multimodal_split_manifest.json`；
- 建库和回答：`Qwen/Qwen3-VL-4B-Instruct`；
- embedding：`Qwen/Qwen3-VL-Embedding-2B`，`http://127.0.0.1:8001/v1`，2048 维；
- 回答端口：8013、8014、8015；可分别放在 GPU3、GPU4、GPU5；
- `temperature=0`，QA `max_tokens=512`，timeout 180 秒，retries 2；
- `top_k=7`；
- judge：OpenRouter `openai/gpt-4o-mini`，judge 调用不计入 MB/QA cost、latency 或 calls。

每个 benchmark 必须是独立 harness 子进程、独立 result/state/checkpoint 目录。新 Run-ID 必须重新建 MemVerse memory，不得复制旧 Run-ID 的 `memory/`、`.checkpoint/`、snapshot 或 retrieval trace。

## 2. 固定输入、数量与已核对 hash

MemVerse 当前落在 harness 的通用 fixed-chunk 分支，由 `src/benchmarks/fixed_chunks.py` 读取：

| 输入 | 行数 | SHA256 |
|---|---:|---|
| `data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl` | 3962 | `93e1d0b778addfbf4d8df8c387c588a1a5333f19864c41b6436e54e9dd0e189e` |
| `data/h2hmem/chunks_dyadic.jsonl` | 2645 | `e046039299b40b0458a7b6e88f482c4081664a41aaf10f3861d2aff511e821b9` |
| `data/h2hmem/chunks_multiparty.jsonl` | 866 | `75ebf39c87ce60ea98313270754c0fdb46bdf5458491f8138e69313999113e51` |
| `data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl` | 15437 | `75e89446f6d991b4a2e5b471ab754d2d7f8942615badc73930c4385d153fe748` |

Manifest SHA256 是 `590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36`。当前 QA prompt hash 是：

- Mem-Gallery：`ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff`；
- H2HMEM：`4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f`；
- WorldMemArena：`6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250`。

禁止重新切分、改写或缺失时静默重建 JSONL。Harness 只可按 test manifest 筛选 sample 和 question ID；`run_manifest.json` 必须记录 chunk path、格式、总行数和 SHA256。

## 3. MemVerse 原版流程与 Top-7 定义

### 3.1 原版建库链

以当前纳入仓库的 MemVerse 源码为证据，原版调用链是：

1. `app.py` 的 `/insert` 调用 `orchestrator.handle_insert()`。
2. `handle_insert()` 为文本建立 entry；图片、视频、音频分别通过 `process_image`、`process_video`、`process_audio` 生成 caption，并写入 conversation JSON。
3. `orchestrator.update_long_term_memory()` 调用 `MemoryKB.build_memory.process_memory()`。
4. `process_memory()` 保持原版三个 system prompt，分别生成 core、episodic、semantic memory；每行含 `id/timestamp/input_text/output_text/embedding`，写入三个 JSONL。
5. `insert_chunks_from_json()` 把三个 memory 的 `output_text` 分别交给三个 LightRAG 实例，完成实体/关系抽取、source chunk 与图/向量索引持久化。

当前 benchmark bridge 是“一条固定 chunk → 一个 entry → 三种 memory summary → session 边界批量写三个图”。这保留了三个 Memory Manager prompt，但用预生成 image caption 代替原版图片处理，并延迟到 session 边界写图；这两点必须在修复验收里明确批准或恢复。

### 3.2 原版检索与回答链

源码中的 `/query` 调用 `orchestrator.handle_query()`：可选 parametric memory → relevance 判断；未启用或不相关时调用 `rag_retrieve()` → `mem_core.aquery(mode="hybrid")` → LightRAG 关键词抽取 → local 实体检索、global 关系检索及图邻接/source chunk 回溯 → LightRAG RAG response；最后 `generate_final_answer()` 生成用户答案。

Benchmark 协议要求外层回答使用各 harness 已有 QA prompt，因此合规 bridge 应把真实原生检索上下文逐项交给当前 benchmark prompt；不得修改三个 MemVerse Memory Manager prompt，也不得把 LightRAG 自己先生成的答案伪装成逐条 retrieved memory。是否保留原版 `generate_final_answer()` 的第二次回答必须在正式运行配置中明确决定，不能同时回答两次却把成本或 provenance 隐藏。

### 3.3 Top-7 的准确含义

原版 LightRAG `QueryParam.top_k` 的代码定义是：local 模式限制相似实体 seed 数，global 模式限制相似关系 seed 数；hybrid 同时运行两路。因此本实验的原生定义应是：

```text
mem_core.aquery(query, QueryParam(mode="hybrid", top_k=7, ...))
                         ├─ entity vector seeds：最多 7
                         └─ relationship vector seeds：最多 7
                            → graph traversal / source chunks / token budget
```

Top-7 不是“core/episodic/semantic 三个聚合字符串取前 7”，也不是从全库收集 provenance 后截断。图展开后的 entity/relation/source chunk 数可能大于 7，所以 trace 必须分别记录原生 seed 排名、图展开来源、最终交给 QA 的 source items，并明确最终 evidence cap 如何满足统一实验规范。若统一规范要求最终 evidence 总数也严格不超过 7，这与上游 `QueryParam.top_k` 的语义不是同一件事，必须先取得确认并在兼容层显式实现，不能悄悄改变原版。

当前 adapter 把上述 `top_k=7` 独立施加到三个 store，随后最多返回三条聚合字符串；正式结果应在运行记录和验收报告中如实标注这一实现差异。

### 3.4 Fallback 规则

允许：原版显式的“parametric memory 不相关时转 long-term RAG”（仅在实验明确启用 parametric memory 时）；原版最终 prompt 在 memory 不足时正常回答；HTTP 同一模型最多重试 2 次；同一签名同一 Run-ID 的 checkpoint 恢复。

禁止：检索或 embedding 失败后换模型/换 provider/换 store；用全库或三类聚合结果替代真实命中；用统一 cosine、旧 snapshot 或旧 trace 代替 LightRAG；丢失图片或伪造图片 provenance；跳过失败题；重建 chunk；对 MemVerse 内部 prompt 做改写；吞掉 build/retrieval 异常后继续计分。

## 4. 已核对 runner 入口与 CLI

正式总入口是：

```text
scripts/run_test_baseline_matrix.py
```

实际 `--help` 已核对的参数为：`--endpoint`（可重复）、`--embedding-base-url`、`--defaults`、`--efficiency-config`、`--split-manifest`、`--output-root`、`--run-id`、`--baseline`（可重复）、`--benchmark`（可重复）和 `--skip-smoke`。

它实际调用：

- `python -m benchmarks.memgallery_harness.eval_memgallery`，并传 `--all-datasets`；
- `python -m benchmarks.h2hmem_harness.eval_h2hmem`，默认 `--variant all`；
- `python -m benchmarks.wma_harness.eval_wma`，数据 staging 只链接 lifelong test sample。

三者共同接受本实验需要的 `--baseline-state-dir/--result-dir/--split-manifest/--split/--top-k/--sample-concurrency/--answer-concurrency/--checkpoint-every/--resume`、answer/executor/embedding 模型参数及 timeout/retries。不要使用教程自行臆造的参数。

## 5. 环境与不可变输入检查

```bash
cd /data/haozhen/Memory-clean/Offline
export MEMVERSE_PYTHON=/data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python

test -x "$MEMVERSE_PYTHON"
"$MEMVERSE_PYTHON" -c 'import openai,numpy,tiktoken,nano_vectordb,networkx,json_repair; print("MemVerse env OK")'
"$MEMVERSE_PYTHON" scripts/run_test_baseline_matrix.py --help >/dev/null

git diff --exit-code -- baselines/MemVerse-main
git rev-parse HEAD:Offline/baselines/MemVerse-main

sha256sum configs/multimodal_split_manifest.json \
  data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunks_multiparty.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl
wc -l data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl \
  data/h2hmem/chunks_dyadic.jsonl \
  data/h2hmem/chunks_multiparty.jsonl \
  data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl

PYTHONPATH=src "$MEMVERSE_PYTHON" - <<'PY'
from pathlib import Path
from scripts.run_test_baseline_matrix import load_selection
s = load_selection(Path('configs/multimodal_split_manifest.json').resolve())
print({b: len(s.question_ids_for_benchmark(b)) for b in ('Mem-Gallery','H2HMEM','WorldMemArena')})
PY
```

数量检查必须输出 `{'Mem-Gallery': 275, 'H2HMEM': 360, 'WorldMemArena': 440}`。当前 `git rev-parse` 输出是 `247d7836c568ffac6cd1832c43a26e089f6a69bd`，但它尚未被 registry 固定为已验证的官方 tree；正式运行前应把实际采用的 pin 写入 registry/run manifest，再与该 pin 比较，不能永远硬编码当前未认证 tree。Git object path 始终相对仓库根，所以即使 shell 位于 `Offline/` 也使用 `HEAD:Offline/baselines/MemVerse-main`。

## 6. 启动和检查服务

先检查已有服务，不得终止不属于本任务的进程：

```bash
nvidia-smi -i 3,4,5
for port in 8013 8014 8015; do curl -fsS "http://127.0.0.1:${port}/v1/models"; done
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-VL-Embedding-2B","input":["dimension check"]}' \
  | "$MEMVERSE_PYTHON" -c 'import json,sys; d=json.load(sys.stdin); assert len(d["data"][0]["embedding"])==2048; print("embedding dim=2048")'
```

如果 8001 未启动，先确认一张额外空闲 GPU，再使用仓库已有 server；不要与回答 GPU 抢显存：

```bash
: "${EMBED_GPU:?设置一张已确认空闲的 embedding GPU 编号}"
tmux new-session -d -s memverse_embed_8001 \
  "cd /data/haozhen/Memory-clean/Offline && \
   CUDA_VISIBLE_DEVICES=$EMBED_GPU \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u scripts/serve_embeddings.py \
   --host 127.0.0.1 --port 8001 \
   --model Qwen/Qwen3-VL-Embedding-2B --dim 2048 --device cuda:0 \
   --dtype bfloat16 --local-files-only \
   > outputs/embedding_8001.log 2>&1"
```

缺少回答服务时，用已核对的 `scripts/serve_vllm.sh` 启动三份：

```bash
for spec in 3:8013 4:8014 5:8015; do
  gpu=${spec%:*}; port=${spec#*:}
  tmux new-session -d -s "memverse_vllm_${port}" \
    "cd /data/haozhen/Memory-clean/Offline && \
     PATH=/data/haozhen/miniconda3/envs/vllm_repro/bin:\$PATH \
     GPUS=$gpu PORT=$port \
     MODEL=/data/shared_models/Qwen3-VL-4B-Instruct \
     SERVED_NAME=Qwen/Qwen3-VL-4B-Instruct \
     MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 GPU_MEMORY_UTILIZATION=0.90 \
     sh scripts/serve_vllm.sh > outputs/vllm_${port}.log 2>&1"
done
```

服务就绪后再次验证三个 `/v1/models` 都含精确 served name。MemVerse 当前 adapter 还会把 executor 输出上限强制限制为 512；不得让服务默认值替代实验配置。


## 7. tmux 正式启动

以下命令与当前 CLI 一致，使用全新 Run-ID 直接启动正式实验：

```bash
cd /data/haozhen/Memory-clean/Offline
export RUN_ID="memverse_local_$(date +%Y%m%d_%H%M%S)"
export MEMVERSE_PYTHON=/data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python
unset MEMVERSE_REUSE_STATE
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s "memverse_local_${RUN_ID}" \
  "cd /data/haozhen/Memory-clean/Offline && \
   export MEMVERSE_PYTHON=/data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python && \
   unset MEMVERSE_REUSE_STATE && \
   /data/haozhen/Memory-clean/Offline/.venvs/memverse/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults configs/defaults.json \
   --efficiency-config configs/model_efficiency.json \
   --split-manifest configs/multimodal_split_manifest.json \
   --baseline MemVerse \
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

三个 endpoint 创建三个 worker，三个 benchmark 可并行；每个 harness 的 state 自动位于自己的 `result-dir/memory/datasets`。命令中的 `--skip-smoke` 使 runner 直接进入正式任务。

## 8. 监控与断点恢复

```bash
tmux attach -t "memverse_local_${RUN_ID}"
tail -f "outputs/_runs/$RUN_ID/launcher.log"
cat "outputs/_runs/$RUN_ID/status.json"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/memverse__mem_gallery.log"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/memverse__h2hmem.log"
tail -f "outputs/_runs/$RUN_ID/_logs/baseline/memverse__worldmemarena.log"
```

同一 Run-ID 恢复时必须复用完全相同的代码、服务、参数、manifest 与 chunk hash。Harness 总是传 `--resume`；MemVerse adapter 只有在 `MEMVERSE_REUSE_STATE=1` 时才保留未完成的本地 graph/state，否则 `reset()` 会删除该 sample state。因此：首次新 Run-ID 必须 unset；确认签名一致的中断恢复才把 tmux 命令里的 `unset MEMVERSE_REUSE_STATE` 改成 `export MEMVERSE_REUSE_STATE=1`。不得对新 Run-ID 设置复用，也不得从其他 Run-ID 搬 state。

## 9. 输出与完成验收

正式目录：

```text
outputs/Mem-Gallery/MemVerse/<Run-ID>/
outputs/H2HMEM/MemVerse/<Run-ID>/
outputs/WorldMemArena/MemVerse/<Run-ID>/
```

共享核心文件包括 `results.json`、`retrieval_trace.jsonl`、`pipeline_qa.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json`、`efficiency_metrics.json`、`call_metrics.json`、`call_trace.jsonl`、`llm_judge_results.json`、`llm_judge_progress.jsonl`、`llm_judge_checkpoint.json` 和 `llm_judge_metrics.json`；另有 `.checkpoint/`、sample `call_traces/`，H2HMEM 的两个 prediction JSON，以及部分 harness 的 `memory_metrics.json`。

基础数量验收：

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
    pipe=[json.loads(x) for x in (root/'pipeline_qa.jsonl').read_text().splitlines() if x.strip()]
    judge=json.loads((root/'llm_judge_metrics.json').read_text())
    manifest=json.loads((root/'run_manifest.json').read_text())
    assert len(results)==len(traces)==len(pipe)==n
    assert judge['count']==n and judge['judge_errors']==0
    assert manifest['selection_mode']=='strict_manifest'
    assert manifest['split_manifest_sha256']=='590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36'
    assert not any(row.get('error') for row in results)
    print(benchmark, n, 'basic acceptance OK')
PY
```

还必须人工/程序化验收：run manifest 固定官方版本；问题 ID 与 manifest 顺序完全一致；Top-7 是真实 LightRAG seed 并有图展开/source provenance；最终 evidence 不含全库聚合来源；图片只关联真实命中；WMA 每个 checkpoint 先答再写未来且无未来 session；三类 Memory Manager prompt 未改；MB/QA/Judge cost 与调用分类正确。任一不满足则该 Run-ID 作废。

统一汇总列固定为：`Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA)`。F1/EM/Judge 按 QA 平均；MB、QA cost/latency 按 sample；`#Calls` 为 `(MB calls + answer calls) / sample`；retrieval trace 单列保存，Judge 不计入。

## 10. 给 Codex 的快捷指令

> 严格遵循本教程，核对官方版本、原版检索、逐项 provenance/image 和 WMA 时序，使用新 Run-ID、GPU3/4/5 与 8013/8014/8015 在 tmux 直接并行运行三个 test split；完成 conformance 和结果验收，不要静默 fallback 或自行改协议。
