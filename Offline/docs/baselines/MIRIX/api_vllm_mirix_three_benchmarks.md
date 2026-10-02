# API MIRIX 三 Benchmark Test Split 运行教程

## 1. 目标

使用修复后的 MIRIX v0.1.1，通过 OpenRouter API 分别完成以下三个 benchmark 的 test split：

- Mem-Gallery：275 QA
- H2HMEM：360 QA，包含 dyadic 和 multiparty
- WorldMemArena：仅运行 lifelong，共 440 QA

三个任务必须读取已经生成的固定 chunk JSONL，重新构建各自独立的 MIRIX memory，不得复用旧 memory、snapshot 或 retrieval trace。三个 benchmark 可以并行调用 API，但必须使用独立的进程、state 目录、checkpoint 和输出目录，不能共享可变 memory 状态。

本教程仅将本地 vLLM 建库/回答模型替换为指定的 OpenRouter API 模型；数据、chunk、split、Top-K、MIRIX 流程、QA prompt、Judge、指标和验收规则与 `本地vllm_mirix_three_benchmarks.md` 保持一致。

## 2. 固定配置

- 协议：`configs/test_baseline_matrix.json`
- API defaults：`/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json`
- API efficiency config：`/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini`
- API key 文件：`/data/haozhen/Memory-clean/Nvida_api/Openrouter_api`
- Split manifest：`configs/multimodal_split_manifest.json`
- Baseline：`MIRIX`
- MIRIX 上游版本：`v0.1.1`
- MIRIX commit：`ac0a1f2890df5e7435c66d6c2827f34c5c4ce32d`
- Mem-Gallery：275 QA
- H2HMEM：360 QA
- WorldMemArena lifelong：440 QA
- `top_k=7`
- 回答与建库模型：OpenRouter `openai/gpt-5-mini`
- API base URL：`https://openrouter.ai/api/v1`
- QA temperature：`0.0`
- QA reasoning effort：`minimal`
- QA max tokens：`512`
- MIRIX memory-build max tokens：`8192`
- Embedding：`Qwen/Qwen3-VL-Embedding-2B`
- Embedding 端口：`8001`
- Embedding 维度：`2048`
- 请求超时：`180` 秒
- 重试：`2` 次
- LLM Judge：OpenRouter `openai/gpt-4o-mini`
- Judge temperature：`0.0`
- Judge max tokens：`512`
- Judge 调用不计入 MB/QA calls、cost 或 latency

`defaults_gpt-5-mini.json` 中通用的 `executor_max_tokens` 为 512；MIRIX 正式建库由 runner 使用独立的 `mirix_executor_max_tokens=2048` 上限。实际值必须写入 `run_manifest.json`，QA 仍保持 512。MIRIX 建库触及 2048、工具 JSON 不完整或没有产生有效原生工具调用时，必须拒绝不完整内容并记为 baseline 自身的失败建库点；跳过后继续，只有连续 10 个建库点失败才隔离当前样本。

除上述 API 替换外，其余配置仍以 `configs/baselines.json`、`configs/test_baseline_matrix.json` 和 API defaults 为准。运行时不得静默修改固定配置。

API efficiency config 中 `openai/gpt-5-mini` 的固定计算参数为：

- 输入价格：0.25 USD / million tokens
- 输出价格：2.00 USD / million tokens
- latency base：3.47 seconds
- latency input coefficient：0.000003 seconds/token
- latency output coefficient：0.01754386 seconds/token
- latency image coefficient：0.3 seconds/image

## 3. 固定输入

三个 benchmark 必须通过 `src/benchmarks/fixed_chunks.py` 读取以下固定文件：

- Mem-Gallery：`data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl`
- H2HMEM dyadic：`data/h2hmem/chunks_dyadic.jsonl`
- H2HMEM multiparty：`data/h2hmem/chunks_multiparty.jsonl`
- WMA lifelong：`data/wma_qwen3_vl_embedding_2b/chunks_lifelong.jsonl`

要求：

1. 根据 test manifest 精确筛选 sample 和 question ID。
2. 禁止运行时重新切分、改写或静默重建 chunk。
3. 将 chunk 文件绝对路径、SHA256、格式和 chunk 数量写入 `run_manifest.json`。
4. 每次正式实验使用新的 Run-ID，从固定 chunk 重新建库。
5. 中断恢复只能恢复同一 Run-ID、同一 manifest、同一 chunk SHA256、同一 API 模型和同一配置的任务。

## 4. MIRIX 流程约束

### 4.1 Memory 构建

1. 使用干净的 MIRIX v0.1.1 上游代码。
2. 每个固定 chunk 进入 MIRIX 原生 Temporary Message Accumulator。
3. 按原生 20-chunk absorption boundary 触发 Meta Memory Agent。
4. 由 Meta Memory Agent 调用原生 Memory Agents，管理 Core、Episodic、Semantic、Procedural、Resource 和 Knowledge Vault 六层记忆。
5. 某个 memory bank 没有产出可以是 Agent 的正常决策，不得强制补写 Semantic Memory。
6. 禁止 direct insert、semantic fallback 和绕过 Memory Agent 的人工写库。
7. API 并行仅发生在不同 benchmark/sample 的独立 MIRIX 实例之间；同一样本内部的 chunk 顺序不得并行或重排。

### 4.2 检索与回答

1. Core Memory 常驻 Chat Agent 上下文，不参与 Top-K 计数。
2. Episodic、Semantic、Procedural、Resource 和 Knowledge Vault 为五个可检索 memory bank。
3. 原生 Chat Agent 看到 query 后自主调用 MIRIX 检索工具。
4. 每道 QA 中，最终提供给 Chat Agent 的五库 memory 去重后合计不得超过 7 条。
5. Top-7 必须同时覆盖显式 tool search 和 system prompt automatic prefetch；不得把非 Core prefetch 排除在 Top-7 之外。
6. 如果自动预载 memory 与工具检索 memory 的并集合超过 7 条，必须停止正式任务并报告，不能只在 trace 中标记后继续运行。
7. 正式回答必须由 MIRIX 原生 Chat Agent 生成；不得绕过 Chat Agent 调用统一回答 client。
8. Chat Agent 使用各 benchmark 代码中现有的 QA prompt；不得改写 MIRIX 内部 Agent prompt 或 benchmark QA prompt。
9. 每道 QA 后恢复 Chat Agent 的消息历史和 topic，防止不同问题之间泄漏。

### 4.3 WMA checkpoint

WMA 必须按 checkpoint 顺序执行：

1. 只写入截至当前 checkpoint 可见的 session/chunk。
2. 完成当前 checkpoint 的检索和回答。
3. 当前 QA 结束后，才能继续写入未来 session。
4. `visible_sessions`、retrieved memory provenance 和 gold evidence 均不得包含未来信息。

## 5. 正式启动前检查

进入工作目录：

```bash
cd /data/haozhen/Memory-clean/Offline
```

准备并校验 MIRIX v0.1.1：

```bash
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python \
  scripts/prepare_mirix_v011.py
```

检查两个 API 配置文件存在且 JSON 有效；不得输出 API key：

```bash
/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json >/dev/null

/data/haozhen/miniconda3/envs/pipeline_repro/bin/python -m json.tool \
  /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini >/dev/null
```

安全地加载 OpenRouter key：

```bash
export OPENAI_API_KEY="$(tr -d '\r\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api)"
test -n "$OPENAI_API_KEY"
```

不得使用 `echo`、shell tracing 或日志打印 API key。

检查 OpenRouter 模型可用，但不要打印请求头：

```bash
curl -fsS https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  | /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -c \
    'import json,sys; d=json.load(sys.stdin); assert any(x.get("id")=="openai/gpt-5-mini" for x in d.get("data", []))'
```

Embedding 仍使用本地服务，必须检查端口 8001 和输出维度 2048：

```bash
curl -fsS http://127.0.0.1:8001/v1/models
```

API 建库和回答不需要本地 vLLM GPU；只有本地 Embedding 服务需要相应 GPU 资源。


## 6. 三个 Benchmark 并行运行

选择一个全新的 Run-ID，例如 `0913_mirix_gpt5mini_api_v1`。向 matrix runner 重复传入三次同一个 OpenRouter endpoint，创建三个独立 worker；MIRIX 的三个 benchmark 会分别被一个 worker 领取并并行运行。

```bash
RUN_ID=0913_mirix_gpt5mini_api_v1
mkdir -p "outputs/_runs/$RUN_ID"

tmux new-session -d -s mirix_gpt5mini_api \
  "cd /data/haozhen/Memory-clean/Offline && \
   export OPENAI_API_KEY=\$(tr -d '\\r\\n' < /data/haozhen/Memory-clean/Nvida_api/Openrouter_api) && \
   /data/haozhen/miniconda3/envs/pipeline_repro/bin/python -u \
   scripts/run_test_baseline_matrix.py \
   --defaults /data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json \
   --efficiency-config /data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini \
   --baseline MIRIX \
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

三个 endpoint 参数代表三个并行 worker，不代表三套不同 API。每个 benchmark 的 memory database 和 state 必须保存在自己的 result directory 中。

该入口应完成服务预检、固定 chunk 重新建库、检索、QA、输出验证和 LLM Judge，并通过 `--skip-smoke` 直接进入正式任务。

如果 API 出现持续 429、provider timeout 或吞吐明显下降，应停止并报告真实并发、错误率和请求耗时；不得静默降低并发、改模型或跳过失败样本。

## 7. 监控与断点恢复

查看 tmux：

```bash
tmux attach -t mirix_gpt5mini_api
```

查看日志和状态：

```bash
tail -f outputs/_runs/<Run-ID>/launcher.log
cat outputs/_runs/<Run-ID>/status.json
```

检查三个正式任务分别拥有不同的 PID、result directory 和 baseline state directory。不得仅凭主进程存在判断三个任务均在推进。

中断后，使用完全相同的 Run-ID、参数、API 模型、manifest 和 chunk 文件重新执行第 6 节命令。Runner 只能恢复已验证兼容的 checkpoint；配置或输入 hash 不一致时必须拒绝恢复。

如果要从头重跑，不得覆盖旧目录，应使用新的 Run-ID。

## 8. Outputs

正式结果分别保存在：

```text
/data/haozhen/Memory-clean/Offline/outputs/Mem-Gallery/MIRIX/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/H2HMEM/MIRIX/<Run-ID>/
/data/haozhen/Memory-clean/Offline/outputs/WorldMemArena/MIRIX/<Run-ID>/
```

每个目录必须包含：

- `results.json`
- `retrieval_trace.jsonl`
- `pipeline_qa.jsonl`
- `memory/memory_snapshot.jsonl`
- `run_manifest.json`
- `metrics.json`
- `efficiency_metrics.json`
- `call_trace.jsonl`
- `llm_judge_results.jsonl`
- `llm_judge_metrics.json`
- 运行日志和 checkpoint 状态

三个 benchmark 并行时仍然不得复用或覆盖任何上述文件。

## 9. 完成验收

三个任务分别检查：

1. QA 数严格为 275、360、440。
2. question ID、顺序和 sample 集与 test manifest 完全一致。
3. `run_manifest.json` 中的 split、manifest SHA256、chunk path/SHA256、API 模型、prompt SHA256、Top-K 和 MIRIX commit 正确。
4. 所有固定 chunk 都经过原生 MIRIX memory 构建链；无 direct insert 或 semantic fallback。
5. `memory_snapshot.jsonl` 能区分六层 memory；允许 Agent 合理地让某一 bank 为空。
6. 每道 QA 的检索来自原生 Chat Agent 工具，最终五库 memory 并集不超过 7，Core 单独常驻。
7. 最终回答来自原生 MIRIX Chat Agent；无统一回答 client fallback。
8. WMA 每道题只使用 checkpoint 当时可见的历史，无未来 session 泄漏。
9. 无 answer error、重复题、未解析 tool call、API 429 遗留或未完成 Judge。
10. Judge 数量与 QA 数一致，且 Judge 调用未计入效率指标。
11. 三个任务的 memory state、checkpoint、trace 和结果目录相互独立。

## 10. Metric

结果统一按以下字段及顺序汇总，不增删或改名：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- `F1`、`EM`、`Judge`：test QA 的 qa-wise average。
- `Cost-MB`、`Lat-MB`：memory 构建阶段的 USD/sample 与 seconds/sample。
- `Cost-QA`、`Lat-QA`：检索及回答阶段合计的 USD/sample 与 seconds/sample。
- `#Calls (MB+QA)`：按 `(MB calls + answer calls) / sample 数` 汇报。
- Retrieval calls 单独保存在 call trace 中，不计入 `#Calls (MB+QA)`。
- Judge calls 不计入 calls、cost 或 latency。
- Cost 和 latency 必须使用 `/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini` 中 `openai/gpt-5-mini` 的系数计算。

## 11. 必须停止并报告的情况

出现以下任一情况时，不得自行改变算法后继续：

- 固定 chunk 文件或 manifest 缺失、hash 改变；
- MIRIX 上游 commit 不一致或源码不干净；
- system prompt automatic prefetch 与显式检索的五库 memory 并集超过 7；
- 原生 Chat Agent、Meta Memory Agent 或 Memory Agent 被绕过；
- 出现 direct insert、semantic fallback 或统一回答 client fallback；
- WMA 检索到未来 session；
- QA 数、question ID、prompt hash 或 Judge 数不匹配；
- OpenRouter API key、模型、provider 或计价配置不一致；
- API 请求持续出现 401、402、429、timeout 或 provider error；
- tool call 被截断、重复循环或无法解析；
- 需要修改原版 Agent prompt、benchmark QA prompt 或实验固定配置。

遇到上述情况，Codex 应先提供文件位置、真实调用链、日志证据、已产生的 API cost 和建议方案，获得用户确认后再修改或继续付费运行。

## 12. 给 Codex 的快捷指令

以后可以直接发送：

> 严格遵循本教程，使用指定 defaults 和 efficiency config，通过 OpenRouter `openai/gpt-5-mini` 以新 Run-ID 并行运行 MIRIX 在 Mem-Gallery、H2HMEM 和 WMA lifelong 的 test split。核对固定 chunk、原生六层建库、原生 Chat Agent 和严格全局 Top-7，并完成监控与结果验收。
