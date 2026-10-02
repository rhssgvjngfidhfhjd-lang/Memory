# Baseline Test Matrix Plan

## 1.目标

在 GPU 3、4、5 上完成 7 个非 HiveMem baseline 的 test split 实验。三个 GPU 各运行一个相同配置的 vLLM worker，任务采用 shortest-job-first 队列调度并支持断点续跑。

## 2.固定配置

- 协议：`configs/test_baseline_matrix.json`
- Baseline：M3-Agent-caption、M2A、AUGUSTUSMemory、MMA、MIRIX、OmniSimpleMem、MemVerse
- Mem-Gallery：275 QA
- H2HMEM：360 QA
- WorldMemArena lifelong：440 QA
- `top_k=7`
- 回答与建库模型：`Qwen/Qwen3-VL-4B-Instruct`
- Embedding：`Qwen/Qwen3-VL-Embedding-2B`，端口 8001，维度 2048
- 回答端口：8013、8014、8015
- LLM Judge：OpenRouter `openai/gpt-4o-mini`

## 3.执行

1. 在 GPU 3、4、5 启动 vLLM，并完成服务预检。
2. 运行 smoke test；通过后启动 21 组正式实验。
3. 按 `run_test_baseline_matrix.py` 中的 `JOB_ORDER` 从预计耗时短到长分配任务。
4. 三个 benchmark 均严格读取当前已有的固定 chunk JSONL，并依据 test manifest 筛选样本，禁止运行时重新切分或静默重建 chunk
5.若有多agent，运行原生 Chat Agent 自主检索，保证每道 QA 最终返回的 memory 总数不超过 top_k=7
6. 记录 F1、EM、Judge、MB/QA calls、cost、latency，以及运行配置和断点状态。

## 4.完成条件

- 每组 QA 数严格符合 275/360/440，且 question ID 与 test manifest 完全一致。
- 每组均生成 `results.json`、`retrieval_trace.jsonl`、`memory/memory_snapshot.jsonl`、`run_manifest.json`、`metrics.json` 和调用追踪。
- 无回答错误、重复题目、超出 `top_k=7` 的检索结果或未完成 Judge。

## 5. Metric

所有实验结果统一按以下字段及顺序汇总，不增删或改名：

| Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

- `Cost-MB`、`Lat-MB`：memory 构建阶段的平均成本（USD/sample）与平均估算延迟（seconds/sample）。
- `Cost-QA`：检索及回答阶段合计的平均成本（USD/sample）。
- `Lat-QA`：检索及回答阶段合计的平均估算延迟（seconds/QA）；分子汇总 retrieval+answer 的调用、输入/输出 token 和输入图片延迟，分母固定为 QA 总数，不能使用 sample 数。
- `#Calls (MB+QA)`：沿用统一报表口径，按 `(MB calls + answer calls) / sample 数` 汇报；retrieval calls 单独保存在调用追踪中，不计入该列。

## 6. Outputs 文件保存规范

每组实验使用独立运行目录，统一保存为：

`/data/haozhen/Memory-clean/Offline/outputs/<Benchmark>/<Baseline>/<Run-ID>/`

举例：本次 Prompt 重跑的 `Run-ID` 为 `0909_custom_prompt_qa`。例如：

`/data/haozhen/Memory-clean/Offline/outputs/Mem-Gallery/M2A/0909_custom_prompt_qa/`

不同 benchmark、baseline 和运行批次之间不得复用或覆盖输出目录。
