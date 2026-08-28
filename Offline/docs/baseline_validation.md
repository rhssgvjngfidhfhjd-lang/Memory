# Baseline 接入验证记录

日期：2026-08-22

最近回归：2026-08-28，`117 passed, 2 skipped`，并通过 `compileall`。

## H2HMem chunk 生成

已按“连续同 speaker 先合并、相邻两个 speaker block 再配对”的规则生成：

- `data/h2hmem/chunks_dyadic.jsonl`：2,645 chunks，283 sessions；
- `data/h2hmem/chunks_multiparty.jsonl`：866 chunks，25 sessions。

校验覆盖原始 7,076 条非空发言和 1,265 张图片，无消息或图片遗漏，所有图片
路径存在，chunk ID 唯一；输出不包含 `questions`、`original_answer` 或
`answer_session` 等 QA/答案字段。

WorldMemArena 的当前运行范围已收敛为 `WorldMemArena/lifelong`；已删除的
`agent` 数据不会再被默认扫描。失去数据来源的 `--small` 选项同时移除，指定
少量样本继续使用可重复传入的 `--sample-id`。

## 统一输出布局回归

原生 baseline 的默认 memory state 已统一为：

```text
outputs/<benchmark>/<baseline>/memory/datasets/<sample>/
```

Mem-Gallery 和 WorldMemArena 的规范化 memory snapshot 均写入：

```text
outputs/<benchmark>/<baseline>/memory/memory_snapshot.jsonl
```

结果、指标、检索轨迹和 manifest 仍直接写在 baseline 根目录。HiveMem 保持
`<index-root>/datasets/<sample>` 原生索引布局；推荐将 `--index-root` 指向对应
baseline 目录下的 `memory/`。

## 自动化验证

```powershell
$env:PYTHONPATH = "src"
C:\Users\USER\anaconda3\python.exe -m compileall -q src tests
C:\Users\USER\anaconda3\python.exe -m pytest -q
```

结果：`94 passed, 1 skipped`。跳过项是项目原有条件跳过项。

新增测试覆盖：

- registry 名称和 alias；
- 统一协议到 answer context 的 provenance 转换；
- Mem-Gallery native baseline 的 reset/ingest/end-session/retrieve/snapshot/close；
- WMA checkpoint 只写入可见 session，避免 future leakage。
- 孤立 Unicode surrogate 在进入 JSON RPC 和 embedding 请求前会被安全处理。

## 两个 benchmark 的真实 CLI 冒烟

生成与回答端点使用本机 SSH 转发的真实服务
`http://127.0.0.1:18001/v1`，模型为 `Qwen/Qwen3-VL-4B-Instruct`；embedding
端点使用临时 8 维确定性 OpenAI-compatible 服务。随后分别用真实命令行入口
运行一条 Mem-Gallery 样例和一条 WMA 样例，baseline 使用真实
`M3-Agent-caption` 子进程 adapter。

结果：

- Mem-Gallery：真实数据集 `Dog_Behavior_Research_Academic_Life`，完成 1 个问题，
  无 answer error，生成 `results.json` 与 `retrieval_trace.jsonl`；
- WMA：真实样例 `mobile_01`，完成 1 个问题，无 answer error，生成
  `results.json`、`retrieval_trace.jsonl` 与 `pipeline_qa.jsonl`；
- answer 与 embedding 确实使用两个不同端口；
- 两边最终答案都通过统一 answer client 产生。

成功输出位于：
`outputs/validation_m3_real_llm_20260822_134706/`。

本轮两条样例的 F1、EM 和 retrieval hitrate 均为 0。这里仅证明数据解析、
baseline 写入/检索、统一回答和结果落盘链路可运行；临时 embedding 不具备语义
检索能力，因此这些数值不能作为 baseline 质量结论。

## Native adapter 冒烟

使用临时统一模型端点，已实际执行以下 adapter 的 reset、ingest、retrieve 和
snapshot：

| Baseline | 结果 | 说明 |
| --- | --- | --- |
| AUGUSTUSMemory | 通过 | 使用 18001 真实 LLM；共享 multimodal embedding endpoint |
| OmniSimpleMem | 通过 | 使用 18001 真实 LLM；text/image 使用共享远程 embedding 配置 |
| M3-Agent-caption | 通过 | 在两个真实 benchmark CLI 中完成独立 worker 与 VideoGraph 检索 |
| M2A | 待隔离依赖 | 当前解释器缺少 `langchain_openai` |
| MIRIX | 待隔离依赖 | 当前解释器缺少 `demjson3` |
| MMA | 待隔离依赖 | 已修复大写源码目录与 `mma` 包名不匹配；下一缺失项为 `demjson3` |
| MemVerse | 待隔离依赖 | 当前解释器缺少 `pipmaster` |

`MGMemory` 已于 2026-08-28 从可运行 baseline 注册表移除；其独立源码目录此前
已经不存在。AUGUSTUSMemory vendored 的 MemEngine 上游包仍包含同名内部类和
操作实现，为保持上游包结构完整而保留，但不会作为实验方法暴露。

待隔离依赖的项目已完成 adapter、配置映射和静态编译；完整 native 运行需要先
按 `baselines/README.md` 安装各自隔离依赖并设置对应 `*_PYTHON` 变量。它们不应
被装入 HiveMem 根环境，因为不同上游项目的 Python/OpenAI 版本约束可能冲突。

## 当前本机模型端口

检查时的状态：

- `18000/v1/models` 与 `18001/v1/models` 均可访问，模型都是
  `Qwen/Qwen3-VL-4B-Instruct`；
- 两个端口都是 SSH 转发，18000 有其他程序的历史请求，故本次选用 18001；
- 测试结束后 18001 的 vLLM 指标为 running=0、waiting=0；
- 未发现可用的真实 embedding 服务。

因此本次是使用真实生成模型的端到端运行验证，但不是正式质量评测。正式
benchmark 还需要让 `configs/defaults.json` 中的 embedding endpoint 提供匹配的
语义 embedding 模型，并为剩余四个 baseline 准备隔离环境。
