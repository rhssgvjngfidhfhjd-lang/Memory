# 0909 回答 Prompt 统一计划

## 1. 目标

Mem-Gallery、H2HMem 和 WorldMemArena 的 QA 严格采用仓库根目录 `answer_prompts.py` 中定义的自定义 Prompt，并分别通过各 harness 自己的 `prompts.py` 生成 system/user messages，不再保留或调用现有的官方及旧版回答 Prompt。

仅替换回答阶段 Prompt 和答案解析；Memory Bank、检索结果、LLM Judge Prompt、数据划分及指标口径保持不变。

## 2. Prompt 组织

- `/data/haozhen/Memory-clean/answer_prompts.py` 作为本次修改的对照标准，不作为正式运行时依赖。
- 将其中三个 benchmark 的具体 Prompt 分别写入现有 harness：
  - Mem-Gallery：`Offline/src/benchmarks/memgallery_harness/runner/prompts.py`
  - H2HMem：`Offline/src/benchmarks/h2hmem_harness/prompts.py`
  - WorldMemArena：`Offline/src/benchmarks/wma_harness/runner/prompts.py`
- 每个 harness 对外提供 `build_answer_messages()`、`parse_answer_response()` 和 `prompt_sha256()`，正式运行只从对应 harness 导入。
- 通用模块只负责严格解析 `<answer>...</answer>` 等非 Prompt 逻辑，不保存 benchmark 专属 Prompt 文本。
- 测试逐项比较三个 harness 的 messages 与根目录 `answer_prompts.py` 的输出，防止内容漂移。
- Prompt 哈希变化后，不得复用旧 QA 答案或回答缓存。

## 3. 回答流程

1. 复用已有 Memory Bank 和冻结的 retrieval。
2. 将最终检索项按原顺序转换为 `memory_evidence`，不改变 top-k 和图扩展结果。
3. 调用对应 harness 的 `build_answer_messages()` 生成完整 system/user messages。
4. 在不修改消息文本的前提下附加检索图片和问题图片。
5. 调用统一回答模型，并严格提取唯一且非空的 `<answer>...</answer>`；格式错误记为回答错误。

## 4. 代码修改

- 重写三个 harness 的 Prompt 文件：删除现有回答 Prompt，写入 `answer_prompts.py` 中对应 benchmark 的 system/user Prompt 构造逻辑及哈希。
- 新增共享答案解析模块：严格提取唯一且非空的 `<answer>...</answer>`，但不存放 Prompt 文本。
- 修改三个 harness 的回答函数，使其只调用各自的 Prompt 文件：
  - `Offline/src/benchmarks/memgallery_harness/eval_memgallery.py`
  - `Offline/src/benchmarks/h2hmem_harness/eval_h2hmem.py`
  - `Offline/src/benchmarks/wma_harness/eval_wma.py`
- 修改 `Offline/src/benchmarks/memgallery_harness/runner/answer_client.py`：支持发送预构造 messages，并保证附图时不插入额外文本。
- 修改 `Offline/scripts/evidence_policy.py` 和 `Offline/src/evidence_policy/rollout.py`：PPO 在选完 evidence 后调用对应 harness 的 Prompt 构造器。
- 修改 `Offline/scripts/rerun_qa_from_frozen_retrieval.py`：忽略旧产物中的 Prompt，仅复用 retrieval 并重新生成回答。
- 修改 `Offline/scripts/run_test_baseline_matrix.py`：分别校验三个 harness 的 Prompt 哈希。
- 更新对应测试和运行文档，移除旧 Prompt 入口说明。

## 5. 兼容与产物

- 不覆盖已有实验目录；新结果使用新的 Run-ID。
- 旧 Memory Bank、memory snapshot、retrieval trace 可以复用。
- 旧 QA 答案、旧 Prompt 字段和旧回答缓存不可复用。
- `results.json` 保存解析后的答案，并额外保留模型原始 `<answer>` 响应用于审计。

## 6. 验收标准

- 三个 benchmark 的实际请求 messages 与 `answer_prompts.py` 输出逐字一致。
- 真实图片仍正常发送，且不会改变 Prompt 文本。
- Mem-Gallery 的 CD/VS、H2HMem 各 question type、WMA 缺失信息规则均通过测试。
- QA replay 和 PPO 不再调用旧 Prompt；静态扫描无遗留运行入口。
- 全量单元测试及每个 benchmark 一题的端到端 smoke test 均通过后，才允许重跑 QA。
