# PPO 加入 QA Cost Reward 实施计划

## 1. 目标与边界

在现有 PPO 质量奖励上增加回答阶段成本惩罚，使 policy 在准确率与 evidence 成本之间学习取舍：

\[
R_i=F1_i-\lambda\alpha_t^{scale}C_{norm,i}
\]

- 成本范围仅为 QA 最终回答 LLM；包含该请求的实际重试用量。
- 不包含 memory bank 构建、embedding、非 LLM retrieval 和 LLM Judge。
- F1、EM、Judge 及统一报表的 `Cost-MB`/`Cost-QA` 口径不变；reward cost 另行记录。
- 本计划适用于 MemGallery、H2HMEM 和 WMA，每个 benchmark 使用独立的归一化状态。
- 正式 QA 必须继续使用现有 prompt 文案、接口、version 和 hash，不为 cost reward 修改 prompt。

## 2. 成本定义

价格只从 `Offline/configs/model_efficiency.json` 的当前模型 profile 读取，不在训练代码里重复硬编码。当前 Qwen3-VL-4B 口径为输入 `$0.05/1M tokens`、输出 `$0.25/1M tokens`。

### 2.1 实际成本

\[
C_i=\frac{P_i\times0.05+O_i\times0.25}{10^6}
\]

`P_i` 和 `O_i` 分别是服务端返回的 `prompt_tokens` 和 `completion_tokens`。多次 attempt 使用已累加的精确 usage；缺失 usage 或请求失败的 rollout 不进入 PPO buffer 和 cost 窗口，不得静默估算。

### 2.2 逐题 BaseCost

`B_i` 是同一题在所有召回 MAU 都选择 `00000` 时，仅保留 system prompt、question、输出格式约束和题目自带 query image 的基础输入成本：

\[
B_i=\frac{P_i^{00000}\times0.05}{10^6}
\]

- `00000` 表示不选 summary、dialogue、caption、memory image 和 VP；不删除问题自带图像。
- `P_i^{00000}` 由 cost 模块在内部从原 prompt 剥离 evidence 段后，用同一模型 tokenizer/processor 计数；不额外调用 LLM，也不改变实际请求 prompt。
- BaseCost 按题计算，不用全数据集的单一常数。由于定义为“仅输入 prompt”，其中不加 completion cost。

额外 evidence 成本为：

\[
\Delta C_i=\max(C_i-B_i,0)
\]

进入归一化前先做平方根变换，减弱极长 evidence 和异常长输出形成的成本长尾：

\[
C'_i=\sqrt{\Delta C_i}
\]

## 3. 滑动窗口 Cost-Min、Cost-Max 与归一化

每个 benchmark 独立维护最近 `512` 个成功 train rollout 的配对窗口
`(C'_i, F1_i)`，以 P5/P95 抑制两端离群点：

\[
C'_{min,t}=P5(W_t),\qquad \widetilde C'_{max,t}=P95(W_t)
\]

\[
S_0=\widetilde C'_{max,0}-C'_{min,0}
\]

\[
C'_{max,t}=C'_{min,t}+\max(
\widetilde C'_{max,t}-C'_{min,t},\;0.25S_0,\;\epsilon_{range})
\]

\[
C_{norm,i}=clip\left(
\frac{C'_i-C'_{min,t}}{C'_{max,t}-C'_{min,t}},0,1\right)
\]

日志中的逐题美元边界为：

\[
C_{min}(q_i,t)=B_i+(C'_{min,t})^2
\]

\[
C_{max}(q_i,t)=B_i+(C'_{max,t})^2
\]

使用同一窗口中的近期训练 reward 对齐任务奖励与成本奖励的波动尺度：

\[
\alpha_t^{scale}=\frac{std(F1)}{std(C_{norm})+\epsilon_{std}}
\]

最终奖励为：

\[
R_i=F1_i-\lambda\alpha_t^{scale}C_{norm,i}
\]

- `window_size=512`，`min_window_size=128`，`lower_quantile=0.05`，`upper_quantile=0.95`，`initial_range_floor_ratio=0.25`。
- 前 128 个有效 train rollout 仅收集 cost，使用 `R=F1`；收集完后固定初始 P5、P95 和区间 `S_0`。
- `0.25 * S_0` 是区间下限，避免 policy 成本分布收缩后归一化范围塌缩。
- 每个 rollout batch 开始时快照一次 P5/P95、有效区间和动态尺度，同一 batch 所有样本共用；PPO update 完成后才把本 batch 的 `(C'_i, F1_i)` 加入窗口。
- 窗口跨 epoch 保留，validation/test 只读取冻结状态，不更新窗口。
- `lambda` 是可控的成本权衡，默认 `0.1`；`alpha_t^{scale}` 只负责动态尺度对齐。正式实验至少对比 `lambda=0`、`0.05`、`0.1`、`0.2`，其中 `lambda=0` 必须与原 F1-only reward 等价。

## 4. 训练时序

1. 使用 policy 选 evidence，生成回答并取得精确 usage。
2. 计算 `F1_i`、`C_i`、`B_i`、`Delta C_i` 和 `C'_i`。
3. 使用本 batch 开始时冻结的 Cost-Min、Cost-Max 和 `alpha_t^{scale}` 计算 `C_norm,i` 与最终 reward。
4. 将最终 reward 写入 PPO buffer，完成参数更新。
5. 将本 batch 的有效 `(C'_i, F1_i)` 追加到窗口，超过 512 时同步淘汰最旧配对。
6. 先将 rollout、metrics 和 checkpoint 落盘，再尝试上传 W&B；W&B 失败不中断训练。

## 5. 代码改动（不新增源码文件）

- `Offline/src/benchmarks/memgallery_harness/runner/metrics.py`
  - 增加可复用的单条 usage 成本计算函数，统一价格、token 校验和精度。
- `answer_prompts.py` 及三个 benchmark 的 prompt 镜像文件
  - 不做任何改动；继续使用当前正式 prompt 和严格 evidence 校验。
- `Offline/src/evidence_policy/ppo.py`
  - `SlidingCostNormalizer` 封装平方根变换、配对窗口、warm-up、P5/P95、区间下限、动态尺度、batch snapshot 和 state 序列化。
- `Offline/src/evidence_policy/rollout.py`
  - `EvidenceRollout` 记录原始/基础/增量/变换/归一化成本、动态边界、reward 标准差、动态尺度和有效成本权重。
- `Offline/scripts/evidence_policy.py`
  - 仅在 cost 计算内部从原 prompt 剥离 evidence 段并计算 `C_min`；按第 4 节时序组装 shaped reward，不改变实际回答 prompt。
  - checkpoint 保存并恢复窗口内容、有效样本数、初始 P5/P95、初始区间和当前快照，保证断点续训不改变归一化状态。
- `Offline/configs/evidence_policy*.json`
  - 在 `reward` 中配置 `cost_tradeoff_lambda`、`cost_transform`、窗口、分位数、区间下限及两个独立 epsilon。
- W&B 与本地 JSON/JSONL
  - 每次 update 记录成本链路、动态 Cost-Min/Max、两个 reward 标准差、`alpha_t^{scale}`、有效权重、两端裁剪率和全 `00000` 率。

## 6. 验收标准

1. 三个 benchmark 的正常 evidence prompt 文案、version/hash 不因 cost 计算改变；PPO 的 `00000` 仍使用既有空 evidence 分支。
2. 单元测试覆盖平方根、成本公式、`C_norm in [0,1]`、warm-up、P5/P95、区间下限、动态尺度、同 batch 固定快照、batch 后更新及 val/test 不更新。
3. 中断前后 normalizer 状态完全一致；缺失 usage 和失败 rollout 不污染 PPO buffer/窗口。
4. `lambda=0` 回归测试与现有 F1-only reward 一致；三个 benchmark 各完成小规模 smoke test，无旧 prompt cache 误复用。
5. 正式结果同时报告 F1/EM/Judge、平均 QA cost、平均 normalized cost、全 `00000` 率和 evidence level 分布，不只报告 shaped reward。

## 7. 实施顺序

1. 先实现逐题 BaseCost token 计数和单条成本函数，补齐测试。
2. 实现 `SlidingCostNormalizer` 及 checkpoint 状态。
3. 接入 PPO batch 时序、本地记录和 W&B。
4. 运行单元测试与三基准 smoke test；通过后再开展 `lambda` 消融和全量训练。
