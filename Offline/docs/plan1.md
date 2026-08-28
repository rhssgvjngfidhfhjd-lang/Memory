# PPO Evidence Selection Policy 实现计划

## 1. 目标

在现有 Mem-Gallery 检索与问答链路之间加入一个基于 PPO 的 Evidence
Selection Policy。

现有 retrieval 根据 query embedding 从固定 memory bank 中返回 Top-5 MAU。
Policy 不改变 retrieval 的召回结果、顺序或分数，而是针对每个 retrieved
MAU 独立选择送给下游 VLM 的 evidence，构建 Evidence Chain，并使用最终问答
F1 作为 PPO reward。

整体流程：

```text
Query
  -> existing retrieval
  -> Top-5 MAU
  -> per-MAU PPO policy
  -> Evidence Chain
  -> Qwen3-VL-4B-Instruct
  -> Answer
  -> F1 reward
  -> PPO update
```

第一阶段只实现框架，不修改 MAU 构建方法、不优化 retrieval、不训练 VLM，也不使用
LLM Judge 作为在线 reward。

## 2. 固定实验对象

固定使用以下 memory bank：

```text
Offline/outputs/hive_mem_token_rerun_0810/c_insert_only
```

该 memory bank 已验证：

- 包含 20 个 persona/dataset；
- 共 8192 个 MAU；
- 每个 MAU 恰好有一个 `source_dialogue_id`；
- 一个 dialogue 可以生成多个 MAU；
- 2609 个 MAU 同时具有 image 和 caption；
- 5583 个 MAU 既没有 image，也没有 caption；
- 不存在只有 image 或只有 caption 的 MAU。

`dialogue` 不是 MAU 工程 metadata。本文中的 dialogue 只表示该 MAU 来源轮次的
完整原始对话：

```text
user: <原始 user 文本>
assistant: <原始 assistant 文本>
```

运行时根据 MAU 的 `source_dialogue_ids[0]` 从
`Mem-Gallery/benchmark/data/dialog/*.json` 查回，不修改现有 MAU 文件结构。

## 3. 已确认的 Policy 输入

Policy 对 Top-5 中每个 MAU 独立运行。若 retrieval 得到 `MAU_1 ... MAU_5`，则
并行形成：

```text
query_embedding + MAU_1.summary_embedding
query_embedding + MAU_2.summary_embedding
...
query_embedding + MAU_5.summary_embedding
```

同一个共享 MLP 分别处理这 5 个输入。判断某个 MAU 时看不到其他 retrieved
MAU，也不使用 Top-K pooled/global context。

Policy 输入明确不包含：

- MAU image embedding；
- dialogue embedding；
- caption embedding；
- 原始文本或原始图片；
- 其他 MAU 的表示。

复用现有 query embedding 和 MAU summary embedding，不新增 encoder。

## 4. 已确认的 Action Space

Action 不是 4 个相互独立的 Bernoulli 开关，而是两个 categorical 二选一。

### 4.1 文本 evidence head

每个 retrieved MAU 都必须执行一次：

```text
summary | dialogue
```

- 不要求必须选择 summary；
- 不允许 summary 和 dialogue 同时选择；
- 不允许二者都不选择；
- 不允许丢弃整个 MAU。

### 4.2 视觉 evidence head

仅当 QA 类别是 VS/VR，且当前 MAU 有图时执行：

```text
image | caption
```

边界规则：

- 无图 MAU：不执行视觉动作，也不添加视觉 evidence；
- 非 VS/VR QA：保持现有规则，不把 retrieved MAU 图片发送给 VLM；
- 非 VS/VR QA 检索到有图 MAU：固定使用 caption，不产生视觉 policy 动作；
- VS/VR QA 检索到有图 MAU：由 policy 在 image 和 caption 间二选一；
- 不考虑一个 MAU 内包含多张图的情况；
- 不设置 Evidence Chain 最大图片数或 token 数。

因此，有图 MAU 在 VS/VR 中有四种 joint evidence 组合：

```text
summary + image
summary + caption
dialogue + image
dialogue + caption
```

## 5. Episode、概率与 Reward

一次 QA 是一个单步 episode（contextual bandit）：

1. 固定 retrieval 返回 Top-5 MAU；
2. Policy 为每个 MAU 采样文本动作；
3. 对符合条件的 MAU采样视觉动作；
4. 所有 per-MAU 动作共同组成一次 joint action；
5. 构建一条 Evidence Chain；
6. VLM 仅调用一次并生成最终答案；
7. 使用最终答案相对标准答案的 F1 作为整次 episode 的共同 reward。

Joint action 的 log probability 为所有实际执行的 categorical action log
probability 之和：

```text
joint_log_prob = sum(text_log_prob_i) + sum(applicable_visual_log_prob_i)
```

某个 MAU 不获得独立 reward，Top-5 中所有动作共同承担最终 F1 reward。

第一版：

```text
reward = answer_f1
```

- 不加入 evidence token penalty；
- 不加入 image cost；
- 不加入 EM bonus；
- 不在线调用 LLM Judge；
- Reward 通过可替换接口实现，以便后续增加其他 reward。

单步 episode 中不存在跨时间 credit assignment；GAE/return 接口仍保留，但单步
return 可直接由 reward 和终止 value 计算。

## 6. 下游 VLM 与 Rollout

训练与评估固定使用：

```text
Qwen/Qwen3-VL-4B-Instruct
temperature = 0
固定 system prompt
固定 question prompt
固定生成参数
```

VLM 保持冻结。Policy 只控制 Evidence Chain，不修改回答模型。

为减少重复 VLM 调用，缓存：

```text
(query_id, evidence_action_signature, rollout_config_hash)
    -> answer, F1, error, token/image statistics
```

缓存使用本地文件实现，不新增依赖。`rollout_config_hash` 至少覆盖模型名、prompt
版本和生成参数，避免错误复用旧答案。

## 7. 数据划分

按 persona/dataset 划分，避免同一 persona 的记忆同时进入训练与测试：

```text
12 train / 4 validation / 4 test
```

具体 persona 清单尚未确定。划分时应：

- 三个 split 无 persona 重叠；
- 尽量平衡 AR、CD、FR、KR、MR、TR、TTL、VR、VS 类别和 QA 数量；
- 使用固定随机种子；
- 将最终清单保存为配置文件，确保可复现；
- test split 只用于最终评估，不参与超参数选择。

## 8. Baselines 与推理模式

需要支持以下四种 evidence strategy：

1. `full-evidence`：按当前 evidence schema 使用完整/默认 evidence；
2. `summary-only`：文本固定选择 summary，视觉部分按明确的 baseline 规则执行；
3. `random`：在所有合法 action 中均匀随机采样；
4. `ppo`：使用训练后的 policy。

PPO policy 支持：

- stochastic sampling：训练和 rollout；
- deterministic inference：对每个 categorical head 取 argmax，用于验证和测试。

## 9. 分阶段实现任务与验证方法

### Task 1：数据契约与 split

工作：

- 定义 query、retrieved MAU、dialogue、evidence action 和 episode 的类型；
- 生成并固定 12/4/4 persona split；
- 建立 `source_dialogue_id -> {user, assistant}` 只读索引；
- 校验指定 memory bank 与原始 Mem-Gallery 数据的一致性。

验证：

- 20 个 persona 恰好被分配一次；
- train/validation/test 无交集；
- 每个 MAU 恰好解析出一个 source dialogue；
- 所有 source dialogue 均能查回 user/assistant；
- 类别与 QA 数量分布报告可复现。

### Task 2：Evidence schema 与 Evidence Chain builder

工作：

- 定义文本和视觉 categorical action；
- 将 per-MAU action 转为 summary/dialogue/image/caption evidence；
- 复用现有回答 prompt 和图片编码链路；
- 保持非 VS/VR 不发送 memory image 的现有规则。

验证：

- 四种 VS/VR 合法组合渲染正确；
- 无图 MAU不产生视觉 evidence；
- 非 VS/VR 有图 MAU只输出 caption，不发送图片；
- Evidence Chain 中没有未选择的 evidence；
- 原有 VLM client public API 不被破坏。

### Task 3：Policy/Value MLP

工作：

- 拼接 query embedding 与当前 MAU summary embedding；
- 使用共享 MLP 并行处理 Top-5 MAU；
- 输出 text categorical logits、适用时的 visual categorical logits，以及
  episode value；
- 支持 action availability mask；
- 支持 stochastic sample、deterministic argmax、log-probability 和 entropy。

验证：

- batch、Top-K 和 embedding 维度正确；
- 非适用视觉动作不计入 log probability/entropy；
- 所有采样动作合法；
- deterministic inference 对相同输入稳定；
- 梯度能够从 PPO loss 回传到 MLP 参数。

### Task 4：VLM rollout environment 与缓存

工作：

- 串联 retrieval、policy、Evidence Chain、VLM、F1 reward；
- 固定 VLM 与 prompt 配置；
- 实现可恢复的 rollout cache；
- 记录答案、reward、action、log probability、value 和错误。

验证：

- 单个 QA 能完整运行；
- 相同 cache key 第二次不调用 VLM；
- prompt/模型配置变化会产生新 cache key；
- F1 与现有 `metrics.f1_score` 一致；
- VLM 错误被记录，不破坏整个训练 run。

### Task 5：PPO buffer 与更新器

工作：

- 存储单步 transition；
- 计算 return/advantage；
- 实现 clipped policy objective、value loss 和 entropy bonus；
- 支持 minibatch、多 epoch 更新和 gradient clipping；
- 保存 optimizer 与训练状态。

验证：

- 人工 transition 的 return/advantage 数值正确；
- ratio=1 时 clipped objective 符合预期；
- 超出 clip range 时正确截断；
- value loss、entropy 和总 loss 均为有限数；
- 小型合成 batch 更新后参数发生变化。

### Task 6：训练、checkpoint 与恢复

工作：

- 训练入口只读取 train split；
- validation 使用 deterministic policy；
- 保存 policy、value、optimizer、配置、split、随机状态和训练步数；
- 支持从 checkpoint 恢复。

验证：

- 小样本训练能够完成至少一次 PPO update；
- checkpoint 保存后可加载并复现 deterministic action；
- 恢复训练时步数和 optimizer 状态连续；
- validation/test 不参与 PPO update。

### Task 7：评估与 baselines

工作：

- 统一运行 full-evidence、summary-only、random 和 PPO；
- 输出答案质量、action 分布和 evidence 使用统计；
- 按整体、persona、QA category 汇总。

验证：

- 四种 strategy 使用同一 retrieval、VLM 和 split；
- baseline action 满足 schema；
- PPO evaluation 为 deterministic；
- test 结果可从原始逐题记录重新汇总。

### Task 8：测试与 smoke test

工作：

- 为 dialogue lookup、schema、builder、policy、cache、reward、PPO loss 和
  checkpoint 添加单元测试；
- 在少量 train QA 上运行端到端 smoke test。

验证：

- 现有测试继续通过；
- 新增测试覆盖所有 action 边界；
- smoke test 完成 retrieval -> action -> VLM -> F1 -> PPO update；
- 不需要完整 1711 QA 即可验证框架。

## 10. 计划增加的接口

以下为设计接口，名称可在编码前最后确认。

```python
class EvidenceTextAction(Enum):
    SUMMARY = "summary"
    DIALOGUE = "dialogue"


class EvidenceVisualAction(Enum):
    IMAGE = "image"
    CAPTION = "caption"


@dataclass(frozen=True)
class MAUEvidenceAction:
    memory_id: str
    text: EvidenceTextAction
    visual: EvidenceVisualAction | None


@dataclass(frozen=True)
class PolicyObservation:
    query_embedding: torch.Tensor
    summary_embeddings: torch.Tensor
    visual_action_mask: torch.Tensor


@dataclass
class PolicyStep:
    actions: list[MAUEvidenceAction]
    joint_log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class EvidenceSelectionPolicy(nn.Module):
    def sample(self, observation: PolicyObservation) -> PolicyStep: ...
    def select_deterministic(self, observation: PolicyObservation) -> PolicyStep: ...
    def evaluate_actions(
        self,
        observation: PolicyObservation,
        actions: list[MAUEvidenceAction],
    ) -> PolicyStep: ...


class DialogueStore:
    def get(self, dataset: str, dialogue_id: str) -> DialogueEvidence: ...


class EvidenceChainBuilder:
    def build(
        self,
        query_category: str,
        memory_hits: list[MemoryHit],
        actions: list[MAUEvidenceAction],
    ) -> list[dict[str, Any]]: ...


class RewardFunction(Protocol):
    def __call__(self, prediction: str, ground_truth: str) -> float: ...


class F1Reward:
    def __call__(self, prediction: str, ground_truth: str) -> float: ...


class EvidenceSelectionEnv:
    def rollout(self, episode: EvidenceEpisode, strategy: EvidenceStrategy) -> Rollout: ...


class PPOBuffer: ...
class PPOTrainer: ...
```

实现时所有新增函数必须有 type hints，并优先复用：

- `hive_mem.retriever.MemoryHit`；
- `SimpleMemoryIndex` / `GraphExpandedIndex`；
- `VLMAnswerClient`；
- `build_retrieved_memory_context` 可复用的格式化逻辑；
- `metrics.f1_score`；
- `QueryEmbeddingCache`；
- 已保存的 MAU summary vectors。

## 11. 新增文件（精简版）

新增独立 package，避免把 PPO 逻辑塞入 retrieval 或 VLM client；小型职责合并，
第一版只保留 4 个实现模块：

```text
Offline/src/evidence_policy/
  __init__.py
  evidence.py              # 类型、DialogueStore、Evidence Chain、baseline strategies
  policy.py                # shared MLP、categorical heads、采样与确定性推理
  rollout.py               # VLM 环境、F1 reward、本地 rollout cache
  ppo.py                   # buffer、PPO 更新、checkpoint 保存/恢复

Offline/scripts/
  evidence_policy.py       # prepare-split / train / eval 三个子命令

Offline/configs/
  evidence_policy.json     # split、模型参数、PPO 参数

Offline/tests/
  test_evidence_policy.py
```

只有文件增长到职责不清时再拆分。

## 12. 与现有代码的接入

第一版通过统一脚本编排 retrieval、Evidence Chain 和 VLM，直接复用
`SimpleMemoryIndex`、`QueryEmbeddingCache`、`VLMAnswerClient` 与现有 prompt/metrics。
现有 `memory_items` 已能表达 dialogue/caption/image evidence，因此无需修改：

```text
Offline/src/benchmarks/memgallery_harness/eval_memgallery.py
Offline/src/benchmarks/memgallery_harness/runner/answer_client.py
```

这样旧评测入口、默认行为和 public API 均保持不变。

不计划修改：

- MAU 构建流程；
- `MAU` / `MAUBank` 的持久化 schema；
- retrieval 排名逻辑；
- query embedding 和 summary embedding 生成逻辑；
- VLM 模型参数；
- 与 PPO 无关的 baseline。

## 13. 第一版实现默认值与运行前条件

为保持第一版简单可运行，采用以下可配置默认值：

1. 12/4/4 persona 清单由固定种子的平衡划分脚本生成并写入配置；
2. `full-evidence` 使用 dialogue，VS/VR 有图时使用 image，其他情况使用 caption；
3. `summary-only` 使用 summary，所有有图情况使用 caption；
4. 同一 source dialogue 的多个 MAU 都选择 dialogue 时保留重复项，不去重；
5. MLP 使用两层 hidden=256 的 GELU，所有超参数均写入配置；
6. episode value 对所有 per-MAU hidden state 做 mean pooling；
7. 旧 memory bank 的 `image_paths` 在 rollout 时映射到当前
   `Mem-Gallery/benchmark/data/image`；
8. 指定 memory bank 对应的 2048 维 query embedding cache 当前不存在，正式训练前
   必须先生成并验证。缺失时命令应快速失败并给出明确提示。

## 14. Coding Rules

- 优先复用现有函数，不重复造轮子；
- 不修改与当前任务无关的代码；
- 不随意改变已有 public API；
- 不新增依赖，除非确实必要；
- 新增函数必须有 type hints；
- 保持现有项目代码风格；
- 不删除现有逻辑，除非任务明确要求；
- 遇到不确定设计先说明假设并确认。
