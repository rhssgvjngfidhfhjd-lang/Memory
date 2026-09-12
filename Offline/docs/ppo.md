
# HiveMem 三基准 PPO 全量实验计划
## 0. 首先把这个任务（ppo以及codex自身）转入tumx，防止关机
## 1. 实验目标

使用 HiveMem 在以下三个 benchmark 上分别训练并评测独立的 PPO Evidence Policy：

- MemGallery
- H2HMEM
- WorldMemArena（WMA）

三个实验使用独立模型端点、独立 checkpoint、独立 rollout cache 和独立结果目录，不共享 PPO 状态。

## 2. 固定实验配置

- 检索配置：**5 个向量召回 + 2 个图扩展结果**
- 图扩展模式：`append`
- PPO epoch：6
- Rollout batch size：16
- 每轮验证：训练到 50% 时一次，epoch 结束时一次
- 初始验证：训练开始前一次
- 验证范围：完整 validation split
- 测试范围：完整 test split
- VP：`qwen3vl4b_all_v1`
- VP 严格覆盖：启用
- WMA：不得排除 MB，test 必须为 **440 QA**

## 3. 数据划分

| Benchmark | Train | Validation | Test | 总 QA |
|---|---:|---:|---:|---:|
| MemGallery | 940 | 312 | 275 | 1527 |
| H2HMEM | 1063 | 363 | 360 | 1786 |
| WMA | 1265 | 385 | 440 | 2090 |

每个实验包含 1 次初始验证和每个 epoch 的 2 次验证，共 **13 次完整 validation**。

## 4. GPU 与端点分配

| Benchmark | GPU | vLLM 端点 |
|---|---:|---|
| MemGallery | GPU 3 | `http://127.0.0.1:8013/v1` |
| H2HMEM | GPU 4 | `http://127.0.0.1:8014/v1` |
| WMA | GPU 5 | `http://127.0.0.1:8015/v1` |

三个 PPO 实验并行运行。PPO policy 网络建议放在 CPU，GPU 专门用于 vLLM 推理，避免额外占用显存。

## 5. 运行前检查

必须全部通过后才能正式训练：

1. 三个 vLLM 端点模型检查通过。
2. memory bank 与 query embedding cache 完整。
3. 三个 benchmark 的 VP audit 均无缺失。
4. H2HMEM caption 已生成并回填；若暂不处理，需要明确记录 H2HMEM 使用的是无 caption 的 VP/证据配置。
5. WMA 配置确认 `excluded_categories=[]`。
6. 输出目录不存在，避免误续跑旧实验。

## 6. 输出命名

根目录：

```text
/data/haozhen/Memory-clean/Offline/outputs/PPO
```

命名格式：

```text
{Benchmark}_{MMDD}PPO_{当日实验序号}
```

本日第一次运行示例：

```text
MemGallery_0907PPO_a
H2HMEM_0907PPO_a
WMA_0907PPO_a
```

同一 benchmark 当日重跑依次使用 `_b`、`_c`，禁止覆盖已有实验。

## 7. W&B 实时日志与容错
三个 benchmark 分别创建独立 W&B run。
W&B run name 与本地实验目录名保持一致，例如 H2HMEM_0907PPO_a。
训练开始时调用 wandb.init()，记录完整配置、数据划分、Git commit、VP signature 和输出目录。
每次 PPO 参数更新后调用 wandb.log()，上传 loss、reward、KL、entropy、clip fraction、value error、learning rate 等指标。
每次 initial、half-epoch 和 end-of-epoch validation 完成后上传 F1、EM、retrieval hit rate、reward、errors 和 evidence action 分布。
本地 JSON/JSONL 和 checkpoint 是唯一权威结果，必须先成功落盘，再尝试上传 W&B。
W&B 初始化或上传异常只能记录 warning，不得中断训练。
本地保留完整的 update step、validation phase 和指标，结束后通过 upload_evidence_policy_wandb.py 补传。
wandb.finish() 异常同样不得影响训练完成状态。
## 8. 执行顺序

1. 运行三个 benchmark 的配置、数据和 VP 预检。
2. 同时启动三套 6-epoch PPO 训练。
3. 持续监控进程、GPU、epoch、rollout 数量、失败数和 endpoint 状态。
4. 训练完成后分别使用 `epoch_005.pt` 跑完整 test。
5. 汇总三个 benchmark 的训练、验证和测试指标。
6. 若中断，从最后一个完整 epoch checkpoint 断点续跑，不覆盖已提交指标。

## 9. 必须保留的产物

每个实验目录至少包含：

- `checkpoints/epoch_000.pt` 至 `epoch_005.pt`
- `ppo_metrics.jsonl`
- `rollout_cache.jsonl`
- 每轮 train rollouts
- 初始及每轮 validation metrics/rollouts
- `eval/test_ppo/metrics.json`
- `eval/test_ppo/rollouts.jsonl`
- 完整运行日志

## 10. 验收标准

- 三个实验均完成 6 个 epoch。
- Train、validation、test 数量与表格完全一致。
- WMA test 必须为 **440**，不能再出现 400。
- VP 缺失数和 crop 缺失数均为 0。
- 最终 test 错误请求数为 0。
- 最终报告至少包含 F1、EM、LLM Judge（如接入）、retrieval hit rate、reward、evidence action 分布、调用量及 token/cost 指标。