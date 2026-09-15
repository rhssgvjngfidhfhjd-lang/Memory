严格遵循 /data/haozhen/Memory-clean/Offline/configs/ppo.md 执行并完成三个 benchmark 的 PPO 全量训练、测试、W&B 实时记录及结果验收；持续监控至完成，失败时自动断点续跑，不覆盖已有实验目录

原版：
原始 dialogue round
→ 原版 observation
→ 结构化多模态 Context Node
→ CoPe + graph traversal
→ 最多 10 条结构化 memory
→ 保留 dialogue/image
→ 回答模型

当前：
公共扩展 chunk
→ 图片被降为路径字符串
→ 改写版 embedding + 静默 fallback
→ CoPe + graph traversal
→ 最多 10 条 memory
→ 整体 str(list)
→ 错误解析 dialogue ID
→ 1 条无 provenance、无图片的聚合 evidence
→ 回答模型

好了，现在我差不都修改完和官方代码对齐了，你再次检查一下修改后的代码和官方的区别
以下区别可以不算错误
1.建库的chunk没有遵循官方原版而是和ourmethod的chunk一样
2.top-k改为固定top7
3.QA的prompt更改

当前可以按照/data/haozhen/Memory-clean/Offline/docs/baselines.md来跑我们修复的miyix吗，不用真的做，直接回答
### 用api来跑baseline
/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json这是api，/data/haozhen/Memory-clean/Nvida_api/defaults_gpt-5-mini.json和/data/haozhen/Memory-clean/Nvida_api/config_gpt-5-mini是配置，尝试遵循/data/haozhen/Memory-clean/Offline/docs/baselines.md里的配置来跑三个benchmark的test部分，注意三个benchmark可以用api并行跑