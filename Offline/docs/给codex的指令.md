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


当前的三个gpu是空的，请按照本地两个md文件来跑一遍试试，有什么问题先和我汇报

我需要你确认1.是否遵循wma的checkpoint不泄露原则
2.最后给回答agent的信息是topk=7，中途你建库检索agent是top几我不管
3.使用的是/data/haozhen/Memory-clean/Offline/src/benchmarks/.../runner/prompts.py里的prompt在最后回答时

持续监控，防止出现超过512超过1024这种问题，连续出现则及时叫停实验

我又加了两个新的benchmark/data/haozhen/Memory-clean/MemEye和/data/haozhen/Memory-clean/MEMLENS，你去看一下，然后在/data/haozhen/Memory-clean/Offline/src/benchmarks/memeye_harness和/data/haozhen/Memory-clean/Offline/src/benchmarks/memlens_harness新增m3agent的接口（尽量复用原有的文件）

我需要你把cost和call写成这样的公式

我又要跑两个新的benchmark，写了新的/data/haozhen/Memory-clean/Offline/src/benchmarks/memeye_harness，/data/haozhen/Memory-clean/Offline/src/benchmarks/memlens_harness，你用/data/haozhen/Memory-clean/Nvida_api/Openrouter_api这个api来看看，模型选取qwen3.5-9b,其他参数保持不变（Embedding-0.6B → 2048维、Top‑5，其余默认），分别测试memverse能否在这3+2个benchmark上跑出结果，
重新整理一下这个任务，发个prompt给我


base second + 输入token数量*输入系数 + 输出token数量*输出系数 + 输入图片数量*图片系数

