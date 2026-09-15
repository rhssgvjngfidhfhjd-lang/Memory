进入“严格原版复现模式”。
以 <官方仓库路径或明确 commit SHA> 为唯一实现依据，严格复用原版的建库、Memory Manager、检索、Chat Agent 和回答流程。
!!!注意此处的topk默认为7，prompt默认和对应benchmark的qa prompt（如Offline/src/benchmarks/.../prompts.py）一样，不许更改qa prompt，baseline内部agent的prompt 必须原样保留，如果不确定何时用agent的prompt何时用qa prompt请向我询问确认

禁止：
- 简化、重写或替代原算法流程；
- 加入统一 chunk、、冻结检索或统一回答 Prompt；
- 静默 fallback；
- 擅自修复上游 bug；
- 根据论文描述臆造代码中不存在的流程。

执行前必须：
1. 阅读原版入口及完整调用链；
2. 用“文件路径 + 函数名 + 原代码证据”说明原版流程；
3. 列出所有计划修改及其是否偏离原版；
4. 遇到实验规范与原版冲突时停止并询问，不得自行折中；
5. 等我确认后才能修改。（比如我们默认的topk=7会和一些baseline自身方法冲突）


### 现有问题
/data/haozhen/Memory-clean/Offline/baselines里的baselines不按原版官方代码输出

### 模型选择
1.api位置：/data/haozhen/Memory-clean/Nvida_api/Openrouter_api
2.使用模型：qwen/qwen3-vl-8b-instruct
3.模型配置：和/data/haozhen/Memory-clean/Offline/configs/defaults.json一样

### 解决方案
1.阅读原版代码
2.写出原版预期调用链+列出每个阶段的预计产物
3.执行一次完整流程
可选择三个benchmark里的test部分中的一个dataset来作为一次完整流程，注意严格遵循其中的qa prompt，绝对不允许更改
4.收集真实产物与 trace，自动逐项比较
5.输出验收报告

### Adapter和Runner
看能否复用/data/haozhen/Memory-clean/Offline已有的

### Chunk
不要按原有的，请遵循我们当前默认chunk
Mem-Gallery：每个 dialogue round 一个 chunk，包含该轮 user + assistant + image_caption；图片作为该 chunk 的视觉输入。见 [omni_inputs.py (line 32)](/data/haozhen/Memory-clean/Offline/src/benchmarks/baseline_runtime/omni_inputs.py:32)。
H2HMEM：每个原始对话 turn 一个 chunk，不按 session 合并。见 [omni_inputs.py (line 113)](/data/haozhen/Memory-clean/Offline/src/benchmarks/baseline_runtime/omni_inputs.py:113)。
WorldMemArena：同样每个原始 source turn 一个 chunk。见 [omni_inputs.py (line 125)](/data/haozhen/Memory-clean/Offline/src/benchmarks/baseline_runtime/omni_inputs.py:125)。


### 输出给用户的产物

1.原版预期调用链+列出每个阶段的预计产物
2.真实调用链+每个阶段的真实产物位置


### 关于预计产物
“预计产物”需要写成可验证条件，而不只是文字描述，例如：
文件名和位置
schema 和必要字段
Raw/Semantic/Image memory 是否存在
evidence 引用是否有效
retrieval 数量是否为 7
Prompt hash
QA 数量
是否经过原版 Agent/Manager
