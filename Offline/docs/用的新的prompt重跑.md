使用 GPU 1 和 GPU 2，并行完成 H2HMEM 与 WMA 的新版官方 prompt 全量 PPO test。

具体要求：

1. 本次只重新运行 QA test：
   - 不重新训练 PPO。
   - 不重建 memory bank。
   - 不重建 query embedding。
   - 不重新提取 VP。
   - 不运行 LLM Judge。

2. 必须加载我们已经训练完成的 PPO policy，禁止使用 FULL、SUMMARY、随机或其他 evidence strategy：
   - H2HMEM checkpoint：
     /data/haozhen/Memory-clean/Offline/outputs/PPO/H2HMEM_0907PPO_b/checkpoints/epoch_005.pt
   - WMA checkpoint：
     /data/haozhen/Memory-clean/Offline/outputs/PPO/WMA_0907PPO_a/checkpoints/epoch_005.pt
   - 使用 strategy=ppo、deterministic=true，让已训练的 MLP policy 为每条 QA 选择 evidence level。

3. 必须使用当前已经同步到 PPO 的最新版官方 benchmark prompt：
   - 根据 PPO policy 实际选择的 evidence 动态构造 prompt。
   - evidence context 只能拼接一次。
   - 正确解析 H2HMEM 的 {"reasoning_process": ..., "system_answer": ...}。
   - 正确解析 WMA 的 {"answer": ...}。
   - 使用解析后的最终答案计算 F1 和 EM。

4. H2HMEM：
   - 使用 GPU 1。
   - 在 GPU 1 上启动或复用 Qwen/Qwen3-VL-4B-Instruct vLLM，端点使用 http://127.0.0.1:8011/v1。
   - 复用旧 H2HMEM memory bank、query embedding 和 qwen3vl4b_all_v1 VP。
   - 完整运行 360 条 test QA。
   - 明确记录：旧 memory bank 不包含新补齐的 caption，因此本次是无 caption bank。

5. WMA：
   - 使用 GPU 2。
   - 在 GPU 2 上启动或复用 Qwen/Qwen3-VL-4B-Instruct vLLM，端点使用 http://127.0.0.1:8012/v1。
   - 复用旧 WMA memory bank、query embedding 和 qwen3vl4b_all_v1 VP。
   - excluded_categories 必须为空，不得排除 MB。
   - 完整运行 440 条 test QA。

6. 两个 benchmark 都保持原训练设置：
   - 5 个向量召回 + 2 个图扩展结果。
   - graph mode=append。
   - 使用旧 checkpoint 学到的 deterministic PPO evidence policy。
   - 只重新进行 evidence 选择后的最终回答调用。

7. 新 prompt 必须产生全新的 QA 请求：
   - 不得复用旧 prompt 的 rollout cache。
   - 保留新的 raw JSON response 和解析后的 answer。
   - 在结果中记录 prompt_version、prompt_source 和 prompt_sha256。

8. 不得覆盖任何旧结果。新输出目录为：
   - /data/haozhen/Memory-clean/Offline/outputs/PPO/H2HMEM_0909PromptTest_a/eval/test_ppo
   - /data/haozhen/Memory-clean/Offline/outputs/PPO/WMA_0909PromptTest_a/eval/test_ppo
   如果目录已经存在，依次改用 _b、_c。

9. 两个任务并行放入 tmux 后台运行，关闭当前终端后仍应继续。启动前检查：
   - GPU 1、GPU 2 是否空闲。
   - 两个 vLLM 端点的模型是否正确。
   - checkpoint 是否完整可加载。