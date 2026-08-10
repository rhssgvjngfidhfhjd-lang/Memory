# HiveMem 流水线（步骤 → 代码 → 输入/输出）

## 1. 切块  `scripts/build_chunks.py::main` → `src/embedding/chunk_builder.py::build_chunks_from_directory / write_chunks_jsonl`
输入：Mem-Gallery 原始对话 `Mem-Gallery/benchmark/data/dialog/*.json`
默认输出：`data/qwen3_vl_embedding_2b/chunks_no_profile.jsonl`（每对话轮一条，均值 ~163 tok，persona 已剥离）

## 2. 建库（MAU 生成 + 同步编码）  `hive_mem/build_memories.py::main` → `hive_mem/builder.py::MemGalleryMemoryBuilder.build`
prompt1 拼装：`hive_mem/executor.py::MemoryExecutor._build_prompt`（本体清单来自 `entity_schema.py::ontology_prompt_block`）
LLM 调用与解析：`executor.py::execute / _parse_response`，入库 `executor.py::apply_to_memory_bank`
向量编码（就在本步）：`embedding/qwen3vl_embedding.py::QwenMemoryEmbedder.embed_texts / embed_images`
输入：chunks_no_profile.jsonl + `configs/profiles.json`（persona 注入 prompt）
输出：`outputs/<run>/datasets/<数据集>/memories.jsonl`（MAU：summary+entities+attributes）+ `vectors/text.npy` + `vectors/image.npy`/`vectors/image_mask.npy`

## 3. 建图  `hive_mem/build_memory_edges.py::main / build_temporal_chain`（可选 LLM 边：`classify_event_relations`）
输入：memories.jsonl（含 entities 字段）
输出：落盘边写回 memories.jsonl 的 `links`（时间链 prev/next；可选 EVENT_RELATION 入 related）+ `edges_manifest.json` + `conflict_candidates.json`
注：实体边/属性边不落盘——检索加载时由 `derive_entity_pairs / derive_attribute_pairs` 现场推导（DF 过滤 + 度数封顶）

## 4. query embedding（离线一次性）  `embedding/build_query_embeddings.py::main`（缓存键：`benchmarks/memgallery_harness/retrieval/query_embedding_cache.py::make_query_id`）
输入：Mem-Gallery 全部 1711 题（问题文本 + VS/VR 查询图）
默认输出：`data/qwen3_vl_embedding_2b/query_embeddings/{vectors.npy, metadata.jsonl, manifest.json}`（评测只读缓存，保证各实验查询向量逐比特一致）

## 5. 检索  `hive_mem/retriever.py::GraphExpandedIndex.search`（append 模式 `_search_append`；纯向量对照 `SimpleMemoryIndex.search`）
输入：查询向量（读第 4 步缓存）+ 记忆库目录（jsonl+npy）
输出：top-5 向量命中 + ≤2 条图一跳追加（`MemoryHit` 列表，`via` 标注 vector/graph；VS/VR/TTL 类附带图文向量 max-fusion）

## 6. 答题  `benchmarks/memgallery_harness/eval_memgallery.py::run_dataset`（上下文打包：`SimpleMemoryMemGalleryAdapter.recall`，同文件）
prompt2 拼装：`runner/prompts.py::SYSTEM_PROMPT + format_question_prompt`（+profile 注入 system prompt 尾部，config 默认开）
LLM 调用：`runner/answer_client.py::VLMAnswerClient.answer`
输入：top-5(+2) 记忆（文本+图片）+ 问题 + 类别约束 + profile
输出：`results.json`（逐题答案+检索溯源）+ `retrieval_trace.jsonl`

## 7. 指标（两波）
7a 自动指标  `benchmarks/memgallery_harness/runner/metrics.py::summarize_results / provenance_hit / f1_score / exact_match`
输入：results.json
输出：`metrics.json`（F1 / EM / HIT@5，总体 + 分类别）
7b LLM Judge  `scripts/judge_results_llm_parallel.py::build_prompt / judge_one`（gpt-oss-120b，NVIDIA API）
输入：results.json
输出：`llm_judge_results.json`（逐题判决+理由）+ `llm_judge_metrics.json`（JudgeAcc）+ `summary.json`（F1/EM/HIT 与 JudgeAcc 合并汇总）
