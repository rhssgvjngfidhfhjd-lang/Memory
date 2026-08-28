# Archive

过时但暂不删除的数据/代码归档。规则：当前流程用不到、又可能有回溯价值的文件放这里；
确认再也用不到时整目录删除即可。恢复 = `mv` 回原位置。

## data/

| 文件 | 来历 | 归档原因（2026-08-06） |
|------|------|------|
| chunks.jsonl | mode-c 全量 chunk（含 persona 前缀，均值 326 tok） | 被 chunks_no_profile.jsonl（163 tok + configs/profiles.json 注入）取代；最早的 agentmem_rerun_20260802 库用它建的 |
| chunks_image_companions_only.jsonl | 早期图片消融变体 | 对应实验线已结束 |
| chunks_image_subset.jsonl | 早期图片子集消融变体 | 同上 |
| chunks_separate_vectors.jsonl | 早期图文分向量消融变体 | 同上 |
| chunks_pack256.jsonl / chunks_pack512.jsonl | pack_chunks.py 产出的多轮打包 chunk（软预算 256/512 tok） | 多轮打包实验尚未开跑；开跑时 mv 回 data/processed/（注意 pack_chunks.py 脚本本身已被删除，需要时从 git 恢复） |
| metadata.jsonl.bak_offline_paths | query_embeddings 元数据的路径迁移备份 | 迁移已完成且验证通过 |

## code/

| 文件 | 来历 | 归档原因（2026-08-07） |
|------|------|------|
| extract_entities.py | 离线实体补抽/别名仲裁工具（属性 schema 当初靠它先行验证） | 统一输出上线后主流程零依赖；独有能力：不重建库即可重抽实体（--force）、别名仲裁（--alias-merge → entity_aliases.json，消费方 build_memory_edges.load_alias_map 仍在役）、entity_stats 报告。恢复 = mv 回 src/hive_mem/ 并把 tests 里两处 import 从 entity_schema 换回；注意其 EXTRACTION_PROMPT 未跟进 ### 分节风格，复用前建议对齐 |

## outputs_old/

（用户归档的历史实验产物。）
