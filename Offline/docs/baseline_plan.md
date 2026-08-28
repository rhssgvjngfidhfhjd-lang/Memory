# Offline Baseline 统一接入方案（步骤 2）

## 1. 文档状态

- 目标目录：`E:\code\AAA-Memory\Offline`
- 当前阶段：实现步骤 2——代码阅读、架构设计、文件和接口规划
- 本文不包含步骤 3 的验证方案，也不代表已经开始修改实现代码
- benchmark 的实际运行入口必须位于：
  - `Offline/src/benchmarks/memgallery_harness`
  - `Offline/src/benchmarks/wma_harness`
- `E:\code\AAA-Memory\Mem-Gallery` 和
  `E:\code\AAA-Memory\WorldMemArena` 只作为数据、prompt、官方指标和已有实现的参考来源，
  不作为 Offline 运行时必须导入的 Python 包
- 后续第三个 benchmark 目标是 `E:\code\AAA-Memory\H2HMEM-main`；MemEye 暂不在
  当前 baseline 运行范围内

## 2. 最终目标

让 `Offline/baselines` 中的所有有效 baseline 项目，通过统一入口运行
Mem-Gallery 和 WorldMemArena（WMA）两个 benchmark，并与当前 HiveMem 形成可比较的对照组。

需要满足：

1. 两个 benchmark 共用同一套 baseline 生命周期接口。
2. 相同实验变量只从 `Offline/configs/defaults.json` 读取一次。
3. 所有 baseline 使用相同的回答模型、回答 prompt、top-k、超时、重试、judge 配置和数据范围。
4. baseline 自身特有的算法结构可以不同，但不得在 adapter 中偷偷替换模型、top-k 或数据范围。
5. WMA 必须按照 checkpoint 流式写入，禁止看到未来 session。
6. 输出继续兼容现有 results、retrieval trace、metrics 和 WMA 官方 evaluator。
7. 第三方项目尽量不改源码；只有硬编码阻止配置注入时才做最小修改。
8. 不把全部第三方依赖安装进 Offline 主环境。

## 3. 当前代码结构和问题

### 3.1 已有可复用能力

`src/embedding/chunk_builder.py` 已提供统一 `Chunk`：

```python
@dataclass
class Chunk:
    chunk_id: str
    text: str
    images: list[str]
    metadata: dict[str, Any]
```

并且已经实现：

- `build_chunks_from_data()`：Mem-Gallery 对话转 round-level chunk。
- `build_wma_chunks_from_data()`：WMA session 对话转 round-level chunk。
- 统一的 `session_id`、`dialogue_id`、`image_ids`、`image_paths`、timestamp 等元数据。

因此 baseline 层直接消费 `Chunk`，不再新增一套 Turn/Dialogue 数据模型。

现有回答链也应继续复用：

- `src/benchmarks/memgallery_harness/runner/answer_client.py`
- `src/benchmarks/wma_harness/runner/answer_client.py`
- 两个 harness 的 prompt 和 metrics 模块

### 3.2 当前耦合点

`memgallery_harness/eval_memgallery.py` 中的
`SimpleMemoryMemGalleryAdapter` 直接创建 `SimpleMemoryIndex`。

`wma_harness/eval_wma.py` 中的 `WMAIndexAdapter` 也直接创建
`SimpleMemoryIndex`，并通过 `allowed_session_ids` 实现 checkpoint 过滤。

这导致：

- 只能运行 HiveMem 生成的索引。
- 其他 baseline 没有统一的写入、session 结束、检索和清理入口。
- WMA 的未来信息隔离依赖 HiveMem 特有的过滤参数。
- memory metrics 默认假设磁盘上存在 HiveMem 的 `memories.jsonl`。

### 3.3 有效 baseline 清单

有效项目共 7 个：

1. `AUGUSTUSMemory`
2. `M2A`
3. `MIRIX`
4. `OmniSimpleMem`
5. `MMA-main`
6. `MemVerse-main`
7. `m3-agent-master`

以下内容不是 baseline：

- `default_config`：MemEngine 配置模板。
- `_clients`：旧的通用客户端目录。
- `*.zip`：源码归档。
- `MMA-main/MMA-Bench`：MMA 自带 benchmark，不是记忆方法。

## 4. 总体架构

```text
Mem-Gallery/WMA dataset
        |
        v
existing chunk_builder.py
        |
        v
BaselineAdapter protocol
        |
        +-- HiveMemAdapter (主环境内运行)
        |
        +-- BaselineProcess (持久子进程 + JSONL RPC)
                |
                +-- MemEngineAdapter (AUGUSTUSMemory)
                +-- OmniSimpleMemAdapter
                +-- M2AAdapter
                +-- MirixFamilyAdapter (MIRIX/MMA)
                +-- MemVerseAdapter
                +-- M3AgentAdapter
        |
        v
canonical RetrievedMemory[]
        |
        v
existing answer_client + prompt + metrics
        |
        v
existing result files / WMA pipeline_qa.jsonl
```

设计重点是让 adapter 只负责记忆系统，不负责 benchmark 最终回答。
最终回答统一交给现有 answer client，保证回答模型和 prompt 相同。

## 5. 统一 Python 接口

### 5.1 `BaselineAdapter`

文件：`src/benchmarks/baseline_runtime/protocol.py`

```python
from abc import ABC, abstractmethod
from pathlib import Path

from embedding.chunk_builder import Chunk


class BaselineAdapter(ABC):
    @abstractmethod
    def reset(self, sample_id: str, state_dir: Path) -> None:
        """创建当前 sample 的空白记忆状态。"""

    @abstractmethod
    def ingest(self, chunk: Chunk) -> None:
        """按顺序写入一个 dialogue round。"""

    @abstractmethod
    def end_session(self, session_id: str) -> None:
        """通知 baseline 当前 session 已结束并刷新内部缓冲。"""

    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """只检索，不更新记忆状态。"""

    @abstractmethod
    def snapshot(self) -> list[MemoryRecord]:
        """导出当前可见记忆，用于统计和 provenance。"""

    def close(self) -> None:
        """释放数据库、文件句柄、后台线程等资源。"""
```

### 5.2 检索请求

```python
@dataclass(frozen=True)
class RetrievalRequest:
    query_id: str
    text: str
    category: str
    top_k: int
    query_image: str | None = None
    visible_session_ids: tuple[str, ...] = ()
    query_vector: list[float] | None = None
```

字段说明：

- `query_id`：稳定 ID，用于 resume、trace 和错误定位。
- `text`：所有 baseline 都必须可用的原始问题。
- `category`：VS、VR、TTL、VFR 等 benchmark 类型。
- `top_k`：统一从根配置读取。
- `query_image`：视觉问题可选图片。
- `visible_session_ids`：WMA checkpoint 当前允许看到的 session。
- `query_vector`：HiveMem 可复用 query cache；其他 baseline 可以忽略。

### 5.3 统一检索结果

```python
@dataclass
class RetrievedMemory:
    memory_id: str
    text: str
    score: float | None = None
    session_id: str = ""
    source_dialogue_ids: list[str] = field(default_factory=list)
    image_ids: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    items: list[RetrievedMemory]
    trace: dict[str, Any]
```

`RetrievedMemory` 可直接转换为当前 answer client 使用的 `memory_items`。
adapter 不得把 backend 私有对象直接传给 harness。

### 5.4 Memory snapshot

```python
@dataclass
class MemoryRecord:
    memory_id: str
    text: str
    session_id: str
    source_dialogue_ids: list[str]
    image_ids: list[str]
    image_paths: list[str]
    backend_type: str
    metadata: dict[str, Any]
```

snapshot 用于：

- 统计 baseline 实际保留的 memory 数量和 token 数。
- 将 backend memory ID 映射回 session/dialogue。
- 在 trace 中记录真实 provenance。
- 检查 WMA checkpoint 是否存在未来记忆。

## 6. 子进程协议和依赖隔离

### 6.1 为什么必须隔离

当前依赖存在直接冲突：

- Offline：Python 3.10+、`openai<2`。
- M2A：Python 3.12+、`openai>=2.15`、NumPy 2.4+。
- MIRIX/MMA：包含自己的数据库、Pydantic、FastAPI 和大量可选依赖。
- m3-agent：包含固定的模型和媒体处理依赖。

因此不能把所有 requirements 合并进根 `pyproject.toml`。

### 6.2 RPC 操作

`BaselineProcess` 启动一个持续运行的 worker，通过 stdin/stdout 交换一行一个 JSON：

```json
{"id": 1, "op": "init", "baseline": "M2A", "config": {}, "state_dir": "..."}
{"id": 2, "op": "reset", "sample_id": "sample_001", "state_dir": "..."}
{"id": 3, "op": "ingest", "chunk": {}}
{"id": 4, "op": "end_session", "session_id": "S01"}
{"id": 5, "op": "retrieve", "request": {}}
{"id": 6, "op": "snapshot"}
{"id": 7, "op": "close"}
```

响应统一为：

```json
{"id": 5, "ok": true, "result": {}}
```

或：

```json
{
  "id": 5,
  "ok": false,
  "error": {
    "type": "RuntimeError",
    "message": "...",
    "traceback": "..."
  }
}
```

约束：

- worker 的协议输出只能写 stdout。
- baseline 自带的 print/log 重定向到 stderr。
- 每个请求有独立 timeout。
- worker 异常退出时，harness 记录失败，不自动切换成其他算法。
- 每个 sample 使用独立 state directory。
- `close()` 后必须等待子进程退出，不能遗留下载或服务进程。

## 7. 配置归一

### 7.1 唯一实验配置来源

`configs/defaults.json` 是实验配置的唯一事实来源。需要补充以下公共字段：

```json
{
  "answer_model": "...",
  "answer_base_url": "...",
  "answer_temperature": 0.0,
  "num_predict": 512,
  "request_timeout": 180,
  "retries": 2,
  "top_k": 5,

  "executor_model": "...",
  "executor_base_url": "...",
  "executor_temperature": 0.0,
  "executor_visual_input": "image",

  "embedding_model": "...",
  "embedding_base_url": "...",
  "embedding_dim": 2048,
  "embedding_api_key_env": "EMBEDDING_API_KEY",

  "baseline_worker_timeout": 180,
  "baseline_strict_config": true
}
```

实际字段应复用已有名称；只新增目前不存在的字段，不建立第二套同义配置。

### 7.2 `configs/baselines.json`

该文件只是 baseline 注册表，不保存实验参数：

```json
{
  "AUGUSTUSMemory": {
    "adapter": "memengine",
    "source_root": "baselines/AUGUSTUSMemory/upstream",
    "python_env": "MEMENGINE_PYTHON"
  },
  "M2A": {
    "adapter": "m2a",
    "source_root": "baselines/M2A",
    "python_env": "M2A_PYTHON"
  }
}
```

`python_env` 是环境变量名，不在仓库中提交机器专用绝对路径。
未设置时可以使用当前解释器，但依赖不兼容时必须给出清晰错误。

### 7.3 “所有 config 一样”的精确定义

以下受控变量必须完全一致：

- benchmark 数据文件和样本范围。
- 可见 session/checkpoint。
- `top_k`。
- 最终 answer model、endpoint、temperature、max tokens、timeout、retry。
- 最终 answer prompt 和图片输入规则。
- memory 构建阶段使用的 LLM model/endpoint/temperature。
- 共享 embedding model/endpoint/dimension（baseline 支持配置时）。
- judge model 和 judge prompt。

以下是算法定义的一部分，不能强行设成相同：

- AUGUSTUS 的 concept graph 和检索权重。
- OmniSimpleMem 的 hybrid search/graph 结构。
- MIRIX/MMA 的 memory partitions。
- MemVerse 的 core/episodic/semantic RAG。
- M3-Agent 的 episodic/semantic graph。

adapter 启动时生成 `resolved_config`，写入 run manifest。若 native backend
最终使用的模型配置与根配置不一致且 `baseline_strict_config=true`，直接失败，不能静默 fallback。

## 8. benchmark 执行流程

### 8.1 Mem-Gallery

对每个 dataset 文件：

1. 读取 JSON。
2. 使用现有 `build_chunks_from_data()` 生成有序 chunks。
3. `adapter.reset(dataset_name, state_dir)`。
4. 按 chunk 顺序调用 `ingest()`。
5. session 变化时调用 `end_session()`。
6. 所有 session 写入完毕后开始 QA。
7. 每个问题调用 `retrieve()`。
8. 将统一 memory items 交给现有 `VLMAnswerClient.answer()`。
9. 保持当前 `results.json` 和 `retrieval_trace.jsonl` 字段。
10. 调用 `snapshot()` 写入 baseline memory 统计。
11. `close()`。

Mem-Gallery QA 不允许调用 baseline 自带的最终问答函数，否则不同 baseline
会使用不同 prompt 和不同 answer agent。

### 8.2 WMA

WMA 必须改成在线 checkpoint 流程：

1. 读取 sample 并计算 ordered sessions。
2. 把每个 checkpoint 映射到它的最后一个 visible session。
3. `adapter.reset(sample_id, state_dir)`。
4. 按 session 顺序 ingest chunks。
5. 调用 `end_session(session_id)` 强制刷新 baseline 缓冲。
6. 当前 session 达到 checkpoint 后，立即处理该 checkpoint 的所有问题。
7. QA 只能 retrieve，不能写回 memory。
8. 处理完成后继续 ingest 后续 session。
9. 保持 `pipeline_qa.jsonl` 与 WMA 官方 evaluator 兼容。

不能对 native baseline 采用“先写入全部 session，再传 allowed_session_ids”的方式，
因为多数 backend 没有可靠的 session filter。

HiveMemAdapter 可以继续使用已有完整索引和 `allowed_session_ids`，但必须通过同一接口返回结果。

## 9. 各 baseline 的具体适配

### 9.1 HiveMem

文件：`adapters/hivemem.py`

- 封装现有 `SimpleMemoryIndex` 和 `GraphExpandedIndex`。
- `ingest()` 为 no-op，因为 HiveMem memory 已由现有构建流程生成。
- `retrieve()` 使用 `query_vector` 和已有 query embedding cache。
- WMA 继续传 `allowed_session_ids`。
- graph retrieval 在 WMA 继续禁止，直到 graph statistics 能按 prefix 构建。
- 返回现有 `MemoryHit.to_context_item()` 的规范化结果。

### 9.2 AUGUSTUSMemory

文件：`adapters/memengine.py`

使用 MemEngine adapter：

```python
MemEngineAdapter(method="AUGUSTUSMemory", ...)
```

adapter 负责：

- 只导入目标 memory class，避免 `memengine.__init__` 一次加载所有可选依赖。
- 将根配置转换为 `MemoryConfig`。
- `ingest()` 调用 native `store(observation)`。
- `retrieve()` 调用 native `recall(observation)`。
- 解析 LinearStorage/TagGraphStorage，导出 snapshot。
- 在 observation 中保留 `dialogue_id`、`session_id`、图片 ID 和路径。
- `MGMemory` 不再作为可运行 baseline 暴露；AUGUSTUSMemory vendored 上游中的
  同名内部类只作为上游包结构保留。

### 9.3 OmniSimpleMem

文件：`adapters/omni_simplemem.py`

- 创建 `OmniMemoryOrchestrator(config, data_dir=state_dir)`。
- text/caption 模式调用 `add_text()`。
- image 模式调用官方图像入口，同时保留 benchmark caption。
- metadata tags 写入 session/dialogue/image ID。
- `retrieve()` 调用 `query(query, top_k)`。
- `snapshot()` 从 hot storage/MAU 导出规范化记录。
- `close()` 必须调用 orchestrator 的关闭方法。

### 9.4 M2A

文件：`adapters/m2a.py`

- 使用 `M2ASystem`、`RawMessageStore`、`SemanticStore` 和 `MemoryManager`。
- 每个 sample 创建独立 raw DB 和 semantic DB。
- 写入阶段复用官方 evaluation wrapper 的 update-only 模式。
- 检索阶段直接使用 `SemanticStore.hybrid_search()`，不调用 `question()`，以保证最终回答统一。
- 不采用已有外部 adapter 的 lexical fallback，也不无提示地把每个 turn 强写进 semantic store。
- 如果 LLM 没有按算法要求写入 memory，应在 trace 中如实记录，而不是改变算法。

M2A 当前 schema 对 embedding dimension 有硬编码。如果共享 embedding dimension 与 schema 不同，
需要把 dimension 改为构造参数；这是配置参数化，不改变检索算法。

可能需要最小修改：

- `baselines/M2A/agent/config.py`：embedding dimension 配置字段。
- `baselines/M2A/agent/stores/semantic.py`：Milvus schema 使用配置维度。

### 9.5 MIRIX / MMA

文件：`adapters/mirix_family.py`

共享数据库初始化、写入、snapshot 和 partition 查询代码：

```python
MirixFamilyAdapter(backend="MIRIX", confidence=False)
MirixFamilyAdapter(backend="MMA", confidence=True)
```

- 使用各项目自己的 `AgentWrapper.send_message(..., memorizing=True)`。
- 每个 sample 创建独立临时数据库目录。
- session 结束时强制 flush/absorb 已缓存内容。
- snapshot 覆盖 episodic、semantic、procedural、resource、knowledge vault 等 partition。
- MIRIX 按 native manager 搜索结果规范化。
- MMA 必须使用其 confidence module 对候选进行 Source/Time/Consensus 排序；不能退化成普通 MIRIX。
- 问题检索不能通过 `send_message()`，防止 QA 修改记忆。
- runtime 配置应覆盖 upstream 中硬编码的 endpoint 和 model whitelist。

### 9.6 MemVerse

文件：`adapters/memverse.py`

- 为每个 sample 创建独立的 conversation、memory_chunks 和 LightRAG 工作目录。
- 复用 MemVerse 的 core、episodic、semantic memory 生成逻辑。
- 查询时分别查询三类 RAG，而不是只调用当前 `rag_retrieve()` 中的 core memory。
- 最终回答仍由 Offline answer client 完成，不调用 `generate_final_answer()`。
- `use_pm` 默认关闭；Parametric Memory 属于另一个需要训练和部署的组件，若以后启用必须作为明确实验变体。
- 写入 LightRAG 时保留 source dialogue marker，以便恢复 provenance；marker 在传给回答模型前移除。
- adapter 显式注入根配置，覆盖当前代码中的 `gpt-4o-mini` 和
  `text-embedding-3-small` 硬编码。

优先在 adapter 中构造 LightRAG 和 client；只有无法注入时，才最小修改：

- `baselines/MemVerse-main/orchestrator.py`
- `baselines/MemVerse-main/MemoryKB/build_memory.py`

修改内容仅限：路径参数化、model/base_url 参数化、暴露纯检索函数。

### 9.7 m3-agent

文件：`adapters/m3_agent.py`

m3-agent 原生输入是长视频片段，而 Mem-Gallery/WMA 输入是对话 round 和静态图片，
两种数据没有完全等价的官方入口。因此注册名称应为：

```text
M3-Agent-caption
```

兼容策略：

- 一个 dialogue round 对应一个递增 clip ID。
- round 文本和 benchmark caption 写为 episodic memory。
- 若启用语义抽取，使用统一 executor model 生成 semantic memory。
- 图片作为当前 clip 的视觉内容，无法提供音频时不伪造 voice memory。
- 使用 upstream `VideoGraph`、`process_memories`、text node 和 graph retrieval。
- query embedding 使用根配置，不使用源码中硬编码的 Azure client 和
  `text-embedding-3-large`。
- clip ID 与 `session_id/dialogue_id` 的 sidecar 映射用于 provenance 和 WMA checkpoint。

run manifest 必须记录：

```json
{
  "baseline": "M3-Agent-caption",
  "compatibility_mode": "dialogue_round_as_clip",
  "audio_enabled": false
}
```

这样能运行并保持结果可解释，但不能把它描述为官方完整视频设置下的 M3-Agent 结果。

## 10. provenance 和结果兼容

每次 ingest 后，adapter 维护：

```text
backend memory id
    -> source chunk ids
    -> source dialogue ids
    -> session ids
    -> image ids/paths
```

优先级：

1. backend 原生 metadata。
2. 写入时附带的 source marker/tag。
3. 每个 session 结束时通过 snapshot diff 关联新增 memory ID。

不得仅根据当前 session 猜测旧 memory 的来源。对于合并/更新记忆，
`source_dialogue_ids` 应取所有参与源记录的并集。

现有输出字段继续保留：

- `retrieved_ids`
- `retrieved_source_groups`
- `retrieval_top_k`
- `memory_context`
- WMA 的 `visible_sessions`、`gold_sessions` 和 future evidence 字段

run manifest 新增：

- baseline 名称和 adapter 名称。
- baseline 源码目录。
- worker Python 版本。
- resolved common config（密钥只记录环境变量名，不记录值）。
- adapter capabilities。
- compatibility mode。
- baseline 源码版本/目录 hash（可获得时）。

## 11. 文件级修改清单

### 11.1 新增文件

```text
src/benchmarks/baseline_runtime/__init__.py
src/benchmarks/baseline_runtime/protocol.py
src/benchmarks/baseline_runtime/registry.py
src/benchmarks/baseline_runtime/process.py
src/benchmarks/baseline_runtime/worker.py
src/benchmarks/baseline_runtime/config.py
src/benchmarks/baseline_runtime/output_layout.py
src/benchmarks/baseline_runtime/provenance.py
src/benchmarks/baseline_runtime/adapters/__init__.py
src/benchmarks/baseline_runtime/adapters/hivemem.py
src/benchmarks/baseline_runtime/adapters/memengine.py
src/benchmarks/baseline_runtime/adapters/omni_simplemem.py
src/benchmarks/baseline_runtime/adapters/m2a.py
src/benchmarks/baseline_runtime/adapters/mirix_family.py
src/benchmarks/baseline_runtime/adapters/memverse.py
src/benchmarks/baseline_runtime/adapters/m3_agent.py
configs/baselines.json
baselines/README.md
```

职责：

- `protocol.py`：唯一公共数据结构和 adapter 抽象接口。
- `registry.py`：名称解析、adapter factory 和 capability 查询。
- `process.py`：父进程侧 worker 管理和 JSONL RPC。
- `worker.py`：子进程命令分发。
- `config.py`：读取根 defaults，进行严格字段校验和 native 映射。
- `output_layout.py`：统一结果目录下的 memory snapshot 和原生 state 路径。
- `provenance.py`：backend ID、chunk、dialogue 和 session 的关联。
- `adapters/*`：只包含各 baseline 特有的薄适配逻辑。
- `configs/baselines.json`：源码位置和解释器环境变量。
- `baselines/README.md`：环境准备、baseline 名称和入口说明。

### 11.2 修改 Offline 主代码

#### `configs/defaults.json`

- 补齐 answer temperature。
- 补齐 embedding endpoint/API key env。
- 增加 worker timeout 和 strict config。
- 保持已有字段兼容，避免重命名造成当前 HiveMem 命令失效。

#### `src/benchmarks/memgallery_harness/eval_memgallery.py`

- 删除文件内的 HiveMem 专用 adapter 类，迁入 `adapters/hivemem.py`。
- 新增 `--baseline`。
- `--index-root` 只对 HiveMem 必需。
- 增加 baseline state/output directory 参数。
- 原生 state 默认写入 `<result-dir>/memory/datasets/<dataset>`。
- 规范化 memory snapshot 写入 `<result-dir>/memory/memory_snapshot.jsonl`。
- 使用 chunk builder 构建并写入 baseline。
- 统一调用 `RetrievalResult`。
- 保留当前 resume、prompt、answer 和结果结构。

#### `src/benchmarks/wma_harness/eval_wma.py`

- 删除文件内的 HiveMem 专用 adapter 类，迁入 `adapters/hivemem.py`。
- 新增 `--baseline`。
- 将 `prepare_sample_jobs()` 拆成“流式 ingest/checkpoint retrieval”和“并发 answer”两部分。
- native baseline 按 checkpoint 查询，HiveMem 使用 prefix filter。
- 保持 answer 阶段的线程池；同一个 stateful adapter 不并发 ingest/retrieve。
- 保持 `to_pipeline_qa_record()` 输出兼容。
- 原生 state 默认写入 `<result-dir>/memory/datasets/<sample>`。
- 规范化 memory snapshot 写入 `<result-dir>/memory/memory_snapshot.jsonl`。

#### `src/benchmarks/memgallery_harness/runner/metrics.py`

- `write_memory_metrics()` 增加接受 adapter snapshot 的路径。
- HiveMem 继续读取原有文件布局。
- 其他 baseline 使用规范化 `MemoryRecord` 统计。

#### `src/benchmarks/*/runner/answer_client.py`

- 不重写客户端。
- 只在必要时补一个从 `RetrievedMemory` 转换为 `memory_items` 的公共 helper。
- WMA 继续继承 Mem-Gallery answer client。

### 11.3 baseline 目录整理

保留：

```text
baselines/AUGUSTUSMemory/README.md
```

并修改 README，使其指向 Offline 新的 baseline registry，而不是当前不存在的
`eval_framework.cli`。

### 11.4 `default_config` 和 `_clients`

`baselines/default_config`：

- 当前阶段保留。
- 它为 AUGUSTUSMemory 提供 storage、operation、retrieval 等算法结构模板。
- adapter 必须用根 defaults 覆盖其中的 model、endpoint、embedding 和 top-k。
- 后续可随 MemEngine 一起移动到 `baselines/memengine/default_config`，但不应把它当成全项目公共配置。

`baselines/_clients`：

- 当前 Offline 代码没有消费者。
- 最终回答已由 harness answer client 统一负责。
- baseline 内部工具调用仍使用各项目 native client，但配置来自根 defaults。
- baseline runtime 接入完成后可以删除 `_clients`。

zip 文件：

- 不进入 registry。
- 不参与 import、运行或版本判断。
- 是否删除属于单独清理操作，不影响本方案实现。

## 12. 实现顺序

本节是步骤 2 的代码实施顺序，不是步骤 3 的验证方案。

1. 建立 `protocol.py`、config resolver 和 registry。
2. 把两个现有 HiveMem adapter 迁入统一接口，保证现有行为不变。
3. 改造 Mem-Gallery harness 使用统一 adapter。
4. 改造 WMA 为流式 checkpoint 生命周期。
5. 接入已有参考实现覆盖的五类：MemEngine、OmniSimpleMem、M2A、MIRIX。
6. 在 MIRIX family adapter 中增加 MMA confidence 分支。
7. 实现 MemVerse 独立工作目录和三类 RAG 查询。
8. 实现并明确标注 M3-Agent caption compatibility mode。
9. 接入 snapshot/provenance 和统一 memory metrics。
10. 最后处理重复 MemEngine 源码、`_clients` 和 README 清理。

任何一步都不应改变现有 answer prompt、judge prompt 或结果字段含义。

## 13. 明确不采用的方案

### 13.1 不直接 import WorldMemArena 的 `eval_framework`

原因：

- 用户指定运行目标是 Offline 自己的 `src/benchmarks`。
- 会让 Offline 依赖外部兄弟仓库的目录位置。
- 其 schemas、config 和 pipeline 与 Offline 当前结构不同。
- 部分已有 adapter 为了跑通使用了 lexical fallback 或强制写入，会改变 baseline 行为。

可以参考并精简其已有 adapter，但 Offline 必须拥有自己的最小接口实现。

### 13.2 不为每个 benchmark 写一套 baseline adapter

同一个 adapter 同时服务 Mem-Gallery 和 WMA。benchmark 差异只存在于数据调度和问题格式，
不应复制 backend 接入代码。

### 13.3 不让 baseline 自己生成最终答案

M2A、MemVerse、MIRIX 等都带有自己的回答 agent。如果直接使用，会导致：

- prompt 不同。
- answer model 或温度不同。
- 有的 QA 会写回 memory。
- 无法公平比较纯 memory 能力。

所以 baseline 只输出检索结果，最终答案始终由 Offline answer client 生成。

### 13.4 不合并所有 requirements

第三方 baseline 通过独立 Python worker 隔离。根 `pyproject.toml` 只保留 Offline/HiveMem
自身依赖，不加入 M2A、MIRIX、MMA 或 m3-agent 的完整依赖集合。

## 14. 已知边界

- m3-agent 在这两个 benchmark 上只能形成明确标注的对话兼容变体。
- MemVerse Parametric Memory 默认不启用，启用时必须作为单独实验变体。
- WMA graph retrieval 继续保持关闭，直到图统计能够按照 checkpoint prefix 构建。
- 某些 baseline 会把多个源 round 合并成一条记忆，provenance 必须允许一对多。
- 如果 baseline 不支持根配置指定的 embedding dimension，应先参数化 schema；不能静默换成另一个 embedding。
- baseline worker 的环境路径属于机器部署信息，不应写死在仓库配置中。

---

本文到此仍属于实现步骤 2。实现步骤 3（验证方案）需在用户明确回复“下一步”后再给出。
