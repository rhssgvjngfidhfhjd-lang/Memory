### 建边方式
0.新图的 node 使用原生 chunk。后续新建图，不保留当前旧图中的 `prev/next` 时间边、LLM 事件关系边、旧实体共同性边或旧属性共同性边。
1.定义 Ti，仅由 chunk 文本明确表达的属性集合。定义 Vi，仅由该 chunk 所含原图直接观察到的属性集合。

Ti 和 Vi 中的属性 \(a\) 统一定义为：

\[
a=(attribute,value)
\]

因此：

\[
T_i,V_i\subseteq\{(attribute,value)\}
\]

`entity` 可以保留为每个属性的可选元数据，用于溯源和检查属性在原 chunk 中属于谁，但不是属性 \(a\) 的一部分。`entity` 不参与属性相等判断、集合相交、\(df(a)\) 统计、边权计算或检索时的属性编码。

两个不同实体只要具有相同的 `(attribute, value)`，就视为共享同一属性。例如 Lumi 和 Kiki 都具有 `(coat_color, white)` 时，该属性可用于两个 chunk 之间的后续建边判断。

1.1 属性规范化与相等判断

属性名和属性值先经过确定性规范化，包括：

- 转为小写并去除多余空格。
- 统一单复数形式。
- 将确定的同义属性名映射为同一标准属性名，例如 `color` 与 `coat_color`。
- 将确定的同义属性值映射为同一标准值，例如 `sitting` 与 `seated`。

规范化后的属性为：

\[
a=(\operatorname{norm}(attribute),\operatorname{norm}(value))
\]

两个属性仅当规范化后的属性名和属性值都完全相等时，才视为同一属性。不使用 embedding 相似度判断两个属性是否相同。

2.Vi 的属性范围

\(V_i\) 的属性 key 使用封闭集合。VLM 只能从封闭集合中选择属性 key，不得自行创造新的视觉属性名。初版封闭属性 key 集合为：

\[
\mathcal A_V=
\{\text{color},\text{appearance},\text{shape},\text{material},
\text{texture},\text{pattern},\text{count},\text{visible\_state},
\text{action},\text{pose},\text{position},\text{spatial\_relation},
\text{ocr\_text}\}
\]

在视觉属性 \(a=(attribute,value)\) 中，`attribute` 是上述封闭集合中的视觉描述维度，`value` 是从图片中直接观察到的具体信息。例如：

\[
(\text{color},\text{blue}),\quad
(\text{material},\text{wood}),\quad
(\text{pose},\text{sitting}),\quad
(\text{count},3)
\]

因此 `color` 等 key 只限定 VLM 提取的视觉维度，`blue` 等 value 才是图片中的实际可见内容。不将 `blue notebook` 这样的完整描述短语直接作为属性 \(a\)。

具体含义如下：

- 颜色：`color`
- 整体可见外观：`appearance`
- 形状：`shape`
- 材质：`material`
- 表面纹理：`texture`
- 图案：`pattern`
- 数量：`count`
- 可见状态：`visible_state`
- 动作：`action`
- 姿态：`pose`
- 空间位置：`position`
- 实体之间的空间关系：`spatial_relation`
- 图片中可见的文字：`ocr_text`

为了使文本属性与视觉属性能够形成跨模态交集，上述视觉属性 key 同时加入 \(T_i\) 的允许属性范围。当 chunk 文本明确描述颜色、形状、动作、姿态等信息时，\(T_i\) 使用与 \(V_i\) 相同的标准 key。例如：

\[
T_i=\{(\text{color},\text{blue})\},\qquad
V_j=\{(\text{color},\text{blue})\}
\]

两者的区别只在信息来源：\(T_i\) 仍然只能从 chunk 文本提取，\(V_i\) 仍然只能从 chunk 原图提取。

{
  "chunk_id": "chunk_001",
  "summary": "...",
  "Ti": [
    {
      "entity": "Lumi",
      "attribute": "breed",
      "value": ["Maltese"]
    }
  ],
  "Vi": [
    {
      "entity": "Lumi",
      "attribute": "coat_color",
      "value": ["white"]
    },
    {
      "entity": "Lumi",
      "attribute": "pose",
      "value": ["sitting"]
    }
  ]
}

3.关键是 Prompt 必须明确隔离来源：
Ti：只能使用文本，不能使用图片补充。
Vi：只能使用原图中可直接观察的内容，不能使用对话文本或 caption 补充。
同一事实如果文本和图片都能独立支持，可以同时出现在 Ti 和 Vi。
没有图片时 Vi=[]。
无法从图片直接确定的属性不能进入 Vi，例如人物职业、物品主人、偏好、因果关系等。

3.1 一个 chunk 只生成一个 summary

新建库流程必须保证：

\[
1\ \text{chunk}\rightarrow1\ \text{node}\rightarrow1\ \text{summary}
\]

VLM 需要对整个 chunk 输出一个完整 summary，即使 chunk 包含多个事实或主题，也不得拆分成多个 summary 或多个节点。

需修改当前 Prompt 中鼓励拆分的复数表达，特别是 `one memory item per independent fact` 和 `Split unrelated facts into separate memory items`。同时修改解析器，使建库路径不再接受并写入同一 chunk 的多个 summary。

3.2 VLM 输出格式

VLM 对每个 chunk 必须输出且只输出一个 JSON object，不输出 JSON array、多个 object 或 JSON 之外的解释文本。顶层结构为：

```json
{
  "summary": "整个 chunk 的唯一完整摘要",
  "Ti": [
    {
      "entity": "Lumi",
      "attribute": "breed",
      "value": ["Maltese"]
    }
  ],
  "Vi": [
    {
      "entity": "Lumi",
      "attribute": "color",
      "value": ["white"]
    }
  ]
}
```

`summary` 是当前 chunk 的唯一 summary；`Ti` 是仅从 chunk 文本中提取的属性数组；`Vi` 是仅从 chunk 原图中直接观察并提取的属性数组。`entity` 是可选溯源元数据。

同一 `entity + attribute` 下的多个值统一放入 `value` 数组；即使只有一个值，`value` 也使用单元素数组，不使用字符串和数组两种类型。例如：

```json
{
  "entity": "notebook",
  "attribute": "color",
  "value": ["blue", "white"]
}
```

在进入属性规范化和图计算前，将 `value` 数组展开为独立的属性 \(a\)：

\[
(\text{color},\text{blue}),\quad
(\text{color},\text{white})
\]

VLM 输出的常见 JSON 语法错误继续使用 `json_repair` 修复，并保留现有 fallback 能力。解析异常暂不作为本次新图方法的单独设计项。

4.候选边生成规则

对于两个 chunk 节点 \(i,j\)，定义文本—文本共享属性集合：

\[
S^{text}_{ij}=T_i\cap T_j
\]

定义文本—视觉跨模态共享属性集合：

\[
S^{cross}_{ij}=(T_i\cap V_j)\cup(V_i\cap T_j)
\]

只要上述两类共享属性中存在至少一个规范化后的属性，就生成候选无向边 \((i,j)\)：

\[
S^{text}_{ij}\cup S^{cross}_{ij}\neq\varnothing
\quad\Longrightarrow\quad
(i,j)\text{ 进入候选边集合}
\]

不使用纯视觉—视觉属性交集 \(V_i\cap V_j\) 生成候选边，\(V_i\cap V_j\) 也不进入边权计算。

5.边的权重 \(w\)

边权重用于建图阶段的边数量筛选：当一个 chunk 节点具有过多候选边时，根据边权重保留 top-\(k\) neighbors。该权重暂不定义为检索阶段的相关性分数。

对属性 \(a\) 使用 IDF weighting：

\[
\operatorname{IDF}(a)=\log\frac{N}{df(a)}
\]

其中：

- \(N\)：当前 dataset 实际构建的这张图中的 chunk 节点总数。
- \(df(a)\)：当前 dataset 的图中包含属性 \(a\) 的 chunk 节点数。对于同一 chunk，属性 \(a\) 即使重复出现也只计数一次。
- \(1\le df(a)\le N\)，因此 \(\operatorname{IDF}(a)\ge 0\)。
- 属性越稀有，IDF 权重越高；当属性出现在所有 chunk 节点中时，其权重为 0。

MemGallery 和 H2HMEM 在当前 dataset/sample 内统计。WMA 在每个 checkpoint 只使用截至当前可见的 prefix chunk 重新计算 \(N\)、\(df(a)\)、边权和剪枝结果；未来不可见的 chunk 不参与当前图的任何统计或剪枝。

文本属性重合与跨模态属性重合的系数暂定为 \(\alpha=\beta=1\)，因此候选边 \((i,j)\) 的权重为：

\[
w_{ij}
=
\sum_{a\in S^{text}_{ij}}\log\frac{N}{df(a)}
+
\sum_{a\in S^{cross}_{ij}}\log\frac{N}{df(a)}
\]

其中，\(S^{text}_{ij}\) 和 \(S^{cross}_{ij}\) 按第 4 节定义。

6.全局度约束剪枝

\(K_{\text{edge}}\) 定义为可配置超参数，当前默认值为：

\[
K_{\text{edge}}=4
\]

对所有候选无向边计算 \(w_{ij}\) 并按权重从高到低排序。依次遍历候选边，仅当：

\[
\deg(i)<K_{\text{edge}}
\quad\text{且}\quad
\deg(j)<K_{\text{edge}}
\]

时将无向边 \((i,j)\) 加入最终图，并同时增加两端节点的 degree。因此最终每个节点都满足：

\[
\deg(i)\le K_{\text{edge}}
\]

6.1 最终边的存储结构

每条通过全局度约束剪枝后保留的最终无向边，暂定保存以下字段：

```json
{
  "source": "chunk_001",
  "target": "chunk_023",
  "weight": 4.27,
  "shared_text": [
    {
      "attribute": "preference",
      "value": "hiking"
    }
  ],
  "shared_cross": [
    {
      "attribute": "color",
      "value": "blue"
    }
  ]
}
```

- `source` 和 `target` 标识无向边两端的 chunk 节点。
- `weight` 保存建图剪枝使用的 \(w_{ij}\)。
- `shared_text` 保存 \(S^{text}_{ij}=T_i\cap T_j\)。
- `shared_cross` 保存 \(S^{cross}_{ij}=(T_i\cap V_j)\cup(V_i\cap T_j)\)。
- `entity` 不进入边属性。

边 \((i,j)\) 上的全部共享属性集合为：

\[
B_{ij}=S^{text}_{ij}\cup S^{cross}_{ij}
\]

### 检索方式
1.检索结果保持 5+2：先使用向量检索取 top-5 chunk 节点，记为 \(M_{\text{seed}}\)。这 5 个向量召回结果保持不变。

新图中的节点身份和最终返回的证据单位都是原生 chunk，但 vector top-5 的节点文本表示使用 VLM 生成的 summary embedding：

\[
E_{\text{node}}(i)=E(\operatorname{summary}_i)
\]

向量检索命中 summary 对应的节点后，返回的是该节点对应的完整原生 chunk。

Vector top-5 逻辑复用当前实现。先计算 summary 相似度：

\[
s_i^{text}=\operatorname{cos}(E(q),E(\operatorname{summary}_i))
\]

对配置为视觉类别的问题，如果该节点具有图片向量，则复用当前的 max-fusion：

\[
s_i=
\max\left(
s_i^{text},
\operatorname{cos}(E(q),E_{\text{image}}(m_i))
\right)
\]

其他情况使用 \(s_i=s_i^{text}\)。继续复用当前的 `ACTIVE` 状态和 `allowed_session_ids` 可见性过滤，然后按 \(s_i\) 从高到低取 5 个节点。这里复用的是现有 top-5 检索逻辑；存储对象和向量行将由旧 MAU 改为新的 chunk node。

2.在最终剪枝后的无向图中，收集与 \(M_{\text{seed}}\) 中任一节点直接相连的一跳邻居，排除已经存在于 vector top-5 中的节点，得到图候选节点 \(m_j\)。

3.对于候选节点 \(m_j\)，设边 \((m_i,m_j)\) 的共享属性集合为：

\[
B_{ij}=S^{text}_{ij}\cup S^{cross}_{ij}
\]

如果 \(m_j\) 同时与多个 vector top-5 seed 节点相连，先聚合它与全部 seed 节点之间的共享属性：

\[
B_j=
\bigcup_{m_i\in M_{\text{seed}},\,(m_i,m_j)\in E}
B_{ij}
\]

\(B_j\) 是集合，因此同一属性即使同时出现在多条 seed 连边中，也只计算一次。

4.对每个候选节点 \(m_j\) 计算查询相关分数：

\[
S(m_j\mid q)
=
\sum_{a\in B_j}
\operatorname{IDF}(a)
\operatorname{sim}(E(q),E(a))
\]

其中：

- \(E(q)\) 沿用当前的 query embedding。纯文本问题编码问题文本；带查询图片的问题沿用当前的多模态 query embedding。
- 对规范化后的属性 \(a=(attribute,value)\)，将它序列化为 `attribute: value` 文本，再使用与 query 相同的 embedding 模型得到 \(E(a)\)。例如 `(coat_color, white)` 序列化为 `coat_color: white`。
- 每个唯一属性只需编码一次，属性 embedding 可以预先计算并缓存。
- \(\operatorname{sim}\) 使用 cosine similarity：

\[
\operatorname{sim}(E(q),E(a))
=
\frac{E(q)\cdot E(a)}
{\lVert E(q)\rVert\lVert E(a)\rVert}
\]

其中，\(w_{ij}\) 只用于建图阶段的全局度约束剪枝；\(S(m_j\mid q)\) 用于当前查询下的图候选节点排序。

5.将一跳图候选节点按 \(S(m_j\mid q)\) 从高到低排序，选择得分最高的两个不同节点，追加到原 vector top-5 之后，得到最终 5+2 检索结果。

如果实际一跳图候选节点不足两个，是否补齐以及如何补齐暂不设计，等真实运行遇到后再确定。
