1.任务目标：Object-centric Visual Primitive Extractor
构建一个**面向视觉记忆的物体级视觉原语提取器**：
输入任意 RGB 图像，在不知道场景中有什么物体、无需指定类别的情况下，自动发现、定位并结构化表示可记忆的视觉实体。

2.说明
image：主要是当前E:\code\AAA-Memory项目里3个benchmark：（E:\code\AAA-Memory\Mem-Gallery，E:\code\AAA-Memory\H2HMEM-main，E:\code\AAA-Memory\WorldMemArena）的图片
Visual Primitive：(以下统称vp):可独立定位、跟踪、检索的视觉实体，优先提取最小的、完整的、可独立检索的主体，不对主体进行无意义拆分
举例：
独立人物、动物、商品、车辆、家具等完整实体。
身份证、海报、票据等完整文档。
天气面板、评分面板等完整且可独立检索的语义区域。
同类物体的多个实例分别输出。

排除：
孤立文字行、普通图标、装饰元素。
人的手、脚等身体部位，除非它本身是关注主体。
大面积无意义背景，如天空、墙壁、地面。
同一主体的重复框。
无法独立描述或检索的碎片。

3.实现步骤：
Step 1：Object Discovery
使用当前已有视觉模型qwen3vl-4b-instruct，发现场景中的物体：
Image
  ↓
qwen3vl-4b-instruct 一次调用
  ↓
Discovery + coarse bbox
输出举例：
[
  {
    "label": "green Tsingtao beer can",
    "bbox_norm": [245, 147, 736, 918]
  }
]
  ↓
规则校验 / 去重
  ↓
必要时只对不确定候选二次定位



Step2：Crop
接收原始 RGB 图像以及 Step 1 输出的物体标签和 bbox。完成图像方向统一、bbox 合法性检查、归一化坐标到像素坐标的转换，并将每个有效 bbox 裁剪为独立 VP 图片。每张原图保存一个 record.json，记录原图与多个 VP crop 之间的一对多关系。
原始图像 + Step 1 输出
  ↓
统一图像方向和 RGB 格式
  ↓
bbox 校验
  ↓
归一化坐标转像素坐标
  ↓
逐个裁剪 VP
  ↓
保存 crop + record.json

生成vp存放路径
visual_primitives/
    └── qwen3vl4b_v1/
        ├── run.json
        ├── items/
        │   ├── img_a81f203c/
        │   │   ├── record.json
        │   │   ├── vp_0001.jpg
        │   │   ├── vp_0002.jpg
        │   │   └── preview.jpg
        │   └── img_b98e0741/
        │       ├── record.json
        │       └── vp_0001.png
        ├── exports/
        │   ├── images.jsonl
        │   └── primitives.jsonl
        └── failures.jsonl


###明确当前版本不做什么
V1 非目标包括：
不做跨图片实体匹配和跟踪。
不生成 embedding 或检索索引。
不做主体分割或 mask。
不做 OCR 专项抽取。
不构建物体关系图。
不判断 kind。
不复制或修改原始 benchmark 图像。
当前“可跟踪、可检索”表示 VP 输出具备这些后续用途，不表示当前 extractor 已经实现跟踪和检索。

