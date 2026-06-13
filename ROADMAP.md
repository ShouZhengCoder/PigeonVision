# 信鸽品种识别系统 — 技术路线总文档

> **本文档是所有开发工作的唯一权威参考。Agent 在执行任何阶段任务前必须完整阅读本文档。**

---

## 服务器环境

- **项目根目录**：`/home/u2023312335/project/learn/PigeonVision`
- **Python 环境**：服务器预装，建议使用 venv 或 conda 管理依赖
- **Agent 工具**：OpenAI Codex CLI

以下所有路径均相对于项目根目录，绝对路径前缀为 `/home/u2023312335/project/learn/PigeonVision/`。

---

## 项目概述

基于虹膜图像分析的信鸽品种识别系统，提供两个核心功能：

- **功能一（比对）**：输入两张虹膜图像，输出欧氏距离数值及是否属于同一品种/家族的判断。
- **功能二（检索）**：输入一张虹膜图像，从数据库中检索最相似的 Top-K 条记录，返回对应鸽子的环号（PG_ID）和品系名（BLOOD）。

整体 Pipeline：

```
原始鸽眼图 (31,896 张)
  └─[Stage 2: YOLOv5]─→ 眼部 bbox 裁剪 (25,766 张)
       └─[Stage 3: U-Net分割 + Daugman 椭圆展开 + iris mask过滤]─→ 64×512 三通道虹膜纹理图 (25,690 张)
            └─[Stage 4: IrisEncoder]─→ 256/512-dim L2归一化特征向量
                 ├─[Stage 5: /compare]─→ 欧氏距离 + 阈值判断 (当前 AUC 99.5%)
                 └─[Stage 5: /search]─→ FAISS IndexFlatL2 Top-K (当前 Hit@1 70.9%, Hit@10 87.8%)

模型配置：
  Relation-SupCon 256d 单路模型 (当前最佳):
    └── Relation-SupCon Encoder: ResNet34, 256-dim (关系感知加权 SupCon Loss)
  
  FAISS:
    ├── relation_supcon_256d_eval: train gallery + val query, 用于评估
    └── relation_supcon_256d: 全量 success PNG, Flask 生产检索库（25,690 vectors）
  API: Flask, /search + /compare + /health endpoints
```

---

## 项目目录结构

```
PigeonVision/
├── AGENTS.md                         ← Codex CLI 自动读取的项目上下文
├── ROADMAP.md                        ← 本文档（技术路线完整参考）
│
├── data/
│   ├── extracted/                    ← 原始鸽眼图片（已解压）
│   │   ├── 1/                        ← 图片直接在此目录下，如 123456.jpg
│   │   ├── 2/
│   │   ├── ...
│   │   └── 12/
│   ├── unet_labelme_80/              ← 80 张手工标注样本（images / annotations / masks）
│   └── datasetXGN/
│       ├── anotations/               ← 9,979 个 JSON 标注文件
│       ├── blood.csv                 ← 28,910 条血统
│       ├── pigeon.csv                ← 113,844 条鸽子记录
│       ├── relations.csv             ← 250,207 条血统-图片关系
│       └── img_list.txt              ← 31,900 行图片 ID
│
├── src/
│   ├── stage1_data/                  ← 数据整理脚本 (含多标签 meta 构建、rebuild_pairs)
│   ├── stage2_detection/             ← YOLOv5 眼部检测
│   ├── stage3_preprocess/            ← U-Net 虹膜分割、Daugman 椭圆展开与 iris mask 过滤
│   ├── stage4_siamese/               ← IrisEncoder 训练 (Triplet/SupCon/ArcFace/Proxy-Anchor)
│   ├── stage5_server/                ← Flask 后端服务
│   └── stage6_android/               ← Android 客户端：NCNN YOLO 裁眼 + Flask HTTP 调用
│
├── configs/
│   ├── yolov5.yaml                   ← YOLOv5 训练配置
│   ├── unet.yaml                     ← U-Net 训练配置
│   └── siamese.yaml                  ← 孪生网络训练配置
│
├── checkpoints/
│   ├── detection/                    ← YOLOv5 权重
│   ├── segmentation/                 ← U-Net 权重
│   └── siamese/                      ← 孪生网络权重
│
├── outputs/
│   ├── img_index.csv                 ← 图片ID到文件路径的索引（Stage 1 生成）
│   ├── eye_crops/                    ← YOLO 裁剪的眼部图像
│   ├── iris_normalized/              ← U-Net + 椭圆展开 + mask 过滤后的虹膜图像（64×512）
│   └── features/
│       ├── relation_supcon_256d_eval/ ← 评估库：train gallery + val query 指标
│       │   ├── feature_db.npy
│       │   ├── feature_db_meta.csv
│       │   ├── faiss_index.bin
│       │   ├── eval_metrics.json
│       │   └── threshold.json
│       └── relation_supcon_256d/     ← 生产库：Flask 默认读取的全量检索库
│           ├── feature_db.npy
│           ├── feature_db_meta.csv
│           ├── faiss_index.bin
│           ├── threshold.json
│           └── build_manifest.json
│
└── logs/                             ← 训练日志
```

---

## 图片文件路径规则

原始图片分散在 `data/extracted/1/` 到 `data/extracted/12/` 中，图片文件名为 `{img_id}.jpg`，但**不知道某个 img_id 在哪个子目录**。

**Stage 1 必须优先完成的任务**：构建 `outputs/img_index.csv`，记录每个 img_id 对应的完整路径。

```python
# 查找图片的标准方法（在所有脚本中统一使用）
def load_img_index(index_path="outputs/img_index.csv"):
    """返回 dict: img_id(str) -> absolute_path(str)"""
    import pandas as pd
    df = pd.read_csv(index_path)
    return dict(zip(df['img_id'].astype(str), df['path']))
```

img_index.csv 格式：
```
img_id,path
123456,/home/u2023312335/project/learn/PigeonVision/data/extracted/3/123456.jpg
...
```

---

## 关键数据格式说明

### 标注 JSON（data/extracted/datasetXGN/anotations/*.json）

```json
{
  "img": "7918.jpg",
  "height": 600,
  "weidth": 700,
  "bbs": [
    {"label": "eye", "bbx": [x1, y1, x2, y2]},
    {"label": "mouse", "bbx": [...]}
  ]
}
```

**只保留 label == "eye"**，过滤 "mouse" 和 "900"。

### relations.csv（样本对构建的主数据源）

```
B05-6045278,606803
B01-6455003,606803
B02-6113358,606803
...
```

**无 header 行**，两列：第一列血统 ID（blood_id），第二列图片 ID（img_id）。长表格式，标准 CSV，直接 `pd.read_csv` 即可，**不需要 `field_size_limit`**。

数据规模（实测）：
- 250,207 行，81,751 个唯一 blood_id，46,788 个唯一图片
- 理论最大正样本对数：3,553,098 对
- blood.csv 的全部数据是 relations.csv 的子集（blood.csv 不含额外信息，脚本中不再使用）

读取方式：
```python
import pandas as pd
rel = pd.read_csv('data/extracted/datasetXGN/relations.csv',
                  header=None, names=['blood_id', 'img_id'])
rel['img_id'] = rel['img_id'].astype(str)
```

### blood.csv（已弃用，仅保留备查）

宽表格式，第一列血统 ID，后续列为图片 ID，行超长需要 `csv.field_size_limit(10**7)`。其全部数据已包含在 relations.csv 中，所有脚本改用 relations.csv，不再引用 blood.csv。

### pigeon.csv

| 字段 | 说明 |
|------|------|
| ID | 图片 ID（与标注 JSON、relations.csv 一致） |
| PG_ID | 环号（检索结果展示，如 `NL15-1273729`） |
| BLOOD | 品系名（如 `郝斯特.贺尔曼斯`、`根特布朗格"无环号"`） |
| EYE | 眼色（`黄眼`、`砂眼`） |
| COLOR | 羽色 |
| SEX | 性别 |
| IMG | 原始图片 URL |

实际共 11 列：`ID, PID, CID, SID, NAME, COLOR, EYE, PG_ID, SEX, BLOOD, IMG`。

### YOLO 标注格式

```
<class_id> <cx> <cy> <w> <h>    # 归一化到 [0,1]，class_id=0
```

转换公式：
```python
cx = (x1 + x2) / 2 / img_width
cy = (y1 + y2) / 2 / img_height
w  = (x2 - x1) / img_width
h  = (y2 - y1) / img_height
```

---

## 阶段一：数据整理

**目标**：构建图片索引、YOLOv5 检测数据集、孪生网络样本对列表。

### 输出文件

| 文件 | 内容 |
|------|------|
| `outputs/img_index.csv` | img_id → 文件绝对路径 |
| `data/yolo_dataset/labels/train/*.txt` | YOLO 格式标注 |
| `data/yolo_dataset/labels/val/*.txt` | YOLO 格式标注 |
| `data/yolo_dataset/train.txt` | 训练集图片路径列表 |
| `data/yolo_dataset/val.txt` | 验证集图片路径列表 |
| `data/yolo_dataset/data.yaml` | YOLOv5 数据集配置 |
| `data/pairs_train.csv` | 孪生网络训练对（img_id_a, img_id_b, label） |
| `data/pairs_val.csv` | 孪生网络验证对 |

### 脚本

**src/stage1_data/build_img_index.py**
- 遍历 `data/extracted/1/` 到 `data/extracted/12/`
- 收集所有 `.jpg`/`.jpeg` 文件，img_id = 文件名去后缀
- 写入 `outputs/img_index.csv`（列：img_id, path）

**src/stage1_data/convert_annotations.py**
- 读取 `data/extracted/datasetXGN/anotations/` 下所有 JSON，只保留 `eye` 标签
- **先查 img_index.csv，跳过图片不存在的标注文件，并统计缺失数量**
- 转为 YOLO 格式 .txt，按 8:2 分割（seed=42）
- 写入 `data/yolo_dataset/`，生成 `data.yaml`（train/val 均为绝对路径列表文件）

**src/stage1_data/build_pairs.py**
- 读取 `data/extracted/datasetXGN/relations.csv`（无 header，names=['blood_id','img_id']，img_id 强制转 str）
- 与标注集（anotations/ 目录中有 JSON 的 img_id）取交集，构建初始样本对
- 正样本对：同 blood_id 且两张图都在标注集内，label=1；用 `itertools.combinations` 枚举
- **必须去重**：同一对图片可能共属多个 blood_id，枚举后用 `set(frozenset(p) for p in pairs)` 去重
- 负样本对：随机采样，正:负 = 1:2（seed=42）
- 写入 `data/pairs_train.csv` 和 `data/pairs_val.csv`
- **注意**：这是 Stage 1 的初始版本，仅用于 Stage 4 扩充前的快速验证。Stage 3 完成后需用 `rebuild_pairs.py` 重建。

### 验收标准
- `outputs/img_index.csv` 行数接近 31,900
- `data/yolo_dataset/data.yaml` 存在，nc=1，所有路径均实际存在
- `data/pairs_train.csv` 正样本对 ≥ 150,000（基于标注集与 relations.csv 交集，去重后）

---

## 阶段二：目标检测（YOLOv5）

**目标**：训练眼部检测模型，对全量图片推理产出眼部裁剪图。

### 模型选型

使用 YOLOv5s（`ultralytics` 库）：单类（eye），预训练迁移，命令行驱动。

```bash
pip install ultralytics
yolo train data=data/yolo_dataset/data.yaml model=yolov5s.pt epochs=100 batch=16 imgsz=416 project=checkpoints/detection name=exp
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `src/stage2_detection/train.py` | 训练封装脚本 |
| `src/stage2_detection/infer_all.py` | 全量推理 + 裁剪 |
| `checkpoints/detection/exp/weights/best.pt` | 最优权重 |
| `outputs/eye_crops/<img_id>.jpg` | 裁剪的眼部图（置信度≥0.7） |
| `outputs/eye_crops/crop_meta.csv` | img_id, x1, y1, x2, y2, confidence |

### infer_all.py 关键逻辑

- 从 `outputs/img_index.csv` 读取全量图片路径
- 每张图取置信度最高的 eye 框
- bbox 四边扩展 10%（不超图片边界）后裁剪保存
- 支持 `--resume`，跳过 crop_meta.csv 中已有记录的图片

### 验收标准
- 验证集 mAP@0.5 ≥ 0.85
- `outputs/eye_crops/crop_meta.csv` 文件数 ≥ 25,000 条（含 confidence=0 的未检测到记录）
- `outputs/eye_crops/` 中实际图片数 ≥ 20,000

---

## 阶段三：虹膜图像预处理（U-Net + 椭圆展开 + iris mask 过滤）

**目标**：将眼部裁剪图转化为 64×512 三通道虹膜纹理图。

### 算法流程

```
eye_crop.jpg
  → resize 到 256×256
  → U-Net 语义分割（0=background, 1=iris, 2=pupil）
  → 取 pupil / iris 最大连通域
  → cv2.fitEllipse 拟合内外椭圆
  → 椭圆版 Daugman 展开：角向 512 步，径向 64 步，双线性插值
  → 用同一组 remap 坐标展开 iris mask，mask 外填 127 中性灰
  → 输出 64×512 三通道 PNG
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `configs/unet.yaml` | U-Net 训练配置 |
| `src/stage3_preprocess/train_unet.py` | U-Net 训练脚本 |
| `src/stage3_preprocess/iris_localize.py` | 分割+归一化核心函数 |
| `src/stage3_preprocess/batch_normalize.py` | 批量处理 |
| `src/stage3_preprocess/unet_common.py` | U-Net 公共模块与数据处理 |
| `src/stage3_preprocess/visualize_samples.py` | 训练/推理结果可视化 |
| `checkpoints/segmentation/best.pt` | 最优分割权重 |
| `outputs/iris_normalized/<img_id>.png` | 64×512 三通道虹膜图 |
| `outputs/iris_normalized/normalize_meta.csv` | img_id, status, cx, cy, r_inner, r_outer, ellipse, mask_confidence |

### 训练约定

- 输入尺寸固定为 `256×256`
- 输入通道固定为 1 通道灰度，训练和推理必须一致
- U-Net 采用纯 PyTorch 实现，编码/解码块使用 `Conv + GroupNorm + ReLU`
- `base_channels=32` 时使用 `GroupNorm(num_groups=8)`
- 三类分割：`0=background`、`1=iris`、`2=pupil`
- 80 张 Labelme 标注图固定拆分为 `64 train / 16 val`，seed=42
- 仅对 train 做同步增强：水平翻转、垂直翻转、旋转 ±30°、亮度/对比度 jitter
- 损失函数采用 `CrossEntropy + Dice`

### 椭圆拟合、展开与 mask 过滤

- 对 `pupil` 和 `iris` 的最大连通域分别执行 `cv2.fitEllipse`
- 若椭圆拟合失败或连通域过小，则该样本记为 `failed`
- `mask_confidence` 定义为 iris/pupil 区域对应 softmax 概率的均值
- 默认阈值先取 `0.7`，低于该值直接记为 `failed`

椭圆展开时，对每个角度 `θ` 先计算边界半径：

```python
r(θ) = ab / sqrt((b·cosθ)^2 + (a·sinθ)^2)
```

其中 `pupil` 和 `iris` 椭圆各算一次，得到 `r_pupil(θ)` 与 `r_iris(θ)`，再在 `[r_pupil(θ), r_iris(θ)]` 之间做 64 步径向插值，最终 remap 成 64×512 三通道图。图像和 `iris mask` 使用同一组 remap 坐标展开，mask 外像素填充为 127，避免眼睑、背景或椭圆外缘内容进入有效虹膜纹理。

### 验收标准
- 训练集和验证集的输入尺寸、通道数、归一化方式完全一致
- val 集能输出稳定的 iris / pupil 分割结果，并能完成椭圆拟合
- 抽样 10 张确认 `image -> mask -> ellipse -> normalized png` 流程正确
- 生成 `outputs/iris_normalized/samples_vis.png` 便于人工检查
- `normalize_meta.csv` 中要明确区分 `success` / `failed`，并记录 `mask_confidence`

---

## 阶段三点五：重建样本对（Stage 3 完成后必须执行）

**目标**：用全量 iris_normalized 图片替换 Stage 1 的初始样本对，大幅扩充孪生网络训练数据。

**src/stage1_data/rebuild_pairs.py**
- 读取 `outputs/iris_normalized/normalize_meta.csv`，取 status=success 的全部 img_id（全量可用集）
- 读取 `data/extracted/datasetXGN/relations.csv`（无 header，names=['blood_id','img_id']），过滤到全量可用集
- 用全量成功归一化图片重建正/负样本对（逻辑同 build_pairs.py）
- **覆盖写入** `data/pairs_train.csv` 和 `data/pairs_val.csv`

预期效果：正样本对从约 200,000 扩充到 1,000,000+，显著提升孪生网络训练质量。

### 验收标准
- `data/pairs_train.csv` 正样本对数量明显多于 Stage 1 版本
- 构建时只使用 `outputs/iris_normalized/normalize_meta.csv` 中 `status=success` 的样本

---

## 阶段四：IrisEncoder 训练（度量学习）

**目标**：训练 IrisEncoder，将 64×512 虹膜图映射为 256-dim L2 归一化特征向量。

### 网络结构

```python
class IrisEncoder(nn.Module):
    # ResNet34/50 backbone（pretrained ImageNet，去分类头）
    # AdaptiveAvgPool2d(1) → flatten → Linear(in_features, feat_dim) → BatchNorm1d → L2 normalize
    # 输入：64×512 RGB（三通道归一化虹膜 PNG）
    # 输出：256-dim 单位向量（默认）/ 512-dim
```

支持的 backbone：`resnet18`、`resnet34`、`resnet50`，默认 `resnet34`。

### 损失函数

**主方案：Batch Hard Triplet Loss + PK Sampler**

```python
# PK 采样：每 batch 选 P=16 个 blood_name，每个 blood_name 取 K=4 张图 → batch=64
# 对每个 anchor，选 hardest positive + hardest negative（基于 blood_id 重合）
L = max(d(anchor, hardest_positive) - d(anchor, hardest_negative) + margin, 0)
# margin = 0.3
```

**多正样本变体**：若 blood_id 重合则将对应样本视为正样本（支持多标签）。

### 训练配置（configs/siamese.yaml）

```yaml
feat_dim: 256
backbone: resnet34
in_channels: 3
triplet_margin: 0.3
warmup_epochs: 3
batch_size: 64
lr: 0.0003
epochs: 80
scheduler: cosine
input_shape: [64, 512]
classes_per_batch: 16      # P
samples_per_class: 4       # K
min_images_per_blood: 5
min_images_per_blood_id: 2
min_db_images_per_blood: 20
```

### 实验方案（已完成的 8 个实验）

| 实验 | 方法 | 效果 | 状态 |
|------|------|------|:--:|
| 1 | 评估修正 (blood_id 多标签) | 旧评估 R@1 42%→实际 17.5%，Compare AUC 62%→65% | ✅ |
| 2 | 代码健壮性优化 | Python 3.9、cv2 检查、类型安全 | ✅ |
| 3 | SupCon 多标签训练 | Compare AUC +7.5%，但检索略降 | ✅ |
| 4 | PK-SupCon + Warm-Start | 模型退化 | ❌ |
| 5 | Multi-Positive Triplet | 嵌入弥散，不如原始 | ❌ |
| 6 | Proxy-Anchor Loss | 未收敛 | ❌ |
| 7 | ArcFace Loss | 7a (BCE) 不收敛，7b (单标签) 未运行 | ⚠️ |
| **8** | **MoE Concat 512d** | R@1 25.9%, R@10 51.8%, AUC 71.0% | ✅ |
| **9** | **Relation-SupCon + 全量检索库** | **当前最优**：Hit@1 70.9%, Hit@10 87.8%, mAP 60.7%, AUC 99.5% | ✅ |

**最佳模型**：Relation-SupCon 256d 单路编码器，Flask 生产检索默认使用 `outputs/features/relation_supcon_256d/`。旧三路融合 `fusion_1024d_full` 方案保留为对照基线。

详细实验记录见 `docs/experiments.md`。

### 输出文件

| 文件 | 内容 |
|------|------|
| `src/stage4_siamese/model.py` | IrisEncoder 定义 (ResNet34/50) |
| `src/stage4_siamese/dataset.py` | TripletMetaDataset + PK sampler |
| `src/stage4_siamese/loss.py` | batch_hard_triplet_loss / batch_hard_triplet_loss_multi |
| `src/stage4_siamese/loss_supcon.py` | SupCon Loss |
| `src/stage4_siamese/loss_proxy_anchor.py` | Proxy-Anchor Loss |
| `src/stage4_siamese/train.py` | 训练主脚本 (Triplet) |
| `src/stage4_siamese/train_multilabel.py` | SupCon 训练脚本 |
| `src/stage4_siamese/train_relation_supcon.py` | 关系感知加权 SupCon 训练脚本 |
| `src/stage4_siamese/train_arcface.py` | ArcFace 训练脚本 |
| `src/stage4_siamese/train_proxy_anchor.py` | Proxy-Anchor 训练脚本 |
| `src/stage4_siamese/build_db.py` | 构建 FAISS 特征数据库 + 双标准评估 |
| `src/stage4_siamese/build_db_fusion.py` | Relation-SupCon 256d / 多 encoder 特征库构建，支持 `--mode eval/full` |
| `src/stage4_siamese/relation_metrics.py` | 多标签 blood_id 评估指标 |
| `checkpoints/siamese/best.pt` | 最优 Triplet 编码器权重 |
| `checkpoints/siamese/supcon/best.pt` | 最优 SupCon 编码器权重 |
| `checkpoints/siamese/relation_supcon/best.pt` | 当前生产 Relation-SupCon 编码器权重 |
| `outputs/features/relation_supcon_256d_eval/feature_db.npy` | 评估 gallery 特征矩阵（train only） |
| `outputs/features/relation_supcon_256d_eval/eval_metrics.json` | val query 对 train gallery 的评估指标 |
| `outputs/features/relation_supcon_256d_eval/threshold.json` | 评估集上得到的 Compare 阈值 |
| `outputs/features/relation_supcon_256d/feature_db.npy` | 生产全量特征矩阵（全部 success PNG） |
| `outputs/features/relation_supcon_256d/feature_db_meta.csv` | img_id, pg_id, blood, blood_id, blood_name |
| `outputs/features/relation_supcon_256d/faiss_index.bin` | Flask 默认读取的 FAISS IndexFlatL2 |
| `outputs/features/relation_supcon_256d/threshold.json` | Flask 默认读取的阈值 JSON |

### build_db_fusion.py 逻辑

- `--mode eval`：读取 `data/train_meta.csv` 作为 gallery，读取 `data/val_meta.csv` 作为 query；两者都必须落在 `normalize_meta.csv status=success` 且 PNG 存在集合内；默认输出到 `outputs/features/relation_supcon_256d_eval/`；**不得把 val/query 混入评估 gallery**。
- `--mode full`：读取 `outputs/iris_normalized/normalize_meta.csv` 中全部 `status=success` 的 img_id，并确认 `outputs/iris_normalized/<img_id>.png` 存在；关联 `pigeon.csv` 的 `PG_ID/BLOOD` 和 `relations.csv` 的 blood_id；默认输出到 `outputs/features/relation_supcon_256d/`，供 Flask 默认读取。
- 两种模式都保存 `feature_db.npy`、`feature_db_meta.csv` 和 `faiss_index.bin`；eval 额外保存 `eval_metrics.json` / `eval_comparison.json` / `threshold.json`；full 复制 eval 阈值并保存 `build_manifest.json`。
- 当前默认 encoder 为 `relation_supcon`，输出 256d L2 归一化特征；旧 1024d 融合可通过 `--encoders triplet,relation_supcon,arcface` 复现。

命令：

```bash
python src/stage4_siamese/build_db_fusion.py --mode eval
python src/stage4_siamese/build_db_fusion.py --mode full
```

### 评估标准

**多标签 blood_id 评估（新标准，更准确）**：
- 两张图共享任意 blood_id → 血脉相关
- 反映真实血脉关系（每图平均 6.4 个 blood_id，91.8% 图像有多重血脉）

**Search 指标**：
- **Hit@K (`hit_at_k`)**：Top-K 中是否至少有一个血脉相关样本；历史表格中的 R@K 多数是这个口径
- **avg_relevant@K (`avg_relevant_at_k`)**：Top-K 中平均相关样本个数
- **Precision@K (`precision_at_k`)**：Top-K 中相关样本比例，等于 `avg_relevant_at_k / K`
- **Recall@K (`recall_at_k`)**：Top-K 相关样本数 / gallery 中全部相关样本数，按 query 平均
- **nDCG@K (`ndcg_at_k`)**：使用二值相关性计算的排序质量
- **mAP**：平均精度均值

**Compare 指标**：AUC、Balanced Accuracy、EER

**PG_ID 聚合评估**：
- eval 模式额外把 gallery 中同一 `PG_ID` 的多张图聚合成一个 centroid，再用 image-level query 评估 PG_ID 级别 Top-K 指标，输出到 `pg_id_centroid_search`。

### 验收标准
- Search Hit@1 ≥ 20%（基于多标签 blood_id 评估）
- Compare AUC ≥ 65%
- `outputs/features/relation_supcon_256d_eval/faiss_index.bin` 存在，且评估 gallery 只含 train
- `outputs/features/relation_supcon_256d/faiss_index.bin` 存在，feature_db_meta.csv 行数接近 `normalize_meta.csv` 中 success PNG 数（当前 25,690）

---

## 阶段五：后端服务

**目标**：Flask 服务，暴露 HTTP 接口。

### 接口定义

**POST /compare**（multipart: image_a, image_b, eye_crop=0/1）
```json
{"distance": 0.83, "same_family": true, "threshold": 0.72}
```

**POST /search**（multipart: image, top_k=20, eye_crop=0/1）
```json
{"results": [{"rank":1, "img_id":"571835", "pg_id":"2016-26-0571835", "blood_name":"桑杰士", "distance":0.21, "image_url":"/image/571835"}]}
```

**GET /image/<img_id>** → 返回 `outputs/img_index.csv` 中的原始鸽眼图，用于客户端展示检索缩略图

**GET /health** → `{"status": "ok", "gallery_size": 25690, "breed_count": 1234}`（数量随成功归一化 PNG 略有变化）

**GET /** → Web 演示页面

### Pipeline 逻辑

```python
def process_image(img_bytes):
    img = PIL.Image.open(io.BytesIO(img_bytes))
    w, h = img.size

    # 情况一：已是归一化虹膜图（宽高比约 8:1，如 512×64）
    if w / h > 4:
        iris_img = img.resize((512, 64))

    # 情况二：眼部特写（接近方形）
    # 直接走 U-Net 分割 + Daugman 椭圆展开 + iris mask 过滤
    elif 0.5 < w / h < 2.0:
        iris_arr = iris_segment_and_normalize(np.array(img))
        if iris_arr is None:
            raise ValueError("虹膜分割失败")

    # 情况三：原始全图（含鸽身背景）
    # 先走 YOLO 检测裁剪眼部，再走情况二
    else:
        bbox = yolo_detect_eye(img)
        if bbox is None:
            raise ValueError("未检测到眼部区域")
        eye_crop = img.crop(bbox)
        iris_arr = iris_segment_and_normalize(np.array(eye_crop))

    # IrisEncoder → FAISS search / distance compare
    return encoder(transform(iris_img))
```

### 模型加载

启动时加载三个模型：
- YOLOv5（眼部检测，`checkpoints/detection/exp/weights/best.pt`）
- U-Net（虹膜分割，`checkpoints/segmentation/best.pt`）
- IrisEncoder（特征提取，默认 `checkpoints/siamese/relation_supcon/best.pt`）
- FAISS Index（检索，默认读取 `outputs/features/relation_supcon_256d/faiss_index.bin`）
- Metadata（默认读取 `outputs/features/relation_supcon_256d/feature_db_meta.csv`）
- Threshold（优先读取 `outputs/features/relation_supcon_256d/threshold.json`）

`relation_supcon_256d` 是生产检索库，应由 `build_db_fusion.py --mode full` 构建，包含 `normalize_meta.csv` 中全部 `status=success` 且 PNG 存在的样本。`relation_supcon_256d_eval` 只用于离线评估，不应作为 Flask 生产 gallery。

### 验收标准
- 两个接口均能返回正确 JSON
- Web 页面能上传图片并展示结果
- 服务绑定 127.0.0.1:8080
- `/health` 的 `gallery_size` 接近当前全量成功归一化数量（约 25,690），而不是 train-only 评估 gallery

---

## 阶段六：Android 部署

**目标**：实现 Android 客户端，负责拍照/选图、NCNN YOLOv5s 眼部检测裁剪，并调用 Stage 5 Flask 服务完成品种比对和检索。

### 架构

```text
Android:
  原始图片 -> NCNN YOLOv5s 检测眼部 -> JPEG 眼部裁剪图

Flask:
  eye_crop=1 -> U-Net 虹膜分割 -> Daugman 椭圆展开 -> iris mask 过滤 -> IrisEncoder -> FAISS / distance
```

Android 端不做 U-Net 分割、IrisEncoder 特征提取或 FAISS 检索。

### 客户端功能

- 服务端地址输入、持久化和 `/health` 连接测试
- 品种比对：两张图分别裁眼，POST `/compare`，附带 `eye_crop=1`
- 品种检索：一张图裁眼，POST `/search`，附带 `eye_crop=1` 和自定义 `top_k`
- 检索 Top-K 范围 `1-100`，默认 `20`
- 检索结果每页 10 条分页，展示排名、品系名、PG_ID、blood_id、img_id、距离和原始鸽眼缩略图

### 输出文件

| 文件 | 内容 |
|------|------|
| `src/stage6_android/app/src/main/assets/yolo/model.ncnn.param` | NCNN YOLOv5s 参数 |
| `src/stage6_android/app/src/main/assets/yolo/model.ncnn.bin` | NCNN YOLOv5s 权重 |
| `src/stage6_android/app/src/main/cpp/yolo_jni.cpp` | NCNN JNI 推理与 bbox 解析 |
| `src/stage6_android/app/src/main/java/com/pigeonvision/app/YoloDetector.kt` | Kotlin 检测封装 |
| `src/stage6_android/app/src/main/java/com/pigeonvision/app/ApiClient.kt` | OkHttp API 客户端 |
| `src/stage6_android/app/src/main/java/com/pigeonvision/app/ui/` | Jetpack Compose UI |
| `src/stage6_android/README.md` | Android 使用说明 |

### 验收标准

- `./gradlew :app:assembleDebug` 成功生成 debug APK
- `./gradlew testDebugUnitTest` 通过
- 真机可选择/拍摄图片，未检测到眼部时提示用户靠近拍摄
- `/compare` 返回距离和 same_family
- `/search` 按自定义 Top-K 返回结果，分页展示缩略图

---

## 依赖汇总

```
# 训练环境（pip install -r requirements.txt）
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.5.0
numpy
pandas
faiss-cpu
ultralytics
Pillow
tqdm
PyYAML
tensorboard
onnx
onnxruntime

# 服务环境
flask>=2.0
gunicorn
```

---

## 常见陷阱

1. **relations.csv 无 header**：本地文件第一行是数据，读取时必须 `header=None, names=['blood_id','img_id']`，否则第一条记录丢失。
2. **blood.csv 已弃用**：不要在新脚本中引用 blood.csv，其数据已全部包含在 relations.csv 中。
3. **标注噪声**：只保留 `label == "eye"`，过滤 "mouse" 和 "900"。
4. **图片查找**：所有脚本通过 `outputs/img_index.csv` 查找图片路径，禁止硬编码子目录。
5. **IrisEncoder 输入**：Stage 4 输入为 64×512 RGB（三通道归一化 PNG）；不同 backbone 的输入尺寸需一致。
6. **FAISS 向量**：存入前确保已 L2 归一化（此时 L2 距离 ≡ 余弦距离）。
7. **多标签评估**：使用 blood_id 重合评估（非 blood_name 匹配），每张图平均 6.4 个 blood_id。
8. **阶段依赖**：Stage 3 依赖 Stage 2 输出和 `data/unet_labelme_80/` 标注集；Stage 3.5 依赖 Stage 3 输出；Stage 4 依赖 Stage 3.5 输出。Stage 1 最优先执行。
