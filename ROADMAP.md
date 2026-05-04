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
原始鸽眼图
  └─[Stage 2: YOLOv5]─→ 眼部 bbox 裁剪
       └─[Stage 3: Hough归一化]─→ 64×512 虹膜纹理图
            └─[Stage 4: 孪生网络]─→ 128-dim L2归一化特征向量
                 ├─[Stage 5: 比对接口]─→ 欧氏距离 + 阈值判断
                 └─[Stage 5: 检索接口]─→ FAISS Top-K + PG_ID/BLOOD
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
│   └── datasetXGN/
│       ├── anotations/               ← 9,979 个 JSON 标注文件
│       ├── blood.csv                 ← 28,910 条血统
│       ├── pigeon.csv                ← 113,844 条鸽子记录
│       ├── relations.csv             ← 250,207 条血统-图片关系
│       └── img_list.txt              ← 31,900 行图片 ID
│
├── src/
│   ├── stage1_data/                  ← 数据整理脚本
│   ├── stage2_detection/             ← YOLOv5 目标检测
│   ├── stage3_preprocess/            ← 虹膜定位与归一化
│   ├── stage4_siamese/               ← 孪生网络训练
│   ├── stage5_server/                ← Flask 后端服务
│   └── stage6_android/               ← Android 部署
│
├── configs/
│   ├── yolov5.yaml                   ← YOLOv5 训练配置
│   └── siamese.yaml                  ← 孪生网络训练配置
│
├── checkpoints/
│   ├── detection/                    ← YOLOv5 权重
│   └── siamese/                      ← 孪生网络权重
│
├── outputs/
│   ├── img_index.csv                 ← 图片ID到文件路径的索引（Stage 1 生成）
│   ├── eye_crops/                    ← YOLO 裁剪的眼部图像
│   ├── iris_normalized/              ← Hough 归一化后的虹膜图像（64×512）
│   └── features/
│       ├── feature_db.npy            ← 特征向量矩阵（N×128）
│       ├── feature_db_meta.csv       ← img_id, pg_id, blood
│       └── faiss_index.bin           ← FAISS 索引
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

### 标注 JSON（data/datasetXGN/anotations/*.json）

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
rel = pd.read_csv('data/datasetXGN/relations.csv',
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
- 读取 `data/datasetXGN/anotations/` 下所有 JSON，只保留 `eye` 标签
- **先查 img_index.csv，跳过图片不存在的标注文件，并统计缺失数量**
- 转为 YOLO 格式 .txt，按 8:2 分割（seed=42）
- 写入 `data/yolo_dataset/`，生成 `data.yaml`（train/val 均为绝对路径列表文件）

**src/stage1_data/build_pairs.py**
- 读取 `data/datasetXGN/relations.csv`（无 header，names=['blood_id','img_id']，img_id 强制转 str）
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

## 阶段三：虹膜图像预处理

**目标**：将眼部裁剪图转化为 64×512 虹膜纹理图。

### 算法流程

```
eye_crop.jpg
  → 灰度化 → 高斯模糊(5×5)
  → 水平/垂直一维投影 → 瞳孔粗略中心 (cx, cy)
  → 以 (cx,cy) 为中心做 Canny(50, 150)
  → cv2.HoughCircles 找内圆（瞳孔边界）和外圆（虹膜/巩膜边界）
  → Daugman 极坐标展开：径向 64 步，角向 512 步，双线性插值
  → 输出 64×512 灰度 PNG
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `src/stage3_preprocess/iris_localize.py` | 定位+归一化核心函数 |
| `src/stage3_preprocess/batch_normalize.py` | 批量处理 |
| `outputs/iris_normalized/<img_id>.png` | 64×512 虹膜图 |
| `outputs/iris_normalized/normalize_meta.csv` | img_id, status, cx, cy, r_inner, r_outer |

### 关键参数（可调）

```python
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_DP = 1
HOUGH_MIN_DIST = 20
HOUGH_PARAM1 = 50   # Canny高阈值
HOUGH_PARAM2 = 30   # 累加器阈值（越小越多圆，越大越严格）
R_INNER_RANGE = (0.10, 0.25)   # 内圆半径/图像宽度
R_OUTER_RANGE = (0.30, 0.50)   # 外圆半径/图像宽度
```

### 验收标准
- 成功率 ≥ 70%（相对于 eye_crops 数量）
- 抽样 20 张可视化确认纹理展开正确（生成 `outputs/iris_normalized/samples_vis.png`）

---

## 阶段三点五：重建样本对（Stage 3 完成后必须执行）

**目标**：用全量 iris_normalized 图片替换 Stage 1 的初始样本对，大幅扩充孪生网络训练数据。

**src/stage1_data/rebuild_pairs.py**
- 读取 `outputs/iris_normalized/normalize_meta.csv`，取 status=success 的全部 img_id（全量可用集）
- 读取 `data/datasetXGN/relations.csv`（无 header，names=['blood_id','img_id']），过滤到全量可用集
- 用全量成功归一化图片重建正/负样本对（逻辑同 build_pairs.py）
- **覆盖写入** `data/pairs_train.csv` 和 `data/pairs_val.csv`

预期效果：正样本对从约 200,000 扩充到 1,000,000+，显著提升孪生网络训练质量。

### 验收标准
- `data/pairs_train.csv` 正样本对数量明显多于 Stage 1 版本

---

## 阶段四：孪生网络训练

**目标**：训练 IrisEncoder，将虹膜图映射为 128-dim L2 归一化特征向量。

### 网络结构

```python
class IrisEncoder(nn.Module):
    # MobileNetV2 backbone（pretrained，去分类头）
    # AdaptiveAvgPool2d(1) → flatten → FC(1280, 128) → L2 normalize
    # 输入：128×128 RGB（3通道，由 64×512 灰度图复制通道后 resize）
    # 输出：128-dim 单位向量
```

### 损失函数

Contrastive Loss：
```python
d = ||f_a - f_b||_2
L = y * d^2 + (1-y) * max(margin - d, 0)^2
# y=1: 同血统（正对）；y=0: 不同血统（负对）；margin=1.0
```

### 训练配置（configs/siamese.yaml）

```yaml
feat_dim: 128
margin: 1.0
batch_size: 64
lr: 0.001
epochs: 100
optimizer: adam
scheduler: cosine
input_size: 128
normalize_mean: [0.5, 0.5, 0.5]
normalize_std: [0.5, 0.5, 0.5]
checkpoint_dir: checkpoints/siamese/
iris_dir: outputs/iris_normalized/
pairs_train: data/pairs_train.csv
pairs_val: data/pairs_val.csv
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `src/stage4_siamese/model.py` | IrisEncoder 定义 |
| `src/stage4_siamese/dataset.py` | PairDataset |
| `src/stage4_siamese/loss.py` | contrastive_loss |
| `src/stage4_siamese/train.py` | 训练主脚本 |
| `src/stage4_siamese/build_db.py` | 构建 FAISS 特征数据库 |
| `checkpoints/siamese/best.pt` | 最优编码器权重 |
| `outputs/features/feature_db.npy` | N×128 特征矩阵 |
| `outputs/features/feature_db_meta.csv` | img_id, pg_id, blood |
| `outputs/features/faiss_index.bin` | FAISS IndexFlatL2 |

### build_db.py 逻辑

- 读取 `outputs/iris_normalized/normalize_meta.csv`，得到 status=success 的全部 img_id 集合（成功集，str 类型）
- 读取 `pigeon.csv`，将 `ID` 列转为 str，筛选 BLOOD 字段样本数 ≥ 50 的品系
- **取两者交集**（即：该图片既有成功的归一化结果，又有 BLOOD 标签）
- 对交集图片提取特征，保存 feature_db.npy、feature_db_meta.csv
- 构建 FAISS IndexFlatL2(128)，序列化保存

### 验收标准
- 验证集 Recall@1 ≥ 0.5
- `outputs/features/faiss_index.bin` 存在，feature_db_meta.csv 行数 ≥ 15,000

---

## 阶段五：后端服务

**目标**：Flask 服务，暴露两个 HTTP 接口。

### 接口定义

**POST /compare**（multipart: image_a, image_b）
```json
{"distance": 0.83, "result": "可能是同一家族", "threshold": 1.0}
```

**POST /search**（multipart: image, top_k=4）
```json
{"results": [{"rank":1, "img_id":"571835", "pg_id":"2016-26-0571835", "blood":"桑杰士", "distance":0.21}]}
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `src/stage5_server/app.py` | Flask 主应用 |
| `src/stage5_server/pipeline.py` | 推理 Pipeline |
| `src/stage5_server/threshold.py` | 阈值标定（ROC 曲线） |
| `src/stage5_server/templates/index.html` | Web 演示页面 |
| `src/stage5_server/requirements.txt` | 依赖列表 |

### Pipeline 逻辑

用户上传的图片分两种情况，pipeline 需要分别处理：

```python
def process_image(img_bytes):
    img = PIL.Image.open(io.BytesIO(img_bytes))
    w, h = img.size

    # 情况一：已是归一化虹膜图（宽高比约 8:1，如 512×64）
    if w / h > 4:
        iris_img = img.convert("RGB").resize((128, 128))

    # 情况二：眼部特写（接近方形或圆形，如截图里那种虹膜特写）
    # 直接走 Hough 定位+归一化
    elif 0.5 < w / h < 2.0:
        iris_arr = iris_localize_and_normalize(np.array(img.convert("L")))
        if iris_arr is None:
            raise ValueError("虹膜定位失败，请上传清晰的眼部图像")
        iris_img = Image.fromarray(iris_arr).convert("RGB").resize((128, 128))

    # 情况三：原始全图（含鸽身背景），先走 YOLO 检测裁剪眼部，再走情况二
    else:
        bbox = yolo_detect_eye(img)
        if bbox is None:
            raise ValueError("未检测到眼部区域，请上传包含眼睛的鸽子图像")
        eye_crop = img.crop(bbox)
        iris_arr = iris_localize_and_normalize(np.array(eye_crop.convert("L")))
        if iris_arr is None:
            raise ValueError("虹膜定位失败")
        iris_img = Image.fromarray(iris_arr).convert("RGB").resize((128, 128))

    return encoder(ToTensor(Normalize(iris_img)))
```

Demo 演示时用情况二（眼部特写）即可，与截图效果一致。

### 验收标准
- 两个接口均能返回正确 JSON
- Web 页面能上传图片并展示结果

---

## 阶段六：Android 部署

**目标**：将 IrisEncoder 部署到 Android，实现离线推理。

### 模型转换流程

```bash
# 1. PyTorch → ONNX
python src/stage6_android/export_onnx.py
# 输出: checkpoints/siamese/encoder.onnx，输入(1,3,128,128)，输出(1,128)

# 2. ONNX → NCNN
onnx2ncnn encoder.onnx encoder.param encoder.bin

# 3. 量化 float16
ncnnoptimize encoder.param encoder.bin encoder_fp16.param encoder_fp16.bin 1
```

### JNI 接口

```java
public class IrisEncoder {
    public native boolean init(String paramPath, String binPath);
    public native float[] encode(Bitmap bitmap);  // 返回 128-dim float[]
}
```

### 输出文件

| 文件 | 内容 |
|------|------|
| `src/stage6_android/export_onnx.py` | ONNX 导出脚本 |
| `src/stage6_android/jni/iris_encoder.cpp` | NCNN JNI C++ |
| `src/stage6_android/android_app/` | Android Studio 工程 |
| `src/stage6_android/DEPLOY.md` | 部署步骤说明 |

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
5. **MobileNetV2 输入**：预训练模型需要 3 通道 RGB，归一化后的灰度虹膜图需 `img.convert('RGB')` 再 resize 到 128×128。
6. **FAISS 向量**：存入前确保已 L2 归一化（此时 L2 距离 ≡ 余弦距离）。
7. **阶段依赖**：Stage 3 依赖 Stage 2 输出；Stage 4 依赖 Stage 3 输出；Stage 5 依赖 Stage 4 输出。Stage 1 最优先执行。
