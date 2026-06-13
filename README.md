# PigeonVision — 信鸽虹膜品种识别系统

基于虹膜图像分析的信鸽品种识别系统，支持**比对**和**检索**两个核心功能。

## 快速开始

### 1. 克隆代码

```bash
git clone git@github.com:ShouZhengCoder/PigeonVision.git
cd PigeonVision
```

### 2. 下载数据与模型

所有大文件（原始图片、中间产物、模型权重）托管于 Hugging Face。

**方式 A：克隆到项目内（推荐）**

```bash
git clone https://huggingface.co/datasets/jshouEX/pigeon-breed-image-dataset
python scripts/setup_data.py
```

**方式 B：自定义路径**

```bash
git clone https://huggingface.co/datasets/jshouEX/pigeon-breed-image-dataset /path/to/pigeon-data
export PIGEONVISION_DATA=/path/to/pigeon-data
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 运行服务

Flask 默认读取生产检索库 `outputs/features/relation_supcon_256d/`。如果该目录缺失，先构建评估库和生产库：

```bash
python src/stage4_siamese/build_db_fusion.py --mode eval
python src/stage4_siamese/build_db_fusion.py --mode full
```

```bash
conda activate pigeonvision
python src/stage5_server/app.py --host 0.0.0.0 --port 8080
```

访问 http://localhost:8080 使用 Web 界面。Android 真机需要和服务端在同一局域网内，并在 App 顶部填写电脑的局域网地址，例如 `http://192.168.1.23:8080`。

### 5. 运行 Android 客户端

Android 工程位于 `src/stage6_android/`。首次运行前确认：

- `src/stage6_android/app/src/main/assets/yolo/model.ncnn.param` 和 `model.ncnn.bin` 存在
- 服务端 `/health` 可访问
- 真机与服务端电脑在同一局域网

```bash
cd src/stage6_android
./gradlew :app:assembleDebug
```

生成 APK：`src/stage6_android/app/build/outputs/apk/debug/app-debug.apk`。

App 支持两种任务：

- **品种比对**：选择或拍摄两张鸽子图，App 端用 NCNN YOLO 裁剪眼部后提交 `/compare`
- **品种检索**：选择或拍摄一张鸽子图，自定义 Top-K（1-100），结果按每页 10 条分页展示，并显示服务端返回的原始鸽眼图缩略图

## 效果

| 任务 | 指标 | 值 |
|------|------|:--:|
| Search | Hit@1 | **70.9%** |
| | Hit@10 | **87.8%** |
| | mAP | **60.7%** |
| Compare | AUC | **99.5%** |

> 当前服务器优化版采用单路 `relation_supcon_256d` 特征，生产检索库 `outputs/features/relation_supcon_256d/` 使用 `normalize_meta.csv` 中全部 `status=success` 且 PNG 存在的虹膜图，当前全量 25,690 张。旧三路融合 `fusion_1024d_full` 方案为 Hit@1 64.3%、Hit@10 85.7%、mAP 50.4%、Compare AUC 98.0%。详细实验记录见 [docs/experiments.md](docs/experiments.md) 和 [docs/performance_improvement_journey.md](docs/performance_improvement_journey.md)。

## 项目结构

```
PigeonVision/
├── src/
│   ├── stage1_data/         # 数据整理：图片索引、YOLO标注、样本对、多标签meta
│   ├── stage2_detection/    # YOLOv5 眼部检测训练与推理
│   ├── stage3_preprocess/   # U-Net 虹膜分割 + Daugman 椭圆展开 + iris mask过滤
│   ├── stage4_siamese/      # IrisEncoder 训练 + FAISS 特征库 + 多模型融合
│   ├── stage5_server/       # Flask 后端服务 (默认 Relation-SupCon 256d)
│   └── stage6_android/      # Android 客户端：NCNN YOLO 裁眼 + HTTP 调用
├── configs/                 # 训练配置文件
├── scripts/                 # 工具脚本
│   ├── setup_data.py        # 从 HF 仓库解压数据 + 合并新文件
│   └── sync_hf.py           # 上传本地数据到 HF 仓库
├── data/                    # 元数据 CSV（大文件在 HF）
│   ├── train_meta.csv / val_meta.csv               # 单标签元数据
│   ├── train_multi_meta.csv / val_multi_meta.csv   # 多标签元数据
│   └── pairs_train.csv / pairs_val.csv             # 样本对
├── outputs/                 # 中间产物（图片/特征在 HF）
│   ├── eye_crops/           # YOLO眼部裁剪 + crop_meta.csv
│   ├── iris_normalized/     # 64×512 三通道虹膜归一化图 + normalize_meta.csv
│   └── features/
│       ├── relation_supcon_256d_eval/  # 评估库：train gallery + val query 指标
│       ├── relation_supcon_256d/       # 生产库：Flask 默认读取的全量检索库
│       ├── fusion_1024d_full/          # 旧三路融合生产库
│       └── *.json              # 旧版评估指标
└── ROADMAP.md               # 技术路线总文档
```

## 特征库构建

`build_db_fusion.py` 支持两种模式：

```bash
# 评估库：只把 train 放入 gallery，val 仅作为 query，不混入 gallery
python src/stage4_siamese/build_db_fusion.py --mode eval

# 生产库：使用 outputs/iris_normalized/normalize_meta.csv 中全部 success PNG
python src/stage4_siamese/build_db_fusion.py --mode full
```

默认 encoder 为 `relation_supcon`。`--mode eval` 默认输出 `outputs/features/relation_supcon_256d_eval/`，写入 `eval_metrics.json`、`eval_comparison.json`、`threshold.json` 和评估 gallery 的 FAISS 文件。`--mode full` 默认输出 `outputs/features/relation_supcon_256d/`，写入 Flask 直接读取的 `feature_db.npy`、`feature_db_meta.csv`、`faiss_index.bin`、`threshold.json`。

检索指标口径：

- `hit_at_k`：Top-K 内是否至少有 1 个相关样本，按 query 求平均；历史文档里的 R@K 多数是这个口径。
- `avg_relevant_at_k`：Top-K 内平均相关样本个数。
- `precision_at_k`：`avg_relevant_at_k / k`。
- `recall_at_k`：Top-K 相关样本数 / gallery 中全部相关样本数，按 query 求平均。
- `ndcg_at_k`：用二值相关性计算的排序质量。

## 数据托管策略

| 内容 | 位置 | 说明 |
|------|------|------|
| 源码、配置、CSV 元数据 | **GitHub** | `src/`, `configs/`, `data/*.csv`, `outputs/**/*.csv` |
| 原始鸽眼图 | **Hugging Face** | `data/extracted/{1..12}/*.jpg` |
| YOLO 眼部裁剪 | **Hugging Face** | `outputs/eye_crops/*.jpg` |
| 虹膜归一化图 | **Hugging Face** | `outputs/iris_normalized/*.png` |
| 模型权重 | **Hugging Face** | `checkpoints/` 全部内容 |
| 特征向量 + FAISS | **Hugging Face** | `outputs/features/*.npy`, `*.bin` |

Git 只同步源码、配置、文档和小型元数据 CSV；原始图片、眼部裁剪图、归一化 PNG、模型权重和 FAISS 二进制均通过 Hugging Face 或服务器本地重跑生成。归一化逻辑变更后，不要把旧 `outputs/iris_normalized/normalize_meta.csv` 当作新结果提交。

## API 接口

### POST /compare

```bash
curl -X POST http://localhost:8080/compare \
  -F "image_a=@iris1.png" \
  -F "image_b=@iris2.png" \
  -F "eye_crop=1"
```

返回: `{"distance": 0.83, "same_family": false, "threshold": 1.3}`

### POST /search

```bash
curl -X POST http://localhost:8080/search \
  -F "image=@iris.png" \
  -F "top_k=20" \
  -F "eye_crop=1"
```

返回: `{"results": [{"rank": 1, "img_id": "571835", "pg_id": "2016-26-0571835", "blood_name": "桑杰士", "distance": 0.21, "image_url": "/image/571835"}, ...]}`

`top_k` 默认 20，最大 100。`eye_crop=1` 表示上传内容已经是 Android 端裁好的眼部图，服务端会跳过 YOLO 检测，直接进入 U-Net 分割、mask 过滤归一化和特征检索。

### GET /image/<img_id>

返回 `outputs/img_index.csv` 中对应的原始鸽眼图，用于 Android 检索结果缩略图展示。

## 技术栈

- **目标检测**: YOLOv5s (ultralytics)
- **虹膜分割**: U-Net (PyTorch, GroupNorm)
- **虹膜归一化**: Daugman 椭圆展开 + iris mask 过滤非虹膜采样点
- **特征编码**: ResNet34 + Relation-SupCon → 256d L2 归一化特征
- **向量检索**: FAISS IndexFlatL2
- **后端服务**: Flask
- **检索效果**: 当前 Hit@1 70.9%, Hit@10 87.8%, mAP 60.7%, Compare AUC 99.5%

## 许可证

待定
