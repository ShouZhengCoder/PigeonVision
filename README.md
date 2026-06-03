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

```bash
python src/stage5_server/app.py --host 0.0.0.0 --port 5000
```

访问 http://localhost:5000 使用 Web 界面。Android 真机需要和服务端在同一局域网内，并在 App 顶部填写电脑的局域网地址，例如 `http://192.168.1.23:5000`。

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
| Search | R@1 | **35.5%** |
| | R@10 | **59.1%** |
| | mAP | 16.2% |
| Compare | AUC | **73.1%** |
| | BalAcc | 66.9% |

> 图库 22K 张，多标签 blood_id 重合评估。详细实验记录见 [docs/experiments.md](docs/experiments.md)。

## 项目结构

```
PigeonVision/
├── src/
│   ├── stage1_data/         # 数据整理：图片索引、YOLO标注、样本对、多标签meta
│   ├── stage2_detection/    # YOLOv5 眼部检测训练与推理
│   ├── stage3_preprocess/   # U-Net 虹膜分割 + 椭圆Daugman归一化
│   ├── stage4_siamese/      # IrisEncoder 训练 + FAISS 特征库 + 多模型融合
│   ├── stage5_server/       # Flask 后端服务 (默认 Concat 1024d 融合模型)
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
│   ├── iris_normalized/     # 64×512 虹膜归一化图 + normalize_meta.csv
│   └── features/
│       ├── fusion_1024d_full/  # 默认：Concat 1024d 特征库 + FAISS索引
│       └── *.json              # 评估指标
└── ROADMAP.md               # 技术路线总文档
```

## 数据托管策略

| 内容 | 位置 | 说明 |
|------|------|------|
| 源码、配置、CSV 元数据 | **GitHub** | `src/`, `configs/`, `data/*.csv`, `outputs/**/*.csv` |
| 原始鸽眼图 | **Hugging Face** | `data/extracted/{1..12}/*.jpg` |
| YOLO 眼部裁剪 | **Hugging Face** | `outputs/eye_crops/*.jpg` |
| 虹膜归一化图 | **Hugging Face** | `outputs/iris_normalized/*.png` |
| 模型权重 | **Hugging Face** | `checkpoints/` 全部内容 |
| 特征向量 + FAISS | **Hugging Face** | `outputs/features/*.npy`, `*.bin` |

## API 接口

### POST /compare

```bash
curl -X POST http://localhost:5000/compare \
  -F "image_a=@iris1.png" \
  -F "image_b=@iris2.png" \
  -F "eye_crop=1"
```

返回: `{"distance": 0.83, "same_family": false, "threshold": 1.3}`

### POST /search

```bash
curl -X POST http://localhost:5000/search \
  -F "image=@iris.png" \
  -F "top_k=20" \
  -F "eye_crop=1"
```

返回: `{"results": [{"rank": 1, "img_id": "571835", "pg_id": "2016-26-0571835", "blood_name": "桑杰士", "distance": 0.21, "image_url": "/image/571835"}, ...]}`

`top_k` 默认 20，最大 100。`eye_crop=1` 表示上传内容已经是 Android 端裁好的眼部图，服务端会跳过 YOLO 检测，直接进入 U-Net 分割和特征检索。

### GET /image/<img_id>

返回 `outputs/img_index.csv` 中对应的原始鸽眼图，用于 Android 检索结果缩略图展示。

## 技术栈

- **目标检测**: YOLOv5s (ultralytics)
- **虹膜分割**: U-Net (PyTorch, GroupNorm)
- **椭圆展开**: Daugman 极坐标重映射
- **特征编码**: ResNet34/50 + Triplet/SupCon/ArcFace → Concat 1024d 融合
- **向量检索**: FAISS IndexFlatL2
- **后端服务**: Flask
- **检索效果**: R@1 35.5%, R@10 59.1%, Compare AUC 73.1%

## 许可证

待定
