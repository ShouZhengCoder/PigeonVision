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
python src/stage5_server/app.py
```

访问 http://localhost:5000 使用 Web 界面。

## 项目结构

```
PigeonVision/
├── src/
│   ├── stage1_data/         # 数据整理：图片索引、YOLO标注、样本对
│   ├── stage2_detection/    # YOLOv5 眼部检测训练与推理
│   ├── stage3_preprocess/   # U-Net 虹膜分割 + 椭圆展开归一化
│   ├── stage4_siamese/      # Triplet 孪生网络训练 + FAISS 特征库
│   └── stage5_server/       # Flask 后端服务
├── configs/                 # 训练配置文件
├── scripts/                 # 工具脚本
│   ├── setup_data.py        # 从 HF 仓库创建数据/权重符号链接
│   └── sync_hf.py           # 将本地数据同步到 HF 仓库
├── data/                    # 元数据 CSV（大文件在 HF）
│   ├── pairs_train.csv
│   ├── pairs_val.csv
│   ├── train_meta.csv
│   └── val_meta.csv
├── outputs/                 # 中间产物 CSV（图片在 HF）
│   ├── eye_crops/           # crop_meta.csv
│   ├── iris_normalized/     # normalize_meta.csv
│   └── features/            # eval_metrics.json, threshold.json
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
  -F "image_b=@iris2.png"
```

返回: `{"distance": 0.83, "same_family": false, "threshold": 1.0}`

### POST /search

```bash
curl -X POST http://localhost:5000/search \
  -F "image=@iris.png" \
  -F "top_k=10"
```

## 技术栈

- **目标检测**: YOLOv5s (ultralytics)
- **虹膜分割**: U-Net (PyTorch, GroupNorm)
- **椭圆展开**: Daugman 极坐标重映射
- **特征编码**: ResNet34 + Triplet Loss
- **向量检索**: FAISS IndexFlatL2
- **后端服务**: Flask

## 许可证

待定
