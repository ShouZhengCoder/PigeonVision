---
license: mit
task_categories:
  - image-classification
  - image-segmentation
tags:
  - pigeon
  - iris-recognition
  - biometrics
  - computer-vision
pretty_name: Pigeon Breed Iris Image Dataset
size_categories:
  - 10M<n<100M
---

# Pigeon Breed Iris Image Dataset

信鸽虹膜品种识别数据集，配套 [PigeonVision](https://github.com/ShouZhengCoder/PigeonVision) 项目使用。

## 目录结构

```
pigeon-breed-image-dataset/
├── data/
│   ├── extracted/           # 原始鸽眼照片 (31,900 张)
│   │   ├── 1/  ~ 12/       # 按子目录分组的 JPG 图片
│   │   └── ...
│   └── unet_labelme_80/    # U-Net 虹膜分割手工标注 (80 组)
│       ├── images/          # 原始眼部特写
│       ├── annotations/     # Labelme JSON 标注
│       └── masks/           # 分割 mask (0=背景, 1=虹膜, 2=瞳孔)
│
├── outputs/
│   ├── eye_crops/           # YOLOv5 眼部检测裁剪结果 (25,700 张)
│   ├── iris_normalized/     # U-Net 分割 + 椭圆展开归一化虹膜 (25,700 张)
│   │                        # 尺寸: 64×512 灰度 PNG
│   └── features/            # 孪生网络编码的特征向量 + FAISS 索引
│       ├── feature_db.npy   # N×256 L2 归一化特征矩阵
│       ├── faiss_index.bin  # FAISS IndexFlatL2 索引
│       └── feature_db_meta.csv  # img_id, pg_id, blood
│
└── checkpoints/
    ├── detection/           # YOLOv5s 眼部检测模型权重
    ├── segmentation/        # U-Net 虹膜分割模型权重
    └── siamese/             # ResNet34 孪生网络编码器权重
```

## 数据说明

### 原始鸽眼图 (data/extracted/)

来自信鸽养殖场的标准鸽眼摄影，包含 31,900 张 JPG 彩色图片。每张图片对应一只鸽子，图片 ID 与 `pigeon.csv` 中的 ID 字段对应。

### 品系标注

标注数据位于 GitHub 仓库的 `data/extracted/datasetXGN/` 目录下：
- `pigeon.csv` — 鸽子记录（ID, PG_ID 环号, BLOOD 品系, 眼色, 羽色等）
- `relations.csv` — 血统-图片关系映射
- `anotations/` — 9,979 个 JSON 标注文件（眼部 bounding box）

### 虹膜归一化流程

```
原始鸽眼图 → YOLOv5 眼部检测 → 眼部裁剪 → U-Net 三分类分割
→ 椭圆拟合 (pupil/iris) → Daugman 椭圆展开 → 64×512 虹膜纹理图
```

## 使用方式

```bash
# 克隆数据仓库
git clone https://huggingface.co/datasets/jshouEX/pigeon-breed-image-dataset

# 配合 PigeonVision 项目使用
git clone https://github.com/ShouZhengCoder/PigeonVision.git
cd PigeonVision
python scripts/setup_data.py --hf-dir ../pigeon-breed-image-dataset
```

或通过环境变量指定数据路径：

```bash
export PIGEONVISION_DATA=/path/to/pigeon-breed-image-dataset
```

## 引用

如果你使用了本数据集，请引用 PigeonVision 项目。

## 许可证

MIT License
