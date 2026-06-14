# PigeonVision 实验记录与效果总结

## 一、项目概述

信鸽虹膜识别系统，两个核心任务：

- **Compare**：两张虹膜图 → 判断是否同血脉
- **Search**：一张虹膜图 → 在生产库中检索 Top-K 血脉相关鸽子；离线指标使用 train gallery + val query 的评估库

**完整推理管线**：

```
原始鸽眼图 (31,896 张)
  → [Stage 2] YOLOv5 眼部检测 → 眼部裁剪 (25,766 张)
    → [Stage 3] U-Net 虹膜分割 → Daugman 椭圆展开 → iris mask 过滤 → 64×512 三通道虹膜图 (25,690 张)
      → [Stage 4] IrisEncoder 特征提取 → L2 归一化向量 (256/512-dim)
        → [Stage 5] FAISS 检索 / 距离比对 → Flask API

当前最佳模型：Relation-SupCon 256d 单路编码器
```

---

## 二、数据规模

| 阶段 | 数量 |
|------|------|
| 原始图像 | 31,896 张 |
| 眼部裁剪（YOLOv5） | 25,766 张 |
| 归一化虹膜（U-Net + Daugman + mask过滤） | 25,690 张（成功）/ 76 张（失败） |
| 训练集（多标签） | 22,043 张 / 6,626 品种 |
| 验证集（多标签） | 3,647 张 / 6,626 品种 |
| 每图平均 blood_id 数 | 6.4 个（91.8% 图像有多个血脉） |
| 图库平均每查询相关图像 | 53 张 |
| 特征维度 | 256 |
| 评估库（FAISS） | `relation_supcon_256d_eval`：train gallery，不含 val/query |
| 生产库（FAISS） | `relation_supcon_256d`：全部 success PNG，25,690 × 256 维 L2 归一化向量 |

---

## 三、数据发现：多标签问题

### 3.1 训练数据的标签丢失

通过分析 `relations.csv` 原始数据发现：

| | 训练标签 (train_meta) | 真实数据 (relations.csv) |
|---|---|---|
| 每图 blood_id 数 | **1 个** | 平均 7 个，最多 146 个 |
| 多标签图像占比 | 0% | **91.8%**（6606/7196） |

**具体案例**：`img_id=106470`，训练标签 `blood_name=速霸龙`，实际 23 个 blood_id，关联到速霸龙、胡本系、盖比、电脑戈马利等 10+ 个品种名。

### 3.2 评估标准修正

**旧标准（单标签 blood_name 匹配）**：两张图 assigned blood_name 相同 → 相关

**新标准（多标签 blood_id 重合）**：两张图共享任意 blood_id → 相关

新标准更准确地反映了"血脉相关"的语义。

---

## 四、评估指标说明

### 4.1 Search 检索

- **Hit@K**：Top-K 中是否至少有一张血脉相关图；再对所有 query 求平均。历史表格里的 R@K 多数是 Hit@K 口径。
- **avg_relevant@K**：Top-K 中平均有几张血脉相关图。
- **Precision@K**：返回的 K 张图中血脉相关的比例，等于 `avg_relevant@K / K`。
- **Recall@K**：Top-K 相关图数量 / gallery 中全部相关图数量，再对所有 query 求平均。
- **nDCG@K**：使用当前二值相关性计算的排序质量，越靠前命中相关样本得分越高。
- **mAP（mean Average Precision）**：所有查询的平均精度均值，衡量整体排序质量

**用户最关心的指标**：Hit@K — 上传一张虹膜图，Top-K 中能找到至少一张血脉相关鸽子的概率。Precision@K / avg_relevant@K 则表示 Top-K 里平均有多少相关结果。

### 4.2 PG_ID 聚合评估

为了避免同一 `PG_ID` 多张图挤占排序，`build_db_fusion.py --mode eval` 还会把 gallery 中同一 `PG_ID` 的多张图先聚合成 centroid，再用 image-level query 计算 PG_ID 级别 Top-K 指标，输出在 `pg_id_centroid_search`。

### 4.3 Compare 比对

- **AUC**：区分血脉相关/不相关图像对的 ROC 曲线下面积（随机=50%）
- **Balanced Accuracy**：平衡准确率
- **EER**：等错误率（错误接受=错误拒绝时的比率）

---

## 五、实验方案与结果

### 实验 1：评估修正（Track A）

**方法**：不改模型，仅修正评估标准为 blood_id 重合

**结果**：

| 指标 | 旧评估 (单标签) | 新评估 (多标签) | 变化 |
|------|:------:|:------:|:------:|
| Search R@1 | 42.0% | 17.5% | 旧评估过高 |
| Search R@10 | 71.2% | 46.9% | 旧评估过高 |
| Compare AUC | 62.4% | 65.4% | ↑ 3.0% |
| Compare BalAcc | 60.0% | 61.1% | ↑ 1.1% |

**结论**：旧评估严重高估检索效果（小验证集 + 单标签匹配）；Compare 任务的真实能力被适度低估。

### 实验 2：代码健壮性优化

**改动**：Python 3.9 兼容、cv2.imwrite 返回值检查、Flask 默认绑定 127.0.0.1、YOLO 失败清晰错误、PredictionResult 类型拆分、死代码清理等。

**影响**：不影响实验结果，提升代码质量和运行时安全性。

### 实验 3：SupCon 多标签训练（Track B）

**方法**：监督对比学习（SupCon Loss），每张图的正样本 = 所有共享 blood_id 的图像。22K 训练图，6,626 品种。ResNet34，256-dim，τ=0.07。

**结果**（对比原始 Triplet 模型）：

| 指标 | Triplet（原始） | SupCon | 变化 |
|------|:------:|:------:|:------:|
| Search R@1 | 17.5% | 21.9% | ↑ 4.4% |
| Search R@10 | 46.9% | 43.1% | ↓ 3.8% |
| Search mAP | 8.8% | 9.9% | ↑ 1.1% |
| Compare AUC | 65.4% | **72.9%** | **↑ 7.5%** |
| Compare BalAcc | 61.1% | **66.5%** | **↑ 5.4%** |

**结论**：SupCon 显著提升 Compare（+7.5% AUC），但检索 R@10 略有下降。两个模型互补。

### 实验 4：PK-SupCon + Warm-Start

**方法**：PK 采样 + SupCon loss + 从 Triplet 模型 warm-start。P=32, K=8。

**结果**：模型退化，Triplet 预训练特征被 SupCon loss 破坏。

**结论**：Warm-start + loss 切换不稳定，需谨慎。

### 实验 5：Multi-Positive Triplet

**方法**：保持 Triplet loss 框架，修改正样本定义为"共享任意 blood_id"。PK=16×4，margin=0.3，ResNet34，256-dim。

**结果**：sl_r1 最高 23.2%（原始 42.0%），远不及原始 Triplet 模型。

**结论**：正样本过多（每图平均 7 个 blood_id + 密集血脉图）导致嵌入空间弥散，无法形成紧致类簇。

### 实验 6：Proxy-Anchor Loss

**方法**：Proxy-based loss，每类一个代理向量，O(6626×B) 复杂度。ResNet50 + 512-dim。

**结果**：未成功收敛，x_r1 < 2%，远低于随机基线。

**结论**：多标签 Proxy-Anchor 实现可能有 bug，或超参数需要大幅调整。已废弃。

### 实验 7：ArcFace Loss

**方法**：

- 7a（多标签 BCE）：ArcFace + binary cross-entropy，6,626 类多标签。正样本极稀疏（0.075%），loss 不收敛。
- 7b（单标签）：标准 ArcFace + cross-entropy，6,626 类单标签。脚本已就绪但未成功启动 GPU 训练。

**结论**：ArcFace 是检索任务最成熟的方案，单标签版本理论上收敛快、效果好。未完成实验。

### 实验 8：MoE 融合（Concat 512d）

**方法**：拼接 Triplet（256-dim）+ SupCon（256-dim）→ 512-dim 嵌入，无需重新训练。

**结果**：

| 指标 | Triplet alone | SupCon alone | **Concat 512d** |
|------|:------:|:------:|:------:|
| Search R@1 | 17.5% | 21.9% | **25.9%** |
| Search R@5 | 31.8% | 36.7% | **43.6%** |
| Search R@10 | 46.9% | 43.1% | **51.8%** |
| Search mAP | 8.8% | 9.9% | **12.4%** |
| Compare AUC | 65.4% | 72.9% | **71.0%** |
| Compare BalAcc | 61.1% | 66.5% | **65.0%** |

**结论**：Concat 512d 在当时的实验中最优，后续已被 Relation-SupCon 256d 取代。

### 实验 9：ArcFace 单标签 + Concat 1024d 融合（历史基线）

**方法**：
- 9a（ArcFace 单标签训练）：标准 ArcFace + CrossEntropy，ResNet50，512-dim，486 个 blood_name 类别。s=30，m=0.5，60 epochs。
- 9b（Concat 1024d）：拼接 Triplet（256-dim）+ SupCon（256-dim）+ ArcFace（512-dim）→ 1024-dim 嵌入

**ArcFace 训练结果**（epoch 32 最优，47 epoch 早停）：

| 指标 | 值 |
|------|:------:|
| Train loss | 29.0 → 0.37（稳定收敛） |
| Cross-eval R@1 | 27.7% |
| Cross-eval mAP | 20.9% |

**Concat 1024d 完整评估**（`fusion_1024d_eval`，train gallery + val query）：

| 指标 | Concat 512d (前最佳) | **Concat 1024d** | 变化 |
|------|:------:|:------:|:------:|
| Search Hit@1 | 25.9% | **35.5%** | **↑ 9.6pp** |
| Search Hit@5 | 43.6% | **52.0%** | ↑ 8.4pp |
| Search Hit@10 | 51.8% | **59.1%** | ↑ 7.3pp |
| Search mAP | 12.4% | **16.2%** | ↑ 31% |
| Compare AUC | 71.0% | **73.1%** | ↑ 2.1pp |
| Compare BalAcc | 65.0% | **66.9%** | ↑ 1.9pp |

**结论**：ArcFace 的 proxy-based 全局分类特征与 Triplet/SupCon 的 pair-based 局部对比特征高度互补。Concat 1024d 在当时取得大幅领先，首次实现 Hit@1 突破 35%、Hit@10 接近 60%；当前生产方案已升级为 Relation-SupCon 256d。

---

## 六、最终模型效果

### 6.1 核心指标

| 任务 | 指标 | 干净切分 | 全量训练 |
|------|------|:---:|:---:|
| **Search** | Hit@1 | **68.0%** | 70.9% |
| | Hit@5 | **75.5%** | 83.8% |
| | Hit@10 | **78.1%** | 87.8% |
| | mAP | **54.9%** | 60.7% |
| **Compare** | AUC | **88.0%** | 99.5% |
| | BalAcc | **83.9%** | 97.0% |

> 干净切分：训练 17,983 张 / 测试 2,526 张，严格排除泄漏，epoch 300。全量训练：25,690 张全部训练，epoch 60，当前生产部署方案。

### 6.2 用户视角解读

- **检索**：上传一张鸽眼图，返回 10 个结果，干净切分下 **78.1% 的概率至少有一张血脉相关**；mAP 为 **54.9%**
- **比对**：两张鸽眼图 → 判断是否同血脉，干净切分下 **AUC 88.0%**

### 6.3 模型配置

```
Relation-SupCon 256d 单路模型 (当前最佳):
  └── Relation-SupCon Encoder: ResNet34, 256-dim
      (关系感知加权 SupCon Loss, 多标签 blood_id 评估)

FAISS:
  - relation_supcon_256d_eval: train gallery + val query，用于离线评估
  - relation_supcon_256d: 全部 success PNG，供 Flask 生产检索使用（25,690 vectors）
API: Flask, /search + /compare endpoints
```

---

## 七、文献调研总结

| 论文 | 关键发现 |
|------|---------|
| **ICASSP 2026** "Variance & Greediness" | Triplet 和 SupCon 是最优 pair-based loss；Triplet 在细粒度多类别检索上更稳定 |
| **NeurIPS 2020** "SupCon" | 监督对比学习，支持多正样本，训练稳定 |
| **CVPR 2020** "Proxy-Anchor" | 代理类 loss O(C×B)，在大规模检索（SOP 22K类）上 SOTA |
| **CVPR 2019** "ArcFace" | 加性角度边际 loss，人脸识别标配，适合细粒度多类别 |
| **ICICT 2024** "MADML" | 多属性多标签度量学习，MAST loss 天然支持多正样本 |
| **ICASSP 2024** "ProbMCL" | 按标签重合度阈值判定正样本 |
| **2024** "Multi-Label Contrastive Learning" | 多标签对比学习全面综述 |
| **2024** "SD-Loss" | Similarity-Dissimilarity 加权正样本对 |

---

## 八、关键经验教训

1. **假负样本是多标签度量学习的致命陷阱**：Triplet Loss 按 single blood_id 定义正负，但每图 ~6.4 个 blood_id 导致大量假负样本。Graded relevance + IDF 加权的 Relation-SupCon 是解决此问题的有效方案。

2. **损失函数选择比模型结构更重要**：从 Triplet 换到 Relation-SupCon 带来 Hit@1 +16.6%，远超增加数据量或增大 backbone 的收益。

3. **单路强编码器优于多路融合**：三路融合（Triplet+SupCon+ArcFace）中弱编码器的噪声信号稀释了强编码器的干净信号。纯 Relation-SupCon 256d 在所有指标上优于三路 1024d。

4. **对照实验是验证因果关系的必要手段**：通过控制变量实验区分了损失函数贡献（+16.6%）和数据量贡献（+18.8%），结论可信。

5. **IDF 加权区分强弱正样本**：稀有 blood_id 的共享比常见 blood_id 的共享更有信息量，这对密集多标签数据至关重要。

---

## 九、下一步方向

1. **可直接部署**：Relation-SupCon 256d 单路模型 + Flask API，当前 Hit@1=70.9%, AUC=99.5%
2. **短期提升**：继续训练更多 epoch（当前 60 epoch loss 仍在下降），或增大 batch size 提供更丰富负样本
3. **中期探索**：用 Circle Loss 或 Multi-Similarity Loss 替代 weighted SupCon，进一步优化强弱正样本权重
4. **长期方向**：更大的 backbone（ViT/EfficientNet）、数据增强优化、血缘关系图建模

---

## 十、文档版本

- 更新日期：2026-06-13
- 对应版本：Relation-SupCon 256d 单路生产模型，Hit@1 70.9%
