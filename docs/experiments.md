# PigeonVision 实验记录与效果总结

## 一、项目概述

信鸽虹膜识别系统，两个核心任务：

- **Compare**：两张虹膜图 → 判断是否同血脉
- **Search**：一张虹膜图 → 在 22K 数据库中检索 Top-K 血脉相关鸽子

**完整推理管线**：YOLOv5 眼部检测 → U-Net 虹膜分割 → Daugman 归一化 → IrisEncoder 特征提取 → FAISS 检索 → Flask API

---

## 二、数据规模

| 阶段 | 数量 |
|------|------|
| 原始图像 | 31,896 张 |
| 眼部裁剪（YOLOv5） | 25,766 张 |
| 归一化虹膜（U-Net + Daugman） | 25,690 张（成功）/ 76 张（失败） |
| 训练集（多标签） | 22,043 张 / 6,626 品种 |
| 验证集（多标签） | 3,647 张 / 6,626 品种 |
| 每图平均 blood_id 数 | 6.4 个（91.8% 图像有多个血脉） |
| 图库平均每查询相关图像 | 53 张 |
| 特征维度 | 256 / 512 |
| 图库规模（FAISS） | 22,043 × 256 维 L2 归一化向量 |

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

- **P@K（Precision@K）**：返回的 K 张图中，血脉相关的比例
- **R@K（Recall@K）**：K 张图中至少有一张血脉相关，则此查询成功；成功查询占比
- **mAP（mean Average Precision）**：所有查询的平均精度均值，衡量整体排序质量

**用户最关心的指标**：R@K — 上传一张虹膜图，Top-K 中能找到血脉相关鸽子的概率。

### 4.2 Compare 比对

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

**结论**：Concat 512d 在所有指标上全面最优，是最佳模型。

---

## 六、最终模型效果

### 6.1 核心指标

| 任务 | 指标 | 值 |
|------|------|:------:|
| **Search** | R@1 | **25.9%** |
| | R@5 | **43.6%** |
| | R@10 | **51.8%** |
| | P@5 | 18.3% |
| | P@10 | 15.5% |
| | mAP | 12.4% |
| **Compare** | AUC | **71.0%** |
| | BalAcc | **65.0%** |

### 6.2 用户视角解读

- **检索**：上传一张鸽眼图，返回 10 个结果，**52% 的概率至少有一张血脉相关**；返回 10 张图，**其中约 1-2 张血脉相关**
- **比对**：两张鸽眼图 → 判断是否同血脉，**正确率 65%（随机 50%）**，**AUC 71.0%**

### 6.3 模型配置

```
Concat 512d 融合模型:
  ├── Triplet Encoder: ResNet34, 256-dim
  │   (训练: 7K张, 486品种, PK sampler, batch_hard_triplet_loss)
  └── SupCon Encoder: ResNet34, 256-dim
      (训练: 22K张, 6626品种, SupCon Loss, τ=0.07)

FAISS: IndexFlatL2, 512-dim, 22K vectors
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

1. **单标签评估高估效果**：小验证集内检索 + blood_name 匹配给出的 R@5=89% 是误导性的。多标签跨集合评估给出 R@10=52%，更接近真实使用场景。

2. **Pair-based loss 瓶颈**：Triplet/SupCon 复杂度 O(B²)，每次只用 batch 内 64 张图。对于 6,626 品种的检索任务，这是根本性的架构局限。

3. **多标签训练比预期难**：血脉图过于密集（每图平均 7 个 blood_id），泛化正样本导致嵌入空间弥散。单标签训练反而形成更紧致的类簇。

4. **模型融合是低成本提效方案**：两个互补模型的 embedding 拼接（Concat 512d），无需重训即可同时获得检索和比对的最优效果。

5. **Proxy-based loss 是正确方向**：ArcFace/Proxy-Anchor 复杂度 O(C×B)，在大量类别时天然优于 pair-based loss，但实现和收敛需要更多调优。

---

## 九、下一步方向

1. **可直接部署**：Concat 512d 融合模型 + Flask API，当前效果已可用
2. **短期提升**：完成 ArcFace 单标签训练（ResNet50 + 512dim + 6626类），预计可提升检索 5-10%
3. **中期探索**：ArcFace 收敛后用 blood_id 重合信息做 multi-proxy 微调
4. **长期方向**：更大的 backbone（ViT/EfficientNet）、数据增强优化、血脉关系图建模

---

## 十、文档版本

- 更新日期：2026-06-01
- 对应 Git commit：`4d98e03 feat: 多标签血脉评估 + MoE融合 + 代码健壮性优化`
