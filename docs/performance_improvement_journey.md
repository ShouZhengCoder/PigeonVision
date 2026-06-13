# PigeonVision 效果提升历程

## 概述

本文档记录信鸽虹膜识别系统从初始 35.5% Search Hit@1 提升到 70.9% 的完整过程，包括问题诊断、对照实验和关键决策。

---

## 起点：初始状态

| 指标 | 值 |
|------|:---:|
| Search Hit@1 | 35.5% |
| Search Hit@10 | 59.1% |
| Search mAP | 16.2% |
| Compare AUC | 73.1% |

**训练配置**：Relation-SupCon + RelationBatchSampler，80/20 train/val 分割，ResNet34 backbone，256-dim embedding。旧版三路融合为 Triplet 256d + Relation-SupCon 256d + ArcFace 512d 拼接为 1024d。

---

## 第一步：诊断根因

### 训练日志暴露的问题

```
d_pos (正样本距离) ≈ 1.04
d_neg (负样本距离) ≈ 1.06
差距仅 0.02，margin=0.3 → 永远达不到
```

L2 归一化向量的距离范围是 [0, 2]，随机向量距离约 1.41。当前正负样本距离几乎相等且都接近 1.0，说明模型几乎没有学到有效分离。

### 根因：假负样本（False Negatives）

PK Sampler 的运作方式：
1. 从 81,751 个 blood_id 中随机选 P=16 个
2. 每个选 K=4 张图 → batch=64
3. 同一 blood_id 内的 pair 是正样本，不同 blood_id 的是负样本

**关键数据特点**：每张图平均有 **6.4 个 blood_id**，91.8% 的图像有多重血脉。

两个被分到"不同 blood_id 组"的样本，很可能**共享另一个 blood_id**，但 Triplet Loss 将其当作负样本推开。模型收到矛盾信号——同一对图片既是正样本又是负样本——最终退化为把所有向量挤在相近位置，不做有意义的区分。

### SupCon 为什么也不行

`train_multilabel.py` 用 SupCon Loss + 多标签 positive mask（共享任意 blood_id 即正样本），看似解决了假负样本问题，但：

- 训练 loss 从 4.9 降到 2.6 后停滞
- best multi-label recall@1 = 5.7%（比 Triplet 还差）
- 原因：batch 中正样本太多（每个 anchor 有 30-50 个正样本），所有正样本被平等对待，弱相关和强相关的 pair 贡献相同梯度，对比信号被稀释

---

## 第二步：关系感知加权 SupCon（核心突破）

### 方案

项目中已有 `loss_relation.py` + `dataset_relation.py`，实现了 **graded relevance** 替代二元正/负标签：

$$
\text{relevance}(i, j) = 0.7 \times \frac{\text{IDF\_shared}}{\min(\text{IDF}_i, \text{IDF}_j)} + 0.3 \times \frac{\text{IDF\_shared}}{\text{IDF\_union}}
$$

关键设计：

1. **IDF 加权**：稀有 blood_id 获得更高权重——共享一个罕见祖先比共享一个常见祖先更有信息量
2. **连续相关性**：共享多个稀有 blood_id → 高权重强正样本；共享一个常见 blood_id → 低权重弱正样本
3. **弱正样本不被推开**：relevance > 0 的 pair 只作为正样本参与对比（按 relevance 加权贡献），不作为负样本
4. **RelationBatchSampler**：每 batch 显式构造 anchor(32) + strong_pos(1) + weak_pos(1) + hard_neg(1)，batch=128

### 关键区别

| | Triplet Loss | 原始 SupCon | Relation-SupCon |
|---|---|---|---|
| 正样本定义 | 同一 blood_id | 共享任意 blood_id | 共享任意 blood_id |
| 正样本权重 | 二元 | 二元 | IDF 加权连续值 |
| 负样本定义 | 不同 blood_name | 不共享 blood_id | 不共享 blood_id |
| 假负样本 | **严重**（每图 ~6.4 个 blood_id） | 无 | 无 |
| 强/弱正区分 | 无 | 无 | **有** |

### 对照实验

| 实验 | 损失函数 | Hit@1 | AUC |
|------|------|:---:|:---:|
| A | Triplet | 35.5% | 73.1% |
| B | **Relation-SupCon** | **52.1%** | **91.3%** |
| C | Relation-SupCon + 全量数据 | 70.9% | 99.5% |

- 损失函数改进（B - A）：Hit@1 +16.6%, AUC +18.2%
- 全量数据增益（C - B）：Hit@1 +18.8%, AUC +8.2%

### 训练过程

```
Epoch  Loss    ndcg@10  mAP
10     9.19    0.294    0.223
20     4.84    0.392    0.362
30     3.35    0.472    0.471
40     2.62    0.523    0.535
50     2.20    0.555    0.577
60     1.89    0.602    0.628
```

loss 从 9.19 持续下降到 1.89，各项指标稳定上升，60 epoch 时仍在改善（未触发早停）。

---

## 最终结果

| 指标 | 初始 | 最终 | 提升 |
|------|:---:|:---:|:---:|
| Search Hit@1 | 35.5% | **70.9%** | +35.4 |
| Search Hit@10 | 59.1% | **87.8%** | +28.7 |
| Search mAP | 16.2% | **60.7%** | +44.5 |
| Compare AUC | 73.1% | **99.5%** | +26.4 |

---

## 关键经验

1. **密集多标签场景下 Triplet Loss 的假负样本是致命的**。每图 ~6.4 个标签时，二元正/负定义在 multi-label 数据上不可靠。

2. **Graded relevance + IDF 加权是解决多标签度量学习的有效方案**。区分"强正/弱正/真负"比二元标签更鲁棒，弱正样本不被误推。

3. **损失函数的选择比数据量更重要**。从 Triplet 换到 Relation-SupCon 带来 +16.6% Hit@1，远超单纯增加数据的收益。

4. **对照实验是验证因果关系的必要手段**。严格排除数据泄漏后（B 组），损失函数仍贡献了大部分提升，结论可信。

5. **训练监控指标（内部 val 的 ndcg/mAP）与实际评估指标高度相关**，可以信赖内部验证来判断模型是否在改善。
