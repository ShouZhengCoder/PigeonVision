# Iris-Based Kinship Recognition in Pigeons via Pedigree-Graph Supervised Representation Learning

> **Branch `paper`** - 论文工作分支。课程大作业（stages 1-6 部署系统）在 `master` 分支 / `course-final` tag。
> 课程相关的旧文档已移至 `archive/course/`。

面向 CCF-B 表征学习期刊投稿。三个贡献：

1. **数据集 + 证明虹膜可判定血缘**：虹膜特征距离与族谱亲缘强相关（Spearman −0.70，AUC 0.91，分层单调）。
2. **族谱图编码定义血缘远近**：解析血统书原文构建族谱图，两套等价编码（字面递归拼接+LCS / 贡献向量+亲缘系数），单调性验证通过。
3. **族谱亲缘监督训练（方法）**：用族谱 k 做 graded SupCon，公平对比下 graded nDCG +19% 胜 IDF 基线。

## 目录结构（论文相关）

```
paper/                      # 论文材料
├── paper_draft.md          # 论文初稿（intro/related/method/dataset/experiments+ablation）
├── dataset_card_draft.md   # 数据集卡（含 Phase B 实证）
├── experiments_table.md    # 实验对比表
├── 汇报_老师.md             # 阶段汇报
├── phaseA_kinship_validation.md   # 贡献②验证报告
├── phaseB_iris_kinship_*.md       # 贡献①/③评估报告（full/split/kinship_v2/idf_150/ped_only_150）
└── fig/                    # 论文图

src/stage7_kinship/         # 论文核心代码
├── parse_pedigree.py       # 解析 details.txt 族谱
├── build_pedigree_graph.py # 建族谱图
├── kinship_encoding.py     # 贡献向量 + 亲缘系数 k（形式 B）
├── kinship_literal.py      # 字面递归拼接 + LCS（形式 A，贴老师原意）
├── validate_kinship.py     # 单调性验证
├── visualize_pedigree.py   # 族谱图可视化
└── phaseB_iris_kinship.py  # 虹膜-亲缘相关性评估

src/stage1_data/            # 数据 pipeline（论文数据集）
src/stage2_detection/       # YOLOv5 眼部检测
src/stage3_preprocess/      # U-Net 虹膜分割 + Daugman 归一化
src/stage4_siamese/         # IrisEncoder + graded SupCon（Phase C 改动）
data/pedigree/              # 族谱产物（parsed_pedigree/graph/vectors/codes）
configs/                    # 训练配置
scripts/                    # 工具脚本（含 sync_hf, run_phaseD_ablation）
```

> `src/stage5_server/`、`src/stage6_android/` 是课程部署系统（Flask + Android），论文不直接使用但保留作编码器落地。

## 复现

```bash
# 1. 族谱编码（贡献②）
python src/stage7_kinship/parse_pedigree.py
python src/stage7_kinship/build_pedigree_graph.py
python src/stage7_kinship/kinship_encoding.py
python src/stage7_kinship/validate_kinship.py

# 2. 虹膜-亲缘相关性（贡献①）
python src/stage7_kinship/phaseB_iris_kinship.py --tag split

# 3. 族谱监督训练（贡献③, Phase C）
python src/stage4_siamese/train_relation_supcon.py \
  --kinship-source pedigree --positive-cutoff 0.15 --fallback-scale 0.3 \
  --output-dir checkpoints/siamese/relation_supcon_kinship_v2 --epochs 150

# 4. 消融（Phase D）
bash scripts/run_phaseD_ablation.sh
```

## 关键结果

| 模型 (150ep) | Spearman(虹膜,k) | graded nDCG@20 | AUC 任意亲缘 | 分层单调 |
|---|:---:|:---:|:---:|:---:|
| IDF 基线 | −0.711 | 0.467 | 0.921 | ✅ |
| **族谱监督 hybrid（我们）** | **−0.716** | **0.558（+19%）** | **0.931** | ✅ |
| 纯族谱（无兜底） | −0.601 | 0.280 | 0.882 | ❌ |

详见 `paper/experiments_table.md` 与 `paper/paper_draft.md`。

## 数据托管

- 源码/配置/小型 CSV：GitHub（本仓库）
- 原始图/裁剪/归一化图/权重/特征/FAISS：HuggingFace `jshouEX/pigeon-breed-image-dataset`
