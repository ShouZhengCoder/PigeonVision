#!/usr/bin/env bash
# Phase D 消融: 训练 + 评估多组配置, 编译对比表。
# 用法: bash scripts/run_phaseD_ablation.sh
# 注意: 每组训练 ~2-3h, 串行跑。先确认 v2 已锁定再运行。
set -e
cd "$(dirname "$0")/.."

EPOCHS=150
WARM=checkpoints/siamese/best.pt
COMMON="--epochs $EPOCHS --eval-every 5 --warm-start $WARM"

run_one () {
  local name="$1"; shift
  local cfg="$1"; shift
  local extra="$1"
  local ckpt="checkpoints/siamese/relation_supcon_${name}"
  local feat="outputs/features/relation_supcon_256d_${name}"
  if [ -f "$ckpt/best.pt" ]; then echo "[skip] $name 已有 best.pt"; else
    python3 src/stage4_siamese/train_relation_supcon.py \
      --kinship-source $cfg --output-dir "$ckpt" $COMMON $extra
  fi
  python3 src/stage4_siamese/build_db_fusion.py --mode full \
    --checkpoint "$ckpt/best.pt" --output-dir "$feat" 2>&1 | tail -2
  python3 src/stage7_kinship/phaseB_iris_kinship.py \
    --features "$feat/feature_db.npy" --meta "$feat/feature_db_meta.csv" \
    --tag "$name" 2>&1 | tail -16
}

# D1: kinship source 消融 (idf / pedigree_hybrid / pedigree_only)
run_one idf_150        idf         ""
run_one ped_hybrid_150 pedigree    "--fallback-scale 0.3 --positive-cutoff 0.15"
run_one ped_only_150   pedigree    "--fallback-scale 0.0 --positive-cutoff 0.15"

echo "=== Phase D 消融完成, 见 paper/phaseB_iris_kinship_*.md ==="
