#!/usr/bin/env python3
"""Sync new/changed files to Hugging Face dataset repo in a single commit.

Usage:
  python scripts/sync_hf.py                     # dry-run
  python scripts/sync_hf.py --execute           # upload (one commit, avoids rate limit)
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_folder

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "jshouEX/pigeon-breed-image-dataset"
REPO_TYPE = "dataset"

# Files to sync (relative to project root)
SYNC_FILES = [
    # Metadata CSVs
    "data/train_meta.csv",
    "data/val_meta.csv",
    "data/train_multi_meta.csv",
    "data/val_multi_meta.csv",
    "data/blood_id_map.json",
    "data/pairs_train.csv",
    "data/pairs_val.csv",
    "data/yolo_dataset/data.yaml",
    # Output metadata
    "outputs/img_index.csv",
    "outputs/eye_crops/crop_meta.csv",
    "outputs/iris_normalized/normalize_meta.csv",
    # Feature DBs (single model)
    "outputs/features/feature_db.npy",
    "outputs/features/feature_db_meta.csv",
    "outputs/features/faiss_index.bin",
    "outputs/features/eval_metrics.json",
    "outputs/features/eval_comparison.json",
    "outputs/features/threshold.json",
    # Relation-SupCon 256d (current default)
    "outputs/features/relation_supcon_256d/faiss_index.bin",
    "outputs/features/relation_supcon_256d/feature_db.npy",
    "outputs/features/relation_supcon_256d/feature_db_meta.csv",
    "outputs/features/relation_supcon_256d/build_manifest.json",
    "outputs/features/relation_supcon_256d/eval_metrics.json",
    "outputs/features/relation_supcon_256d/eval_comparison.json",
    "outputs/features/relation_supcon_256d/threshold.json",
    # Relation-SupCon 256d PG_ID centroid (search mode=pg)
    "outputs/features/relation_supcon_256d_pg/faiss_index.bin",
    "outputs/features/relation_supcon_256d_pg/feature_db.npy",
    "outputs/features/relation_supcon_256d_pg/feature_db_meta.csv",
    "outputs/features/relation_supcon_256d_pg/build_manifest.json",
    "outputs/features/relation_supcon_256d_pg/threshold.json",
    # Fusion 1024d (legacy)
    "outputs/features/fusion_1024d_full/faiss_index.bin",
    "outputs/features/fusion_1024d_full/feature_db.npy",
    "outputs/features/fusion_1024d_full/feature_db_meta.csv",
    "outputs/features/fusion_1024d_full/eval_metrics.json",
    "outputs/features/fusion_1024d_full/eval_comparison.json",
    "outputs/features/fusion_1024d_full/threshold.json",
    "outputs/features/fusion_1024d_full/build_manifest.json",
    # Checkpoints
    "checkpoints/detection/exp/weights/best.pt",
    "checkpoints/segmentation/best.pt",
    "checkpoints/siamese/best.pt",
    "checkpoints/siamese/last.pt",
    "checkpoints/siamese/supcon/best.pt",
    "checkpoints/siamese/supcon/last.pt",
    "checkpoints/siamese/relation_supcon/best.pt",
    "checkpoints/siamese/arcface_v2/best.pt",
    "checkpoints/siamese/arcface/best.pt",
    "checkpoints/siamese/pk_supcon/best.pt",
    "checkpoints/siamese/pk_supcon_v2/best.pt",
    "checkpoints/siamese/proxy_anchor/best.pt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync to HF in a single commit.")
    parser.add_argument("--execute", action="store_true", help="Actually upload.")
    parser.add_argument("--repo", default=REPO_ID, help="HF repo ID.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolved: list[Path] = []
    missing: list[str] = []
    for rel in SYNC_FILES:
        p = ROOT / rel
        if p.is_file():
            resolved.append(p)
        else:
            missing.append(rel)

    if missing:
        print(f"[warn] {len(missing)} files not found:")
        for m in missing:
            print(f"  - {m}")

    total_sz = sum(p.stat().st_size for p in resolved)
    print(f"Files to sync: {len(resolved)}, Total: {total_sz / 1024 / 1024:.1f} MB")
    if not args.execute:
        print("[DRY RUN] 使用 --execute 执行上传。")
        return 0

    # Copy to temp dir with repo structure, upload in ONE commit
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for fp in resolved:
            dest = tmp / fp.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fp, dest)

        print(f"Uploading to {args.repo} (single commit, {len(resolved)} files)...")
        create_repo(args.repo, repo_type=REPO_TYPE, exist_ok=True)
        upload_folder(
            folder_path=str(tmp),
            repo_id=args.repo,
            repo_type=REPO_TYPE,
            path_in_repo="",
            commit_message=f"sync: {len(resolved)} files, {total_sz / 1024 / 1024:.1f} MB",
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
