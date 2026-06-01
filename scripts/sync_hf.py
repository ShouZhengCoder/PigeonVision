#!/usr/bin/env python3
"""Sync local data/output/checkpoint files to Hugging Face dataset repo.

Usage:
  python scripts/sync_hf.py                     # dry-run (list what would change)
  python scripts/sync_hf.py --execute           # actually upload
  python scripts/sync_hf.py --execute --resume  # skip already-uploaded files
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "jshouEX/pigeon-breed-image-dataset"
REPO_TYPE = "dataset"

# Local path -> HF path mapping
# Only sync new/changed files; bulk images already on HF from prior uploads.
SYNC_PLAN = [
    # --- Metadata CSVs (small, frequently updated) ---
    ("data", "data", "train_meta.csv", "训练元数据"),
    ("data", "data", "val_meta.csv", "验证元数据"),
    ("data", "data", "train_multi_meta.csv", "多标签训练元数据"),
    ("data", "data", "val_multi_meta.csv", "多标签验证元数据"),
    ("data", "data", "blood_id_map.json", "血脉ID映射"),
    ("data", "data", "pairs_train.csv", "训练样本对"),
    ("data", "data", "pairs_val.csv", "验证样本对"),
    ("data/yolo_dataset", "data/yolo_dataset", "data.yaml", "YOLO 数据集配置"),
    # --- Output metadata ---
    ("outputs/eye_crops", "outputs/eye_crops", "crop_meta.csv", "眼部裁剪元数据"),
    ("outputs/iris_normalized", "outputs/iris_normalized", "normalize_meta.csv", "归一化元数据"),
    ("outputs", "outputs", "img_index.csv", "图片索引"),
    # --- Feature databases ---
    ("outputs/features", "outputs/features", "*.json", "评估指标 JSON"),
    ("outputs/features", "outputs/features", "feature_db.npy", "特征向量 (单模型)"),
    ("outputs/features", "outputs/features", "feature_db_meta.csv", "特征库元数据"),
    ("outputs/features", "outputs/features", "faiss_index.bin", "FAISS 索引"),
    # --- Fusion 1024d (new) ---
    ("outputs/features/fusion_1024d_full", "outputs/features/fusion_1024d_full", "faiss_index.bin", "FAISS 索引 (1024d)"),
    ("outputs/features/fusion_1024d_full", "outputs/features/fusion_1024d_full", "feature_db.npy", "特征向量 (1024d)"),
    ("outputs/features/fusion_1024d_full", "outputs/features/fusion_1024d_full", "feature_db_meta.csv", "特征库元数据 (1024d)"),
    ("outputs/features/fusion_1024d_full", "outputs/features/fusion_1024d_full", "*.json", "评估指标 (1024d)"),
    # --- Checkpoints ---
    ("checkpoints/detection", "checkpoints/detection", "**/best.pt", "YOLO 检测权重"),
    ("checkpoints/segmentation", "checkpoints/segmentation", "best.pt", "U-Net 分割权重"),
    ("checkpoints/siamese", "checkpoints/siamese", "best.pt", "Triplet 编码器"),
    ("checkpoints/siamese", "checkpoints/siamese", "last.pt", "Triplet 编码器 (last)"),
    # --- New experiment checkpoints ---
    ("checkpoints/siamese", "checkpoints/siamese", "supcon/best.pt", "SupCon 编码器"),
    ("checkpoints/siamese", "checkpoints/siamese", "supcon/last.pt", "SupCon 编码器 (last)"),
    ("checkpoints/siamese", "checkpoints/siamese", "arcface_v2/best.pt", "ArcFace v2 编码器"),
    ("checkpoints/siamese", "checkpoints/siamese", "arcface/best.pt", "ArcFace v1 编码器"),
    ("checkpoints/siamese", "checkpoints/siamese", "pk_supcon/best.pt", "PK-SupCon 编码器"),
    ("checkpoints/siamese", "checkpoints/siamese", "pk_supcon_v2/best.pt", "PK-SupCon v2 编码器"),
    ("checkpoints/siamese", "checkpoints/siamese", "proxy_anchor/best.pt", "Proxy-Anchor 编码器"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync local files to Hugging Face dataset repo.")
    parser.add_argument("--execute", action="store_true", help="Actually upload. Default is dry-run.")
    parser.add_argument("--resume", action="store_true", help="Skip files that already exist on HF.")
    parser.add_argument("--repo", default=REPO_ID, help="HF repo ID.")
    return parser.parse_args()


def collect_files(local_dir: Path, pattern: str) -> list[Path]:
    """Return sorted list of files matching pattern under local_dir."""
    if "*" in pattern or "**" in pattern:
        files = sorted(local_dir.glob(pattern))
    else:
        path = local_dir / pattern
        files = [path] if path.exists() else []
    return [f for f in files if f.is_file()]


def main() -> int:
    args = parse_args()
    api = HfApi()

    try:
        existing_files = set(api.list_repo_files(args.repo, repo_type=REPO_TYPE))
    except Exception as exc:
        print(f"[error] 获取远程文件列表失败 (--resume 降级为全量上传): {exc}", flush=True)
        existing_files = set()

    total_files = 0
    total_size = 0
    to_upload: list[tuple[Path, str]] = []

    for local_rel, hf_rel, pattern, desc in SYNC_PLAN:
        local_dir = ROOT / local_rel
        if not local_dir.exists():
            print(f"[skip] {local_rel} — directory not found")
            continue

        files = collect_files(local_dir, pattern)
        if not files:
            print(f"[skip] {local_rel} — no files matching {pattern}")
            continue

        print(f"[{desc}] {local_rel} → {hf_rel} ({len(files)} files)")
        for file_path in files:
            rel = file_path.relative_to(local_dir)
            hf_path = f"{hf_rel}/{rel.as_posix()}"
            if args.resume and hf_path in existing_files:
                continue
            total_files += 1
            total_size += file_path.stat().st_size
            to_upload.append((file_path, hf_path))

    if not to_upload:
        print("\n所有文件已同步，无需上传。")
        return 0

    print(f"\n待上传: {total_files} 个文件, {total_size / 1024 / 1024:.1f} MB")
    if not args.execute:
        print("[DRY RUN] 使用 --execute 参数执行实际上传。")
        return 0

    create_repo(args.repo, repo_type=REPO_TYPE, exist_ok=True)
    for file_path, hf_path in tqdm(to_upload, desc="uploading"):
        upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=hf_path,
            repo_id=args.repo,
            repo_type=REPO_TYPE,
        )

    print(f"\n上传完成: {len(to_upload)} 个文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
