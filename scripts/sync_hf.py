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
# Each entry: (local_dir, hf_target_dir, glob_pattern, description)
SYNC_PLAN = [
    # --- data/ ---
    ("data/extracted", "data/extracted", "*/*.jpg", "原始鸽眼图"),
    ("data/unet_labelme_80", "data/unet_labelme_80", "**/*", "U-Net 标注样本"),
    # --- outputs/ ---
    ("outputs/eye_crops", "outputs/eye_crops", "*.jpg", "YOLO 眼部裁剪"),
    ("outputs/iris_normalized", "outputs/iris_normalized", "*.png", "归一化虹膜图"),
    ("outputs/features", "outputs/features", "*.npy", "特征向量"),
    ("outputs/features", "outputs/features", "*.bin", "FAISS 索引"),
    # --- checkpoints/ ---
    ("checkpoints/detection", "checkpoints/detection", "**/best.pt", "YOLO 检测权重"),
    ("checkpoints/segmentation", "checkpoints/segmentation", "best.pt", "U-Net 分割权重"),
    ("checkpoints/siamese", "checkpoints/siamese", "best.pt", "孪生网络权重"),
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
    except Exception:
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

    ensure_repo_exists = True
    for file_path, hf_path in tqdm(to_upload, desc="uploading"):
        if ensure_repo_exists:
            try:
                create_repo(args.repo, repo_type=REPO_TYPE, exist_ok=True)
            except Exception:
                pass
            ensure_repo_exists = False
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
