#!/usr/bin/env python3
"""Set up local data from Hugging Face dataset clone.

Run this after cloning both repos:
  git clone git@github.com:ShouZhengCoder/PigeonVision.git
  cd PigeonVision
  git clone https://huggingface.co/datasets/jshouEX/pigeon-breed-image-dataset
  python scripts/setup_data.py

This creates symlinks so the project can find data/output/checkpoint files.
If you prefer to keep the HF clone elsewhere, set PIGEONVISION_DATA env var instead.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_DIR_DEFAULT = ROOT / "pigeon-breed-image-dataset"

LINKS = [
    # (link_path, target_under_hf)
    ("data/extracted", "data/extracted"),
    ("data/unet_labelme_80", "data/unet_labelme_80"),
    ("outputs/eye_crops", "outputs/eye_crops"),
    ("outputs/iris_normalized", "outputs/iris_normalized"),
    ("outputs/features", "outputs/features"),
    ("checkpoints", "checkpoints"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up data symlinks from HF clone.")
    parser.add_argument("--hf-dir", type=Path, default=HF_DIR_DEFAULT, help="Path to cloned HF dataset repo.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing symlinks/dirs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_dir = args.hf_dir.resolve()
    if not hf_dir.exists():
        print(f"HF 仓库未找到: {hf_dir}")
        print(f"请先克隆: git clone https://huggingface.co/datasets/jshouEX/pigeon-breed-image-dataset {hf_dir}")
        return 1

    for link_rel, target_rel in LINKS:
        link_path = (ROOT / link_rel).resolve()
        target_path = (hf_dir / target_rel).resolve()

        if not target_path.exists():
            print(f"[warn] HF 中缺少: {target_rel}，跳过")
            continue

        if link_path.exists() or link_path.is_symlink():
            if args.force:
                if link_path.is_symlink():
                    link_path.unlink()
                elif link_path.is_dir():
                    import shutil
                    shutil.rmtree(link_path)
            else:
                print(f"[skip] 已存在: {link_rel}")
                continue

        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(target_path, target_is_directory=target_path.is_dir())
        print(f"[link] {link_rel} → {target_path}")

    print("\n数据链接已就绪。可通过以下方式验证:")
    print(f"  ls {ROOT / 'data' / 'extracted'}")
    print(f"  ls {ROOT / 'checkpoints'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
