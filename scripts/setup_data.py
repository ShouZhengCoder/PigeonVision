#!/usr/bin/env python3
"""Set up local data from Hugging Face dataset clone.

Run this after cloning both repos:
  git clone git@github.com:ShouZhengCoder/PigeonVision.git
  cd PigeonVision
  git clone https://huggingface.co/datasets/jshouEX/pigeon-breed-image-dataset
  python scripts/setup_data.py

The HF repo contains tar archives that get extracted into the project.
"""

from __future__ import annotations

import argparse
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_DIR_DEFAULT = ROOT / "pigeon-breed-image-dataset"

ARCHIVES = [
    ("data_extracted.tar", "原始鸽眼图 + U-Net 标注"),
    ("outputs.tar", "眼部裁剪 + 虹膜归一化 + 特征库"),
    ("checkpoints.tar", "模型权重"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up data from HF tar archives.")
    parser.add_argument("--hf-dir", type=Path, default=HF_DIR_DEFAULT, help="Path to cloned HF dataset repo.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_dir = args.hf_dir.resolve()
    if not hf_dir.exists():
        print(f"HF 仓库未找到: {hf_dir}")
        print(f"请先克隆: git clone https://huggingface.co/datasets/jshouEX/pigeon-breed-image-dataset {hf_dir}")
        return 1

    for archive_name, desc in ARCHIVES:
        archive_path = hf_dir / archive_name
        if not archive_path.exists():
            print(f"[warn] HF 中缺少: {archive_name}，跳过")
            continue

        file_size_gb = archive_path.stat().st_size / (1024 ** 3)
        print(f"[{desc}] 解压 {archive_name} ({file_size_gb:.1f} GB)...")
        with tarfile.open(archive_path) as tar:
            tar.extractall(path=ROOT, filter="data")
        print(f"  完成")

    print("\n数据已就绪。验证:")
    print(f"  ls {ROOT / 'data' / 'extracted' / '1'}")
    print(f"  ls {ROOT / 'checkpoints' / 'siamese'}")
    print(f"  ls {ROOT / 'outputs' / 'iris_normalized'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
