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
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_DIR_DEFAULT = ROOT / "pigeon-breed-image-dataset"

ARCHIVES = [
    ("data_extracted.tar", "原始鸽眼图 + U-Net 标注"),
    ("outputs.tar", "眼部裁剪 + 虹膜归一化 + 旧特征库"),
    ("checkpoints.tar", "旧模型权重"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up data from HF dataset repo.")
    parser.add_argument("--hf-dir", type=Path, default=HF_DIR_DEFAULT, help="Path to cloned HF dataset repo.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hf_dir = args.hf_dir.resolve()
    if not hf_dir.exists():
        print(f"HF 仓库未找到: {hf_dir}")
        print(f"请先克隆: git clone https://huggingface.co/datasets/jshouEX/pigeon-breed-image-dataset {hf_dir}")
        return 1

    # 1. Extract tar archives
    for archive_name, desc in ARCHIVES:
        archive_path = hf_dir / archive_name
        if not archive_path.exists():
            print(f"[warn] HF 中缺少: {archive_name}，跳过")
            continue
        file_size_gb = archive_path.stat().st_size / (1024 ** 3)
        print(f"[{desc}] 解压 {archive_name} ({file_size_gb:.1f} GB)...")
        with tarfile.open(archive_path) as tar:
            kw: dict[str, object] = {}
            if sys.version_info >= (3, 12):
                kw["filter"] = "data"
            tar.extractall(path=ROOT, **kw)  # type: ignore[arg-type]

    # 2. Merge individual files from HF clone (new checkpoints, fusion features, CSVs)
    # These were uploaded separately and aren't in the old tar archives
    copied = 0
    for item in hf_dir.iterdir():
        if item.name.endswith(".tar") or item.name.startswith("."):
            continue
        dest = ROOT / item.name
        if item.is_dir():
            for src_file in item.rglob("*"):
                if not src_file.is_file():
                    continue
                rel = src_file.relative_to(hf_dir)
                dst_file = ROOT / rel
                if not dst_file.exists():
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    copied += 1
        elif item.is_file() and not dest.exists():
            shutil.copy2(item, dest)
            copied += 1

    if copied:
        print(f"[merge] 从 HF 合并了 {copied} 个新文件")

    print("\n数据已就绪。验证:")
    print(f"  ls {ROOT / 'data' / 'extracted' / '1'}")
    print(f"  ls {ROOT / 'checkpoints' / 'siamese'}")
    print(f"  ls {ROOT / 'checkpoints' / 'siamese' / 'arcface_v2'}")
    print(f"  ls {ROOT / 'outputs' / 'features' / 'fusion_1024d_full'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
