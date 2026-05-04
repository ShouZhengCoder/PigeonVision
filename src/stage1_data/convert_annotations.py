from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from _common import ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert annotation JSONs to YOLO labels.")
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=ROOT / "data" / "extracted" / "datasetXGN" / "anotations",
        help="Directory containing annotation JSON files.",
    )
    parser.add_argument(
        "--img-index",
        type=Path,
        default=ROOT / "outputs" / "img_index.csv",
        help="img_index.csv path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "yolo_dataset",
        help="YOLO dataset root.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip label files that already exist.",
    )
    return parser.parse_args()


def load_img_index(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["img_id"].astype(str), df["path"].astype(str)))


def to_yolo_bbox(bbx: list[float], width: float, height: float) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = map(float, bbx)
    if width <= 0 or height <= 0:
        return None
    x1 = max(0.0, min(x1, width))
    x2 = max(0.0, min(x2, width))
    y1 = max(0.0, min(y1, height))
    y2 = max(0.0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def write_label(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for cx, cy, w, h in boxes:
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main() -> int:
    args = parse_args()
    if not args.img_index.exists():
        raise FileNotFoundError(f"Missing img_index: {args.img_index}")

    img_index = load_img_index(args.img_index)
    ann_paths = sorted(args.annotation_dir.glob("*.json"))

    ensure_dir(args.output_dir / "labels" / "train")
    ensure_dir(args.output_dir / "labels" / "val")

    rng = random.Random(args.seed)

    samples: list[dict[str, object]] = []
    skipped_missing = 0
    skipped_invalid_boxes = 0

    for ann_path in tqdm(ann_paths, desc="read annotations"):
        img_id = ann_path.stem
        img_path = img_index.get(img_id)
        if img_path is None:
            skipped_missing += 1
            continue
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        boxes: list[tuple[float, float, float, float]] = []
        width = float(data.get("weidth", data.get("width", 0)))
        height = float(data.get("height", 0))
        for bb in data.get("bbs", []):
            if bb.get("label") != "eye":
                continue
            yolo = to_yolo_bbox(bb.get("bbx", []), width, height)
            if yolo is None:
                skipped_invalid_boxes += 1
                continue
            boxes.append(yolo)
        samples.append({"img_id": img_id, "img_path": img_path, "boxes": boxes})

    img_ids = [sample["img_id"] for sample in samples]
    rng.shuffle(img_ids)
    split_idx = int(len(img_ids) * args.train_ratio)
    train_ids = set(img_ids[:split_idx])

    train_paths: list[str] = []
    val_paths: list[str] = []
    train_count = 0
    val_count = 0
    skipped_existing = 0

    for sample in tqdm(samples, desc="write labels"):
        img_id = sample["img_id"]  # type: ignore[assignment]
        img_path = str(sample["img_path"])  # type: ignore[assignment]
        boxes = sample["boxes"]  # type: ignore[assignment]
        is_train = img_id in train_ids
        label_dir = args.output_dir / "labels" / ("train" if is_train else "val")
        label_path = label_dir / f"{img_id}.txt"
        if args.resume and label_path.exists():
            skipped_existing += 1
        else:
            write_label(label_path, boxes)
        if is_train:
            train_paths.append(img_path)
            train_count += 1
        else:
            val_paths.append(img_path)
            val_count += 1

    train_txt = args.output_dir / "train.txt"
    val_txt = args.output_dir / "val.txt"
    train_txt.write_text("\n".join(train_paths) + ("\n" if train_paths else ""), encoding="utf-8")
    val_txt.write_text("\n".join(val_paths) + ("\n" if val_paths else ""), encoding="utf-8")

    data_yaml = args.output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {args.output_dir.resolve()}",
                f"train: {train_txt.resolve()}",
                f"val: {val_txt.resolve()}",
                "nc: 1",
                "names: ['eye']",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"processed images: {len(samples)}")
    print(f"skipped missing images: {skipped_missing}")
    print(f"invalid eye boxes skipped: {skipped_invalid_boxes}")
    print(f"train images: {train_count}")
    print(f"val images: {val_count}")
    if args.resume:
        print(f"existing label files skipped: {skipped_existing}")
    print(f"wrote: {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
