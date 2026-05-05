from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from iris_localize import localize_iris, normalize_iris


FIELDNAMES = ["img_id", "status", "cx", "cy", "r_inner", "r_outer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch normalize iris crops to 64x512.")
    parser.add_argument(
        "--crop-meta",
        type=Path,
        default=ROOT / "outputs" / "eye_crops" / "crop_meta.csv",
        help="Crop metadata CSV.",
    )
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=ROOT / "outputs" / "eye_crops",
        help="Eye crop directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "iris_normalized",
        help="Normalized iris output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already processed images found in normalize_meta.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of confidence>0 rows to process.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=100,
        help="Write metadata every N processed rows.",
    )
    return parser.parse_args()


def load_existing(meta_path: Path) -> set[str]:
    if not meta_path.exists():
        return set()
    df = pd.read_csv(meta_path, dtype={"img_id": str})
    if "img_id" not in df.columns:
        return set()
    return set(df["img_id"].astype(str))


def append_rows(meta_path: Path, rows: list[dict[str, object]], write_header: bool) -> None:
    with meta_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    crop_meta = resolve_root_path(args.crop_meta)
    crop_dir = resolve_root_path(args.crop_dir)
    output_dir = resolve_root_path(args.output_dir)
    ensure_dir(output_dir)

    if not crop_meta.exists():
        raise FileNotFoundError(f"Missing crop metadata: {crop_meta}")

    meta_path = output_dir / "normalize_meta.csv"
    if meta_path.exists() and not args.resume:
        meta_path.unlink()
    existing_ids = load_existing(meta_path) if args.resume else set()

    df = pd.read_csv(crop_meta, dtype={"img_id": str})
    if "confidence" not in df.columns:
        raise ValueError(f"Missing confidence column in {crop_meta}")
    df = df[df["confidence"].astype(float) > 0].copy()
    if args.limit is not None:
        df = df.head(args.limit)
    df["img_id"] = df["img_id"].astype(str)
    pending = df[~df["img_id"].isin(existing_ids)].copy()

    print(f"crop_meta: {crop_meta}")
    print(f"total positive crops: {len(df)}")
    print(f"pending: {len(pending)}")

    rows_buffer: list[dict[str, object]] = []
    meta_exists = meta_path.exists() and meta_path.stat().st_size > 0
    processed = 0
    success = 0

    for row in tqdm(pending.itertuples(index=False), total=len(pending), desc="normalize"):
        img_id = str(row.img_id)
        img_path = crop_dir / f"{img_id}.jpg"
        status = "failed"
        cx = cy = r_inner = r_outer = -1
        if img_path.exists():
            image = cv2.imread(str(img_path))
            if image is not None:
                localized = localize_iris(image)
                if localized is not None:
                    cx, cy, r_inner, r_outer = localized
                    try:
                        normalized = normalize_iris(image, cx, cy, r_inner, r_outer)
                    except Exception:
                        localized = None
                    else:
                        cv2.imwrite(str(output_dir / f"{img_id}.png"), normalized)
                        status = "success"
                        success += 1
        rows_buffer.append(
            {
                "img_id": img_id,
                "status": status,
                "cx": cx,
                "cy": cy,
                "r_inner": r_inner,
                "r_outer": r_outer,
            }
        )

        processed += 1
        if len(rows_buffer) >= args.flush_every:
            append_rows(meta_path, rows_buffer, write_header=not meta_exists)
            meta_exists = True
            rows_buffer.clear()

        if processed % 1000 == 0:
            rate = success / processed if processed else 0.0
            print(f"processed={processed}, success={success}, success_rate={rate:.4f}")

    if rows_buffer:
        append_rows(meta_path, rows_buffer, write_header=not meta_exists)

    rate = success / processed if processed else 0.0
    print(f"processed={processed}")
    print(f"success={success}")
    print(f"success_rate={rate:.4f}")
    print(f"wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

