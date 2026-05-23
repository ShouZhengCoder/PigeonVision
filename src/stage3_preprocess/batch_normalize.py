from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from iris_localize import UNetPredictor, localize_iris, normalize_iris
from unet_common import DEFAULT_MASK_CONFIDENCE, IMAGE_SIZE, PredictionResult


FIELDNAMES = [
    "img_id",
    "status",
    "reason",
    "mask_confidence",
    "source_width",
    "source_height",
    "input_size",
    "cx",
    "cy",
    "r_inner",
    "r_outer",
    "pupil_cx",
    "pupil_cy",
    "pupil_a",
    "pupil_b",
    "pupil_angle",
    "iris_cx",
    "iris_cy",
    "iris_a",
    "iris_b",
    "iris_angle",
]


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
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "segmentation" / "best.pt",
        help="Trained U-Net checkpoint.",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, defaults to auto.")
    parser.add_argument("--input-size", type=int, default=IMAGE_SIZE, help="U-Net input size.")
    parser.add_argument("--in-channels", type=int, default=1, help="U-Net input channels.")
    parser.add_argument("--num-classes", type=int, default=3, help="U-Net output classes.")
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net base channel width.")
    parser.add_argument("--num-groups", type=int, default=8, help="GroupNorm groups.")
    parser.add_argument(
        "--mask-confidence-threshold",
        type=float,
        default=DEFAULT_MASK_CONFIDENCE,
        help="Minimum mean softmax confidence for success.",
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


def make_failed_row(
    img_id: str,
    reason: str,
    source_width: int = -1,
    source_height: int = -1,
    input_size: int = IMAGE_SIZE,
    mask_confidence: float = 0.0,
) -> dict[str, object]:
    row = {
        "img_id": img_id,
        "status": "failed",
        "reason": reason,
        "mask_confidence": round(float(mask_confidence), 6),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "input_size": int(input_size),
        "cx": -1,
        "cy": -1,
        "r_inner": -1,
        "r_outer": -1,
        "pupil_cx": -1,
        "pupil_cy": -1,
        "pupil_a": -1,
        "pupil_b": -1,
        "pupil_angle": -1,
        "iris_cx": -1,
        "iris_cy": -1,
        "iris_a": -1,
        "iris_b": -1,
        "iris_angle": -1,
    }
    return row


def main() -> int:
    args = parse_args()
    crop_meta = resolve_root_path(args.crop_meta)
    crop_dir = resolve_root_path(args.crop_dir)
    output_dir = resolve_root_path(args.output_dir)
    checkpoint = resolve_root_path(args.checkpoint)
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

    predictor = UNetPredictor(
        checkpoint_path=checkpoint,
        device=args.device,
        input_size=args.input_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        num_groups=args.num_groups,
    )

    print(f"crop_meta: {crop_meta}")
    print(f"checkpoint: {checkpoint}")
    print(f"total positive crops: {len(df)}")
    print(f"pending: {len(pending)}")
    print(f"device: {predictor.device}")

    rows_buffer: list[dict[str, object]] = []
    meta_exists = meta_path.exists() and meta_path.stat().st_size > 0
    processed = 0
    success = 0
    confidences: list[float] = []

    for row in tqdm(pending.itertuples(index=False), total=len(pending), desc="normalize"):
        img_id = str(row.img_id)
        img_path = crop_dir / f"{img_id}.jpg"
        if not img_path.exists():
            rows_buffer.append(make_failed_row(img_id, "missing crop image"))
            confidences.append(0.0)
            processed += 1
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            rows_buffer.append(make_failed_row(img_id, "failed to read crop image"))
            confidences.append(0.0)
            processed += 1
            continue

        prediction: PredictionResult = localize_iris(
            image,
            predictor=predictor,
            mask_confidence_threshold=args.mask_confidence_threshold,
        )

        if prediction.success:
            try:
                normalized = normalize_iris(image, prediction)
            except Exception as exc:
                prediction = PredictionResult(
                    success=False,
                    status="failed",
                    reason=f"normalize failed: {exc}",
                    mask_confidence=prediction.mask_confidence,
                    source_width=prediction.source_width,
                    source_height=prediction.source_height,
                    input_size=prediction.input_size,
                    cx=-1,
                    cy=-1,
                    r_inner=-1,
                    r_outer=-1,
                    pupil=prediction.pupil,
                    iris=prediction.iris,
                    mask=prediction.mask,
                )
            else:
                cv2.imwrite(str(output_dir / f"{img_id}.png"), normalized)
                success += 1

        confidences.append(float(prediction.mask_confidence))
        rows_buffer.append(prediction.to_meta_row(img_id))

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

    failure = processed - success
    rate = success / processed if processed else 0.0
    failure_rate = failure / processed if processed else 0.0
    conf_arr = np.asarray(confidences, dtype=np.float32) if confidences else np.asarray([], dtype=np.float32)
    conf_mean = float(conf_arr.mean()) if conf_arr.size else 0.0
    conf_p25 = float(np.percentile(conf_arr, 25)) if conf_arr.size else 0.0
    conf_p75 = float(np.percentile(conf_arr, 75)) if conf_arr.size else 0.0
    below_threshold_ratio = float((conf_arr < args.mask_confidence_threshold).mean()) if conf_arr.size else 0.0

    print(f"processed={processed}")
    print(f"success={success}")
    print(f"failure={failure}")
    print(f"success_rate={rate:.4f}")
    print(f"mask_confidence_mean={conf_mean:.6f}")
    print(f"mask_confidence_p25={conf_p25:.6f}")
    print(f"mask_confidence_p75={conf_p75:.6f}")
    print(f"mask_confidence_below_{args.mask_confidence_threshold:.2f}_ratio={below_threshold_ratio:.4f}")
    print(f"wrote: {meta_path}")
    if args.limit is not None and failure_rate > 0.1:
        print("failure_rate_above_10_percent, stopping before full run", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
