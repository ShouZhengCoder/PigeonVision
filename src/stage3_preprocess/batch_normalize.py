from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from iris_localize import UNetPredictor, daugman_normalize_color, extract_iris_region, localize_iris
from unet_common import DEFAULT_MASK_CONFIDENCE, IMAGE_SIZE, EllipseSpec, PredictionResult


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
PNG_FAST_WRITE = [cv2.IMWRITE_PNG_COMPRESSION, 1]


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
        "--export-iris-dir",
        type=Path,
        default=ROOT / "outputs" / "iris_extracted",
        help="Directory for original-size black-background extracted iris PNGs.",
    )
    parser.add_argument(
        "--no-export-iris",
        action="store_true",
        help="Disable extracted iris debug image export.",
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


def iris_ellipse_from_meta(row: pd.Series) -> EllipseSpec | None:
    required = ["iris_cx", "iris_cy", "iris_a", "iris_b", "iris_angle"]
    if any(key not in row.index for key in required):
        return None
    try:
        cx = float(row["iris_cx"])
        cy = float(row["iris_cy"])
        a = float(row["iris_a"])
        b = float(row["iris_b"])
        angle = float(row["iris_angle"])
    except (TypeError, ValueError):
        return None
    if a <= 0 or b <= 0:
        return None
    return EllipseSpec(cx=cx, cy=cy, a=a, b=b, angle_deg=angle, label="iris")


def export_existing_iris_images(
    meta_path: Path,
    crop_dir: Path,
    export_dir: Path,
    target_ids: set[str],
) -> tuple[int, int, int]:
    if not meta_path.exists():
        return 0, 0, 0
    meta = pd.read_csv(meta_path, dtype={"img_id": str})
    if "img_id" not in meta.columns or "status" not in meta.columns:
        return 0, 0, 0
    meta["img_id"] = meta["img_id"].astype(str)
    meta = meta[(meta["status"] == "success") & meta["img_id"].isin(target_ids)].copy()

    exported = 0
    skipped = 0
    failed = 0
    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="export_existing_iris"):
        row_s = pd.Series(row._asdict())
        img_id = str(row_s["img_id"])
        out_path = export_dir / f"{img_id}.png"
        if out_path.exists():
            skipped += 1
            continue

        iris = iris_ellipse_from_meta(row_s)
        if iris is None:
            failed += 1
            continue

        img_path = crop_dir / f"{img_id}.jpg"
        image = cv2.imread(str(img_path))
        if image is None:
            failed += 1
            continue

        input_size = int(row_s.get("input_size", IMAGE_SIZE))
        extracted = extract_iris_region(image, iris, input_size=input_size)
        if cv2.imwrite(str(out_path), extracted, PNG_FAST_WRITE):
            exported += 1
        else:
            failed += 1
    return exported, skipped, failed


def main() -> int:
    args = parse_args()
    crop_meta = resolve_root_path(args.crop_meta)
    crop_dir = resolve_root_path(args.crop_dir)
    output_dir = resolve_root_path(args.output_dir)
    export_iris_dir = None if args.no_export_iris else resolve_root_path(args.export_iris_dir)
    checkpoint = resolve_root_path(args.checkpoint)
    ensure_dir(output_dir)
    if export_iris_dir is not None:
        ensure_dir(export_iris_dir)

    if not crop_meta.exists():
        raise FileNotFoundError(f"Missing crop metadata: {crop_meta}")

    meta_path = output_dir / "normalize_meta.csv"
    if meta_path.exists() and not args.resume:
        backup_path = meta_path.with_suffix(".csv.bak")
        shutil.copy2(meta_path, backup_path)
        meta_path.unlink()
        print(f"backed up existing normalize_meta to: {backup_path}")
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
    print(f"checkpoint: {checkpoint}")
    if export_iris_dir is not None:
        print(f"export_iris_dir: {export_iris_dir}")
    print(f"total positive crops: {len(df)}")
    print(f"pending: {len(pending)}")

    if export_iris_dir is not None and args.resume:
        exported, skipped, export_failed = export_existing_iris_images(
            meta_path=meta_path,
            crop_dir=crop_dir,
            export_dir=export_iris_dir,
            target_ids=set(df["img_id"].astype(str)),
        )
        print(f"export_existing_iris_exported={exported}")
        print(f"export_existing_iris_skipped={skipped}")
        print(f"export_existing_iris_failed={export_failed}")

    if pending.empty:
        print("pending=0, no U-Net inference needed")
        return 0

    predictor = UNetPredictor(
        checkpoint_path=checkpoint,
        device=args.device,
        input_size=args.input_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        base_channels=args.base_channels,
        num_groups=args.num_groups,
    )
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
                normalized = daugman_normalize_color(image, prediction)
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
                if export_iris_dir is not None:
                    extracted = extract_iris_region(image, prediction)
                    cv2.imwrite(str(export_iris_dir / f"{img_id}.png"), extracted, PNG_FAST_WRITE)
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
