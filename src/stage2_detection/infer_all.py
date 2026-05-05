from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from _common import ROOT, ensure_dir, resolve_root_path


FIELDNAMES = ["img_id", "x1", "y1", "x2", "y2", "confidence"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detector on all indexed images.")
    parser.add_argument(
        "--weights",
        type=Path,
        help="Path to best.pt. Defaults to newest best.pt under checkpoints/detection.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "checkpoints" / "detection",
        help="Directory searched for best.pt when --weights is omitted.",
    )
    parser.add_argument(
        "--img-index",
        type=Path,
        default=ROOT / "outputs" / "img_index.csv",
        help="outputs/img_index.csv path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "eye_crops",
        help="Directory for cropped eye images and crop_meta.csv.",
    )
    parser.add_argument("--conf", type=float, default=0.7, help="Detection confidence threshold.")
    parser.add_argument("--expand", type=float, default=0.1, help="BBox expansion ratio.")
    parser.add_argument("--resume", action="store_true", help="Skip image ids already in crop_meta.csv.")
    parser.add_argument("--limit", type=int, help="Optional max image count for smoke tests.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Ultralytics device value. Default lets ultralytics auto-select.",
    )
    return parser.parse_args()


def find_latest_best(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No best.pt found under {checkpoint_dir}")
    return candidates[0]


def load_existing(meta_path: Path) -> set[str]:
    if not meta_path.exists():
        return set()
    df = pd.read_csv(meta_path, dtype={"img_id": str})
    if "img_id" not in df.columns:
        return set()
    return set(df["img_id"].astype(str))


def choose_best_eye(result, conf_threshold: float) -> tuple[float, float, float, float, float] | None:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best: tuple[float, float, float, float, float] | None = None
    best_conf = -1.0
    for i in range(len(boxes)):
        conf = float(boxes.conf[i].item())
        if conf < conf_threshold or conf <= best_conf:
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
        best = (x1, y1, x2, y2, conf)
        best_conf = conf
    return best


def expand_bbox(
    box: tuple[float, float, float, float, float], image_width: int, image_height: int, ratio: float
) -> tuple[int, int, int, int, float]:
    x1, y1, x2, y2, conf = box
    width = x2 - x1
    height = y2 - y1
    pad_x = width * ratio
    pad_y = height * ratio
    ex1 = max(0, int(round(x1 - pad_x)))
    ey1 = max(0, int(round(y1 - pad_y)))
    ex2 = min(image_width, int(round(x2 + pad_x)))
    ey2 = min(image_height, int(round(y2 + pad_y)))
    return ex1, ey1, ex2, ey2, conf


def append_rows(meta_path: Path, rows: list[dict[str, object]], write_header: bool) -> None:
    with meta_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    weights = resolve_root_path(args.weights) if args.weights else find_latest_best(args.checkpoint_dir)
    img_index = resolve_root_path(args.img_index)
    output_dir = resolve_root_path(args.output_dir)
    if not img_index.exists():
        raise FileNotFoundError(f"Missing img_index: {img_index}")
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    ensure_dir(output_dir)
    meta_path = output_dir / "crop_meta.csv"
    if meta_path.exists() and not args.resume:
        meta_path.unlink()
    done_ids = load_existing(meta_path) if args.resume else set()

    df = pd.read_csv(img_index, dtype={"img_id": str})
    if args.limit:
        df = df.head(args.limit)

    pending = df[~df["img_id"].astype(str).isin(done_ids)].copy()
    model = YOLO(str(weights))
    print(f"weights: {weights}")
    print(f"total indexed images: {len(df)}")
    print(f"pending images: {len(pending)}")

    rows_buffer: list[dict[str, object]] = []
    meta_exists = meta_path.exists() and meta_path.stat().st_size > 0
    processed = 0
    detected = 0

    for row in tqdm(pending.itertuples(index=False), total=len(pending), desc="infer"):
        img_id = str(row.img_id)
        img_path = str(row.path)
        image = cv2.imread(img_path)
        if image is None:
            rows_buffer.append({"img_id": img_id, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "confidence": 0.0})
        else:
            predict_kwargs = {"source": image, "conf": args.conf, "verbose": False}
            if args.device:
                predict_kwargs["device"] = args.device
            results = model.predict(**predict_kwargs)
            best = choose_best_eye(results[0], args.conf) if results else None
            if best is None:
                rows_buffer.append({"img_id": img_id, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "confidence": 0.0})
            else:
                h, w = image.shape[:2]
                x1, y1, x2, y2, conf = expand_bbox(best, w, h, args.expand)
                if x2 > x1 and y2 > y1:
                    crop = image[y1:y2, x1:x2]
                    cv2.imwrite(str(output_dir / f"{img_id}.jpg"), crop)
                    detected += 1
                    rows_buffer.append(
                        {
                            "img_id": img_id,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": round(conf, 6),
                        }
                    )
                else:
                    rows_buffer.append(
                        {"img_id": img_id, "x1": 0, "y1": 0, "x2": 0, "y2": 0, "confidence": 0.0}
                    )

        processed += 1
        if len(rows_buffer) >= 100:
            append_rows(meta_path, rows_buffer, write_header=not meta_exists)
            meta_exists = True
            rows_buffer.clear()
        if processed % 1000 == 0:
            print(f"processed={processed}, detected={detected}, detection_rate={detected / processed:.4f}")

    if rows_buffer:
        append_rows(meta_path, rows_buffer, write_header=not meta_exists)

    print(f"processed: {processed}")
    print(f"detected: {detected}")
    print(f"detection_rate: {detected / processed:.4f}" if processed else "detection_rate: 0.0000")
    print(f"wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
