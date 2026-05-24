from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from _common import ROOT, ensure_dir, resolve_root_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize normalized iris samples.")
    parser.add_argument(
        "--meta",
        type=Path,
        default=ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv",
        help="Normalization metadata CSV.",
    )
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=ROOT / "outputs" / "eye_crops",
        help="Eye crop directory.",
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=ROOT / "outputs" / "iris_normalized",
        help="Normalized iris directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "iris_normalized" / "samples_vis.png",
        help="Output visualization path.",
    )
    parser.add_argument("--num", type=int, default=20, help="Number of samples to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_image(path: Path, grayscale: bool = False) -> np.ndarray:
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(path)
    return image


def resize_keep_aspect(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(target_width / width, target_height / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_height, target_width, 3), 245, dtype=np.uint8)
    y0 = (target_height - new_h) // 2
    x0 = (target_width - new_w) // 2
    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def make_panel(crop: np.ndarray, norm: np.ndarray, img_id: str) -> np.ndarray:
    panel_h = 150
    left_w = 260
    right_w = 512
    gap = 12
    left = resize_keep_aspect(crop, panel_h - 28, left_w)
    right = resize_keep_aspect(norm, panel_h - 28, right_w)
    panel = np.full((panel_h, left_w + right_w + gap, 3), 255, dtype=np.uint8)
    panel[18 : 18 + left.shape[0], : left.shape[1]] = left
    panel[18 : 18 + right.shape[0], left_w + gap : left_w + gap + right.shape[1]] = right
    cv2.putText(panel, img_id, (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return panel


def main() -> int:
    args = parse_args()
    meta = resolve_root_path(args.meta)
    crop_dir = resolve_root_path(args.crop_dir)
    normalized_dir = resolve_root_path(args.normalized_dir)
    output = resolve_root_path(args.output)
    ensure_dir(output.parent)

    if not meta.exists():
        raise FileNotFoundError(f"Missing normalize metadata: {meta}")

    df = pd.read_csv(meta, dtype={"img_id": str})
    if "status" not in df.columns:
        raise ValueError(f"Missing status column in {meta}")
    df = df[df["status"] == "success"].copy()
    if df.empty:
        raise RuntimeError("No successful normalized samples found")

    sample_n = min(args.num, len(df))
    sampled = df.sample(n=sample_n, random_state=args.seed).reset_index(drop=True)

    panels: list[np.ndarray] = []
    for row in sampled.itertuples(index=False):
        img_id = str(row.img_id)
        crop = load_image(crop_dir / f"{img_id}.jpg", grayscale=False)
        norm = load_image(normalized_dir / f"{img_id}.png", grayscale=False)
        panels.append(make_panel(crop, norm, img_id))

    cols = 4 if len(panels) >= 4 else len(panels)
    rows = int(np.ceil(len(panels) / cols))
    tile_h, tile_w = panels[0].shape[:2]
    grid = np.full((rows * tile_h, cols * tile_w, 3), 248, dtype=np.uint8)

    for idx, panel in enumerate(panels):
        r = idx // cols
        c = idx % cols
        y = r * tile_h
        x = c * tile_w
        grid[y : y + tile_h, x : x + tile_w] = panel

    cv2.imwrite(str(output), grid)
    print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
