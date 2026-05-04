from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from _common import ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build img_id to absolute path index.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "extracted",
        help="Root directory containing numbered image folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "img_index.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rebuilding if output already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.output.exists():
        import pandas as pd

        df = pd.read_csv(args.output)
        print(f"img_index already exists: {args.output} ({len(df)} rows)")
        return 0

    ensure_dir(args.output.parent)
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    for idx in tqdm(range(1, 13), desc="scan dirs"):
        subdir = args.data_root / str(idx)
        if not subdir.exists():
            print(f"[warn] missing directory: {subdir}")
            continue
        for path in sorted(subdir.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".jpg", ".jpeg"}:
                continue
            img_id = path.stem
            if img_id in seen:
                continue
            rows.append((img_id, str(path.resolve())))
            seen.add(img_id)

    rows.sort(key=lambda item: item[0])
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["img_id", "path"])
        writer.writerows(rows)

    print(f"total images: {len(rows)}")
    print(f"wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
