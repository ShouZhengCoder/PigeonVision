from __future__ import annotations

import argparse
import itertools
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
    parser = argparse.ArgumentParser(description="Build pair CSVs for siamese training.")
    parser.add_argument(
        "--relations",
        type=Path,
        default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv",
        help="relations.csv path.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=ROOT / "data" / "extracted" / "datasetXGN" / "anotations",
        help="Annotation directory for valid image ids.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=ROOT / "data" / "pairs_train.csv",
        help="Train pairs CSV path.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=ROOT / "data" / "pairs_val.csv",
        help="Val pairs CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing outputs when present.",
    )
    return parser.parse_args()


def normalize_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def main() -> int:
    args = parse_args()
    if args.resume and args.train_output.exists() and args.val_output.exists():
        print(f"pair outputs already exist: {args.train_output}, {args.val_output}")
        return 0

    valid_imgs = {p.stem for p in args.annotation_dir.glob("*.json")}
    rel = pd.read_csv(args.relations, header=None, names=["blood_id", "img_id"])
    rel["img_id"] = rel["img_id"].astype(str)
    rel = rel[rel["img_id"].isin(valid_imgs)].drop_duplicates(subset=["blood_id", "img_id"])

    blood_to_imgs: dict[str, list[str]] = {}
    img_to_bloods: dict[str, set[str]] = {}

    for blood_id, group in rel.groupby("blood_id"):
        imgs = sorted(set(group["img_id"].astype(str)))
        if len(imgs) < 2:
            continue
        blood_to_imgs[str(blood_id)] = imgs
        for img_id in imgs:
            img_to_bloods.setdefault(img_id, set()).add(str(blood_id))

    pos_pairs_set: set[tuple[str, str]] = set()
    for imgs in tqdm(blood_to_imgs.values(), desc="build positives"):
        for a, b in itertools.combinations(imgs, 2):
            pos_pairs_set.add(normalize_pair(a, b))

    pos_pairs = sorted(pos_pairs_set)
    target_neg = len(pos_pairs) * 2

    img_ids = sorted(img_to_bloods.keys())
    rng = random.Random(args.seed)
    neg_pairs_set: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max(target_neg * 20, 1000)

    while len(neg_pairs_set) < target_neg and attempts < max_attempts:
        a, b = rng.sample(img_ids, 2)
        attempts += 1
        pair = normalize_pair(a, b)
        if pair in pos_pairs_set or pair in neg_pairs_set:
            continue
        if img_to_bloods[a].isdisjoint(img_to_bloods[b]):
            neg_pairs_set.add(pair)

    if len(neg_pairs_set) < target_neg:
        raise RuntimeError(
            f"Could only sample {len(neg_pairs_set)} negative pairs out of required {target_neg}"
        )

    neg_pairs = sorted(neg_pairs_set)

    combined = [(a, b, 1) for a, b in pos_pairs] + [(a, b, 0) for a, b in neg_pairs]
    rng.shuffle(combined)
    split_idx = int(len(combined) * args.train_ratio)
    train_rows = combined[:split_idx]
    val_rows = combined[split_idx:]

    train_df = pd.DataFrame(train_rows, columns=["img_id_a", "img_id_b", "label"])
    val_df = pd.DataFrame(val_rows, columns=["img_id_a", "img_id_b", "label"])

    ensure_dir(args.train_output.parent)
    train_df.to_csv(args.train_output, index=False)
    val_df.to_csv(args.val_output, index=False)

    print(f"positive pairs: {len(pos_pairs)}")
    print(f"negative pairs: {len(neg_pairs)}")
    print(f"train rows: {len(train_df)}")
    print(f"val rows: {len(val_df)}")
    print(f"wrote: {args.train_output}, {args.val_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
