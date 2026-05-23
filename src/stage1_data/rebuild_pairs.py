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
    parser = argparse.ArgumentParser(description="Rebuild pair CSVs from normalized iris set.")
    parser.add_argument(
        "--relations",
        type=Path,
        default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv",
        help="relations.csv path.",
    )
    parser.add_argument(
        "--normalize-meta",
        type=Path,
        default=ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv",
        help="normalize_meta.csv path.",
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
    return parser.parse_args()


def normalize_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def resolve_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("None of the candidate paths exist: " + ", ".join(str(p) for p in candidates))


def build_outputs(rows: list[tuple[str, str, int]], train_ratio: float, seed: int):
    rng = random.Random(seed)
    train_rows: list[tuple[str, str, int]] = []
    val_rows: list[tuple[str, str, int]] = []
    for row in rows:
        if rng.random() < train_ratio:
            train_rows.append(row)
        else:
            val_rows.append(row)
    return train_rows, val_rows


def summarize_labels(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "label" not in df.columns:
        return {"positive": 0, "negative": 0, "total": 0}
    labels = df["label"].astype(int)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    return {"positive": pos, "negative": neg, "total": int(len(df))}


def load_existing_summary(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"positive": 0, "negative": 0, "total": 0}
    return summarize_labels(pd.read_csv(path))


def main() -> int:
    args = parse_args()
    relations = resolve_existing(
        args.relations,
        ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv",
    )
    normalize_meta = resolve_existing(
        args.normalize_meta,
        ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv",
    )

    success_ids = set(
        pd.read_csv(normalize_meta, dtype={"img_id": str})
        .query("status == 'success'")["img_id"]
        .astype(str)
    )

    rel = pd.read_csv(relations, header=None, names=["blood_id", "img_id"])
    rel["img_id"] = rel["img_id"].astype(str)
    rel = rel[rel["img_id"].isin(success_ids)].drop_duplicates(subset=["blood_id", "img_id"])

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

    rows = [(a, b, 1) for a, b in pos_pairs] + [(a, b, 0) for a, b in neg_pairs]
    rng.shuffle(rows)
    train_rows, val_rows = build_outputs(rows, args.train_ratio, args.seed)

    train_df = pd.DataFrame(train_rows, columns=["img_id_a", "img_id_b", "label"])
    val_df = pd.DataFrame(val_rows, columns=["img_id_a", "img_id_b", "label"])

    old_train_summary = load_existing_summary(args.train_output)
    old_val_summary = load_existing_summary(args.val_output)
    old_summary = {
        "positive": old_train_summary["positive"] + old_val_summary["positive"],
        "negative": old_train_summary["negative"] + old_val_summary["negative"],
        "total": old_train_summary["total"] + old_val_summary["total"],
    }

    ensure_dir(args.train_output.parent)
    train_df.to_csv(args.train_output, index=False)
    val_df.to_csv(args.val_output, index=False)

    new_summary = summarize_labels(pd.concat([train_df, val_df], ignore_index=True))
    delta_positive = new_summary["positive"] - old_summary["positive"]
    delta_negative = new_summary["negative"] - old_summary["negative"]
    delta_total = new_summary["total"] - old_summary["total"]

    print(f"positive pairs: {len(pos_pairs)}")
    print(f"negative pairs: {len(neg_pairs)}")
    print(f"train rows: {len(train_df)}")
    print(f"val rows: {len(val_df)}")
    print(
        "old_summary: "
        f"positive={old_summary['positive']} negative={old_summary['negative']} total={old_summary['total']}"
    )
    print(
        "new_summary: "
        f"positive={new_summary['positive']} negative={new_summary['negative']} total={new_summary['total']}"
    )
    print(
        "delta_summary: "
        f"positive={delta_positive:+d} negative={delta_negative:+d} total={delta_total:+d}"
    )
    print(f"wrote: {args.train_output}, {args.val_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
