from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _common import ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image-level triplet metadata for Stage 4.")
    parser.add_argument("--relations", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    parser.add_argument("--normalize-meta", type=Path, default=ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv")
    parser.add_argument("--pigeon-csv", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv")
    parser.add_argument("--iris-dir", type=Path, default=ROOT / "outputs" / "iris_normalized")
    parser.add_argument("--train-output", type=Path, default=ROOT / "data" / "train_meta.csv")
    parser.add_argument("--val-output", type=Path, default=ROOT / "data" / "val_meta.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-blood-name-images", type=int, default=5)
    parser.add_argument("--min-blood-id-images", type=int, default=2)
    return parser.parse_args()


def read_pigeon_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype={"ID": str})
    except pd.errors.ParserError:
        df = pd.read_csv(path, dtype={"ID": str}, engine="python", on_bad_lines="skip")
        print(f"warning: skipped malformed rows while reading {path}")
    df["ID"] = df["ID"].astype(str)
    df["BLOOD"] = df["BLOOD"].fillna("").astype(str).str.strip()
    return df[["ID", "BLOOD"]].drop_duplicates(subset=["ID"])


def choose_one_blood_id_per_image(rows: pd.DataFrame) -> pd.DataFrame:
    # relations.csv is multi-label for many images. Triplet training needs one anchor label per image,
    # so choose the most represented eligible blood_id for each image, with blood_id as a stable tie-breaker.
    counts = rows.groupby("blood_id")["img_id"].nunique().rename("blood_id_count")
    rows = rows.merge(counts, on="blood_id", how="left")
    rows = rows.sort_values(["img_id", "blood_id_count", "blood_id"], ascending=[True, False, True])
    rows = rows.drop_duplicates(subset=["img_id"], keep="first")
    return rows.drop(columns=["blood_id_count"]).reset_index(drop=True)


def filter_counts(rows: pd.DataFrame, min_blood_name_images: int, min_blood_id_images: int) -> pd.DataFrame:
    rows = rows.copy()
    blood_name_counts = rows.groupby("blood_name")["img_id"].nunique()
    keep_names = set(blood_name_counts[blood_name_counts >= int(min_blood_name_images)].index)
    rows = rows[rows["blood_name"].isin(keep_names)].copy()

    blood_id_counts = rows.groupby("blood_id")["img_id"].nunique()
    keep_ids = set(blood_id_counts[blood_id_counts >= int(min_blood_id_images)].index)
    rows = rows[rows["blood_id"].isin(keep_ids)].copy()
    return rows.reset_index(drop=True)


def stratified_split_by_blood_id(rows: pd.DataFrame, val_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for _blood_id, group in rows.groupby("blood_id", sort=True):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * float(val_ratio))))
        if len(indices) - val_count < 1:
            val_count = len(indices) - 1
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        train_parts.append(rows.loc[train_idx])
        val_parts.append(rows.loc[val_idx])

    if not train_parts or not val_parts:
        raise RuntimeError("Could not build train/val triplet metadata; no splittable blood_id groups.")
    train_df = pd.concat(train_parts).sort_values(["blood_name", "blood_id", "img_id"]).reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values(["blood_name", "blood_id", "img_id"]).reset_index(drop=True)
    return train_df, val_df


def summarize(df: pd.DataFrame) -> str:
    return (
        f"rows={len(df)} images={df['img_id'].nunique()} "
        f"blood_ids={df['blood_id'].nunique()} blood_names={df['blood_name'].nunique()}"
    )


def main() -> int:
    args = parse_args()
    normalize_df = pd.read_csv(args.normalize_meta, dtype={"img_id": str})
    success_ids = set(normalize_df[normalize_df["status"] == "success"]["img_id"].astype(str))

    rel = pd.read_csv(args.relations, header=None, names=["blood_id", "img_id"], dtype={"blood_id": str, "img_id": str})
    rel["img_id"] = rel["img_id"].astype(str)
    rel["blood_id"] = rel["blood_id"].astype(str)
    rel = rel[rel["img_id"].isin(success_ids)].drop_duplicates(subset=["blood_id", "img_id"])

    pigeon = read_pigeon_csv(args.pigeon_csv)
    rows = rel.merge(pigeon, left_on="img_id", right_on="ID", how="inner")
    rows = rows.rename(columns={"BLOOD": "blood_name"})
    rows = rows[rows["blood_name"].fillna("").astype(str).str.strip() != ""].copy()
    rows["blood_name"] = rows["blood_name"].astype(str).str.strip()
    rows = rows[rows["img_id"].map(lambda img_id: (args.iris_dir / f"{img_id}.png").exists())]
    rows = rows[["img_id", "blood_id", "blood_name"]].drop_duplicates()

    before_rows = len(rows)
    rows = filter_counts(rows, args.min_blood_name_images, args.min_blood_id_images)
    rows = choose_one_blood_id_per_image(rows)
    rows = filter_counts(rows, args.min_blood_name_images, args.min_blood_id_images)
    train_df, val_df = stratified_split_by_blood_id(rows, args.val_ratio, args.seed)

    columns = ["img_id", "blood_id", "blood_name"]
    ensure_dir(args.train_output.parent)
    train_df[columns].to_csv(args.train_output, index=False)
    val_df[columns].to_csv(args.val_output, index=False)

    print(f"candidate relation rows: {before_rows}")
    print(f"filtered image rows: {len(rows)}")
    print(f"train: {summarize(train_df)}")
    print(f"val: {summarize(val_df)}")
    print(f"val image leakage into train: {len(set(train_df['img_id']) & set(val_df['img_id']))}")
    print(f"wrote: {args.train_output}, {args.val_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
