"""Build multi-label training metadata with full blood_id sets from relations.csv.

Each image gets its COMPLETE set of blood_ids, not just the single one assigned
in the original train_meta. This enables multi-positive contrastive training.

Output columns: img_id, blood_ids (pipe-separated), blood_name (for negative mining)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from _common import ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multi-label metadata from relations.csv.")
    parser.add_argument(
        "--relations",
        type=Path,
        default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv",
    )
    parser.add_argument(
        "--pigeon-csv",
        type=Path,
        default=ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv",
    )
    parser.add_argument(
        "--normalize-meta",
        type=Path,
        default=ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv",
    )
    parser.add_argument(
        "--iris-dir",
        type=Path,
        default=ROOT / "outputs" / "iris_normalized",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images-per-blood-name", type=int, default=3)
    parser.add_argument(
        "--train-output",
        type=Path,
        default=ROOT / "data" / "train_multi_meta.csv",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=ROOT / "data" / "val_multi_meta.csv",
    )
    parser.add_argument(
        "--blood-id-map-output",
        type=Path,
        default=ROOT / "data" / "blood_id_map.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Load success images from normalize_meta
    norm = pd.read_csv(args.normalize_meta, dtype={"img_id": str})
    success_ids = set(norm[norm["status"] == "success"]["img_id"].astype(str))
    print(f"归一化成功图像: {len(success_ids)}")

    # 2. Load relations: blood_id -> img_id (multi-label!)
    rel = pd.read_csv(
        args.relations,
        header=None,
        names=["blood_id", "img_id"],
        dtype={"blood_id": str, "img_id": str},
    )
    rel = rel.dropna(subset=["blood_id", "img_id"])
    rel["blood_id"] = rel["blood_id"].astype(str).str.strip()
    rel["img_id"] = rel["img_id"].astype(str).str.strip()
    rel = rel[(rel["blood_id"] != "") & (rel["img_id"] != "")]
    rel = rel.drop_duplicates(subset=["blood_id", "img_id"])

    # 3. Build img_id -> set[blood_id] mapping for success-only images
    img_to_bloods: dict[str, set[str]] = defaultdict(set)
    for row in tqdm(rel.itertuples(index=False), total=len(rel), desc="build blood_id sets"):
        img_id = str(row.img_id)
        if img_id in success_ids:
            img_to_bloods[img_id].add(str(row.blood_id))

    # Only keep images with at least one blood_id AND exist on disk
    valid_ids: list[str] = []
    for img_id in sorted(img_to_bloods):
        if (args.iris_dir / f"{img_id}.png").exists():
            valid_ids.append(img_id)
    print(f"有效图像 (有blood_id + 虹膜文件存在): {len(valid_ids)}")

    # 4. Load pigeon data for blood_name
    try:
        pigeon = pd.read_csv(args.pigeon_csv, dtype={"ID": str})
    except pd.errors.ParserError:
        pigeon = pd.read_csv(args.pigeon_csv, dtype={"ID": str}, engine="python", on_bad_lines="skip")
        print("warning: skipped malformed rows in pigeon.csv")
    pigeon["ID"] = pigeon["ID"].astype(str)
    pigeon["BLOOD"] = pigeon["BLOOD"].fillna("").astype(str).str.strip()
    pigeon = pigeon[pigeon["BLOOD"] != ""]
    img_to_blood_name = dict(zip(pigeon["ID"], pigeon["BLOOD"]))

    # 5. Build multi-label metadata rows
    rows: list[dict] = []
    blood_id_to_idx: dict[str, int] = {}
    for img_id in tqdm(valid_ids, desc="build rows"):
        blood_ids = sorted(img_to_bloods[img_id])
        blood_name = img_to_blood_name.get(img_id, "")

        # Assign indices to blood_ids we haven't seen
        for bid in blood_ids:
            if bid not in blood_id_to_idx:
                blood_id_to_idx[bid] = len(blood_id_to_idx)

        blood_id_indices = [blood_id_to_idx[bid] for bid in blood_ids]
        rows.append({
            "img_id": img_id,
            "blood_ids": "|".join(blood_ids),
            "blood_id_indices": json.dumps(blood_id_indices),
            "blood_name": blood_name,
            "num_blood_ids": len(blood_ids),
        })

    df = pd.DataFrame(rows)
    print(f"总图像: {len(df)}, 总 blood_id 数: {len(blood_id_to_idx)}")
    print(f"平均 blood_id/图: {df['num_blood_ids'].mean():.1f}")
    print(f"分布: min={df['num_blood_ids'].min()}, median={df['num_blood_ids'].median():.0f}, max={df['num_blood_ids'].max()}")

    # 6. Train/val split by blood_name (stratified)
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=args.seed).values
    train_indices: list[int] = []
    val_indices: list[int] = []

    # Group by blood_name for stratified split
    name_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, row in df.iterrows():
        name_to_indices[row["blood_name"]].append(i)

    rng2 = pd.Series(range(len(df))).sample(frac=1, random_state=args.seed + 1).values
    for name, indices in name_to_indices.items():
        indices_arr = pd.array(indices)
        rng2_subset = pd.Series(indices_arr).sample(frac=1, random_state=args.seed).values
        val_n = max(1, int(round(len(indices_arr) * args.val_ratio)))
        if len(indices_arr) - val_n < args.min_images_per_blood_name:
            val_n = max(0, len(indices_arr) - args.min_images_per_blood_name)
        val_indices.extend(rng2_subset[:val_n].tolist())
        train_indices.extend(rng2_subset[val_n:].tolist())

    train_df = df.iloc[train_indices].sort_values("img_id").reset_index(drop=True)
    val_df = df.iloc[val_indices].sort_values("img_id").reset_index(drop=True)

    # Remove val images from train
    val_img_set = set(val_df["img_id"])
    train_df = train_df[~train_df["img_id"].isin(val_img_set)]

    # 7. Save
    ensure_dir(args.train_output.parent)
    train_df.to_csv(args.train_output, index=False)
    val_df.to_csv(args.val_output, index=False)

    with open(args.blood_id_map_output, "w", encoding="utf-8") as f:
        json.dump(blood_id_to_idx, f, ensure_ascii=False)

    print(f"train: {len(train_df)} images, {train_df['blood_name'].nunique()} blood_names")
    print(f"val:   {len(val_df)} images, {val_df['blood_name'].nunique()} blood_names")
    val_leak = len(set(val_df["img_id"]) & set(train_df["img_id"]))
    print(f"val image leakage into train: {val_leak}")
    print(f"wrote: {args.train_output}, {args.val_output}, {args.blood_id_map_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
