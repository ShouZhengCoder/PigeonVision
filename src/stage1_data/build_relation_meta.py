"""Build unified relation metadata for multi-label bloodline retrieval."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import ROOT, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data/relation_meta.csv from normalized PNGs and relations.csv.")
    parser.add_argument("--relations", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    parser.add_argument("--pigeon-csv", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv")
    parser.add_argument("--normalize-meta", type=Path, default=ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv")
    parser.add_argument("--iris-dir", type=Path, default=ROOT / "outputs" / "iris_normalized")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "relation_meta.csv")
    parser.add_argument("--blood-id-map-output", type=Path, default=ROOT / "data" / "relation_blood_id_map.json")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print stats without writing outputs.")
    parser.add_argument("--resume", action="store_true", help="Skip rebuild when output already exists.")
    return parser.parse_args()


def load_success_ids(normalize_meta: Path, iris_dir: Path) -> tuple[list[str], dict[str, int]]:
    if not normalize_meta.exists():
        raise FileNotFoundError(f"normalize_meta not found: {normalize_meta}")
    df = pd.read_csv(normalize_meta, dtype={"img_id": str, "status": str})
    required = {"img_id", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{normalize_meta} missing columns: {sorted(missing)}")
    success = df[df["status"].astype(str) == "success"].copy()
    success["img_id"] = success["img_id"].fillna("").astype(str).str.strip()
    success = success[success["img_id"] != ""].drop_duplicates(subset=["img_id"], keep="first")
    success_ids = success["img_id"].tolist()
    existing = [img_id for img_id in success_ids if (iris_dir / f"{img_id}.png").exists()]
    return existing, {"normalize_success": int(len(success_ids)), "png_exists": int(len(existing))}


def load_relations(relations: Path, valid_ids: set[str]) -> dict[str, set[str]]:
    rel = pd.read_csv(relations, header=None, names=["blood_id", "img_id"], dtype=str)
    rel = rel.dropna(subset=["blood_id", "img_id"]).copy()
    rel["blood_id"] = rel["blood_id"].astype(str).str.strip()
    rel["img_id"] = rel["img_id"].astype(str).str.strip()
    rel = rel[(rel["blood_id"] != "") & (rel["img_id"] != "")]
    rel = rel[rel["img_id"].isin(valid_ids)].drop_duplicates(subset=["blood_id", "img_id"])

    img_to_bloods: dict[str, set[str]] = defaultdict(set)
    for row in tqdm(rel.itertuples(index=False), total=len(rel), desc="relations"):
        img_to_bloods[str(row.img_id)].add(str(row.blood_id))
    return img_to_bloods


def read_pigeon_metadata(pigeon_csv: Path) -> pd.DataFrame:
    try:
        pigeon = pd.read_csv(pigeon_csv, dtype=str)
    except pd.errors.ParserError:
        pigeon = pd.read_csv(pigeon_csv, dtype=str, engine="python", on_bad_lines="skip")
        print(f"warning: skipped malformed rows while reading {pigeon_csv}")
    required = {"ID", "PG_ID", "BLOOD"}
    missing = required - set(pigeon.columns)
    if missing:
        raise ValueError(f"{pigeon_csv} missing columns: {sorted(missing)}")
    cols = [col for col in ("ID", "PG_ID", "BLOOD") if col in pigeon.columns]
    pigeon = pigeon[cols].fillna("").copy()
    pigeon["ID"] = pigeon["ID"].astype(str).str.strip()
    pigeon["PG_ID"] = pigeon["PG_ID"].astype(str).str.strip()
    pigeon["BLOOD"] = pigeon["BLOOD"].astype(str).str.strip()
    pigeon = pigeon[pigeon["ID"] != ""].drop_duplicates(subset=["ID"], keep="first")
    return pigeon.rename(columns={"ID": "img_id", "PG_ID": "pg_id", "BLOOD": "blood_name"})


def compute_idf(img_to_bloods: dict[str, set[str]]) -> dict[str, float]:
    n = len(img_to_bloods)
    df: dict[str, int] = defaultdict(int)
    for blood_ids in img_to_bloods.values():
        for blood_id in blood_ids:
            df[str(blood_id)] += 1
    return {blood_id: float(np.log((n + 1.0) / (count + 1.0))) for blood_id, count in df.items()}


def split_groups(rows: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> pd.Series:
    total = float(train_ratio + val_ratio + test_ratio)
    if total <= 0:
        raise ValueError("split ratios must be positive")
    train_ratio = float(train_ratio) / total
    val_ratio = float(val_ratio) / total
    test_ratio = float(test_ratio) / total

    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, row in rows.iterrows():
        pg_id = str(row.get("pg_id", "")).strip()
        group_key = f"pg:{pg_id}" if pg_id else f"img:{row['img_id']}"
        group_to_indices[group_key].append(int(idx))

    rng = np.random.default_rng(seed)
    groups = list(group_to_indices.items())
    rng.shuffle(groups)
    total_rows = len(rows)
    targets = {"train": train_ratio * total_rows, "val": val_ratio * total_rows, "test": test_ratio * total_rows}
    counts = {"train": 0, "val": 0, "test": 0}
    split = pd.Series([""] * len(rows), index=rows.index, dtype=object)

    for _group_key, indices in groups:
        # Greedy assignment by relative fill. This preserves PG_ID isolation while
        # keeping row counts close to the requested ratios.
        underfill = {name: targets[name] - counts[name] for name in targets}
        name = max(underfill, key=underfill.get)
        if underfill[name] <= 0:
            name = min(counts, key=counts.get)
        split.iloc[indices] = name
        counts[name] += len(indices)
    return split


def leakage_report(rows: pd.DataFrame) -> dict[str, int]:
    with_pg = rows[rows["pg_id"].fillna("").astype(str).str.strip() != ""]
    leaks = 0
    for _pg_id, group in with_pg.groupby("pg_id", sort=False):
        if group["split"].nunique() > 1:
            leaks += 1
    return {"pg_id_leak_count": int(leaks), "pg_id_groups": int(with_pg["pg_id"].nunique())}


def related_gallery_stats(rows: pd.DataFrame) -> dict[str, dict[str, int]]:
    split_sets = {
        split: [set(str(value).split("|")) for value in group["blood_ids"]]
        for split, group in rows.groupby("split", sort=False)
    }
    train_gallery: list[set[str]] = split_sets.get("train", [])
    stats: dict[str, dict[str, int]] = {}
    for split in ("val", "test"):
        total = 0
        with_train_related = 0
        no_train_related = 0
        for row in rows[rows["split"] == split].itertuples(index=False):
            total += 1
            blood_ids = set(str(row.blood_ids).split("|"))
            has_related = any(bool(blood_ids & gallery_ids) for gallery_ids in train_gallery)
            with_train_related += int(has_related)
            no_train_related += int(not has_related)
        stats[split] = {
            "queries": int(total),
            "with_train_gallery_related": int(with_train_related),
            "no_train_gallery_related": int(no_train_related),
        }
    return stats


def main() -> int:
    args = parse_args()
    if args.resume and args.output.exists():
        print(f"--resume: output exists, skipping rebuild: {args.output}")
        return 0

    success_ids, success_stats = load_success_ids(args.normalize_meta, args.iris_dir)
    valid_id_set = set(success_ids)
    img_to_bloods = load_relations(args.relations, valid_id_set)
    img_to_bloods = {
        img_id: blood_ids
        for img_id, blood_ids in img_to_bloods.items()
        if blood_ids and (args.iris_dir / f"{img_id}.png").exists()
    }
    if not img_to_bloods:
        raise RuntimeError("No normalized images with relation labels found")

    idf = compute_idf(img_to_bloods)
    blood_id_to_idx = {blood_id: idx for idx, blood_id in enumerate(sorted(idf))}
    rows: list[dict[str, object]] = []
    for img_id in tqdm(sorted(img_to_bloods), desc="rows"):
        blood_ids = sorted(img_to_bloods[img_id])
        rows.append(
            {
                "img_id": img_id,
                "blood_ids": "|".join(blood_ids),
                "blood_id_indices": json.dumps([blood_id_to_idx[b] for b in blood_ids], separators=(",", ":")),
                "num_blood_ids": int(len(blood_ids)),
                "sum_blood_idf": float(sum(idf.get(b, 0.0) for b in blood_ids)),
            }
        )
    rows_df = pd.DataFrame(rows)
    pigeon = read_pigeon_metadata(args.pigeon_csv)
    rows_df = rows_df.merge(pigeon, on="img_id", how="left")
    rows_df["pg_id"] = rows_df["pg_id"].fillna("").astype(str).str.strip()
    rows_df["blood_name"] = rows_df["blood_name"].fillna("").astype(str).str.strip()
    rows_df["has_pg_id"] = (rows_df["pg_id"] != "").astype(int)
    rows_df["has_blood_name"] = (rows_df["blood_name"] != "").astype(int)
    rows_df["split"] = split_groups(rows_df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    rows_df = rows_df[
        [
            "img_id",
            "blood_ids",
            "blood_id_indices",
            "blood_name",
            "pg_id",
            "num_blood_ids",
            "sum_blood_idf",
            "has_pg_id",
            "has_blood_name",
            "split",
        ]
    ].sort_values(["split", "img_id"]).reset_index(drop=True)

    leak = leakage_report(rows_df)
    split_counts = rows_df["split"].value_counts().to_dict()
    gallery_stats = related_gallery_stats(rows_df)
    summary = {
        **success_stats,
        "rows": int(len(rows_df)),
        "unique_blood_ids": int(len(blood_id_to_idx)),
        "avg_blood_ids_per_image": float(rows_df["num_blood_ids"].mean()),
        "with_pg_id": int(rows_df["has_pg_id"].sum()),
        "with_blood_name": int(rows_df["has_blood_name"].sum()),
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "leakage": leak,
        "gallery_related": gallery_stats,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if leak["pg_id_leak_count"] != 0:
        raise RuntimeError(f"PG_ID split leakage detected: {leak['pg_id_leak_count']}")
    if args.dry_run:
        print("dry-run: not writing outputs")
        return 0

    ensure_dir(args.output.parent)
    rows_df.to_csv(args.output, index=False)
    with args.blood_id_map_output.open("w", encoding="utf-8") as f:
        json.dump(blood_id_to_idx, f, ensure_ascii=False, indent=2)
    print(f"wrote: {args.output}")
    print(f"wrote: {args.blood_id_map_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
