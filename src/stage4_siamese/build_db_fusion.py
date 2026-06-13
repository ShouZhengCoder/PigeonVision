"""Build FAISS feature DB using multi-encoder fusion.

Modes:
- eval: train images are the gallery, val images are the query set.
- full: all successful normalized iris PNGs are the production gallery.

Triplet 256d + Relation-SupCon 256d + ArcFace 512d -> per-part L2 norm
-> concat -> global L2 norm -> 1024d.
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform, load_rgb_image
from model import IrisEncoder
from relation_metrics import (
    compute_cross_compare_metrics_by_blood_ids,
    compute_cross_compare_metrics_by_related_breeds,
    compute_cross_search_metrics_by_blood_ids,
    compute_cross_search_metrics_by_related_breeds,
    compute_cross_search_metrics_by_sets,
    load_blood_id_sets,
    load_related_blood_names,
)


DEFAULT_EVAL_DIR = ROOT / "outputs" / "features" / "fusion_1024d_eval"
DEFAULT_FULL_DIR = ROOT / "outputs" / "features" / "fusion_1024d_full"
IRIS_DIR = ROOT / "outputs" / "iris_normalized"
FUSION_ENCODERS = (
    ("triplet", ROOT / "checkpoints" / "siamese" / "best.pt", 256, "resnet34"),
    ("relation_supcon", ROOT / "checkpoints" / "siamese" / "relation_supcon" / "best.pt", 256, "resnet34"),
    ("arcface", ROOT / "checkpoints" / "siamese" / "arcface_v2" / "best.pt", 512, "resnet50"),
)
FUSION_DESCRIPTION = "triplet_256d + relation_supcon_256d + arcface_512d"
SEARCH_PRINT_KEYS = (
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "precision_at_1",
    "precision_at_5",
    "precision_at_10",
    "avg_relevant_at_1",
    "avg_relevant_at_5",
    "avg_relevant_at_10",
    "ndcg_at_1",
    "ndcg_at_5",
    "ndcg_at_10",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mAP",
)


class IrisDbDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, img_dir: Path, transform) -> None:
        self.rows = rows.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        img_id = str(row["img_id"])
        image = load_rgb_image(self.img_dir / f"{img_id}.png")
        return img_id, self.transform(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1024d fusion feature DB.")
    parser.add_argument(
        "--mode",
        choices=("eval", "full"),
        default="eval",
        help="eval: train gallery + val queries; full: all successful normalized PNGs for Flask production search.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to fusion_1024d_eval or fusion_1024d_full by mode.")
    parser.add_argument("--train-meta", type=Path, default=ROOT / "data" / "train_meta.csv")
    parser.add_argument("--val-meta", type=Path, default=ROOT / "data" / "val_meta.csv")
    parser.add_argument("--normalize-meta", type=Path, default=IRIS_DIR / "normalize_meta.csv")
    parser.add_argument("--pigeon-csv", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv")
    parser.add_argument("--relations", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    parser.add_argument("--threshold-source", type=Path, default=None, help="Threshold JSON to copy in --mode full.")
    parser.add_argument("--limit", type=int, default=None, help="Use first N gallery/query rows for smoke tests.")
    parser.add_argument("--compare-max-pairs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def torch_load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError:
        # Project checkpoints are produced by our own training jobs and may
        # include config values such as pathlib.PosixPath, which PyTorch 2.6
        # rejects under the new weights_only=True default.
        return torch.load(path, map_location=device, weights_only=False)


def load_encoder(checkpoint_path: Path, feat_dim: int, backbone: str, device: torch.device) -> IrisEncoder:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    encoder = IrisEncoder(feat_dim=feat_dim, backbone=backbone, pretrained=False, in_channels=3).to(device)
    state = torch_load_checkpoint(checkpoint_path, device)
    model_state = state.get("model_state", state) if isinstance(state, dict) else state
    encoder.load_state_dict(model_state)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract_features(
    encoder: IrisEncoder,
    rows: pd.DataFrame,
    img_dir: Path,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> np.ndarray:
    dataset = IrisDbDataset(rows, img_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features: list[np.ndarray] = []
    for _img_ids, images in tqdm(loader, desc=desc):
        feats = encoder(images.to(device, non_blocking=True)).detach().cpu().numpy().astype("float32")
        features.append(feats)
    if not features:
        return np.empty((0, encoder.feat_dim), dtype="float32")
    return np.concatenate(features, axis=0).astype("float32")


def l2_normalize(features: np.ndarray) -> np.ndarray:
    return (features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)).astype("float32")


def extract_fusion_features(
    rows: pd.DataFrame,
    encoders: list[tuple[str, IrisEncoder, int]],
    img_dir: Path,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    role: str,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for name, encoder, dim in encoders:
        print(f"\nExtracting {role} {name} ({dim}d)...")
        features = extract_features(
            encoder,
            rows,
            img_dir,
            transform,
            device,
            batch_size,
            num_workers,
            f"{role} {name}",
        )
        parts.append(l2_normalize(features))
    return l2_normalize(np.concatenate(parts, axis=1).astype("float32"))


def load_meta_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"img_id": str, "blood_id": str, "blood_name": str})
    required = {"img_id", "blood_id", "blood_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["img_id", "blood_id", "blood_name"]].copy()
    out["img_id"] = out["img_id"].fillna("").astype(str).str.strip()
    out["blood_id"] = out["blood_id"].fillna("").astype(str).str.strip()
    out["blood_name"] = out["blood_name"].fillna("").astype(str).str.strip()
    out = out[(out["img_id"] != "") & (out["blood_name"] != "")]
    return out.drop_duplicates(subset=["img_id"], keep="first").reset_index(drop=True)


def load_success_img_ids(normalize_meta: Path, img_dir: Path) -> tuple[list[str], int, int]:
    if not normalize_meta.exists():
        raise FileNotFoundError(f"normalize_meta not found: {normalize_meta}")
    normalize_df = pd.read_csv(normalize_meta, dtype={"img_id": str, "status": str})
    required = {"img_id", "status"}
    missing = required - set(normalize_df.columns)
    if missing:
        raise ValueError(f"{normalize_meta} missing columns: {sorted(missing)}")
    success = normalize_df[normalize_df["status"].astype(str) == "success"].copy()
    success["img_id"] = success["img_id"].fillna("").astype(str).str.strip()
    success = success[success["img_id"] != ""].drop_duplicates(subset=["img_id"], keep="first")
    success_ids = success["img_id"].tolist()
    existing_ids = [img_id for img_id in success_ids if (img_dir / f"{img_id}.png").exists()]
    return existing_ids, len(success_ids), len(existing_ids)


def filter_available_rows(rows: pd.DataFrame, normalize_meta: Path, img_dir: Path, name: str) -> pd.DataFrame:
    success_ids, success_count, _png_count = load_success_img_ids(normalize_meta, img_dir)
    success_set = set(success_ids)

    before = len(rows)
    out = rows[rows["img_id"].astype(str).isin(success_set)].copy()
    after_status_and_png = len(out)
    print(
        f"{name}: kept {len(out)}/{before} rows "
        f"(normalize_success={success_count}, status_success_and_png={after_status_and_png})"
    )
    return out.reset_index(drop=True)


def read_pigeon_metadata(pigeon_csv: Path) -> pd.DataFrame:
    try:
        pigeon_df = pd.read_csv(pigeon_csv, dtype=str)
    except pd.errors.ParserError:
        pigeon_df = pd.read_csv(pigeon_csv, dtype=str, engine="python", on_bad_lines="skip")
        print(f"warning: skipped malformed rows while reading {pigeon_csv}")
    required = {"ID", "PG_ID", "BLOOD"}
    missing = required - set(pigeon_df.columns)
    if missing:
        raise ValueError(f"{pigeon_csv} missing columns: {sorted(missing)}")
    cols = [col for col in ("ID", "PG_ID", "BLOOD", "EYE", "COLOR", "SEX") if col in pigeon_df.columns]
    pigeon_df = pigeon_df[cols].copy().fillna("")
    pigeon_df["ID"] = pigeon_df["ID"].astype(str).str.strip()
    pigeon_df = pigeon_df[pigeon_df["ID"] != ""].drop_duplicates(subset=["ID"], keep="first")
    rename = {"ID": "img_id", "PG_ID": "pg_id", "BLOOD": "blood_name"}
    return pigeon_df.rename(columns=rename).reset_index(drop=True)


def load_pg_id_map(pigeon_csv: Path) -> dict[str, str]:
    pigeon_df = read_pigeon_metadata(pigeon_csv)
    return dict(zip(pigeon_df["img_id"].astype(str), pigeon_df["pg_id"].fillna("").astype(str)))


def read_relation_summary(relations: Path) -> pd.DataFrame:
    rel = pd.read_csv(relations, header=None, names=["blood_id", "img_id"], dtype=str)
    rel = rel.dropna(subset=["blood_id", "img_id"]).copy()
    rel["blood_id"] = rel["blood_id"].astype(str).str.strip()
    rel["img_id"] = rel["img_id"].astype(str).str.strip()
    rel = rel[(rel["blood_id"] != "") & (rel["img_id"] != "")]
    rel = rel.drop_duplicates(subset=["blood_id", "img_id"])
    summary = rel.groupby("img_id", sort=False)["blood_id"].agg(lambda values: "|".join(sorted(set(values))))
    return summary.reset_index()


def build_full_rows(normalize_meta: Path, img_dir: Path, pigeon_csv: Path, relations: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    success_ids, success_count, png_count = load_success_img_ids(normalize_meta, img_dir)
    rows = pd.DataFrame({"img_id": success_ids})
    relation_summary = read_relation_summary(relations)
    pigeon_meta = read_pigeon_metadata(pigeon_csv)

    rows = rows.merge(relation_summary, on="img_id", how="left")
    rows = rows.merge(pigeon_meta, on="img_id", how="left")
    rows["blood_id"] = rows["blood_id"].fillna("").astype(str)
    rows["pg_id"] = rows["pg_id"].fillna("").astype(str)
    rows["blood_name"] = rows["blood_name"].fillna("").astype(str)
    rows["blood"] = rows["blood_name"]

    stats = {
        "normalize_success": int(success_count),
        "png_exists": int(png_count),
        "rows": int(len(rows)),
        "with_relations": int((rows["blood_id"] != "").sum()),
        "with_pigeon_meta": int((rows["pg_id"] != "").sum()),
        "with_blood_name": int((rows["blood_name"] != "").sum()),
    }
    return rows.reset_index(drop=True), stats


def metric_payload(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in metrics.items()}


def feature_meta_from_rows(rows: pd.DataFrame, pigeon_csv: Path) -> pd.DataFrame:
    feature_meta = rows.copy()
    if "pg_id" not in feature_meta.columns:
        pg_id_map = load_pg_id_map(pigeon_csv)
        feature_meta["pg_id"] = feature_meta["img_id"].astype(str).map(pg_id_map).fillna("")
    if "blood" not in feature_meta.columns:
        feature_meta["blood"] = feature_meta.get("blood_name", "").fillna("").astype(str)
    for column in ("blood_id", "blood_name"):
        if column not in feature_meta.columns:
            feature_meta[column] = ""
    return feature_meta[["img_id", "pg_id", "blood", "blood_id", "blood_name"]].fillna("")


def save_feature_db(features: np.ndarray, rows: pd.DataFrame, output_dir: Path, pigeon_csv: Path) -> int:
    import faiss

    if len(rows) != len(features):
        raise ValueError(f"row/feature mismatch: rows={len(rows)}, features={len(features)}")
    np.save(output_dir / "feature_db.npy", features.astype("float32"))
    feature_meta = feature_meta_from_rows(rows, pigeon_csv)
    feature_meta.to_csv(output_dir / "feature_db_meta.csv", index=False)

    index = faiss.IndexFlatL2(int(features.shape[1]))
    index.add(features.astype("float32"))
    faiss.write_index(index, str(output_dir / "faiss_index.bin"))
    if int(index.ntotal) != len(feature_meta):
        raise RuntimeError(f"FAISS/meta mismatch after write: index={index.ntotal}, meta={len(feature_meta)}")
    return int(index.ntotal)


def blood_sets_for_img_ids(img_ids: list[str], blood_id_sets: dict[str, frozenset[str]]) -> list[frozenset[str]]:
    return [blood_id_sets.get(str(img_id), frozenset()) for img_id in img_ids]


def compute_pg_id_centroids(
    features: np.ndarray,
    rows: pd.DataFrame,
    blood_id_sets: dict[str, frozenset[str]],
) -> tuple[np.ndarray, list[str], list[frozenset[str]]]:
    if "pg_id" not in rows.columns:
        raise ValueError("rows missing pg_id column")
    buckets: dict[str, list[int]] = {}
    for index, row in enumerate(rows.itertuples(index=False)):
        pg_id = str(getattr(row, "pg_id", "")).strip()
        if pg_id:
            buckets.setdefault(pg_id, []).append(index)

    centroid_features: list[np.ndarray] = []
    centroid_pg_ids: list[str] = []
    centroid_sets: list[frozenset[str]] = []
    img_ids = rows["img_id"].astype(str).tolist()
    for pg_id in sorted(buckets):
        indices = buckets[pg_id]
        centroid = np.mean(features[indices], axis=0, dtype=np.float32)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        union: set[str] = set()
        for idx in indices:
            union.update(blood_id_sets.get(img_ids[idx], frozenset()))
        if not union:
            continue
        centroid_pg_ids.append(pg_id)
        centroid_features.append(centroid.astype("float32"))
        centroid_sets.append(frozenset(union))

    if not centroid_features:
        return np.empty((0, features.shape[1]), dtype="float32"), [], []
    return np.stack(centroid_features).astype("float32"), centroid_pg_ids, centroid_sets


def copy_threshold_for_full(output_dir: Path, threshold_source: Path | None) -> Path:
    candidates: list[Path] = []
    if threshold_source is not None:
        candidates.append(resolve_root_path(threshold_source))
    else:
        candidates.append(DEFAULT_EVAL_DIR / "threshold.json")
    for candidate in candidates:
        if candidate.exists():
            if candidate.resolve() != (output_dir / "threshold.json").resolve():
                shutil.copyfile(candidate, output_dir / "threshold.json")
            return candidate
    raise FileNotFoundError(
        "No threshold.json found. Run --mode eval first or pass --threshold-source before building full production DB."
    )


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_eval_mode(args: argparse.Namespace, encoders: list[tuple[str, IrisEncoder, int]], transform, device: torch.device) -> int:
    iris_dir = IRIS_DIR
    train_meta = resolve_root_path(args.train_meta)
    val_meta = resolve_root_path(args.val_meta)
    normalize_meta = resolve_root_path(args.normalize_meta)
    output_dir = ensure_dir(resolve_root_path(args.output_dir or DEFAULT_EVAL_DIR))
    pigeon_csv = resolve_root_path(args.pigeon_csv)
    relations = resolve_root_path(args.relations)

    train_rows = filter_available_rows(load_meta_rows(train_meta), normalize_meta, iris_dir, "train_meta")
    val_rows = filter_available_rows(load_meta_rows(val_meta), normalize_meta, iris_dir, "val_meta")
    val_ids = set(val_rows["img_id"].astype(str))
    gallery_rows = train_rows[~train_rows["img_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    query_rows = val_rows.reset_index(drop=True)

    pg_id_map = load_pg_id_map(pigeon_csv)
    gallery_rows["pg_id"] = gallery_rows["img_id"].astype(str).map(pg_id_map).fillna("")
    query_rows["pg_id"] = query_rows["img_id"].astype(str).map(pg_id_map).fillna("")

    if args.limit is not None:
        gallery_rows = gallery_rows.head(args.limit).copy()
        query_rows = query_rows.head(args.limit).copy()
    if gallery_rows.empty or query_rows.empty:
        raise RuntimeError("No eligible gallery/query images for fusion database evaluation.")
    print(f"Mode: eval")
    print(f"Gallery(train only): {len(gallery_rows)}, Query(val only): {len(query_rows)}")

    gallery_features = extract_fusion_features(
        gallery_rows,
        encoders,
        iris_dir,
        transform,
        device,
        args.batch_size,
        args.num_workers,
        "gallery",
    )
    query_features = extract_fusion_features(
        query_rows,
        encoders,
        iris_dir,
        transform,
        device,
        args.batch_size,
        args.num_workers,
        "query",
    )
    total_dim = int(gallery_features.shape[1])
    ntotal = save_feature_db(gallery_features, gallery_rows, output_dir, pigeon_csv)

    gallery_img_ids = gallery_rows["img_id"].astype(str).tolist()
    query_img_ids = query_rows["img_id"].astype(str).tolist()
    gallery_blood_names = gallery_rows["blood_name"].astype(str).tolist()
    query_blood_names = query_rows["blood_name"].astype(str).tolist()
    related_blood_names = load_related_blood_names(relations, pigeon_csv)
    blood_id_sets = load_blood_id_sets(relations)

    print("\nEvaluating image-level retrieval/compare...")
    search_metrics_new = compute_cross_search_metrics_by_blood_ids(
        query_features,
        query_img_ids,
        gallery_features,
        gallery_img_ids,
        blood_id_sets,
    )
    compare_metrics_new = compute_cross_compare_metrics_by_blood_ids(
        query_features,
        query_img_ids,
        gallery_features,
        gallery_img_ids,
        blood_id_sets,
        max_pairs=int(args.compare_max_pairs),
        seed=int(args.seed),
    )
    search_metrics_old = compute_cross_search_metrics_by_related_breeds(
        query_features,
        query_img_ids,
        gallery_features,
        gallery_blood_names,
        related_blood_names,
    )
    compare_metrics_old = compute_cross_compare_metrics_by_related_breeds(
        query_features,
        query_img_ids,
        query_blood_names,
        gallery_features,
        gallery_img_ids,
        gallery_blood_names,
        related_blood_names,
        max_pairs=int(args.compare_max_pairs),
        seed=int(args.seed),
    )

    pg_centroid_features, pg_ids, pg_sets = compute_pg_id_centroids(gallery_features, gallery_rows, blood_id_sets)
    query_sets = blood_sets_for_img_ids(query_img_ids, blood_id_sets)
    pg_id_metrics = compute_cross_search_metrics_by_sets(query_features, query_sets, pg_centroid_features, pg_sets)

    payload = {
        "mode": "eval",
        "gallery_role": "train",
        "query_role": "val",
        "fusion": FUSION_DESCRIPTION,
        "total_dim": total_dim,
        "gallery_size": ntotal,
        "query_size": int(len(query_rows)),
        "pg_id_centroid_gallery_size": int(len(pg_ids)),
        "search": metric_payload(search_metrics_new),
        "pg_id_centroid_search": metric_payload(pg_id_metrics),
        "compare": metric_payload(compare_metrics_new),
    }
    write_json(output_dir / "eval_metrics.json", payload)
    write_json(
        output_dir / "eval_comparison.json",
        {
            "mode": "eval",
            "single_label": {
                "search": metric_payload(search_metrics_old),
                "compare": metric_payload(compare_metrics_old),
            },
            "multi_label": {
                "search": metric_payload(search_metrics_new),
                "pg_id_centroid_search": metric_payload(pg_id_metrics),
                "compare": metric_payload(compare_metrics_new),
            },
        },
    )
    if "threshold" in compare_metrics_new:
        write_json(output_dir / "threshold.json", {"threshold": float(compare_metrics_new["threshold"]), "source": "eval_multi_label"})

    print_eval_summary(search_metrics_new, compare_metrics_new, pg_id_metrics, output_dir, total_dim)
    return 0


def run_full_mode(args: argparse.Namespace, encoders: list[tuple[str, IrisEncoder, int]], transform, device: torch.device) -> int:
    iris_dir = IRIS_DIR
    normalize_meta = resolve_root_path(args.normalize_meta)
    output_dir = ensure_dir(resolve_root_path(args.output_dir or DEFAULT_FULL_DIR))
    pigeon_csv = resolve_root_path(args.pigeon_csv)
    relations = resolve_root_path(args.relations)

    rows, stats = build_full_rows(normalize_meta, iris_dir, pigeon_csv, relations)
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
        stats["limited_rows"] = int(len(rows))
    if rows.empty:
        raise RuntimeError("No successful normalized PNGs for full production database.")
    print("Mode: full")
    print(
        "Full production rows: "
        f"{len(rows)} (normalize_success={stats['normalize_success']}, png_exists={stats['png_exists']}, "
        f"with_relations={stats['with_relations']}, with_pigeon_meta={stats['with_pigeon_meta']})"
    )

    features = extract_fusion_features(
        rows,
        encoders,
        iris_dir,
        transform,
        device,
        args.batch_size,
        args.num_workers,
        "full",
    )
    total_dim = int(features.shape[1])
    ntotal = save_feature_db(features, rows, output_dir, pigeon_csv)
    threshold_from = copy_threshold_for_full(output_dir, args.threshold_source)
    write_json(
        output_dir / "build_manifest.json",
        {
            "mode": "full",
            "fusion": FUSION_DESCRIPTION,
            "encoders": [
                {
                    "name": name,
                    "checkpoint": str(path),
                    "feat_dim": int(dim),
                    "backbone": backbone,
                }
                for name, path, dim, backbone in FUSION_ENCODERS
            ],
            "total_dim": total_dim,
            "gallery_size": ntotal,
            "metadata_stats": stats,
            "threshold_source": str(threshold_from),
        },
    )
    print(f"Saved full production DB: gallery_size={ntotal}, dim={total_dim}")
    print(f"Threshold copied from: {threshold_from}")
    print(f"Saved to {output_dir}/")
    return 0


def print_eval_summary(
    search_metrics: dict[str, float],
    compare_metrics: dict[str, float],
    pg_id_metrics: dict[str, float],
    output_dir: Path,
    total_dim: int,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Concat {total_dim}d Fusion eval (multi-label blood_id)")
    print(f"{'=' * 60}")
    for key in SEARCH_PRINT_KEYS:
        if key in search_metrics:
            print(f"  search_{key:<22}: {search_metrics[key]:.4f}")
    for key in ("accuracy", "balanced_accuracy", "auc", "eer"):
        print(f"  compare_{key:<21}: {compare_metrics[key]:.4f}")
    print("\nPG_ID centroid gallery metrics:")
    for key in SEARCH_PRINT_KEYS:
        if key in pg_id_metrics:
            print(f"  pg_id_{key:<23}: {pg_id_metrics[key]:.4f}")
    print(f"\nSaved to {output_dir}/")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print("Loading encoders...")
    encoders = [
        (name, load_encoder(path, dim, backbone, device), dim)
        for name, path, dim, backbone in FUSION_ENCODERS
    ]

    transform = default_transform(input_shape=(64, 512), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), train=False)
    if args.mode == "eval":
        return run_eval_mode(args, encoders, transform, device)
    if args.mode == "full":
        return run_full_mode(args, encoders, transform, device)
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
