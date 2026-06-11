"""Build FAISS feature DB using multi-encoder fusion.

Triplet 256d + SupCon 256d + ArcFace 512d -> L2-norm -> concat -> 1024d.
"""
from __future__ import annotations

import argparse
import json
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
    load_blood_id_sets,
    load_related_blood_names,
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
    parser = argparse.ArgumentParser(description="Build 1024d fusion feature DB and evaluate val queries.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "features" / "fusion_1024d_full_candidate",
    )
    parser.add_argument("--train-meta", type=Path, default=ROOT / "data" / "train_meta.csv")
    parser.add_argument("--val-meta", type=Path, default=ROOT / "data" / "val_meta.csv")
    parser.add_argument("--normalize-meta", type=Path, default=ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv")
    parser.add_argument("--pigeon-csv", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv")
    parser.add_argument("--relations", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    parser.add_argument("--limit", type=int, default=None, help="Use first N gallery/query rows for smoke tests.")
    return parser.parse_args()


def torch_load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_encoder(checkpoint_path: Path, feat_dim: int, backbone: str, device: torch.device) -> IrisEncoder:
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


def filter_available_rows(rows: pd.DataFrame, normalize_meta: Path, img_dir: Path, name: str) -> pd.DataFrame:
    if not normalize_meta.exists():
        raise FileNotFoundError(f"normalize_meta not found: {normalize_meta}")
    normalize_df = pd.read_csv(normalize_meta, dtype={"img_id": str, "status": str})
    required = {"img_id", "status"}
    missing = required - set(normalize_df.columns)
    if missing:
        raise ValueError(f"{normalize_meta} missing columns: {sorted(missing)}")
    success_ids = set(normalize_df[normalize_df["status"] == "success"]["img_id"].astype(str))

    before = len(rows)
    out = rows[rows["img_id"].astype(str).isin(success_ids)].copy()
    after_status = len(out)
    out = out[out["img_id"].astype(str).map(lambda img_id: (img_dir / f"{img_id}.png").exists())].copy()
    print(
        f"{name}: kept {len(out)}/{before} rows "
        f"(status_success={after_status}, png_exists={len(out)})"
    )
    return out.reset_index(drop=True)


def load_pg_id_map(pigeon_csv: Path) -> dict[str, str]:
    try:
        pigeon_df = pd.read_csv(pigeon_csv, dtype={"ID": str})
    except pd.errors.ParserError:
        pigeon_df = pd.read_csv(pigeon_csv, dtype={"ID": str}, engine="python", on_bad_lines="skip")
        print(f"warning: skipped malformed rows while reading {pigeon_csv}")
    required = {"ID", "PG_ID"}
    missing = required - set(pigeon_df.columns)
    if missing:
        raise ValueError(f"{pigeon_csv} missing columns: {sorted(missing)}")
    pigeon_df["ID"] = pigeon_df["ID"].astype(str)
    pigeon_df["PG_ID"] = pigeon_df["PG_ID"].fillna("").astype(str)
    pigeon_df = pigeon_df.drop_duplicates(subset=["ID"], keep="first")
    return dict(zip(pigeon_df["ID"], pigeon_df["PG_ID"]))


def metric_payload(metrics: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 6) for key, value in metrics.items()}


def main() -> int:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    iris_dir = ROOT / "outputs" / "iris_normalized"
    train_meta = resolve_root_path(args.train_meta)
    val_meta = resolve_root_path(args.val_meta)
    normalize_meta = resolve_root_path(args.normalize_meta)
    output_dir = ensure_dir(resolve_root_path(args.output_dir))
    pigeon_csv = resolve_root_path(args.pigeon_csv)
    relations = resolve_root_path(args.relations)

    print("Loading encoders...")
    encoders = [
        ("triplet", load_encoder(ROOT / "checkpoints" / "siamese" / "best.pt", 256, "resnet34", device), 256),
        ("supcon", load_encoder(ROOT / "checkpoints" / "siamese" / "supcon" / "best.pt", 256, "resnet34", device), 256),
        ("arcface", load_encoder(ROOT / "checkpoints" / "siamese" / "arcface_v2" / "best.pt", 512, "resnet50", device), 512),
    ]

    train_rows = filter_available_rows(load_meta_rows(train_meta), normalize_meta, iris_dir, "train_meta")
    val_rows = filter_available_rows(load_meta_rows(val_meta), normalize_meta, iris_dir, "val_meta")
    val_ids = set(val_rows["img_id"].astype(str))
    gallery_rows = train_rows[~train_rows["img_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    query_rows = val_rows.reset_index(drop=True)
    if args.limit is not None:
        gallery_rows = gallery_rows.head(args.limit).copy()
        query_rows = query_rows.head(args.limit).copy()
    if gallery_rows.empty or query_rows.empty:
        raise RuntimeError("No eligible gallery/query images for fusion database evaluation.")
    print(f"Gallery: {len(gallery_rows)}, Query: {len(query_rows)}")

    transform = default_transform(input_shape=(64, 512), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), train=False)

    all_gallery: list[np.ndarray] = []
    all_query: list[np.ndarray] = []
    for name, encoder, dim in encoders:
        print(f"\nExtracting {name} ({dim}d)...")
        gallery_features = extract_features(
            encoder,
            gallery_rows,
            iris_dir,
            transform,
            device,
            args.batch_size,
            args.num_workers,
            f"gallery {name}",
        )
        query_features = extract_features(
            encoder,
            query_rows,
            iris_dir,
            transform,
            device,
            args.batch_size,
            args.num_workers,
            f"query {name}",
        )
        all_gallery.append(gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-12))
        all_query.append(query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-12))

    gallery_features = np.concatenate(all_gallery, axis=1).astype("float32")
    query_features = np.concatenate(all_query, axis=1).astype("float32")
    total_dim = int(gallery_features.shape[1])
    print(f"\nFused dimension: {total_dim}")

    np.save(output_dir / "feature_db.npy", gallery_features)
    pg_id_map = load_pg_id_map(pigeon_csv)
    feature_meta = gallery_rows[["img_id", "blood_id", "blood_name"]].copy()
    feature_meta["pg_id"] = feature_meta["img_id"].astype(str).map(pg_id_map).fillna("")
    feature_meta["blood"] = feature_meta["blood_name"]
    feature_meta = feature_meta[["img_id", "pg_id", "blood", "blood_id", "blood_name"]]
    feature_meta.to_csv(output_dir / "feature_db_meta.csv", index=False)
    import faiss

    index = faiss.IndexFlatL2(total_dim)
    index.add(gallery_features)
    faiss.write_index(index, str(output_dir / "faiss_index.bin"))

    gallery_img_ids = gallery_rows["img_id"].astype(str).tolist()
    query_img_ids = query_rows["img_id"].astype(str).tolist()
    gallery_blood_names = gallery_rows["blood_name"].astype(str).tolist()
    query_blood_names = query_rows["blood_name"].astype(str).tolist()
    related_blood_names = load_related_blood_names(relations, pigeon_csv)
    blood_id_sets = load_blood_id_sets(relations)

    print("\nEvaluating...")
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
        max_pairs=200000,
        seed=42,
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
        max_pairs=200000,
        seed=42,
    )

    payload = {
        "fusion": "triplet_256d + supcon_256d + arcface_512d",
        "total_dim": total_dim,
        "search": metric_payload(search_metrics_new),
        "compare": metric_payload(compare_metrics_new),
    }
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (output_dir / "eval_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "single_label": {
                    "search": metric_payload(search_metrics_old),
                    "compare": metric_payload(compare_metrics_old),
                },
                "multi_label": {
                    "search": metric_payload(search_metrics_new),
                    "compare": metric_payload(compare_metrics_new),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if "threshold" in compare_metrics_new:
        with (output_dir / "threshold.json").open("w", encoding="utf-8") as f:
            json.dump({"threshold": float(compare_metrics_new["threshold"])}, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Concat {total_dim}d Fusion (multi-label blood_id)")
    print(f"{'=' * 60}")
    for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mAP"):
        print(f"  search_{key:<18}: {search_metrics_new[key]:.4f}")
    for key in ("accuracy", "balanced_accuracy", "auc", "eer"):
        print(f"  compare_{key:<18}: {compare_metrics_new[key]:.4f}")
    print(f"\nSaved to {output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
