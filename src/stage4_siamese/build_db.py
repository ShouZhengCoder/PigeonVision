from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform, load_rgb_image, load_triplet_meta
from model import IrisEncoder
from relation_metrics import (
    compute_cross_compare_metrics_by_related_breeds,
    compute_cross_search_metrics_by_related_breeds,
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
    parser = argparse.ArgumentParser(description="Build train-gallery feature DB and evaluate val queries.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "siamese.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-meta", type=Path, default=None)
    parser.add_argument("--val-meta", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Use first N train/query rows for smoke tests.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def load_encoder(checkpoint_path: Path, feat_dim: int, backbone: str, device: torch.device) -> tuple[IrisEncoder, int]:
    encoder = IrisEncoder(feat_dim=feat_dim, backbone=backbone, pretrained=False, in_channels=3).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    encoder.load_state_dict(model_state)
    encoder.eval()
    epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
    return encoder, epoch


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
        images = images.to(device, non_blocking=True)
        feats = encoder(images).detach().cpu().numpy().astype("float32")
        features.append(feats)
    if not features:
        return np.empty((0, encoder.feat_dim), dtype="float32")
    return np.concatenate(features, axis=0).astype("float32")


def save_eval_metrics(output_dir: Path, epoch: int, compare_metrics: dict[str, float], search_metrics: dict[str, float]) -> None:
    payload = {
        "epoch": int(epoch),
        "compare": {key: float(value) for key, value in compare_metrics.items()},
        "search": {key: float(value) for key, value in search_metrics.items()},
    }
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (output_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": float(compare_metrics["threshold"]), "epoch": int(epoch)}, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = args.checkpoint or (resolve_root_path(config["checkpoint_dir"]) / "best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    img_dir = resolve_root_path(config["iris_dir"])
    train_meta = resolve_root_path(args.train_meta or config.get("train_meta", ROOT / "data" / "train_meta.csv"))
    val_meta = resolve_root_path(args.val_meta or config.get("val_meta", ROOT / "data" / "val_meta.csv"))
    output_dir = ensure_dir(resolve_root_path(args.output_dir))
    pigeon_csv = resolve_root_path(config.get("pigeon_csv", ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv"))
    relations_path = resolve_root_path(
        config.get("relations", ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    )
    related_blood_names = load_related_blood_names(relations_path, pigeon_csv)

    train_rows = load_triplet_meta(train_meta)
    val_rows = load_triplet_meta(val_meta)
    val_img_ids = set(val_rows["img_id"].astype(str))
    gallery_rows = train_rows[~train_rows["img_id"].astype(str).isin(val_img_ids)].drop_duplicates(subset=["img_id"]).reset_index(drop=True)
    query_rows = val_rows.drop_duplicates(subset=["img_id"]).reset_index(drop=True)
    if args.limit is not None:
        gallery_rows = gallery_rows.head(args.limit).copy()
        query_rows = query_rows.head(args.limit).copy()
    if gallery_rows.empty or query_rows.empty:
        raise RuntimeError("No eligible gallery/query images for feature database evaluation.")

    transform = default_transform(
        input_shape=config["input_shape"],
        mean=config.get("normalize_mean", [0.5, 0.5, 0.5]),
        std=config.get("normalize_std", [0.5, 0.5, 0.5]),
        train=False,
    )
    encoder, checkpoint_epoch = load_encoder(
        checkpoint_path,
        int(config.get("feat_dim", 256)),
        str(config.get("backbone", "resnet34")),
        device,
    )

    gallery_features = extract_features(encoder, gallery_rows, img_dir, transform, device, args.batch_size, args.num_workers, "extract gallery")
    query_features = extract_features(encoder, query_rows, img_dir, transform, device, args.batch_size, args.num_workers, "extract val queries")
    gallery_img_ids = gallery_rows["img_id"].astype(str).tolist()
    query_img_ids = query_rows["img_id"].astype(str).tolist()
    gallery_blood_names = gallery_rows["blood_name"].astype(str).tolist()
    query_blood_names = query_rows["blood_name"].astype(str).tolist()

    feature_path = output_dir / "feature_db.npy"
    meta_path = output_dir / "feature_db_meta.csv"
    index_path = output_dir / "faiss_index.bin"
    np.save(feature_path, gallery_features)
    pg_id_map = load_pg_id_map(pigeon_csv)
    feature_meta = gallery_rows[["img_id", "blood_id", "blood_name"]].copy()
    feature_meta["pg_id"] = feature_meta["img_id"].astype(str).map(pg_id_map).fillna("")
    feature_meta["blood"] = feature_meta["blood_name"]
    feature_meta = feature_meta[["img_id", "pg_id", "blood", "blood_id", "blood_name"]]
    feature_meta.to_csv(meta_path, index=False)
    index = faiss.IndexFlatL2(int(config.get("feat_dim", 256)))
    index.add(gallery_features.astype("float32"))
    faiss.write_index(index, str(index_path))

    search_metrics = compute_cross_search_metrics_by_related_breeds(
        query_features,
        query_img_ids,
        gallery_features,
        gallery_blood_names,
        related_blood_names,
    )
    compare_metrics = compute_cross_compare_metrics_by_related_breeds(
        query_features,
        query_img_ids,
        query_blood_names,
        gallery_features,
        gallery_img_ids,
        gallery_blood_names,
        related_blood_names,
        max_pairs=int(config.get("compare_eval_pairs", 200000)),
        seed=int(config.get("seed", 42)),
    )
    save_eval_metrics(output_dir, checkpoint_epoch, compare_metrics, search_metrics)

    print(f"入库图片数: {len(gallery_rows)}")
    print(f"覆盖品系数: {gallery_rows['blood_name'].nunique()}")
    print(f"query图片数: {len(query_rows)}")
    print(f"query品系数: {query_rows['blood_name'].nunique()}")
    print(f"feature shape: {gallery_features.shape}")
    print(
        "metrics: "
        f"recall@1={search_metrics['recall_at_1']:.6f} "
        f"recall@5={search_metrics['recall_at_5']:.6f} "
        f"recall@10={search_metrics['recall_at_10']:.6f} "
        f"mAP={search_metrics['mAP']:.6f} "
        f"compare_bal_acc={compare_metrics['balanced_accuracy']:.6f} "
        f"auc={compare_metrics['auc']:.6f} "
        f"threshold={compare_metrics['threshold']:.6f}"
    )
    print(f"wrote: {feature_path}, {meta_path}, {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
