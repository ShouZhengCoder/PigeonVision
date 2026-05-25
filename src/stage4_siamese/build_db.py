from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform, load_rgb_image, load_triplet_meta
from model import IrisEncoder


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


def compute_cross_search_metrics(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
) -> dict[str, float]:
    if len(query_features) == 0 or len(gallery_features) == 0:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mAP": 0.0}

    index = faiss.IndexFlatL2(gallery_features.shape[1])
    index.add(gallery_features.astype("float32"))
    max_rank = len(gallery_features)
    recall_hits = {1: 0, 5: 0, 10: 0}
    aps: list[float] = []
    valid = 0

    chunk = 256
    for start in range(0, len(query_features), chunk):
        end = min(start + chunk, len(query_features))
        _distances, indices = index.search(query_features[start:end].astype("float32"), max_rank)
        for local_i, ranked_idx in enumerate(indices):
            query_label = query_labels[start + local_i]
            relevant_total = int(np.sum(gallery_labels == query_label))
            if relevant_total == 0:
                continue
            valid += 1
            ranked_labels = gallery_labels[ranked_idx]
            relevant = ranked_labels == query_label
            for k in (1, 5, 10):
                recall_hits[k] += int(np.any(relevant[: min(k, len(relevant))]))
            hits = 0
            precision_sum = 0.0
            for rank, is_relevant in enumerate(relevant, start=1):
                if is_relevant:
                    hits += 1
                    precision_sum += hits / rank
            aps.append(precision_sum / max(relevant_total, 1))

    return {
        "recall_at_1": recall_hits[1] / max(valid, 1),
        "recall_at_5": recall_hits[5] / max(valid, 1),
        "recall_at_10": recall_hits[10] / max(valid, 1),
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }


def compute_cross_compare_metrics(
    query_features: np.ndarray,
    query_labels: np.ndarray,
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    max_pairs: int = 200000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    label_to_gallery: dict[str, np.ndarray] = {
        str(label): np.where(gallery_labels == label)[0]
        for label in np.unique(gallery_labels)
    }
    positive_pairs: list[tuple[int, int]] = []
    query_order = np.arange(len(query_labels))
    rng.shuffle(query_order)
    target_pos = max_pairs // 2
    for query_idx in query_order:
        candidates = label_to_gallery.get(str(query_labels[query_idx]))
        if candidates is None or len(candidates) == 0:
            continue
        gallery_idx = int(rng.choice(candidates))
        positive_pairs.append((int(query_idx), gallery_idx))
        if len(positive_pairs) >= target_pos:
            break

    negative_pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(negative_pairs) < len(positive_pairs) and attempts < max_pairs * 20:
        attempts += 1
        query_idx = int(rng.integers(0, len(query_labels)))
        gallery_idx = int(rng.integers(0, len(gallery_labels)))
        if query_labels[query_idx] == gallery_labels[gallery_idx]:
            continue
        negative_pairs.append((query_idx, gallery_idx))

    pair_n = min(len(positive_pairs), len(negative_pairs))
    if pair_n == 0:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.0}

    pairs = positive_pairs[:pair_n] + negative_pairs[:pair_n]
    y_true = np.asarray([1] * pair_n + [0] * pair_n, dtype=np.int32)
    dists = np.asarray(
        [float(np.linalg.norm(query_features[q] - gallery_features[g])) for q, g in pairs],
        dtype=np.float32,
    )
    thresholds = np.unique(np.quantile(dists, np.linspace(0, 1, 256)))
    best_threshold = float(thresholds[0])
    best_acc = -1.0
    best_bal = -1.0
    for threshold in thresholds:
        preds = (dists <= threshold).astype(np.int32)
        acc = accuracy_score(y_true, preds)
        bal = balanced_accuracy_score(y_true, preds)
        if bal > best_bal:
            best_threshold = float(threshold)
            best_acc = float(acc)
            best_bal = float(bal)
    try:
        auc = float(roc_auc_score(y_true, -dists))
    except ValueError:
        auc = 0.0
    fpr, tpr, _roc_thresholds = roc_curve(y_true, -dists)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return {"accuracy": best_acc, "balanced_accuracy": best_bal, "auc": auc, "eer": eer, "threshold": best_threshold}


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
    gallery_labels = gallery_rows["blood_name"].astype(str).to_numpy()
    query_labels = query_rows["blood_name"].astype(str).to_numpy()

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

    search_metrics = compute_cross_search_metrics(query_features, query_labels, gallery_features, gallery_labels)
    compare_metrics = compute_cross_compare_metrics(
        query_features,
        query_labels,
        gallery_features,
        gallery_labels,
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
