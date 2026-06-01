"""Build FAISS feature DB using multi-encoder fusion (Concat 1024d).

Triplet 256d + SupCon 256d + ArcFace 512d -> L2-norm -> concat -> 1024d.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import faiss, numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform, load_rgb_image, load_triplet_meta
from model import IrisEncoder
from relation_metrics import (
    compute_cross_compare_metrics_by_blood_ids,
    compute_cross_compare_metrics_by_related_breeds,
    compute_cross_search_metrics_by_blood_ids,
    compute_cross_search_metrics_by_related_breeds,
    load_blood_id_sets, load_related_blood_names,
)


class IrisDbDataset(Dataset):
    def __init__(self, rows, img_dir, transform):
        self.rows = rows.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows.iloc[i]
        return str(row["img_id"]), self.transform(load_rgb_image(self.img_dir / f"{str(row['img_id'])}.png"))


def load_encoder(ckpt, feat_dim, backbone, device):
    encoder = IrisEncoder(feat_dim=feat_dim, backbone=backbone, pretrained=False, in_channels=3).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    ms = state.get("model_state", state)
    encoder.load_state_dict(ms)
    encoder.eval()
    return encoder


@torch.no_grad()
def extract(encoder, rows, img_dir, transform, device, bs, nw, desc):
    ds = IrisDbDataset(rows, img_dir, transform)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda"))
    feats = []
    for _ids, imgs in tqdm(loader, desc=desc):
        feats.append(encoder(imgs.to(device, non_blocking=True)).cpu().numpy().astype("float32"))
    return np.concatenate(feats, axis=0) if feats else np.empty((0, 1), dtype="float32")


def load_meta_img_ids(path: Path) -> pd.DataFrame:
    """Load unique img_ids from meta CSV. Handles both single-label and multi-label formats."""
    df = pd.read_csv(path, dtype={"img_id": str})
    if "blood_name" not in df.columns:
        df["blood_name"] = ""
    # Keep only img_id and blood_name, drop duplicates
    out = df[["img_id"]].copy()
    out["blood_name"] = df.get("blood_name", "")
    return out.drop_duplicates(subset=["img_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "features" / "fusion_1024d")
    ap.add_argument("--train-meta", type=Path, default=None)
    ap.add_argument("--val-meta", type=Path, default=None)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    iris_dir = ROOT / "outputs" / "iris_normalized"
    train_meta = args.train_meta or (ROOT / "data" / "train_meta.csv")
    val_meta = args.val_meta or (ROOT / "data" / "val_meta.csv")
    pigeon_csv = ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv"
    relations = ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv"

    print("Loading encoders...")
    enc_triplet = load_encoder(ROOT / "checkpoints" / "siamese" / "best.pt", 256, "resnet34", device)
    enc_supcon = load_encoder(ROOT / "checkpoints" / "siamese" / "supcon" / "best.pt", 256, "resnet34", device)
    enc_arcface = load_encoder(ROOT / "checkpoints" / "siamese" / "arcface_v2" / "best.pt", 512, "resnet50", device)
    encoders = [
        ("triplet", enc_triplet, 256),
        ("supcon", enc_supcon, 256),
        ("arcface", enc_arcface, 512),
    ]

    train_rows = load_meta_img_ids(train_meta)
    val_rows = load_meta_img_ids(val_meta)
    val_ids = set(val_rows["img_id"].astype(str))
    gallery_rows = train_rows[~train_rows["img_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    query_rows = val_rows.reset_index(drop=True)
    print(f"Gallery: {len(gallery_rows)}, Query: {len(query_rows)}")

    transform = default_transform(input_shape=(64, 512), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), train=False)

    all_gallery, all_query = [], []
    for name, enc, dim in encoders:
        print(f"\nExtracting {name} ({dim}d)...")
        gf = extract(enc, gallery_rows, iris_dir, transform, device, args.batch_size, args.num_workers, f"gallery {name}")
        qf = extract(enc, query_rows, iris_dir, transform, device, args.batch_size, args.num_workers, f"query {name}")
        all_gallery.append(gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12))
        all_query.append(qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-12))

    gallery_feats = np.concatenate(all_gallery, axis=1).astype("float32")
    query_feats = np.concatenate(all_query, axis=1).astype("float32")
    total_dim = gallery_feats.shape[1]
    print(f"\nFused dimension: {total_dim}")

    out = ensure_dir(args.output_dir)
    np.save(out / "feature_db.npy", gallery_feats)
    meta_cols = ["img_id"]
    for c in ["blood_id", "blood_name"]:
        if c in gallery_rows.columns:
            meta_cols.append(c)
    gallery_rows[meta_cols].to_csv(out / "feature_db_meta.csv", index=False)
    index = faiss.IndexFlatL2(total_dim)
    index.add(gallery_feats)
    faiss.write_index(index, str(out / "faiss_index.bin"))

    gids = gallery_rows["img_id"].astype(str).tolist()
    qids = query_rows["img_id"].astype(str).tolist()
    gbloods = gallery_rows["blood_name"].astype(str).tolist() if "blood_name" in gallery_rows.columns else [""] * len(gallery_rows)
    qbloods = query_rows["blood_name"].astype(str).tolist() if "blood_name" in query_rows.columns else [""] * len(query_rows)
    related = load_related_blood_names(relations, pigeon_csv)
    bid_sets = load_blood_id_sets(relations)

    print("\nEvaluating...")
    sm_new = compute_cross_search_metrics_by_blood_ids(query_feats, qids, gallery_feats, gids, bid_sets)
    cm_new = compute_cross_compare_metrics_by_blood_ids(query_feats, qids, gallery_feats, gids, bid_sets, max_pairs=200000, seed=42)
    sm_old = compute_cross_search_metrics_by_related_breeds(query_feats, qids, gallery_feats, gbloods, related)
    cm_old = compute_cross_compare_metrics_by_related_breeds(query_feats, qids, qbloods, gallery_feats, gids, gbloods, related, max_pairs=200000, seed=42)

    print(f"\n{'='*60}")
    print(f"Concat {total_dim}d Fusion (multi-label blood_id)")
    print(f"{'='*60}")
    for k in ("recall_at_1", "recall_at_5", "recall_at_10", "mAP"):
        print(f"  search_{k:<18}: {sm_new[k]:.4f}")
    for k in ("accuracy", "balanced_accuracy", "auc", "eer"):
        print(f"  compare_{k:<18}: {cm_new[k]:.4f}")

    payload = {
        "fusion": "triplet_256d + supcon_256d + arcface_512d",
        "total_dim": total_dim,
        "search": {k: round(float(v), 6) for k, v in sm_new.items()},
        "compare": {k: round(float(v), 6) for k, v in cm_new.items()},
    }
    with (out / "eval_metrics.json").open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (out / "eval_comparison.json").open("w") as f:
        json.dump({"single_label": {"search": sm_old, "compare": cm_old},
                   "multi_label":  {"search": sm_new, "compare": cm_new}}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
