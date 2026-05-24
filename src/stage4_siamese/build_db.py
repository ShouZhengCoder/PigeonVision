from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_existing, resolve_root_path
from dataset import default_transform
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
        with Image.open(self.img_dir / f"{img_id}.png") as image:
            image = image.convert("RGB")
            return img_id, self.transform(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build iris feature database and FAISS index.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "siamese.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--normalize-meta", type=Path, default=ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv")
    parser.add_argument("--pigeon-csv", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "features")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Use first N eligible rows for smoke tests.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_encoder(checkpoint_path: Path, feat_dim: int, device: torch.device) -> IrisEncoder:
    encoder = IrisEncoder(feat_dim=feat_dim, pretrained=False).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
    encoder.load_state_dict(model_state)
    encoder.eval()
    return encoder


def select_db_rows(normalize_meta: Path, pigeon_csv: Path, img_dir: Path) -> pd.DataFrame:
    normalize_df = pd.read_csv(normalize_meta, dtype={"img_id": str})
    success_ids = set(normalize_df[normalize_df["status"] == "success"]["img_id"].astype(str))

    try:
        pigeon_df = pd.read_csv(pigeon_csv, dtype={"ID": str})
    except pd.errors.ParserError:
        pigeon_df = pd.read_csv(pigeon_csv, dtype={"ID": str}, engine="python", on_bad_lines="skip")
        print(f"warning: skipped malformed rows while reading {pigeon_csv}")
    pigeon_df["ID"] = pigeon_df["ID"].astype(str)
    pigeon_df["BLOOD"] = pigeon_df["BLOOD"].fillna("").astype(str).str.strip()
    pigeon_df = pigeon_df[pigeon_df["BLOOD"] != ""]

    blood_counts = pigeon_df["BLOOD"].value_counts()
    keep_bloods = set(blood_counts[blood_counts >= 50].index)
    pigeon_df = pigeon_df[pigeon_df["BLOOD"].isin(keep_bloods)]
    pigeon_df = pigeon_df[pigeon_df["ID"].isin(success_ids)]
    pigeon_df = pigeon_df[pigeon_df["ID"].map(lambda img_id: (img_dir / f"{img_id}.png").exists())]

    rows = pd.DataFrame(
        {
            "img_id": pigeon_df["ID"].astype(str),
            "pg_id": pigeon_df["PG_ID"].fillna("").astype(str),
            "blood": pigeon_df["BLOOD"].astype(str),
        }
    )
    return rows.drop_duplicates(subset=["img_id"]).sort_values("img_id").reset_index(drop=True)


@torch.no_grad()
def extract_features(
    encoder: IrisEncoder,
    rows: pd.DataFrame,
    img_dir: Path,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    dataset = IrisDbDataset(rows, img_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features: list[np.ndarray] = []
    for _, images in tqdm(loader, desc="extract features"):
        images = images.to(device, non_blocking=True)
        feats = encoder(images).detach().cpu().numpy().astype("float32")
        features.append(feats)
    if not features:
        return np.empty((0, encoder.fc.out_features), dtype="float32")
    return np.concatenate(features, axis=0).astype("float32")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = args.checkpoint or (resolve_root_path(config["checkpoint_dir"]) / "best.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    normalize_meta = resolve_root_path(args.normalize_meta)
    pigeon_csv = resolve_existing(args.pigeon_csv, ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv")
    img_dir = resolve_root_path(config["iris_dir"])
    output_dir = ensure_dir(resolve_root_path(args.output_dir))

    rows = select_db_rows(normalize_meta, pigeon_csv, img_dir)
    if args.limit is not None:
        rows = rows.head(args.limit).copy()
    if rows.empty:
        raise RuntimeError("No eligible images for feature database.")

    transform = default_transform(
        input_shape=config["input_shape"],
        mean=config["normalize_mean"],
        std=config["normalize_std"],
    )
    encoder = load_encoder(checkpoint_path, int(config["feat_dim"]), device)
    features = extract_features(encoder, rows, img_dir, transform, device, args.batch_size, args.num_workers)

    feature_path = output_dir / "feature_db.npy"
    meta_path = output_dir / "feature_db_meta.csv"
    index_path = output_dir / "faiss_index.bin"

    np.save(feature_path, features)
    rows.to_csv(meta_path, index=False)
    index = faiss.IndexFlatL2(int(config["feat_dim"]))
    index.add(features)
    faiss.write_index(index, str(index_path))

    print(f"入库图片数: {len(rows)}")
    print(f"覆盖品系数: {rows['blood'].nunique()}")
    print(f"feature shape: {features.shape}")
    print(f"wrote: {feature_path}, {meta_path}, {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
