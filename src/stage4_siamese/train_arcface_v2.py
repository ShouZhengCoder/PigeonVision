"""ArcFace single-label training for iris retrieval.

Standard ArcFace: one blood_name per image → CrossEntropyLoss.
This avoids the sparse-positive problem of multi-label BCE (0.075% positive ratio).

Reference: Deng et al., "ArcFace: Additive Angular Margin Loss", CVPR 2019.
"""
from __future__ import annotations
import argparse, json, logging, math, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform, load_rgb_image, split_meta_for_training
from model import IrisEncoder
from relation_metrics import load_blood_id_sets, compute_cross_search_metrics_by_blood_ids


# ── ArcFace Loss (standard single-label) ───────────────────
class ArcFace(nn.Module):
    """ArcFace with standard cross-entropy for single-label classification."""

    def __init__(self, num_classes: int, feat_dim: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.s = float(s)
        self.m = float(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ArcFace forward.

        Args:
            embeddings: (B, D) L2-normalized features
            labels: (B,) long tensor, class indices

        Returns:
            scalar cross-entropy loss
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # cos(theta): (B, C)
        cosine = embeddings @ weight_norm.T

        # One-hot labels to select target logits
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        cosine_target = (cosine * one_hot).sum(dim=1)

        sine = torch.sqrt((1.0 - cosine_target.pow(2)).clamp(1e-12, 1.0))

        # cos(theta + m) = cos*cos_m - sin*sin_m
        phi = cosine_target * self.cos_m - sine * self.sin_m

        # For theta > pi - m, use cos(theta) - m*sin(m)
        cond = cosine_target > self.th
        phi = torch.where(cond, phi, cosine_target - self.mm)

        # Replace target logit with margin-applied version
        output = cosine * self.s
        output.scatter_(1, labels.unsqueeze(1), phi.unsqueeze(1) * self.s)

        return F.cross_entropy(output, labels)


# ── Single-Label Dataset ────────────────────────────────────
class ArcSingleDataset(Dataset):
    """Each image → one blood_name label."""

    def __init__(self, meta_path: Path, iris_dir: Path, name_to_label: dict[str, int],
                 transform=None):
        self.iris_dir = Path(iris_dir)
        self.transform = transform or default_transform(train=False)
        df = pd.read_csv(meta_path, dtype={"img_id": str, "blood_name": str})
        self.rows: list[dict] = []
        skipped = 0
        for _, r in df.iterrows():
            name = str(r["blood_name"]).strip()
            label = name_to_label.get(name, -1)
            if label < 0:
                skipped += 1
                continue
            self.rows.append({"img_id": str(r["img_id"]), "label": label})
        if skipped:
            print(f"[ArcSingleDataset] {meta_path}: skipped {skipped} rows with unknown blood_name")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = load_rgb_image(self.iris_dir / f"{r['img_id']}.png")
        return {"img_id": r["img_id"], "image": self.transform(img), "label": r["label"]}


def collate_single(batch: list[dict]) -> dict:
    return {"img_ids": [b["img_id"] for b in batch],
            "images": torch.stack([b["image"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long)}


# ── Blood-Name → Label mapping (single-label) ───────────────
def build_single_label_map(train_meta_path: Path, val_meta_path: Path):
    """Build name_to_label from train_meta.csv (single-label)."""
    df_tr = pd.read_csv(train_meta_path, dtype={"img_id": str, "blood_name": str})
    df_val = pd.read_csv(val_meta_path, dtype={"img_id": str, "blood_name": str})
    all_names = sorted(set(df_tr["blood_name"].astype(str).str.strip().unique())
                       | set(df_val["blood_name"].astype(str).str.strip().unique()))
    name_to_label = {n: i for i, n in enumerate(all_names)}
    return name_to_label, len(all_names)


# ── LR schedule ─────────────────────────────────────────────
def get_lr(ep: int, tot: int, wu: int, base: float) -> float:
    if wu > 0 and ep <= wu:
        return 1e-5 + (base - 1e-5) * (max(1, ep) - 1) / max(wu - 1, 1)
    s = max(ep - wu, 0) if wu > 0 else max(ep - 1, 0)
    tc = max(tot - wu, 1) if wu > 0 else max(tot - 1, 1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(s, tc) / tc))


@torch.no_grad()
def extract_eval(encoder, ds, indices, device, bs, nw):
    loader = DataLoader(torch.utils.data.Subset(ds, indices), batch_size=bs,
                        shuffle=False, num_workers=nw, collate_fn=collate_single)
    ids, feats = [], []
    for b in loader:
        feats.append(encoder(b["images"].to(device)).cpu().numpy())
        ids.extend(b["img_ids"])
    return ids, np.concatenate(feats).astype(np.float32) if feats else np.empty((0, 256), dtype=np.float32)


# ── Main ────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--feat-dim", type=int, default=512)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--lr-head", type=float, default=0.001)
    ap.add_argument("--s", type=float, default=30.0)
    ap.add_argument("--m", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--gallery-size", type=int, default=3000)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("arcface_v2")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    for h in [logging.FileHandler(ensure_dir(ROOT / "logs") / "arcface_v2.log", encoding="utf-8"),
              logging.StreamHandler()]:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    writer = SummaryWriter(log_dir=str(ROOT / "logs" / "tensorboard" / "arcface_v2"))

    iris_dir = ROOT / "outputs" / "iris_normalized"
    train_meta = ROOT / "data" / "train_meta.csv"
    val_meta_path = ROOT / "data" / "val_meta.csv"

    # Merge train + val meta, split internally for validation monitoring
    all_rows = pd.concat([
        pd.read_csv(train_meta, dtype={"img_id": str, "blood_name": str}),
        pd.read_csv(val_meta_path, dtype={"img_id": str, "blood_name": str}),
    ], ignore_index=True)
    tr_rows, val_rows = split_meta_for_training(all_rows, val_ratio=0.1, seed=42, group_col="blood_id")
    ckpt_dir = ensure_dir(ROOT / "checkpoints" / "siamese" / "arcface_v2")
    train_meta = ckpt_dir / "_train_meta.csv"
    val_meta = ckpt_dir / "_val_meta.csv"
    tr_rows.to_csv(train_meta, index=False)
    val_rows.to_csv(val_meta, index=False)

    name_to_label, num_classes = build_single_label_map(train_meta, val_meta)

    tr_ds = ArcSingleDataset(train_meta, iris_dir, name_to_label,
                             transform=default_transform(input_shape=(64, 512), train=True))
    val_ds = ArcSingleDataset(val_meta, iris_dir, name_to_label,
                              transform=default_transform(input_shape=(64, 512), train=False))
    val_idx = list(range(len(val_ds)))

    blood_id_sets = load_blood_id_sets(ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")

    rng = np.random.default_rng(42)
    gal_idx = sorted(rng.choice(len(tr_ds), min(args.gallery_size, len(tr_ds)), replace=False).tolist())

    encoder = IrisEncoder(feat_dim=args.feat_dim, backbone=args.backbone,
                          pretrained=True, in_channels=3).to(device)
    arcface = ArcFace(num_classes, args.feat_dim, s=args.s, m=args.m).to(device)

    optimizer = Adam([
        {"params": encoder.parameters(), "lr": args.lr},
        {"params": arcface.parameters(), "lr": args.lr_head},
    ])

    best_map, best_ep, no_imp = -1.0, 0, 0

    logger.info("=== ArcFace V2 Single-Label Training ===")
    logger.info("classes=%d dim=%d backbone=%s lr=%.6f lr_head=%.6f s=%.1f m=%.2f epochs=%d batch=%d",
                num_classes, args.feat_dim, args.backbone, args.lr, args.lr_head,
                args.s, args.m, args.epochs, args.batch_size)
    logger.info("train samples=%d val samples=%d", len(tr_ds), len(val_ds))

    seed = 42
    tr_indices = list(range(len(tr_ds)))

    for epoch in range(1, args.epochs + 1):
        lr = get_lr(epoch, args.epochs, args.warmup_epochs, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * (args.lr_head / args.lr)

        encoder.train(); arcface.train()
        np.random.seed(seed + epoch); np.random.shuffle(tr_indices)
        tr_loader = DataLoader(torch.utils.data.Subset(tr_ds, tr_indices),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=collate_single)
        total_loss, steps = 0.0, 0
        for batch in tqdm(tr_loader, desc=f"train {epoch:02d}/{args.epochs}"):
            images = batch["images"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            embs = encoder(images)
            loss = arcface(embs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); steps += 1
        avg_loss = total_loss / max(steps, 1)

        ml = {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mAP": 0.0}
        if epoch % args.eval_every == 0:
            encoder.eval()
            gids, gf = extract_eval(encoder, tr_ds, gal_idx, device, args.batch_size, args.num_workers)
            vids, vf = extract_eval(encoder, val_ds, val_idx, device, args.batch_size, args.num_workers)
            if len(vf) >= 2 and len(gf) >= 2:
                ml = compute_cross_search_metrics_by_blood_ids(
                    vf, [str(x) for x in vids], gf, [str(x) for x in gids], blood_id_sets)

        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("metric/x_mAP", ml["mAP"], epoch)
        writer.add_scalar("metric/x_r1", ml["recall_at_1"], epoch)
        writer.add_scalar("metric/x_r5", ml["recall_at_5"], epoch)
        writer.add_scalar("metric/x_r10", ml["recall_at_10"], epoch)

        cur = ml.get("mAP", 0)
        best_flag = 0
        if cur > best_map:
            best_map, best_ep, no_imp, best_flag = cur, epoch, 0, 1
            torch.save({"model_state": encoder.state_dict(),
                        "arcface_state": arcface.state_dict(),
                        "best_mAP": best_map, "best_epoch": best_ep,
                        "feat_dim": args.feat_dim, "num_classes": num_classes,
                        "name_to_label": name_to_label, "s": args.s, "m": args.m,
                        "backbone": args.backbone},
                       ckpt_dir / "best.pt")
        else:
            no_imp += 1

        logger.info("epoch=%02d loss=%.4f lr=%.6f x_r1=%.4f x_r5=%.4f x_r10=%.4f x_mAP=%.4f best=%d",
                    epoch, avg_loss, lr, ml["recall_at_1"], ml["recall_at_5"],
                    ml["recall_at_10"], ml["mAP"], best_flag)

        if epoch > args.warmup_epochs + 5 and no_imp >= args.patience:
            logger.info("early stop epoch=%d best=%d best_mAP=%.4f", epoch, best_ep, best_map)
            break

    writer.close()
    logger.info("done. best_epoch=%d best_mAP=%.4f", best_ep, best_map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
