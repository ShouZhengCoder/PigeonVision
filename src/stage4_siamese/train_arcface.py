"""ArcFace multi-label training for iris retrieval."""
from __future__ import annotations
import argparse, json, logging, math, sys
from collections import defaultdict
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

# ── Dataset ──────────────────────────────────────────────
class ArcDataset(Dataset):
    def __init__(self, meta_path, iris_dir, transform=None):
        self.iris_dir = Path(iris_dir)
        self.transform = transform or default_transform(train=False)
        df = pd.read_csv(meta_path, dtype={"img_id": str, "blood_name": str})
        self.rows = df.reset_index(drop=True)
        self.blood_names = [str(r["blood_name"]) for _, r in self.rows.iterrows()]
        self.bid_indices = [json.loads(r) for r in self.rows["blood_id_indices"]]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        return {"img_id": str(r["img_id"]),
                "image": self.transform(load_rgb_image(self.iris_dir / f"{str(r['img_id'])}.png")),
                "bid_idx": self.bid_indices[i],
                "blood_name": str(r["blood_name"])}

def collate(batch):
    return {"img_ids": [b["img_id"] for b in batch],
            "images": torch.stack([b["image"] for b in batch]),
            "bid_idx": [b["bid_idx"] for b in batch],
            "blood_names": [b["blood_name"] for b in batch]}

# ── Multi-Label ArcFace Loss ─────────────────────────────
class MultiLabelArcFace(nn.Module):
    """ArcFace with multi-label support via binary cross-entropy.
    
    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss", CVPR 2019.
    Multi-label: each sample has multiple positive classes (blood_names from blood_ids).
    """
    def __init__(self, num_classes, feat_dim, s=30.0, m=0.5):
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
    
    def forward(self, embeddings, positive_mask):
        """ArcFace forward.
        
        Args:
            embeddings: (B, D) normalized features
            positive_mask: (B, C) bool, True for positive classes
        
        Returns:
            scalar loss (binary cross-entropy)
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # cos(theta): (B, C)
        cosine = embeddings @ weight_norm.T
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(1e-12, 1.0))
        
        # cos(theta + m) = cos*cos_m - sin*sin_m
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Only apply margin to positive classes
        # For theta > pi - m, use cos(theta) - m*sin(m) instead
        cond = cosine > self.th
        phi = torch.where(cond, phi, cosine - self.mm)
        
        # For negative classes, use original cosine
        output = torch.where(positive_mask, phi, cosine)
        output = output * self.s
        
        # Binary cross-entropy: multiple positives per sample
        positive_mask_f = positive_mask.float()
        # Mean over all (sample, class) pairs
        loss = F.binary_cross_entropy_with_logits(
            output, positive_mask_f, reduction='mean')
        return loss

# ── Blood-ID → Name Label mapping ────────────────────────
def build_name_map(meta_path):
    df = pd.read_csv(meta_path, dtype={"img_id": str, "blood_name": str})
    names = sorted(df["blood_name"].astype(str).unique())
    name_to_label = {n: i for i, n in enumerate(names)}
    bid_to_names: dict[int, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        bids = json.loads(row["blood_id_indices"])
        name = str(row["blood_name"])
        for b in bids:
            bid_to_names[int(b)].add(name)
    return bid_to_names, name_to_label, len(names)

def build_pos_mask(batch_bid_idx, bid_name_map, name_to_label, num_classes, device):
    B = len(batch_bid_idx)
    mask = torch.zeros(B, num_classes, dtype=torch.bool, device=device)
    for i, bids in enumerate(batch_bid_idx):
        names = set()
        for b in bids:
            names.update(bid_name_map.get(int(b), set()))
        for name in names:
            label = name_to_label.get(name, -1)
            if 0 <= label < num_classes:
                mask[i, label] = True
    return mask

# ── LR ────────────────────────────────────────────────────
def get_lr(ep, tot, wu, base):
    if wu > 0 and ep <= wu:
        return 1e-5 + (base - 1e-5) * (max(1,ep)-1) / max(wu-1,1)
    s = max(ep-wu,0) if wu>0 else max(ep-1,0)
    tc = max(tot-wu,1) if wu>0 else max(tot-1,1)
    return base * 0.5 * (1.0 + math.cos(math.pi*min(s,tc)/tc))

@torch.no_grad()
def extract_eval(encoder, ds, indices, device, bs, nw):
    loader = DataLoader(torch.utils.data.Subset(ds, indices), batch_size=bs,
                        shuffle=False, num_workers=nw, collate_fn=collate)
    ids, feats = [], []
    for b in loader:
        feats.append(encoder(b["images"].to(device)).cpu().numpy())
        ids.extend(b["img_ids"])
    return ids, np.concatenate(feats).astype(np.float32) if feats else np.empty((0,256), dtype=np.float32)

# ── Main ─────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--feat-dim", type=int, default=512)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--lr", type=float, default=0.0001)
    ap.add_argument("--s", type=float, default=30.0)
    ap.add_argument("--m", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--warm-start", type=Path, default=ROOT/"checkpoints"/"siamese"/"best.pt")
    args = ap.parse_args()
    device = torch.device(args.device)
    
    logger = logging.getLogger("arcface")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    for h in [logging.FileHandler(ensure_dir(ROOT/"logs")/"arcface.log", encoding="utf-8"),
              logging.StreamHandler()]:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    writer = SummaryWriter(log_dir=str(ROOT/"logs"/"tensorboard"/"arcface"))
    
    iris_dir = ROOT / "outputs" / "iris_normalized"
    # Merge train + val multi_meta, split internally for validation monitoring
    all_df = pd.concat([
        pd.read_csv(ROOT/"data"/"train_multi_meta.csv", dtype={"img_id": str, "blood_name": str}),
        pd.read_csv(ROOT/"data"/"val_multi_meta.csv", dtype={"img_id": str, "blood_name": str}),
    ], ignore_index=True)
    tr_rows, val_rows = split_meta_for_training(all_df, val_ratio=0.1, seed=42, group_col="blood_name")
    ckpt_dir = ensure_dir(ROOT/"checkpoints"/"siamese"/"arcface")
    tr_meta_path = ckpt_dir / "_train_multi_meta.csv"
    val_meta_path = ckpt_dir / "_val_multi_meta.csv"
    tr_rows.to_csv(tr_meta_path, index=False)
    val_rows.to_csv(val_meta_path, index=False)

    tr_ds = ArcDataset(tr_meta_path, iris_dir,
                       transform=default_transform(input_shape=(64,512), train=True))
    val_ds = ArcDataset(val_meta_path, iris_dir,
                        transform=default_transform(input_shape=(64,512), train=False))
    val_idx = list(range(len(val_ds)))

    bid_name_map, name_to_label, num_classes = build_name_map(ROOT/"data"/"train_multi_meta.csv")
    blood_id_sets = load_blood_id_sets(ROOT/"data"/"extracted"/"datasetXGN"/"relations.csv")
    
    rng = np.random.default_rng(42)
    gal_idx = sorted(rng.choice(len(tr_ds), min(3000, len(tr_ds)), replace=False).tolist())
    
    encoder = IrisEncoder(feat_dim=args.feat_dim, backbone=args.backbone,
                          pretrained=True, in_channels=3).to(device)
    
    logger.info("training from scratch (backbone=%s, dim=%s)", args.backbone, args.feat_dim)
    
    loss_fn = MultiLabelArcFace(num_classes, args.feat_dim, s=args.s, m=args.m).to(device)
    # Only train the classifier weight + backbone
    optimizer = Adam([
        {"params": encoder.parameters(), "lr": args.lr},
        {"params": loss_fn.parameters(), "lr": args.lr * 10},
    ])

    best_map, best_ep, no_imp = -1.0, 0, 0
    
    logger.info("classes=%s dim=%s backbone=%s lr=%.6f s=%.1f m=%.2f",
                num_classes, args.feat_dim, args.backbone, args.lr, args.s, args.m)
    
    seed = 42
    tr_indices = list(range(len(tr_ds)))
    
    for epoch in range(1, args.epochs + 1):
        lr = get_lr(epoch, args.epochs, 5, args.lr)
        for pg in optimizer.param_groups: pg["lr"] = lr * (10 if pg is optimizer.param_groups[1] else 1)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * 10
        
        encoder.train(); loss_fn.train()
        np.random.seed(seed + epoch); np.random.shuffle(tr_indices)
        tr_loader = DataLoader(torch.utils.data.Subset(tr_ds, tr_indices),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=collate)
        total_loss, steps = 0.0, 0
        for batch in tqdm(tr_loader, desc=f"train {epoch}"):
            images = batch["images"].to(device, non_blocking=True)
            pos_mask = build_pos_mask(batch["bid_idx"], bid_name_map, name_to_label, num_classes, device)
            if not pos_mask.any(): steps += 1; continue
            optimizer.zero_grad(set_to_none=True)
            embs = encoder(images)
            loss = loss_fn(embs, pos_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); steps += 1
        avg_loss = total_loss / max(steps, 1)
        
        ml = {"recall_at_1":0.0,"recall_at_5":0.0,"recall_at_10":0.0,"mAP":0.0}
        if epoch % 2 == 0:
            encoder.eval()
            gids, gf = extract_eval(encoder, tr_ds, gal_idx, device, args.batch_size, args.num_workers)
            vids, vf = extract_eval(encoder, val_ds, val_idx, device, args.batch_size, args.num_workers)
            if len(vf) >= 2 and len(gf) >= 2:
                ml = compute_cross_search_metrics_by_blood_ids(
                    vf, [str(x) for x in vids], gf, [str(x) for x in gids], blood_id_sets)
        
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("metric/x_mAP", ml["mAP"], epoch)
        writer.add_scalar("metric/x_r1", ml["recall_at_1"], epoch)
        
        cur = ml.get("mAP", 0)
        best_flag = 0
        if cur > best_map:
            best_map, best_ep, no_imp, best_flag = cur, epoch, 0, 1
            torch.save({"model_state": encoder.state_dict(), "weight": loss_fn.weight.data,
                        "best_mAP": best_map, "best_epoch": best_ep,
                        "feat_dim": args.feat_dim, "num_classes": num_classes,
                        "name_to_label": name_to_label, "s": args.s, "m": args.m},
                       ckpt_dir/"best.pt")
        else:
            no_imp += 1
        
        logger.info("epoch=%s loss=%.4f lr=%.6f x_r1=%.4f x_r5=%.4f x_mAP=%.4f best=%s",
                    epoch, avg_loss, lr, ml["recall_at_1"], ml["recall_at_5"], ml["mAP"], best_flag)
        
        if epoch > 10 and no_imp >= args.patience:
            logger.info("early stop epoch=%s best=%s best_mAP=%.4f", epoch, best_ep, best_map); break
    
    writer.close()
    logger.info("done. best_epoch=%s best_mAP=%.4f", best_ep, best_map)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
