"""Proxy-Anchor training with multi-label support for iris retrieval."""
from __future__ import annotations
import argparse, json, logging, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform, load_rgb_image, split_meta_for_training
from model import IrisEncoder
from loss_proxy_anchor import ProxyAnchorLoss, build_multi_label_positive_mask
from relation_metrics import load_blood_id_sets, compute_cross_search_metrics_by_blood_ids

# ── Dataset ──────────────────────────────────────────────
class ProxyDataset(Dataset):
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

# ── Blood-ID → Name Label mapping ────────────────────────
def build_blood_id_name_map(meta_path, blood_id_map_path):
    """Build mapping: blood_id index → set of blood_name label indices."""
    df = pd.read_csv(meta_path, dtype={"img_id": str, "blood_name": str})
    names = sorted(df["blood_name"].astype(str).unique())
    name_to_label = {n: i for i, n in enumerate(names)}
    
    bid_to_names: dict[int, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        bids = json.loads(row["blood_id_indices"])
        name = str(row["blood_name"])
        for b in bids:
            bid_to_names[int(b)].add(name)
    
    return {b: {name_to_label[n] for n in ns if n in name_to_label}
            for b, ns in bid_to_names.items()}, name_to_label

# ── LR schedule ──────────────────────────────────────────
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
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--feat-dim", type=int, default=512)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--lr-backbone", type=float, default=0.0001)
    ap.add_argument("--lr-proxy", type=float, default=0.01)
    ap.add_argument("--scale", type=float, default=1.0/9)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()
    device = torch.device(args.device)
    
    logger = logging.getLogger("proxy")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    for h in [logging.FileHandler(ensure_dir(ROOT/"logs")/"proxy_anchor.log", encoding="utf-8"),
              logging.StreamHandler()]:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    writer = SummaryWriter(log_dir=str(ROOT/"logs"/"tensorboard"/"proxy_anchor"))
    
    iris_dir = ROOT / "outputs" / "iris_normalized"
    # Merge train + val multi_meta, split internally for validation monitoring
    all_df = pd.concat([
        pd.read_csv(ROOT/"data"/"train_multi_meta.csv", dtype={"img_id": str, "blood_name": str}),
        pd.read_csv(ROOT/"data"/"val_multi_meta.csv", dtype={"img_id": str, "blood_name": str}),
    ], ignore_index=True)
    tr_rows, val_rows = split_meta_for_training(all_df, val_ratio=0.1, seed=42, group_col="blood_name")
    ckpt_dir = ensure_dir(ROOT/"checkpoints"/"siamese"/"proxy_anchor")
    tr_meta_path = ckpt_dir / "_train_multi_meta.csv"
    val_meta_path = ckpt_dir / "_val_multi_meta.csv"
    tr_rows.to_csv(tr_meta_path, index=False)
    val_rows.to_csv(val_meta_path, index=False)

    tr_ds = ProxyDataset(tr_meta_path, iris_dir,
                         transform=default_transform(input_shape=(64,512), train=True))
    val_ds = ProxyDataset(val_meta_path, iris_dir,
                          transform=default_transform(input_shape=(64,512), train=False))
    val_idx = list(range(len(val_ds)))

    # Build blood_id -> name_label mapping
    bid_name_map, name_to_label = build_blood_id_name_map(
        ROOT/"data"/"train_multi_meta.csv", ROOT/"data"/"blood_id_map.json")
    num_classes = len(name_to_label)
    
    blood_id_sets = load_blood_id_sets(ROOT/"data"/"extracted"/"datasetXGN"/"relations.csv")
    
    # Pre-select gallery subset for cross-eval
    rng = np.random.default_rng(42)
    gal_n = min(3000, len(tr_ds))
    gal_idx = sorted(rng.choice(len(tr_ds), gal_n, replace=False).tolist())
    
    encoder = IrisEncoder(feat_dim=args.feat_dim, backbone=args.backbone,
                          pretrained=True, in_channels=3).to(device)
    loss_fn = ProxyAnchorLoss(num_classes, args.feat_dim, scale=args.scale, margin=args.margin).to(device)
    
    optimizer = Adam([
        {"params": encoder.parameters(), "lr": args.lr_backbone},
        {"params": loss_fn.parameters(), "lr": args.lr_proxy},
    ])

    best_map, best_ep, no_imp = -1.0, 0, 0
    
    logger.info("classes=%s dim=%s backbone=%s lr_b=%.6f lr_p=%.4f scale=%.4f margin=%.3f",
                num_classes, args.feat_dim, args.backbone,
                args.lr_backbone, args.lr_proxy, args.scale, args.margin)
    
    seed = 42
    tr_indices = list(range(len(tr_ds)))
    
    for epoch in range(1, args.epochs + 1):
        lr_b = get_lr(epoch, args.epochs, 5, args.lr_backbone)
        lr_p = get_lr(epoch, args.epochs, 5, args.lr_proxy)
        optimizer.param_groups[0]["lr"] = lr_b
        optimizer.param_groups[1]["lr"] = lr_p
        
        encoder.train(); loss_fn.train()
        np.random.seed(seed + epoch); np.random.shuffle(tr_indices)
        tr_loader = DataLoader(torch.utils.data.Subset(tr_ds, tr_indices),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=collate)
        total_loss, steps = 0.0, 0
        offset = 0
        for batch in tqdm(tr_loader, desc=f"train {epoch}"):
            images = batch["images"].to(device, non_blocking=True)
            bs = len(images)
            ds_idx = tr_indices[offset:offset+bs]; offset += bs
            
            pos_mask = build_multi_label_positive_mask(
                [batch["bid_idx"][k] for k in range(bs)], bid_name_map, num_classes, device)
            
            if not pos_mask.any(): steps += 1; continue
            
            optimizer.zero_grad(set_to_none=True)
            embs = encoder(images)
            loss = loss_fn(embs, pos_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item(); steps += 1
        avg_loss = total_loss / max(steps, 1)
        
        # Eval every 2 epochs
        ml = {"recall_at_1":0.0,"recall_at_5":0.0,"recall_at_10":0.0,"mAP":0.0}
        if epoch % 2 == 0:
            encoder.eval()
            gids, gf = extract_eval(encoder, tr_ds, gal_idx, device, args.batch_size, args.num_workers)
            vids, vf = extract_eval(encoder, val_ds, val_idx, device, args.batch_size, args.num_workers)
            if len(vf) >= 2 and len(gf) >= 2:
                ml = compute_cross_search_metrics_by_blood_ids(
                    vf, [str(x) for x in vids], gf, [str(x) for x in gids], blood_id_sets)
        
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("metric/x_r1", ml["recall_at_1"], epoch)
        writer.add_scalar("metric/x_mAP", ml["mAP"], epoch)
        
        cur = ml.get("mAP", 0)
        best_flag = 0
        if cur > best_map:
            best_map, best_ep, no_imp, best_flag = cur, epoch, 0, 1
            torch.save({"model_state": encoder.state_dict(), "proxy_state": loss_fn.state_dict(),
                        "best_mAP": best_map, "best_epoch": best_ep, "name_to_label": name_to_label,
                        "feat_dim": args.feat_dim, "num_classes": num_classes},
                       ckpt_dir/"best.pt")
        else:
            no_imp += 1
        
        logger.info("epoch=%s loss=%.4f lr_b=%.6f lr_p=%.6f x_r1=%.4f x_r5=%.4f x_mAP=%.4f best=%s",
                    epoch, avg_loss, lr_b, lr_p, ml["recall_at_1"], ml["recall_at_5"], ml["mAP"], best_flag)
        
        if epoch > 10 and no_imp >= args.patience:
            logger.info("early stop epoch=%s best=%s best_mAP=%.4f", epoch, best_ep, best_map); break
    
    writer.close()
    logger.info("done. best_epoch=%s best_mAP=%.4f", best_ep, best_map)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
