"""PK-SupCon v2: cross-eval monitoring + extended training."""
from __future__ import annotations
import argparse, json, logging, math, random
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform, load_rgb_image, split_meta_for_training
from model import IrisEncoder
from relation_metrics import load_blood_id_sets, compute_cross_search_metrics_by_blood_ids

class PKMLSampler(Sampler):
    def __init__(self, names, P=32, K=8, seed=42):
        self.P, self.K, self.seed, self.epoch = P, K, seed, 0
        by_n = defaultdict(list)
        for i, n in enumerate(names):
            if n: by_n[str(n)].append(i)
        self.by_n = {n: idxs for n, idxs in by_n.items() if len(idxs) >= K}
        self.names = sorted(self.by_n)
        self.batches = max(1, sum(len(v) for v in self.by_n.values()) // (P*K))
    def __len__(self): return self.batches
    def set_epoch(self, e): self.epoch = e
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches):
            c = rng.sample(self.names, self.P) if len(self.names) >= self.P else [rng.choice(self.names) for _ in range(self.P)]
            b = []
            for n in c:
                p = self.by_n[n]
                b.extend(rng.sample(p, self.K) if len(p) >= self.K else rng.choices(p, k=self.K))
            rng.shuffle(b); yield b

class PKDS(Dataset):
    def __init__(self, meta_path, iris_dir, transform=None):
        self.iris_dir = Path(iris_dir)
        self.transform = transform or default_transform(train=False)
        df = pd.read_csv(meta_path, dtype={"img_id": str, "blood_name": str})
        self.rows = df.reset_index(drop=True)
        self.names = [str(r["blood_name"]) for _, r in self.rows.iterrows()]
        self.bids = [json.loads(r) for r in self.rows["blood_id_indices"]]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        return {"img_id": str(r["img_id"]),
                "image": self.transform(load_rgb_image(self.iris_dir / f"{str(r['img_id'])}.png")),
                "bid_idx": self.bids[i], "blood_name": str(r["blood_name"])}

def pk_collate(b):
    return {"img_ids": [x["img_id"] for x in b], "images": torch.stack([x["image"] for x in b]),
            "bid_idx": [x["bid_idx"] for x in b], "blood_names": [x["blood_name"] for x in b]}

class Proj(nn.Module):
    def __init__(self, i=256, h=128, o=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i,h), nn.ReLU(inplace=True), nn.Linear(h,o))
    def forward(self, x): return F.normalize(self.net(x), p=2, dim=1)

def supcon(embs, pos, t=0.07):
    B = embs.size(0)
    embs = F.normalize(embs, p=2, dim=1)
    sim = (embs @ embs.T) / t
    eye = torch.eye(B, dtype=torch.bool, device=embs.device)
    pos = pos & ~eye
    has = pos.any(dim=1)
    if not has.all():
        v = has; pos = pos[v][:,v]; sim = sim[v][:,v]; eye = eye[v][:,v]; B = v.sum().item()
        if B < 2: return embs.sum() * 0.0
    sim = sim - sim.max(dim=1,keepdim=True).values.detach()
    es = torch.exp(sim) * ~eye
    d = es.sum(dim=1,keepdim=True).clamp(min=1e-8)
    n = (es * pos.float()).sum(dim=1,keepdim=True).clamp(min=1e-8)
    np = pos.float().sum(dim=1).clamp(min=1)
    return (-torch.log(n/d).squeeze() / np).mean()

def get_lr(ep, tot, wu, base):
    if wu > 0 and ep <= wu:
        return 1e-5 + (base - 1e-5) * (max(1,ep)-1) / max(wu-1,1)
    s = max(ep-wu,0) if wu>0 else max(ep-1,0)
    tc = max(tot-wu,1) if wu>0 else max(tot-1,1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(s,tc) / tc))

@torch.no_grad()
def extract(encoder, ds, indices, device, bs, nw):
    loader = DataLoader(torch.utils.data.Subset(ds, indices), batch_size=bs, shuffle=False,
                        num_workers=nw, collate_fn=pk_collate)
    ids, feats = [], []
    for b in loader:
        feats.append(F.normalize(encoder(b["images"].to(device)), p=2, dim=1).cpu().numpy())
        ids.extend(b["img_ids"])
    return ids, np.concatenate(feats).astype(np.float32) if feats else np.empty((0,256), dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=32); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80); ap.add_argument("--lr", type=float, default=0.0003)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--warm-start", type=Path, default=ROOT/"checkpoints"/"siamese"/"best.pt")
    args = ap.parse_args()
    device = torch.device(args.device)
    bs = args.P * args.K
    seed = 42

    logger = logging.getLogger("pk2")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    for h in [logging.FileHandler(ensure_dir(ROOT/"logs")/"pk_supcon_v2.log", encoding="utf-8"), logging.StreamHandler()]:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); logger.addHandler(h)
    writer = SummaryWriter(log_dir=str(ROOT/"logs"/"tensorboard"/"pk_supcon_v2"))

    iris_dir = ROOT / "outputs" / "iris_normalized"
    # Merge train + val multi_meta, split internally for validation monitoring
    all_df = pd.concat([
        pd.read_csv(ROOT/"data"/"train_multi_meta.csv", dtype={"img_id": str, "blood_name": str}),
        pd.read_csv(ROOT/"data"/"val_multi_meta.csv", dtype={"img_id": str, "blood_name": str}),
    ], ignore_index=True)
    tr_rows, val_rows = split_meta_for_training(all_df, val_ratio=0.1, seed=42, group_col="blood_name")
    ckpt_dir = ensure_dir(ROOT/"checkpoints"/"siamese"/"pk_supcon_v2")
    tr_meta_path = ckpt_dir / "_train_multi_meta.csv"
    val_meta_path = ckpt_dir / "_val_multi_meta.csv"
    tr_rows.to_csv(tr_meta_path, index=False)
    val_rows.to_csv(val_meta_path, index=False)

    tr_ds = PKDS(tr_meta_path, iris_dir,
                 transform=default_transform(input_shape=(64,512), train=True))
    val_ds = PKDS(val_meta_path, iris_dir,
                  transform=default_transform(input_shape=(64,512), train=False))
    blood_id_sets = load_blood_id_sets(ROOT/"data"/"extracted"/"datasetXGN"/"relations.csv")

    # Pre-select gallery subset for cross-eval (5000 random train images)
    rng = np.random.default_rng(seed)
    gal_n = min(5000, len(tr_ds))
    gal_idx = sorted(rng.choice(len(tr_ds), gal_n, replace=False).tolist())
    val_idx = list(range(len(val_ds)))

    encoder = IrisEncoder(feat_dim=256, backbone="resnet34", pretrained=True, in_channels=3).to(device)
    proj = Proj(i=256, h=128, o=128).to(device)
    if args.warm_start.exists():
        st = torch.load(args.warm_start, map_location=device)
        encoder.load_state_dict(st.get("model_state", st), strict=False)
        logger.info("warm-start ok")

    optimizer = Adam(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr)
    tr_sampler = PKMLSampler(tr_ds.names, P=args.P, K=args.K, seed=seed)

    best_mAP, best_ep, no_imp = -1.0, 0, 0
    for epoch in range(1, args.epochs + 1):
        lr = get_lr(epoch, args.epochs, 5, args.lr)
        for pg in optimizer.param_groups: pg["lr"] = lr
        encoder.train(); proj.train(); tr_sampler.set_epoch(epoch)
        tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=pk_collate)
        total_loss, steps = 0.0, 0
        for batch in tqdm(tr_loader, desc=f"train {epoch}"):
            images = batch["images"].to(device, non_blocking=True)
            B = len(images)
            sets = [set(b) for b in batch["bid_idx"]]
            pos = torch.zeros(B, B, dtype=torch.bool, device=device)
            for i in range(B):
                for j in range(i+1, B):
                    if sets[i] & sets[j]: pos[i,j] = pos[j,i] = True
            if not pos.sum(): steps += 1; continue
            optimizer.zero_grad(set_to_none=True)
            loss = supcon(proj(encoder(images)), pos, t=args.temp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item(); steps += 1
        avg_loss = total_loss / max(steps, 1)

        # Cross-eval every even epoch: val features vs train gallery subset
        ml = {"recall_at_1":0.0,"recall_at_5":0.0,"recall_at_10":0.0,"mAP":0.0}
        if epoch % 2 == 0:
            encoder.eval()
            gids, gfeats = extract(encoder, tr_ds, gal_idx, device, bs, args.num_workers)
            vids, vfeats = extract(encoder, val_ds, val_idx, device, bs, args.num_workers)
            if len(vfeats) >= 2 and len(gfeats) >= 2:
                ml = compute_cross_search_metrics_by_blood_ids(
                    vfeats, [str(x) for x in vids], gfeats, [str(x) for x in gids], blood_id_sets)

        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("metric/x_ml_r1", ml["recall_at_1"], epoch)
        writer.add_scalar("metric/x_ml_mAP", ml["mAP"], epoch)

        cur = ml.get("mAP", 0)  # use mAP for early stopping (recall@1 unreliable)
        best_flag = 0
        if cur > best_mAP:
            best_mAP, best_ep, no_imp, best_flag = cur, epoch, 0, 1
            torch.save({"model_state":encoder.state_dict(),"proj_state":proj.state_dict(),
                        "best_mAP":best_mAP,"best_epoch":best_ep}, ckpt_dir/"best.pt")
        else:
            no_imp += 1
        torch.save({"epoch":epoch,"model_state":encoder.state_dict(),"proj_state":proj.state_dict(),
                    "optimizer_state":optimizer.state_dict(),
                    "best_mAP":best_mAP,"best_epoch":best_ep}, ckpt_dir/"last.pt")

        logger.info("epoch=%s loss=%.4f lr=%.6f x_r1=%.4f x_r5=%.4f x_mAP=%.4f best=%s",
                    epoch, avg_loss, lr, ml["recall_at_1"], ml["recall_at_5"], ml["mAP"], best_flag)
        if epoch > 10 and no_imp >= args.patience:
            logger.info("early stop epoch=%s best=%s best_mAP=%.4f", epoch, best_ep, best_mAP); break

    writer.close()
    logger.info("done. best_epoch=%s best_mAP=%.4f", best_ep, best_mAP)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
