"""PK-SupCon: PK sampler + SupCon loss + projection head + warm-start."""
from __future__ import annotations
import argparse, json, logging, math, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
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

class PKMultiLabelSampler(Sampler):
    def __init__(self, blood_names, P=32, K=8, seed=42):
        self.P, self.K, self.seed, self.epoch = P, K, seed, 0
        by_name = defaultdict(list)
        for idx, name in enumerate(blood_names):
            if name: by_name[str(name)].append(idx)
        self.by_name = {n: idxs for n, idxs in by_name.items() if len(idxs) >= K}
        self.names = sorted(self.by_name)
        if not self.names: raise ValueError(f"no class with >= {K} samples")
        self.batches = max(1, sum(len(v) for v in self.by_name.values()) // (P * K))
    def __len__(self): return self.batches
    def set_epoch(self, epoch): self.epoch = epoch
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches):
            chosen = rng.sample(self.names, self.P) if len(self.names) >= self.P else [rng.choice(self.names) for _ in range(self.P)]
            batch = []
            for n in chosen:
                pool = self.by_name[n]
                batch.extend(rng.sample(pool, self.K) if len(pool) >= self.K else rng.choices(pool, k=self.K))
            rng.shuffle(batch)
            yield batch

class PKDataset(Dataset):
    def __init__(self, meta_path, iris_dir, transform=None):
        self.iris_dir, self.transform = Path(iris_dir), transform or default_transform(train=False)
        df = pd.read_csv(meta_path, dtype={"img_id": str, "blood_name": str})
        self.rows = df.reset_index(drop=True)
        self.blood_names = [str(r["blood_name"]) for _, r in self.rows.iterrows()]
        self.bid_indices = [json.loads(r) for r in self.rows["blood_id_indices"]]
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        return {"img_id": str(row["img_id"]),
                "image": self.transform(load_rgb_image(self.iris_dir / f"{str(row['img_id'])}.png")),
                "bid_idx": self.bid_indices[idx], "blood_name": str(row["blood_name"])}

def pk_collate(batch):
    return {"img_ids": [b["img_id"] for b in batch],
            "images": torch.stack([b["image"] for b in batch]),
            "bid_idx": [b["bid_idx"] for b in batch]}

class ProjHead(nn.Module):
    def __init__(self, in_d=256, hid=128, out_d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_d, hid), nn.ReLU(inplace=True), nn.Linear(hid, out_d))
    def forward(self, x): return F.normalize(self.net(x), p=2, dim=1)

def supcon_proj(embs, pos_mask, t=0.07):
    B = embs.size(0)
    embs = F.normalize(embs, p=2, dim=1)
    sim = (embs @ embs.T) / t
    eye = torch.eye(B, dtype=torch.bool, device=embs.device)
    pos_mask = pos_mask & ~eye
    has = pos_mask.any(dim=1)
    if not has.all():
        v = has; pos_mask = pos_mask[v][:,v]; sim = sim[v][:,v]; eye = eye[v][:,v]; B = v.sum().item()
        if B < 2: return embs.sum() * 0.0
    sim = sim - sim.max(dim=1,keepdim=True).values.detach()
    exp_sim = torch.exp(sim) * ~eye
    denom = exp_sim.sum(dim=1,keepdim=True).clamp(min=1e-8)
    num = (exp_sim * pos_mask.float()).sum(dim=1,keepdim=True).clamp(min=1e-8)
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    return (-torch.log(num/denom).squeeze() / n_pos).mean()

def get_lr(ep, total, warmup, base):
    if warmup > 0 and ep <= warmup:
        return 1e-5 + (base - 1e-5) * (max(1,ep)-1) / max(warmup-1,1)
    step = max(ep-warmup,0) if warmup>0 else max(ep-1,0)
    total_c = max(total-warmup,1) if warmup>0 else max(total-1,1)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(step,total_c) / total_c))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--P", type=int, default=32); p.add_argument("--K", type=int, default=8)
    p.add_argument("--epochs", type=int, default=80); p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--temp", type=float, default=0.07); p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--device", default="cuda"); p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=20); p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--warm-start", type=Path, default=ROOT/"checkpoints"/"siamese"/"best.pt")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bs = args.P * args.K  # 256

    logger = logging.getLogger("pk_sc")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    for h in [logging.FileHandler(ensure_dir(ROOT/"logs")/"pk_supcon.log", encoding="utf-8"), logging.StreamHandler()]:
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s")); logger.addHandler(h)
    writer = SummaryWriter(log_dir=str(ROOT/"logs"/"tensorboard"/"pk_supcon"))

    iris_dir = ROOT / "outputs" / "iris_normalized"
    # Merge train + val multi_meta, split internally for validation monitoring
    all_df = pd.concat([
        pd.read_csv(ROOT/"data"/"train_multi_meta.csv", dtype={"img_id": str, "blood_name": str}),
        pd.read_csv(ROOT/"data"/"val_multi_meta.csv", dtype={"img_id": str, "blood_name": str}),
    ], ignore_index=True)
    tr_rows, val_rows = split_meta_for_training(all_df, val_ratio=0.1, seed=42, group_col="blood_name")
    ckpt_dir = ensure_dir(ROOT/"checkpoints"/"siamese"/"pk_supcon")
    tr_meta_path = ckpt_dir / "_train_multi_meta.csv"
    val_meta_path = ckpt_dir / "_val_multi_meta.csv"
    tr_rows.to_csv(tr_meta_path, index=False)
    val_rows.to_csv(val_meta_path, index=False)

    tr_ds = PKDataset(tr_meta_path, iris_dir,
                      transform=default_transform(input_shape=(64,512), train=True))
    val_ds = PKDataset(val_meta_path, iris_dir,
                       transform=default_transform(input_shape=(64,512), train=False))
    val_idx = list(range(len(val_ds)))
    blood_id_sets = load_blood_id_sets(ROOT/"data"/"extracted"/"datasetXGN"/"relations.csv")

    tr_sampler = PKMultiLabelSampler(tr_ds.blood_names, P=args.P, K=args.K)

    encoder = IrisEncoder(feat_dim=256, backbone="resnet34", pretrained=True, in_channels=3).to(device)
    proj = ProjHead(in_d=256, hid=args.proj_dim, out_d=args.proj_dim).to(device)

    ws = args.warm_start
    if ws.exists():
        st = torch.load(ws, map_location=device)
        encoder.load_state_dict(st.get("model_state", st), strict=False)
        logger.info("warm-start from %s", ws)

    optimizer = Adam(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr)
    best_metric, best_ep, no_imp, start_ep = -1.0, 0, 0, 1

    for epoch in range(start_ep, args.epochs + 1):
        lr = get_lr(epoch, args.epochs, 5, args.lr)
        for pg in optimizer.param_groups: pg["lr"] = lr
        encoder.train(); proj.train(); tr_sampler.set_epoch(epoch)
        tr_loader = DataLoader(tr_ds, batch_sampler=tr_sampler, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=pk_collate)
        total_loss, steps = 0.0, 0
        for batch in tqdm(tr_loader, desc=f"train {epoch}"):
            images = batch["images"].to(device, non_blocking=True)
            B = len(images)
            # Build blood_id overlap mask
            sets = [set(b) for b in batch["bid_idx"]]
            pos = torch.zeros(B, B, dtype=torch.bool, device=device)
            for i in range(B):
                for j in range(i+1, B):
                    if sets[i] & sets[j]: pos[i,j] = pos[j,i] = True
            if not pos.sum(): steps += 1; continue
            optimizer.zero_grad(set_to_none=True)
            feats = encoder(images)
            loss = supcon_proj(proj(feats), pos, t=args.temp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item(); steps += 1
        avg_loss = total_loss / max(steps, 1)

        ml = {"recall_at_1":0.0,"recall_at_5":0.0,"recall_at_10":0.0,"mAP":0.0}
        if epoch % 2 == 0 or epoch == 1:
            encoder.eval()
            eloader = DataLoader(torch.utils.data.Subset(val_ds, val_idx), batch_size=bs, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pk_collate)
            vids, vfeats = [], []
            for batch in tqdm(eloader, desc=f"val {epoch}"):
                vfeats.append(F.normalize(encoder(batch["images"].to(device)), p=2, dim=1).detach().cpu().numpy())
                vids.extend(batch["img_ids"])
            vfeats = np.concatenate(vfeats).astype(np.float32) if vfeats else np.empty((0,256), dtype=np.float32)
            if len(vfeats) >= 2:
                ml = compute_cross_search_metrics_by_blood_ids(vfeats, [str(x) for x in vids],
                                                               vfeats, [str(x) for x in vids],
                                                               blood_id_sets, exclude_self=True)
        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("metric/ml_r1", ml["recall_at_1"], epoch)
        cur = ml.get("recall_at_1", 0)
        if cur > best_metric: best_metric, best_ep, no_imp = cur, epoch, 0
        else: no_imp += 1
        logger.info("epoch=%s loss=%.4f lr=%.6f ml_r1=%.4f ml_r5=%.4f ml_mAP=%.4f best=%s",
                    epoch, avg_loss, lr, ml["recall_at_1"], ml["recall_at_5"], ml["mAP"], int(cur>best_metric or cur==best_metric))
        if epoch > 5 and no_imp >= args.patience:
            logger.info("early stop epoch=%s best=%s", epoch, best_ep); break
    torch.save({"model_state":encoder.state_dict(),"proj_state":proj.state_dict(),
                "best_metric":best_metric,"best_epoch":best_ep}, ckpt_dir/"best.pt")
    logger.info("done. best_epoch=%s best_ml_r1=%.4f", best_ep, best_metric)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
