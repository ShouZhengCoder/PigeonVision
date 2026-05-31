"""Multi-label SupCon training with full blood_id information."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform
from dataset_multilabel import MultiLabelIrisDataset, collate_multilabel
from loss_supcon import supcon_loss, supcon_loss_with_weights
from model import IrisEncoder
from relation_metrics import load_blood_id_sets, compute_cross_search_metrics_by_blood_ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SupCon iris encoder with multi-label data.")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "siamese.yaml")
    p.add_argument("--train-meta", type=Path, default=ROOT / "data" / "train_multi_meta.csv")
    p.add_argument("--val-meta", type=Path, default=ROOT / "data" / "val_multi_meta.csv")
    p.add_argument("--blood-id-map", type=Path, default=ROOT / "data" / "blood_id_map.json")
    p.add_argument("--relations", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--use-sd-weights", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--smoke-min-recall", type=float, default=0.02)
    p.add_argument("--skip-smoke-gate", action="store_true")
    return p.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("supcon_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in [logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def get_lr(epoch, total_epochs, warmup_epochs, base_lr):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        progress = (max(1, epoch) - 1) / max(warmup_epochs - 1, 1)
        return 1e-5 + (base_lr - 1e-5) * progress
    if warmup_epochs <= 0:
        step, total = max(epoch - 1, 0), max(total_epochs - 1, 1)
    else:
        step, total = max(epoch - warmup_epochs, 0), max(total_epochs - warmup_epochs, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(step, total) / total))


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)


@torch.no_grad()
def extract_embs(encoder, ds, device, indices, batch_size, num_workers, desc):
    encoder.eval()
    ids_out, name_labels_out, feats = [], [], []
    subset = torch.utils.data.Subset(ds, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate_multilabel)
    for batch in tqdm(loader, desc=desc):
        images = batch["images"].to(device, non_blocking=True)
        emb = encoder(images).detach().cpu().numpy().astype(np.float32)
        feats.append(emb)
        ids_out.extend(batch["img_ids"])
        name_labels_out.extend(batch["blood_name_labels"].numpy().tolist())
    if not feats:
        return [], np.empty((0, encoder.feat_dim), dtype=np.float32), np.array([], dtype=np.int64)
    return ids_out, np.concatenate(feats, axis=0).astype(np.float32), np.array(name_labels_out, dtype=np.int64)


def compute_search_sl(features, labels):
    n = len(labels)
    if n < 2:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0}
    norms = np.sum(features * features, axis=1, keepdims=True)
    dist2 = norms + norms.T - 2.0 * (features @ features.T)
    np.fill_diagonal(dist2, np.inf)
    order = np.argsort(dist2, axis=1)
    recall = {}
    for k in (1, 5, 10):
        hits = valid = 0
        for i in range(n):
            pos = np.where(labels == labels[i])[0]
            pos = pos[pos != i]
            if len(pos) == 0:
                continue
            valid += 1
            if np.any(labels[order[i, :k]] == labels[i]):
                hits += 1
        recall[f"recall_at_{k}"] = hits / max(valid, 1)
    return recall


def save_ckpt(path, epoch, encoder, optimizer, best_metric, best_epoch, no_improve, config):
    ensure_dir(path.parent)
    torch.save({
        "epoch": int(epoch), "model_state": encoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": float(best_metric), "best_epoch": int(best_epoch),
        "no_improve_epochs": int(no_improve), "config": config,
    }, path)


def main():
    args = parse_args()
    with resolve_root_path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    feat_dim = int(config.get("feat_dim", 256))
    backbone = str(config.get("backbone", "resnet34"))
    base_lr = float(args.lr or config.get("lr", 0.0003))
    total_epochs = int(args.epochs or config.get("epochs", 80))
    warmup_epochs = int(config.get("warmup_epochs", 3))
    batch_size = int(args.batch_size)
    seed = int(config.get("seed", 42))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_dir = ensure_dir(resolve_root_path(config["checkpoint_dir"]) / "supcon")
    logger = setup_logger(ROOT / "logs" / "supcon_train.log")
    writer = SummaryWriter(log_dir=str(ROOT / "logs" / "tensorboard" / "supcon"))

    iris_dir = resolve_root_path(config["iris_dir"])
    tr_ds = MultiLabelIrisDataset(resolve_root_path(args.train_meta), resolve_root_path(args.blood_id_map), iris_dir,
                                  transform=default_transform(input_shape=config["input_shape"], train=True,
                                   mean=config.get("normalize_mean", [0.5]*3), std=config.get("normalize_std", [0.5]*3)))
    val_ds = MultiLabelIrisDataset(resolve_root_path(args.val_meta), resolve_root_path(args.blood_id_map), iris_dir,
                                   transform=default_transform(input_shape=config["input_shape"], train=False,
                                    mean=config.get("normalize_mean", [0.5]*3), std=config.get("normalize_std", [0.5]*3)))
    val_indices = list(range(len(val_ds)))
    blood_id_sets = load_blood_id_sets(resolve_root_path(args.relations))

    logger.info("train=%s val=%s blood_names=%s dim=%s backbone=%s lr=%.6f temp=%.3f",
                tr_ds.num_images, val_ds.num_images, tr_ds.num_blood_names, feat_dim, backbone, base_lr, args.temperature)

    encoder = IrisEncoder(feat_dim=feat_dim, backbone=backbone, pretrained=True, in_channels=3).to(device)
    optimizer = Adam(encoder.parameters(), lr=base_lr)

    start_epoch = 1
    best_metric = -1.0
    best_epoch = 0
    no_improve = 0
    last_path = ckpt_dir / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device)
        encoder.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_metric = float(state.get("best_metric", -1))
        best_epoch = int(state.get("best_epoch", 0))
        no_improve = int(state.get("no_improve_epochs", 0))
        start_epoch = int(state["epoch"]) + 1
        logger.info("resumed epoch=%s", start_epoch)

    exit_code = 0
    tr_indices = list(range(len(tr_ds)))

    for epoch in range(start_epoch, total_epochs + 1):
        current_lr = get_lr(epoch, total_epochs, warmup_epochs, base_lr)
        set_lr(optimizer, current_lr)

        # Train
        encoder.train()
        np.random.seed(seed + epoch)
        np.random.shuffle(tr_indices)
        tr_loader = DataLoader(
            torch.utils.data.Subset(tr_ds, tr_indices),
            batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_multilabel,
        )
        total_loss, steps = 0.0, 0
        offset = 0
        for batch in tqdm(tr_loader, desc=f"train {epoch}"):
            images = batch["images"].to(device, non_blocking=True)
            bs = len(images)
            ds_idx = tr_indices[offset:offset + bs]
            offset += bs
            pos_mask = tr_ds.build_batch_positive_mask(ds_idx, device)
            if not pos_mask.sum():
                steps += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            embs = encoder(images)
            if args.use_sd_weights:
                bid_sets = tr_ds.get_blood_id_sets_for_indices(ds_idx)
                loss = supcon_loss_with_weights(embs, pos_mask, bid_sets, temperature=args.temperature)
            else:
                loss = supcon_loss(embs, pos_mask, temperature=args.temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        avg_loss = total_loss / max(steps, 1)

        # Eval every 2 epochs
        search_ml = {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mAP": 0.0}
        search_sl = {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0}
        if epoch % 2 == 0 or epoch == start_epoch:
            vids, vfeats, vlabels = extract_embs(encoder, val_ds, device, val_indices, batch_size, args.num_workers, f"val {epoch}")
            vids_str = [str(x) for x in vids]
            if len(vfeats) >= 2:
                search_ml = compute_cross_search_metrics_by_blood_ids(vfeats, vids_str, vfeats, vids_str, blood_id_sets, exclude_self=True)
                search_sl = compute_search_sl(vfeats, vlabels)

        writer.add_scalar("loss/train", avg_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)
        writer.add_scalar("metric/ml_recall_at_1", search_ml["recall_at_1"], epoch)
        writer.add_scalar("metric/sl_recall_at_1", search_sl.get("recall_at_1", 0), epoch)

        cur_metric = search_ml.get("recall_at_1", 0)
        best_flag = 0
        if cur_metric > best_metric:
            best_metric, best_epoch, no_improve, best_flag = cur_metric, epoch, 0, 1
            save_ckpt(ckpt_dir / "best.pt", epoch, encoder, optimizer, best_metric, best_epoch, no_improve, config)
        else:
            no_improve += 1
        save_ckpt(last_path, epoch, encoder, optimizer, best_metric, best_epoch, no_improve, config)

        logger.info("epoch=%s loss=%.6f lr=%.7f ml_r1=%.6f ml_r5=%.6f ml_mAP=%.6f sl_r1=%.6f best=%s",
                    epoch, avg_loss, current_lr, search_ml["recall_at_1"], search_ml["recall_at_5"],
                    search_ml.get("mAP", 0), search_sl.get("recall_at_1", 0), best_flag)

        if epoch == int(config.get("smoke_gate_epoch", 5)) and not args.skip_smoke_gate:
            if search_ml.get("recall_at_1", 0) < args.smoke_min_recall:
                logger.error("smoke gate failed epoch=%s", epoch)
                exit_code = 2
                break
        if epoch > warmup_epochs and no_improve >= args.patience:
            logger.info("early stop epoch=%s best=%s", epoch, best_epoch)
            break

    writer.close()
    logger.info("done. best_epoch=%s best_ml_r1=%.6f", best_epoch, best_metric)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
