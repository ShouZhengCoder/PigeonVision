from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import (
    BloodIdPKSampler,
    TripletMetaDataset,
    add_triplet_label_columns,
    build_triplet_label_maps,
    default_transform,
    load_triplet_meta,
)
from loss import batch_hard_triplet_loss
from model import IrisEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train triplet-loss iris encoder.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "siamese.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/siamese/last.pt.")
    parser.add_argument("--limit-train", type=int, default=None, help="Use first N train rows for smoke tests.")
    parser.add_argument("--limit-val", type=int, default=None, help="Use first N val rows for smoke tests.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weights.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on search_recall_at_1.")
    parser.add_argument("--smoke-min-recall", type=float, default=0.02, help="Smoke threshold for probe_recall_at_1.")
    parser.add_argument("--skip-smoke-gate", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("siamese_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def maybe_limit_df(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit >= len(df):
        return df
    return df.head(limit).copy()


def set_optimizer_lr(optimizer: Adam, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


def get_warmup_lr(epoch: int, warmup_epochs: int, base_lr: float, warmup_start_lr: float = 1e-5) -> float:
    if warmup_epochs <= 0 or warmup_epochs == 1:
        return float(base_lr)
    epoch = max(1, min(int(epoch), int(warmup_epochs)))
    progress = (epoch - 1) / max(warmup_epochs - 1, 1)
    return float(warmup_start_lr + (base_lr - warmup_start_lr) * progress)


def get_epoch_lr(epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return get_warmup_lr(epoch, warmup_epochs, base_lr)
    if warmup_epochs <= 0:
        cosine_step = max(epoch - 1, 0)
        cosine_epochs = max(total_epochs - 1, 1)
    else:
        cosine_step = max(epoch - warmup_epochs, 0)
        cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_step = min(cosine_step, cosine_epochs)
    return float(base_lr * 0.5 * (1.0 + math.cos(math.pi * cosine_step / cosine_epochs)))


def make_pk_loader(
    dataset: TripletMetaDataset,
    rows: pd.DataFrame,
    batch_size: int,
    classes_per_batch: int,
    samples_per_class: int,
    batches_per_epoch: int,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, BloodIdPKSampler]:
    sampler = BloodIdPKSampler(
        rows["blood_id_label"].tolist(),
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        batches_per_epoch=batches_per_epoch or math.ceil(len(rows) / max(batch_size, 1)),
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return loader, sampler


def make_eval_loader(dataset: TripletMetaDataset, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")


def run_epoch(
    encoder: IrisEncoder,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    margin: float,
    desc: str,
) -> tuple[float, float, float]:
    training = optimizer is not None
    encoder.train(training)
    total_loss = 0.0
    total_pos = 0.0
    total_neg = 0.0
    total_count = 0
    with torch.set_grad_enabled(training):
        for _img_ids, images, blood_ids, blood_names in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            blood_ids = blood_ids.to(device, non_blocking=True)
            blood_names = blood_names.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embeddings = encoder(images)
            loss, d_pos, d_neg = batch_hard_triplet_loss(embeddings, blood_ids, blood_names, margin=margin)
            if training:
                loss.backward()
                optimizer.step()
            batch_size_now = images.size(0)
            total_loss += float(loss.item()) * batch_size_now
            total_pos += float(d_pos.item()) * batch_size_now
            total_neg += float(d_neg.item()) * batch_size_now
            total_count += batch_size_now
    denom = max(total_count, 1)
    return total_loss / denom, total_pos / denom, total_neg / denom


@torch.no_grad()
def extract_embeddings(
    encoder: IrisEncoder,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    encoder.eval()
    ordered_ids: list[str] = []
    blood_ids: list[int] = []
    blood_names: list[int] = []
    features: list[np.ndarray] = []
    for img_ids, images, batch_blood_ids, batch_blood_names in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        emb = encoder(images).detach().cpu().numpy().astype(np.float32)
        features.append(emb)
        ordered_ids.extend([str(img_id) for img_id in img_ids])
        blood_ids.extend(batch_blood_ids.numpy().astype(np.int64).tolist())
        blood_names.extend(batch_blood_names.numpy().astype(np.int64).tolist())
    if not features:
        return ordered_ids, np.empty((0, encoder.feat_dim), dtype=np.float32), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    return (
        ordered_ids,
        np.concatenate(features, axis=0).astype(np.float32),
        np.asarray(blood_ids, dtype=np.int64),
        np.asarray(blood_names, dtype=np.int64),
    )


def compute_compare_metrics(features: np.ndarray, labels: np.ndarray, max_pairs: int = 200000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n < 2:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.0}

    label_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(idx)

    same_pairs: list[tuple[int, int]] = []
    for indices in label_to_indices.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                same_pairs.append((indices[i], indices[j]))
                if len(same_pairs) >= max_pairs // 2:
                    break
            if len(same_pairs) >= max_pairs // 2:
                break
        if len(same_pairs) >= max_pairs // 2:
            break
    if not same_pairs:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.0}

    neg_pairs: list[tuple[int, int]] = []
    attempts = 0
    target_neg = min(len(same_pairs), max_pairs // 2)
    while len(neg_pairs) < target_neg and attempts < max_pairs * 10:
        i, j = rng.integers(0, n, size=2)
        attempts += 1
        if i == j or labels[i] == labels[j]:
            continue
        neg_pairs.append((int(i), int(j)))
    if not neg_pairs:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.0}

    pair_n = min(len(same_pairs), len(neg_pairs))
    pairs = same_pairs[:pair_n] + neg_pairs[:pair_n]
    y_true = np.asarray([1] * pair_n + [0] * pair_n, dtype=np.int32)
    dists = np.asarray([float(np.linalg.norm(features[i] - features[j])) for i, j in pairs], dtype=np.float32)

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


def compute_search_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    n = len(labels)
    if n < 2:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mAP": 0.0}

    norms = np.sum(features * features, axis=1, keepdims=True)
    dist2 = norms + norms.T - 2.0 * (features @ features.T)
    np.fill_diagonal(dist2, np.inf)
    order = np.argsort(dist2, axis=1)

    recall_at: dict[str, float] = {}
    aps: list[float] = []
    for k in (1, 5, 10):
        hits = 0
        valid = 0
        for i in range(n):
            positives = np.where(labels == labels[i])[0]
            positives = positives[positives != i]
            if positives.size == 0:
                continue
            valid += 1
            topk = order[i, :k]
            if np.any(labels[topk] == labels[i]):
                hits += 1
        recall_at[f"recall_at_{k}"] = hits / max(valid, 1)

    for i in range(n):
        positives = np.where(labels == labels[i])[0]
        positives = positives[positives != i]
        if positives.size == 0:
            continue
        y = (labels[order[i]] == labels[i]).astype(np.int32)
        hits = 0
        precision_sum = 0.0
        for rank, is_pos in enumerate(y, start=1):
            if is_pos:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / max(len(positives), 1))
    return {
        "recall_at_1": recall_at["recall_at_1"],
        "recall_at_5": recall_at["recall_at_5"],
        "recall_at_10": recall_at["recall_at_10"],
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }


def compute_probe_recall(features: np.ndarray, labels: np.ndarray, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    query_indices: list[int] = []
    gallery_indices: list[int] = []
    for label in np.unique(labels):
        indices = np.where(labels == label)[0]
        if len(indices) < 2:
            continue
        query = int(rng.choice(indices))
        query_indices.append(query)
        gallery_indices.extend([int(idx) for idx in indices if idx != query])
    if not query_indices or not gallery_indices:
        return 0.0
    gallery = features[gallery_indices]
    gallery_labels = labels[gallery_indices]
    hits = 0
    for query in query_indices:
        distances = np.linalg.norm(gallery - features[query], axis=1)
        nearest = int(np.argmin(distances))
        hits += int(gallery_labels[nearest] == labels[query])
    return hits / len(query_indices)


def save_eval_metrics(output_dir: Path, epoch: int, compare_metrics: dict[str, float], search_metrics: dict[str, float], probe_recall: float) -> None:
    ensure_dir(output_dir)
    payload = {
        "epoch": int(epoch),
        "probe_recall_at_1": float(probe_recall),
        "compare": {key: float(value) for key, value in compare_metrics.items()},
        "search": {key: float(value) for key, value in search_metrics.items()},
    }
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (output_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": float(compare_metrics["threshold"]), "epoch": int(epoch)}, f, ensure_ascii=False, indent=2)


def save_checkpoint(
    path: Path,
    epoch: int,
    encoder: IrisEncoder,
    optimizer: Adam,
    best_metric: float,
    best_epoch: int,
    no_improve_epochs: int,
    config: dict,
    blood_id_to_label: dict[str, int],
    blood_name_to_label: dict[str, int],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "no_improve_epochs": int(no_improve_epochs),
            "config": config,
            "blood_id_to_label": blood_id_to_label,
            "blood_name_to_label": blood_name_to_label,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir = ensure_dir(resolve_root_path(config["checkpoint_dir"]))
    logger = setup_logger(ROOT / "logs" / "siamese_train.log")
    writer = SummaryWriter(log_dir=str(ROOT / "logs" / "tensorboard" / "siamese_triplet"))

    total_epochs = int(config["epochs"])
    warmup_epochs = int(config.get("warmup_epochs", 3))
    smoke_gate_epoch = int(config.get("smoke_gate_epoch", 5))
    base_lr = float(config["lr"])
    patience = int(args.patience)
    margin = float(config.get("triplet_margin", 0.3))
    batch_size = int(config.get("batch_size", 64))
    classes_per_batch = int(config.get("classes_per_batch", 16))
    samples_per_class = int(config.get("samples_per_class", 4))
    seed = int(config.get("seed", 42))

    train_meta = resolve_root_path(config.get("train_meta", ROOT / "data" / "train_meta.csv"))
    val_meta = resolve_root_path(config.get("val_meta", ROOT / "data" / "val_meta.csv"))
    iris_dir = resolve_root_path(config["iris_dir"])

    train_rows = load_triplet_meta(train_meta)
    val_rows = load_triplet_meta(val_meta)
    train_rows = maybe_limit_df(train_rows, args.limit_train)
    val_rows = maybe_limit_df(val_rows, args.limit_val)
    blood_id_to_label, blood_name_to_label = build_triplet_label_maps(train_rows, val_rows)
    train_rows = add_triplet_label_columns(train_rows, blood_id_to_label, blood_name_to_label)
    val_rows = add_triplet_label_columns(val_rows, blood_id_to_label, blood_name_to_label)

    train_ds = TripletMetaDataset(
        train_rows,
        iris_dir,
        transform=default_transform(
            input_shape=config["input_shape"],
            mean=config.get("normalize_mean", [0.5, 0.5, 0.5]),
            std=config.get("normalize_std", [0.5, 0.5, 0.5]),
            train=True,
        ),
    )
    val_ds = TripletMetaDataset(
        val_rows,
        iris_dir,
        transform=default_transform(
            input_shape=config["input_shape"],
            mean=config.get("normalize_mean", [0.5, 0.5, 0.5]),
            std=config.get("normalize_std", [0.5, 0.5, 0.5]),
            train=False,
        ),
    )

    batches_per_epoch = int(config.get("batches_per_epoch", 0) or math.ceil(len(train_rows) / max(batch_size, 1)))
    train_loader, train_sampler = make_pk_loader(
        train_ds,
        train_rows,
        batch_size,
        classes_per_batch,
        samples_per_class,
        batches_per_epoch,
        seed,
        args.num_workers,
        device,
    )
    val_loss_loader, val_loss_sampler = make_pk_loader(
        val_ds,
        val_rows,
        batch_size,
        classes_per_batch,
        samples_per_class,
        max(1, math.ceil(len(val_rows) / max(batch_size, 1))),
        seed + 1000,
        args.num_workers,
        device,
    )

    encoder = IrisEncoder(
        feat_dim=int(config.get("feat_dim", 256)),
        backbone=str(config.get("backbone", "resnet34")),
        pretrained=not args.no_pretrained,
        in_channels=int(config.get("in_channels", 3)),
    ).to(device)
    optimizer = Adam(encoder.parameters(), lr=base_lr)

    start_epoch = 1
    best_metric = -1.0
    best_epoch = 0
    no_improve_epochs = 0
    last_epoch = 0
    last_val_loss = float("inf")
    last_path = checkpoint_dir / "last.pt"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        state = torch.load(last_path, map_location=device)
        encoder.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        best_metric = float(state.get("best_metric", best_metric))
        best_epoch = int(state.get("best_epoch", state.get("epoch", 0)))
        no_improve_epochs = int(state.get("no_improve_epochs", 0))
        start_epoch = int(state["epoch"]) + 1
        logger.info("resumed from %s at epoch %s", last_path, start_epoch)

    logger.info(
        "device=%s train_rows=%s val_rows=%s blood_ids=%s blood_names=%s epochs=%s batch_size=%s lr=%s backbone=%s margin=%s warmup_epochs=%s smoke_gate_epoch=%s patience=%s",
        device,
        len(train_rows),
        len(val_rows),
        train_rows["blood_id"].nunique(),
        train_rows["blood_name"].nunique(),
        total_epochs,
        batch_size,
        base_lr,
        config.get("backbone", "resnet34"),
        margin,
        warmup_epochs,
        smoke_gate_epoch,
        patience,
    )

    smoke_probe_history: list[float] = []
    exit_code = 0
    for epoch in range(start_epoch, total_epochs + 1):
        last_epoch = epoch
        train_sampler.set_epoch(epoch)
        val_loss_sampler.set_epoch(epoch)
        current_lr = get_epoch_lr(epoch, total_epochs, warmup_epochs, base_lr)
        set_optimizer_lr(optimizer, current_lr)

        train_loss, train_pos, train_neg = run_epoch(encoder, train_loader, optimizer, device, margin, f"train {epoch}")
        last_val_loss, val_pos, val_neg = run_epoch(encoder, val_loss_loader, None, device, margin, f"val {epoch}")

        val_eval_loader = make_eval_loader(val_ds, batch_size, args.num_workers, device)
        _ordered_ids, val_features, val_blood_ids, val_blood_names = extract_embeddings(encoder, val_eval_loader, device, desc=f"val features {epoch}")
        probe_recall = compute_probe_recall(val_features, val_blood_ids, seed=seed)
        compare_metrics = compute_compare_metrics(val_features, val_blood_names, max_pairs=int(config.get("compare_eval_pairs", 200000)), seed=seed)
        search_metrics = compute_search_metrics(val_features, val_blood_names)
        save_eval_metrics(ROOT / "outputs" / "features", epoch, compare_metrics, search_metrics, probe_recall)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", last_val_loss, epoch)
        writer.add_scalar("distance/train_pos", train_pos, epoch)
        writer.add_scalar("distance/train_neg", train_neg, epoch)
        writer.add_scalar("distance/val_pos", val_pos, epoch)
        writer.add_scalar("distance/val_neg", val_neg, epoch)
        writer.add_scalar("lr", current_lr, epoch)
        writer.add_scalar("metric/probe_recall_at_1", probe_recall, epoch)
        writer.add_scalar("metric/search_recall_at_1", search_metrics["recall_at_1"], epoch)

        smoke_probe_history.append(float(probe_recall))
        smoke_tail = smoke_probe_history[-3:]
        smoke_rising = len(smoke_tail) == 3 and smoke_tail[0] < smoke_tail[1] < smoke_tail[2]

        current_metric = search_metrics["recall_at_1"]
        best_flag = 0
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            no_improve_epochs = 0
            best_flag = 1
            save_checkpoint(
                checkpoint_dir / "best.pt",
                epoch,
                encoder,
                optimizer,
                best_metric,
                best_epoch,
                no_improve_epochs,
                config,
                blood_id_to_label,
                blood_name_to_label,
            )
        else:
            no_improve_epochs += 1

        save_checkpoint(
            last_path,
            epoch,
            encoder,
            optimizer,
            best_metric,
            best_epoch,
            no_improve_epochs,
            config,
            blood_id_to_label,
            blood_name_to_label,
        )

        logger.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f d_pos=%.6f d_neg=%.6f probe_recall_at_1=%.6f lr=%.8f search_recall_at_1=%.6f search_recall_at_5=%.6f search_recall_at_10=%.6f search_map=%.6f compare_bal_acc=%.6f best=%s",
            epoch,
            train_loss,
            last_val_loss,
            val_pos,
            val_neg,
            probe_recall,
            current_lr,
            search_metrics["recall_at_1"],
            search_metrics["recall_at_5"],
            search_metrics["recall_at_10"],
            search_metrics["mAP"],
            compare_metrics["balanced_accuracy"],
            best_flag,
        )

        if epoch <= warmup_epochs:
            continue

        if not args.skip_smoke_gate and epoch == smoke_gate_epoch:
            if probe_recall < float(args.smoke_min_recall) and not smoke_rising:
                logger.error(
                    "smoke gate failed at epoch=%s probe_recall_at_1=%.6f threshold=%.6f rising=%s; stop training",
                    epoch,
                    probe_recall,
                    float(args.smoke_min_recall),
                    smoke_rising,
                )
                exit_code = 2
                break
            logger.info(
                "smoke gate check at epoch=%s probe_recall_at_1=%.6f threshold=%.6f rising=%s",
                epoch,
                probe_recall,
                float(args.smoke_min_recall),
                smoke_rising,
            )

        if epoch > smoke_gate_epoch and no_improve_epochs >= patience:
            logger.info(
                "early stopping triggered at epoch=%s best_epoch=%s best_metric=%.6f patience=%s no_improve_epochs=%s",
                epoch,
                best_epoch,
                best_metric,
                patience,
                no_improve_epochs,
            )
            break

    writer.close()
    best_path = checkpoint_dir / "best.pt"
    final_eval_epoch = last_epoch
    if best_path.exists():
        best_state = torch.load(best_path, map_location=device)
        encoder.load_state_dict(best_state["model_state"])
        final_eval_epoch = int(best_state.get("epoch", best_epoch or last_epoch))

    val_eval_loader = make_eval_loader(val_ds, batch_size, args.num_workers, device)
    _ordered_ids, val_features, val_blood_ids, val_blood_names = extract_embeddings(encoder, val_eval_loader, device, desc="final val features")
    probe_recall = compute_probe_recall(val_features, val_blood_ids, seed=seed)
    compare_metrics = compute_compare_metrics(val_features, val_blood_names, max_pairs=int(config.get("compare_eval_pairs", 200000)), seed=seed)
    search_metrics = compute_search_metrics(val_features, val_blood_names)
    save_eval_metrics(ROOT / "outputs" / "features", final_eval_epoch, compare_metrics, search_metrics, probe_recall)
    logger.info(
        "training finished actual_epochs=%s final_val_loss=%.6f best_epoch=%s best_metric=%.6f no_improve_epochs=%s probe_recall_at_1=%.6f compare_balanced_accuracy=%.6f search_recall_at_1=%.6f search_recall_at_5=%.6f search_recall_at_10=%.6f search_map=%.6f checkpoint=%s",
        last_epoch,
        last_val_loss,
        best_epoch,
        best_metric,
        no_improve_epochs,
        probe_recall,
        compare_metrics["balanced_accuracy"],
        search_metrics["recall_at_1"],
        search_metrics["recall_at_5"],
        search_metrics["recall_at_10"],
        search_metrics["mAP"],
        best_path,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
