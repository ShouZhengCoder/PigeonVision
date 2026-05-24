from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import (
    IrisClassDataset,
    PKBatchSampler,
    add_label_column,
    default_transform,
    load_blood_rows,
    split_by_blood,
)
from model import IrisEncoder, SubCenterArcFace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ArcFace iris blood classifier.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "siamese.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/siamese/last.pt.")
    parser.add_argument("--limit-train", type=int, default=None, help="Use first N train samples for smoke tests.")
    parser.add_argument("--limit-val", type=int, default=None, help="Use first N val samples for smoke tests.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weights.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on search_recall_at_1.")
    parser.add_argument("--smoke-min-recall", type=float, default=0.02, help="Smoke gate threshold for probe_recall_at_1.")
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
    if warmup_epochs <= 0:
        return float(base_lr)
    if warmup_epochs == 1:
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


def run_epoch(
    encoder: IrisEncoder,
    head: torch.nn.Module,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    label_smoothing: float,
    desc: str,
) -> float:
    training = optimizer is not None
    encoder.train(training)
    head.train(training)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0
    total_count = 0
    with torch.set_grad_enabled(training):
        for _img_id, images, labels in tqdm(loader, desc=desc):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            feats = encoder(images)
            logits = head(feats, labels)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


@torch.no_grad()
def extract_embeddings(
    encoder: IrisEncoder,
    loader: DataLoader,
    device: torch.device,
    desc: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    encoder.eval()
    ordered_ids: list[str] = []
    labels: list[int] = []
    feats: list[np.ndarray] = []
    for img_ids, images, batch_labels in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        emb = encoder(images).detach().cpu().numpy().astype(np.float32)
        feats.append(emb)
        ordered_ids.extend([str(img_id) for img_id in img_ids])
        labels.extend(batch_labels.numpy().astype(np.int64).tolist())
    return ordered_ids, np.concatenate(feats, axis=0), np.asarray(labels, dtype=np.int64)


def compute_compare_metrics(features: np.ndarray, labels: np.ndarray, max_pairs: int = 200000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n < 2:
        return {"accuracy": 0.0, "balanced_accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.0}

    same_pairs: list[tuple[int, int]] = []
    label_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(int(label), []).append(idx)
    for indices in label_to_indices.values():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                same_pairs.append((indices[i], indices[j]))

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

    same_n = min(len(same_pairs), target_neg)
    neg_n = min(len(neg_pairs), same_n)
    pairs = same_pairs[:same_n] + neg_pairs[:neg_n]
    y_true = np.asarray([1] * same_n + [0] * neg_n, dtype=np.int32)
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
    fpr, tpr, roc_thresholds = roc_curve(y_true, -dists)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return {
        "accuracy": best_acc,
        "balanced_accuracy": best_bal,
        "auc": auc,
        "eer": eer,
        "threshold": best_threshold,
    }


def compute_search_metrics(features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    n = len(labels)
    if n < 2:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mAP": 0.0}

    norms = np.sum(features * features, axis=1, keepdims=True)
    dist2 = norms + norms.T - 2.0 * (features @ features.T)
    np.fill_diagonal(dist2, np.inf)
    order = np.argsort(dist2, axis=1)

    recall_at = {}
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
        ranked = order[i]
        y = (labels[ranked] == labels[i]).astype(np.int32)
        hits = 0
        precision_sum = 0.0
        for rank, is_pos in enumerate(y, start=1):
            if is_pos:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / max(len(positives), 1))
    m_ap = float(np.mean(aps)) if aps else 0.0
    return {
        "recall_at_1": recall_at["recall_at_1"],
        "recall_at_5": recall_at["recall_at_5"],
        "recall_at_10": recall_at["recall_at_10"],
        "mAP": m_ap,
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


def embed_visualizations(features: np.ndarray, labels: np.ndarray, output_dir: Path, epoch: int) -> dict[str, str]:
    ensure_dir(output_dir)
    sample_n = min(len(features), 2000)
    if sample_n == 0:
        return {}
    idx = np.linspace(0, len(features) - 1, sample_n, dtype=int)
    X = features[idx]
    y = labels[idx]
    pca = PCA(n_components=2, random_state=42).fit_transform(X)
    paths: dict[str, str] = {}

    def _plot(points: np.ndarray, name: str) -> str:
        fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
        sc = ax.scatter(points[:, 0], points[:, 1], c=y, s=10, cmap="tab20", alpha=0.8, linewidths=0)
        ax.set_title(f"{name} epoch {epoch}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        out = output_dir / f"{name}_epoch_{epoch:03d}.png"
        fig.tight_layout()
        fig.savefig(out)
        plt.close(fig)
        return str(out)

    paths["pca"] = _plot(pca, "pca")
    try:
        import umap

        umap_points = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, max(2, len(X) - 1))).fit_transform(X)
        paths["umap"] = _plot(umap_points, "umap")
    except Exception as exc:
        print(f"warning: UMAP unavailable for epoch {epoch}: {exc}")
    return paths


def save_checkpoint(
    path: Path,
    epoch: int,
    encoder: IrisEncoder,
    head: torch.nn.Module,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    best_metric: float,
    best_epoch: int,
    no_improve_epochs: int,
    config: dict,
    label_to_blood: dict[int, str],
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "no_improve_epochs": no_improve_epochs,
            "label_to_blood": label_to_blood,
            "config": config,
        },
        path,
    )


def save_eval_metrics(output_dir: Path, epoch: int, compare_metrics: dict[str, float], search_metrics: dict[str, float], probe_recall: float) -> None:
    ensure_dir(output_dir)
    payload = {
        "epoch": int(epoch),
        "probe_recall_at_1": float(probe_recall),
        "compare": {k: float(v) for k, v in compare_metrics.items()},
        "search": {k: float(v) for k, v in search_metrics.items()},
    }
    with (output_dir / "eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with (output_dir / "threshold.json").open("w", encoding="utf-8") as f:
        json.dump({"threshold": float(compare_metrics["threshold"]), "epoch": int(epoch)}, f, ensure_ascii=False, indent=2)


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
    log_path = ROOT / "logs" / "siamese_train.log"
    logger = setup_logger(log_path)
    writer = SummaryWriter(log_dir=str(ROOT / "logs" / "tensorboard" / "siamese_v2"))

    warmup_epochs = int(config.get("warmup_epochs", 3))
    smoke_gate_epoch = int(config.get("smoke_gate_epoch", 5))
    base_lr = float(config["lr"])
    patience = int(args.patience)
    smoke_min_recall = float(args.smoke_min_recall)
    total_epochs = int(config["epochs"])
    cosine_epochs = max(total_epochs - warmup_epochs, 1)

    normalize_meta = resolve_root_path(config["normalize_meta"]) if "normalize_meta" in config else ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv"
    pigeon_csv = resolve_root_path(config["pigeon_csv"]) if "pigeon_csv" in config else ROOT / "data" / "extracted" / "datasetXGN" / "pigeon.csv"
    iris_dir = resolve_root_path(config["iris_dir"])

    rows = load_blood_rows(
        normalize_meta=normalize_meta,
        pigeon_csv=pigeon_csv,
        img_dir=iris_dir,
        min_images_per_blood=int(config.get("min_images_per_blood", 5)),
    )
    train_rows, val_rows = split_by_blood(rows, val_ratio=float(config.get("val_ratio", 0.2)), seed=int(config.get("seed", 42)))
    if args.limit_train is not None:
        train_rows = maybe_limit_df(train_rows, args.limit_train)
    if args.limit_val is not None:
        val_rows = maybe_limit_df(val_rows, args.limit_val)
    train_rows, val_rows, label_to_blood = add_label_column(train_rows, val_rows)

    label_to_blood_inv = {int(label): blood for blood, label in label_to_blood.items()}
    train_ds = IrisClassDataset(
        train_rows,
        iris_dir,
        transform=default_transform(
            input_shape=config["input_shape"],
            mean=config.get("normalize_mean", [0.5, 0.5, 0.5]),
            std=config.get("normalize_std", [0.5, 0.5, 0.5]),
            train=True,
        ),
    )
    val_ds = IrisClassDataset(
        val_rows,
        iris_dir,
        transform=default_transform(
            input_shape=config["input_shape"],
            mean=config.get("normalize_mean", [0.5, 0.5, 0.5]),
            std=config.get("normalize_std", [0.5, 0.5, 0.5]),
            train=False,
        ),
    )

    batch_size = int(config.get("batch_size", 64))
    classes_per_batch = int(config.get("classes_per_batch", 16))
    samples_per_class = int(config.get("samples_per_class", 4))
    train_sampler = PKBatchSampler(
        labels=train_rows["label"].tolist(),
        classes_per_batch=classes_per_batch,
        samples_per_class=samples_per_class,
        batches_per_epoch=int(config.get("batches_per_epoch", math.ceil(len(train_rows) / max(batch_size, 1)))),
        seed=int(config.get("seed", 42)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder = IrisEncoder(
        feat_dim=int(config.get("feat_dim", 256)),
        backbone=str(config.get("backbone", "resnet18")),
        pretrained=not args.no_pretrained,
        in_channels=int(config.get("in_channels", 3)),
    ).to(device)
    head = SubCenterArcFace(
        feat_dim=int(config.get("feat_dim", 256)),
        num_classes=int(len(train_rows["label"].unique())),
        subcenters=int(config.get("arcface_subcenters", 3)),
        scale=float(config.get("arcface_scale", 30.0)),
        margin=float(config.get("arcface_margin", 0.2)),
    ).to(device)
    optimizer = Adam(list(encoder.parameters()) + list(head.parameters()), lr=base_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs)

    start_epoch = 1
    best_metric = -1.0
    best_epoch = 0
    no_improve_epochs = 0
    last_val_loss = float("inf")
    last_epoch = 0
    last_path = checkpoint_dir / "last.pt"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        state = torch.load(last_path, map_location=device)
        encoder.load_state_dict(state["model_state"])
        if "head_state" in state:
            head.load_state_dict(state["head_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best_metric = float(state.get("best_metric", best_metric))
        best_epoch = int(state.get("best_epoch", state.get("epoch", 0)))
        no_improve_epochs = int(state.get("no_improve_epochs", 0))
        start_epoch = int(state["epoch"]) + 1
        logger.info("resumed from %s at epoch %s", last_path, start_epoch)

    logger.info(
        "device=%s train_images=%s val_images=%s classes=%s epochs=%s batch_size=%s lr=%s backbone=%s warmup_epochs=%s smoke_gate_epoch=%s patience=%s",
        device,
        len(train_rows),
        len(val_rows),
        len(train_rows["label"].unique()),
        config["epochs"],
        batch_size,
        config["lr"],
        config.get("backbone", "resnet18"),
        warmup_epochs,
        smoke_gate_epoch,
        patience,
    )

    smoke_probe_history: list[float] = []
    exit_code = 0

    for epoch in range(start_epoch, total_epochs + 1):
        last_epoch = epoch
        train_sampler.set_epoch(epoch)
        current_lr = get_epoch_lr(epoch, total_epochs, warmup_epochs, base_lr)
        set_optimizer_lr(optimizer, current_lr)
        train_loss = run_epoch(
            encoder,
            head,
            train_loader,
            optimizer,
            device,
            float(config.get("label_smoothing", 0.05)),
            f"train {epoch}",
        )
        encoder.eval()
        head.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_count = 0
            for _img_id, images, labels in tqdm(val_loader, desc=f"val {epoch}"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                feats = encoder(images)
                logits = head(feats, labels)
                loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=float(config.get("label_smoothing", 0.05)))
                batch_size_now = images.size(0)
                total_loss += loss.item() * batch_size_now
                total_count += batch_size_now
            last_val_loss = total_loss / max(total_count, 1)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", last_val_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        val_loader_eval = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        _ordered_ids, val_features, val_labels = extract_embeddings(encoder, val_loader_eval, device, desc=f"val features {epoch}")
        probe_recall = compute_probe_recall(val_features, val_labels, seed=int(config.get("seed", 42)))
        compare_metrics = compute_compare_metrics(val_features, val_labels, max_pairs=int(config.get("compare_eval_pairs", 200000)))
        search_metrics = compute_search_metrics(val_features, val_labels)
        viz_paths = embed_visualizations(val_features, val_labels, ROOT / "outputs" / "features" / "embeddings", epoch)
        save_eval_metrics(ROOT / "outputs" / "features", epoch, compare_metrics, search_metrics, probe_recall)

        smoke_probe_history.append(float(probe_recall))
        smoke_tail = smoke_probe_history[-3:]
        smoke_rising = len(smoke_tail) == 3 and smoke_tail[0] < smoke_tail[1] < smoke_tail[2]

        writer.add_scalar("metric/probe_recall_at_1", probe_recall, epoch)
        writer.add_scalar("metric/compare_balanced_accuracy", compare_metrics["balanced_accuracy"], epoch)
        writer.add_scalar("metric/search_recall_at_1", search_metrics["recall_at_1"], epoch)
        writer.add_scalar("metric/search_recall_at_5", search_metrics["recall_at_5"], epoch)
        writer.add_scalar("metric/search_recall_at_10", search_metrics["recall_at_10"], epoch)
        writer.add_scalar("metric/search_map", search_metrics["mAP"], epoch)

        message = (
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={last_val_loss:.6f} "
            f"probe_recall_at_1={probe_recall:.6f} lr={current_lr:.8f} compare_acc={compare_metrics['accuracy']:.6f} "
            f"compare_bal_acc={compare_metrics['balanced_accuracy']:.6f} search_recall_at_1={search_metrics['recall_at_1']:.6f} "
            f"search_recall_at_5={search_metrics['recall_at_5']:.6f} search_recall_at_10={search_metrics['recall_at_10']:.6f} "
            f"search_map={search_metrics['mAP']:.6f}"
        )

        current_metric = search_metrics["recall_at_1"]
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                epoch,
                encoder,
                head,
                optimizer,
                scheduler,
                best_metric,
                best_epoch,
                no_improve_epochs,
                config,
                label_to_blood=label_to_blood_inv,
            )
            message += " best=1"
        else:
            no_improve_epochs += 1
            message += " best=0"
        save_checkpoint(
            last_path,
            epoch,
            encoder,
            head,
            optimizer,
            scheduler,
            best_metric,
            best_epoch,
            no_improve_epochs,
            config,
            label_to_blood=label_to_blood_inv,
        )
        if viz_paths:
            message += f" viz={viz_paths}"
        logger.info(message)

        if epoch <= warmup_epochs:
            continue

        if not args.skip_smoke_gate and epoch == smoke_gate_epoch:
            if probe_recall < smoke_min_recall and not smoke_rising:
                logger.error(
                    "smoke gate failed at epoch=%s probe_recall_at_1=%.6f threshold=%.6f rising=%s; stop training",
                    epoch,
                    probe_recall,
                    smoke_min_recall,
                    smoke_rising,
                )
                exit_code = 2
                break
            logger.info(
                "smoke gate check at epoch=%s probe_recall_at_1=%.6f threshold=%.6f rising=%s",
                epoch,
                probe_recall,
                smoke_min_recall,
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
        if "head_state" in best_state:
            head.load_state_dict(best_state["head_state"])
        final_eval_epoch = int(best_state.get("epoch", best_epoch or last_epoch))
    val_loader_eval = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    _ordered_ids, val_features, val_labels = extract_embeddings(encoder, val_loader_eval, device, desc="final val features")
    probe_recall = compute_probe_recall(val_features, val_labels, seed=int(config.get("seed", 42)))
    compare_metrics = compute_compare_metrics(val_features, val_labels, max_pairs=int(config.get("compare_eval_pairs", 200000)))
    search_metrics = compute_search_metrics(val_features, val_labels)
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
