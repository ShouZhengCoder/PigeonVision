"""Pairwise relation ranking fine-tune after relation SupCon pretraining."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform
from dataset_relation import RelationBatchSampler, RelationDataset
from loss_relation import pairwise_relation_ranking_loss
from model import IrisEncoder
from relation_metrics import build_blood_id_idf, compute_cross_search_metrics_by_graded_blood_ids
from train_relation_supcon import blood_id_sets_from_rows, extract_features, torch_load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relation ranking fine-tune.")
    parser.add_argument("--relation-meta", type=Path, default=ROOT / "data" / "relation_meta.csv")
    parser.add_argument("--iris-dir", type=Path, default=ROOT / "outputs" / "iris_normalized")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "siamese" / "relation_ranker")
    parser.add_argument("--warm-start", type=Path, default=ROOT / "checkpoints" / "siamese" / "relation_supcon" / "best.pt")
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--triplets-per-epoch", type=int, default=8192)
    parser.add_argument("--margin-sp", type=float, default=0.08)
    parser.add_argument("--margin-wn", type=float, default=0.15)
    parser.add_argument("--strong-threshold", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--eval-gallery-size", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("relation_ranker")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_model_state(encoder: IrisEncoder, path: Path, device: torch.device, logger: logging.Logger) -> None:
    if not path.exists():
        raise FileNotFoundError(f"warm-start checkpoint not found: {path}")
    state = torch_load(path, device)
    model_state = state.get("model_state", state) if isinstance(state, dict) else state
    missing, unexpected = encoder.load_state_dict(model_state, strict=False)
    logger.info("warm-start loaded from %s missing=%s unexpected=%s", path, len(missing), len(unexpected))


def image_batch(dataset: RelationDataset, indices: list[int], device: torch.device) -> torch.Tensor:
    return torch.stack([dataset[int(index)].image for index in indices]).to(device, non_blocking=True)


def save_checkpoint(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    output_dir = ensure_dir(resolve_root_path(args.output_dir))
    logger = setup_logger(output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    relation_meta = resolve_root_path(args.relation_meta)
    if not relation_meta.exists():
        raise FileNotFoundError(f"relation_meta not found: {relation_meta}. Run src/stage1_data/build_relation_meta.py first.")

    train_ds = RelationDataset(
        relation_meta,
        resolve_root_path(args.iris_dir),
        split="train",
        transform=default_transform(input_shape=(64, 512), train=True),
        limit=int(args.limit_train) or None,
    )
    eval_gallery_ds = RelationDataset(
        relation_meta,
        resolve_root_path(args.iris_dir),
        split="train",
        transform=default_transform(input_shape=(64, 512), train=False),
        limit=int(args.limit_train) or None,
    )
    val_ds = RelationDataset(
        relation_meta,
        resolve_root_path(args.iris_dir),
        split="val",
        transform=default_transform(input_shape=(64, 512), train=False),
        limit=int(args.limit_val) or None,
    )
    blood_id_sets = blood_id_sets_from_rows(eval_gallery_ds.rows, val_ds.rows)
    idf = build_blood_id_idf(blood_id_sets)
    sampler = RelationBatchSampler(
        train_ds,
        anchors_per_batch=1,
        strong_threshold=float(args.strong_threshold),
        batches_per_epoch=1,
        seed=int(args.seed),
    )

    encoder = IrisEncoder(feat_dim=int(args.feat_dim), backbone=str(args.backbone), pretrained=False, in_channels=3).to(device)
    optimizer = Adam(encoder.parameters(), lr=float(args.lr))
    start_epoch = 1
    best_primary = -1.0
    best_tie = -1.0
    last_path = output_dir / "last.pt"
    if args.resume and last_path.exists():
        state = torch_load(last_path, device)
        encoder.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_primary = float(state.get("best_primary", -1.0))
        best_tie = float(state.get("best_tie", -1.0))
        logger.info("resumed from %s epoch=%s", last_path, start_epoch)
    else:
        load_model_state(encoder, resolve_root_path(args.warm_start), device, logger)

    rng = np.random.default_rng(int(args.seed))
    py_rng = __import__("random").Random(int(args.seed))
    gallery_indices = list(range(len(eval_gallery_ds)))
    if int(args.eval_gallery_size) > 0 and len(gallery_indices) > int(args.eval_gallery_size):
        gallery_indices = sorted(rng.choice(gallery_indices, size=int(args.eval_gallery_size), replace=False).tolist())
    val_indices = list(range(len(val_ds)))
    config = vars(args).copy()
    config["device"] = str(device)
    logger.info("train=%s val=%s device=%s", len(train_ds), len(val_ds), device)

    for epoch in range(start_epoch, int(args.epochs) + 1):
        encoder.train()
        anchors = rng.choice(
            sampler.positive_anchors,
            size=int(args.triplets_per_epoch),
            replace=len(sampler.positive_anchors) < int(args.triplets_per_epoch),
            p=sampler.anchor_weights,
        ).astype(int).tolist()
        triplets: list[tuple[int, int, int, int]] = []
        for anchor in anchors:
            strong, weak, neg = sampler.sample_tuple(anchor, py_rng)
            if strong is None or weak is None or neg is None:
                continue
            if strong == weak:
                continue
            triplets.append((int(anchor), int(strong), int(weak), int(neg)))

        total_loss = 0.0
        steps = 0
        for start in tqdm(range(0, len(triplets), int(args.batch_size)), desc=f"relation rank {epoch}"):
            batch = triplets[start : start + int(args.batch_size)]
            if not batch:
                continue
            a_idx, s_idx, w_idx, n_idx = [list(values) for values in zip(*batch)]
            optimizer.zero_grad(set_to_none=True)
            a = encoder(image_batch(train_ds, a_idx, device))
            s = encoder(image_batch(train_ds, s_idx, device))
            w = encoder(image_batch(train_ds, w_idx, device))
            n = encoder(image_batch(train_ds, n_idx, device))
            loss = pairwise_relation_ranking_loss(a, s, w, n, margin_sp=float(args.margin_sp), margin_wn=float(args.margin_wn))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        metrics = {"graded_ndcg_at_10": 0.0, "avg_relevance_at_10": 0.0, "precision_at_10": 0.0, "mAP": 0.0}
        if int(args.eval_every) > 0 and epoch % int(args.eval_every) == 0:
            gids, gfeats = extract_features(encoder, eval_gallery_ds, gallery_indices, device, int(args.batch_size), int(args.num_workers))
            vids, vfeats = extract_features(encoder, val_ds, val_indices, device, int(args.batch_size), int(args.num_workers))
            if len(gfeats) and len(vfeats):
                metrics = compute_cross_search_metrics_by_graded_blood_ids(
                    vfeats,
                    vids,
                    gfeats,
                    gids,
                    blood_id_sets,
                    idf=idf,
                    exclude_self=False,
                )

        primary = float(metrics.get("graded_ndcg_at_10", 0.0))
        tie = float(metrics.get("avg_relevance_at_10", 0.0))
        is_best = primary > best_primary or (primary == best_primary and tie > best_tie)
        if is_best:
            best_primary = primary
            best_tie = tie
            save_checkpoint(
                output_dir / "best.pt",
                {
                    "epoch": epoch,
                    "model_state": encoder.state_dict(),
                    "best_primary": best_primary,
                    "best_tie": best_tie,
                    "config": config,
                    "metrics": metrics,
                },
            )
        save_checkpoint(
            last_path,
            {
                "epoch": epoch,
                "model_state": encoder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_primary": best_primary,
                "best_tie": best_tie,
                "config": config,
                "metrics": metrics,
            },
        )
        logger.info(
            "epoch=%s loss=%.5f triplets=%s ndcg10=%.5f avg_rel10=%.5f p10=%.5f mAP=%.5f best=%s",
            epoch,
            avg_loss,
            len(triplets),
            float(metrics.get("graded_ndcg_at_10", 0.0)),
            float(metrics.get("avg_relevance_at_10", 0.0)),
            float(metrics.get("precision_at_10", 0.0)),
            float(metrics.get("mAP", 0.0)),
            int(is_best),
        )
    logger.info("done best_graded_ndcg_at_10=%.6f best_avg_relevance_at_10=%.6f", best_primary, best_tie)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
