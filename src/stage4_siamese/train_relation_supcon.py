"""Train relation-aware weighted SupCon encoder."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import default_transform, split_meta_for_training
from dataset_relation import RelationBatchSampler, RelationDataset, collate_relation
from loss_relation import kinship_relevance_matrix, relation_relevance_matrix, weighted_supcon_loss

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from model import IrisEncoder
from relation_metrics import build_blood_id_idf, compute_cross_search_metrics_by_graded_blood_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relation-aware weighted SupCon training.")
    parser.add_argument("--relation-meta", type=Path, default=ROOT / "data" / "relation_meta.csv")
    parser.add_argument("--iris-dir", type=Path, default=ROOT / "outputs" / "iris_normalized")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints" / "siamese" / "relation_supcon")
    parser.add_argument("--warm-start", type=Path, default=ROOT / "checkpoints" / "siamese" / "best.pt")
    parser.add_argument("--kinship-source", choices=["idf", "pedigree"], default="idf",
                        help="Relevance source: idf (blood_id heuristic, baseline) or pedigree (Phase C hybrid k).")
    parser.add_argument("--kinship-vectors", type=Path, default=ROOT / "data" / "pedigree" / "contribution_vectors.csv",
                        help="Phase A contribution vectors (used when --kinship-source pedigree).")
    parser.add_argument("--fallback-scale", type=float, default=1.0,
                        help="Scale IDF fallback in hybrid kinship (0-1; <1 downweights non-structured pairs so pedigree k dominates).")
    parser.add_argument("--positive-cutoff", type=float, default=None,
                        help="Relevance below this treated as negative (only k>=cutoff are graded positives); preserves tier separation.")
    parser.add_argument("--select-by", choices=["idf", "kinship"], default="idf",
                        help="Val metric to select best.pt: idf (graded_nDCG) or kinship (pedigree-k Spearman, aligned with Phase B).")
    parser.add_argument("--feat-dim", type=int, default=256)
    parser.add_argument("--backbone", default="resnet34")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--negative-margin", type=float, default=None)
    parser.add_argument("--anchors-per-batch", type=int, default=32)
    parser.add_argument("--strong-pos-per-anchor", type=int, default=1)
    parser.add_argument("--weak-pos-per-anchor", type=int, default=1)
    parser.add_argument("--hard-neg-per-anchor", type=int, default=1)
    parser.add_argument("--strong-threshold", type=float, default=0.35)
    parser.add_argument("--batches-per-epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128, help="Eval extraction batch size.")
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
    logger = logging.getLogger("relation_supcon")
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


def torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def load_warm_start(encoder: IrisEncoder, path: Path, device: torch.device, logger: logging.Logger) -> None:
    if not path.exists():
        logger.info("warm-start checkpoint not found: %s", path)
        return
    state = torch_load(path, device)
    model_state = state.get("model_state", state) if isinstance(state, dict) else state
    missing, unexpected = encoder.load_state_dict(model_state, strict=False)
    logger.info("warm-start loaded from %s missing=%s unexpected=%s", path, len(missing), len(unexpected))


def pipe_set(value: object) -> frozenset[str]:
    return frozenset(part.strip() for part in str(value).split("|") if part.strip())


def blood_id_sets_from_rows(*frames: pd.DataFrame) -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for frame in frames:
        for row in frame[["img_id", "blood_ids"]].itertuples(index=False):
            out[str(row.img_id)] = pipe_set(row.blood_ids)
    return out


@torch.no_grad()
def extract_features(
    encoder: IrisEncoder,
    dataset: RelationDataset,
    indices: list[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[list[str], np.ndarray]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_relation,
    )
    ids: list[str] = []
    feats: list[np.ndarray] = []
    encoder.eval()
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        feat = encoder(images).detach().cpu().numpy().astype("float32")
        feats.append(feat)
        ids.extend([str(x) for x in batch["img_ids"]])
    if not feats:
        return ids, np.empty((0, encoder.feat_dim), dtype="float32")
    return ids, np.concatenate(feats, axis=0).astype("float32")


def save_checkpoint(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def pedigree_k_val_metric(feats, ids, kinship_fn, n_sample: int = 4000) -> dict:
    """Phase B 口径的 val 指标: Spearman(iris_dist, pedigree_k) + AUC(kin vs unrelated)。
    用于 --select-by kinship 时选 best.pt(对齐 Phase B 目标, 而非 IDF nDCG)。"""
    feats = np.asarray(feats, dtype=np.float32)
    feats = feats / np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    n = len(ids)
    if n < 2:
        return {"spearman": 0.0, "auc": 0.0}
    rng = np.random.default_rng(12345)
    m = min(n_sample, n * (n - 1) // 2)
    a = rng.integers(0, n, size=m)
    b = rng.integers(0, n, size=m)
    mask = a != b
    a, b = a[mask], b[mask]
    dist = np.linalg.norm(feats[a] - feats[b], axis=1)
    k = np.array([float(kinship_fn(str(ids[int(a[i])]), str(ids[int(b[i])]))) for i in range(len(a))])
    sp = 0.0
    if len(set(k.tolist())) > 1:
        s, _ = spearmanr(dist, k)
        sp = float(s) if s == s else 0.0
    label = (k > 0.1).astype(int)
    auc = 0.0
    if 0 < label.sum() < len(label):
        auc = float(roc_auc_score(label, -dist))
    return {"spearman": sp, "auc": auc}


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

    train_transform = default_transform(input_shape=(64, 512), train=True)
    eval_transform = default_transform(input_shape=(64, 512), train=False)

    train_ds = RelationDataset(
        relation_meta,
        resolve_root_path(args.iris_dir),
        split="train",
        transform=train_transform,
        limit=int(args.limit_train) or None,
        kinship_vectors=resolve_root_path(args.kinship_vectors) if args.kinship_source == "pedigree" else None,
        fallback_scale=float(args.fallback_scale) if args.kinship_source == "pedigree" else 1.0,
    )
    val_ds = RelationDataset(
        relation_meta,
        resolve_root_path(args.iris_dir),
        split="val",
        transform=eval_transform,
        limit=int(args.limit_val) or None,
    )
    eval_gallery_ds = RelationDataset(
        relation_meta,
        resolve_root_path(args.iris_dir),
        split="train",
        transform=eval_transform,
        limit=int(args.limit_train) or None,
    )

    logger.info("train=%s val=%s", len(train_ds), len(val_ds))

    blood_id_sets = blood_id_sets_from_rows(eval_gallery_ds.rows, val_ds.rows)
    idf = build_blood_id_idf(blood_id_sets)

    sampler = RelationBatchSampler(
        train_ds,
        anchors_per_batch=int(args.anchors_per_batch),
        strong_pos_per_anchor=int(args.strong_pos_per_anchor),
        weak_pos_per_anchor=int(args.weak_pos_per_anchor),
        hard_neg_per_anchor=int(args.hard_neg_per_anchor),
        strong_threshold=float(args.strong_threshold),
        batches_per_epoch=int(args.batches_per_epoch) or None,
        seed=int(args.seed),
        use_kinship=(args.kinship_source == "pedigree"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_relation,
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
        load_warm_start(encoder, resolve_root_path(args.warm_start), device, logger)

    rng = np.random.default_rng(int(args.seed))
    gallery_indices = list(range(len(train_ds)))
    if int(args.eval_gallery_size) > 0 and len(gallery_indices) > int(args.eval_gallery_size):
        gallery_indices = sorted(rng.choice(gallery_indices, size=int(args.eval_gallery_size), replace=False).tolist())
    eval_val_indices = list(range(len(val_ds)))

    config = vars(args).copy()
    config["device"] = str(device)
    logger.info("train=%s val=%s batches=%s device=%s", len(train_ds), len(val_ds), len(sampler), device)

    for epoch in range(start_epoch, int(args.epochs) + 1):
        sampler.set_epoch(epoch)
        encoder.train()
        total_loss = 0.0
        steps = 0
        skipped = 0
        for batch in tqdm(train_loader, desc=f"relation supcon {epoch}"):
            images = batch["images"].to(device, non_blocking=True)
            if args.kinship_source == "pedigree":
                relevance = kinship_relevance_matrix(batch["img_ids"], train_ds.kinship, device)
            else:
                relevance = relation_relevance_matrix(batch["blood_id_indices"], train_ds.idf_by_index, device)
            if not (relevance > 0.0).any():
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            embeddings = encoder(images)
            loss = weighted_supcon_loss(
                embeddings,
                relevance,
                temperature=float(args.temperature),
                negative_margin=args.negative_margin,
                positive_cutoff=args.positive_cutoff,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        metrics = {"graded_ndcg_at_10": 0.0, "avg_relevance_at_10": 0.0, "precision_at_10": 0.0, "mAP": 0.0}
        kin_val = {"spearman": 0.0, "auc": 0.0}
        if int(args.eval_every) > 0 and epoch % int(args.eval_every) == 0:
            gids, gfeats = extract_features(encoder, train_ds, gallery_indices, device, int(args.batch_size), int(args.num_workers))
            vids, vfeats = extract_features(encoder, val_ds, eval_val_indices, device, int(args.batch_size), int(args.num_workers))
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
                if args.select_by == "kinship" and args.kinship_source == "pedigree":
                    kin_val = pedigree_k_val_metric(vfeats, vids, train_ds.kinship)

        if args.select_by == "kinship" and args.kinship_source == "pedigree":
            primary = -float(kin_val["spearman"])  # Spearman 更负=更好 -> 取负作 primary(更高=更好)
            tie = float(kin_val["auc"])
        else:
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
            "epoch=%s loss=%.5f skipped=%s ndcg10=%.5f avg_rel10=%.5f p10=%.5f mAP=%.5f best=%s sampler=%s",
            epoch,
            avg_loss,
            skipped,
            float(metrics.get("graded_ndcg_at_10", 0.0)),
            float(metrics.get("avg_relevance_at_10", 0.0)),
            float(metrics.get("precision_at_10", 0.0)),
            float(metrics.get("mAP", 0.0)),
            int(is_best),
            json.dumps(sampler.last_epoch_stats, ensure_ascii=False),
        )

    logger.info("done best_graded_ndcg_at_10=%.6f best_avg_relevance_at_10=%.6f", best_primary, best_tie)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
