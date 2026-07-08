"""Relation-aware contrastive and ranking losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def relation_relevance_matrix(
    batch_blood_id_indices: list[list[int]],
    idf_by_index: dict[int, float],
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(batch_blood_id_indices)
    out = torch.zeros(batch_size, batch_size, dtype=torch.float32, device=device)
    sets = [set(int(v) for v in values) for values in batch_blood_id_indices]
    weights = [sum(float(idf_by_index.get(int(v), 0.0)) for v in values) for values in sets]
    for i in range(batch_size):
        left = sets[i]
        if not left:
            continue
        for j in range(i + 1, batch_size):
            right = sets[j]
            shared = left & right
            if not shared:
                continue
            shared_w = sum(float(idf_by_index.get(int(v), 0.0)) for v in shared)
            union_w = sum(float(idf_by_index.get(int(v), 0.0)) for v in (left | right))
            min_w = min(weights[i], weights[j])
            if shared_w <= 0.0 or union_w <= 0.0 or min_w <= 0.0:
                continue
            relevance = 0.7 * (shared_w / min_w) + 0.3 * (shared_w / union_w)
            out[i, j] = float(relevance)
            out[j, i] = float(relevance)
    return out.clamp_(0.0, 1.0)


def kinship_relevance_matrix(
    batch_img_ids: list[str],
    kinship_fn,
    device: torch.device,
) -> torch.Tensor:
    """Graded relevance matrix from a pedigree-kinship function (Phase C).

    kinship_fn(a_img_id, b_img_id) -> float in [0, 1] (hybrid: pedigree dot
    where both structured, IDF heuristic fallback otherwise). Symmetric;
    zero means unrelated. Replaces relation_relevance_matrix when
    --kinship-source pedigree, feeding the same weighted_supcon_loss.
    """
    batch_size = len(batch_img_ids)
    out = torch.zeros(batch_size, batch_size, dtype=torch.float32, device=device)
    for i in range(batch_size):
        a = batch_img_ids[i]
        for j in range(i + 1, batch_size):
            k = float(kinship_fn(a, batch_img_ids[j]))
            if k > 0.0:
                out[i, j] = k
                out[j, i] = k
    return out.clamp_(0.0, 1.0)


def weighted_supcon_loss(
    embeddings: torch.Tensor,
    relevance: torch.Tensor,
    temperature: float = 0.07,
    negative_margin: float | None = None,
) -> torch.Tensor:
    """Supervised contrastive loss weighted by graded relation relevance.

    Positive pairs contribute proportionally to relevance. Clear negatives are
    pairs with zero relevance. Weakly related pairs are positives, not negatives.
    If negative_margin is supplied, negatives use max(0, margin - relevance).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {tuple(embeddings.shape)}")
    batch_size = embeddings.size(0)
    if relevance.shape != (batch_size, batch_size):
        raise ValueError(f"relevance shape {tuple(relevance.shape)} != ({batch_size}, {batch_size})")

    embeddings = F.normalize(embeddings, p=2, dim=1)
    relevance = relevance.to(device=embeddings.device, dtype=torch.float32).clamp(0.0, 1.0)
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    relevance = relevance.masked_fill(eye, 0.0)
    has_positive = (relevance > 0.0).any(dim=1)
    if not has_positive.any():
        return embeddings.sum() * 0.0

    sim = (embeddings @ embeddings.T) / float(temperature)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim).masked_fill(eye, 0.0)

    positive_weight = relevance
    if negative_margin is None:
        negative_weight = ((relevance <= 0.0) & ~eye).to(torch.float32)
    else:
        negative_weight = torch.clamp(float(negative_margin) - relevance, min=0.0).masked_fill(eye, 0.0)

    numerator = (exp_sim * positive_weight).sum(dim=1)
    denominator = (exp_sim * (positive_weight + negative_weight)).sum(dim=1)
    valid = has_positive & (numerator > 0.0) & (denominator > 0.0)
    if not valid.any():
        return embeddings.sum() * 0.0
    positive_mass = positive_weight.sum(dim=1).clamp(min=1e-8)
    loss = -torch.log((numerator[valid] + 1e-8) / (denominator[valid] + 1e-8))
    loss = loss / positive_mass[valid]
    return loss.mean()


def pairwise_relation_ranking_loss(
    anchor: torch.Tensor,
    strong: torch.Tensor,
    weak: torch.Tensor,
    negative: torch.Tensor,
    margin_sp: float = 0.08,
    margin_wn: float = 0.15,
) -> torch.Tensor:
    anchor = F.normalize(anchor, p=2, dim=1)
    strong = F.normalize(strong, p=2, dim=1)
    weak = F.normalize(weak, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    d_sp = torch.linalg.vector_norm(anchor - strong, dim=1)
    d_wp = torch.linalg.vector_norm(anchor - weak, dim=1)
    d_neg = torch.linalg.vector_norm(anchor - negative, dim=1)
    return (F.relu(d_sp + float(margin_sp) - d_wp) + F.relu(d_wp + float(margin_wn) - d_neg)).mean()
