"""SupCon (Supervised Contrastive) Loss for multi-label iris embedding learning.

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
Multi-label extension: positive pairs are defined by shared blood_ids.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(
    embeddings: torch.Tensor,
    blood_id_mask: torch.Tensor,  # (N, N) bool: True if images i,j share any blood_id
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised contrastive loss with multi-label positive mask.

    Args:
        embeddings: L2-normalized feature vectors (N, D)
        blood_id_mask: (N, N) bool tensor where True = shared blood_id(s)
        temperature: temperature scaling (default 0.07 from SupCon paper)

    Returns:
        scalar loss
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={tuple(embeddings.shape)}")

    batch_size = embeddings.size(0)
    if blood_id_mask.shape != (batch_size, batch_size):
        raise ValueError(
            f"blood_id_mask shape {tuple(blood_id_mask.shape)} != "
            f"expected ({batch_size}, {batch_size})"
        )

    # Normalize and compute similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = embeddings @ embeddings.T  # (N, N) cosine similarity
    sim = sim / temperature

    # Remove self from positives
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = blood_id_mask & ~eye

    # Check that every anchor has at least one positive
    has_positive = positive_mask.any(dim=1)
    if not has_positive.all():
        # Skip anchors without positives (shouldn't happen with good data)
        valid = has_positive
        positive_mask = positive_mask[valid][:, valid]
        sim = sim[valid][:, valid]
        eye = eye[valid][:, valid]
        batch_size = valid.sum().item()
        if batch_size < 2:
            return embeddings.sum() * 0.0  # zero gradient

    # For numerical stability: subtract max per row
    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - sim_max

    # exp of all similarities
    exp_sim = torch.exp(sim)
    exp_sim = exp_sim * ~eye  # exclude self

    # Denominator: sum of exp over all except self
    denom = exp_sim.sum(dim=1, keepdim=True)

    # Numerator: sum of exp over positives
    # Replace non-positives with 0 for the sum
    num = (exp_sim * positive_mask.float()).sum(dim=1, keepdim=True)

    # Avoid division by zero
    num = num + (num == 0).float() * 1e-8
    denom = denom + (denom == 0).float() * 1e-8

    # Loss per anchor
    loss_per_anchor = -torch.log(num / denom).squeeze()

    # Normalize by number of positives per anchor (to avoid bias towards many-positive anchors)
    n_positives = positive_mask.float().sum(dim=1).clamp(min=1)
    loss = (loss_per_anchor / n_positives).mean()

    return loss


def supcon_loss_with_weights(
    embeddings: torch.Tensor,
    blood_id_mask: torch.Tensor,  # (N, N) bool
    blood_id_sets_indices: list[list[int]],  # list of blood_id index sets per image
    temperature: float = 0.07,
) -> torch.Tensor:
    """SupCon loss with Similarity-Dissimilarity weighting (Track C).

    Each positive pair is weighted by:
      weight = |S ∩ T| / |S| * 1 / (1 + |T \ (S ∩ T)|)

    where S = anchor's blood_id set, T = sample's blood_id set.

    Args:
        embeddings: L2-normalized features (N, D)
        blood_id_mask: (N, N) bool: shared blood_id(s)
        blood_id_sets_indices: list of blood_id index lists per image
        temperature: temperature scaling

    Returns:
        scalar loss
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={tuple(embeddings.shape)}")

    batch_size = embeddings.size(0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    sim = embeddings @ embeddings.T / temperature

    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    positive_mask = blood_id_mask & ~eye

    has_positive = positive_mask.any(dim=1)
    if not has_positive.all():
        valid = has_positive
        positive_mask = positive_mask[valid][:, valid]
        sim = sim[valid][:, valid]
        eye = eye[valid][:, valid]
        blood_id_sets_indices = [blood_id_sets_indices[i] for i in range(batch_size) if valid[i]]
        batch_size = valid.sum().item()
        if batch_size < 2:
            return embeddings.sum() * 0.0

    # Compute SD weights
    weights = torch.zeros(batch_size, batch_size, device=embeddings.device)
    for i in range(batch_size):
        s_set = set(blood_id_sets_indices[i])
        s_len = len(s_set)
        if s_len == 0:
            continue
        for j in range(batch_size):
            if i == j or not positive_mask[i, j]:
                continue
            t_set = set(blood_id_sets_indices[j])
            intersection = len(s_set & t_set)
            extra = len(t_set - s_set)
            if intersection > 0:
                k_s = intersection / s_len          # similarity factor
                k_d = 1.0 / (1.0 + extra)           # dissimilarity factor
                weights[i, j] = k_s * k_d

    sim_max = sim.max(dim=1, keepdim=True).values.detach()
    sim = sim - sim_max
    exp_sim = torch.exp(sim) * ~eye

    denom = exp_sim.sum(dim=1, keepdim=True)
    weighted_num = (exp_sim * weights).sum(dim=1, keepdim=True)
    weighted_num = weighted_num + (weighted_num == 0).float() * 1e-8
    denom = denom + (denom == 0).float() * 1e-8

    n_positives = positive_mask.float().sum(dim=1).clamp(min=1)
    loss = (-torch.log(weighted_num / denom).squeeze() / n_positives).mean()
    return loss
