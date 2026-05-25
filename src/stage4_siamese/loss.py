from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    label: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    label = label.to(device=feat_a.device, dtype=feat_a.dtype).view(-1)
    distance = torch.norm(feat_a - feat_b, dim=1)
    positive = label * distance.pow(2)
    negative = (1.0 - label) * torch.clamp(margin - distance, min=0.0).pow(2)
    return (positive + negative).mean()


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    blood_ids: torch.Tensor,
    blood_names: torch.Tensor,
    margin: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch-hard triplet loss for pigeon iris embedding learning.

    Positive samples are images sharing the same `blood_id`.
    Negative samples are images with a different `blood_name`.
    Samples with the same `blood_name` but different `blood_id` are ignored as negatives.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape={tuple(embeddings.shape)}")

    blood_ids = blood_ids.to(device=embeddings.device).view(-1)
    blood_names = blood_names.to(device=embeddings.device).view(-1)
    if embeddings.size(0) != blood_ids.numel() or embeddings.size(0) != blood_names.numel():
        raise ValueError("embeddings, blood_ids, and blood_names must have the same batch size")

    embeddings = F.normalize(embeddings, p=2, dim=1)
    distances = torch.cdist(embeddings, embeddings, p=2)
    batch_size = embeddings.size(0)
    eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

    positive_mask = (blood_ids[:, None] == blood_ids[None, :]) & ~eye
    negative_mask = blood_names[:, None] != blood_names[None, :]
    valid_anchor = positive_mask.any(dim=1) & negative_mask.any(dim=1)

    if not torch.any(valid_anchor):
        zero = embeddings.sum() * 0.0
        return zero, zero.detach(), zero.detach()

    pos_dist = distances.masked_fill(~positive_mask, -1.0).max(dim=1).values
    neg_dist = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values
    valid_pos = pos_dist[valid_anchor]
    valid_neg = neg_dist[valid_anchor]
    loss = torch.clamp(valid_pos - valid_neg + float(margin), min=0.0).mean()
    return loss, valid_pos.detach().mean(), valid_neg.detach().mean()
