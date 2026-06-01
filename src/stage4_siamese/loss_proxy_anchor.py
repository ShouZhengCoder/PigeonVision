"""Proxy-Anchor Loss with multi-label support for image retrieval.

Reference: Kim et al., "Proxy Anchor Loss for Deep Metric Learning", CVPR 2020.
Multi-label extension: each sample can have multiple positive proxies (blood_names
reachable through its blood_ids).

Key advantage over pair-based losses (Triplet, SupCon):
  - Complexity O(C×B) instead of O(B²)
  - Uses ALL classes in every batch, not just the batch samples
  - Proven SOTA on SOP (22K classes), InShop (7K classes), CUB-200
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ProxyAnchorLoss(nn.Module):
    """Proxy-Anchor loss with multi-label support.

    Args:
        num_classes: number of blood_names (proxies)
        feat_dim: embedding dimension
        scale: scale factor α (inverse temperature), default 1/9 ≈ 0.111
        margin: proxy-anchor margin δ, default 0.1
    """

    def __init__(self, num_classes: int, feat_dim: int, scale: float = 1.0 / 9, margin: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feat_dim = int(feat_dim)
        self.scale = float(scale)
        self.margin = float(margin)
        self.proxies = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(
        self,
        embeddings: torch.Tensor,          # (B, D) L2-normalized
        positive_mask: torch.Tensor,        # (B, C) bool: True if sample i is positive for proxy j
    ) -> torch.Tensor:
        """Compute Proxy-Anchor loss.

        Args:
            embeddings: (B, D) L2-normalized feature vectors
            positive_mask: (B, C) bool, positive_mask[i,j]=True if sample i
                           shares any blood_id with proxy j's blood_name

        Returns:
            scalar loss
        """
        B, D = embeddings.shape
        C = self.num_classes
        device = embeddings.device

        if embeddings.size(0) != positive_mask.size(0):
            raise ValueError("embeddings and positive_mask batch size mismatch")
        if positive_mask.size(1) != C:
            raise ValueError(f"positive_mask has {positive_mask.size(1)} columns, expected {C}")

        # Normalize embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)  # (C, D)

        # Cosine similarity: (B, C)
        cos = embeddings @ proxies.T  # (B, C)

        # --- Positive term: pull samples towards their positive proxies ---
        # For each proxy, consider its positive samples
        # pos_mask: (B, C) bool
        pos_mask = positive_mask.to(device)

        # For each proxy with at least one positive sample:
        #   loss_pos_p = log(1 + Σ_{x∈pos} exp(-α * (cos(x,p) - δ)))
        pos_cos = cos * pos_mask.float()  # zero out non-positives
        pos_cos = pos_cos.masked_fill(~pos_mask, float("-inf"))

        # Only consider proxies that have at least one positive in this batch
        has_pos = pos_mask.any(dim=0)  # (C,) bool
        if has_pos.any():
            # For each proxy, gather similarities of its positive samples
            # Use max over positives as a stable approximation,
            # then sum exp(-α*(s-δ)) over all positives
            pos_term_per_proxy = torch.zeros(C, device=device)
            for j in range(C):
                if not has_pos[j]:
                    continue
                p_indices = pos_mask[:, j].nonzero(as_tuple=True)[0]
                p_sims = cos[p_indices, j]  # similarities of positive samples to proxy j
                pos_term_per_proxy[j] = torch.log(
                    1.0 + torch.sum(torch.exp(-self.scale * (p_sims - self.margin)))
                )
            pos_loss = pos_term_per_proxy[has_pos].mean()
        else:
            pos_loss = torch.tensor(0.0, device=device)

        # --- Negative term: push samples away from their negative proxies ---
        # For each proxy, consider its negative samples
        neg_mask = (~pos_mask).to(device)  # (B, C) bool

        # For each proxy with at least one negative sample:
        #   loss_neg_p = log(1 + Σ_{x∈neg} exp(α * (cos(x,p) + δ)))
        has_neg = neg_mask.any(dim=0)  # (C,) bool
        if has_neg.any():
            neg_term_per_proxy = torch.zeros(C, device=device)
            for j in range(C):
                if not has_neg[j]:
                    continue
                n_indices = neg_mask[:, j].nonzero(as_tuple=True)[0]
                n_sims = cos[n_indices, j]
                neg_term_per_proxy[j] = torch.log(
                    1.0 + torch.sum(torch.exp(self.scale * (n_sims + self.margin)))
                )
            neg_loss = neg_term_per_proxy[has_neg].mean()
        else:
            neg_loss = torch.tensor(0.0, device=device)

        return (pos_loss + neg_loss) * 0.5

    def get_proxies(self) -> torch.Tensor:
        """Return normalized proxies for evaluation/FAISS."""
        return F.normalize(self.proxies, p=2, dim=1)


def build_multi_label_positive_mask(
    batch_blood_id_indices: list[list[int]],
    blood_id_to_name_labels: dict[int, set[int]],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Build (B, C) positive mask from blood_id overlap.

    A sample is positive for proxy j if any of its blood_ids maps to
    a blood_name whose label is j.

    Args:
        batch_blood_id_indices: per-sample list of blood_id integer indices
        blood_id_to_name_labels: maps blood_id index -> set of blood_name labels
        num_classes: total number of blood_name classes (proxies)
        device: torch device

    Returns:
        (B, C) bool tensor
    """
    B = len(batch_blood_id_indices)
    mask = torch.zeros(B, num_classes, dtype=torch.bool, device=device)
    for i, bid_indices in enumerate(batch_blood_id_indices):
        name_labels = set()
        for bid in bid_indices:
            name_labels.update(blood_id_to_name_labels.get(bid, set()))
        for label in name_labels:
            if 0 <= label < num_classes:
                mask[i, label] = True
    return mask
