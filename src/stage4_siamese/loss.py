from __future__ import annotations

import torch


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
