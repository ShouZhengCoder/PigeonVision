from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class IrisEncoder(nn.Module):
    def __init__(self, feat_dim: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        self.backbone = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class SiameseNet(nn.Module):
    def __init__(self, encoder: IrisEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else IrisEncoder()

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_a = self.encoder(img_a)
        feat_b = self.encoder(img_b)
        return feat_a, feat_b
