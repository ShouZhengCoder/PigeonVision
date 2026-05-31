from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


BACKBONES = {"resnet18", "resnet34"}


def _build_resnet(backbone: str, pretrained: bool, in_channels: int) -> tuple[nn.Module, int]:
    if backbone == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif backbone == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}. Expected one of {sorted(BACKBONES)}")

    if in_channels != 3:
        raise ValueError("Stage 4 RGB training expects in_channels=3")

    in_features = model.fc.in_features
    model.fc = nn.Identity()
    return model, in_features


class IrisEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        backbone: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.backbone_name = backbone
        self.in_channels = int(in_channels)
        self.backbone, in_features = _build_resnet(backbone, pretrained=pretrained, in_channels=self.in_channels)
        self.embedding = nn.Sequential(
            nn.Linear(in_features, self.feat_dim, bias=False),
            nn.BatchNorm1d(self.feat_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)
