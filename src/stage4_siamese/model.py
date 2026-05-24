from __future__ import annotations

import math

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


class ArcMarginProduct(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.20) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(self.num_classes, self.feat_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels = labels.to(device=features.device, dtype=torch.long).view(-1, 1)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels, 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class SubCenterArcFace(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        subcenters: int = 3,
        scale: float = 30.0,
        margin: float = 0.20,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.subcenters = int(subcenters)
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(self.num_classes * self.subcenters, self.feat_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine = cosine_all.view(-1, self.num_classes, self.subcenters).max(dim=2).values
        if labels is None:
            return cosine * self.scale

        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0, max=1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels = labels.to(device=features.device, dtype=torch.long).view(-1, 1)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels, 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class SiameseNet(nn.Module):
    def __init__(self, encoder: IrisEncoder | None = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else IrisEncoder()

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_a = self.encoder(img_a)
        feat_b = self.encoder(img_b)
        return feat_a, feat_b
