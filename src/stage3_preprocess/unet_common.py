from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


IMAGE_SIZE = 256
NORMALIZED_SHAPE = (64, 512)
INPUT_CHANNELS = 1
NUM_CLASSES = 3
BASE_CHANNELS = 32
GROUP_NORM_GROUPS = 8
DEFAULT_MASK_CONFIDENCE = 0.7
MIN_COMPONENT_AREA = 50
MIN_CONTOUR_POINTS = 5


def _resolve_num_groups(num_channels: int, requested_groups: int) -> int:
    groups = math.gcd(num_channels, requested_groups)
    return max(1, groups)


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def resize_gray_image(image_bgr: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    gray = _to_gray(image_bgr)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


def resize_mask(mask: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)


def gray_to_tensor(image_gray: np.ndarray) -> torch.Tensor:
    image = image_gray.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return torch.from_numpy(image).unsqueeze(0)


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.int64))


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = GROUP_NORM_GROUPS) -> None:
        super().__init__()
        g1 = _resolve_num_groups(out_channels, num_groups)
        g2 = _resolve_num_groups(out_channels, num_groups)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(g2, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = GROUP_NORM_GROUPS) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels, num_groups=num_groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = GROUP_NORM_GROUPS) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels, out_channels, num_groups=num_groups)

    def forward(self, x_low: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        x_low = self.up(x_low)
        diff_y = x_skip.size(2) - x_low.size(2)
        diff_x = x_skip.size(3) - x_low.size(3)
        if diff_y != 0 or diff_x != 0:
            x_low = F.pad(
                x_low,
                [
                    diff_x // 2,
                    diff_x - diff_x // 2,
                    diff_y // 2,
                    diff_y - diff_y // 2,
                ],
            )
        x = torch.cat([x_skip, x_low], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = INPUT_CHANNELS,
        num_classes: int = NUM_CLASSES,
        base_channels: int = BASE_CHANNELS,
        num_groups: int = GROUP_NORM_GROUPS,
    ) -> None:
        super().__init__()
        b = int(base_channels)
        self.inc = DoubleConv(in_channels, b, num_groups=num_groups)
        self.down1 = Down(b, b * 2, num_groups=num_groups)
        self.down2 = Down(b * 2, b * 4, num_groups=num_groups)
        self.down3 = Down(b * 4, b * 8, num_groups=num_groups)
        self.down4 = Down(b * 8, b * 16, num_groups=num_groups)
        self.up1 = Up(b * 16 + b * 8, b * 8, num_groups=num_groups)
        self.up2 = Up(b * 8 + b * 4, b * 4, num_groups=num_groups)
        self.up3 = Up(b * 4 + b * 2, b * 2, num_groups=num_groups)
        self.up4 = Up(b * 2 + b, b, num_groups=num_groups)
        self.outc = OutConv(b, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


@dataclass(frozen=True)
class EllipseSpec:
    cx: float
    cy: float
    a: float
    b: float
    angle_deg: float
    label: str

    def equivalent_radius(self) -> float:
        return (self.a + self.b) / 2.0

    def to_dict(self, prefix: str) -> dict[str, object]:
        return {
            f"{prefix}_cx": float(self.cx),
            f"{prefix}_cy": float(self.cy),
            f"{prefix}_a": float(self.a),
            f"{prefix}_b": float(self.b),
            f"{prefix}_angle": float(self.angle_deg),
        }


@dataclass(frozen=True)
class PredictionResult:
    success: bool
    status: str
    reason: str
    mask_confidence: float
    source_width: int
    source_height: int
    input_size: int
    cx: int
    cy: int
    r_inner: int
    r_outer: int
    pupil: EllipseSpec | None
    iris: EllipseSpec | None
    mask: np.ndarray | None = None

    def to_meta_row(self, img_id: str) -> dict[str, object]:
        row: dict[str, object] = {
            "img_id": img_id,
            "status": self.status,
            "reason": self.reason,
            "mask_confidence": round(float(self.mask_confidence), 6),
            "source_width": int(self.source_width),
            "source_height": int(self.source_height),
            "input_size": int(self.input_size),
            "cx": int(self.cx),
            "cy": int(self.cy),
            "r_inner": int(self.r_inner),
            "r_outer": int(self.r_outer),
        }
        if self.pupil is not None:
            row.update(self.pupil.to_dict("pupil"))
        else:
            row.update(
                {
                    "pupil_cx": -1,
                    "pupil_cy": -1,
                    "pupil_a": -1,
                    "pupil_b": -1,
                    "pupil_angle": -1,
                }
            )
        if self.iris is not None:
            row.update(self.iris.to_dict("iris"))
        else:
            row.update(
                {
                    "iris_cx": -1,
                    "iris_cy": -1,
                    "iris_a": -1,
                    "iris_b": -1,
                    "iris_angle": -1,
                }
            )
        return row


def _normalize_fit_ellipse(raw_ellipse: tuple[tuple[float, float], tuple[float, float], float], label: str) -> EllipseSpec:
    (cx, cy), (axis_1, axis_2), angle = raw_ellipse
    if axis_1 >= axis_2:
        a = float(axis_1) / 2.0
        b = float(axis_2) / 2.0
        angle_deg = float(angle)
    else:
        a = float(axis_2) / 2.0
        b = float(axis_1) / 2.0
        angle_deg = float(angle + 90.0)
    angle_deg = angle_deg % 180.0
    return EllipseSpec(float(cx), float(cy), a, b, angle_deg, label=label)


def largest_component_mask(binary_mask: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray | None:
    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)
    binary_mask = (binary_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return None
    best_label = -1
    best_area = 0
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label_idx
    if best_label < 0 or best_area < min_area:
        return None
    return (labels == best_label).astype(np.uint8)


def fit_ellipse_from_mask(binary_mask: np.ndarray, label: str, min_area: int = MIN_COMPONENT_AREA) -> EllipseSpec | None:
    component = largest_component_mask(binary_mask, min_area=min_area)
    if component is None:
        return None
    contours, _hierarchy = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < MIN_CONTOUR_POINTS:
        return None
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    try:
        raw = cv2.fitEllipse(contour)
    except cv2.error:
        return None
    return _normalize_fit_ellipse(raw, label=label)


def ellipse_radius_at_angle(ellipse: EllipseSpec, theta: np.ndarray) -> np.ndarray:
    angle = math.radians(float(ellipse.angle_deg))
    rel = theta - angle
    cos_rel = np.cos(rel)
    sin_rel = np.sin(rel)
    a = float(ellipse.a)
    b = float(ellipse.b)
    denom = np.sqrt((b * cos_rel) ** 2 + (a * sin_rel) ** 2)
    denom = np.where(denom <= 1e-6, 1e-6, denom)
    return (a * b) / denom


def normalize_iris_from_ellipses(
    image_gray: np.ndarray,
    pupil: EllipseSpec,
    iris: EllipseSpec,
    shape: tuple[int, int] = NORMALIZED_SHAPE,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    if image_gray.ndim == 3:
        image_gray = _to_gray(image_gray)
    if image_gray.dtype != np.uint8:
        image_gray = np.clip(image_gray, 0, 255).astype(np.uint8)

    if center is None:
        cx = (float(pupil.cx) + float(iris.cx)) / 2.0
        cy = (float(pupil.cy) + float(iris.cy)) / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])

    radial_steps, angular_steps = shape
    theta = np.linspace(0.0, 2.0 * np.pi, angular_steps, endpoint=False, dtype=np.float32)
    pupil_r = ellipse_radius_at_angle(pupil, theta)
    iris_r = ellipse_radius_at_angle(iris, theta)

    if np.any(iris_r <= pupil_r + 1e-3):
        raise ValueError("iris boundary must be outside pupil boundary")

    radial = np.linspace(0.0, 1.0, radial_steps, dtype=np.float32)[:, None]
    radii = pupil_r[None, :] + radial * (iris_r - pupil_r)[None, :]
    map_x = cx + radii * np.cos(theta)[None, :]
    map_y = cy + radii * np.sin(theta)[None, :]
    normalized = cv2.remap(
        image_gray,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return normalized


def compute_mask_confidence(probabilities: np.ndarray, predicted_mask: np.ndarray) -> float:
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (C, H, W)")
    if predicted_mask.shape != probabilities.shape[1:]:
        raise ValueError("predicted_mask shape mismatch")
    foreground = predicted_mask != 0
    if not np.any(foreground):
        return 0.0
    class_indices = predicted_mask[foreground]
    y_idx, x_idx = np.nonzero(foreground)
    selected = probabilities[class_indices, y_idx, x_idx]
    return float(np.mean(selected)) if selected.size else 0.0


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask == 1] = (0, 165, 255)
    canvas[mask == 2] = (0, 0, 255)
    return canvas


def overlay_mask(image_gray: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    if image_gray.ndim == 2:
        canvas = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        canvas = image_gray.copy()
    colors = colorize_mask(mask)
    overlay = canvas.astype(np.float32)
    overlay[mask > 0] = overlay[mask > 0] * (1.0 - alpha) + colors[mask > 0].astype(np.float32) * alpha
    return overlay.astype(np.uint8)


def ellipse_to_cv2(ellipse: EllipseSpec) -> tuple[tuple[float, float], tuple[float, float], float]:
    axes = (float(ellipse.a) * 2.0, float(ellipse.b) * 2.0)
    return (float(ellipse.cx), float(ellipse.cy)), axes, float(ellipse.angle_deg)


def build_unet(
    in_channels: int = INPUT_CHANNELS,
    num_classes: int = NUM_CLASSES,
    base_channels: int = BASE_CHANNELS,
    num_groups: int = GROUP_NORM_GROUPS,
) -> UNet:
    return UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        num_groups=num_groups,
    )

