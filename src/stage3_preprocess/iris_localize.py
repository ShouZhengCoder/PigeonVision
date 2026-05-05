from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_DP = 1
HOUGH_MIN_DIST = 20
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
R_INNER_RANGE = (0.10, 0.25)
R_OUTER_RANGE = (0.30, 0.50)
DETECT_MAX_SIDE = 256


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


def _to_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def _blur_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(_to_gray(img_bgr), (5, 5), 0)


def _estimate_center(gray_blur: np.ndarray) -> tuple[int, int]:
    vertical = gray_blur.mean(axis=0)
    horizontal = gray_blur.mean(axis=1)
    return int(np.argmin(vertical)), int(np.argmin(horizontal))


def _downscale_for_detection(gray: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = gray.shape[:2]
    max_side = max(height, width)
    if max_side <= DETECT_MAX_SIDE:
        return gray, 1.0
    scale = DETECT_MAX_SIDE / float(max_side)
    resized = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _roi_bounds(width: int, height: int, cx: int, cy: int, radius: int) -> tuple[int, int, int, int]:
    radius = max(1, radius)
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius)
    y2 = min(height, cy + radius)
    return x1, y1, x2, y2


def _circles_from_hough(img: np.ndarray, r_min: int, r_max: int) -> list[Circle]:
    if r_min <= 0 or r_max <= 0 or r_min >= r_max:
        return []
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MIN_DIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=int(r_min),
        maxRadius=int(r_max),
    )
    if circles is None:
        return []
    return [Circle(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]


def _detect_circle(gray: np.ndarray, center_hint: tuple[int, int], r_min: int, r_max: int) -> Circle | None:
    height, width = gray.shape[:2]
    cx_hint, cy_hint = center_hint
    roi_radius = int(max(r_max * 1.6, min(width, height) * 0.45))
    x1, y1, x2, y2 = _roi_bounds(width, height, cx_hint, cy_hint, roi_radius)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    edges = cv2.Canny(roi, CANNY_LOW, CANNY_HIGH)
    circles = _circles_from_hough(edges, r_min, r_max)
    if not circles:
        circles = _circles_from_hough(roi, r_min, r_max)
    if not circles:
        return None
    best = min(
        circles,
        key=lambda c: ((c.cx + x1) - cx_hint) ** 2 + ((c.cy + y1) - cy_hint) ** 2,
    )
    return Circle(best.cx + x1, best.cy + y1, best.r)


def _fallback_circle(center_hint: tuple[int, int], width: int, kind: str) -> Circle:
    cx, cy = center_hint
    radius = int(round(width * (0.18 if kind == "inner" else 0.40)))
    return Circle(float(cx), float(cy), float(radius))


def localize_iris(img_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray_blur = _blur_gray(img_bgr)
    height, width = gray_blur.shape[:2]
    if min(height, width) < 20:
        return None

    detect_gray, scale = _downscale_for_detection(gray_blur)
    cx_hint, cy_hint = _estimate_center(detect_gray)
    scaled_width = detect_gray.shape[1]
    inner_min = max(1, int(round(scaled_width * R_INNER_RANGE[0])))
    inner_max = max(inner_min + 1, int(round(scaled_width * R_INNER_RANGE[1])))
    outer_min = max(inner_max + 1, int(round(scaled_width * R_OUTER_RANGE[0])))
    outer_max = max(outer_min + 1, int(round(scaled_width * R_OUTER_RANGE[1])))

    inner = _detect_circle(detect_gray, (cx_hint, cy_hint), inner_min, inner_max)
    outer = _detect_circle(detect_gray, (cx_hint, cy_hint), outer_min, outer_max)
    if inner is None:
        inner = _fallback_circle((cx_hint, cy_hint), scaled_width, "inner")
    if outer is None:
        outer = _fallback_circle((cx_hint, cy_hint), scaled_width, "outer")

    if scale != 1.0:
        inv = 1.0 / scale
        inner = Circle(inner.cx * inv, inner.cy * inv, inner.r * inv)
        outer = Circle(outer.cx * inv, outer.cy * inv, outer.r * inv)

    cx = int(round((inner.cx + outer.cx) / 2.0))
    cy = int(round((inner.cy + outer.cy) / 2.0))
    r_inner = int(round(inner.r))
    r_outer = int(round(outer.r))
    if r_inner <= 0 or r_outer <= 0 or r_outer <= r_inner:
        return None
    return cx, cy, r_inner, r_outer


def normalize_iris(
    img_bgr: np.ndarray,
    cx: int,
    cy: int,
    r_inner: int,
    r_outer: int,
    shape: tuple[int, int] = (64, 512),
) -> np.ndarray:
    gray = _to_gray(img_bgr)
    if r_outer <= r_inner:
        raise ValueError("r_outer must be greater than r_inner")

    radial_steps, angular_steps = shape
    theta = np.linspace(0.0, 2.0 * np.pi, angular_steps, endpoint=False, dtype=np.float32)
    radial = np.linspace(0.0, 1.0, radial_steps, dtype=np.float32)[:, None]
    radii = (float(r_inner) + radial * float(r_outer - r_inner)).astype(np.float32)

    cos_theta = np.cos(theta).astype(np.float32)[None, :]
    sin_theta = np.sin(theta).astype(np.float32)[None, :]

    map_x = cx + radii * cos_theta
    map_y = cy + radii * sin_theta
    return cv2.remap(
        gray,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def visualize_localization(
    img_bgr: np.ndarray,
    cx: int,
    cy: int,
    r_inner: int,
    r_outer: int,
) -> np.ndarray:
    if img_bgr.ndim == 2:
        canvas = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        canvas = img_bgr.copy()
    cv2.circle(canvas, (int(cx), int(cy)), int(r_inner), (0, 255, 0), 2)
    cv2.circle(canvas, (int(cx), int(cy)), int(r_outer), (0, 0, 255), 2)
    cv2.circle(canvas, (int(cx), int(cy)), 2, (255, 0, 0), -1)
    return canvas

