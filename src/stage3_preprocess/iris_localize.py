from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from _common import ROOT, resolve_root_path
from unet_common import (
    DEFAULT_MASK_CONFIDENCE,
    IMAGE_SIZE,
    NORMALIZED_SHAPE,
    EllipseSpec,
    PredictionResult,
    build_unet,
    compute_mask_confidence,
    ellipse_to_cv2,
    fit_ellipse_from_mask,
    gray_to_tensor,
    normalize_iris_color_from_ellipses,
    normalize_iris_from_ellipses,
    overlay_mask,
    resize_gray_image,
)


DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "segmentation" / "best.pt"


class UNetPredictor:
    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str | torch.device | None = None,
        input_size: int = IMAGE_SIZE,
        in_channels: int = 1,
        num_classes: int = 3,
        base_channels: int = 32,
        num_groups: int = 8,
    ) -> None:
        self.checkpoint_path = resolve_root_path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        if self.in_channels != 1:
            raise ValueError("Stage 3 v1 uses grayscale 1-channel input only")
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.num_groups = int(num_groups)

        self.model = build_unet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
            num_groups=self.num_groups,
        ).to(self.device)
        self._load_weights()
        self.model.eval()

    def _load_weights(self) -> None:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Missing segmentation checkpoint: {self.checkpoint_path}")
        state = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(state, dict):
            for key in ("model_state", "state_dict", "model"):
                if key in state:
                    self.model.load_state_dict(state[key])
                    return
        if isinstance(state, dict):
            self.model.load_state_dict(state)
            return
        raise ValueError(f"Unsupported checkpoint format: {self.checkpoint_path}")

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray, mask_confidence_threshold: float = DEFAULT_MASK_CONFIDENCE) -> PredictionResult:
        source_height, source_width = image_bgr.shape[:2]
        image_gray = resize_gray_image(image_bgr, self.input_size)
        tensor = gray_to_tensor(image_gray).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred_mask = probs.argmax(axis=0).astype(np.uint8)
        mask_confidence = compute_mask_confidence(probs, pred_mask)

        pupil_ellipse: EllipseSpec | None = None
        iris_ellipse: EllipseSpec | None = None
        reason = "ok"

        pupil_mask = (pred_mask == 2).astype(np.uint8)
        iris_mask = (pred_mask == 1).astype(np.uint8)

        pupil_ellipse = fit_ellipse_from_mask(pupil_mask, label="pupil")
        if pupil_ellipse is None:
            reason = "pupil ellipse fit failed"

        iris_ellipse = fit_ellipse_from_mask(iris_mask, label="iris")
        if iris_ellipse is None and reason == "ok":
            reason = "iris ellipse fit failed"

        if pupil_ellipse is not None and iris_ellipse is not None:
            center_distance = float(
                ((pupil_ellipse.cx - iris_ellipse.cx) ** 2 + (pupil_ellipse.cy - iris_ellipse.cy) ** 2) ** 0.5
            )
            radius_scale = max(float(iris_ellipse.a), float(iris_ellipse.b), 1.0)
            if center_distance > max(20.0, 0.25 * radius_scale):
                reason = "ellipse centers too far apart"

            if iris_ellipse.equivalent_radius() <= pupil_ellipse.equivalent_radius():
                reason = "iris ellipse is not outside pupil ellipse"

        if mask_confidence < mask_confidence_threshold and reason == "ok":
            reason = f"mask confidence below threshold ({mask_confidence:.4f} < {mask_confidence_threshold:.4f})"

        success = reason == "ok" and pupil_ellipse is not None and iris_ellipse is not None
        if success:
            center_x = int(round((pupil_ellipse.cx + iris_ellipse.cx) / 2.0))
            center_y = int(round((pupil_ellipse.cy + iris_ellipse.cy) / 2.0))
            inner_radius = int(round(pupil_ellipse.equivalent_radius()))
            outer_radius = int(round(iris_ellipse.equivalent_radius()))
        else:
            center_x = center_y = inner_radius = outer_radius = -1

        return PredictionResult(
            success=success,
            status="success" if success else "failed",
            reason=reason,
            mask_confidence=float(mask_confidence),
            source_width=int(source_width),
            source_height=int(source_height),
            input_size=int(self.input_size),
            cx=center_x,
            cy=center_y,
            r_inner=inner_radius,
            r_outer=outer_radius,
            pupil=pupil_ellipse,
            iris=iris_ellipse,
            mask=pred_mask,
        )


@lru_cache(maxsize=4)
def get_default_predictor(
    checkpoint_path: str = str(DEFAULT_CHECKPOINT),
    device: str | None = None,
    input_size: int = IMAGE_SIZE,
    in_channels: int = 1,
    num_classes: int = 3,
    base_channels: int = 32,
    num_groups: int = 8,
) -> UNetPredictor:
    return UNetPredictor(
        checkpoint_path=Path(checkpoint_path),
        device=device,
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        num_groups=num_groups,
    )


def localize_iris(
    img_bgr: np.ndarray,
    predictor: UNetPredictor | None = None,
    mask_confidence_threshold: float = DEFAULT_MASK_CONFIDENCE,
) -> PredictionResult:
    predictor = predictor or get_default_predictor()
    return predictor.predict(img_bgr, mask_confidence_threshold=mask_confidence_threshold)


def normalize_iris(
    img_bgr: np.ndarray,
    prediction_or_pupil: PredictionResult | EllipseSpec,
    iris: EllipseSpec | None = None,
    shape: tuple[int, int] = NORMALIZED_SHAPE,
) -> np.ndarray:
    if isinstance(prediction_or_pupil, PredictionResult):
        prediction = prediction_or_pupil
        if not prediction.success or prediction.pupil is None or prediction.iris is None:
            raise ValueError(prediction.reason)
        pupil = prediction.pupil
        iris = prediction.iris
        input_size = prediction.input_size
    else:
        pupil = prediction_or_pupil
        if iris is None:
            raise ValueError("iris ellipse is required")
        input_size = IMAGE_SIZE

    image_gray = resize_gray_image(img_bgr, input_size)
    return normalize_iris_from_ellipses(image_gray, pupil, iris, shape=shape)


def daugman_normalize_color(
    img_bgr: np.ndarray,
    prediction_or_pupil: PredictionResult | EllipseSpec,
    iris: EllipseSpec | None = None,
    shape: tuple[int, int] = NORMALIZED_SHAPE,
) -> np.ndarray:
    if isinstance(prediction_or_pupil, PredictionResult):
        prediction = prediction_or_pupil
        if not prediction.success or prediction.pupil is None or prediction.iris is None:
            raise ValueError(prediction.reason)
        pupil = prediction.pupil
        iris = prediction.iris
        input_size = prediction.input_size
    else:
        pupil = prediction_or_pupil
        if iris is None:
            raise ValueError("iris ellipse is required")
        input_size = IMAGE_SIZE

    if img_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = img_bgr
    image_bgr = cv2.resize(image_bgr, (int(input_size), int(input_size)), interpolation=cv2.INTER_AREA)
    return normalize_iris_color_from_ellipses(image_bgr, pupil, iris, shape=shape)


def extract_iris_region(
    img_bgr: np.ndarray,
    prediction_or_iris: PredictionResult | EllipseSpec,
    input_size: int = IMAGE_SIZE,
) -> np.ndarray:
    if isinstance(prediction_or_iris, PredictionResult):
        prediction = prediction_or_iris
        if not prediction.success or prediction.iris is None:
            raise ValueError(prediction.reason)
        iris = prediction.iris
        input_size = prediction.input_size
    else:
        iris = prediction_or_iris

    if img_bgr.ndim == 2:
        image_gray = img_bgr
    else:
        image_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask_small = np.zeros((int(input_size), int(input_size)), dtype=np.uint8)
    cv2.ellipse(mask_small, ellipse_to_cv2(iris), 255, thickness=-1)
    mask = cv2.resize(
        mask_small,
        (int(image_gray.shape[1]), int(image_gray.shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )

    extracted = np.zeros_like(image_gray)
    extracted[mask > 0] = image_gray[mask > 0]
    return extracted


def visualize_localization(
    img_bgr: np.ndarray,
    prediction: PredictionResult,
) -> np.ndarray:
    image_gray = resize_gray_image(img_bgr, prediction.input_size)
    if prediction.mask is None:
        canvas = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        canvas = overlay_mask(image_gray, prediction.mask)

    if prediction.pupil is not None:
        cv2.ellipse(canvas, ellipse_to_cv2(prediction.pupil), (0, 255, 0), 2)
    if prediction.iris is not None:
        cv2.ellipse(canvas, ellipse_to_cv2(prediction.iris), (0, 0, 255), 2)

    text = f"{prediction.status} conf={prediction.mask_confidence:.3f}"
    if prediction.reason and prediction.reason != "ok":
        text = f"{text} {prediction.reason}"
    cv2.putText(canvas, text[:80], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    return canvas
