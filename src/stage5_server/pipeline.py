from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
STAGE3_DIR = ROOT / "src" / "stage3_preprocess"
STAGE4_DIR = ROOT / "src" / "stage4_siamese"
for import_dir in (STAGE4_DIR, STAGE3_DIR):
    import_path = str(import_dir)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from iris_localize import UNetPredictor, daugman_normalize_color  # noqa: E402
from model import IrisEncoder  # noqa: E402
from unet_common import DEFAULT_MASK_CONFIDENCE, NORMALIZED_SHAPE, IrisSegmentationSuccess  # noqa: E402


# Default fusion config: (checkpoint, feat_dim, backbone)
FUSION_ENCODERS = [
    (ROOT / "checkpoints" / "siamese" / "best.pt", 256, "resnet34"),
    (ROOT / "checkpoints" / "siamese" / "supcon" / "best.pt", 256, "resnet34"),
    (ROOT / "checkpoints" / "siamese" / "arcface_v2" / "best.pt", 512, "resnet50"),
]
FUSION_DEFAULT_DIR = ROOT / "outputs" / "features" / "fusion_1024d_full"
DEFAULT_SEARCH_TOP_K = 20
MAX_SEARCH_TOP_K = 100


class IrisPipeline:
    def __init__(
        self,
        siamese_checkpoint: str | Path = ROOT / "checkpoints" / "siamese" / "best.pt",
        detection_checkpoint: str | Path = ROOT / "checkpoints" / "detection" / "exp" / "weights" / "best.pt",
        segmentation_checkpoint: str | Path = ROOT / "checkpoints" / "segmentation" / "best.pt",
        faiss_index_path: str | Path = ROOT / "outputs" / "features" / "faiss_index.bin",
        feature_meta_path: str | Path = ROOT / "outputs" / "features" / "feature_db_meta.csv",
        threshold_path: str | Path = ROOT / "outputs" / "features" / "threshold.json",
        siamese_config_path: str | Path = ROOT / "configs" / "siamese.yaml",
        unet_config_path: str | Path = ROOT / "configs" / "unet.yaml",
        img_index_path: str | Path = ROOT / "outputs" / "img_index.csv",
        detection_confidence: float = 0.7,
        detection_expand_ratio: float = 0.1,
        device: str | torch.device | None = None,
        fusion: bool = True,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fusion = bool(fusion)
        self.siamese_checkpoint = self._resolve(siamese_checkpoint)
        self.detection_checkpoint = self._resolve(detection_checkpoint)
        self.segmentation_checkpoint = self._resolve(segmentation_checkpoint)
        self.siamese_config_path = self._resolve(siamese_config_path)
        self.unet_config_path = self._resolve(unet_config_path)
        self.img_index_path = self._resolve(img_index_path)

        self.siamese_config = self._load_yaml(self.siamese_config_path)
        self.unet_config = self._load_yaml(self.unet_config_path)
        self.detection_confidence = float(detection_confidence)
        self.detection_expand_ratio = float(detection_expand_ratio)

        # In fusion mode, the service must use the matching 1024d gallery.
        if self.fusion:
            faiss_index_path = FUSION_DEFAULT_DIR / "faiss_index.bin"
            feature_meta_path = FUSION_DEFAULT_DIR / "feature_db_meta.csv"
            # Use fusion-specific threshold if available, fall back to global threshold.
            fusion_threshold = FUSION_DEFAULT_DIR / "threshold.json"
            if fusion_threshold.exists():
                threshold_path = fusion_threshold
            else:
                threshold_path = ROOT / "outputs" / "features" / "threshold.json"

        self.faiss_index_path = self._resolve(faiss_index_path)
        self.feature_meta_path = self._resolve(feature_meta_path)
        self.threshold_path = self._resolve(threshold_path)
        self.threshold = self._load_threshold(self.threshold_path)
        self.detector = self._load_detector()
        self.encoders = self._load_encoders()
        self.segmenter = self._load_segmenter()
        self.index = self._load_faiss_index()
        self.meta = self._load_feature_meta()
        self.img_index = self._load_img_index()
        self._validate_index_meta()

        self.input_shape = tuple(int(v) for v in self.siamese_config.get("input_shape", NORMALIZED_SHAPE))
        if len(self.input_shape) != 2:
            raise ValueError(f"Invalid input_shape in {self.siamese_config_path}: {self.input_shape}")
        self.normalize_mean = np.asarray(self.siamese_config.get("normalize_mean", [0.5, 0.5, 0.5]), dtype=np.float32)
        self.normalize_std = np.asarray(self.siamese_config.get("normalize_std", [0.5, 0.5, 0.5]), dtype=np.float32)
        if self.normalize_mean.shape != (3,) or self.normalize_std.shape != (3,):
            raise ValueError("normalize_mean and normalize_std must have 3 values")

    @property
    def gallery_size(self) -> int:
        return int(self.index.ntotal)

    @property
    def breed_count(self) -> int:
        if "blood_name" not in self.meta.columns:
            return 0
        return int(self.meta["blood_name"].fillna("").astype(str).nunique())

    def encode(self, img_bytes: bytes, eye_crop: bool = False) -> np.ndarray:
        image_bgr = self._decode_image(img_bytes)
        normalized_bgr = self._prepare_normalized_iris(image_bgr, eye_crop=eye_crop)
        tensor = self._normalized_bgr_to_tensor(normalized_bgr).to(self.device)
        with torch.no_grad():
            if len(self.encoders) == 1:
                embedding = self.encoders[0](tensor).detach().cpu().numpy()[0].astype(np.float32)
            else:
                parts = []
                for enc in self.encoders:
                    f = enc(tensor).detach().cpu().numpy()[0].astype(np.float32)
                    f = f / (np.linalg.norm(f) + 1e-12)
                    parts.append(f)
                embedding = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-12:
            raise ValueError("编码失败：embedding 范数为 0")
        return (embedding / norm).astype(np.float32)

    def _prepare_normalized_iris(self, image_bgr: np.ndarray, eye_crop: bool = False) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        # Case 1: already a normalized iris strip (aspect ratio >= 4:1)
        if height > 0 and width / height >= 4.0:
            return image_bgr
        # Case 2: caller confirms this is already an eye crop (e.g. from Android YOLO)
        if eye_crop:
            return self._segment_and_normalize(image_bgr)
        # Case 3: full pigeon image — run YOLO detection first
        eye_bgr = self._detect_eye_crop(image_bgr)
        return self._segment_and_normalize(eye_bgr)

    def _segment_and_normalize(self, eye_bgr: np.ndarray) -> np.ndarray:
        prediction = self.segmenter.predict(
            eye_bgr,
            mask_confidence_threshold=float(DEFAULT_MASK_CONFIDENCE),
        )
        if not isinstance(prediction, IrisSegmentationSuccess):
            raise ValueError(
                f"虹膜分割失败：{prediction.reason}。请上传清晰的眼部特写，"
                "或直接上传 64×512 的归一化虹膜图。"
            )
        if prediction.mask_confidence < float(DEFAULT_MASK_CONFIDENCE):
            raise ValueError(
                f"虹膜分割失败：mask_confidence={prediction.mask_confidence:.4f} "
                f"低于阈值 {float(DEFAULT_MASK_CONFIDENCE):.4f}。请上传清晰的眼部特写。"
            )

        try:
            return daugman_normalize_color(eye_bgr, prediction, shape=NORMALIZED_SHAPE)
        except Exception as exc:
            raise ValueError(f"虹膜展开失败：{exc}") from exc

    def _detect_eye_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        results = self.detector.predict(
            source=image_bgr,
            conf=self.detection_confidence,
            verbose=False,
            device=self._ultralytics_device(),
        )
        if not results:
            raise ValueError("眼部检测失败：YOLO模型无输出，请确认检测模型已正确加载")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            raise ValueError("眼部检测失败：未能检测到鸽眼，请上传包含清晰眼部特写的图片")

        best_index = -1
        best_conf = -1.0
        for idx in range(len(boxes)):
            conf = float(boxes.conf[idx].item())
            if conf >= self.detection_confidence and conf > best_conf:
                best_conf = conf
                best_index = idx
        if best_index < 0:
            raise ValueError(
                f"眼部检测失败：置信度最高的检测框 ({best_conf:.2f}) 低于阈值 "
                f"{self.detection_confidence}，请上传更清晰的眼部特写图片"
            )

        x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[best_index].tolist()]
        image_height, image_width = image_bgr.shape[:2]
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = box_width * self.detection_expand_ratio
        pad_y = box_height * self.detection_expand_ratio
        ex1 = max(0, int(round(x1 - pad_x)))
        ey1 = max(0, int(round(y1 - pad_y)))
        ex2 = min(image_width, int(round(x2 + pad_x)))
        ey2 = min(image_height, int(round(y2 + pad_y)))
        if ex2 <= ex1 or ey2 <= ey1:
            raise ValueError("眼部检测失败：检测框无效")
        return image_bgr[ey1:ey2, ex1:ex2].copy()

    def compare(self, img_bytes_a: bytes, img_bytes_b: bytes, eye_crop: bool = False) -> dict[str, Any]:
        feat_a = self.encode(img_bytes_a, eye_crop=eye_crop)
        feat_b = self.encode(img_bytes_b, eye_crop=eye_crop)
        distance = float(np.linalg.norm(feat_a - feat_b))
        return {
            "distance": distance,
            "same_family": bool(distance < self.threshold),
            "threshold": float(self.threshold),
        }

    def search(self, img_bytes: bytes, top_k: int = DEFAULT_SEARCH_TOP_K, eye_crop: bool = False) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        top_k = min(int(top_k), MAX_SEARCH_TOP_K)
        feature = self.encode(img_bytes, eye_crop=eye_crop).reshape(1, -1).astype(np.float32)
        k = min(int(top_k), int(self.index.ntotal))
        distances, indices = self.index.search(feature, k)

        results: list[dict[str, Any]] = []
        for rank, (distance, index_id) in enumerate(zip(distances[0], indices[0]), start=1):
            if index_id < 0:
                continue
            row = self.meta.iloc[int(index_id)]
            item = {
                "rank": int(rank),
                "img_id": str(row.get("img_id", "")),
                "blood_id": str(row.get("blood_id", "")),
                "blood_name": str(row.get("blood_name", row.get("blood", ""))),
                "distance": float(distance),
            }
            if "pg_id" in row.index:
                item["pg_id"] = str(row.get("pg_id", ""))
            results.append(item)
        return results

    def gallery_image_path(self, img_id: str) -> Path | None:
        path = self.img_index.get(str(img_id))
        if path is None:
            return None
        if path.is_file():
            return path

        marker = "PigeonVision/"
        raw = str(path)
        if marker in raw:
            relocated = ROOT / raw.split(marker, 1)[1]
            if relocated.is_file():
                return relocated
        return None

    def _load_encoders(self) -> list[IrisEncoder]:
        if not self.fusion:
            return [self._load_single_encoder(self.siamese_checkpoint)]
        encoders = []
        for ckpt_path, feat_dim, backbone in FUSION_ENCODERS:
            resolved = self._resolve(ckpt_path)
            if not resolved.exists():
                raise FileNotFoundError(f"缺少编码器权重：{resolved}")
            encoders.append(self._load_single_encoder(resolved, feat_dim, backbone))
        return encoders

    def _load_single_encoder(self, checkpoint_path: Path, feat_dim: int = 256, backbone: str = "resnet34") -> IrisEncoder:
        state = torch.load(checkpoint_path, map_location=self.device)
        checkpoint_config = state.get("config", {}) if isinstance(state, dict) else {}
        dim = int(checkpoint_config.get("feat_dim", feat_dim))
        arch = str(checkpoint_config.get("backbone", backbone))
        encoder = IrisEncoder(feat_dim=dim, backbone=arch, pretrained=False, in_channels=3).to(self.device)
        model_state = state["model_state"] if isinstance(state, dict) and "model_state" in state else state
        encoder.load_state_dict(model_state)
        encoder.eval()
        return encoder

    def _load_detector(self) -> YOLO:
        checkpoint = self.detection_checkpoint
        if not checkpoint.exists():
            candidates = sorted(
                (ROOT / "checkpoints" / "detection").rglob("best.pt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(f"缺少 YOLO 眼部检测权重：{checkpoint}")
            checkpoint = candidates[0]
            self.detection_checkpoint = checkpoint
        return YOLO(str(checkpoint))

    def _load_segmenter(self) -> UNetPredictor:
        if not self.segmentation_checkpoint.exists():
            raise FileNotFoundError(f"缺少 U-Net 权重：{self.segmentation_checkpoint}")
        return UNetPredictor(
            checkpoint_path=self.segmentation_checkpoint,
            device=self.device,
            input_size=int(self.unet_config.get("input_size", 256)),
            in_channels=int(self.unet_config.get("in_channels", 1)),
            num_classes=int(self.unet_config.get("num_classes", 3)),
            base_channels=int(self.unet_config.get("base_channels", 32)),
            num_groups=int(self.unet_config.get("num_groups", 8)),
        )

    def _load_faiss_index(self):
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"缺少 FAISS 索引：{self.faiss_index_path}")
        return faiss.read_index(str(self.faiss_index_path))

    def _load_feature_meta(self) -> pd.DataFrame:
        if not self.feature_meta_path.exists():
            raise FileNotFoundError(f"缺少特征库元数据：{self.feature_meta_path}")
        meta = pd.read_csv(self.feature_meta_path, dtype=str).fillna("")
        required = {"img_id", "blood_name"}
        missing = required - set(meta.columns)
        if missing:
            raise ValueError(f"{self.feature_meta_path} 缺少列：{sorted(missing)}")
        return meta.reset_index(drop=True)

    def _load_img_index(self) -> dict[str, Path]:
        if not self.img_index_path.exists():
            raise FileNotFoundError(f"缺少图片索引：{self.img_index_path}")
        index = pd.read_csv(self.img_index_path, dtype=str).fillna("")
        required = {"img_id", "path"}
        missing = required - set(index.columns)
        if missing:
            raise ValueError(f"{self.img_index_path} 缺少列：{sorted(missing)}")
        return {
            str(row.img_id): Path(str(row.path))
            for row in index.itertuples(index=False)
            if str(row.img_id) and str(row.path)
        }

    def _validate_index_meta(self) -> None:
        if int(self.index.ntotal) != len(self.meta):
            raise ValueError(
                f"FAISS 索引数量与元数据行数不一致：index={self.index.ntotal}, meta={len(self.meta)}"
            )
        expected_dim = 1024 if self.fusion else int(self.siamese_config.get("feat_dim", 256))
        if int(self.index.d) != expected_dim:
            raise ValueError(f"FAISS 维度与配置不一致：index.d={self.index.d}, expected={expected_dim}")

    def _normalized_bgr_to_tensor(self, normalized_bgr: np.ndarray) -> torch.Tensor:
        height, width = self.input_shape
        if normalized_bgr.shape[:2] != (height, width):
            normalized_bgr = cv2.resize(normalized_bgr, (width, height), interpolation=cv2.INTER_AREA)
        normalized_rgb = cv2.cvtColor(normalized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized_rgb = (normalized_rgb - self.normalize_mean[None, None, :]) / self.normalize_std[None, None, :]
        tensor = torch.from_numpy(normalized_rgb.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(dtype=torch.float32)

    @staticmethod
    def _decode_image(img_bytes: bytes) -> np.ndarray:
        if not img_bytes:
            raise ValueError("图片为空")
        encoded = np.frombuffer(img_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("图片解码失败，请上传有效图片")
        return image_bgr

    @staticmethod
    def _resolve(path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"缺少配置文件：{path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件格式错误：{path}")
        return data

    @staticmethod
    def _load_threshold(path: Path) -> float:
        if not path.exists():
            raise FileNotFoundError(f"缺少阈值文件：{path}")
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "threshold" not in payload:
            raise ValueError(f"{path} 缺少 threshold 字段")
        return float(payload["threshold"])

    def _ultralytics_device(self) -> int | str:
        if self.device.type != "cuda":
            return "cpu"
        return int(self.device.index) if self.device.index is not None else 0
