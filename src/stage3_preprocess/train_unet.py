from __future__ import annotations

import argparse
import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from unet_common import (
    BASE_CHANNELS,
    GROUP_NORM_GROUPS,
    IMAGE_SIZE,
    NUM_CLASSES,
    UNet,
    colorize_mask,
    gray_to_tensor,
    mask_to_tensor,
    overlay_mask,
    resize_gray_image,
    resize_mask,
)


@dataclass(frozen=True)
class Sample:
    img_id: str
    image_path: Path
    mask_path: Path


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        input_size: int = IMAGE_SIZE,
        augment: bool = False,
        rotate_degrees: float = 30.0,
        brightness_jitter: float = 0.2,
        contrast_jitter: float = 0.2,
        flip_prob: float = 0.5,
    ) -> None:
        self.samples = samples
        self.input_size = int(input_size)
        self.augment = bool(augment)
        self.rotate_degrees = float(rotate_degrees)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self.flip_prob = float(flip_prob)

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if random.random() < self.flip_prob:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
        if abs(angle) > 1e-3:
            matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            mask = cv2.warpAffine(
                mask,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        alpha = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
        beta = random.uniform(-self.brightness_jitter, self.brightness_jitter) * 255.0
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(sample.image_path)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(sample.mask_path)

        image = resize_gray_image(image, self.input_size)
        mask = resize_mask(mask, self.input_size)

        if self.augment:
            image, mask = self._apply_augment(image, mask)

        return gray_to_tensor(image), mask_to_tensor(mask), sample.img_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net for pigeon iris segmentation.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "unet.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr.")
    parser.add_argument("--patience", type=int, default=None, help="Early stop when val dice_fg stops improving.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/segmentation/last.pt.")
    parser.add_argument("--limit-train", type=int, default=None, help="Use first N train samples for smoke tests.")
    parser.add_argument("--limit-val", type=int, default=None, help="Use first N val samples for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", action="store_true", help="Disable train augmentation.")
    parser.add_argument("--vis-samples", type=int, default=4)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("stage3_unet_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_samples(images_dir: Path, masks_dir: Path) -> list[Sample]:
    samples: list[Sample] = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        mask_path = masks_dir / f"{image_path.stem}_mask.png"
        if mask_path.exists():
            samples.append(Sample(image_path.stem, image_path, mask_path))
    return samples


def split_samples(samples: list[Sample], train_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    shuffled = samples[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def maybe_limit(samples: list[Sample], limit: int | None) -> list[Sample]:
    if limit is None or limit >= len(samples):
        return samples
    return samples[:limit]


def dice_loss(logits: torch.Tensor, target: torch.Tensor, foreground_classes: tuple[int, ...] = (1, 2)) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    losses: list[torch.Tensor] = []
    smooth = 1.0
    for cls in foreground_classes:
        prob = probs[:, cls]
        truth = target_oh[:, cls]
        intersection = (prob * truth).sum(dim=(1, 2))
        denominator = prob.sum(dim=(1, 2)) + truth.sum(dim=(1, 2))
        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        losses.append(1.0 - dice.mean())
    return torch.stack(losses).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(logits, target)
    dice = dice_loss(logits, target)
    loss = ce + dice
    return loss, {"ce": float(ce.item()), "dice": float(dice.item())}


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_CLASSES) -> dict[str, float]:
    pred = logits.argmax(dim=1)
    eps = 1e-6
    metrics: dict[str, float] = {}
    dice_values: list[float] = []
    iou_values: list[float] = []
    for cls in range(num_classes):
        pred_c = pred == cls
        target_c = target == cls
        intersection = (pred_c & target_c).sum().item()
        pred_sum = pred_c.sum().item()
        target_sum = target_c.sum().item()
        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        union = pred_sum + target_sum - intersection
        iou = (intersection + eps) / (union + eps)
        metrics[f"dice_{cls}"] = float(dice)
        metrics[f"iou_{cls}"] = float(iou)
        dice_values.append(float(dice))
        iou_values.append(float(iou))
    fg_dice = [metrics["dice_1"], metrics["dice_2"]]
    fg_iou = [metrics["iou_1"], metrics["iou_2"]]
    metrics["dice_fg"] = float(sum(fg_dice) / len(fg_dice))
    metrics["iou_fg"] = float(sum(fg_iou) / len(fg_iou))
    metrics["dice_mean"] = float(sum(dice_values) / len(dice_values))
    metrics["iou_mean"] = float(sum(iou_values) / len(iou_values))
    return metrics


def make_sample_panel(image_tensor: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, sample_id: str) -> np.ndarray:
    image = ((image_tensor.squeeze(0).cpu().numpy() * 0.5) + 0.5) * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    gt = target.cpu().numpy().astype(np.uint8)
    pred_np = pred.cpu().numpy().astype(np.uint8)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt_overlay = overlay_mask(image, gt)
    pred_overlay = overlay_mask(image, pred_np)

    labels = [
        ("image", image_bgr),
        ("target", gt_overlay),
        ("pred", pred_overlay),
    ]
    panels: list[np.ndarray] = []
    for label, panel_img in labels:
        panel = panel_img.copy()
        cv2.putText(panel, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)

    row = np.hstack(panels)
    cv2.putText(row, sample_id, (8, 252), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return row


@torch.no_grad()
def save_visualizations(
    model: UNet,
    dataset: SegmentationDataset,
    device: torch.device,
    output_path: Path,
    num_samples: int = 4,
) -> None:
    if len(dataset) == 0:
        return
    sample_count = min(num_samples, len(dataset))
    indices = list(range(sample_count))
    panels: list[np.ndarray] = []
    model.eval()
    for idx in indices:
        image_tensor, target_tensor, sample_id = dataset[idx]
        logits = model(image_tensor.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu()
        panels.append(make_sample_panel(image_tensor, target_tensor, pred, sample_id))

    cols = 1
    grid = np.vstack(panels)
    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), grid)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: UNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    best_score: float,
    config: dict,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: UNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    device: torch.device,
) -> tuple[int, float]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    epoch = int(state.get("epoch", 0))
    best_score = float(state.get("best_score", 0.0))
    return epoch, best_score


def write_metrics_row(path: Path, row: dict[str, object], write_header: bool) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(
    model: UNet,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    epoch: int,
    desc: str,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    count = 0
    inter = np.zeros(NUM_CLASSES, dtype=np.float64)
    pred_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
    target_sum = np.zeros(NUM_CLASSES, dtype=np.float64)

    with torch.set_grad_enabled(training):
        for images, targets, _sample_ids in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, _parts = segmentation_loss(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            loss_total += float(loss.item()) * batch_size
            count += batch_size

            pred = logits.argmax(dim=1)
            for cls in range(NUM_CLASSES):
                pred_c = pred == cls
                target_c = targets == cls
                inter[cls] += (pred_c & target_c).sum().item()
                pred_sum[cls] += pred_c.sum().item()
                target_sum[cls] += target_c.sum().item()

    eps = 1e-6
    dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
    union = pred_sum + target_sum - inter
    iou = (inter + eps) / (union + eps)
    metrics = {
        "dice_0": float(dice[0]),
        "dice_1": float(dice[1]),
        "dice_2": float(dice[2]),
        "iou_0": float(iou[0]),
        "iou_1": float(iou[1]),
        "iou_2": float(iou[2]),
        "dice_fg": float((dice[1] + dice[2]) / 2.0),
        "iou_fg": float((iou[1] + iou[2]) / 2.0),
        "dice_mean": float(dice.mean()),
        "iou_mean": float(iou.mean()),
    }
    return loss_total / max(count, 1), metrics


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr
    patience = int(args.patience if args.patience is not None else config.get("patience", 10))

    seed_everything(int(args.seed))

    if int(config.get("in_channels", 1)) != 1:
        raise ValueError("Stage 3 v1 uses grayscale 1-channel input only")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir = ensure_dir(resolve_root_path(config["checkpoint_dir"]))
    log_path = ROOT / "logs" / "segmentation_train.log"
    logger = setup_logger(log_path)
    metrics_path = ROOT / "logs" / "segmentation_metrics.csv"
    vis_dir = ROOT / "logs" / "segmentation_vis"
    ensure_dir(vis_dir)
    if metrics_path.exists() and not args.resume:
        metrics_path.unlink()

    images_dir = resolve_root_path(config["images_dir"])
    masks_dir = resolve_root_path(config["masks_dir"])
    samples = collect_samples(images_dir, masks_dir)
    if not samples:
        raise RuntimeError(f"No paired image/mask samples found under {images_dir} and {masks_dir}")

    train_samples, val_samples = split_samples(samples, float(config.get("train_ratio", 0.8)), int(args.seed))
    train_samples = maybe_limit(train_samples, args.limit_train)
    val_samples = maybe_limit(val_samples, args.limit_val)

    train_dataset = SegmentationDataset(
        train_samples,
        input_size=int(config["input_size"]),
        augment=not args.no_augment,
        rotate_degrees=float(config.get("rotate_degrees", 30.0)),
        brightness_jitter=float(config.get("brightness_jitter", 0.2)),
        contrast_jitter=float(config.get("contrast_jitter", 0.2)),
        flip_prob=float(config.get("flip_prob", 0.5)),
    )
    val_dataset = SegmentationDataset(
        val_samples,
        input_size=int(config["input_size"]),
        augment=False,
    )

    def seed_worker(worker_id: int) -> None:
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
    )

    model = UNet(
        in_channels=int(config.get("in_channels", 1)),
        num_classes=int(config.get("num_classes", NUM_CLASSES)),
        base_channels=int(config.get("base_channels", BASE_CHANNELS)),
        num_groups=int(config.get("num_groups", GROUP_NORM_GROUPS)),
    ).to(device)
    optimizer = Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))

    start_epoch = 1
    best_score = float("-inf")
    best_epoch = 0
    best_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0
    last_epoch_run = 0
    last_path = checkpoint_dir / "last.pt"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        start_epoch, best_score = load_checkpoint(last_path, model, optimizer, scheduler, device)
        start_epoch += 1
        logger.info("resumed from %s at epoch %s", last_path, start_epoch)

    logger.info(
        "device=%s train_samples=%s val_samples=%s epochs=%s batch_size=%s lr=%s",
        device,
        len(train_dataset),
        len(val_dataset),
        config["epochs"],
        config["batch_size"],
        config["lr"],
    )

    metrics_header_written = metrics_path.exists() and metrics_path.stat().st_size > 0
    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        last_epoch_run = epoch
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, device, epoch, f"train {epoch}")
        val_loss, val_metrics = run_epoch(model, val_loader, None, device, epoch, f"val {epoch}")
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_dice_fg": round(train_metrics["dice_fg"], 6),
            "val_dice_fg": round(val_metrics["dice_fg"], 6),
            "train_iou_fg": round(train_metrics["iou_fg"], 6),
            "val_iou_fg": round(val_metrics["iou_fg"], 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        write_metrics_row(metrics_path, row, write_header=not metrics_header_written)
        metrics_header_written = True

        message = (
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"train_dice_fg={train_metrics['dice_fg']:.6f} val_dice_fg={val_metrics['dice_fg']:.6f} "
            f"train_iou_fg={train_metrics['iou_fg']:.6f} val_iou_fg={val_metrics['iou_fg']:.6f}"
        )

        if val_metrics["dice_fg"] > best_score:
            best_score = val_metrics["dice_fg"]
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, scheduler, best_score, config)
            save_visualizations(
                model,
                val_dataset,
                device,
                vis_dir / f"best_epoch_{epoch}.png",
                num_samples=int(args.vis_samples),
            )
            message += " best=1"
        else:
            epochs_without_improvement += 1
            message += " best=0"

        save_checkpoint(last_path, epoch, model, optimizer, scheduler, best_score, config)
        logger.info(message)

        if epochs_without_improvement >= patience:
            logger.info(
                "early stopping triggered at epoch=%s patience=%s best_epoch=%s best_dice_fg=%.6f",
                epoch,
                patience,
                best_epoch,
                best_score,
            )
            break

    if best_metrics is None:
        best_metrics = {"dice_fg": best_score, "dice_1": float("nan"), "dice_2": float("nan")}
    logger.info(
        "training finished actual_epochs=%s best_epoch=%s val_dice_fg=%.6f iris_dice=%.6f pupil_dice=%.6f checkpoint=%s",
        last_epoch_run,
        best_epoch,
        best_metrics.get("dice_fg", float("nan")),
        best_metrics.get("dice_1", float("nan")),
        best_metrics.get("dice_2", float("nan")),
        checkpoint_dir / "best.pt",
    )
    print(
        f"final val_dice_fg={best_metrics.get('dice_fg', float('nan')):.6f} "
        f"iris_dice={best_metrics.get('dice_1', float('nan')):.6f} "
        f"pupil_dice={best_metrics.get('dice_2', float('nan')):.6f} "
        f"actual_epochs={last_epoch_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
