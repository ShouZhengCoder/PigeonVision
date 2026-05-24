from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from dataset import PairDataset, default_transform
from loss import contrastive_loss
from model import IrisEncoder, SiameseNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train siamese iris encoder.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "siamese.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override config lr.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/siamese/last.pt.")
    parser.add_argument("--limit-train", type=int, default=None, help="Use first N train pairs for smoke tests.")
    parser.add_argument("--limit-val", type=int, default=None, help="Use first N val pairs for smoke tests.")
    parser.add_argument("--recall-every", type=int, default=10)
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet weights for quick smoke tests.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_path: Path) -> logging.Logger:
    ensure_dir(log_path.parent)
    logger = logging.getLogger("siamese_train")
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


def maybe_subset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def run_epoch(
    model: SiameseNet,
    loader: DataLoader,
    optimizer: Adam | None,
    margin: float,
    device: torch.device,
    desc: str,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    with torch.set_grad_enabled(training):
        for img_a, img_b, label in tqdm(loader, desc=desc):
            img_a = img_a.to(device, non_blocking=True)
            img_b = img_b.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            feat_a, feat_b = model(img_a, img_b)
            loss = contrastive_loss(feat_a, feat_b, label, margin=margin)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = img_a.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)


class IrisImageDataset(Dataset):
    def __init__(self, img_ids: list[str], img_dir: Path, transform) -> None:
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int):
        img_id = self.img_ids[index]
        with Image.open(self.img_dir / f"{img_id}.png") as image:
            image = image.convert("RGB")
            return img_id, self.transform(image)


@torch.no_grad()
def extract_val_features(
    img_ids: list[str],
    img_dir: Path,
    transform,
    encoder: IrisEncoder,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[list[str], torch.Tensor]:
    dataset = IrisImageDataset(img_ids, img_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features: list[torch.Tensor] = []
    ordered_ids: list[str] = []
    encoder.eval()
    for batch_ids, images in tqdm(loader, desc="val features"):
        images = images.to(device, non_blocking=True)
        features.append(encoder(images).detach().cpu())
        ordered_ids.extend([str(img_id) for img_id in batch_ids])
    return ordered_ids, torch.cat(features, dim=0)


@torch.no_grad()
def recall_at_1(
    encoder: IrisEncoder,
    pairs_csv: Path,
    img_dir: Path,
    transform,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    df = pd.read_csv(pairs_csv, dtype={"img_id_a": str, "img_id_b": str})
    df_pos = df[df["label"].astype(int) == 1].copy()
    if df_pos.empty:
        return 0.0

    available = {
        img_id
        for img_id in pd.concat([df["img_id_a"], df["img_id_b"]]).astype(str).unique()
        if (img_dir / f"{img_id}.png").exists()
    }
    df_pos = df_pos[df_pos["img_id_a"].isin(available) & df_pos["img_id_b"].isin(available)]
    if df_pos.empty:
        return 0.0

    img_ids = sorted(available)
    ordered_ids, features = extract_val_features(img_ids, img_dir, transform, encoder, device, batch_size, num_workers)
    features = features.to(device)
    id_to_index = {img_id: i for i, img_id in enumerate(ordered_ids)}

    anchor_indices = torch.tensor([id_to_index[str(x)] for x in df_pos["img_id_a"]], device=device)
    target_indices = torch.tensor([id_to_index[str(x)] for x in df_pos["img_id_b"]], device=device)
    hits = 0
    chunk_size = 512
    for start in tqdm(range(0, len(anchor_indices), chunk_size), desc="recall@1"):
        end = min(start + chunk_size, len(anchor_indices))
        anchors = features[anchor_indices[start:end]]
        dist2 = 2.0 - 2.0 * anchors @ features.T
        row = torch.arange(end - start, device=device)
        dist2[row, anchor_indices[start:end]] = float("inf")
        nearest = torch.argmin(dist2, dim=1)
        hits += (nearest == target_indices[start:end]).sum().item()
    return hits / len(anchor_indices)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: SiameseNet,
    optimizer: Adam,
    scheduler: CosineAnnealingLR,
    best_val_loss: float,
    best_epoch: int,
    config: dict,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.encoder.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "config": config,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["lr"] = args.lr

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir = ensure_dir(resolve_root_path(config["checkpoint_dir"]))
    log_path = ROOT / "logs" / "siamese_train.log"
    logger = setup_logger(log_path)
    writer = SummaryWriter(log_dir=str(ROOT / "logs" / "tensorboard" / "siamese"))

    transform = default_transform(
        input_shape=config["input_shape"],
        mean=config["normalize_mean"],
        std=config["normalize_std"],
    )
    iris_dir = resolve_root_path(config["iris_dir"])
    train_csv = resolve_root_path(config["pairs_train"])
    val_csv = resolve_root_path(config["pairs_val"])

    train_dataset = maybe_subset(PairDataset(train_csv, iris_dir, transform=transform), args.limit_train)
    val_dataset = maybe_subset(PairDataset(val_csv, iris_dir, transform=transform), args.limit_val)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    encoder = IrisEncoder(feat_dim=int(config["feat_dim"]), pretrained=not args.no_pretrained)
    model = SiameseNet(encoder).to(device)
    optimizer = Adam(model.parameters(), lr=float(config["lr"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config["epochs"]))

    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    last_val_loss = float("inf")
    last_epoch = start_epoch - 1
    last_path = checkpoint_dir / "last.pt"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        state = torch.load(last_path, map_location=device)
        model.encoder.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        best_state_path = checkpoint_dir / "best.pt"
        if best_state_path.exists():
            best_state = torch.load(best_state_path, map_location=device)
            best_epoch = int(best_state.get("epoch", state.get("epoch", 0)))
        else:
            best_epoch = int(state.get("best_epoch", state.get("epoch", 0)))
        start_epoch = int(state["epoch"]) + 1
        logger.info("resumed from %s at epoch %s", last_path, start_epoch)

    logger.info(
        "device=%s train_pairs=%s val_pairs=%s epochs=%s batch_size=%s lr=%s",
        device,
        len(train_dataset),
        len(val_dataset),
        config["epochs"],
        config["batch_size"],
        config["lr"],
    )

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        last_epoch = epoch
        train_loss = run_epoch(model, train_loader, optimizer, float(config["margin"]), device, f"train {epoch}")
        val_loss = run_epoch(model, val_loader, None, float(config["margin"]), device, f"val {epoch}")
        last_val_loss = val_loss
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        message = f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(checkpoint_dir / "best.pt", epoch, model, optimizer, scheduler, best_val_loss, best_epoch, config)
            message += " best=1"
        else:
            message += " best=0"
        save_checkpoint(last_path, epoch, model, optimizer, scheduler, best_val_loss, best_epoch, config)

        if args.recall_every > 0 and epoch % args.recall_every == 0 and args.limit_val is None:
            recall = recall_at_1(
                model.encoder,
                val_csv,
                iris_dir,
                transform,
                device,
                int(config["batch_size"]),
                args.num_workers,
            )
            writer.add_scalar("metric/recall_at_1", recall, epoch)
            message += f" recall_at_1={recall:.6f}"

        logger.info(message)

    writer.close()
    best_state = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.encoder.load_state_dict(best_state["model_state"])
    final_recall = recall_at_1(
        model.encoder,
        val_csv,
        iris_dir,
        transform,
        device,
        int(config["batch_size"]),
        args.num_workers,
    )
    logger.info(
        "training finished actual_epochs=%s final_val_loss=%.6f best_epoch=%s best_val_loss=%.6f recall_at_1=%.6f checkpoint=%s",
        last_epoch,
        last_val_loss,
        best_epoch,
        best_val_loss,
        final_recall,
        checkpoint_dir / "best.pt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
