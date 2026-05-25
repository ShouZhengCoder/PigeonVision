from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


class RandomHorizontalRoll:
    def __init__(self, max_shift: int = 32) -> None:
        self.max_shift = int(max_shift)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return tensor
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        return torch.roll(tensor, shifts=shift, dims=-1)


class RandomVerticalRoll:
    def __init__(self, max_shift: int = 4) -> None:
        self.max_shift = int(max_shift)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.max_shift <= 0:
            return tensor
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
        return torch.roll(tensor, shifts=shift, dims=-2)


class AddGaussianNoise:
    def __init__(self, std: float = 0.02, probability: float = 0.5) -> None:
        self.std = float(std)
        self.probability = float(probability)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0 or float(torch.rand(()).item()) > self.probability:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


class RandomHorizontalOcclusion:
    def __init__(self, max_height: int = 8, probability: float = 0.35) -> None:
        self.max_height = int(max_height)
        self.probability = float(probability)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.max_height <= 0 or float(torch.rand(()).item()) > self.probability:
            return tensor
        height = tensor.shape[-2]
        band_h = int(torch.randint(1, min(self.max_height, height) + 1, (1,)).item())
        y0 = int(torch.randint(0, height - band_h + 1, (1,)).item())
        tensor = tensor.clone()
        tensor[..., y0 : y0 + band_h, :] = 0.0
        return tensor


def default_transform(
    input_shape: tuple[int, int] | list[int] = (64, 512),
    mean: list[float] | tuple[float, ...] = (0.5, 0.5, 0.5),
    std: list[float] | tuple[float, ...] = (0.5, 0.5, 0.5),
    train: bool = False,
    roll_shift: int = 32,
    vertical_shift: int = 4,
    noise_std: float = 0.02,
    occlusion_max_height: int = 8,
):
    height, width = int(input_shape[0]), int(input_shape[1])
    ops: list[object] = [transforms.Resize((height, width))]
    if train:
        ops.append(transforms.ColorJitter(brightness=0.15, contrast=0.15))
    ops.append(transforms.ToTensor())
    if train:
        ops.extend(
            [
                RandomHorizontalRoll(roll_shift),
                RandomVerticalRoll(vertical_shift),
                AddGaussianNoise(noise_std),
                RandomHorizontalOcclusion(occlusion_max_height),
            ]
        )
    ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def load_rgb_image(path: str | Path) -> Image.Image:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)


def load_triplet_meta(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, dtype={"img_id": str, "blood_id": str, "blood_name": str})
    required = {"img_id", "blood_id", "blood_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df[["img_id", "blood_id", "blood_name"]].copy()
    df["img_id"] = df["img_id"].astype(str)
    df["blood_id"] = df["blood_id"].astype(str)
    df["blood_name"] = df["blood_name"].fillna("").astype(str).str.strip()
    df = df[df["blood_name"] != ""].drop_duplicates(subset=["img_id", "blood_id", "blood_name"])
    return df.sort_values(["blood_name", "blood_id", "img_id"]).reset_index(drop=True)


def build_triplet_label_maps(*frames: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    blood_ids = sorted({str(value) for frame in frames for value in frame["blood_id"].astype(str).unique()})
    blood_names = sorted({str(value) for frame in frames for value in frame["blood_name"].astype(str).unique()})
    return ({value: idx for idx, value in enumerate(blood_ids)}, {value: idx for idx, value in enumerate(blood_names)})


def add_triplet_label_columns(
    df: pd.DataFrame,
    blood_id_to_label: dict[str, int],
    blood_name_to_label: dict[str, int],
) -> pd.DataFrame:
    out = df.copy()
    out["blood_id_label"] = out["blood_id"].astype(str).map(blood_id_to_label).astype(int)
    out["blood_name_label"] = out["blood_name"].astype(str).map(blood_name_to_label).astype(int)
    return out


def load_blood_rows(
    normalize_meta: str | Path,
    pigeon_csv: str | Path,
    img_dir: str | Path,
    min_images_per_blood: int = 5,
) -> pd.DataFrame:
    normalize_df = pd.read_csv(normalize_meta, dtype={"img_id": str})
    success_ids = set(normalize_df[normalize_df["status"] == "success"]["img_id"].astype(str))

    try:
        pigeon_df = pd.read_csv(pigeon_csv, dtype={"ID": str})
    except pd.errors.ParserError:
        pigeon_df = pd.read_csv(pigeon_csv, dtype={"ID": str}, engine="python", on_bad_lines="skip")
        print(f"warning: skipped malformed rows while reading {pigeon_csv}")

    img_dir = Path(img_dir)
    pigeon_df["ID"] = pigeon_df["ID"].astype(str)
    pigeon_df["BLOOD"] = pigeon_df["BLOOD"].fillna("").astype(str).str.strip()
    pigeon_df = pigeon_df[(pigeon_df["BLOOD"] != "") & pigeon_df["ID"].isin(success_ids)]
    pigeon_df = pigeon_df[pigeon_df["ID"].map(lambda img_id: (img_dir / f"{img_id}.png").exists())]
    pigeon_df = pigeon_df.drop_duplicates(subset=["ID"]).copy()

    counts = pigeon_df["BLOOD"].value_counts()
    keep_bloods = set(counts[counts >= int(min_images_per_blood)].index)
    pigeon_df = pigeon_df[pigeon_df["BLOOD"].isin(keep_bloods)].copy()
    rows = pd.DataFrame(
        {
            "img_id": pigeon_df["ID"].astype(str),
            "pg_id": pigeon_df["PG_ID"].fillna("").astype(str),
            "blood": pigeon_df["BLOOD"].astype(str),
        }
    )
    return rows.sort_values(["blood", "img_id"]).reset_index(drop=True)


def split_by_blood(rows: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(int(seed))
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    for _blood, group in rows.groupby("blood", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * float(val_ratio))))
        if len(indices) - val_count < 1:
            val_count = len(indices) - 1
        val_idx = indices[:val_count]
        train_idx = indices[val_count:]
        train_parts.append(rows.loc[train_idx])
        val_parts.append(rows.loc[val_idx])
    train_df = pd.concat(train_parts).sort_values("img_id").reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values("img_id").reset_index(drop=True)
    return train_df, val_df


def add_label_column(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    bloods = sorted(train_df["blood"].astype(str).unique())
    label_to_blood = {blood: idx for idx, blood in enumerate(bloods)}
    train_df = train_df.copy()
    val_df = val_df[val_df["blood"].isin(label_to_blood)].copy()
    train_df["label"] = train_df["blood"].map(label_to_blood).astype(int)
    val_df["label"] = val_df["blood"].map(label_to_blood).astype(int)
    return train_df, val_df, label_to_blood


class IrisClassDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, img_dir: str | Path, transform=None) -> None:
        self.rows = rows.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform if transform is not None else default_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        img_id = str(row["img_id"])
        image = load_rgb_image(self.img_dir / f"{img_id}.png")
        tensor = self.transform(image)
        return img_id, tensor, torch.tensor(int(row["label"]), dtype=torch.long)


class TripletMetaDataset(Dataset):
    """Dataset for batch-hard triplet learning.

    Positive rule: same `blood_id`.
    Negative rule: different `blood_name`.
    Same `blood_name` with different `blood_id` is deliberately skipped as a negative.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        img_dir: str | Path,
        transform=None,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform if transform is not None else default_transform(train=False)
        required = {"img_id", "blood_id", "blood_name", "blood_id_label", "blood_name_label"}
        missing = required - set(self.rows.columns)
        if missing:
            raise ValueError(f"TripletMetaDataset rows missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        img_id = str(row["img_id"])
        image = load_rgb_image(self.img_dir / f"{img_id}.png")
        tensor = self.transform(image)
        return (
            img_id,
            tensor,
            torch.tensor(int(row["blood_id_label"]), dtype=torch.long),
            torch.tensor(int(row["blood_name_label"]), dtype=torch.long),
        )


class PKBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: list[int] | np.ndarray | pd.Series,
        classes_per_batch: int = 16,
        samples_per_class: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.labels = [int(label) for label in labels]
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.batches_per_epoch = int(batches_per_epoch or math.ceil(len(self.labels) / max(self.batch_size, 1)))
        self.seed = int(seed)
        self.epoch = 0

        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            by_label[label].append(index)
        self.by_label = dict(by_label)
        self.unique_labels = sorted(self.by_label)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            if len(self.unique_labels) >= self.classes_per_batch:
                labels = rng.sample(self.unique_labels, self.classes_per_batch)
            else:
                labels = [rng.choice(self.unique_labels) for _ in range(self.classes_per_batch)]

            batch: list[int] = []
            for label in labels:
                indices = self.by_label[label]
                replace = len(indices) < self.samples_per_class
                if replace:
                    batch.extend(rng.choice(indices) for _ in range(self.samples_per_class))
                else:
                    batch.extend(rng.sample(indices, self.samples_per_class))
            rng.shuffle(batch)
            yield batch


class BloodIdPKSampler(Sampler[list[int]]):
    def __init__(
        self,
        blood_id_labels: list[int] | np.ndarray | pd.Series,
        classes_per_batch: int = 16,
        samples_per_class: int = 4,
        batches_per_epoch: int | None = None,
        seed: int = 42,
        min_samples_per_class: int = 2,
    ) -> None:
        self.labels = [int(label) for label in blood_id_labels]
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.batches_per_epoch = int(batches_per_epoch or math.ceil(len(self.labels) / max(self.batch_size, 1)))
        self.seed = int(seed)
        self.epoch = 0

        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            by_label[label].append(index)
        self.by_label = {
            label: indices
            for label, indices in by_label.items()
            if len(indices) >= int(min_samples_per_class)
        }
        self.unique_labels = sorted(self.by_label)
        if not self.unique_labels:
            raise ValueError("BloodIdPKSampler requires at least one blood_id with >=2 images")

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            if len(self.unique_labels) >= self.classes_per_batch:
                labels = rng.sample(self.unique_labels, self.classes_per_batch)
            else:
                labels = [rng.choice(self.unique_labels) for _ in range(self.classes_per_batch)]

            batch: list[int] = []
            for label in labels:
                indices = self.by_label[label]
                replace = len(indices) < self.samples_per_class
                if replace:
                    batch.extend(rng.choice(indices) for _ in range(self.samples_per_class))
                else:
                    batch.extend(rng.sample(indices, self.samples_per_class))
            rng.shuffle(batch)
            yield batch


class PairDataset(Dataset):
    def __init__(self, pairs_csv: str | Path, img_dir: str | Path, transform=None) -> None:
        self.pairs_csv = Path(pairs_csv)
        self.img_dir = Path(img_dir)
        self.transform = transform if transform is not None else default_transform(train=False)

        df = pd.read_csv(self.pairs_csv, dtype={"img_id_a": str, "img_id_b": str})
        required = {"img_id_a", "img_id_b", "label"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{self.pairs_csv} missing columns: {sorted(missing_cols)}")

        self.rows: list[tuple[str, str, float]] = []
        for row in df.itertuples(index=False):
            img_id_a = str(row.img_id_a)
            img_id_b = str(row.img_id_b)
            if (self.img_dir / f"{img_id_a}.png").exists() and (self.img_dir / f"{img_id_b}.png").exists():
                self.rows.append((img_id_a, img_id_b, float(row.label)))

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, img_id: str) -> torch.Tensor:
        image = load_rgb_image(self.img_dir / f"{img_id}.png")
        return self.transform(image)

    def __getitem__(self, index: int):
        img_id_a, img_id_b, label = self.rows[index]
        return self._load_image(img_id_a), self._load_image(img_id_b), torch.tensor(label, dtype=torch.float32)
