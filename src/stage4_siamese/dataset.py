from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform(
    input_size: int = 128,
    mean: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: list[float] | tuple[float, float, float] = (0.5, 0.5, 0.5),
):
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class PairDataset(Dataset):
    def __init__(self, pairs_csv: str | Path, img_dir: str | Path, transform=None) -> None:
        self.pairs_csv = Path(pairs_csv)
        self.img_dir = Path(img_dir)
        self.transform = transform if transform is not None else default_transform()

        df = pd.read_csv(self.pairs_csv, dtype={"img_id_a": str, "img_id_b": str})
        required = {"img_id_a", "img_id_b", "label"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{self.pairs_csv} missing columns: {sorted(missing_cols)}")

        self.rows: list[tuple[str, str, float]] = []
        missing_count = 0
        max_warnings = 20
        for row in df.itertuples(index=False):
            img_id_a = str(row.img_id_a)
            img_id_b = str(row.img_id_b)
            path_a = self.img_dir / f"{img_id_a}.png"
            path_b = self.img_dir / f"{img_id_b}.png"
            if not path_a.exists() or not path_b.exists():
                missing_count += 1
                if missing_count <= max_warnings:
                    print(f"warning: skip missing pair {img_id_a}, {img_id_b}")
                continue
            self.rows.append((img_id_a, img_id_b, float(row.label)))

        if missing_count > max_warnings:
            print(f"warning: skipped {missing_count} pairs with missing image files")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, img_id: str) -> torch.Tensor:
        path = self.img_dir / f"{img_id}.png"
        with Image.open(path) as image:
            image = image.convert("RGB")
            return self.transform(image)

    def __getitem__(self, index: int):
        img_id_a, img_id_b, label = self.rows[index]
        img_a = self._load_image(img_id_a)
        img_b = self._load_image(img_id_b)
        return img_a, img_b, torch.tensor(label, dtype=torch.float32)
