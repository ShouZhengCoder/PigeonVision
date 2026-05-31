"""Multi-label dataset for SupCon training with full blood_id sets.

Each sample returns (img_id, image, blood_id_set_indices, blood_name_label)
where blood_id_set_indices encodes the complete list of blood_ids for that image.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset import default_transform, load_rgb_image


class MultiLabelSample(NamedTuple):
    img_id: str
    image: torch.Tensor
    blood_id_indices: list[int]  # list of blood_id integer indices
    blood_name_label: int
    blood_name: str


class MultiLabelIrisDataset(Dataset):
    """Dataset returning full blood_id sets for multi-positive contrastive learning.

    Each image is associated with MULTIPLE blood_ids (not just one).
    Two images are "positive pairs" if they share at least one blood_id.
    """

    def __init__(
        self,
        meta_path: str | Path,
        blood_id_map_path: str | Path,
        iris_dir: str | Path,
        transform=None,
    ) -> None:
        self.iris_dir = Path(iris_dir)
        self.transform = transform if transform is not None else default_transform(train=False)

        # Load metadata
        df = pd.read_csv(meta_path, dtype={"img_id": str, "blood_name": str})
        required = {"img_id", "blood_ids", "blood_id_indices", "blood_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {meta_path}: {sorted(missing)}")

        self.rows = df.reset_index(drop=True)
        self.blood_name_to_label = self._build_name_label_map()

        # Parse blood_id indices (stored as JSON array in CSV)
        self._blood_id_indices: list[list[int]] = []
        for raw in self.rows["blood_id_indices"]:
            self._blood_id_indices.append(json.loads(raw))

        # Build blood_id -> name label mapping for fast overlap checking
        self._blood_id_to_name_label: dict[int, int] = {}
        for i, row in self.rows.iterrows():
            for bid_idx in self._blood_id_indices[i]:
                if bid_idx not in self._blood_id_to_name_label:
                    self._blood_id_to_name_label[bid_idx] = self.blood_name_to_label.get(
                        str(row["blood_name"]), -1
                    )

        # Pre-index: map blood_name_label -> list of row indices for efficient negative finding
        self._label_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, row in self.rows.iterrows():
            label = self.blood_name_to_label.get(str(row["blood_name"]), -1)
            self._label_to_indices[label].append(i)

    def _build_name_label_map(self) -> dict[str, int]:
        names = sorted(self.rows["blood_name"].astype(str).unique())
        return {name: idx for idx, name in enumerate(names)}

    @property
    def num_blood_names(self) -> int:
        return len(self.blood_name_to_label)

    @property
    def num_images(self) -> int:
        return len(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        img_id = str(row["img_id"])
        image = load_rgb_image(self.iris_dir / f"{img_id}.png")
        tensor = self.transform(image)
        return MultiLabelSample(
            img_id=img_id,
            image=tensor,
            blood_id_indices=self._blood_id_indices[index],
            blood_name_label=self.blood_name_to_label.get(str(row["blood_name"]), -1),
            blood_name=str(row["blood_name"]),
        )

    def get_blood_id_indices(self, index: int) -> list[int]:
        return self._blood_id_indices[index]

    def get_blood_name_label(self, index: int) -> int:
        return self.blood_name_to_label.get(str(self.rows.iloc[index]["blood_name"]), -1)

    def build_batch_positive_mask(
        self,
        indices: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Build (B, B) bool mask: True if images i,j share any blood_id."""
        batch_size = len(indices)
        mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)

        # For small batches, O(B^2) is fine
        bid_sets = [set(self._blood_id_indices[idx]) for idx in indices]
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if bid_sets[i] & bid_sets[j]:
                    mask[i, j] = True
                    mask[j, i] = True
        return mask

    def get_blood_id_sets_for_indices(self, indices: list[int]) -> list[list[int]]:
        """Return blood_id index lists for given batch indices (for SD weighting)."""
        return [self._blood_id_indices[idx] for idx in indices]


def collate_multilabel(batch: list[MultiLabelSample]) -> dict:
    """Custom collate function for MultiLabelIrisDataset.

    Returns a dict to preserve the list-of-lists structure for blood_id_indices.
    """
    return {
        "img_ids": [s.img_id for s in batch],
        "images": torch.stack([s.image for s in batch]),
        "blood_id_indices": [s.blood_id_indices for s in batch],
        "blood_name_labels": torch.tensor([s.blood_name_label for s in batch], dtype=torch.long),
        "blood_names": [s.blood_name for s in batch],
    }
