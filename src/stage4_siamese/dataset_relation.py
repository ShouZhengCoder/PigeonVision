"""Relation-aware dataset and sampler for multi-label bloodline training."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from dataset import default_transform, load_rgb_image


class RelationSample(NamedTuple):
    index: int
    img_id: str
    image: torch.Tensor
    blood_id_indices: list[int]
    blood_name: str
    pg_id: str


def _parse_indices(raw: object) -> list[int]:
    if isinstance(raw, list):
        return [int(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    return [int(x) for x in json.loads(text)]


def _set_weight(values: set[int], idf: dict[int, float]) -> float:
    return float(sum(float(idf.get(int(value), 0.0)) for value in values))


def relation_score(left: set[int], right: set[int], idf: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    shared = left & right
    if not shared:
        return 0.0
    union = left | right
    union_weight = _set_weight(union, idf)
    min_weight = min(_set_weight(left, idf), _set_weight(right, idf))
    shared_weight = _set_weight(shared, idf)
    if union_weight <= 0.0 or min_weight <= 0.0:
        return 0.0
    jaccard = shared_weight / union_weight
    overlap = shared_weight / min_weight
    return float(0.7 * overlap + 0.3 * jaccard)


class RelationDataset(Dataset):
    # Phase C: pedigree k 归一化常数(全同胞观测均值 0.715), 使 pedigree k 与 IDF 同尺度 [0,1]
    PED_K_NORM = 0.715

    def __init__(
        self,
        meta_path: str | Path,
        iris_dir: str | Path,
        split: str | None = None,
        transform=None,
        limit: int | None = None,
        kinship_vectors: str | Path | None = None,
    ) -> None:
        self.meta_path = Path(meta_path)
        self.iris_dir = Path(iris_dir)
        self.transform = transform if transform is not None else default_transform(train=False)
        df = pd.read_csv(self.meta_path, dtype=str).fillna("")
        required = {"img_id", "blood_ids", "blood_id_indices"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{self.meta_path} missing columns: {sorted(missing)}")
        if split is not None and "split" in df.columns:
            df = df[df["split"].astype(str) == str(split)].copy()
        df["img_id"] = df["img_id"].astype(str).str.strip()
        df = df[df["img_id"] != ""].drop_duplicates(subset=["img_id"], keep="first")
        df = df[df["img_id"].map(lambda img_id: (self.iris_dir / f"{img_id}.png").exists())].copy()
        if limit is not None and int(limit) > 0:
            df = df.head(int(limit)).copy()
        self.rows = df.reset_index(drop=True)
        if self.rows.empty:
            raise RuntimeError(f"No rows available from {self.meta_path} split={split!r}")

        self.blood_id_indices: list[list[int]] = [_parse_indices(raw) for raw in self.rows["blood_id_indices"]]
        self.blood_id_sets: list[set[int]] = [set(values) for values in self.blood_id_indices]
        self.blood_names = self.rows.get("blood_name", pd.Series([""] * len(self.rows))).fillna("").astype(str).tolist()
        self.pg_ids = self.rows.get("pg_id", pd.Series([""] * len(self.rows))).fillna("").astype(str).tolist()
        self.img_ids = self.rows["img_id"].astype(str).tolist()
        self.idf_by_index = self._build_idf()
        self.sum_idf = np.asarray([_set_weight(ids, self.idf_by_index) for ids in self.blood_id_sets], dtype=np.float64)
        self.inverted = self._build_inverted()
        self.kinship_vec = self._load_kinship_vectors(kinship_vectors)
        self.blood_id_sets_by_img = {self.img_ids[i]: self.blood_id_sets[i] for i in range(len(self.img_ids))}

    def _load_kinship_vectors(self, path: str | Path | None) -> dict[str, dict[str, float]]:
        """加载 Phase A 的 contribution_vectors.csv -> {img_id: {ancestor: contrib}}。"""
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        df = pd.read_csv(p, dtype=str)
        vec: dict[str, dict[str, float]] = defaultdict(dict)
        for iid, anc, c in zip(df["img_id"].astype(str), df["ancestor"].astype(str),
                                df["contribution"].astype(float)):
            vec[iid][anc] = float(c)
        return dict(vec)

    def kinship(self, a_img_id: str, b_img_id: str) -> float:
        """Hybrid 亲缘 k ∈ [0,1]: 两端都结构化 -> pedigree 点积(归一化); 否则 IDF 启发式兜底。"""
        va = self.kinship_vec.get(str(a_img_id))
        vb = self.kinship_vec.get(str(b_img_id))
        if va and vb:
            small, large = (va, vb) if len(va) <= len(vb) else (vb, va)
            k = sum(c * large.get(x, 0.0) for x, c in small.items()) / self.PED_K_NORM
            return float(max(0.0, min(1.0, k)))
        # IDF 兜底(覆盖无结构系谱的鸽子)
        sa = self.blood_id_sets_by_img.get(str(a_img_id), set())
        sb = self.blood_id_sets_by_img.get(str(b_img_id), set())
        return relation_score(sa, sb, self.idf_by_index)

    def _build_idf(self) -> dict[int, float]:
        n = len(self.blood_id_sets)
        df: dict[int, int] = defaultdict(int)
        for ids in self.blood_id_sets:
            for value in ids:
                df[int(value)] += 1
        return {int(value): float(np.log((n + 1.0) / (count + 1.0))) for value, count in df.items()}

    def _build_inverted(self) -> dict[int, list[int]]:
        inverted: dict[int, list[int]] = defaultdict(list)
        for idx, ids in enumerate(self.blood_id_sets):
            for value in ids:
                inverted[int(value)].append(idx)
        return dict(inverted)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> RelationSample:
        row = self.rows.iloc[int(index)]
        img_id = str(row["img_id"])
        image = load_rgb_image(self.iris_dir / f"{img_id}.png")
        return RelationSample(
            index=int(index),
            img_id=img_id,
            image=self.transform(image),
            blood_id_indices=self.blood_id_indices[int(index)],
            blood_name=self.blood_names[int(index)],
            pg_id=self.pg_ids[int(index)],
        )

    def candidate_scores(self, index: int, use_kinship: bool = False) -> list[tuple[int, float]]:
        query = self.blood_id_sets[int(index)]
        candidates: set[int] = set()
        for value in query:
            candidates.update(self.inverted.get(int(value), []))
        candidates.discard(int(index))
        if use_kinship and self.kinship_vec:
            qid = self.img_ids[int(index)]
            scored = [(idx, self.kinship(qid, self.img_ids[idx])) for idx in candidates]
        else:
            scored = [
                (idx, relation_score(query, self.blood_id_sets[idx], self.idf_by_index))
                for idx in candidates
            ]
        return [(idx, score) for idx, score in scored if score > 0.0]

    def relevance(self, left: int, right: int) -> float:
        return relation_score(self.blood_id_sets[int(left)], self.blood_id_sets[int(right)], self.idf_by_index)

    def has_positive(self, index: int) -> bool:
        return any(len(self.inverted.get(int(value), [])) > 1 for value in self.blood_id_sets[int(index)])


class RelationBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: RelationDataset,
        anchors_per_batch: int = 32,
        strong_pos_per_anchor: int = 1,
        weak_pos_per_anchor: int = 1,
        hard_neg_per_anchor: int = 1,
        strong_threshold: float = 0.35,
        batches_per_epoch: int | None = None,
        seed: int = 42,
        use_kinship: bool = False,
    ) -> None:
        self.dataset = dataset
        self.anchors_per_batch = int(anchors_per_batch)
        self.strong_pos_per_anchor = int(strong_pos_per_anchor)
        self.weak_pos_per_anchor = int(weak_pos_per_anchor)
        self.hard_neg_per_anchor = int(hard_neg_per_anchor)
        self.strong_threshold = float(strong_threshold)
        self.seed = int(seed)
        self.use_kinship = bool(use_kinship)
        self.epoch = 0
        self.positive_anchors = [idx for idx in range(len(dataset)) if dataset.has_positive(idx)]
        if not self.positive_anchors:
            raise RuntimeError("RelationBatchSampler found no anchors with positive candidates")
        self.anchor_weights = np.asarray([max(dataset.sum_idf[idx], 1e-6) for idx in self.positive_anchors], dtype=np.float64)
        self.anchor_weights = self.anchor_weights / self.anchor_weights.sum()
        default_batches = max(1, len(self.positive_anchors) // max(self.anchors_per_batch, 1))
        self.batches_per_epoch = int(batches_per_epoch) if batches_per_epoch else default_batches
        self.last_epoch_stats: dict[str, float] = {}
        self._candidate_cache: dict[int, list[tuple[int, float]]] = {}
        self._name_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, name in enumerate(dataset.blood_names):
            if str(name).strip():
                self._name_to_indices[str(name)].append(idx)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _candidates(self, index: int) -> list[tuple[int, float]]:
        if int(index) not in self._candidate_cache:
            self._candidate_cache[int(index)] = self.dataset.candidate_scores(
                int(index), use_kinship=self.use_kinship)
        return self._candidate_cache[int(index)]

    def _sample_from(self, rng: random.Random, candidates: list[int], k: int) -> list[int]:
        if not candidates or k <= 0:
            return []
        if len(candidates) >= k:
            return rng.sample(candidates, k)
        return [rng.choice(candidates) for _ in range(k)]

    def sample_tuple(self, anchor: int, rng: random.Random) -> tuple[int | None, int | None, int | None]:
        scored = self._candidates(anchor)
        if not scored:
            return None, None, None
        strong = [idx for idx, score in scored if score >= self.strong_threshold]
        weak = [idx for idx, score in scored if 0.0 < score < self.strong_threshold]
        ordered = [idx for idx, _score in sorted(scored, key=lambda item: item[1], reverse=True)]
        strong_idx = rng.choice(strong) if strong else ordered[0]
        weak_idx = rng.choice(weak) if weak else (rng.choice(strong) if strong else ordered[min(1, len(ordered) - 1)])
        neg_idx = self._sample_negative(anchor, rng)
        return int(strong_idx), int(weak_idx), int(neg_idx) if neg_idx is not None else None

    def _sample_negative(self, anchor: int, rng: random.Random) -> int | None:
        positives = {idx for idx, _score in self._candidates(anchor)}
        positives.add(int(anchor))
        blood_name = self.dataset.blood_names[int(anchor)]
        hard_pool = [idx for idx in self._name_to_indices.get(str(blood_name), []) if idx not in positives]
        if hard_pool:
            return int(rng.choice(hard_pool))
        for _ in range(100):
            idx = int(rng.randrange(len(self.dataset)))
            if idx not in positives and self.dataset.relevance(anchor, idx) == 0.0:
                return idx
        return None

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        np_rng = np.random.default_rng(self.seed + self.epoch)
        stats = defaultdict(float)
        for _ in range(self.batches_per_epoch):
            anchors = np_rng.choice(
                self.positive_anchors,
                size=self.anchors_per_batch,
                replace=len(self.positive_anchors) < self.anchors_per_batch,
                p=self.anchor_weights,
            ).astype(int).tolist()
            batch: list[int] = []
            for anchor in anchors:
                batch.append(int(anchor))
                scored = self._candidates(anchor)
                if scored:
                    stats["anchors_with_positive"] += 1.0
                stats["anchors"] += 1.0
                strong = [idx for idx, score in scored if score >= self.strong_threshold]
                weak = [idx for idx, score in scored if 0.0 < score < self.strong_threshold]
                ordered = [idx for idx, _score in sorted(scored, key=lambda item: item[1], reverse=True)]
                if not strong and ordered:
                    strong = [ordered[0]]
                if not weak and strong:
                    weak = strong
                sampled_strong = self._sample_from(rng, strong, self.strong_pos_per_anchor)
                sampled_weak = self._sample_from(rng, weak, self.weak_pos_per_anchor)
                sampled_neg = [idx for idx in (self._sample_negative(anchor, rng) for _ in range(self.hard_neg_per_anchor)) if idx is not None]
                stats["strong_pos"] += len(sampled_strong)
                stats["weak_pos"] += len(sampled_weak)
                stats["hard_neg"] += len(sampled_neg)
                batch.extend(sampled_strong)
                batch.extend(sampled_weak)
                batch.extend(sampled_neg)
            self.last_epoch_stats = {
                "anchors_with_positive_ratio": float(stats["anchors_with_positive"] / max(stats["anchors"], 1.0)),
                "avg_strong_pos_per_batch": float(stats["strong_pos"] / max(self.batches_per_epoch, 1)),
                "avg_weak_pos_per_batch": float(stats["weak_pos"] / max(self.batches_per_epoch, 1)),
                "avg_hard_neg_per_batch": float(stats["hard_neg"] / max(self.batches_per_epoch, 1)),
            }
            yield batch


def collate_relation(batch: list[RelationSample]) -> dict:
    return {
        "indices": [int(sample.index) for sample in batch],
        "img_ids": [sample.img_id for sample in batch],
        "images": torch.stack([sample.image for sample in batch]),
        "blood_id_indices": [sample.blood_id_indices for sample in batch],
        "blood_names": [sample.blood_name for sample in batch],
        "pg_ids": [sample.pg_id for sample in batch],
    }
