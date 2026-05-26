from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve


EMPTY_METRICS = {"accuracy": 0.0, "balanced_accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.0}
EMPTY_SEARCH_METRICS = {"recall_at_1": 0.0, "recall_at_5": 0.0, "recall_at_10": 0.0, "mAP": 0.0}


def _read_pigeon_blood_names(path: str | Path) -> pd.DataFrame:
    try:
        pigeon = pd.read_csv(path, dtype={"ID": str})
    except pd.errors.ParserError:
        pigeon = pd.read_csv(path, dtype={"ID": str}, engine="python", on_bad_lines="skip")
        print(f"warning: skipped malformed rows while reading {path}")
    required = {"ID", "BLOOD"}
    missing = required - set(pigeon.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    pigeon = pigeon[["ID", "BLOOD"]].copy()
    pigeon["ID"] = pigeon["ID"].astype(str).str.strip()
    pigeon["BLOOD"] = pigeon["BLOOD"].fillna("").astype(str).str.strip()
    pigeon = pigeon[(pigeon["ID"] != "") & (pigeon["BLOOD"] != "")]
    return pigeon.drop_duplicates(subset=["ID"], keep="first")


def load_related_blood_names(
    relations_path: str | Path,
    pigeon_csv: str | Path,
) -> dict[str, frozenset[str]]:
    rel = pd.read_csv(
        relations_path,
        header=None,
        names=["blood_id", "img_id"],
        dtype={"blood_id": str, "img_id": str},
    )
    rel = rel.dropna(subset=["blood_id", "img_id"]).copy()
    rel["blood_id"] = rel["blood_id"].astype(str).str.strip()
    rel["img_id"] = rel["img_id"].astype(str).str.strip()
    rel = rel[(rel["blood_id"] != "") & (rel["img_id"] != "")]
    rel = rel.drop_duplicates(subset=["blood_id", "img_id"])

    pigeon = _read_pigeon_blood_names(pigeon_csv)
    rel_with_names = rel.merge(pigeon, left_on="img_id", right_on="ID", how="inner")

    blood_id_to_names = {
        str(blood_id): frozenset(group["BLOOD"].astype(str))
        for blood_id, group in rel_with_names.groupby("blood_id", sort=False)
    }

    related: dict[str, frozenset[str]] = {}
    for img_id, group in rel.groupby("img_id", sort=False):
        names: set[str] = set()
        for blood_id in group["blood_id"].astype(str):
            names.update(blood_id_to_names.get(blood_id, frozenset()))
        if names:
            related[str(img_id)] = frozenset(names)
    return related


def _ordered_related_sets(
    img_ids: list[str] | np.ndarray,
    related_blood_names: dict[str, frozenset[str]],
) -> list[frozenset[str]]:
    return [related_blood_names.get(str(img_id), frozenset()) for img_id in img_ids]


def _normalize_names(blood_names: list[str] | np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray([str(value).strip() for value in blood_names], dtype=object)


def _is_related(
    left_name: str,
    left_related_names: frozenset[str],
    right_name: str,
    right_related_names: frozenset[str],
) -> bool:
    return (right_name in left_related_names) or (left_name in right_related_names)


def _finalize_compare_metrics(dists: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    if len(dists) == 0 or len(np.unique(y_true)) < 2:
        return dict(EMPTY_METRICS)

    thresholds = np.unique(np.quantile(dists, np.linspace(0, 1, 256)))
    best_threshold = float(thresholds[0])
    best_acc = -1.0
    best_bal = -1.0
    for threshold in thresholds:
        preds = (dists <= threshold).astype(np.int32)
        acc = accuracy_score(y_true, preds)
        bal = balanced_accuracy_score(y_true, preds)
        if bal > best_bal:
            best_threshold = float(threshold)
            best_acc = float(acc)
            best_bal = float(bal)

    try:
        auc = float(roc_auc_score(y_true, -dists))
    except ValueError:
        auc = 0.0
    fpr, tpr, _roc_thresholds = roc_curve(y_true, -dists)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    return {"accuracy": best_acc, "balanced_accuracy": best_bal, "auc": auc, "eer": eer, "threshold": best_threshold}


def _name_index(blood_names: np.ndarray) -> dict[str, list[int]]:
    by_name: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(blood_names):
        if name:
            by_name[str(name)].append(index)
    return dict(by_name)


def _relevant_mask(
    query_related_names: frozenset[str],
    candidate_blood_names: np.ndarray,
    exclude_index: int | None = None,
) -> np.ndarray:
    relevant = np.asarray([str(name) in query_related_names for name in candidate_blood_names], dtype=bool)
    if exclude_index is not None and 0 <= exclude_index < len(relevant):
        relevant[exclude_index] = False
    return relevant


def compute_search_metrics_by_related_breeds(
    features: np.ndarray,
    img_ids: list[str] | np.ndarray,
    blood_names: list[str] | np.ndarray,
    related_blood_names: dict[str, frozenset[str]],
) -> dict[str, float]:
    n = len(img_ids)
    if n < 2:
        return dict(EMPTY_SEARCH_METRICS)

    related_sets = _ordered_related_sets(img_ids, related_blood_names)
    names = _normalize_names(blood_names)
    norms = np.sum(features * features, axis=1, keepdims=True)
    dist2 = norms + norms.T - 2.0 * (features @ features.T)
    np.fill_diagonal(dist2, np.inf)
    order = np.argsort(dist2, axis=1)

    recall_hits = {1: 0, 5: 0, 10: 0}
    aps: list[float] = []
    valid = 0
    for i in range(n):
        query_related = related_sets[i]
        if not query_related:
            continue
        relevant = _relevant_mask(query_related, names, exclude_index=i)
        relevant_total = int(np.sum(relevant))
        if relevant_total == 0:
            continue
        valid += 1
        ranked_relevant = relevant[order[i]]
        for k in (1, 5, 10):
            recall_hits[k] += int(np.any(ranked_relevant[: min(k, len(ranked_relevant))]))
        hits = 0
        precision_sum = 0.0
        for rank, is_relevant in enumerate(ranked_relevant, start=1):
            if is_relevant:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / relevant_total)

    return {
        "recall_at_1": recall_hits[1] / max(valid, 1),
        "recall_at_5": recall_hits[5] / max(valid, 1),
        "recall_at_10": recall_hits[10] / max(valid, 1),
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }


def compute_cross_search_metrics_by_related_breeds(
    query_features: np.ndarray,
    query_img_ids: list[str] | np.ndarray,
    gallery_features: np.ndarray,
    gallery_blood_names: list[str] | np.ndarray,
    related_blood_names: dict[str, frozenset[str]],
    chunk_size: int = 256,
) -> dict[str, float]:
    if len(query_img_ids) == 0 or len(gallery_blood_names) == 0:
        return dict(EMPTY_SEARCH_METRICS)

    query_related_sets = _ordered_related_sets(query_img_ids, related_blood_names)
    gallery_names = _normalize_names(gallery_blood_names)
    gallery_norms = np.sum(gallery_features * gallery_features, axis=1)
    recall_hits = {1: 0, 5: 0, 10: 0}
    aps: list[float] = []
    valid = 0

    for start in range(0, len(query_features), int(chunk_size)):
        end = min(start + int(chunk_size), len(query_features))
        query_chunk = query_features[start:end]
        query_norms = np.sum(query_chunk * query_chunk, axis=1, keepdims=True)
        dist2 = query_norms + gallery_norms[None, :] - 2.0 * (query_chunk @ gallery_features.T)
        order = np.argsort(dist2, axis=1)
        for local_i, ranked_idx in enumerate(order):
            query_related = query_related_sets[start + local_i]
            if not query_related:
                continue
            relevant = _relevant_mask(query_related, gallery_names)
            relevant_total = int(np.sum(relevant))
            if relevant_total == 0:
                continue
            valid += 1
            ranked_relevant = relevant[ranked_idx]
            for k in (1, 5, 10):
                recall_hits[k] += int(np.any(ranked_relevant[: min(k, len(ranked_relevant))]))
            hits = 0
            precision_sum = 0.0
            for rank, is_relevant in enumerate(ranked_relevant, start=1):
                if is_relevant:
                    hits += 1
                    precision_sum += hits / rank
            aps.append(precision_sum / relevant_total)

    return {
        "recall_at_1": recall_hits[1] / max(valid, 1),
        "recall_at_5": recall_hits[5] / max(valid, 1),
        "recall_at_10": recall_hits[10] / max(valid, 1),
        "mAP": float(np.mean(aps)) if aps else 0.0,
    }


def compute_compare_metrics_by_related_breeds(
    features: np.ndarray,
    img_ids: list[str] | np.ndarray,
    blood_names: list[str] | np.ndarray,
    related_blood_names: dict[str, frozenset[str]],
    max_pairs: int = 200000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    names = _normalize_names(blood_names)
    related_sets = _ordered_related_sets(img_ids, related_blood_names)
    valid_indices = [idx for idx, related_names in enumerate(related_sets) if related_names and names[idx]]
    if len(valid_indices) < 2:
        return dict(EMPTY_METRICS)

    name_to_indices = _name_index(names)
    target_pos = int(max_pairs) // 2
    positive_pairs_set: set[tuple[int, int]] = set()
    for i in valid_indices:
        candidates: set[int] = set()
        for related_name in related_sets[i]:
            candidates.update(name_to_indices.get(related_name, []))
        for j in sorted(candidates):
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            positive_pairs_set.add(pair)
            if len(positive_pairs_set) >= target_pos:
                break
        if len(positive_pairs_set) >= target_pos:
            break
    if not positive_pairs_set:
        return dict(EMPTY_METRICS)

    positive_pairs = sorted(positive_pairs_set)
    target_neg = min(len(positive_pairs), target_pos)
    negative_pairs_set: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(target_neg * 50, 1000)
    while len(negative_pairs_set) < target_neg and attempts < max_attempts:
        attempts += 1
        i, j = [int(v) for v in rng.choice(valid_indices, size=2, replace=False)]
        pair = (i, j) if i < j else (j, i)
        if pair in positive_pairs_set or pair in negative_pairs_set:
            continue
        if not _is_related(names[i], related_sets[i], names[j], related_sets[j]):
            negative_pairs_set.add(pair)

    negative_pairs = sorted(negative_pairs_set)
    pair_n = min(len(positive_pairs), len(negative_pairs))
    if pair_n == 0:
        return dict(EMPTY_METRICS)

    pairs = positive_pairs[:pair_n] + negative_pairs[:pair_n]
    y_true = np.asarray([1] * pair_n + [0] * pair_n, dtype=np.int32)
    dists = np.asarray([float(np.linalg.norm(features[i] - features[j])) for i, j in pairs], dtype=np.float32)
    return _finalize_compare_metrics(dists, y_true)


def compute_cross_compare_metrics_by_related_breeds(
    query_features: np.ndarray,
    query_img_ids: list[str] | np.ndarray,
    query_blood_names: list[str] | np.ndarray,
    gallery_features: np.ndarray,
    gallery_img_ids: list[str] | np.ndarray,
    gallery_blood_names: list[str] | np.ndarray,
    related_blood_names: dict[str, frozenset[str]],
    max_pairs: int = 200000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    query_names = _normalize_names(query_blood_names)
    gallery_names = _normalize_names(gallery_blood_names)
    query_related_sets = _ordered_related_sets(query_img_ids, related_blood_names)
    gallery_related_sets = _ordered_related_sets(gallery_img_ids, related_blood_names)
    gallery_name_to_indices = _name_index(gallery_names)
    gallery_related_name_to_indices: dict[str, list[int]] = defaultdict(list)
    for gallery_idx, related_names in enumerate(gallery_related_sets):
        for related_name in related_names:
            gallery_related_name_to_indices[related_name].append(gallery_idx)
    valid_queries = [idx for idx, related_names in enumerate(query_related_sets) if related_names and query_names[idx]]
    valid_gallery = [idx for idx, related_names in enumerate(gallery_related_sets) if related_names and gallery_names[idx]]
    if not valid_queries or not valid_gallery:
        return dict(EMPTY_METRICS)

    positive_pairs: list[tuple[int, int]] = []
    query_order = np.asarray(valid_queries, dtype=np.int64)
    rng.shuffle(query_order)
    target_pos = int(max_pairs) // 2
    for query_idx in query_order:
        candidates: set[int] = set()
        for related_name in query_related_sets[int(query_idx)]:
            candidates.update(gallery_name_to_indices.get(related_name, []))
        candidates.update(gallery_related_name_to_indices.get(str(query_names[int(query_idx)]), []))
        if not candidates:
            continue
        gallery_idx = int(rng.choice(sorted(candidates)))
        positive_pairs.append((int(query_idx), gallery_idx))
        if len(positive_pairs) >= target_pos:
            break

    if not positive_pairs:
        return dict(EMPTY_METRICS)

    target_neg = min(len(positive_pairs), target_pos)
    negative_pairs_set: set[tuple[int, int]] = set()
    attempts = 0
    max_attempts = max(target_neg * 50, 1000)
    while len(negative_pairs_set) < target_neg and attempts < max_attempts:
        attempts += 1
        query_idx = int(rng.choice(valid_queries))
        gallery_idx = int(rng.choice(valid_gallery))
        pair = (query_idx, gallery_idx)
        if pair in negative_pairs_set:
            continue
        if not _is_related(
            query_names[query_idx],
            query_related_sets[query_idx],
            gallery_names[gallery_idx],
            gallery_related_sets[gallery_idx],
        ):
            negative_pairs_set.add(pair)

    negative_pairs = sorted(negative_pairs_set)
    pair_n = min(len(positive_pairs), len(negative_pairs))
    if pair_n == 0:
        return dict(EMPTY_METRICS)

    pairs = positive_pairs[:pair_n] + negative_pairs[:pair_n]
    y_true = np.asarray([1] * pair_n + [0] * pair_n, dtype=np.int32)
    dists = np.asarray(
        [float(np.linalg.norm(query_features[q] - gallery_features[g])) for q, g in pairs],
        dtype=np.float32,
    )
    return _finalize_compare_metrics(dists, y_true)


def compute_probe_recall_by_related_breeds(
    features: np.ndarray,
    img_ids: list[str] | np.ndarray,
    blood_names: list[str] | np.ndarray,
    related_blood_names: dict[str, frozenset[str]],
) -> float:
    names = _normalize_names(blood_names)
    related_sets = _ordered_related_sets(img_ids, related_blood_names)
    valid_indices = [idx for idx, related_names in enumerate(related_sets) if related_names and names[idx]]
    if len(valid_indices) < 2:
        return 0.0

    hits = 0
    valid = 0
    for query in valid_indices:
        gallery_indices = [idx for idx in valid_indices if idx != query]
        if not gallery_indices:
            continue
        valid += 1
        gallery = features[gallery_indices]
        distances = np.linalg.norm(gallery - features[query], axis=1)
        nearest = gallery_indices[int(np.argmin(distances))]
        hits += int(names[nearest] in related_sets[query])
    return hits / max(valid, 1)
