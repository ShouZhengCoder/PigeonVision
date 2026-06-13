from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, roc_curve


EMPTY_METRICS = {"accuracy": 0.0, "balanced_accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.0}
SEARCH_KS = (1, 5, 10)
GRADED_SEARCH_KS = (1, 5, 10)


def _empty_search_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {"valid_queries": 0.0, "mean_relevant_total": 0.0, "mAP": 0.0}
    for k in SEARCH_KS:
        metrics[f"hit_at_{k}"] = 0.0
        metrics[f"recall_at_{k}"] = 0.0
        metrics[f"avg_relevant_at_{k}"] = 0.0
        metrics[f"precision_at_{k}"] = 0.0
        metrics[f"ndcg_at_{k}"] = 0.0
    return metrics


EMPTY_SEARCH_METRICS = _empty_search_metrics()


def _init_search_acc() -> dict[str, object]:
    return {
        "valid": 0,
        "relevant_total_sum": 0,
        "ap_sum": 0.0,
        "hit": {k: 0.0 for k in SEARCH_KS},
        "recall": {k: 0.0 for k in SEARCH_KS},
        "avg_relevant": {k: 0.0 for k in SEARCH_KS},
        "precision": {k: 0.0 for k in SEARCH_KS},
        "ndcg": {k: 0.0 for k in SEARCH_KS},
    }


def _ndcg_at(ranked_relevant: np.ndarray, relevant_total: int, k: int) -> float:
    if relevant_total <= 0:
        return 0.0
    top = ranked_relevant[: min(k, len(ranked_relevant))].astype(np.float32)
    if len(top) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2, dtype=np.float32))
    dcg = float(np.sum(top * discounts))
    ideal_len = min(int(relevant_total), int(k))
    if ideal_len <= 0:
        return 0.0
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_len + 2, dtype=np.float32))
    idcg = float(np.sum(ideal_discounts))
    return dcg / idcg if idcg > 0 else 0.0


def _graded_ndcg_at(ranked_relevance: np.ndarray, ideal_relevance: np.ndarray, k: int) -> float:
    top = np.asarray(ranked_relevance[: min(int(k), len(ranked_relevance))], dtype=np.float32)
    if len(top) == 0:
        return 0.0
    ideal = np.sort(np.asarray(ideal_relevance, dtype=np.float32))[::-1][: len(top)]
    if len(ideal) == 0 or float(np.sum(ideal)) <= 0.0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(top) + 2, dtype=np.float32))
    dcg = float(np.sum(top * discounts))
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision(ranked_relevant: np.ndarray, relevant_total: int) -> float:
    if relevant_total <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(ranked_relevant, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / relevant_total


def build_blood_id_idf(
    blood_id_sets: dict[str, frozenset[str]],
    img_ids: list[str] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute IDF for blood_ids over the evaluation/training universe."""
    if img_ids is None:
        selected_sets = list(blood_id_sets.values())
    else:
        selected_sets = [blood_id_sets.get(str(img_id), frozenset()) for img_id in img_ids]
    n = len(selected_sets)
    df: dict[str, int] = defaultdict(int)
    for ids in selected_sets:
        for blood_id in ids:
            df[str(blood_id)] += 1
    return {blood_id: float(np.log((n + 1.0) / (count + 1.0))) for blood_id, count in df.items()}


def blood_set_weight(blood_ids: frozenset[str] | set[str], idf: dict[str, float]) -> float:
    return float(sum(float(idf.get(str(blood_id), 0.0)) for blood_id in blood_ids))


def weighted_jaccard(
    left: frozenset[str] | set[str],
    right: frozenset[str] | set[str],
    idf: dict[str, float],
) -> float:
    if not left or not right:
        return 0.0
    union = set(left) | set(right)
    if not union:
        return 0.0
    denom = blood_set_weight(union, idf)
    if denom <= 0:
        return 0.0
    return float(blood_set_weight(set(left) & set(right), idf) / denom)


def weighted_overlap(
    left: frozenset[str] | set[str],
    right: frozenset[str] | set[str],
    idf: dict[str, float],
) -> float:
    if not left or not right:
        return 0.0
    left_weight = blood_set_weight(left, idf)
    right_weight = blood_set_weight(right, idf)
    denom = min(left_weight, right_weight)
    if denom <= 0:
        return 0.0
    return float(blood_set_weight(set(left) & set(right), idf) / denom)


def blood_id_relevance(
    left: frozenset[str] | set[str],
    right: frozenset[str] | set[str],
    idf: dict[str, float],
) -> float:
    """IDF-weighted graded bloodline relevance in [0, 1]."""
    if not left or not right or not (set(left) & set(right)):
        return 0.0
    return float(0.7 * weighted_overlap(left, right, idf) + 0.3 * weighted_jaccard(left, right, idf))


def shared_idf_quantile(
    left: frozenset[str] | set[str],
    right: frozenset[str] | set[str],
    idf: dict[str, float],
    quantiles: tuple[float, float, float] | None = None,
) -> str:
    shared = [float(idf.get(str(blood_id), 0.0)) for blood_id in (set(left) & set(right))]
    if not shared:
        return "none"
    if quantiles is None:
        values = np.asarray([float(v) for v in idf.values()], dtype=np.float32)
        if len(values) == 0:
            return "unknown"
        q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    else:
        q25, q50, q75 = quantiles
    score = max(shared)
    if score >= q75:
        return "q4_rare"
    if score >= q50:
        return "q3"
    if score >= q25:
        return "q2"
    return "q1_common"


def _build_inverted_index(sets: list[frozenset[str]]) -> dict[str, list[int]]:
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, blood_ids in enumerate(sets):
        for blood_id in blood_ids:
            inverted[str(blood_id)].append(idx)
    return dict(inverted)


def _candidate_relevance_scores(
    query_set: frozenset[str],
    gallery_sets: list[frozenset[str]],
    gallery_inverted: dict[str, list[int]],
    idf: dict[str, float],
    exclude_index: int | None = None,
) -> dict[int, float]:
    if not query_set:
        return {}
    candidates: set[int] = set()
    for blood_id in query_set:
        candidates.update(gallery_inverted.get(str(blood_id), []))
    if exclude_index is not None:
        candidates.discard(int(exclude_index))
    scores: dict[int, float] = {}
    for idx in candidates:
        score = blood_id_relevance(query_set, gallery_sets[int(idx)], idf)
        if score > 0.0:
            scores[int(idx)] = float(score)
    return scores


def _update_search_acc(acc: dict[str, object], ranked_relevant: np.ndarray, relevant_total: int) -> None:
    acc["valid"] = int(acc["valid"]) + 1
    acc["relevant_total_sum"] = int(acc["relevant_total_sum"]) + int(relevant_total)
    acc["ap_sum"] = float(acc["ap_sum"]) + _average_precision(ranked_relevant, relevant_total)

    hit = acc["hit"]
    recall = acc["recall"]
    avg_relevant = acc["avg_relevant"]
    precision = acc["precision"]
    ndcg = acc["ndcg"]
    assert isinstance(hit, dict)
    assert isinstance(recall, dict)
    assert isinstance(avg_relevant, dict)
    assert isinstance(precision, dict)
    assert isinstance(ndcg, dict)

    for k in SEARCH_KS:
        relevant_count = int(np.sum(ranked_relevant[: min(k, len(ranked_relevant))]))
        hit[k] += float(relevant_count > 0)
        recall[k] += float(relevant_count / max(int(relevant_total), 1))
        avg_relevant[k] += float(relevant_count)
        precision[k] += float(relevant_count / k)
        ndcg[k] += _ndcg_at(ranked_relevant, relevant_total, k)


def _finalize_search_acc(acc: dict[str, object]) -> dict[str, float]:
    valid = int(acc["valid"])
    if valid <= 0:
        return dict(EMPTY_SEARCH_METRICS)

    hit = acc["hit"]
    recall = acc["recall"]
    avg_relevant = acc["avg_relevant"]
    precision = acc["precision"]
    ndcg = acc["ndcg"]
    assert isinstance(hit, dict)
    assert isinstance(recall, dict)
    assert isinstance(avg_relevant, dict)
    assert isinstance(precision, dict)
    assert isinstance(ndcg, dict)

    metrics = {
        "valid_queries": float(valid),
        "mean_relevant_total": float(int(acc["relevant_total_sum"]) / valid),
        "mAP": float(float(acc["ap_sum"]) / valid),
    }
    for k in SEARCH_KS:
        metrics[f"hit_at_{k}"] = float(hit[k] / valid)
        metrics[f"recall_at_{k}"] = float(recall[k] / valid)
        metrics[f"avg_relevant_at_{k}"] = float(avg_relevant[k] / valid)
        metrics[f"precision_at_{k}"] = float(precision[k] / valid)
        metrics[f"ndcg_at_{k}"] = float(ndcg[k] / valid)
    return metrics


def _empty_graded_search_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {
        "valid_queries": 0.0,
        "skipped_queries": 0.0,
        "mean_relevant_total": 0.0,
        "median_relevant_total": 0.0,
        "p90_relevant_total": 0.0,
        "mAP": 0.0,
    }
    for k in GRADED_SEARCH_KS:
        metrics[f"hit_at_{k}"] = 0.0
        metrics[f"precision_at_{k}"] = 0.0
        metrics[f"avg_relevant_at_{k}"] = 0.0
        metrics[f"recall_at_{k}"] = 0.0
        metrics[f"binary_ndcg_at_{k}"] = 0.0
        metrics[f"graded_ndcg_at_{k}"] = 0.0
        metrics[f"avg_relevance_at_{k}"] = 0.0
    return metrics


EMPTY_GRADED_SEARCH_METRICS = _empty_graded_search_metrics()


def _init_graded_search_acc() -> dict[str, object]:
    return {
        "valid": 0,
        "skipped": 0,
        "relevant_totals": [],
        "ap_sum": 0.0,
        "hit": {k: 0.0 for k in GRADED_SEARCH_KS},
        "precision": {k: 0.0 for k in GRADED_SEARCH_KS},
        "avg_relevant": {k: 0.0 for k in GRADED_SEARCH_KS},
        "recall": {k: 0.0 for k in GRADED_SEARCH_KS},
        "binary_ndcg": {k: 0.0 for k in GRADED_SEARCH_KS},
        "graded_ndcg": {k: 0.0 for k in GRADED_SEARCH_KS},
        "avg_relevance": {k: 0.0 for k in GRADED_SEARCH_KS},
    }


def _update_graded_search_acc(
    acc: dict[str, object],
    ranked_relevance: np.ndarray,
    ideal_relevance: np.ndarray,
) -> None:
    ranked_relevance = np.asarray(ranked_relevance, dtype=np.float32)
    ideal_relevance = np.asarray(ideal_relevance, dtype=np.float32)
    relevant_total = int(np.sum(ideal_relevance > 0.0))
    if relevant_total <= 0:
        acc["skipped"] = int(acc["skipped"]) + 1
        return

    acc["valid"] = int(acc["valid"]) + 1
    totals = acc["relevant_totals"]
    assert isinstance(totals, list)
    totals.append(relevant_total)
    acc["ap_sum"] = float(acc["ap_sum"]) + _average_precision(ranked_relevance > 0.0, relevant_total)

    hit = acc["hit"]
    precision = acc["precision"]
    avg_relevant = acc["avg_relevant"]
    recall = acc["recall"]
    binary_ndcg = acc["binary_ndcg"]
    graded_ndcg = acc["graded_ndcg"]
    avg_relevance = acc["avg_relevance"]
    assert isinstance(hit, dict)
    assert isinstance(precision, dict)
    assert isinstance(avg_relevant, dict)
    assert isinstance(recall, dict)
    assert isinstance(binary_ndcg, dict)
    assert isinstance(graded_ndcg, dict)
    assert isinstance(avg_relevance, dict)

    for k in GRADED_SEARCH_KS:
        top = ranked_relevance[: min(k, len(ranked_relevance))]
        relevant_count = int(np.sum(top > 0.0))
        relevance_sum = float(np.sum(top))
        hit[k] += float(relevant_count > 0)
        precision[k] += float(relevant_count / k)
        avg_relevant[k] += float(relevant_count)
        recall[k] += float(relevant_count / max(relevant_total, 1))
        binary_ndcg[k] += _ndcg_at(ranked_relevance > 0.0, relevant_total, k)
        graded_ndcg[k] += _graded_ndcg_at(ranked_relevance, ideal_relevance, k)
        avg_relevance[k] += float(relevance_sum / k)


def _finalize_graded_search_acc(acc: dict[str, object]) -> dict[str, float]:
    valid = int(acc["valid"])
    if valid <= 0:
        return dict(EMPTY_GRADED_SEARCH_METRICS)

    totals = np.asarray(acc["relevant_totals"], dtype=np.float32)
    hit = acc["hit"]
    precision = acc["precision"]
    avg_relevant = acc["avg_relevant"]
    recall = acc["recall"]
    binary_ndcg = acc["binary_ndcg"]
    graded_ndcg = acc["graded_ndcg"]
    avg_relevance = acc["avg_relevance"]
    assert isinstance(hit, dict)
    assert isinstance(precision, dict)
    assert isinstance(avg_relevant, dict)
    assert isinstance(recall, dict)
    assert isinstance(binary_ndcg, dict)
    assert isinstance(graded_ndcg, dict)
    assert isinstance(avg_relevance, dict)

    metrics = {
        "valid_queries": float(valid),
        "skipped_queries": float(int(acc["skipped"])),
        "mean_relevant_total": float(np.mean(totals)) if len(totals) else 0.0,
        "median_relevant_total": float(np.median(totals)) if len(totals) else 0.0,
        "p90_relevant_total": float(np.quantile(totals, 0.90)) if len(totals) else 0.0,
        "mAP": float(float(acc["ap_sum"]) / valid),
    }
    for k in GRADED_SEARCH_KS:
        metrics[f"hit_at_{k}"] = float(hit[k] / valid)
        metrics[f"precision_at_{k}"] = float(precision[k] / valid)
        metrics[f"avg_relevant_at_{k}"] = float(avg_relevant[k] / valid)
        metrics[f"recall_at_{k}"] = float(recall[k] / valid)
        metrics[f"binary_ndcg_at_{k}"] = float(binary_ndcg[k] / valid)
        metrics[f"graded_ndcg_at_{k}"] = float(graded_ndcg[k] / valid)
        metrics[f"avg_relevance_at_{k}"] = float(avg_relevance[k] / valid)
    return metrics


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

    acc = _init_search_acc()
    for i in range(n):
        query_related = related_sets[i]
        if not query_related:
            continue
        relevant = _relevant_mask(query_related, names, exclude_index=i)
        relevant_total = int(np.sum(relevant))
        if relevant_total == 0:
            continue
        ranked_relevant = relevant[order[i]]
        _update_search_acc(acc, ranked_relevant, relevant_total)

    return _finalize_search_acc(acc)


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
    acc = _init_search_acc()

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
            ranked_relevant = relevant[ranked_idx]
            _update_search_acc(acc, ranked_relevant, relevant_total)

    return _finalize_search_acc(acc)


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


# ---------------------------------------------------------------------------
# Blood-ID level multi-label evaluation
# Each image may have multiple blood_ids. Two images are "related" if they
# share at least one blood_id.
# ---------------------------------------------------------------------------

def load_blood_id_sets(relations_path: str | Path) -> dict[str, frozenset[str]]:
    rel = pd.read_csv(
        relations_path,
        header=None,
        names=["blood_id", "img_id"],
        dtype={"blood_id": str, "img_id": str},
    )
    rel = rel.dropna(subset=["blood_id", "img_id"])
    rel["blood_id"] = rel["blood_id"].astype(str).str.strip()
    rel["img_id"] = rel["img_id"].astype(str).str.strip()
    rel = rel[(rel["blood_id"] != "") & (rel["img_id"] != "")]
    rel = rel.drop_duplicates(subset=["blood_id", "img_id"])

    img_to_bloods: dict[str, set[str]] = defaultdict(set)
    for row in rel.itertuples(index=False):
        img_to_bloods[str(row.img_id)].add(str(row.blood_id))
    return {img_id: frozenset(bloods) for img_id, bloods in img_to_bloods.items()}


def _ordered_blood_id_sets(
    img_ids: list[str] | np.ndarray,
    blood_id_sets: dict[str, frozenset[str]],
) -> list[frozenset[str]]:
    return [blood_id_sets.get(str(img_id), frozenset()) for img_id in img_ids]


def _blood_id_relevant_mask(
    query_blood_ids: frozenset[str],
    gallery_blood_id_sets: list[frozenset[str]],
    exclude_index: int | None = None,
) -> np.ndarray:
    relevant = np.asarray([bool(query_blood_ids & g_set) for g_set in gallery_blood_id_sets], dtype=bool)
    if exclude_index is not None and 0 <= exclude_index < len(relevant):
        relevant[exclude_index] = False
    return relevant


def compute_cross_search_metrics_by_sets(
    query_features: np.ndarray,
    query_sets: list[frozenset[str]],
    gallery_features: np.ndarray,
    gallery_sets: list[frozenset[str]],
    chunk_size: int = 256,
) -> dict[str, float]:
    if len(query_features) == 0 or len(gallery_features) == 0:
        return dict(EMPTY_SEARCH_METRICS)
    if len(query_features) != len(query_sets):
        raise ValueError("query_features and query_sets length mismatch")
    if len(gallery_features) != len(gallery_sets):
        raise ValueError("gallery_features and gallery_sets length mismatch")

    gallery_norms = np.sum(gallery_features * gallery_features, axis=1)
    acc = _init_search_acc()
    for start in range(0, len(query_features), int(chunk_size)):
        end = min(start + int(chunk_size), len(query_features))
        query_chunk = query_features[start:end]
        query_norms = np.sum(query_chunk * query_chunk, axis=1, keepdims=True)
        dist2 = query_norms + gallery_norms[None, :] - 2.0 * (query_chunk @ gallery_features.T)
        order = np.argsort(dist2, axis=1)
        for local_i, ranked_idx in enumerate(order):
            query_set = query_sets[start + local_i]
            if not query_set:
                continue
            relevant = _blood_id_relevant_mask(query_set, gallery_sets)
            relevant_total = int(np.sum(relevant))
            if relevant_total == 0:
                continue
            ranked_relevant = relevant[ranked_idx]
            _update_search_acc(acc, ranked_relevant, relevant_total)
    return _finalize_search_acc(acc)


def compute_search_metrics_by_blood_ids(
    features: np.ndarray,
    img_ids: list[str] | np.ndarray,
    blood_id_sets: dict[str, frozenset[str]],
) -> dict[str, float]:
    n = len(img_ids)
    if n < 2:
        return dict(EMPTY_SEARCH_METRICS)

    ordered_sets = _ordered_blood_id_sets(img_ids, blood_id_sets)
    norms = np.sum(features * features, axis=1, keepdims=True)
    dist2 = norms + norms.T - 2.0 * (features @ features.T)
    np.fill_diagonal(dist2, np.inf)
    order = np.argsort(dist2, axis=1)

    acc = _init_search_acc()
    for i in range(n):
        query_set = ordered_sets[i]
        if not query_set:
            continue
        relevant = _blood_id_relevant_mask(query_set, ordered_sets, exclude_index=i)
        relevant_total = int(np.sum(relevant))
        if relevant_total == 0:
            continue
        ranked_relevant = relevant[order[i]]
        _update_search_acc(acc, ranked_relevant, relevant_total)

    return _finalize_search_acc(acc)


def compute_cross_search_metrics_by_blood_ids(
    query_features: np.ndarray,
    query_img_ids: list[str] | np.ndarray,
    gallery_features: np.ndarray,
    gallery_img_ids: list[str] | np.ndarray,
    blood_id_sets: dict[str, frozenset[str]],
    chunk_size: int = 256,
    exclude_self: bool = False,
) -> dict[str, float]:
    if len(query_img_ids) == 0 or len(gallery_img_ids) == 0:
        return dict(EMPTY_SEARCH_METRICS)

    query_sets = _ordered_blood_id_sets(query_img_ids, blood_id_sets)
    gallery_sets = _ordered_blood_id_sets(gallery_img_ids, blood_id_sets)
    gallery_norms = np.sum(gallery_features * gallery_features, axis=1)
    acc = _init_search_acc()

    # Pre-compute self-exclusion mapping when query == gallery
    self_exclude: dict[int, int] = {}
    if exclude_self:
        q_ids = [str(x) for x in query_img_ids]
        g_ids = [str(x) for x in gallery_img_ids]
        id_to_gidx = {id_: i for i, id_ in enumerate(g_ids)}
        for qi, qid in enumerate(q_ids):
            if qid in id_to_gidx:
                self_exclude[qi] = id_to_gidx[qid]

    for start in range(0, len(query_features), int(chunk_size)):
        end = min(start + int(chunk_size), len(query_features))
        query_chunk = query_features[start:end]
        query_norms = np.sum(query_chunk * query_chunk, axis=1, keepdims=True)
        dist2 = query_norms + gallery_norms[None, :] - 2.0 * (query_chunk @ gallery_features.T)
        # Exclude self-matches
        if exclude_self:
            for qi in range(start, end):
                if qi in self_exclude:
                    dist2[qi - start, self_exclude[qi]] = np.inf
        order = np.argsort(dist2, axis=1)
        for local_i, ranked_idx in enumerate(order):
            query_set = query_sets[start + local_i]
            if not query_set:
                continue
            exclude_index = self_exclude.get(start + local_i)
            relevant = _blood_id_relevant_mask(query_set, gallery_sets, exclude_index=exclude_index)
            relevant_total = int(np.sum(relevant))
            if relevant_total == 0:
                continue
            ranked_relevant = relevant[ranked_idx]
            _update_search_acc(acc, ranked_relevant, relevant_total)

    return _finalize_search_acc(acc)


def compute_cross_search_metrics_by_graded_blood_ids(
    query_features: np.ndarray,
    query_img_ids: list[str] | np.ndarray,
    gallery_features: np.ndarray,
    gallery_img_ids: list[str] | np.ndarray,
    blood_id_sets: dict[str, frozenset[str]],
    idf: dict[str, float] | None = None,
    chunk_size: int = 256,
    exclude_self: bool = False,
) -> dict[str, float]:
    """Compute Top-K retrieval metrics with continuous blood-ID relevance."""
    if len(query_img_ids) == 0 or len(gallery_img_ids) == 0:
        return dict(EMPTY_GRADED_SEARCH_METRICS)
    if len(query_features) != len(query_img_ids):
        raise ValueError("query_features and query_img_ids length mismatch")
    if len(gallery_features) != len(gallery_img_ids):
        raise ValueError("gallery_features and gallery_img_ids length mismatch")

    query_sets = _ordered_blood_id_sets(query_img_ids, blood_id_sets)
    gallery_sets = _ordered_blood_id_sets(gallery_img_ids, blood_id_sets)
    if idf is None:
        universe_ids = [str(x) for x in query_img_ids] + [str(x) for x in gallery_img_ids]
        idf = build_blood_id_idf(blood_id_sets, universe_ids)
    gallery_inverted = _build_inverted_index(gallery_sets)
    gallery_norms = np.sum(gallery_features * gallery_features, axis=1)
    acc = _init_graded_search_acc()

    self_exclude: dict[int, int] = {}
    if exclude_self:
        id_to_gidx = {str(img_id): idx for idx, img_id in enumerate(gallery_img_ids)}
        for query_idx, img_id in enumerate(query_img_ids):
            if str(img_id) in id_to_gidx:
                self_exclude[query_idx] = id_to_gidx[str(img_id)]

    for start in range(0, len(query_features), int(chunk_size)):
        end = min(start + int(chunk_size), len(query_features))
        query_chunk = query_features[start:end]
        query_norms = np.sum(query_chunk * query_chunk, axis=1, keepdims=True)
        dist2 = query_norms + gallery_norms[None, :] - 2.0 * (query_chunk @ gallery_features.T)
        if exclude_self:
            for query_idx in range(start, end):
                gallery_idx = self_exclude.get(query_idx)
                if gallery_idx is not None:
                    dist2[query_idx - start, gallery_idx] = np.inf
        order = np.argsort(dist2, axis=1)
        for local_i, ranked_idx in enumerate(order):
            query_idx = start + local_i
            query_set = query_sets[query_idx]
            if not query_set:
                acc["skipped"] = int(acc["skipped"]) + 1
                continue
            relevance_by_idx = _candidate_relevance_scores(
                query_set,
                gallery_sets,
                gallery_inverted,
                idf,
                exclude_index=self_exclude.get(query_idx),
            )
            if not relevance_by_idx:
                acc["skipped"] = int(acc["skipped"]) + 1
                continue
            ranked_relevance = np.asarray(
                [float(relevance_by_idx.get(int(idx), 0.0)) for idx in ranked_idx],
                dtype=np.float32,
            )
            ideal_relevance = np.asarray(list(relevance_by_idx.values()), dtype=np.float32)
            _update_graded_search_acc(acc, ranked_relevance, ideal_relevance)
    return _finalize_graded_search_acc(acc)


def compute_search_metrics_by_graded_blood_ids(
    features: np.ndarray,
    img_ids: list[str] | np.ndarray,
    blood_id_sets: dict[str, frozenset[str]],
    idf: dict[str, float] | None = None,
    chunk_size: int = 256,
) -> dict[str, float]:
    return compute_cross_search_metrics_by_graded_blood_ids(
        features,
        img_ids,
        features,
        img_ids,
        blood_id_sets,
        idf=idf,
        chunk_size=chunk_size,
        exclude_self=True,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or len(right) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def compute_distance_relevance_correlation(
    query_features: np.ndarray,
    query_img_ids: list[str] | np.ndarray,
    gallery_features: np.ndarray,
    gallery_img_ids: list[str] | np.ndarray,
    blood_id_sets: dict[str, frozenset[str]],
    idf: dict[str, float] | None = None,
    max_queries: int = 1000,
    positives_per_query: int = 5,
    negatives_per_query: int = 5,
    seed: int = 42,
    exclude_self: bool = False,
) -> dict[str, float]:
    """Sample query/gallery pairs and correlate embedding distance with relevance."""
    if len(query_features) == 0 or len(gallery_features) == 0:
        return {"pairs": 0.0, "pearson_distance_relevance": 0.0, "spearman_distance_relevance": 0.0}
    rng = np.random.default_rng(seed)
    query_sets = _ordered_blood_id_sets(query_img_ids, blood_id_sets)
    gallery_sets = _ordered_blood_id_sets(gallery_img_ids, blood_id_sets)
    if idf is None:
        universe_ids = [str(x) for x in query_img_ids] + [str(x) for x in gallery_img_ids]
        idf = build_blood_id_idf(blood_id_sets, universe_ids)
    gallery_inverted = _build_inverted_index(gallery_sets)
    id_to_gidx = {str(img_id): idx for idx, img_id in enumerate(gallery_img_ids)}

    valid_queries = [idx for idx, ids in enumerate(query_sets) if ids]
    if max_queries > 0 and len(valid_queries) > max_queries:
        valid_queries = sorted(rng.choice(valid_queries, size=int(max_queries), replace=False).tolist())

    distances: list[float] = []
    relevances: list[float] = []
    all_gallery = np.arange(len(gallery_features), dtype=np.int64)
    for query_idx in valid_queries:
        exclude_idx = id_to_gidx.get(str(query_img_ids[query_idx])) if exclude_self else None
        relevance_by_idx = _candidate_relevance_scores(
            query_sets[query_idx], gallery_sets, gallery_inverted, idf, exclude_index=exclude_idx
        )
        positive_indices = list(relevance_by_idx)
        if positive_indices:
            take = min(int(positives_per_query), len(positive_indices))
            for gallery_idx in rng.choice(positive_indices, size=take, replace=False):
                gi = int(gallery_idx)
                distances.append(float(np.linalg.norm(query_features[query_idx] - gallery_features[gi])))
                relevances.append(float(relevance_by_idx[gi]))

        negative_take = int(negatives_per_query)
        attempts = 0
        chosen_neg: set[int] = set()
        while len(chosen_neg) < negative_take and attempts < negative_take * 20 + 20:
            attempts += 1
            gi = int(rng.choice(all_gallery))
            if gi in chosen_neg or gi in relevance_by_idx or (exclude_idx is not None and gi == exclude_idx):
                continue
            if not gallery_sets[gi]:
                continue
            chosen_neg.add(gi)
        for gi in chosen_neg:
            distances.append(float(np.linalg.norm(query_features[query_idx] - gallery_features[gi])))
            relevances.append(0.0)

    d = np.asarray(distances, dtype=np.float32)
    r = np.asarray(relevances, dtype=np.float32)
    return {
        "pairs": float(len(d)),
        "pearson_distance_relevance": _corrcoef(d, r),
        "spearman_distance_relevance": _corrcoef(_rankdata(d), _rankdata(r)) if len(d) else 0.0,
        "pearson_similarity_relevance": _corrcoef(-d, r),
        "spearman_similarity_relevance": _corrcoef(_rankdata(-d), _rankdata(r)) if len(d) else 0.0,
        "mean_distance": float(np.mean(d)) if len(d) else 0.0,
        "mean_relevance": float(np.mean(r)) if len(r) else 0.0,
    }


def compute_cross_compare_metrics_by_blood_ids(
    query_features: np.ndarray,
    query_img_ids: list[str] | np.ndarray,
    gallery_features: np.ndarray,
    gallery_img_ids: list[str] | np.ndarray,
    blood_id_sets: dict[str, frozenset[str]],
    max_pairs: int = 200000,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    query_sets = _ordered_blood_id_sets(query_img_ids, blood_id_sets)
    gallery_sets = _ordered_blood_id_sets(gallery_img_ids, blood_id_sets)

    valid_queries = [idx for idx, s in enumerate(query_sets) if s]
    valid_gallery = [idx for idx, s in enumerate(gallery_sets) if s]
    if not valid_queries or not valid_gallery:
        return dict(EMPTY_METRICS)

    target_pos = int(max_pairs) // 2
    positive_pairs: list[tuple[int, int]] = []
    query_order = np.asarray(valid_queries, dtype=np.int64)
    rng.shuffle(query_order)
    for qi in query_order:
        for gi in valid_gallery:
            if query_sets[int(qi)] & gallery_sets[int(gi)]:
                positive_pairs.append((int(qi), gi))
                if len(positive_pairs) >= target_pos:
                    break
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
        qi = int(rng.choice(valid_queries))
        gi = int(rng.choice(valid_gallery))
        pair = (qi, gi)
        if pair in negative_pairs_set:
            continue
        if not (query_sets[qi] & gallery_sets[gi]):
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
