"""Evaluate image queries against the PG_ID centroid gallery with graded relevance."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from _common import ROOT, ensure_dir, resolve_root_path
from evaluate_graded import ordered_sets, read_feature_db, normalize_features
from relation_metrics import blood_id_relevance, load_blood_id_sets


DEFAULT_QUERY_DB = ROOT / "outputs" / "features" / "fusion_1024d_full"
DEFAULT_GALLERY_DB = ROOT / "outputs" / "features" / "fusion_1024d_full_pg"
DEFAULT_REPORT_DIR = ROOT / "outputs" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image queries against PG_ID centroid gallery.")
    parser.add_argument("--query-db", type=Path, default=DEFAULT_QUERY_DB)
    parser.add_argument("--gallery-db", type=Path, default=DEFAULT_GALLERY_DB)
    parser.add_argument("--relations", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--sample-query", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-same-pg", action="store_true", help="Do not exclude gallery rows with the same non-empty PG_ID as the query.")
    return parser.parse_args()


def parse_pg_ids(meta: pd.DataFrame) -> list[str]:
    return meta.get("pg_id", pd.Series([""] * len(meta))).fillna("").astype(str).str.strip().tolist()


def build_idf(sets: list[frozenset[str]]) -> dict[str, float]:
    freq: dict[str, int] = defaultdict(int)
    for ids in sets:
        for blood_id in ids:
            freq[str(blood_id)] += 1
    n = len(sets)
    return {blood_id: float(np.log((n + 1.0) / (count + 1.0))) for blood_id, count in freq.items()}


def build_inverted(sets: list[frozenset[str]]) -> dict[str, list[int]]:
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, ids in enumerate(sets):
        for blood_id in ids:
            inverted[str(blood_id)].append(idx)
    return dict(inverted)


def dcg(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(values) + 2, dtype=np.float32))
    return float(np.sum(values.astype(np.float32) * discounts))


def average_precision(ranked_relevance: np.ndarray, relevant_total: int) -> float:
    if relevant_total <= 0:
        return 0.0
    hits = (ranked_relevance > 0.0).astype(np.float32)
    if not np.any(hits):
        return 0.0
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1, dtype=np.float32)
    return float(np.sum(precision * hits) / max(relevant_total, 1))


def relevant_gallery_scores(
    query_set: frozenset[str],
    gallery_sets: list[frozenset[str]],
    gallery_inverted: dict[str, list[int]],
    idf: dict[str, float],
    excluded: set[int],
) -> dict[int, float]:
    candidates: set[int] = set()
    for blood_id in query_set:
        candidates.update(gallery_inverted.get(str(blood_id), []))
    candidates.difference_update(excluded)
    scores: dict[int, float] = {}
    for idx in candidates:
        score = blood_id_relevance(query_set, gallery_sets[int(idx)], idf)
        if score > 0.0:
            scores[int(idx)] = float(score)
    return scores


def evaluate(
    query_features: np.ndarray,
    query_meta: pd.DataFrame,
    query_sets: list[frozenset[str]],
    gallery_features: np.ndarray,
    gallery_meta: pd.DataFrame,
    gallery_sets: list[frozenset[str]],
    idf: dict[str, float],
    top_k: int,
    chunk_size: int,
    exclude_same_pg: bool,
) -> dict[str, Any]:
    query_pg_ids = parse_pg_ids(query_meta)
    gallery_pg_ids = parse_pg_ids(gallery_meta)
    gallery_pg_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, pg_id in enumerate(gallery_pg_ids):
        if pg_id:
            gallery_pg_to_indices[pg_id].append(idx)

    gallery_inverted = build_inverted(gallery_sets)
    gallery_norms = np.sum(gallery_features * gallery_features, axis=1)
    ks = (1, 5, 10)
    acc: dict[str, float] = defaultdict(float)
    relevant_totals: list[int] = []
    topk_strength = {"strong": 0, "weak": 0, "irrelevant": 0}

    for start in tqdm(range(0, len(query_features), int(chunk_size)), desc="pg graded eval"):
        end = min(start + int(chunk_size), len(query_features))
        q = query_features[start:end]
        dist2 = np.sum(q * q, axis=1, keepdims=True) + gallery_norms[None, :] - 2.0 * (q @ gallery_features.T)
        excluded_per_query: dict[int, set[int]] = {}
        if exclude_same_pg:
            for query_idx in range(start, end):
                pg_id = query_pg_ids[query_idx]
                excluded = set(gallery_pg_to_indices.get(pg_id, [])) if pg_id else set()
                if excluded:
                    excluded_per_query[query_idx] = excluded
                    dist2[query_idx - start, list(excluded)] = np.inf
        order = np.argsort(dist2, axis=1)

        for local_i, ranked_idx in enumerate(order):
            query_idx = start + local_i
            query_set = query_sets[query_idx]
            if not query_set:
                acc["skipped_queries"] += 1.0
                continue
            relevance_by_idx = relevant_gallery_scores(
                query_set,
                gallery_sets,
                gallery_inverted,
                idf,
                excluded_per_query.get(query_idx, set()),
            )
            if not relevance_by_idx:
                acc["skipped_queries"] += 1.0
                continue

            ranked_relevance = np.asarray([float(relevance_by_idx.get(int(idx), 0.0)) for idx in ranked_idx], dtype=np.float32)
            ideal_relevance = np.sort(np.asarray(list(relevance_by_idx.values()), dtype=np.float32))[::-1]
            relevant_total = int(len(relevance_by_idx))
            relevant_totals.append(relevant_total)
            acc["valid_queries"] += 1.0
            acc["mAP"] += average_precision(ranked_relevance, relevant_total)

            top = ranked_relevance[:top_k]
            topk_strength["strong"] += int(np.sum(top >= 0.35))
            topk_strength["weak"] += int(np.sum((top > 0.0) & (top < 0.35)))
            topk_strength["irrelevant"] += int(top_k - np.sum(top > 0.0))

            for k in ks:
                top_rel = ranked_relevance[:k]
                hits = top_rel > 0.0
                hit_count = int(np.sum(hits))
                acc[f"hit_at_{k}"] += float(hit_count > 0)
                acc[f"precision_at_{k}"] += float(hit_count / k)
                acc[f"avg_relevant_at_{k}"] += float(hit_count)
                acc[f"recall_at_{k}"] += float(hit_count / max(relevant_total, 1))
                binary_idcg = dcg(np.ones(min(relevant_total, k), dtype=np.float32))
                acc[f"binary_ndcg_at_{k}"] += float(dcg(hits.astype(np.float32)) / binary_idcg) if binary_idcg > 0 else 0.0
                graded_idcg = dcg(ideal_relevance[:k])
                acc[f"graded_ndcg_at_{k}"] += float(dcg(top_rel) / graded_idcg) if graded_idcg > 0 else 0.0
                acc[f"avg_relevance_at_{k}"] += float(np.sum(top_rel) / k)

    valid = float(acc["valid_queries"])
    if valid <= 0:
        return {"valid_queries": 0.0, "skipped_queries": float(acc["skipped_queries"])}

    metrics: dict[str, Any] = {
        "valid_queries": valid,
        "skipped_queries": float(acc["skipped_queries"]),
        "mean_relevant_total": float(np.mean(relevant_totals)) if relevant_totals else 0.0,
        "median_relevant_total": float(np.median(relevant_totals)) if relevant_totals else 0.0,
        "p90_relevant_total": float(np.quantile(np.asarray(relevant_totals), 0.90)) if relevant_totals else 0.0,
        "mAP": float(acc["mAP"] / valid),
        "topk_strength_per_query": {key: float(value / valid) for key, value in topk_strength.items()},
    }
    for key, value in sorted(acc.items()):
        if key in {"valid_queries", "skipped_queries", "mAP"}:
            continue
        metrics[key] = float(value / valid)
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    query_db = resolve_root_path(args.query_db)
    gallery_db = resolve_root_path(args.gallery_db)
    reports_dir = ensure_dir(resolve_root_path(args.reports_dir))

    query_features, query_meta_all = read_feature_db(query_db)
    gallery_features, gallery_meta = read_feature_db(gallery_db)
    query_features = normalize_features(query_features)
    gallery_features = normalize_features(gallery_features)

    blood_id_sets = load_blood_id_sets(resolve_root_path(args.relations))
    query_img_ids_all = query_meta_all["img_id"].astype(str).tolist()
    gallery_img_ids = gallery_meta["img_id"].astype(str).tolist()
    query_sets_all = ordered_sets(query_img_ids_all, blood_id_sets, query_meta_all)
    gallery_sets = ordered_sets(gallery_img_ids, blood_id_sets, gallery_meta)

    rng = np.random.default_rng(int(args.seed))
    if int(args.sample_query) > 0 and int(args.sample_query) < len(query_meta_all):
        query_indices = sorted(rng.choice(len(query_meta_all), size=int(args.sample_query), replace=False).tolist())
    else:
        query_indices = list(range(len(query_meta_all)))
    query_meta = query_meta_all.iloc[query_indices].reset_index(drop=True)
    query_features_sample = query_features[query_indices]
    query_sets = [query_sets_all[idx] for idx in query_indices]

    idf = build_idf(query_sets + gallery_sets)
    exclude_same_pg = not bool(args.include_same_pg)
    metrics = evaluate(
        query_features_sample,
        query_meta,
        query_sets,
        gallery_features,
        gallery_meta,
        gallery_sets,
        idf,
        top_k=int(args.top_k),
        chunk_size=int(args.chunk_size),
        exclude_same_pg=exclude_same_pg,
    )
    payload = {
        "query_db": str(query_db),
        "gallery_db": str(gallery_db),
        "query_size": int(len(query_meta)),
        "gallery_size": int(len(gallery_meta)),
        "sample_query": int(args.sample_query),
        "exclude_same_pg": bool(exclude_same_pg),
        "graded_relevance_search": metrics,
    }
    suffix = "exclude_same_pg" if exclude_same_pg else "include_same_pg"
    report_path = reports_dir / f"graded_metrics_{query_db.name}_to_{gallery_db.name}_{suffix}.json"
    write_json(report_path, payload)

    print(f"Query DB: {query_db}")
    print(f"Gallery DB: {gallery_db}")
    print(f"Query: {len(query_meta)} rows, Gallery: {len(gallery_meta)} rows, exclude_same_pg={exclude_same_pg}")
    print("\nPG graded search summary:")
    for key in ("hit_at_1", "hit_at_5", "hit_at_10", "precision_at_10", "avg_relevance_at_10", "graded_ndcg_at_10", "mAP"):
        print(f"  {key}: {float(metrics.get(key, 0.0)):.6f}")
    print(f"Reports written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
