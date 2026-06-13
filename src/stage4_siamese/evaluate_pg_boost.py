"""Evaluate pseudo-blood boosted PG_ID retrieval.

This is a presentation-oriented reranker: it infers a pseudo bloodline profile
from the nearest visual image anchors, then boosts PG_ID centroid candidates that
share those anchor bloodlines.
"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PG pseudo-blood boost reranking.")
    parser.add_argument("--query-db", type=Path, default=ROOT / "outputs" / "features" / "fusion_1024d_full")
    parser.add_argument("--gallery-db", type=Path, default=ROOT / "outputs" / "features" / "fusion_1024d_full_pg")
    parser.add_argument("--relations", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "outputs" / "reports")
    parser.add_argument("--sample-query", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-k", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--pool-size", type=int, default=1000)
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


def dcg(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(values) + 2, dtype=np.float32))
    return float(np.sum(values.astype(np.float32) * discounts))


def split_pipe_ids(value: str) -> set[str]:
    return {part.strip() for part in str(value).split("|") if part.strip()}


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    query_features, query_meta = read_feature_db(resolve_root_path(args.query_db))
    gallery_features, gallery_meta = read_feature_db(resolve_root_path(args.gallery_db))
    query_features = normalize_features(query_features)
    gallery_features = normalize_features(gallery_features)

    query_img_ids = query_meta["img_id"].astype(str).tolist()
    gallery_img_ids = gallery_meta["img_id"].astype(str).tolist()
    blood_id_sets = load_blood_id_sets(resolve_root_path(args.relations))
    query_sets_all = ordered_sets(query_img_ids, blood_id_sets, query_meta)
    gallery_sets = ordered_sets(gallery_img_ids, blood_id_sets, gallery_meta)
    idf = build_idf(query_sets_all + gallery_sets)

    query_pg_ids = parse_pg_ids(query_meta)
    gallery_pg_ids = parse_pg_ids(gallery_meta)
    gallery_pg_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, pg_id in enumerate(gallery_pg_ids):
        if pg_id:
            gallery_pg_to_indices[pg_id].append(idx)

    rng = np.random.default_rng(int(args.seed))
    if int(args.sample_query) > 0 and int(args.sample_query) < len(query_meta):
        query_indices = sorted(rng.choice(len(query_meta), size=int(args.sample_query), replace=False).tolist())
    else:
        query_indices = list(range(len(query_meta)))

    query_norms_all = np.sum(query_features * query_features, axis=1)
    gallery_norms = np.sum(gallery_features * gallery_features, axis=1)
    top_k = 10
    ks = (1, 5, 10)
    acc: dict[str, float] = defaultdict(float)
    valid = 0
    skipped = 0
    strength = {"strong": 0, "weak": 0, "irrelevant": 0}

    for start in tqdm(range(0, len(query_indices), int(args.chunk_size)), desc="pg boost eval"):
        batch_indices = query_indices[start : start + int(args.chunk_size)]
        q = query_features[batch_indices]
        image_dist2 = np.sum(q * q, axis=1, keepdims=True) + query_norms_all[None, :] - 2.0 * (q @ query_features.T)
        pg_dist2 = np.sum(q * q, axis=1, keepdims=True) + gallery_norms[None, :] - 2.0 * (q @ gallery_features.T)
        for local_i, query_idx in enumerate(batch_indices):
            image_dist2[local_i, int(query_idx)] = np.inf
            pg_id = query_pg_ids[int(query_idx)]
            if pg_id:
                pg_dist2[local_i, gallery_pg_to_indices.get(pg_id, [])] = np.inf

        anchor_indices = np.argsort(image_dist2, axis=1)[:, : int(args.anchor_k)]
        pg_base_order = np.argsort(pg_dist2, axis=1)

        for local_i, query_idx in enumerate(batch_indices):
            query_set = query_sets_all[int(query_idx)]
            if not query_set:
                skipped += 1
                continue
            relevant = {
                gi: blood_id_relevance(query_set, gallery_sets[gi], idf)
                for gi in range(len(gallery_sets))
                if query_set & gallery_sets[gi]
            }
            pg_id = query_pg_ids[int(query_idx)]
            if pg_id:
                for gi in gallery_pg_to_indices.get(pg_id, []):
                    relevant.pop(gi, None)
            relevant = {gi: score for gi, score in relevant.items() if score > 0.0}
            if not relevant:
                skipped += 1
                continue

            pseudo: dict[str, float] = defaultdict(float)
            for rank, anchor_idx in enumerate(anchor_indices[local_i], start=1):
                weight = float(1.0 / np.log2(float(rank) + 1.0))
                for blood_id in query_sets_all[int(anchor_idx)]:
                    pseudo[blood_id] += weight
            if pseudo:
                max_weight = max(pseudo.values())
                pseudo = {blood_id: weight / max_weight for blood_id, weight in pseudo.items() if max_weight > 1e-12}

            pool = pg_base_order[local_i, : min(int(args.pool_size), len(gallery_features))]
            scores = pg_dist2[local_i, pool].astype(np.float32).copy()
            if pseudo:
                for pos, gallery_idx in enumerate(pool):
                    shared = gallery_sets[int(gallery_idx)] & pseudo.keys()
                    if shared:
                        scores[pos] -= float(args.beta) * (sum(pseudo[blood_id] for blood_id in shared) / max(len(gallery_sets[int(gallery_idx)]), 1))
            reranked_pool = pool[np.argsort(scores)]
            order = np.concatenate([reranked_pool, pg_base_order[local_i, len(pool) :]])
            ranked_relevance = np.asarray([relevant.get(int(gi), 0.0) for gi in order], dtype=np.float32)

            valid += 1
            hits_all = (ranked_relevance > 0.0).astype(np.float32)
            if np.any(hits_all):
                precision = np.cumsum(hits_all) / np.arange(1, len(hits_all) + 1, dtype=np.float32)
                acc["mAP"] += float(np.sum(precision * hits_all) / max(len(relevant), 1))
            for k in ks:
                top = ranked_relevance[:k]
                hits = top > 0.0
                hit_count = int(np.sum(hits))
                acc[f"hit_at_{k}"] += float(hit_count > 0)
                acc[f"precision_at_{k}"] += float(hit_count / k)
                acc[f"avg_relevant_at_{k}"] += float(hit_count)
                acc[f"avg_relevance_at_{k}"] += float(np.sum(top) / k)
                ideal = np.sort(np.asarray(list(relevant.values()), dtype=np.float32))[::-1][:k]
                idcg = dcg(ideal)
                acc[f"graded_ndcg_at_{k}"] += float(dcg(top) / idcg) if idcg > 0 else 0.0
            top = ranked_relevance[:top_k]
            strength["strong"] += int(np.sum(top >= 0.35))
            strength["weak"] += int(np.sum((top > 0.0) & (top < 0.35)))
            strength["irrelevant"] += int(top_k - np.sum(top > 0.0))

    if valid <= 0:
        return {"valid_queries": 0, "skipped_queries": skipped}
    metrics: dict[str, Any] = {
        "valid_queries": int(valid),
        "skipped_queries": int(skipped),
        "anchor_k": int(args.anchor_k),
        "beta": float(args.beta),
        "pool_size": int(args.pool_size),
        "topk_strength_per_query": {key: float(value / valid) for key, value in strength.items()},
    }
    for key, value in sorted(acc.items()):
        metrics[key] = float(value / valid)
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    metrics = evaluate(args)
    report = {
        "query_db": str(resolve_root_path(args.query_db)),
        "gallery_db": str(resolve_root_path(args.gallery_db)),
        "sample_query": int(args.sample_query),
        "graded_relevance_search": metrics,
    }
    report_path = ensure_dir(resolve_root_path(args.reports_dir)) / "graded_metrics_pg_boost.json"
    write_json(report_path, report)
    print("\nPG boost summary:")
    for key in ("hit_at_10", "precision_at_10", "avg_relevance_at_10", "graded_ndcg_at_10", "mAP"):
        print(f"  {key}: {float(metrics.get(key, 0.0)):.6f}")
    print(f"Reports written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
