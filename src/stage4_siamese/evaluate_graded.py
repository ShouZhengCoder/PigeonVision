"""Evaluate feature DB retrieval with IDF-weighted graded bloodline relevance."""
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
from relation_metrics import (
    build_blood_id_idf,
    blood_id_relevance,
    compute_cross_search_metrics_by_blood_ids,
    compute_cross_search_metrics_by_graded_blood_ids,
    compute_distance_relevance_correlation,
    load_blood_id_sets,
    shared_idf_quantile,
)


DEFAULT_REPORT_DIR = ROOT / "outputs" / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a feature DB using graded multi-label bloodline relevance.")
    parser.add_argument("--db", type=Path, default=ROOT / "outputs" / "features" / "fusion_1024d_eval")
    parser.add_argument("--relations", type=Path, default=ROOT / "data" / "extracted" / "datasetXGN" / "relations.csv")
    parser.add_argument("--normalize-meta", type=Path, default=ROOT / "outputs" / "iris_normalized" / "normalize_meta.csv")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--sample-query", type=int, default=0, help="Evaluate a deterministic query sample against the full DB.")
    parser.add_argument("--top-k", type=int, default=10, help="Rows per query for error_analysis_topk.csv.")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-error-rows", type=int, default=50000)
    parser.add_argument("--correlation-max-queries", type=int, default=1000)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_feature_db(db_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    feature_path = db_dir / "feature_db.npy"
    meta_path = db_dir / "feature_db_meta.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"feature_db.npy not found: {feature_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"feature_db_meta.csv not found: {meta_path}")
    features = np.load(feature_path).astype("float32")
    meta = pd.read_csv(meta_path, dtype=str).fillna("")
    if "img_id" not in meta.columns:
        raise ValueError(f"{meta_path} missing img_id column")
    if len(features) != len(meta):
        raise ValueError(f"feature/meta row mismatch: features={len(features)}, meta={len(meta)}")
    return features, meta.reset_index(drop=True)


def normalize_features(features: np.ndarray) -> np.ndarray:
    return (features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)).astype("float32")


def parse_pipe_ids(value: str) -> frozenset[str]:
    return frozenset(part.strip() for part in str(value).split("|") if part.strip())


def ordered_sets(img_ids: list[str], blood_id_sets: dict[str, frozenset[str]], meta: pd.DataFrame) -> list[frozenset[str]]:
    meta_sets: dict[str, frozenset[str]] = {}
    if "blood_id" in meta.columns:
        meta_sets = {
            str(row.img_id): parse_pipe_ids(str(row.blood_id))
            for row in meta[["img_id", "blood_id"]].itertuples(index=False)
            if str(row.blood_id).strip()
        }
    return [
        frozenset(set(blood_id_sets.get(str(img_id), frozenset())) | set(meta_sets.get(str(img_id), frozenset())))
        for img_id in img_ids
    ]


def build_inverted(sets: list[frozenset[str]]) -> dict[str, list[int]]:
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, blood_ids in enumerate(sets):
        for blood_id in blood_ids:
            inverted[str(blood_id)].append(idx)
    return dict(inverted)


def candidate_relevance(
    query_set: frozenset[str],
    gallery_sets: list[frozenset[str]],
    inverted: dict[str, list[int]],
    idf: dict[str, float],
    exclude_idx: int | None = None,
) -> dict[int, float]:
    candidates: set[int] = set()
    for blood_id in query_set:
        candidates.update(inverted.get(str(blood_id), []))
    if exclude_idx is not None:
        candidates.discard(int(exclude_idx))
    scores: dict[int, float] = {}
    for idx in candidates:
        score = blood_id_relevance(query_set, gallery_sets[int(idx)], idf)
        if score > 0.0:
            scores[int(idx)] = float(score)
    return scores


def load_mask_confidence(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"img_id": str})
    if "img_id" not in df.columns or "mask_confidence" not in df.columns:
        return {}
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "success"]
    return {
        str(row.img_id): float(row.mask_confidence)
        for row in df[["img_id", "mask_confidence"]].dropna().itertuples(index=False)
    }


def num_blood_bucket(count: int) -> str:
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return ">10"


def relevant_total_bucket(count: int) -> str:
    if count <= 5:
        return "1-5"
    if count <= 20:
        return "6-20"
    if count <= 100:
        return "21-100"
    return ">100"


def mask_bucket(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "missing"
    if value < 0.80:
        return "<0.80"
    if value < 0.90:
        return "0.80-0.90"
    if value < 0.95:
        return "0.90-0.95"
    return ">=0.95"


def summarize_bucket(bucket: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, values in sorted(bucket.items()):
        count = float(values.get("count", 0.0))
        if count <= 0:
            continue
        out[name] = {
            "queries": int(count),
            "precision_at_10": float(values.get("precision_at_10", 0.0) / count),
            "avg_relevance_at_10": float(values.get("avg_relevance_at_10", 0.0) / count),
            "graded_ndcg_at_10": float(values.get("graded_ndcg_at_10", 0.0) / count),
        }
    return out


def bloodline_label_stats(
    db_img_ids: list[str],
    db_sets: list[frozenset[str]],
    meta: pd.DataFrame,
    idf: dict[str, float],
) -> dict[str, Any]:
    freq: dict[str, int] = defaultdict(int)
    for ids in db_sets:
        for blood_id in ids:
            freq[str(blood_id)] += 1
    counts = np.asarray([len(ids) for ids in db_sets], dtype=np.float32)
    freqs = np.asarray(list(freq.values()), dtype=np.float32)
    return {
        "images": int(len(db_img_ids)),
        "images_with_blood_ids": int(sum(bool(ids) for ids in db_sets)),
        "unique_blood_ids": int(len(freq)),
        "blood_id_frequency": {
            "singleton": int(np.sum(freqs == 1)) if len(freqs) else 0,
            "lte_2": int(np.sum(freqs <= 2)) if len(freqs) else 0,
            "gte_5": int(np.sum(freqs >= 5)) if len(freqs) else 0,
            "gte_20": int(np.sum(freqs >= 20)) if len(freqs) else 0,
            "max": int(np.max(freqs)) if len(freqs) else 0,
        },
        "blood_ids_per_image": {
            "mean": float(np.mean(counts)) if len(counts) else 0.0,
            "median": float(np.median(counts)) if len(counts) else 0.0,
            "p90": float(np.quantile(counts, 0.90)) if len(counts) else 0.0,
            "max": int(np.max(counts)) if len(counts) else 0,
        },
        "metadata_missing": {
            "pg_id": int((meta.get("pg_id", pd.Series([""] * len(meta))).fillna("").astype(str).str.strip() == "").sum()),
            "blood_name": int((meta.get("blood_name", meta.get("blood", pd.Series([""] * len(meta)))).fillna("").astype(str).str.strip() == "").sum()),
        },
        "idf": {
            "min": float(min(idf.values())) if idf else 0.0,
            "median": float(np.median(list(idf.values()))) if idf else 0.0,
            "max": float(max(idf.values())) if idf else 0.0,
        },
    }


def run_topk_diagnostics(
    query_features: np.ndarray,
    query_meta: pd.DataFrame,
    gallery_features: np.ndarray,
    gallery_meta: pd.DataFrame,
    query_sets: list[frozenset[str]],
    gallery_sets: list[frozenset[str]],
    idf: dict[str, float],
    mask_confidence: dict[str, float],
    top_k: int,
    chunk_size: int,
    max_error_rows: int,
    reports_dir: Path,
    db_name: str,
    exclude_self: bool,
) -> dict[str, Any]:
    gallery_img_ids = gallery_meta["img_id"].astype(str).tolist()
    query_img_ids = query_meta["img_id"].astype(str).tolist()
    gallery_pg_ids = gallery_meta.get("pg_id", pd.Series([""] * len(gallery_meta))).fillna("").astype(str).tolist()
    query_pg_ids = query_meta.get("pg_id", pd.Series([""] * len(query_meta))).fillna("").astype(str).tolist()
    gallery_blood_names = gallery_meta.get("blood_name", gallery_meta.get("blood", pd.Series([""] * len(gallery_meta)))).fillna("").astype(str).tolist()
    query_blood_names = query_meta.get("blood_name", query_meta.get("blood", pd.Series([""] * len(query_meta)))).fillna("").astype(str).tolist()
    id_to_gidx = {img_id: idx for idx, img_id in enumerate(gallery_img_ids)}
    inverted = build_inverted(gallery_sets)
    gallery_norms = np.sum(gallery_features * gallery_features, axis=1)

    idf_values = np.asarray(list(idf.values()), dtype=np.float32)
    idf_quantiles = tuple(float(x) for x in np.quantile(idf_values, [0.25, 0.50, 0.75])) if len(idf_values) else None

    relevant_totals: list[int] = []
    topk_counts = {"strong": 0, "weak": 0, "irrelevant": 0}
    shared_idf_counts: dict[str, int] = defaultdict(int)
    num_blood_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    relevant_total_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    mask_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    metadata_buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    missing_topk = {"pg_id": 0, "blood_name": 0}
    error_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(query_features), int(chunk_size)), desc="graded diagnostics"):
        end = min(start + int(chunk_size), len(query_features))
        q = query_features[start:end]
        dist2 = np.sum(q * q, axis=1, keepdims=True) + gallery_norms[None, :] - 2.0 * (q @ gallery_features.T)
        if exclude_self:
            for query_idx in range(start, end):
                gallery_idx = id_to_gidx.get(query_img_ids[query_idx])
                if gallery_idx is not None:
                    dist2[query_idx - start, gallery_idx] = np.inf
        top_idx = np.argsort(dist2, axis=1)[:, : int(top_k)]
        for local_i, ranked_idx in enumerate(top_idx):
            query_idx = start + local_i
            query_set = query_sets[query_idx]
            if not query_set:
                continue
            exclude_idx = id_to_gidx.get(query_img_ids[query_idx]) if exclude_self else None
            relevance_by_idx = candidate_relevance(query_set, gallery_sets, inverted, idf, exclude_idx=exclude_idx)
            if not relevance_by_idx:
                continue
            ideal = np.sort(np.asarray(list(relevance_by_idx.values()), dtype=np.float32))[::-1]
            relevant_total = len(relevance_by_idx)
            relevant_totals.append(relevant_total)
            relevance_values = np.asarray([relevance_by_idx.get(int(idx), 0.0) for idx in ranked_idx], dtype=np.float32)
            binary_hits = relevance_values > 0.0
            strong = int(np.sum(relevance_values >= 0.35))
            weak = int(np.sum((relevance_values > 0.0) & (relevance_values < 0.35)))
            irrelevant = int(len(relevance_values) - strong - weak)
            topk_counts["strong"] += strong
            topk_counts["weak"] += weak
            topk_counts["irrelevant"] += irrelevant

            discounts = 1.0 / np.log2(np.arange(2, len(relevance_values) + 2, dtype=np.float32))
            idcg = float(np.sum(ideal[: len(relevance_values)] * discounts[: min(len(ideal), len(relevance_values))]))
            dcg = float(np.sum(relevance_values * discounts))
            summary = {
                "count": 1.0,
                "precision_at_10": float(np.sum(binary_hits) / max(int(top_k), 1)),
                "avg_relevance_at_10": float(np.sum(relevance_values) / max(int(top_k), 1)),
                "graded_ndcg_at_10": float(dcg / idcg) if idcg > 0 else 0.0,
            }
            for bucket in (
                num_blood_buckets[num_blood_bucket(len(query_set))],
                relevant_total_buckets[relevant_total_bucket(relevant_total)],
                mask_buckets[mask_bucket(mask_confidence.get(query_img_ids[query_idx]))],
                metadata_buckets[
                    "missing_pg_id"
                    if not query_pg_ids[query_idx].strip()
                    else "missing_blood_name"
                    if not query_blood_names[query_idx].strip()
                    else "complete"
                ],
            ):
                for key, value in summary.items():
                    bucket[key] += value

            for rank, gallery_idx in enumerate(ranked_idx, start=1):
                gi = int(gallery_idx)
                relevance = float(relevance_by_idx.get(gi, 0.0))
                if relevance >= 0.35:
                    strength = "strong"
                elif relevance > 0.0:
                    strength = "weak"
                else:
                    strength = "irrelevant"
                if relevance > 0.0:
                    shared_idf_counts[shared_idf_quantile(query_set, gallery_sets[gi], idf, idf_quantiles)] += 1
                if not gallery_pg_ids[gi].strip():
                    missing_topk["pg_id"] += 1
                if not gallery_blood_names[gi].strip():
                    missing_topk["blood_name"] += 1
                if len(error_rows) < int(max_error_rows):
                    shared = sorted(query_set & gallery_sets[gi])
                    error_rows.append(
                        {
                            "db": db_name,
                            "query_img_id": query_img_ids[query_idx],
                            "query_pg_id": query_pg_ids[query_idx],
                            "query_blood_name": query_blood_names[query_idx],
                            "query_num_blood_ids": len(query_set),
                            "query_mask_confidence": mask_confidence.get(query_img_ids[query_idx], ""),
                            "rank": rank,
                            "gallery_img_id": gallery_img_ids[gi],
                            "gallery_pg_id": gallery_pg_ids[gi],
                            "gallery_blood_name": gallery_blood_names[gi],
                            "distance": float(np.sqrt(max(float(dist2[local_i, gi]), 0.0))),
                            "relevance": relevance,
                            "strength": strength,
                            "shared_blood_ids": "|".join(shared[:20]),
                            "num_shared_blood_ids": len(shared),
                        }
                    )

    if error_rows:
        pd.DataFrame(error_rows).to_csv(reports_dir / "error_analysis_topk.csv", index=False)

    totals = np.asarray(relevant_totals, dtype=np.float32)
    denom = max(len(relevant_totals), 1)
    return {
        "db": db_name,
        "queries_with_gallery_relevance": int(len(relevant_totals)),
        "relevant_total_distribution": {
            "mean": float(np.mean(totals)) if len(totals) else 0.0,
            "median": float(np.median(totals)) if len(totals) else 0.0,
            "p90": float(np.quantile(totals, 0.90)) if len(totals) else 0.0,
            "max": int(np.max(totals)) if len(totals) else 0,
        },
        "topk_strength_per_query": {key: float(value / denom) for key, value in topk_counts.items()},
        "topk_strength_total": topk_counts,
        "shared_blood_idf_quantile_topk": dict(sorted(shared_idf_counts.items())),
        "query_buckets_num_blood_ids": summarize_bucket(num_blood_buckets),
        "query_buckets_relevant_total": summarize_bucket(relevant_total_buckets),
        "query_buckets_mask_confidence": summarize_bucket(mask_buckets),
        "query_buckets_metadata": summarize_bucket(metadata_buckets),
        "topk_missing_metadata": missing_topk,
        "error_rows_written": int(len(error_rows)),
    }


def main() -> int:
    args = parse_args()
    db_dir = resolve_root_path(args.db)
    reports_dir = ensure_dir(resolve_root_path(args.reports_dir))
    features, meta = read_feature_db(db_dir)
    features = normalize_features(features)
    img_ids = meta["img_id"].astype(str).tolist()
    blood_id_sets = load_blood_id_sets(resolve_root_path(args.relations))
    db_sets = ordered_sets(img_ids, blood_id_sets, meta)
    idf = build_blood_id_idf({img_id: ids for img_id, ids in zip(img_ids, db_sets)}, img_ids)
    mask_conf = load_mask_confidence(resolve_root_path(args.normalize_meta))

    rng = np.random.default_rng(int(args.seed))
    if int(args.sample_query) > 0 and int(args.sample_query) < len(meta):
        query_indices = sorted(rng.choice(len(meta), size=int(args.sample_query), replace=False).tolist())
    else:
        query_indices = list(range(len(meta)))
    query_features = features[query_indices]
    query_meta = meta.iloc[query_indices].reset_index(drop=True)
    query_img_ids = query_meta["img_id"].astype(str).tolist()
    query_sets = [db_sets[idx] for idx in query_indices]

    print(f"DB: {db_dir}")
    print(f"Gallery: {len(meta)} rows, Query: {len(query_meta)} rows")

    binary_metrics = compute_cross_search_metrics_by_blood_ids(
        query_features,
        query_img_ids,
        features,
        img_ids,
        blood_id_sets,
        chunk_size=int(args.chunk_size),
        exclude_self=True,
    )
    graded_metrics = compute_cross_search_metrics_by_graded_blood_ids(
        query_features,
        query_img_ids,
        features,
        img_ids,
        {img_id: ids for img_id, ids in zip(img_ids, db_sets)},
        idf=idf,
        chunk_size=int(args.chunk_size),
        exclude_self=True,
    )
    correlation = compute_distance_relevance_correlation(
        query_features,
        query_img_ids,
        features,
        img_ids,
        {img_id: ids for img_id, ids in zip(img_ids, db_sets)},
        idf=idf,
        max_queries=int(args.correlation_max_queries),
        seed=int(args.seed),
        exclude_self=True,
    )

    db_name = db_dir.name
    stats = bloodline_label_stats(img_ids, db_sets, meta, idf)
    diagnostics = run_topk_diagnostics(
        query_features,
        query_meta,
        features,
        meta,
        query_sets,
        db_sets,
        idf,
        mask_conf,
        top_k=int(args.top_k),
        chunk_size=int(args.chunk_size),
        max_error_rows=int(args.max_error_rows),
        reports_dir=reports_dir,
        db_name=db_name,
        exclude_self=True,
    )

    payload = {
        "db": str(db_dir),
        "gallery_size": int(len(meta)),
        "query_size": int(len(query_meta)),
        "sample_query": int(args.sample_query),
        "binary_overlap_search": {key: float(value) for key, value in binary_metrics.items()},
        "graded_relevance_search": {key: float(value) for key, value in graded_metrics.items()},
    }
    write_json(reports_dir / f"graded_metrics_{db_name}.json", payload)
    write_json(reports_dir / "bloodline_label_stats.json", stats)
    write_json(reports_dir / "relevance_distribution_eval.json", diagnostics)
    write_json(reports_dir / "distance_relevance_correlation.json", correlation)

    print("\nGraded search summary:")
    for key in ("hit_at_1", "hit_at_5", "hit_at_10", "precision_at_10", "avg_relevance_at_10", "graded_ndcg_at_10", "mAP"):
        print(f"  {key}: {graded_metrics.get(key, 0.0):.6f}")
    print(f"Reports written to {reports_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
