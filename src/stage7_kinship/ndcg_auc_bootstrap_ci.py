"""Bootstrap 95% CIs for graded nDCG@20 and any-kin AUC, per encoder.
Reproduces phaseB_iris_kinship.py logic (random.seed(0), same struct pool/queries)
so the point estimates match the paper, then bootstraps over queries (nDCG) and
over pairs (AUC). Run once per encoder; prints + saves JSON.
"""
import argparse, itertools, math, os, random
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def grade(k):
    if k <= 0: return 0
    if k <= 0.1: return 1
    if k <= 0.3: return 2
    return 3

def ndcg(rel_sorted, k=20):
    dcg = sum(rel_sorted[i] / math.log2(i + 2) for i in range(min(k, len(rel_sorted))))
    ideal = sorted(rel_sorted, reverse=True)[:k]
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(min(k, len(ideal))))
    return dcg / idcg if idcg > 0 else 0.0

def auc(d_pos, d_neg):
    d_pos = np.asarray(d_pos, dtype=np.float64); d_neg = np.asarray(d_neg, dtype=np.float64)
    if len(d_pos) == 0 or len(d_neg) == 0: return float('nan')
    s = np.sort(d_neg)
    strict = len(d_neg) - np.searchsorted(s, d_pos, side='right')          # neg > pos
    ties = np.searchsorted(s, d_pos, side='right') - np.searchsorted(s, d_pos, side='left')  # neg == pos
    return float((strict + 0.5 * ties).sum() / (len(d_pos) * len(d_neg)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--vectors', default='data/pedigree/contribution_vectors.csv')
    ap.add_argument('--pedigree', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-json', default='outputs/reports/ndcg_auc_ci.json')
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    random.seed(0); np.random.seed(0)

    feat = np.load(args.features).astype(np.float32)
    meta = pd.read_csv(args.meta, dtype=str); meta['img_id'] = meta['img_id'].astype(str)
    feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    idx = dict(zip(meta['img_id'], range(len(meta))))
    cv = pd.read_csv(args.vectors, dtype=str); cv['img_id'] = cv['img_id'].astype(str)
    vec = defaultdict(dict)
    for i, a, c in zip(cv['img_id'], cv['ancestor'], cv['contribution'].astype(float)):
        vec[i][a] = c
    struct = [i for i in vec if vec[i] and i in idx]
    def k_ped(a, b):
        va, vb = vec.get(a, {}), vec.get(b, {})
        if not va or not vb: return 0.0
        s, l = (va, vb) if len(va) <= len(vb) else (vb, va)
        return sum(c * l.get(x, 0.0) for x, c in s.items())
    ped = pd.read_csv(args.pedigree, dtype=str).fillna(''); ped['img_id'] = ped['img_id'].astype(str)
    par = {r['img_id']: (r['father'], r['mother']) for _, r in ped.iterrows()}
    by_f, by_m = defaultdict(list), defaultdict(list)
    for i in struct:
        f, m = par.get(i, ('', ''))
        if f: by_f[f].append(i)
        if m: by_m[m].append(i)
    full, half, cous, unrel = [], [], [], []
    seen = set()
    for grp in list(by_f.values()) + list(by_m.values()):
        for a, b in itertools.combinations(grp, 2):
            if (a, b) in seen: continue
            seen.add((a, b))
            fa, ma = par.get(a, ('', '')); fb, mb = par.get(b, ('', ''))
            sf = fa and fa == fb; sm = ma and ma == mb
            if sf and sm: full.append((a, b))
            elif sf or sm: half.append((a, b))
    sample = random.sample(struct, min(1000, len(struct)))
    for a, b in itertools.combinations(sample, 2):
        if (a, b) in seen: continue
        fa, ma = par.get(a, ('', '')); fb, mb = par.get(b, ('', ''))
        if (fa and fa == fb) or (ma and ma == mb): continue
        if set(vec[a]) & set(vec[b]): cous.append((a, b))
        else: unrel.append((a, b))
    unrel = random.sample(unrel, min(5000, len(unrel)))
    def d_iris(a, b): return float(np.linalg.norm(feat[idx[a]] - feat[idx[b]]))
    D_full = [d_iris(a, b) for a, b in full]
    D_half = [d_iris(a, b) for a, b in half]
    D_cous = [d_iris(a, b) for a, b in cous]
    D_un = [d_iris(a, b) for a, b in unrel]

    # nDCG@20 (graded k), 300 queries, top-50 pool
    queries = random.sample(struct, min(300, len(struct)))
    pool_ids = struct
    pool_feat = feat[[idx[i] for i in pool_ids]]
    ndcg_per_q = []
    for q in queries:
        fq = feat[idx[q]]
        dist = np.linalg.norm(pool_feat - fq, axis=1)
        order = [o for o in np.argsort(dist) if pool_ids[o] != q][:50]
        rels_k = [grade(k_ped(q, pool_ids[o])) for o in order]
        ndcg_per_q.append(ndcg(rels_k))
    ndcg_per_q = np.array(ndcg_per_q)
    ndcg_mean = float(ndcg_per_q.mean())

    # any-kin AUC (kin vs unrelated)
    d_pos = np.array(D_full + D_half + D_cous)
    d_neg = np.array(D_un)
    auc_val = auc(d_pos, d_neg)

    # bootstrap CIs
    rng = np.random.default_rng(0)
    nq = len(ndcg_per_q)
    ndcg_boots = np.array([ndcg_per_q[rng.integers(0, nq, nq)].mean() for _ in range(args.boot)])
    ndcg_ci = (float(np.percentile(ndcg_boots, 2.5)), float(np.percentile(ndcg_boots, 97.5)))
    np_pos, np_neg = len(d_pos), len(d_neg)
    auc_boots = []
    for _ in range(args.boot):
        p = d_pos[rng.integers(0, np_pos, np_pos)]
        n = d_neg[rng.integers(0, np_neg, np_neg)]
        auc_boots.append(auc(p, n))
    auc_ci = (float(np.percentile(auc_boots, 2.5)), float(np.percentile(auc_boots, 97.5)))

    out = {'tag': args.tag, 'nDCG@20': round(ndcg_mean, 4), 'nDCG@20_CI95': [round(ndcg_ci[0],4), round(ndcg_ci[1],4)],
           'anykin_AUC': round(float(auc_val), 4), 'anykin_AUC_CI95': [round(auc_ci[0],4), round(auc_ci[1],4)],
           'n_queries': nq, 'n_pos': np_pos, 'n_neg': np_neg, 'boot_iters': args.boot}
    print(f'[{args.tag}] nDCG@20={ndcg_mean:.4f} CI[{ndcg_ci[0]:.4f},{ndcg_ci[1]:.4f}] | any-kin AUC={auc_val:.4f} CI[{auc_ci[0]:.4f},{auc_ci[1]:.4f}]')
    # append to json list
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    import json
    prev = []
    if os.path.exists(args.out_json):
        try: prev = json.load(open(args.out_json))
        except: prev = []
    prev = [r for r in prev if r.get('tag') != args.tag]
    prev.append(out)
    json.dump(prev, open(args.out_json, 'w'), indent=2)

if __name__ == '__main__':
    main()
