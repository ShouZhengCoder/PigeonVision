"""Bootstrap CI for the iris-kinship Spearman + same-loft (pg_id) confound analysis,
using the PROOF encoder (relation_supcon_split, 300ep) features. Reuses phaseB_iris_kinship.py
pair generation (random.seed(0), non-empty parent check) so pairs match the -0.699 result.

Outputs:
  - Spearman(iris_dist, k_ped) with bootstrap 95% CI
  - Spearman(iris_dist, k_idf) [training heuristic, for the circularity point]
  - tier iris-distance means, overall and restricted to same-pg_id pairs (confound test)
  - same-pg vs diff-pg unrelated-pair distance (loft-confound magnitude)
"""
import argparse, itertools, math, os, random
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='outputs/features/relation_supcon_256d_split_full/feature_db.npy')
    ap.add_argument('--meta', default='outputs/features/relation_supcon_256d_split_full/feature_db_meta.csv')
    ap.add_argument('--vectors', default='data/pedigree/contribution_vectors.csv')
    ap.add_argument('--pedigree', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--relations', default='data/extracted/datasetXGN/relations.csv')
    ap.add_argument('--rel-meta', default='data/relation_meta.csv')
    ap.add_argument('--out-json', default='outputs/reports/proof_spearman_ci_confound.json')
    ap.add_argument('--boot', type=int, default=1000)
    args = ap.parse_args()
    random.seed(0)

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

    rel = pd.read_csv(args.relations, header=None, names=['blood_id', 'img_id'])
    rel['img_id'] = rel['img_id'].astype(str)
    df_c = rel.groupby('blood_id').size(); n_docs = rel['img_id'].nunique()
    idf = {b: math.log(1 + n_docs / c) for b, c in df_c.items()}
    rel_sets = rel.groupby('img_id')['blood_id'].apply(set).to_dict()
    def k_idf(a, b):
        sa, sb = rel_sets.get(a, set()), rel_sets.get(b, set())
        if not sa or not sb: return 0.0
        sh = sa & sb
        if not sh: return 0.0
        shared = sum(idf[x] for x in sh); ia = sum(idf[x] for x in sa); ib = sum(idf[x] for x in sb)
        return 0.7 * shared / min(ia, ib) + 0.3 * shared / (ia + ib - shared)

    rm = pd.read_csv(args.rel_meta, dtype=str); rm['img_id'] = rm['img_id'].astype(str)
    pg = dict(zip(rm['img_id'], rm['pg_id']))

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
    def same_pg(a, b):
        pa, pb = pg.get(a), pg.get(b)
        return bool(pa) and bool(pb) and pa == pb

    rows = []
    for tier, pairs in [('full', full), ('half', half), ('cousin', cous), ('unrelated', unrel)]:
        for a, b in pairs:
            rows.append({'tier': tier, 'd': d_iris(a, b), 'k': k_ped(a, b),
                         'k_idf': k_idf(a, b), 'same_pg': same_pg(a, b)})
    df = pd.DataFrame(rows)
    print(f'pairs: {len(df)} (full={len(full)}, half={len(half)}, cousin={len(cous)}, unrel={len(unrel)})')

    sp_ped, _ = spearmanr(df['d'], df['k'])
    sp_idf, _ = spearmanr(df['d'], df['k_idf'])
    print(f'Spearman(d, k_ped) = {sp_ped:.4f}')
    print(f'Spearman(d, k_idf) = {sp_idf:.4f}  [training heuristic]')

    # bootstrap CI on Spearman(d, k_ped)
    rng = np.random.default_rng(0)
    n = len(df); boots = []
    d_arr = df['d'].to_numpy(); k_arr = df['k'].to_numpy()
    for _ in range(args.boot):
        s = rng.integers(0, n, n)
        r, _ = spearmanr(d_arr[s], k_arr[s])
        boots.append(r)
    boots = np.array(boots)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    print(f'bootstrap 95% CI for Spearman(d,k): [{ci[0]:.4f}, {ci[1]:.4f}]  ({args.boot} resamples)')

    # confound: tier means overall vs same-pg-only
    tier_means_all = df.groupby('tier')['d'].mean().to_dict()
    tier_means_samepg = df[df['same_pg']].groupby('tier')['d'].mean().to_dict()
    tier_counts_samepg = df[df['same_pg']].groupby('tier').size().to_dict()
    mono_samepg = None
    sp_samepg = None
    sdf = df[df['same_pg']]
    if {'full','half','cousin','unrelated'}.issubset(set(sdf['tier'].unique())):
        sp_samepg, _ = spearmanr(sdf['d'], sdf['k'])
        order = ['full','half','cousin','unrelated']
        mono_samepg = all(tier_means_samepg[order[i]] < tier_means_samepg[order[i+1]] for i in range(3))

    # loft confound magnitude on unrelated: same-pg vs diff-pg
    un = df[df['tier']=='unrelated']
    un_same = un[un['same_pg']]['d'].mean()
    un_diff = un[~un['same_pg']]['d'].mean()

    out = {
        'n_pairs': int(n),
        'n_full': len(full), 'n_half': len(half), 'n_cousin': len(cous), 'n_unrel': len(unrel),
        'spearman_d_kped': float(sp_ped),
        'spearman_d_kidf': float(sp_idf),
        'spearman_ci95': ci,
        'bootstrap_iters': args.boot,
        'tier_mean_iris_dist_all': {k: float(v) for k, v in tier_means_all.items()},
        'tier_mean_iris_dist_samepg': {k: float(v) for k, v in tier_means_samepg.items()},
        'tier_n_samepg': {k: int(v) for k, v in tier_counts_samepg.items()},
        'monotonic_within_samepg': mono_samepg,
        'spearman_within_samepg': float(sp_samepg) if sp_samepg is not None else None,
        'unrelated_samepg_mean_dist': float(un_same),
        'unrelated_diffpg_mean_dist': float(un_diff),
        'loft_confound_on_unrelated': float(un_same - un_diff),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    import json
    with open(args.out_json, 'w') as f: json.dump(out, f, indent=2)
    print('\n=== RESULT ===')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
