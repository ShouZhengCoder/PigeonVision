"""Pedigree-record-error sensitivity: drop X% of each pigeon's ancestor links
(simulating missing/wrong parent records), recompute k, measure Spearman(d, k_perturbed).
Uses the PROOF encoder's iris distances over the 7,144 kin pairs (+5,000 unrelated).
If the correlation survives realistic error rates (10-20%), the result is robust.
"""
import argparse, itertools, os, random
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
    ap.add_argument('--out-json', default='outputs/reports/pedigree_perturbation.json')
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    random.seed(0)

    feat = np.load(args.features).astype(np.float32)
    meta = pd.read_csv(args.meta, dtype=str); meta['img_id'] = meta['img_id'].astype(str)
    feat /= np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-12)
    idx = dict(zip(meta['img_id'], range(len(meta))))
    cv = pd.read_csv(args.vectors, dtype=str); cv['img_id'] = cv['img_id'].astype(str)
    vec_full = defaultdict(dict)
    for i, a, c in zip(cv['img_id'], cv['ancestor'], cv['contribution'].astype(float)):
        vec_full[i][a] = c
    struct = [i for i in vec_full if vec_full[i] and i in idx]

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
        if set(vec_full[a]) & set(vec_full[b]): cous.append((a, b))
        else: unrel.append((a, b))
    unrel = random.sample(unrel, min(5000, len(unrel)))
    pairs = full + half + cous + unrel
    def d_iris(a, b): return float(np.linalg.norm(feat[idx[a]] - feat[idx[b]]))
    D = np.array([d_iris(a, b) for a, b in pairs])
    def kij(vec, a, b):
        va, vb = vec.get(a, {}), vec.get(b, {})
        if not va or not vb: return 0.0
        s, l = (va, vb) if len(va) <= len(vb) else (vb, va)
        return sum(c * l.get(x, 0.0) for x, c in s.items())
    K0 = np.array([kij(vec_full, a, b) for a, b in pairs])
    sp0, _ = spearmanr(D, K0)
    print(f'baseline Spearman(d, k) = {sp0:.4f}  (n={len(pairs)})')

    out = {'baseline_spearman': float(sp0), 'n_pairs': len(pairs), 'levels': {}}
    for p in [0.05, 0.10, 0.20, 0.30, 0.40]:
        sps = []
        for s in range(args.seeds):
            rng = random.Random(1000 * s + 7)
            vec_p = {}
            for i in struct:
                anc = list(vec_full[i].items())
                keep = [x for x in anc if rng.random() >= p]  # drop p% of ancestor links
                vec_p[i] = dict(keep) if keep else {}
            Kp = np.array([kij(vec_p, a, b) for a, b in pairs])
            sp, _ = spearmanr(D, Kp)
            sps.append(sp)
        out['levels'][f'{int(p*100)}pct'] = {'mean': float(np.mean(sps)), 'std': float(np.std(sps))}
        print(f'  drop {int(p*100):2d}% ancestor links: Spearman = {np.mean(sps):.4f} +/- {np.std(sps):.4f}')
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    import json
    json.dump(out, open(args.out_json, 'w'), indent=2)
    print('saved', args.out_json)

if __name__ == '__main__':
    main()
