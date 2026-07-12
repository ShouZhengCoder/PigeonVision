"""Form A (literal ancestor-path code + LCS) vs Form B (founder-contribution vector = 2phi)
consistency check. Reuses validate_kinship.py's exact pair generation (random.seed(0)) so
the pairs match the published Table 1 / Fig 3.

Reports:
  - Form A LCS-sim tier means (monotonicity check)
  - Form B k tier means (sanity, should match validate_kinship)
  - Spearman(Form A sim, Form B k) over sampled pairs (cross-form agreement)
"""
import argparse, itertools, os, random, math
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VALID_RING = __import__('re').compile(r'^[A-Z]{1,4}\d+-[A-Z0-9-]+$')
def ok(r): return bool(r) and bool(VALID_RING.match(str(r))) and len(str(r)) >= 6

def lcs_len(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0: return 0
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0; ai = a[i]
        for j in range(n):
            cur = dp[j + 1]
            dp[j + 1] = prev + 1 if ai == b[j] else max(dp[j], dp[j + 1])
            prev = cur
    return dp[n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vectors', default='data/pedigree/contribution_vectors.csv')
    ap.add_argument('--literal', default='data/pedigree/literal_codes.csv')
    ap.add_argument('--pedigree', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--out-json', default='outputs/reports/formA_consistency.json')
    args = ap.parse_args()
    random.seed(0)

    # Form B vectors
    cv = pd.read_csv(args.vectors, dtype=str)
    cv['img_id'] = cv['img_id'].astype(str)
    vectors = defaultdict(dict)
    for iid, anc, c in zip(cv['img_id'], cv['ancestor'], cv['contribution'].astype(float)):
        vectors[iid][anc] = c
    structured = [i for i in vectors if vectors[i]]

    # Form A literal codes (token lists)
    lc = pd.read_csv(args.literal, dtype=str).fillna('')
    lc['img_id'] = lc['img_id'].astype(str)
    codes = {iid: [t for t in s.split('|') if t] for iid, s in zip(lc['img_id'], lc['literal_code'])}

    ped = pd.read_csv(args.pedigree, dtype=str).fillna('')
    ped['img_id'] = ped['img_id'].astype(str)
    par_of = {r['img_id']: (r['father'], r['mother']) for _, r in ped.iterrows()}

    def kB(a, b):  # Form B dot product
        va, vb = vectors[a], vectors[b]
        return sum(va[x] * vb.get(x, 0.0) for x in va)
    def kA(a, b):  # Form A LCS ratio
        ca, cb = codes.get(a, []), codes.get(b, [])
        if not ca or not cb: return 0.0
        return lcs_len(ca, cb) / max(len(ca), len(cb))

    fullA, halfA, cousA, unrelA = [], [], [], []
    fullB, halfB, cousB, unrelB = [], [], [], []
    by_f, by_m = defaultdict(list), defaultdict(list)
    for i in structured:
        f, m = par_of.get(i, ('', ''))
        if ok(f): by_f[f].append(i)
        if ok(m): by_m[m].append(i)
    seen = set()
    for grp in list(by_f.values()) + list(by_m.values()):
        if len(grp) < 2: continue
        for a, b in itertools.combinations(grp, 2):
            if (a, b) in seen: continue
            seen.add((a, b))
            fa, ma = par_of.get(a, ('', '')); fb, mb = par_of.get(b, ('', ''))
            share_f = ok(fa) and fa == fb; share_m = ok(ma) and ma == mb
            if share_f and share_m:
                fullA.append(kA(a, b)); fullB.append(kB(a, b))
            elif share_f or share_m:
                halfA.append(kA(a, b)); halfB.append(kB(a, b))
    sample = random.sample(structured, min(1000, len(structured)))
    for a, b in itertools.combinations(sample, 2):
        if (a, b) in seen or (b, a) in seen: continue
        fa, ma = par_of.get(a, ('', '')); fb, mb = par_of.get(b, ('', ''))
        if (ok(fa) and fa == fb) or (ok(ma) and ma == mb): continue
        if set(vectors[a]) & set(vectors[b]):
            cousA.append(kA(a, b)); cousB.append(kB(a, b))
        else:
            unrelA.append(kA(a, b)); unrelB.append(kB(a, b))

    def stats(lst):
        a = np.array(lst) if lst else np.array([0.0])
        return {'n': len(lst), 'mean': float(a.mean()), 'median': float(np.median(a)), 'std': float(a.std())}

    # Cross-form agreement over all sampled kin pairs (exclude unrelated zeros for rank signal)
    allA = fullA + halfA + cousA + unrelA
    allB = fullB + halfB + cousB + unrelB
    rho_all, _ = spearmanr(allA, allB)
    kinA = fullA + halfA + cousA
    kinB = fullB + halfB + cousB
    rho_kin, _ = spearmanr(kinA, kinB)

    out = {
        'formA_tiers': {'full': stats(fullA), 'half': stats(halfA), 'cousin': stats(cousA), 'unrelated': stats(unrelA)},
        'formB_tiers': {'full': stats(fullB), 'half': stats(halfB), 'cousin': stats(cousB), 'unrelated': stats(unrelB)},
        'spearman_formA_vs_formB_all_pairs': float(rho_all),
        'spearman_formA_vs_formB_kin_only': float(rho_kin),
        'n_all_pairs': len(allA), 'n_kin_pairs': len(kinA),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    import json
    with open(args.out_json, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
