"""Unsupervised control for the circularity concern: extract features with an
ImageNet-pretrained ResNet-34 (NO iris training, NO kinship labels, NO bloodline
text) and measure Spearman(iris_dist, pedigree_k). If significant, iris carries
genealogical structure intrinsically -- the correlation cannot be a text-derived
artifact. Also report the partial correlation controlling for the IDF proxy.
"""
import argparse, itertools, os, random
from collections import defaultdict
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from PIL import Image
from scipy.stats import spearmanr, pearsonr, rankdata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='outputs/iris_normalized')
    ap.add_argument('--meta', default='outputs/features/relation_supcon_256d_split_full/feature_db_meta.csv')
    ap.add_argument('--vectors', default='data/pedigree/contribution_vectors.csv')
    ap.add_argument('--pedigree', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--relations', default='data/extracted/datasetXGN/relations.csv')
    ap.add_argument('--out', default='outputs/reports/unsupervised_control.json')
    ap.add_argument('--size', type=int, default=224)
    args = ap.parse_args()
    random.seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ImageNet-pretrained ResNet-34, penultimate 512-d features
    import torchvision.models as M
    import torchvision.transforms as T
    net = M.resnet34(weights=M.ResNet34_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval().to(dev)
    tfm = T.Compose([T.Resize((args.size, args.size)), T.ToTensor(),
                     T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.405])])

    meta = pd.read_csv(args.meta, dtype=str); meta['img_id'] = meta['img_id'].astype(str)

    # build pairs FIRST (same logic as phaseB/proof scripts), then extract features
    # only for the individuals appearing in pairs (~2.5k, not all 25.7k).
    cv = pd.read_csv(args.vectors, dtype=str); cv['img_id'] = cv['img_id'].astype(str)
    vec = defaultdict(dict)
    for i, a, c in zip(cv['img_id'], cv['ancestor'], cv['contribution'].astype(float)):
        vec[i][a] = c
    meta_ids = set(meta['img_id'])
    struct = [i for i in vec if vec[i] and i in meta_ids]
    def k_ped(a, b):
        va, vb = vec.get(a, {}), vec.get(b, {})
        if not va or not vb: return 0.0
        s, l = (va, vb) if len(va) <= len(vb) else (vb, va)
        return sum(c * l.get(x, 0.0) for x, c in s.items())
    rel = pd.read_csv(args.relations, header=None, names=['blood_id', 'img_id']); rel['img_id'] = rel['img_id'].astype(str)
    df_c = rel.groupby('blood_id').size(); n_docs = rel['img_id'].nunique()
    idf = {b: np.log(1 + n_docs / c) for b, c in df_c.items()}
    rel_sets = rel.groupby('img_id')['blood_id'].apply(set).to_dict()
    def k_idf(a, b):
        sa, sb = rel_sets.get(a, set()), rel_sets.get(b, set())
        if not sa or not sb: return 0.0
        sh = sa & sb
        if not sh: return 0.0
        shared = sum(idf[x] for x in sh); ia = sum(idf[x] for x in sa); ib = sum(idf[x] for x in sb)
        return 0.7*shared/min(ia,ib) + 0.3*shared/(ia+ib-shared)
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
    pairs = full + half + cous + unrel
    need = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    print(f'extracting ImageNet-ResNet34 features for {len(need)} pair individuals on {dev}...')
    feats = np.zeros((len(need), 512), dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for i in range(0, len(need), bs):
            chunk = need[i:i+bs]
            imgs = []
            for iid in chunk:
                im = Image.open(os.path.join(args.images, f'{iid}.png')).convert('RGB')
                imgs.append(tfm(im))
            x = torch.stack(imgs).to(dev)
            feats[i:i+bs] = net(x).cpu().numpy()
    feats /= np.maximum(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12)
    idx = {iid: k for k, iid in enumerate(need)}
    print('features extracted')
    def d_iris(a, b): return float(np.linalg.norm(feats[idx[a]] - feats[idx[b]]))
    D = np.array([d_iris(a, b) for a, b in pairs])
    K = np.array([k_ped(a, b) for a, b in pairs])
    I = np.array([k_idf(a, b) for a, b in pairs])
    sp, _ = spearmanr(D, K); sp_i, _ = spearmanr(D, I)
    r_d = rankdata(D); r_k = rankdata(K); r_i = rankdata(I)
    def resid(y, x):
        b = np.polyfit(x, y, 1); return y - (b[0]*x + b[1])
    pr, pp = pearsonr(resid(r_d, r_i), resid(r_k, r_i))
    # tier means
    def tier_mean(t, ps):
        ds = [d_iris(a, b) for a, b in ps]
        return float(np.mean(ds)) if ds else float('nan')
    out = {'encoder': 'ImageNet-pretrained ResNet-34 (no iris/kinship/text training)',
           'n_pairs': len(pairs),
           'spearman_d_kped': float(sp), 'spearman_d_kidf': float(sp_i),
           'partial_spearman_d_kped_given_kidf': {'r': float(pr), 'p': float(pp)},
           'tier_mean_iris_dist': {'full': tier_mean('f', full), 'half': tier_mean('h', half),
                                    'cousin': tier_mean('c', cous), 'unrelated': tier_mean('u', unrel)},
           'monotone': bool(tier_mean('f', full) < tier_mean('h', half) < tier_mean('c', cous) < tier_mean('u', unrel))}
    print(f'\n=== UNSUPERVISED CONTROL (ImageNet ResNet-34, no training) ===')
    print(f'Spearman(d, k_ped) = {sp:.4f}   [cf. proof encoder -0.699]')
    print(f'Spearman(d, k_idf) = {sp_i:.4f}')
    print(f'partial Spearman(d, k_ped | k_idf) = {pr:.4f} (p={pp:.2e})')
    print(f'tier means: full {out["tier_mean_iris_dist"]["full"]:.3f} < half {out["tier_mean_iris_dist"]["half"]:.3f} < cousin {out["tier_mean_iris_dist"]["cousin"]:.3f} < unrel {out["tier_mean_iris_dist"]["unrelated"]:.3f}  monotone={out["monotone"]}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import json; json.dump(out, open(args.out, 'w'), indent=2)

if __name__ == '__main__':
    main()
