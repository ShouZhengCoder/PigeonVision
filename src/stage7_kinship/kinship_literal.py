"""Stage 7 — 字面版亲缘编码(严格贴合老师原意)。

老师原话: "找到没有入度的图就是祖先, 给每个祖先唯一编码; 对于后代, 就是
父母节点的编码拼起来; 两个个体的亲疏关系就可通过编码获取, 找相同子序列,
以及编码长度。"

本脚本严格实现:
  - founder(无入度) = 唯一 F 码 (覆盖全图 42,609 个 founder, 不只可达子集)
  - 后代码 = 父码 ++ 母码 递归拼接 (深度截断 max_depth 防指数膨胀)
  - 亲疏 = LCS(码_i, 码_j) / max(|码_i|, |码_j|)  (公共子序列比 + 编码长度)

与 kinship_encoding.py 的向量点积版做对比验证。
"""
import argparse
import itertools
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd

VALID_RING = __import__('re').compile(r'^[A-Z]{1,4}\d+-[A-Z0-9-]+$')


def ok(r):
    return bool(r) and bool(VALID_RING.match(r)) and len(r) >= 6


def lcs_len(a, b):
    """两 token 序列的 LCS 长度(DP)。"""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(m):
        prev = 0
        ai = a[i]
        for j in range(n):
            cur = dp[j + 1]
            dp[j + 1] = prev + 1 if ai == b[j] else max(dp[j], dp[j + 1])
            prev = cur
    return dp[n]


def main():
    ap = argparse.ArgumentParser(description='字面版亲缘编码(递归拼接+LCS)')
    ap.add_argument('--pedigree', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--pigeon', default='data/extracted/datasetXGN/pigeon.csv')
    ap.add_argument('--founders', default='data/pedigree/founders.csv')
    ap.add_argument('--usable-meta', default='outputs/iris_normalized/normalize_meta.csv')
    ap.add_argument('--max-depth', type=int, default=5)
    args = ap.parse_args()
    random.seed(0)

    ped = pd.read_csv(args.pedigree, dtype=str).fillna('')
    ped['img_id'] = ped['img_id'].astype(str)
    ped_row = {r['img_id']: r for _, r in ped.iterrows()}
    pg = pd.read_csv(args.pigeon, on_bad_lines='skip', engine='python')
    pg['ID'] = pg['ID'].astype(str)
    img2ring = {iid: r for iid, r in zip(pg['ID'], pg['PG_ID'].astype(str).str.strip()) if ok(r)}

    # parent_map (与 kinship_encoding 一致: 自身行权威 + 祖父母字段兜底)
    parent_map = {}
    for iid, row in ped_row.items():
        ring = img2ring.get(iid)
        if not ring:
            continue
        if row['father'] or row['mother']:
            parent_map.setdefault(ring, (row['father'], row['mother']))
    for iid, row in ped_row.items():
        ring = img2ring.get(iid)
        if not ring:
            continue
        f, m = row['father'], row['mother']
        if f and ok(f) and f not in parent_map and (row['pgf'] or row['pgm']):
            parent_map[f] = (row['pgf'], row['pgm'])
        if m and ok(m) and m not in parent_map and (row['mgf'] or row['mgm']):
            parent_map[m] = (row['mgf'], row['mgm'])

    # 全图 founder (42,609), 每个分配唯一 F 码
    fdf = pd.read_csv(args.founders, dtype=str)
    all_rings = set(parent_map.keys())
    for f, m in parent_map.values():
        for p in (f, m):
            if ok(p):
                all_rings.add(p)
    founders_set = set(r for r in all_rings if r not in parent_map) | set(fdf['founder'])
    founders = sorted(founders_set)
    fcode = {r: i for i, r in enumerate(founders)}
    # 每个 ring 一个短 token: founder 用 F{idx}, 非 founder 用 N{idx}
    nonf = [r for r in all_rings if r not in founders_set]
    ncode = {r: i + len(founders) for i, r in enumerate(nonf)}
    TOK_F = 0
    TOK_N = 1

    def token(ring):
        if ring in fcode:
            return (TOK_F, fcode[ring])
        return (TOK_N, ncode.get(ring, -1))

    # 递归拼接码: code(child) = code(father) ++ code(mother), 递归到 founder
    # 终止。无深度截断(贴合老师无界递归原意), 用路径内成环守卫处理 5 个真实记录环。
    def literal_code(node, path=()):
        if node in path:           # 成环守卫
            return [token(node)]
        if node not in parent_map:  # founder
            return [token(node)]
        fa, mo = parent_map[node]
        code = []
        new_path = path + (node,)
        if ok(fa):
            code.extend(literal_code(fa, new_path))
        if ok(mo):
            code.extend(literal_code(mo, new_path))
        return code or [token(node)]

    # 编码所有可达节点 + 可用鸽子
    nm = pd.read_csv(args.usable_meta)
    usable = sorted(nm[nm.status == 'success']['img_id'].astype(str).tolist())
    codes = {}
    for iid in usable:
        ring = img2ring.get(iid)
        if ring:
            codes[iid] = literal_code(ring)
        else:
            codes[iid] = []

    # 验证: 全/半同胞/共享祖先/无关 (与 validate_kinship 同款构造)
    struct = [i for i in usable if codes[i]]
    par_of = {i: (ped_row[i]['father'], ped_row[i]['mother']) for i in struct if i in ped_row}
    by_f, by_m = defaultdict(list), defaultdict(list)
    for i in struct:
        f, m = par_of.get(i, ('', ''))
        if ok(f):
            by_f[f].append(i)
        if ok(m):
            by_m[m].append(i)
    full, half, cous, unrel = [], [], [], []
    seen = set()
    for grp in list(by_f.values()) + list(by_m.values()):
        for a, b in itertools.combinations(grp, 2):
            if (a, b) in seen:
                continue
            seen.add((a, b))
            fa, ma = par_of.get(a, ('', ''))
            fb, mb = par_of.get(b, ('', ''))
            sf = ok(fa) and fa == fb
            sm = ok(ma) and ma == mb
            s = lcs_sim(codes[a], codes[b])
            if sf and sm:
                full.append(s)
            elif sf or sm:
                half.append(s)
    sample = random.sample(struct, min(800, len(struct)))
    for a, b in itertools.combinations(sample, 2):
        if (a, b) in seen:
            continue
        fa, ma = par_of.get(a, ('', ''))
        fb, mb = par_of.get(b, ('', ''))
        if (ok(fa) and fa == fb) or (ok(ma) and ma == mb):
            continue
        s = lcs_sim(codes[a], codes[b])
        (cous if set(codes[a]) & set(codes[b]) else unrel).append(s)

    def stats(lst):
        if not lst:
            return 'n=0'
        a = np.array(lst)
        return f'n={len(lst)}, mean={a.mean():.4f}, median={np.median(a):.4f}'

    print('=== 字面版(递归拼接+LCS) 验证 ===')
    print(f'编码节点: 全图 founder {len(founders)} + 可达非founder {len(nonf)}; '
          f'可用鸽子有码 {len(struct)}/{len(usable)}')
    lens = [len(c) for c in codes.values() if c]
    print(f'码长: mean={np.mean(lens):.1f}, max={max(lens)} '
          f'(无深度截断, 递归到 founder; 路径内成环守卫)')
    print('--- 单调性 ---')
    print(f'  全同胞:     {stats(full)}')
    print(f'  半同胞:     {stats(half)}')
    print(f'  共享祖先:   {stats(cous)}')
    print(f'  无共享祖先: {stats(unrel)}')

    # 保存字面码(可用鸽子)
    out = []
    for iid in usable:
        c = codes[iid]
        toks = []
        for t, idx in c:
            toks.append(f'F{idx}' if t == TOK_F else f'N{idx}')
        out.append({'img_id': iid, 'literal_code': '|'.join(toks), 'code_len': len(c)})
    os.makedirs('data/pedigree', exist_ok=True)
    pd.DataFrame(out).to_csv('data/pedigree/literal_codes.csv', index=False)
    print('写出 data/pedigree/literal_codes.csv')


def lcs_sim(a, b):
    if not a or not b:
        return 0.0
    return lcs_len(a, b) / max(len(a), len(b))


if __name__ == '__main__':
    main()
