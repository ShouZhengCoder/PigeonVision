"""Stage 7 — 亲缘编码与远近度量(贡献②核心)。

两套等价表示:
  形式 A(字符串祖先路径码, 对应老师原意): founder 唯一短码 F1..Fn, 后代码 =
    排序的 "F{idx}:{min_depth}" token 序列(祖先到后代的最近辈分)。
  形式 B(founder/祖先贡献向量 + 亲缘系数, 数学度量): v_i[a] = Σ_{路径 i→a}
    (1/2)^depth, 亲缘系数 k(i,j) = v_i · v_j = Σ_{共享祖先 a} v_i[a]*v_j[a]。
    k ≈ 2×标准亲缘系数 φ_ij(founder 视为非近交, 忽略 (1+F_a) 修正)。

parent_map 统一构建: 被拍照鸽子用自身解析行的父母(权威); 非被拍照祖先用提及它
  的子代行的祖父母字段兜底。

hybrid 兜底: 无结构系谱的鸽子(空贡献向量)用 relations.csv 的 blood_id 集合
  (IDF 加权)做 fallback 向量(blood_id 本身即祖先环号, 与结构向量同索引空间)。

展开用路径枚举 DFS, max_depth=6, 路径内成环守卫(避免 5 个真实记录环造成死循环)。
"""
import argparse
import math
import os
import re
from collections import defaultdict

import pandas as pd

VALID_RING = re.compile(r'^[A-Z]{1,4}\d+-[A-Z0-9-]+$')


def ok(ring):
    return bool(ring) and bool(VALID_RING.match(ring)) and len(ring) >= 6


def main():
    ap = argparse.ArgumentParser(description='亲缘编码与度量')
    ap.add_argument('--pedigree', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--pigeon', default='data/extracted/datasetXGN/pigeon.csv')
    ap.add_argument('--relations', default='data/extracted/datasetXGN/relations.csv')
    ap.add_argument('--usable-meta', default='outputs/iris_normalized/normalize_meta.csv')
    ap.add_argument('--out-dir', default='data/pedigree')
    ap.add_argument('--max-depth', type=int, default=6)
    args = ap.parse_args()

    ped = pd.read_csv(args.pedigree, dtype=str).fillna('')
    ped['img_id'] = ped['img_id'].astype(str)
    ped_row = {r['img_id']: r for _, r in ped.iterrows()}

    pg = pd.read_csv(args.pigeon, on_bad_lines='skip', engine='python')
    pg['ID'] = pg['ID'].astype(str)
    pg['PG_ID'] = pg['PG_ID'].astype(str).str.strip()
    img2ring = {iid: r for iid, r in zip(pg['ID'], pg['PG_ID']) if ok(r)}
    ring2img = {}
    for iid, ring in img2ring.items():
        ring2img.setdefault(ring, iid)

    # 1. parent_map(统一): pass1 权威(自身行), pass2 兜底(子代行的祖父母字段)
    parent_map = {}
    for iid, row in ped_row.items():
        ring = img2ring.get(iid)
        if not ring:
            continue
        f, m = row['father'], row['mother']
        if f or m:
            parent_map.setdefault(ring, (f, m))
    for iid, row in ped_row.items():
        ring = img2ring.get(iid)
        if not ring:
            continue
        f, m = row['father'], row['mother']
        if f and ok(f) and f not in parent_map and (row['pgf'] or row['pgm']):
            parent_map[f] = (row['pgf'], row['pgm'])
        if m and ok(m) and m not in parent_map and (row['mgf'] or row['mgm']):
            parent_map[m] = (row['mgf'], row['mgm'])

    # 2. founder = 有族谱记录但无父母的环号(贡献向量的终止节点)
    all_rings = set(parent_map.keys())
    for f, m in parent_map.values():
        for p in (f, m):
            if ok(p):
                all_rings.add(p)
    founders = sorted(r for r in all_rings if r not in parent_map)
    founder_idx = {r: i for i, r in enumerate(founders)}
    pd.DataFrame({'founder': founders, 'code': [f'F{i}' for i in range(len(founders))]})\
        .to_csv(os.path.join(args.out_dir, 'founder_codes.csv'), index=False)

    # 3. 祖先展开(路径枚举, 路径内成环守卫)
    def expand(start):
        contrib = defaultdict(float)
        min_depth = {}
        stack = [(start, 0, 1.0, (start,))]
        while stack:
            node, d, c, path = stack.pop()
            if d > 0:
                contrib[node] += c
                if node not in min_depth or d < min_depth[node]:
                    min_depth[node] = d
            if d >= args.max_depth:
                continue
            fa, mo = parent_map.get(node, ('', ''))
            for parent in (fa, mo):
                if ok(parent) and parent not in path:
                    stack.append((parent, d + 1, c * 0.5, path + (parent,)))
        return contrib, min_depth

    # 4. relations.csv IDF fallback(无结构系谱时用)
    rel = pd.read_csv(args.relations, header=None, names=['blood_id', 'img_id'])
    rel['img_id'] = rel['img_id'].astype(str)
    rel = rel[rel['blood_id'].apply(ok)]
    df_count = rel.groupby('blood_id').size()
    n_docs = rel['img_id'].nunique()
    idf = {b: math.log(1 + n_docs / c) for b, c in df_count.items()}
    rel_sets = rel.groupby('img_id')['blood_id'].apply(set).to_dict()

    # 5. 为可用虹膜集每只鸽子构建贡献向量
    nm = pd.read_csv(args.usable_meta)
    usable = sorted(nm[nm.status == 'success']['img_id'].astype(str).tolist())

    vectors = {}      # img_id -> {ancestor_ring: contrib}
    code_str = {}     # img_id -> 字符串码(Form A)
    src = {}          # img_id -> 'structured' / 'fallback' / 'empty'
    vec_long = []     # (img_id, ancestor, contrib) 持久化
    for iid in usable:
        ring = img2ring.get(iid)
        v, mdepth = expand(ring) if ring else (defaultdict(float), {})
        if v:
            vectors[iid] = dict(v)
            # Form A: founder 子集 + 最近深度(展开时直接记录, 非反推)
            ftoks = sorted((mdepth[a], founder_idx[a]) for a in v if a in founder_idx)
            code_str[iid] = '|'.join(f'F{idx}:{d}' for d, idx in ftoks[:20])
            src[iid] = 'structured'
            for anc, c in v.items():
                vec_long.append((iid, anc, c))
        else:
            bs = rel_sets.get(iid, set())
            if bs:
                vectors[iid] = {b: idf[b] for b in bs if b in idf}
                src[iid] = 'fallback'
                code_str[iid] = '|'.join(sorted(f'F{founder_idx.get(b, -1)}' for b in bs if b in founder_idx)[:20])
            else:
                vectors[iid] = {}
                src[iid] = 'empty'
                code_str[iid] = ''

    # 6. 持久化
    pd.DataFrame(vec_long, columns=['img_id', 'ancestor', 'contribution']).to_csv(
        os.path.join(args.out_dir, 'contribution_vectors.csv'), index=False)
    pd.DataFrame({'img_id': usable, 'ring': [img2ring.get(i, '') for i in usable],
                  'code': [code_str[i] for i in usable], 'source': [src[i] for i in usable]})\
        .to_csv(os.path.join(args.out_dir, 'ancestry_codes.csv'), index=False)

    # 7. 亲缘函数 k(i,j) = 共享祖先贡献点积(仅同源结构向量间语义一致)
    def kinship(i, j):
        vi, vj = vectors.get(i, {}), vectors.get(j, {})
        if not vi or not vj:
            return 0.0
        small, large = (vi, vj) if len(vi) <= len(vj) else (vj, vi)
        return sum(c * large.get(a, 0.0) for a, c in small.items())

    # 8. 统计 + demo
    from collections import Counter
    src_cnt = Counter(src.values())
    print('=== kinship_encoding 统计 ===')
    print(f'可用鸽子数: {len(usable)}')
    print(f'向量来源: {dict(src_cnt)}')
    print(f'founder 数: {len(founders)}')
    print(f'parent_map 覆盖环号: {len(parent_map)}')
    struct = [i for i in usable if src[i] == 'structured']
    nnz = sum(len(vectors[i]) for i in struct)
    print(f'结构向量鸽子: {len(struct)}, 平均祖先数: {nnz/max(len(struct),1):.1f}')

    # demo: 在结构化子集采样找已知关系对, 验证 k 单调性(全同胞>半同胞>共享祖先>无关)
    import itertools, random, statistics as st
    random.seed(0)
    struct = [i for i in usable if src[i] == 'structured']
    par_of = {i: (ped_row[i]['father'], ped_row[i]['mother'])
              for i in struct if i in ped_row}
    sample = random.sample(struct, min(800, len(struct)))
    full, half, cous, unrel = [], [], [], []
    for a, b in itertools.combinations(sample, 2):
        fa, ma = par_of.get(a, ('', ''))
        fb, mb = par_of.get(b, ('', ''))
        share_f = ok(fa) and fa == fb
        share_m = ok(ma) and ma == mb
        k = kinship(a, b)
        if share_f and share_m:
            full.append(k)
        elif share_f or share_m:
            half.append(k)
        elif set(vectors[a]) & set(vectors[b]):
            cous.append(k)
        else:
            unrel.append(k)

    def stats(lst, name):
        if lst:
            print(f'  {name}: n={len(lst)}, mean k={st.mean(lst):.4f}, '
                  f'median={st.median(lst):.4f}, max={max(lst):.4f}')
        else:
            print(f'  {name}: n=0')

    print('--- 亲缘单调性验证(结构化子集采样 800) ---')
    stats(full, '全同胞(共父母) ')
    stats(half, '半同胞(共一亲) ')
    stats(cous, '共享祖先非同胞')
    stats(unrel, '无共享祖先    ')
    if '606803' in code_str:
        print(f'  606803 字符串码(Form A): {code_str["606803"][:120]}')

    print(f'写出 founder_codes.csv / contribution_vectors.csv / ancestry_codes.csv')


if __name__ == '__main__':
    main()
