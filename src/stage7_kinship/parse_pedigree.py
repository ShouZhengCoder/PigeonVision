"""Stage 7 — 解析 details.txt 为结构化族谱表。

每行格式: <img_id>\t<血统书原文>
原文含结构化字段: 父亲:/母亲:/祖父:/祖母:/外祖父:/外祖母: + 足环号
本脚本抽取这 6 个角色字段(每角色取首次出现), 输出 data/pedigree/parsed_pedigree.csv
并打印覆盖率统计(总体 / 与可用虹膜集交集 / 父母双全 / 可链式延深潜力)。

读取约定: details.txt 为 UTF-8(含噪声), 用 errors='replace' 容错。
足环号正则: [A-Z]{1,4}\d[\d-]*\d  覆盖 B05-6045278 / DV02098-08-2100 / NL15-1273729 / CHN2006-01-016434 等格式。
"""
import argparse
import os
import re
import sys
from collections import Counter

import pandas as pd
from tqdm import tqdm

# 角色关键词按"最长前缀优先"排列, 避免外祖父被当成祖父匹配
ROLE_RE = re.compile(r'(外祖父|外祖母|祖父|祖母|父亲|母亲):([A-Z]{1,4}\d[\d-]*\d)')
ROLE_MAP = {'父亲': 'father', '母亲': 'mother', '祖父': 'pgf',
            '祖母': 'pgm', '外祖父': 'mgf', '外祖母': 'mgm'}
ROLE_COLS = ['father', 'mother', 'pgf', 'pgm', 'mgf', 'mgm']


def parse_line(line):
    """返回 (img_id, {col: ring}) 或 None。每角色只取首次出现。"""
    parts = line.rstrip('\n').split('\t', 1)
    if len(parts) < 2:
        return None
    img_id, text = parts[0].strip(), parts[1]
    if not img_id:
        return None
    rec = {c: '' for c in ROLE_COLS}
    seen = set()
    for role, ring in ROLE_RE.findall(text):
        col = ROLE_MAP[role]
        if col not in seen:  # 首次出现 = 个体自身的该角色(后续出现多为祖先叙述)
            rec[col] = ring
            seen.add(col)
    return img_id, rec


def main():
    ap = argparse.ArgumentParser(description='解析 details.txt 族谱字段')
    ap.add_argument('--input', default='data/extracted/datasetXGN/details.txt')
    ap.add_argument('--out', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--pigeon', default='data/extracted/datasetXGN/pigeon.csv',
                    help='pigeon.csv (用于 PG_ID 链式延深统计)')
    ap.add_argument('--usable-meta', default='outputs/iris_normalized/normalize_meta.csv',
                    help='normalize_meta.csv (用于与可用虹膜集求交集)')
    args = ap.parse_args()

    rows = []
    total = skipped = 0
    role_counts = Counter()
    with open(args.input, encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, desc='parse'):
            total += 1
            r = parse_line(line)
            if r is None:
                skipped += 1
                continue
            img_id, rec = r
            rec['img_id'] = img_id
            rows.append(rec)
            for c in ROLE_COLS:
                if rec[c]:
                    role_counts[c] += 1

    df = pd.DataFrame(rows, columns=['img_id'] + ROLE_COLS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    any_role = (df[ROLE_COLS] != '').any(axis=1).sum()
    both_parents = ((df['father'] != '') & (df['mother'] != '')).sum()
    print('=== parse_pedigree 覆盖率 ===')
    print(f'总行数: {total}')
    print(f'解析行数: {len(df)} (跳过 {skipped})')
    print(f'有 >=1 角色字段: {any_role} ({any_role/len(df):.1%})')
    print(f'父母双全: {both_parents} ({both_parents/len(df):.1%})')
    for c in ROLE_COLS:
        print(f'  {c}: {role_counts[c]} ({role_counts[c]/len(df):.1%})')

    # 与可用虹膜集交集
    try:
        nm = pd.read_csv(args.usable_meta)
        usable = set(nm[nm.status == 'success']['img_id'].astype(str))
        df_u = df[df.img_id.astype(str).isin(usable)]
        print(f'--- 可用 success 虹膜集: {len(usable)} ---')
        print(f'  在集合内且有解析记录: {len(df_u)}')
        print(f'  其中 >=1 角色: {(df_u[ROLE_COLS] != "").any(axis=1).sum()}')
        print(f'  其中父母双全: {((df_u.father != "") & (df_u.mother != "")).sum()}')
    except Exception as e:
        print(f'(可用集统计跳过: {e})', file=sys.stderr)

    # 链式延深潜力: 捕获到的父母环号是否本身是被拍照鸽子(出现在 PG_ID)
    try:
        pg = pd.read_csv(args.pigeon, on_bad_lines='skip', engine='python')
        pg_id_set = set(pg['PG_ID'].dropna().astype(str))
        parent_rings = set()
        for c in ROLE_COLS:
            parent_rings.update(df[c].replace('', pd.NA).dropna().astype(str))
        in_pg = sum(1 for r in parent_rings if r in pg_id_set)
        print('--- 链式延深潜力 ---')
        print(f'捕获到的唯一父母/祖父母环号: {len(parent_rings)}')
        print(f'  其中本身是被拍照鸽子(在 PG_ID 中): {in_pg} ({in_pg/max(len(parent_rings),1):.1%})')
    except Exception as e:
        print(f'(pigeon.csv 统计跳过: {e})', file=sys.stderr)

    print(f'写出 {args.out} ({len(df)} 行)')


if __name__ == '__main__':
    main()
