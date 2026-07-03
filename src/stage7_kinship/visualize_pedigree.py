"""Stage 7 — 族谱图可视化(给老师/论文看图)。

两张图:
  1. pedigree_tree_<img_id>.png: 单只鸽子的 3 代族谱树, 标注 founder 码,
     直观展示"祖先唯一编码 + 后代 = 祖先码组合"。
  2. pedigree_component_sample.png: 主连通分量的采样子图(~120 节点),
     founder 与非 founder 分色, 展示"大图"结构。
"""
import argparse
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

VALID_RING = __import__('re').compile(r'^[A-Z]{1,4}\d+-[A-Z0-9-]+$')


def ok(r):
    return bool(r) and bool(VALID_RING.match(r)) and len(r) >= 6


def draw_tree(ped_row, founder_code, img_id, out_path):
    """画 3 代族谱树: 祖父母(顶) -> 父母 -> img_id(底)。"""
    row = ped_row[img_id]
    ring = row.get('ring', img_id)
    f, m = row['father'], row['mother']
    pgf, pgm, mgf, mgm = row['pgf'], row['pgm'], row['mgf'], row['mgm']

    # 节点: (label, x, y, is_founder)
    nodes = []
    edges = []
    # 顶层祖父母
    top = [n for n in [pgf, pgm, mgf, mgm] if ok(n)]
    for n in top:
        nodes.append((n, n, None, 2))
    # 父母
    if ok(f):
        nodes.append((f, f, None, 1))
        if ok(pgf):
            edges.append((pgf, f))
        if ok(pgm):
            edges.append((pgm, f))
    if ok(m):
        nodes.append((m, m, None, 1))
        if ok(mgf):
            edges.append((mgf, m))
        if ok(mgm):
            edges.append((mgm, m))
    # 本人
    nodes.append((img_id, f'{img_id}\n{ring}', None, 0))
    if ok(f):
        edges.append((f, img_id))
    if ok(m):
        edges.append((m, img_id))

    # x 坐标: 祖父母均匀铺开, 父母居中于其父母, 本人居中
    pos = {}
    top_valid = [n for n in [pgf, pgm, mgf, mgm] if ok(n)]
    for i, n in enumerate(top_valid):
        pos[n] = (i * 2 - (len(top_valid) - 1), 2)
    if ok(f):
        xs = [pos[n][0] for n in [pgf, pgm] if ok(n) and n in pos]
        pos[f] = (sum(xs) / len(xs) if xs else -1.5, 1)
    if ok(m):
        xs = [pos[n][0] for n in [mgf, mgm] if ok(n) and n in pos]
        pos[m] = (sum(xs) / len(xs) if xs else 1.5, 1)
    pos[img_id] = (0, 0)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # 边
    for a, b in edges:
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                'k-', lw=1.2, zorder=1)
    # 节点
    for key, label, _, y in nodes:
        x, y = pos[key]
        is_founder = ok(key) and key in founder_code
        color = '#4C9F70' if is_founder else '#3A7CA5'
        fc = '#D9F0DD' if is_founder else '#DCE9F2'
        ax.scatter([x], [y], s=2600, c=fc, edgecolors=color,
                   linewidths=2, zorder=2)
        txt = label
        if is_founder:
            txt = f'{label}\n[{founder_code[key]}]'
        ax.text(x, y, txt, ha='center', va='center', fontsize=7.5,
                zorder=3, family='monospace')
    # 图例
    ax.scatter([], [], s=200, c='#D9F0DD', edgecolors='#4C9F70',
               linewidths=2, label='founder (no parents, code F)')
    ax.scatter([], [], s=200, c='#DCE9F2', edgecolors='#3A7CA5',
               linewidths=2, label='non-founder (has parents)')
    ax.legend(loc='upper right', fontsize=8)

    # 本人字符串码
    code = row.get('code', '')
    if code:
        ax.text(0.01, 0.02, f'{img_id} ancestry path code (Form A):\n{code}',
                transform=ax.transAxes, fontsize=7, family='monospace',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', fc='#FFF8E1', ec='#E0C16A'))

    ax.set_title(f'Pigeon {img_id} ({ring}) - 3-gen pedigree & founder codes',
                 fontsize=11)
    ax.axis('off')
    ax.set_ylim(-0.6, 2.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def draw_component_sample(edges_df, founder_set, out_path, n=120):
    """主连通分量采样子图。"""
    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(r['parent'], r['child'])
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    if not comps:
        return
    main = G.subgraph(comps[0]).copy()
    # 从最大度节点 BFS 采样
    hub = max(main.degree, key=lambda x: x[1])[0]
    seen = {hub}
    frontier = [hub]
    while frontier and len(seen) < n:
        nxt = []
        for v in frontier:
            for u in main.neighbors(v):
                if u not in seen:
                    seen.add(u)
                    nxt.append(u)
                    if len(seen) >= n:
                        break
            if len(seen) >= n:
                break
        frontier = nxt
    sub = main.subgraph(seen).copy()
    pos = nx.spring_layout(sub, seed=7, k=0.9, iterations=80)

    fig, ax = plt.subplots(figsize=(11, 8))
    f_nodes = [n for n in sub if n in founder_set]
    nf_nodes = [n for n in sub if n not in founder_set]
    nx.draw_networkx_nodes(sub, pos, nodelist=f_nodes, node_size=90,
                           node_color='#4C9F70', alpha=0.85, ax=ax,
                           label=f'founder ({len(f_nodes)})')
    nx.draw_networkx_nodes(sub, pos, nodelist=nf_nodes, node_size=90,
                           node_color='#3A7CA5', alpha=0.85, ax=ax,
                           label=f'non-founder ({len(nf_nodes)})')
    nx.draw_networkx_edges(sub, pos, edge_color='#888', width=0.4,
                           alpha=0.5, ax=ax)
    ax.set_title(f'Pedigree graph: sample of largest component '
                 f'({len(sub)} of {len(comps[0])} nodes, BFS from hub {hub})',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='族谱图可视化')
    ap.add_argument('--pedigree', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--codes', default='data/pedigree/founder_codes.csv')
    ap.add_argument('--ancestry', default='data/pedigree/ancestry_codes.csv')
    ap.add_argument('--edges', default='data/pedigree/pedigree_edges.csv')
    ap.add_argument('--pigeon', default='data/extracted/datasetXGN/pigeon.csv')
    ap.add_argument('--target', default='606803')
    ap.add_argument('--out-dir', default='paper/fig')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ped = pd.read_csv(args.pedigree, dtype=str).fillna('')
    ped['img_id'] = ped['img_id'].astype(str)
    pg = pd.read_csv(args.pigeon, on_bad_lines='skip', engine='python')
    pg['ID'] = pg['ID'].astype(str)
    img2ring = dict(zip(pg['ID'], pg['PG_ID'].astype(str).str.strip()))
    anc = pd.read_csv(args.ancestry, dtype=str).fillna('')
    code_of = dict(zip(anc['img_id'].astype(str), anc['code']))
    ped_row = {}
    for _, r in ped.iterrows():
        ped_row[r['img_id']] = {**r.to_dict(), 'ring': img2ring.get(r['img_id'], ''),
                                'code': code_of.get(r['img_id'], '')}
    fc = pd.read_csv(args.codes, dtype=str)
    founder_code = dict(zip(fc['founder'], fc['code']))
    founder_set = set(fc['founder'])

    # 图1: 目标鸽子的族谱树
    if args.target in ped_row:
        p = os.path.join(args.out_dir, f'pedigree_tree_{args.target}.png')
        draw_tree(ped_row, founder_code, args.target, p)
        print(f'图1: {p}')
        print(f'  {args.target} 码: {code_of.get(args.target, "")}')

    # 图2: 主连通分量采样
    edges_df = pd.read_csv(args.edges, dtype=str)
    p2 = os.path.join(args.out_dir, 'pedigree_component_sample.png')
    draw_component_sample(edges_df, founder_set, p2)
    print(f'图2: {p2}')


if __name__ == '__main__':
    main()
