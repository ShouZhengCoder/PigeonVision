"""Stage 7 — 构建信鸽族谱图。

节点 = 足环号(ring)。被拍照鸽子通过 pigeon.csv 的 PG_ID 锚定到环号空间。
边 = parent→child, 带角色(sire/dam)。

建图逻辑(每只被拍照鸽子 P, 环号=PG_ID, 解析行有 6 字段):
  father F, mother M           → F→P(sire), M→P(dam)            [1 代]
  pgf, pgm (F 的父母)          → pgf→F, pgm→F                    [2 代]
  mgf, mgm (M 的父母)          → mgf→M, mgm→M                    [2 代]
链式延深: 若父/母环号本身是被拍照鸽子(在 ring→img_id 中), 则并入它自己解析行的
  父母/祖父母边(去重), 把族谱延深到 3+ 代。

founder = 无入度(无父母记录)的节点。
环检测 = pedigree 应为 DAG, 记录噪声/近交可能造成环, 报告之。
"""
import argparse
import os
import re
import sys
from collections import defaultdict, deque

import pandas as pd

# 合法足环号: 字母前缀+数字年份+ '-' + 数字段(如 B05-6045278 / DV02098-08-2100 / NL15-1273729)。
# 过滤 details.txt 里被截断的畸形环号(如 B98 / AU04 / B839, 无 '-' 或过短), 它们会造成假环。
VALID_RING = re.compile(r'^[A-Z]{1,4}\d+-[A-Z0-9-]+$')


def ok(ring):
    return bool(ring) and bool(VALID_RING.match(ring)) and len(ring) >= 6


def add_edge(parents_of, children_of, child, parent, role, edge_set):
    """加一条 parent→child 边(去重)。"""
    key = (parent, child)
    if key in edge_set:
        return
    edge_set.add(key)
    children_of[parent].add(child)
    parents_of[child].add(parent)


def main():
    ap = argparse.ArgumentParser(description='构建族谱图')
    ap.add_argument('--pedigree', default='data/pedigree/parsed_pedigree.csv')
    ap.add_argument('--pigeon', default='data/extracted/datasetXGN/pigeon.csv')
    ap.add_argument('--out-dir', default='data/pedigree')
    args = ap.parse_args()

    # 1. 加载解析族谱
    ped = pd.read_csv(args.pedigree, dtype=str).fillna('')
    ped['img_id'] = ped['img_id'].astype(str)
    ped_row = {r['img_id']: r for _, r in ped.iterrows()}

    # 2. 加载 pigeon.csv: img_id→PG_ID, 以及 ring→img_id(链式延深用)
    pg = pd.read_csv(args.pigeon, on_bad_lines='skip', engine='python')
    pg['ID'] = pg['ID'].astype(str)
    pg['PG_ID'] = pg['PG_ID'].astype(str).str.strip()
    img2ring = dict(zip(pg['ID'], pg['PG_ID']))
    # ring → 第一个 img_id(同一只鸟可能多次拍照, 取首个用于延深)
    ring2img = {}
    for iid, ring in img2ring.items():
        if ring and ring not in ring2img:
            ring2img[ring] = iid

    # 3. 建图
    children_of = defaultdict(set)   # parent -> {children}
    parents_of = defaultdict(set)    # child -> {parents}
    edge_set = set()                 # (parent, child) 去重
    edges_with_role = []             # (parent, child, role) 输出用

    def emit(parent, child, role):
        if not ok(parent) or not ok(child) or parent == child:
            return
        if (parent, child) in edge_set:
            return
        add_edge(parents_of, children_of, child, parent, role, edge_set)
        edges_with_role.append((parent, child, role))

    chained = 0
    for iid, row in ped_row.items():
        ring = img2ring.get(iid, '')
        if not ring:
            continue  # 无环号, 无法在环号空间定位, 跳过
        f, m = row['father'], row['mother']
        pgf, pgm = row['pgf'], row['pgm']
        mgf, mgm = row['mgf'], row['mgm']
        # 1 代: 父母 → P
        if f:
            emit(f, ring, 'sire')
        if m:
            emit(m, ring, 'dam')
        # 2 代: 祖父母 → 父; 外祖父母 → 母
        if f and pgf:
            emit(pgf, f, 'sire')
        if f and pgm:
            emit(pgm, f, 'dam')
        if m and mgf:
            emit(mgf, m, 'sire')
        if m and mgm:
            emit(mgm, m, 'dam')
        # 链式延深: 若父/母本身被拍照, 并入其解析行的父母边(3+ 代)
        for parent_ring, parent_pgf, parent_pgm, parent_f, parent_m in [
            (f, pgf, pgm, None, None), (m, mgf, mgm, None, None)]:
            if parent_ring and parent_ring in ring2img:
                prow = ped_row.get(ring2img[parent_ring])
                if prow is not None:
                    pf, pm = prow['father'], prow['mother']
                    if pf:
                        emit(pf, parent_ring, 'sire')
                        chained += 1
                    if pm:
                        emit(pm, parent_ring, 'dam')
                        chained += 1
                    # 曾祖父母(深度3): parent 的祖父母 → parent 的父母
                    if pf and prow['pgf']:
                        emit(prow['pgf'], pf, 'sire')
                    if pf and prow['pgm']:
                        emit(prow['pgm'], pf, 'dam')
                    if pm and prow['mgf']:
                        emit(prow['mgf'], pm, 'sire')
                    if pm and prow['mgm']:
                        emit(prow['mgm'], pm, 'dam')

    nodes = set(children_of.keys()) | set(parents_of.keys())
    for r in img2ring.values():
        if ok(r):
            nodes.add(r)

    # 4. founder = 无入度
    founders = sorted([n for n in nodes if len(parents_of.get(n, set())) == 0])

    # 5. 环检测(DFS, 迭代)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}
    cycles = []
    for start in nodes:
        if color[start] != WHITE:
            continue
        stack = [(start, iter(sorted(children_of.get(start, set()))))]
        color[start] = GRAY
        path = [start]
        while stack:
            node, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                color[node] = BLACK
                stack.pop()
                path.pop()
                continue
            if color.get(nxt, WHITE) == GRAY:
                # 找到环
                idx = path.index(nxt) if nxt in path else 0
                cycles.append(path[idx:] + [nxt])
                if len(cycles) >= 20:
                    break
                continue
            if color.get(nxt, WHITE) == WHITE:
                color[nxt] = GRAY
                path.append(nxt)
                stack.append((nxt, iter(sorted(children_of.get(nxt, set())))))
        if len(cycles) >= 20:
            break

    # 6. 深度: 每个节点到 founder 的最长路径(迭代拓扑)
    depth = {}
    # 按入度拓扑排序(忽略环边)
    indeg = {n: len(parents_of.get(n, set())) for n in nodes}
    dq = deque([n for n in nodes if indeg[n] == 0])
    depth0 = {n: 0 for n in dq}
    while dq:
        n = dq.popleft()
        d = depth0[n]
        depth[n] = d
        for c in children_of.get(n, set()):
            indeg[c] -= 1
            depth0[c] = max(depth0.get(c, 0), d + 1)
            if indeg[c] == 0:
                dq.append(c)
    # 环内节点 depth 未定, 标 -1

    # 7. 连通分量(无向)
    seen = set()
    comp_sizes = []
    for n in nodes:
        if n in seen:
            continue
        # BFS
        q = deque([n])
        seen.add(n)
        sz = 0
        while q:
            x = q.popleft()
            sz += 1
            for y in (children_of.get(x, set()) | parents_of.get(x, set())):
                if y not in seen:
                    seen.add(y)
                    q.append(y)
        comp_sizes.append(sz)
    comp_sizes.sort(reverse=True)

    # 8. 输出
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(edges_with_role, columns=['parent', 'child', 'role']).to_csv(
        os.path.join(args.out_dir, 'pedigree_edges.csv'), index=False)
    pd.DataFrame({'founder': founders}).to_csv(
        os.path.join(args.out_dir, 'founders.csv'), index=False)

    photo_rings = set(r for r in img2ring.values() if ok(r))
    photo_in_graph = photo_rings & nodes
    depth_vals = [d for d in depth.values() if d >= 0]
    print('=== build_pedigree_graph 统计 ===')
    print(f'节点数: {len(nodes)} (其中被拍照鸽子 {len(photo_in_graph)})')
    print(f'边数(去重): {len(edge_set)}')
    print(f'链式延深新增边: {chained}')
    print(f'founder 数(无入度): {len(founders)}')
    print(f'环检测: 发现 {len(cycles)} 个环(展示前 5)')
    for c in cycles[:5]:
        print(f'  {" -> ".join(c)}')
    if depth_vals:
        dv = pd.Series(depth_vals)
        print(f'深度(到 founder 最长路径)分布: max={dv.max()}, '
              f'mean={dv.mean():.2f}, 分布: {dv.value_counts().sort_index().to_dict()}')
    print(f'连通分量数: {len(comp_sizes)} (最大 {comp_sizes[0] if comp_sizes else 0}, '
          f'前5: {comp_sizes[:5]})')
    print(f'写出 pedigree_edges.csv ({len(edge_set)} 边), founders.csv ({len(founders)} founder)')


if __name__ == '__main__':
    main()
