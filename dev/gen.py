#!/usr/bin/env python3
"""
Generate hidden-test-like cases for local benchmarking.

The official hidden cases are "of similar size and difficulty" to the six in
test_cases/, so this produces many randomized warehouse floors per level in the
same spirit, plus a larger "stress" set for timing safety and a "robust" set
that exercises schema features the examples don't (one-way aisles, fractional
weights, capacity-limited travel nodes).

Usage:
    python3 dev/gen.py --out dev/cases --seed 1 --per-level 30
"""

import argparse
import heapq
import json
import math
import os
import random


def ceil_w(w):
    return int(math.ceil(w))


def dijkstra_all(n, edges):
    adj = {i: [] for i in range(n)}
    for e in edges:
        w = ceil_w(e["weight"])
        adj[e["from_node"]].append((e["to_node"], w))
        if e.get("bidirectional", True):
            adj[e["to_node"]].append((e["from_node"], w))
    dist = {}
    for s in range(n):
        d = {s: 0}
        pq = [(0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > d.get(u, 1e18):
                continue
            for v, w in adj[u]:
                nd = du + w
                if nd < d.get(v, 1e18):
                    d[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist[s] = d
    return dist


def strongly_connected(n, edges):
    dist = dijkstra_all(n, edges)
    return all(len(dist[s]) == n for s in range(n))


# ---------------------------------------------------------------- topologies

def topo_grid(rng, rows, cols):
    n = rows * cols
    edges = []
    for r in range(rows):
        for c in range(cols):
            u = r * cols + c
            if c + 1 < cols:
                edges.append((u, u + 1))
            if r + 1 < rows:
                edges.append((u, u + cols))
    # Remove a few edges while staying connected
    rng.shuffle(edges)
    kept = list(edges)
    for e in edges:
        if rng.random() < 0.15:
            trial = [x for x in kept if x != e]
            if strongly_connected(n, [{"from_node": a, "to_node": b, "weight": 1} for a, b in trial]):
                kept = trial
    return n, kept


def topo_tree_plus(rng, n, extra):
    edges = []
    for v in range(1, n):
        u = rng.randrange(0, v)
        edges.append((u, v))
    existing = set(edges) | set((b, a) for a, b in edges)
    tries = 0
    while extra > 0 and tries < 200:
        tries += 1
        a, b = rng.sample(range(n), 2)
        if (a, b) not in existing:
            edges.append((a, b))
            existing.add((a, b))
            existing.add((b, a))
            extra -= 1
    return n, edges


def topo_ladder(rng, length):
    # Two parallel corridors joined by rungs - classic narrow-aisle layout
    n = 2 * length
    edges = []
    for i in range(length - 1):
        edges.append((i, i + 1))
        edges.append((length + i, length + i + 1))
    for i in range(length):
        if i == 0 or i == length - 1 or rng.random() < 0.5:
            edges.append((i, length + i))
    return n, edges


def topo_hub(rng, spokes):
    # Parking spurs around a hub, a trunk to stations (like test_case_5)
    n = spokes + 3
    hub = 0
    edges = [(i, hub) for i in range(1, spokes + 1)]
    approach = spokes + 1
    edges.append((hub, approach))
    edges.append((approach, spokes + 2))
    return n, edges


def build_case(rng, level, stress=False, robust=False):
    # --- floor
    if stress:
        kind = rng.choice(["grid", "tree", "ladder"])
    else:
        kind = rng.choice(["grid", "tree", "tree", "ladder", "hub"]) if level > 1 else rng.choice(["grid", "tree", "tree", "ladder"])

    if kind == "grid":
        if stress:
            rows, cols = rng.choice([(4, 5), (5, 5), (4, 6)])
        else:
            rows, cols = rng.choice([(2, 3), (2, 4), (3, 3), (3, 4)] if level > 1 else [(2, 2), (2, 3), (2, 4), (3, 3)])
        n, pairs = topo_grid(rng, rows, cols)
    elif kind == "tree":
        n = rng.randint(14, 24) if stress else rng.randint(4, 9) if level == 1 else rng.randint(6, 11)
        n, pairs = topo_tree_plus(rng, n, rng.randint(1, max(1, n // 3)))
    elif kind == "ladder":
        n, pairs = topo_ladder(rng, rng.randint(6, 10) if stress else rng.randint(3, 5))
    else:
        n, pairs = topo_hub(rng, rng.randint(3, 5))

    ids = list(range(n))

    # --- node roles
    if kind == "hub":
        stations = [n - 1]
        storages = [0] if rng.random() < 0.6 else [0, n - 2]
        if len(storages) == 2:
            # keep the approach node as travel; add a second station on a spur
            pass
    else:
        n_st = 1 if level == 1 and rng.random() < 0.5 else rng.randint(1, 3 if stress else 2)
        n_sto = 1 if level == 1 else rng.randint(1, 4 if stress else (3 if level == 3 else 2))
        n_st = min(n_st, max(1, n // 3))
        n_sto = min(n_sto, max(1, n // 3))
        # stations and storage at different places; prefer low-degree nodes
        deg = {i: 0 for i in ids}
        for a, b in pairs:
            deg[a] += 1
            deg[b] += 1
        order = sorted(ids, key=lambda i: (deg[i] + rng.random() * 2))
        pool = order[: max(n_st + n_sto + 1, n // 2 + 1)]
        rng.shuffle(pool)
        stations = pool[:n_st]
        storages = pool[n_st:n_st + n_sto]
        if not storages:
            storages = [x for x in ids if x not in stations][:1]

    # --- edges
    edges = []
    for a, b in pairs:
        w = rng.choice([1, 2, 2, 2, 3, 3, 4, 5, 6]) if not stress else rng.randint(1, 5)
        if robust and rng.random() < 0.2:
            w = round(rng.uniform(1.0, 5.0), 1)
        e = {"from_node": a, "to_node": b, "weight": w}
        if level == 2:
            if rng.random() < rng.choice([0.4, 0.7, 1.0]):
                e["capacity"] = 1
        elif level == 3 or stress:
            if rng.random() < 0.25:
                e["capacity"] = rng.choice([1, 1, 2])
        edges.append(e)

    if robust:
        # make some edges one-way while keeping strong connectivity
        order = list(range(len(edges)))
        rng.shuffle(order)
        for i in order[: max(1, len(edges) // 4)]:
            edges[i]["bidirectional"] = False
            if rng.random() < 0.5:
                edges[i]["from_node"], edges[i]["to_node"] = edges[i]["to_node"], edges[i]["from_node"]
            if not strongly_connected(n, edges):
                edges[i]["bidirectional"] = True

    # --- nodes
    nodes = []
    for i in ids:
        t = "station" if i in stations else "storage" if i in storages else "travel"
        nd = {"id": i, "name": "%s-%d" % (t.capitalize(), i), "type": t}
        if t == "station" and (level == 3 or stress):
            nd["capacity"] = rng.choice([1, 1, 2, 2, 3])
        if robust and t == "travel" and rng.random() < 0.2:
            nd["capacity"] = rng.choice([1, 2])
        nodes.append(nd)

    # --- drive units
    if level == 1:
        n_units = 1
    elif stress:
        n_units = rng.randint(4, 7)
    else:
        n_units = rng.randint(2, 4)
    cap_of = {nd["id"]: nd.get("capacity") for nd in nodes}
    start_pool = [i for i in ids if i not in stations]
    rng.shuffle(start_pool)
    starts = []
    for i in start_pool:
        if len(starts) >= n_units:
            break
        starts.append(i)
    while len(starts) < n_units:  # tiny graph: allow sharing uncapped nodes
        cands = [i for i in ids if cap_of[i] is None and i not in stations]
        starts.append(rng.choice(cands))
    units = []
    for k, s in enumerate(starts):
        u = {"id": k, "start_node": s}
        if level == 3 or stress:
            u["capacity"] = rng.choice([1, 2, 2, 3]) if level == 3 else rng.choice([1, 2])
        units.append(u)

    # --- pods
    if stress:
        n_pods = rng.randint(15, 30)
        horizon = rng.randint(40, 120)
    elif level == 1:
        n_pods = rng.randint(1, 4)
        horizon = rng.choice([0, 20, 40])
    elif level == 2:
        n_pods = rng.randint(2, 8)
        horizon = rng.choice([0, 8, 15, 25])
    else:
        n_pods = rng.randint(4, 10)
        horizon = rng.choice([10, 20, 30])
    pods = []
    for p in range(n_pods):
        if horizon == 0:
            t = 0
        else:
            t = rng.choice([0, 0] + [rng.randint(0, horizon)] * 3)
            if rng.random() < 0.4:
                t = (t // 5) * 5
        pods.append({
            "id": "P%d" % (p + 1),
            "source_node": rng.choice(storages),
            "destination_station": rng.choice(stations),
            "entry_time": t,
        })
    pods.sort(key=lambda x: (x["entry_time"], x["id"]))

    # --- time budget, generous like the examples but occasionally tight
    dist = dijkstra_all(n, edges)
    work = sum(dist[p["source_node"]][p["destination_station"]] * 2 for p in pods)
    last = max(p["entry_time"] for p in pods)
    base = last + work / max(1, n_units) * rng.choice([0.8, 1.2, 1.5, 2.0]) + 30
    max_steps = int(max(40, min(400 if stress else 250, 10 * round(base / 10))))

    return {
        "metadata": {
            "max_time_steps": max_steps,
            "description": "generated level %d (%s)%s%s" % (level, kind, " stress" if stress else "", " robust" if robust else ""),
        },
        "nodes": nodes,
        "edges": edges,
        "drive_units": units,
        "pods": pods,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--per-level", type=int, default=30)
    ap.add_argument("--stress", type=int, default=0)
    ap.add_argument("--robust", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    def dump(sub, idx, case):
        d = os.path.join(args.out, sub)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "case_%03d.json" % idx), "w") as f:
            json.dump(case, f, indent=1)

    for level in (1, 2, 3):
        for i in range(args.per_level):
            dump("level%d" % level, i, build_case(rng, level))
    for i in range(args.stress):
        dump("stress", i, build_case(rng, 3, stress=True))
    for i in range(args.robust):
        dump("robust", i, build_case(rng, rng.choice([1, 2, 3]), robust=True))


if __name__ == "__main__":
    main()
