"""Reference baseline: Dijkstra next-hop + greedy nearest-pod claiming. No traffic awareness."""
import heapq
import math
from typing import Optional

_cache = {"sig": None, "dist": None, "nxt": None, "claims": {}}


def _prepare(state):
    sig = (tuple((e.from_node, e.to_node, e.weight, e.bidirectional) for e in state.edges),
           tuple(n.id for n in state.nodes))
    if _cache["sig"] == sig and state.current_time_step >= _cache.get("t", 0):
        _cache["t"] = state.current_time_step
        return
    adj = {n.id: [] for n in state.nodes}
    for e in state.edges:
        w = int(math.ceil(e.weight))
        adj.setdefault(e.from_node, []).append((e.to_node, w))
        if e.bidirectional:
            adj.setdefault(e.to_node, []).append((e.from_node, w))
    dist, nxt = {}, {}
    for t in adj:  # reverse dijkstra from every target
        radj = {}
        for u in adj:
            for v, w in adj[u]:
                radj.setdefault(v, []).append((u, w))
        d = {t: 0}
        hop = {}
        pq = [(0, t)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > d[u]:
                continue
            for v, w in radj.get(u, []):
                if du + w < d.get(v, 1e18):
                    d[v] = du + w
                    hop[v] = u
                    heapq.heappush(pq, (du + w, v))
        dist[t], nxt[t] = d, hop
    _cache.update(sig=sig, dist=dist, nxt=nxt, claims={}, t=state.current_time_step)


def drive_unit_next_move(drive_unit_id: int, state) -> Optional[int]:
    _prepare(state)
    dist, nxt, claims = _cache["dist"], _cache["nxt"], _cache["claims"]
    unit = state.get_drive_unit(drive_unit_id)
    here = unit.current_node
    if unit.carrying:
        dests = [state.get_pod(p).destination_station for p in unit.carrying]
        target = min(dests, key=lambda s: dist[s].get(here, 1e18))
    else:
        live = {p.id for p in state.active_pods if p.carried_by is None}
        for k in [k for k, v in claims.items() if v not in live]:
            del claims[k]
        mine = claims.get(drive_unit_id)
        if mine is None:
            taken = set(claims.values())
            best = None
            for p in state.active_pods:
                if p.carried_by is None and p.id not in taken:
                    d = dist[p.current_node].get(here, 1e18)
                    if best is None or d < best[0]:
                        best = (d, p)
            if best is None:
                return None
            mine = claims[drive_unit_id] = best[1].id
        target = state.get_pod(mine).current_node
    if target == here:
        return None
    return nxt[target].get(here)
