"""
bnb: task-sequence search + reservation-based (space-time) path planning,
validated by an exact mini-simulation of the engine.

Python 3.7+, standard library only.
"""
import heapq
import math
import random
import time
from typing import Optional

INF = float("inf")
BIG = 10 ** 9

# ----------------------------------------------------------------- budgets
SOFT_PLAN_BUDGET = 0.080      # seconds per replan (search)
HARD_PLAN_BUDGET = 0.250      # absolute cap for one replan
GAME_BUDGET = 25.0            # total planning seconds per game before cheap mode

_G = {}   # per-game cache
import os as _os
_DEBUG = bool(_os.environ.get('BNB_DEBUG'))


def _exp_score(dt):
    return math.exp(-dt / 50.0)


# =================================================================== static
class Static(object):
    pass


def _signature(state):
    return (tuple((n.id, n.node_type, n.capacity) for n in state.nodes),
            tuple((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional) for e in state.edges),
            tuple(sorted((u.id, u.capacity) for u in state.drive_units)))


def _build_static(state):
    S = Static()
    S.node_ids = [n.id for n in state.nodes]
    S.idx = dict((nid, i) for i, nid in enumerate(S.node_ids))
    N = len(S.node_ids)
    S.N = N
    S.ntype = [n.node_type for n in state.nodes]
    S.ncap = [n.capacity for n in state.nodes]
    S.out = [[] for _ in range(N)]      # (v, w, eid)
    S.ecap = []
    S.eW = []
    for k, e in enumerate(state.edges):
        if e.from_node not in S.idx or e.to_node not in S.idx:
            S.ecap.append(None)
            S.eW.append(1)
            continue
        a, b = S.idx[e.from_node], S.idx[e.to_node]
        w = int(math.ceil(e.weight))
        if w < 1:
            w = 1
        S.ecap.append(e.capacity)
        S.eW.append(w)
        if a != b:
            S.out[a].append((b, w, k))
            if e.bidirectional:
                S.out[b].append((a, w, k))
    # keep only the cheapest parallel edge per (a,b)? keep all (capacities differ)
    D = [[INF] * N for _ in range(N)]
    for s in range(N):
        d = D[s]
        d[s] = 0
        pq = [(0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > d[u]:
                continue
            for v, w, _ in S.out[u]:
                nd = du + w
                if nd < d[v]:
                    d[v] = nd
                    heapq.heappush(pq, (nd, v))
    S.D = D
    S.maxD = max([x for row in D for x in row if x < INF] + [1])
    S.storages = [i for i in range(N) if S.ntype[i] == "storage"]
    S.stations = [i for i in range(N) if S.ntype[i] == "station"]
    S.E = len(state.edges)
    # resource capacity: nodes 0..N-1, edges N..N+E-1
    S.rcap = list(S.ncap) + list(S.ecap)
    S.unit_ids = sorted(u.id for u in state.drive_units)
    S.rank = dict((uid, r) for r, uid in enumerate(S.unit_ids))
    S.M = max(1, len(S.unit_ids))
    S.ucap = dict((u.id, u.capacity) for u in state.drive_units)
    return S


# ================================================================ snapshot
def _read_state(S, state):
    t = state.current_time_step
    units = []
    for u in sorted(state.drive_units, key=lambda x: x.id):
        if u.in_transit:
            rem = u.transit_remaining_time
            arr = t + max(1, int(math.ceil(rem - 1e-9)))
            units.append({"id": u.id, "r": S.rank[u.id], "cap": u.capacity,
                          "node": S.idx[u.current_node], "transit": True,
                          "dest": S.idx[u.transit_destination], "rem": rem, "arr": arr,
                          "carry": list(u.carrying)})
        else:
            units.append({"id": u.id, "r": S.rank[u.id], "cap": u.capacity,
                          "node": S.idx[u.current_node], "transit": False,
                          "dest": None, "rem": 0, "arr": t,
                          "carry": list(u.carrying)})
    pods = {}
    for p in state.active_pods:
        pods[p.id] = {"id": p.id, "src": (S.idx[p.current_node] if p.current_node is not None and p.carried_by is None else None),
                      "dst": S.idx[p.destination_station], "entry": p.entry_time,
                      "by": p.carried_by}
    return t, units, pods


def _unit_key(u):
    return (u["node"], u["transit"], u["dest"], tuple(sorted(u["carry"])))


# ========================================================= estimator / LS
FIFO_PEN = 0.3


def _pkey(p):
    return (p["entry"], p["id"])


def _est_unit(S, u, seq, pods, others=None, bynode=None, want_picks=False):
    """Distance-only estimate. seq: list of (kind, pid) kind 0=pick 1=drop.
    others: node -> list of (moment, rank, pid) picks by other units (FIFO check)."""
    D = S.D
    node = u["dest"] if u["transit"] else u["node"]
    T = u["arr"]
    load = len(u["carry"])
    cap = u["cap"]
    sc = 0.0
    picks = None
    for kind, pid in seq:
        p = pods[pid]
        if kind == 0:
            tgt = p["src"]
            T += D[node][tgt]
            m = T - 1
            if T < p["entry"]:
                T = p["entry"]
            if m < p["entry"]:
                m = p["entry"]
            load += 1
            if load > cap:
                return -1e9, T, None
            if picks is None:
                picks = []
            picks.append((tgt, m, pid))
        else:
            tgt = p["dst"]
            T += D[node][tgt]
            sc += _exp_score(T - 1 - p["entry"])
            load -= 1
        node = tgt
    if picks is not None and bynode is not None:
        r = u["r"]
        viol = 0
        mine = {}
        for (nd, m, pid) in picks:
            mine[pid] = m
        for (nd, m, pid) in picks:
            kp = _pkey(pods[pid])
            lst = bynode.get(nd, ())
            oth = others.get(nd, ()) if others else ()
            for q in lst:
                if _pkey(q) >= kp:
                    break
                if q["entry"] > m:
                    continue
                qid = q["id"]
                if qid in mine:
                    if mine[qid] <= m:
                        continue
                    viol += 1
                    continue
                ok = False
                for (m2, r2, p2) in oth:
                    if p2 == qid:
                        if (m2, r2) < (m, r):
                            ok = True
                        break
                else:
                    ok = True   # unassigned: ignore
                if not ok:
                    viol += 1
            for (m2, r2, p2) in oth:
                if (m2, r2) < (m, r) and _pkey(pods[p2]) > kp and pods[pid]["entry"] <= m2:
                    viol += 1
        sc -= FIFO_PEN * viol
    return sc - 1e-4 * T, T, (picks if want_picks else None)


class Ctx(object):
    pass


def _make_ctx(S, units, pods):
    C = Ctx()
    C.S = S
    C.units = units
    C.pods = pods
    C.ubyid = dict((u["id"], u) for u in units)
    bn = {}
    for p in pods.values():
        if p["src"] is not None:
            bn.setdefault(p["src"], []).append(p)
    for v in bn:
        bn[v].sort(key=_pkey)
    C.bynode = bn
    return C


def _picks_of(C, uid, seq):
    v, T, picks = _est_unit(C.S, C.ubyid[uid], seq, C.pods, None, None, True)
    r = C.ubyid[uid]["r"]
    return [(nd, m, r, pid) for (nd, m, pid) in (picks or [])]


def _others(C, allpicks, uid):
    o = {}
    for u2, pl in allpicks.items():
        if u2 == uid:
            continue
        for (nd, m, r, pid) in pl:
            o.setdefault(nd, []).append((m, r, pid))
    return o


def _uval(C, uid, seq, allpicks):
    return _est_unit(C.S, C.ubyid[uid], seq, C.pods, _others(C, allpicks, uid), C.bynode)[0]


def _best_insert(C, uid, seq, pid, base, others):
    """Best position to insert pick+drop of pid (or only drop if carried)."""
    S = C.S
    u = C.ubyid[uid]
    pods = C.pods
    bn = C.bynode
    best = (-1e18, None)
    L = len(seq)
    carried = pods[pid]["src"] is None
    if carried:
        for j in range(L + 1):
            ns = seq[:j] + [(1, pid)] + seq[j:]
            v = _est_unit(S, u, ns, pods, others, bn)[0]
            if v - base > best[0]:
                best = (v - base, ns)
        return best
    for i in range(L + 1):
        s1 = seq[:i] + [(0, pid)]
        rest = seq[i:]
        for j in range(len(rest) + 1):
            ns = s1 + rest[:j] + [(1, pid)] + rest[j:]
            v = _est_unit(S, u, ns, pods, others, bn)[0]
            if v - base > best[0]:
                best = (v - base, ns)
    return best


def _assign_score(C, assign):
    allpicks = dict((uid, _picks_of(C, uid, sq)) for uid, sq in assign.items())
    tot = 0.0
    for uid, sq in assign.items():
        tot += _uval(C, uid, sq, allpicks)
    return tot


def _local_search(C, assign, movable, deadline, rng=None):
    """assign: dict uid->seq. movable: pod ids that can be re-inserted."""
    pods = C.pods
    allpicks = dict((uid, _picks_of(C, uid, sq)) for uid, sq in assign.items())
    improved = True
    order = list(movable)
    while improved and time.perf_counter() < deadline:
        improved = False
        if rng is not None:
            rng.shuffle(order)
        for pid in order:
            if time.perf_counter() >= deadline:
                break
            owner = None
            for uid, sq in assign.items():
                for k, x in sq:
                    if x == pid:
                        owner = uid
                        break
                if owner is not None:
                    break
            if owner is None:
                continue
            carried = pods[pid]["src"] is None
            reduced = [it_ for it_ in assign[owner] if it_[1] != pid]
            ap2 = dict(allpicks)
            ap2[owner] = _picks_of(C, owner, reduced)
            # baseline total and reduced total
            cur_total = sum(_uval(C, uid, sq, allpicks) for uid, sq in assign.items())
            vals_red = {}
            for uid, sq in assign.items():
                vals_red[uid] = _uval(C, uid, reduced if uid == owner else sq, ap2)
            red_total = sum(vals_red.values())
            best_gain = 1e-9
            best = None
            cands = [owner] if carried else list(assign.keys())
            for uid in cands:
                base_seq = reduced if uid == owner else assign[uid]
                g, ns = _best_insert(C, uid, base_seq, pid, vals_red[uid], _others(C, ap2, uid))
                if ns is None:
                    continue
                gain = red_total + g - cur_total
                if gain > best_gain:
                    best_gain, best = gain, (uid, ns)
            if best is not None:
                uid, ns = best
                trial = dict(assign)
                trial[owner] = reduced
                trial[uid] = ns
                tp = dict(ap2)
                tp[uid] = _picks_of(C, uid, ns)
                tv = sum(_uval(C, u2, sq, tp) for u2, sq in trial.items())
                if tv > cur_total + 1e-9:
                    assign.clear()
                    assign.update(trial)
                    allpicks = tp
                    improved = True
    return assign


def _greedy_insert(C, assign, pids, rng=None):
    allpicks = dict((uid, _picks_of(C, uid, sq)) for uid, sq in assign.items())
    rem = list(pids)
    while rem:
        best = None
        uv = dict((uid, _uval(C, uid, sq, allpicks)) for uid, sq in assign.items())
        for pid in rem:
            opts = []
            for uid in assign:
                g, ns = _best_insert(C, uid, assign[uid], pid, uv[uid], _others(C, allpicks, uid))
                if ns is not None:
                    opts.append((g, uid, ns))
            if not opts:
                continue
            opts.sort(key=lambda x: -x[0])
            regret = opts[0][0] - (opts[1][0] if len(opts) > 1 else -1.0)
            key = regret + opts[0][0]
            if rng is not None:
                key += rng.random() * 0.05
            if best is None or key > best[0]:
                best = (key, pid, opts[0])
        if best is None:
            break
        _, pid, (g, uid, ns) = best
        assign[uid] = ns
        allpicks[uid] = _picks_of(C, uid, ns)
        rem.remove(pid)
    return assign


# ======================================================== reservations
class Res(object):
    def __init__(self, S):
        self.S = S
        self.iv = {}     # rid -> list of [a, b, owner]

    def add(self, rid, a, b, owner):
        if self.S.rcap[rid] is None:
            return
        self.iv.setdefault(rid, []).append((a, b, owner))

    def remove_owner(self, owner):
        for rid in list(self.iv.keys()):
            self.iv[rid] = [x for x in self.iv[rid] if x[2] != owner]

    def blocked(self, rid, a, b):
        c = self.S.rcap[rid]
        if c is None:
            return False
        lst = self.iv.get(rid)
        if not lst:
            return False
        ov = [(x, y) for (x, y, _) in lst if x < b and y > a]
        n = len(ov)
        if n < c:
            return False
        if c <= 0:
            return True
        if c == 1:
            return True
        ev = []
        for x, y in ov:
            ev.append((x if x > a else a, 1))
            ev.append((y, -1))
        ev.sort()
        cur = 0
        for _, d in ev:
            cur += d
            if cur >= c:
                return True
        return False


# ============================================================== A*
def _astar(S, R, r, n0, t0, goal, ready, hazard, tlimit, park=False, max_exp=20000):
    """Earliest (goal, t>=ready) under reservations. Returns list of (dep_t, from, to, arr_t) or None."""
    D = S.D
    M = S.M
    N = S.N
    if D[n0][goal] >= INF:
        return None
    rcap = S.rcap
    blocked = R.blocked
    Dg = [D[v][goal] for v in range(N)]
    start_h = max(Dg[n0], ready - t0, 0)
    heap = [(t0 + start_h, t0, n0)]
    parent = {(n0, t0): None}
    closed = set()
    exp = 0
    while heap:
        f, t, n = heapq.heappop(heap)
        key = (n, t)
        if key in closed:
            continue
        closed.add(key)
        exp += 1
        if exp > max_exp:
            return None
        if n == goal and t >= ready:
            if not park or rcap[n] is None or not blocked(n, t * M + r, BIG * M):
                # reconstruct
                moves = []
                cur = key
                while parent[cur] is not None:
                    prv = parent[cur]
                    if prv[0] != cur[0]:
                        moves.append((prv[1], prv[0], cur[0], cur[1]))
                    cur = prv
                moves.reverse()
                return moves
        if t >= tlimit:
            continue
        # wait
        nk = (n, t + 1)
        if nk not in closed and nk not in parent:
            ok = True
            if rcap[n] is not None and blocked(n, t * M + r, (t + 1) * M + r):
                ok = False
            if ok and hazard is not None and n != goal and hazard(n, t + 1, t + 1):
                ok = False
            if ok:
                parent[nk] = key
                heapq.heappush(heap, (t + 1 + max(Dg[n], ready - t - 1, 0), t + 1, n))
        for v, w, eid in S.out[n]:
            if Dg[v] >= INF:
                continue
            ta = t + w
            if ta > tlimit + S.maxD:
                continue
            nk = (v, ta)
            if nk in parent:
                continue
            a = t * M + r
            if S.ecap[eid] is not None and blocked(N + eid, a, ta * M):
                continue
            if rcap[v] is not None and blocked(v, a, ta * M + r):
                continue
            if hazard is not None and v != goal and hazard(v, ta - 1, ta):
                continue
            parent[nk] = key
            heapq.heappush(heap, (ta + max(Dg[v], ready - ta, 0), ta, v))
    return None


# ============================================================ plan build
def _stops_from_seq(seq, pods):
    """Merge sequence into stops: list of [node, ready, picks(list), drops(list)]."""
    stops = []
    for kind, pid in seq:
        p = pods[pid]
        node = p["src"] if kind == 0 else p["dst"]
        ready = p["entry"] if kind == 0 else 0
        if stops and stops[-1][0] == node:
            stops[-1][1] = max(stops[-1][1], ready)
            (stops[-1][2] if kind == 0 else stops[-1][3]).append(pid)
        else:
            stops.append([node, ready, [pid] if kind == 0 else [], [pid] if kind == 1 else []])
    return stops


def _choose_park(S, u, end_node, end_t, parked_nodes, pods, pick_time, freq):
    """Pick a staging node: near storages (facility location), uncapped, not a station."""
    D = S.D
    N = S.N
    cands = [v for v in range(N) if S.ncap[v] is None and S.ntype[v] != "station" and D[end_node][v] < INF]
    if not cands:
        cands = [v for v in range(N) if S.ntype[v] != "station" and D[end_node][v] < INF]
    if not cands:
        return end_node
    stor = S.storages if S.storages else list(range(N))
    best = None
    for v in cands:
        # avoid storage nodes with pods still waiting (would grab them)
        bad = False
        if u["cap"] > 0:
            for p in pods.values():
                if p["src"] == v and pick_time.get(p["id"], INF) >= end_t:
                    bad = True
                    break
        cost = 0.0
        for s in stor:
            dv = D[v][s]
            for pn in parked_nodes:
                if D[pn][s] < dv:
                    dv = D[pn][s]
            cost += freq.get(s, 1.0) * dv
        cost += 0.05 * D[end_node][v] + (1000.0 if bad else 0.0)
        if best is None or cost < best[0]:
            best = (cost, v)
    return best[1]


def _build_plan(S, t0, units, pods, assign, order, freq, deadline):
    """Prioritized planning. Returns dict uid -> list of moves (dep, from, to, arr), and stats."""
    R = Res(S)
    M = S.M
    N = S.N
    ubyid = dict((u["id"], u) for u in units)
    # pre-register every unit at its node forever (and edges for transit units)
    for u in units:
        if u["transit"]:
            R.add(u["dest"], -BIG, BIG * M, ("pre", u["id"]))
            eid = _find_edge(S, u["node"], u["dest"])
            if eid is not None:
                R.add(N + eid, -BIG, u["arr"] * M, ("fix", u["id"]))
        else:
            R.add(u["node"], -BIG, BIG * M, ("pre", u["id"]))
    pick_time = {}
    plans = {}
    fails = 0
    parked_nodes = []
    horizon_base = S.maxD * 2 + 40
    for uid in order:
        u = ubyid[uid]
        r = u["r"]
        R.remove_owner(("pre", uid))
        seq = assign.get(uid, [])
        stops = _stops_from_seq(seq, pods)
        cur = u["dest"] if u["transit"] else u["node"]
        t = u["arr"]
        load = len(u["carry"])
        cap = u["cap"]
        moves = []
        own = set(pid for _, pid in seq)
        failed = False

        def make_hazard(freecap):
            if freecap <= 0:
                return None

            def hz(v, ta_minus, ta):
                if S.ntype[v] != "storage":
                    return False
                for p in pods.values():
                    if p["src"] == v and p["entry"] <= ta and p["id"] not in own and pick_time.get(p["id"], INF) >= ta_minus:
                        return True
                return False
            return hz

        for st in stops:
            node, ready, picks, drops = st
            hz = make_hazard(cap - load)
            tl = max(t, ready) + horizon_base
            mv = _astar(S, R, r, cur, t, node, ready, hz, tl)
            if mv is None:
                failed = True
                break
            moves.extend(mv)
            if mv:
                t = mv[-1][3]
            if t < ready:
                t = ready
            cur = node
            for pid in picks:
                pick_time[pid] = max(t - 1, pods[pid]["entry"])
            load += len(picks) - len(drops)
        # park
        if not failed:
            pv = _choose_park(S, u, cur, t, parked_nodes, pods, pick_time, freq)
            hz = make_hazard(cap - load)
            mv = _astar(S, R, r, cur, t, pv, t, hz, t + horizon_base, park=True)
            if mv is None and pv != cur:
                mv = _astar(S, R, r, cur, t, cur, t, hz, t + horizon_base, park=True)
                if mv is not None:
                    pv = cur
            if mv is None:
                failed = True
            else:
                moves.extend(mv)
                if mv:
                    t = mv[-1][3]
                cur = pv
                parked_nodes.append(pv)
        if failed:
            fails += 1
        # register intervals
        _register(S, R, u, moves)
        plans[uid] = moves
        if time.perf_counter() > deadline:
            # still register remaining units as waiting forever
            pass
    return plans, fails


def _find_edge(S, a, b):
    for v, w, eid in S.out[a]:
        if v == b:
            return eid
    return None


def _register(S, R, u, moves):
    M = S.M
    N = S.N
    r = u["r"]
    uid = u["id"]
    cur = u["dest"] if u["transit"] else u["node"]
    start = -BIG
    for (dep, a, b, arr) in moves:
        R.add(cur, start, dep * M + r, uid)
        R.add(N + _edge_of(S, a, b, arr - dep), dep * M + r, arr * M, uid)
        cur = b
        start = dep * M + r
    R.add(cur, start, BIG * M, uid)


def _edge_of(S, a, b, w):
    best = None
    for v, ww, eid in S.out[a]:
        if v == b:
            if ww == w:
                return eid
            best = eid
    return best


# ============================================================ exact sim
def _simulate(S, t0, units, pods, plans, horizon):
    """Replay plans through an exact re-implementation of the engine rules.
    Returns (score_sum, delivered dict, snapshots dict step->tuple of unit keys)."""
    M = S.M
    N = S.N
    us = []
    for u in units:
        us.append({"id": u["id"], "cap": u["cap"], "node": u["node"], "transit": u["transit"],
                   "dest": u["dest"], "rem": u["rem"], "eid": (_find_edge(S, u["node"], u["dest"]) if u["transit"] else None),
                   "carry": list(u["carry"]), "moves": plans.get(u["id"], []), "k": 0})
    us.sort(key=lambda x: x["id"])
    waiting = {}   # node -> list of pods sorted by (entry, id)
    pinfo = {}
    for p in pods.values():
        pinfo[p["id"]] = p
        if p["src"] is not None:
            waiting.setdefault(p["src"], []).append(p["id"])
    for v in waiting:
        waiting[v].sort(key=lambda pid: (pinfo[pid]["entry"], pid))
    delivered = {}
    snaps = {}
    deviations = 0

    def dp():
        for x in us:
            if x["transit"]:
                continue
            if x["carry"]:
                keep = []
                for pid in x["carry"]:
                    if pinfo[pid]["dst"] == x["node"]:
                        delivered[pid] = s
                    else:
                        keep.append(pid)
                x["carry"] = keep
            wl = waiting.get(x["node"])
            if wl and len(x["carry"]) < x["cap"]:
                while wl and len(x["carry"]) < x["cap"]:
                    x["carry"].append(wl.pop(0))

    ecap = S.ecap
    ncap = S.ncap
    s = t0
    last = t0
    for x in us:
        if x["moves"]:
            last = max(last, x["moves"][-1][3])
    end = max(last, t0) + 3 + horizon
    npods = len(pinfo)
    while s <= end:
        dp()
        for x in us:
            if x["transit"]:
                continue
            k = x["k"]
            mv = x["moves"]
            if k < len(mv) and mv[k][0] <= s:
                dep, a, b, arr = mv[k]
                if mv[k][0] < s:
                    deviations += 1
                if a != x["node"]:
                    continue
                eid = _edge_of(S, a, b, arr - dep)
                ok = True
                if ecap[eid] is not None:
                    c = 0
                    for y in us:
                        if y["transit"] and y["eid"] == eid:
                            c += 1
                    if c >= ecap[eid]:
                        ok = False
                if ok and ncap[b] is not None:
                    c = 0
                    for y in us:
                        if y["transit"]:
                            if y["dest"] == b:
                                c += 1
                        elif y["node"] == b:
                            c += 1
                    if c >= ncap[b]:
                        ok = False
                if ok:
                    x["transit"] = True
                    x["dest"] = b
                    x["eid"] = eid
                    x["rem"] = S.eW[eid]
                    x["k"] = k + 1
        for x in us:
            if x["transit"]:
                x["rem"] -= 1
                if x["rem"] <= 0:
                    x["node"] = x["dest"]
                    x["transit"] = False
                    x["dest"] = None
                    x["eid"] = None
        dp()
        s += 1
        snaps[s] = tuple((x["node"], x["transit"], x["dest"], tuple(sorted(x["carry"]))) for x in us)
        if len(delivered) == npods and s > last:
            break
    score = 0.0
    for pid, dt in delivered.items():
        score += _exp_score(dt - pinfo[pid]["entry"])
    return score, delivered, snaps, deviations


# ============================================================ planner
def _orderings(S, units, assign, pods, rng, n_extra):
    ids = [u["id"] for u in units]
    ubyid = dict((u["id"], u) for u in units)
    est = {}
    for u in units:
        seq = assign.get(u["id"], [])
        # urgency: earliest estimated delivery minus entry (bigger age first)
        T = u["arr"]
        node = u["dest"] if u["transit"] else u["node"]
        first = INF
        for kind, pid in seq:
            p = pods[pid]
            tgt = p["src"] if kind == 0 else p["dst"]
            T += S.D[node][tgt]
            node = tgt
            if kind == 0 and T < p["entry"]:
                T = p["entry"]
            if kind == 1:
                first = T
                break
        est[u["id"]] = first
    capped = lambda u: (S.ncap[u["dest"] if u["transit"] else u["node"]] is not None)
    outs = []
    outs.append(sorted(ids, key=lambda i: (0 if capped(ubyid[i]) else 1, est[i], i)))
    outs.append(sorted(ids, key=lambda i: (est[i], i)))
    outs.append(sorted(ids))
    outs.append(sorted(ids, key=lambda i: (-len(ubyid[i]["carry"]), est[i], i)))
    outs.append(sorted(ids, key=lambda i: (0 if capped(ubyid[i]) else 1, -len(assign.get(i, [])), i)))
    for _ in range(n_extra):
        o = list(ids)
        rng.shuffle(o)
        outs.append(o)
    res = []
    seen = set()
    for o in outs:
        k = tuple(o)
        if k not in seen:
            seen.add(k)
            res.append(o)
    return res


def _canon(assign):
    return tuple(sorted((uid, tuple(sq)) for uid, sq in assign.items()))


def _replan(S, t, units, pods, g):
    t_start = time.perf_counter()
    cheap = g["plan_time"] > GAME_BUDGET
    soft = t_start + (SOFT_PLAN_BUDGET * (0.25 if cheap else 1.0))
    hard = t_start + HARD_PLAN_BUDGET * (0.5 if cheap else 1.0)
    rng = g["rng"]
    # frequency of storages
    freq = g["freq"]

    carried_pids = []
    base = {}
    for u in units:
        base[u["id"]] = []
    waiting_pids = [pid for pid, p in pods.items() if p["src"] is not None]
    # carried pods: drop items
    for u in units:
        for pid in u["carry"]:
            if pid in pods:
                carried_pids.append(pid)
    # candidate assignments
    cands = {}

    def add_cand(assign):
        sc = _assign_score(S, units, assign, pods)
        cands[_canon(assign)] = (sc, dict((k, list(v)) for k, v in assign.items()))

    # 1) from previous assignment
    prev = g.get("assign")
    a1 = {}
    for u in units:
        a1[u["id"]] = []
    ucarry = dict((u["id"], set(u["carry"])) for u in units)
    placed = set()
    if prev:
        for uid, sq in prev.items():
            if uid not in a1:
                continue
            ns = []
            for kind, pid in sq:
                if pid not in pods:
                    continue
                p = pods[pid]
                if kind == 0:
                    if p["src"] is None:
                        continue
                else:
                    if p["src"] is None and pid not in ucarry[uid]:
                        continue
                ns.append((kind, pid))
            # drop items whose pick is missing for waiting pods -> remove both
            picks = set(pid for k, pid in ns if k == 0)
            ns2 = []
            for kind, pid in ns:
                if kind == 1 and pods[pid]["src"] is not None and pid not in picks:
                    continue
                if kind == 0 and not any(k2 == 1 and p2 == pid for k2, p2 in ns):
                    continue
                ns2.append((kind, pid))
            a1[uid] = ns2
            for _, pid in ns2:
                placed.add(pid)
    # make sure carried pods have drops
    for u in units:
        for pid in u["carry"]:
            if pid not in pods:
                continue
            if not any(k == 1 and p == pid for k, p in a1[u["id"]]):
                g2, ns = _best_insert(S, u, a1[u["id"]], pid, pods, _est_unit(S, u, a1[u["id"]], pods)[0])
                a1[u["id"]] = ns if ns is not None else a1[u["id"]] + [(1, pid)]
            placed.add(pid)
    # validity: capacity feasible?
    for u in units:
        if _est_unit(S, u, a1[u["id"]], pods)[0] < -1e8:
            # rebuild this unit: only carried drops
            removed = [pid for _, pid in a1[u["id"]] if pods[pid]["src"] is not None]
            a1[u["id"]] = [(1, pid) for pid in u["carry"] if pid in pods]
            for pid in removed:
                placed.discard(pid)
    unplaced = [pid for pid in waiting_pids if pid not in placed]
    a1 = _greedy_insert(S, units, a1, pods, sorted(unplaced, key=lambda x: (pods[x]["entry"], x)))
    movable = waiting_pids + carried_pids
    a1 = _local_search(S, units, a1, pods, movable, soft)
    add_cand(a1)
    # 2) fresh greedy
    a2 = dict((u["id"], [(1, pid) for pid in u["carry"] if pid in pods]) for u in units)
    for u in units:
        if len(a2[u["id"]]) > 1:
            # order drops by distance chain via LS later
            pass
    a2 = _greedy_insert(S, units, a2, pods, waiting_pids)
    a2 = _local_search(S, units, a2, pods, movable, soft)
    add_cand(a2)
    # 3) randomized restarts
    tries = 0
    while time.perf_counter() < t_start + (SOFT_PLAN_BUDGET * 0.4 if not cheap else 0.005) and tries < 8 and waiting_pids:
        tries += 1
        a3 = dict((u["id"], [(1, pid) for pid in u["carry"] if pid in pods]) for u in units)
        wp = list(waiting_pids)
        rng.shuffle(wp)
        a3 = _greedy_insert(S, units, a3, pods, wp, rng)
        a3 = _local_search(S, units, a3, pods, movable, soft, rng)
        add_cand(a3)

    ranked = sorted(cands.values(), key=lambda x: -x[0])
    K = 1 if cheap else 4
    ranked = ranked[:K]

    best = None
    evals = 0
    for ci, (esc, assign) in enumerate(ranked):
        n_extra = 0 if cheap else (2 if ci == 0 else 0)
        for order in _orderings(S, units, assign, pods, rng, n_extra):
            if evals > 0 and time.perf_counter() > (soft if evals > 1 else hard):
                break
            plans, fails = _build_plan(S, t, units, pods, assign, order, freq, hard)
            score, delivered, snaps, dev = _simulate(S, t, units, pods, plans, 5)
            evals += 1
            key = (score - 0.001 * fails - 0.0005 * dev)
            if best is None or key > best[0]:
                best = (key, plans, snaps, assign, score, len(delivered))
        if time.perf_counter() > soft and best is not None:
            break
    g["plan_time"] += time.perf_counter() - t_start
    g["n_replans"] += 1
    return best


# ============================================================ fallback
def _fallback(S, uid, state):
    u = None
    for x in state.drive_units:
        if x.id == uid:
            u = x
            break
    if u is None or u.in_transit:
        return None
    here = S.idx[u.current_node]
    pods = dict((p.id, p) for p in state.active_pods)
    targets = []
    if u.carrying:
        for pid in u.carrying:
            if pid in pods:
                targets.append(S.idx[pods[pid].destination_station])
    else:
        for p in state.active_pods:
            if p.carried_by is None and p.current_node is not None:
                targets.append(S.idx[p.current_node])
    if not targets:
        return None
    tgt = min(targets, key=lambda x: S.D[here][x])
    if tgt == here or S.D[here][tgt] >= INF:
        return None
    for v, w, eid in S.out[here]:
        if w + S.D[v][tgt] == S.D[here][tgt]:
            return S.node_ids[v]
    return None


# ============================================================ entry
def _new_game(state, sig):
    S = _build_static(state)
    _G.clear()
    _G["sig"] = sig
    _G["S"] = S
    _G["t"] = -1
    _G["last_uid"] = None
    _G["plans"] = None
    _G["snaps"] = None
    _G["known"] = set()
    _G["assign"] = None
    _G["plan_time"] = 0.0
    _G["n_replans"] = 0
    _G["rng"] = random.Random(12345)
    _G["freq"] = dict((s, 1.0) for s in S.storages)
    _G["moves_now"] = {}
    _G["delivered_n"] = len(state.delivered_pods)


def _step_begin(state):
    g = _G
    S = g["S"]
    t, units, pods = _read_state(S, state)
    # update storage frequency
    for pid, p in pods.items():
        if pid not in g["known"]:
            g["known"].add(pid)
            if p["src"] is not None:
                g["freq"][p["src"]] = g["freq"].get(p["src"], 1.0) + 1.0
    need = False
    snaps = g.get("snaps")
    cur_keys = tuple((u["node"], u["transit"], u["dest"], tuple(sorted(u["carry"]))) for u in units)
    if g.get("plans") is None or snaps is None or t not in snaps:
        need = True
    elif snaps[t] != cur_keys:
        need = True
    elif set(pods.keys()) != g.get("pod_ids_expected", set()) and any(pid not in g.get("pod_ids_seen", set()) for pid in pods):
        need = True
    if need:
        best = _replan(S, t, units, pods, g)
        if best is not None:
            _, plans, snaps2, assign, score, nd = best
            g["plans"] = plans
            g["snaps"] = snaps2
            g["assign"] = assign
            g["plan_t"] = t
            if _DEBUG:
                print("REPLAN t=%d score=%.3f nd=%d assign=%s plans=%s" % (t, score, nd, assign, plans))
    g["pod_ids_seen"] = set(pods.keys()) | g.get("pod_ids_seen", set())
    g["pod_ids_expected"] = set(pods.keys())
    # moves for this step
    mv = {}
    plans = g.get("plans") or {}
    for u in units:
        if u["transit"]:
            continue
        for (dep, a, b, arr) in plans.get(u["id"], []):
            if dep == t and a == u["node"]:
                mv[u["id"]] = S.node_ids[b]
                break
            if dep > t:
                break
    g["moves_now"] = mv


def drive_unit_next_move(drive_unit_id, state):
    try:
        t = state.current_time_step
        g = _G
        sig = None
        new = False
        if not g or t < g.get("t", -1) or (t == g.get("t") and g.get("last_uid") is not None and drive_unit_id <= g["last_uid"]):
            new = True
        elif t != g.get("t"):
            # cheap graph change check occasionally
            if len(state.nodes) != g["S"].N or len(state.edges) != g["S"].E:
                new = True
            elif t == 0 or len(state.delivered_pods) < g.get("delivered_n", 0):
                new = True
        if new:
            sig = _signature(state)
            _new_game(state, sig)
        g = _G
        if t != g["t"]:
            g["t"] = t
            g["last_uid"] = None
            g["delivered_n"] = len(state.delivered_pods)
            _step_begin(state)
        g["last_uid"] = drive_unit_id
        return g["moves_now"].get(drive_unit_id)
    except Exception:
        if _DEBUG:
            raise
        try:
            return _fallback(_G["S"], drive_unit_id, state)
        except Exception:
            return None
