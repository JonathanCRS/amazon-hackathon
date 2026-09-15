"""
Amazon Robotics Hackathon 2026 - "anticipate" routing strategy.

Once per time step (on the first poll of the step) a global plan is built:
  1. Task layer: greedy insertion of storage-node pickups into per-unit trip
     blocks, maximising the sum of exp(-duration/50) with exact modelling of
     involuntary pickups (a visit grabs the oldest waiting pods up to the free
     capacity) and best-permutation delivery ordering.
  2. Anticipation: idle units are spread over storage nodes (weighted by the
     spawn statistics learned online) and wait there, so new pods are picked
     up the step they appear.
  3. Routing: prioritised space-time A* against a reservation table that
     models the engine's capacity rules exactly (including the ascending-id
     polling order), so planned moves are valid when executed.
  4. Stuck detection raises priorities and eventually makes units back off.
Everything is wrapped in try/except with a shortest-path fallback.
"""

import heapq
import math
import time
import itertools
from typing import Optional

INF = 10 ** 9

F_FULL = 0   # counts for everybody
F_HI = 1     # counts only for units with a higher id (departed this step)
F_LO = 2     # counts only for units with a lower id (leaves this step)

SOFT_BUDGET = 0.06     # seconds per step: after this, remaining units get cheap plans
HARD_BUDGET = 0.25
GAME_BUDGET = 40.0     # seconds per game before switching to cheap mode
MAX_EXPAND = 6000

G = {}
import os as _os
_DEBUG = bool(_os.environ.get('ANTICIPATE_DEBUG'))


def _reset(sig):
    G.clear()
    G["sig"] = sig
    G["last_t"] = -1
    G["last_uid"] = -1
    G["plan_t"] = -1
    G["moves"] = {}
    G["seen"] = set()
    G["spawn_cnt"] = {}
    G["prev_first"] = {}
    G["prev_park"] = {}
    G["stuck"] = {}
    G["last_pos"] = {}
    G["game_time"] = 0.0
    G["static"] = None


def _signature(state):
    return (len(state.nodes), len(state.edges),
            tuple(n.id for n in state.nodes[:50]),
            tuple((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional) for e in state.edges[:80]),
            tuple(sorted(u.id for u in state.drive_units)))


# --------------------------------------------------------------------------- static

def _build_static(state):
    S = {}
    ids = [n.id for n in state.nodes]
    idx = {}
    for i in ids:
        if i not in idx:
            idx[i] = len(idx)
    # nodes referenced only by edges / units
    for e in state.edges:
        for x in (e.from_node, e.to_node):
            if x not in idx:
                idx[x] = len(idx)
    for u in state.drive_units:
        if u.current_node not in idx:
            idx[u.current_node] = len(idx)
    n = len(idx)
    rid = [None] * n
    for k, v in idx.items():
        rid[v] = k
    ncap = [None] * n
    ntype = ["travel"] * n
    seen_node = set()
    for nd in state.nodes:
        if nd.id in seen_node:
            continue
        seen_node.add(nd.id)
        ncap[idx[nd.id]] = nd.capacity
        ntype[idx[nd.id]] = nd.node_type
    # first edge connecting u->v defines the move (engine uses get_edge)
    first = {}
    for e in state.edges:
        a, b = idx[e.from_node], idx[e.to_node]
        if (a, b) not in first:
            first[(a, b)] = e
        if e.bidirectional and (b, a) not in first:
            first[(b, a)] = e
    out = [[] for _ in range(n)]
    pinfo = {}
    for (a, b), e in first.items():
        if a == b:
            continue
        try:
            w = float(e.weight)
        except Exception:
            w = 1.0
        W = max(1, int(math.ceil(w - 1e-9)))
        blockers = [(a, b)]
        if e.bidirectional:
            ea, eb = idx[e.from_node], idx[e.to_node]
            blockers = [(ea, eb), (eb, ea)]
        pinfo[(a, b)] = (W, e.capacity, blockers)
        out[a].append((b, W))
    inn = [[] for _ in range(n)]
    for a in range(n):
        for b, W in out[a]:
            inn[b].append((a, W))
    # all pairs shortest path D[a][b]
    D = []
    for s in range(n):
        d = [INF] * n
        d[s] = 0
        pq = [(0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > d[u]:
                continue
            for v, W in out[u]:
                nd_ = du + W
                if nd_ < d[v]:
                    d[v] = nd_
                    heapq.heappush(pq, (nd_, v))
        D.append(d)
    S.update(n=n, idx=idx, rid=rid, ncap=ncap, ntype=ntype, out=out, inn=inn,
             pinfo=pinfo, D=D)
    S["storages"] = [i for i in range(n) if ntype[i] == "storage"]
    S["stations"] = [i for i in range(n) if ntype[i] == "station"]
    S["uncap"] = [i for i in range(n) if ncap[i] is None]
    # parking spot for each storage node: itself if uncapacitated, else the
    # closest uncapacitated node (round trip distance)
    park = {}
    for s in S["storages"]:
        if ncap[s] is None:
            park[s] = s
        else:
            best, bd = s, INF
            for x in S["uncap"]:
                dd = D[x][s] + D[s][x]
                if dd < bd:
                    bd, best = dd, x
            park[s] = best
    S["park"] = park
    return S


# --------------------------------------------------------------------------- reservations

class Res(object):
    __slots__ = ("node", "edge", "perm")

    def __init__(self, n):
        self.node = [dict() for _ in range(n)]
        self.edge = {}
        self.perm = [[] for _ in range(n)]

    def add_node(self, v, s, uid, flag):
        d = self.node[v]
        l = d.get(s)
        if l is None:
            d[s] = [(uid, flag)]
        else:
            l.append((uid, flag))

    def add_edge(self, p, s, uid, flag):
        d = self.edge.get(p)
        if d is None:
            d = self.edge[p] = {}
        l = d.get(s)
        if l is None:
            d[s] = [(uid, flag)]
        else:
            l.append((uid, flag))

    def remove_uid_node(self, v, s, uid):
        l = self.node[v].get(s)
        if l:
            self.node[v][s] = [x for x in l if x[0] != uid]

    @staticmethod
    def _cnt(l, uid, fme):
        c = 0
        for x, f in l:
            if x == uid:
                continue
            if f == F_FULL:
                c += 1
            elif f == F_HI:
                if fme != F_LO or x < uid:
                    c += 1
            else:  # other leaves this step
                if fme != F_HI or x > uid:
                    c += 1
        return c

    def node_count(self, v, s, uid, fme=F_FULL):
        c = 0
        l = self.node[v].get(s)
        if l:
            c = self._cnt(l, uid, fme)
        for T, x in self.perm[v]:
            if T <= s and x != uid:
                c += 1
        return c

    def edge_count(self, blockers, s, uid, fme=F_FULL):
        c = 0
        for q in blockers:
            d = self.edge.get(q)
            if d:
                l = d.get(s)
                if l:
                    c += self._cnt(l, uid, fme)
        return c

    def node_free_forever(self, v, T, cap, uid):
        if cap is None:
            return True
        base = 0
        for T2, x in self.perm[v]:
            if x != uid:
                base += 1
        if base >= cap:
            return False
        for s, l in self.node[v].items():
            if s >= T:
                if self.node_count(v, s, uid) >= cap:
                    return False
        return True


# --------------------------------------------------------------------------- task layer

def _eval_block(S, pos, t, load, pickups):
    """load: list of (dest, entry); pickups: list of (node, [(dest, entry, pid)]).
    Returns (value, end_t, end_pos, stops) or None if infeasible."""
    D = S["D"]
    stops = []
    cur = list(load)
    for s, pods in pickups:
        d = D[pos][s]
        if d >= INF:
            return None
        t += d
        pos = s
        stops.append(s)
        for p in pods:
            cur.append((p[0], p[1]))
    dests = []
    for dst, ent in cur:
        if dst is not None and dst not in dests:
            dests.append(dst)
    if not dests:
        return (0.0, t, pos, stops)
    best = None
    if len(dests) <= 4:
        perms = itertools.permutations(dests)
    else:
        # nearest neighbour order
        order, p0, rem = [], pos, list(dests)
        while rem:
            nx = min(rem, key=lambda x: D[p0][x])
            order.append(nx)
            rem.remove(nx)
            p0 = nx
        perms = [tuple(order)]
    for perm in perms:
        tt, pp, val, ok = t, pos, 0.0, True
        arr = {}
        for st in perm:
            d = D[pp][st]
            if d >= INF:
                ok = False
                break
            tt += d
            pp = st
            arr[st] = tt
        if not ok:
            continue
        for dst, ent in cur:
            if dst is None:
                continue
            dt = max(0, arr[dst] - 1 - ent)
            val += math.exp(-dt / 50.0)
        if best is None or val > best[0] + 1e-12:
            best = (val, tt, pp, list(perm))
    if best is None:
        return None
    return (best[0], best[1], best[2], stops + best[3])


def _assign(S, units, queues, now):
    """units: list of dicts with pos, t, cap, carry(list of (dest, entry)).
    queues: storage idx -> list of pods (dest, entry, pid) in pickup order.
    Returns per-uid plan dict: blocks list."""
    D = S["D"]
    plans = {}
    for u in units:
        blocks = []
        if u["carry"]:
            ev = _eval_block(S, u["pos"], u["t"], u["carry"], [])
            if ev is not None:
                blocks.append({"pos": u["pos"], "t": u["t"], "load": list(u["carry"]),
                               "pick": [], "ev": ev})
        plans[u["uid"]] = blocks
    q = dict((s, list(l)) for s, l in queues.items() if l)
    prev_first = G.get("prev_first", {})
    iters = 0
    while q and iters < 200:
        iters += 1
        best = None
        for u in units:
            if u["cap"] <= 0:
                continue
            uid = u["uid"]
            blocks = plans[uid]
            last = blocks[-1] if blocks else None
            if last is not None:
                tail_pos, tail_t = last["ev"][2], last["ev"][1]
                used = len(last["load"]) + sum(len(p[1]) for p in last["pick"])
                free = u["cap"] - used
            else:
                tail_pos, tail_t = u["pos"], u["t"]
                free = 0
            for s, pods in q.items():
                if D[tail_pos][s] >= INF and (last is None or D[last["pos"]][s] >= INF):
                    continue
                hyst = 0.0
                if prev_first.get(uid) == s and len(blocks) <= 1:
                    hyst = 0.03
                # (a) extend last block
                if last is not None and free > 0:
                    k = min(free, len(pods))
                    newpick = last["pick"] + [(s, pods[:k])]
                    ev = _eval_block(S, last["pos"], last["t"], last["load"], newpick)
                    if ev is not None:
                        delta = ev[0] - last["ev"][0] + (hyst if len(blocks) == 1 else 0.0)
                        if best is None or delta > best[0]:
                            best = (delta, uid, s, k, "ext", ev, None)
                # (b) new block
                k = min(u["cap"], len(pods))
                ev = _eval_block(S, tail_pos, tail_t, [], [(s, pods[:k])])
                if ev is not None:
                    delta = ev[0] + (hyst if not blocks else 0.0)
                    if best is None or delta > best[0]:
                        best = (delta, uid, s, k, "new", ev, (tail_pos, tail_t))
        if best is None:
            break
        delta, uid, s, k, kind, ev, tail = best
        pods = q[s][:k]
        if kind == "ext":
            last = plans[uid][-1]
            last["pick"] = last["pick"] + [(s, pods)]
            last["ev"] = ev
        else:
            plans[uid].append({"pos": tail[0], "t": tail[1], "load": [], "pick": [(s, pods)], "ev": ev})
        q[s] = q[s][k:]
        if not q[s]:
            del q[s]
    return plans


# --------------------------------------------------------------------------- A*

def _astar(S, R, uid, start, t0, legs, free_cap, avoid, now, deadline):
    """legs: list of (goalset(frozenset), park(bool)).
    Returns list of (node, t) states or None."""
    out, pinfo, ncap, D = S["out"], S["pinfo"], S["ncap"], S["D"]
    nl = len(legs)
    # heuristic tables
    hs = []
    for gset, park in legs:
        h = [INF] * S["n"]
        for v in range(S["n"]):
            m = INF
            for g in gset:
                if D[v][g] < m:
                    m = D[v][g]
            h[v] = m
        hs.append(h)
    rest = [0] * (nl + 1)
    for k in range(nl - 2, -1, -1):
        m = INF
        for a in legs[k][0]:
            for b in legs[k + 1][0]:
                if D[a][b] < m:
                    m = D[a][b]
        rest[k] = rest[k + 1] + (m if m < INF else 0)

    def advance(v, t, leg):
        while leg < nl and v in legs[leg][0]:
            if legs[leg][1]:
                if not R.node_free_forever(v, t, ncap[v], uid):
                    break
            leg += 1
        return leg

    leg0 = advance(start, t0, 0)
    if leg0 >= nl:
        return [(start, t0)]
    h0 = hs[leg0][start]
    if h0 >= INF:
        return None
    horizon = t0 + int(h0 + rest[leg0]) * 3 + 40
    cnt = 0
    startkey = (start, t0, leg0)
    parent = {startkey: None}
    gbest = {startkey: 0.0}
    pq = [(h0 + rest[leg0], 0.0, cnt, startkey)]
    expand = 0
    while pq:
        f, g, _, key = heapq.heappop(pq)
        if gbest.get(key, INF) < g - 1e-9:
            continue
        v, t, leg = key
        if leg >= nl:
            path = []
            k = key
            while k is not None:
                path.append((k[0], k[1]))
                k = parent[k]
            path.reverse()
            return path
        expand += 1
        if expand > MAX_EXPAND or (expand & 255) == 0 and time.perf_counter() > deadline:
            return None
        if t >= horizon:
            continue
        # wait
        cap = ncap[v]
        if cap is None or R.node_count(v, t + 1, uid) < cap:
            nk = (v, t + 1, leg)
            ng = g + 1.0
            nleg = advance(v, t + 1, leg)
            if nleg != leg:
                nk = (v, t + 1, nleg)
            if ng < gbest.get(nk, INF):
                gbest[nk] = ng
                parent[nk] = key
                cnt += 1
                hh = hs[nleg][v] + rest[nleg] if nleg < nl else 0
                heapq.heappush(pq, (ng + hh, ng, cnt, nk))
        for w, W in out[v]:
            hw = hs[leg][w]
            if hw >= INF and not (w in legs[leg][0]):
                continue
            _, ecap, blockers = pinfo[(v, w)]
            ok = True
            if ecap is not None:
                for s in range(t, t + W):
                    if R.edge_count(blockers, s, uid, F_HI if s == t else F_FULL) >= ecap:
                        ok = False
                        break
            if not ok:
                continue
            wcap = ncap[w]
            if wcap is not None:
                for s in range(t, t + W + 1):
                    if R.node_count(w, s, uid, F_HI if s == t else F_FULL) >= wcap:
                        ok = False
                        break
            if not ok:
                continue
            ng = g + W + 0.01
            if free_cap > 0 and w in avoid and not (w in legs[leg][0]):
                ng += avoid[w]
            nleg = advance(w, t + W, leg)
            nk = (w, t + W, nleg)
            if ng < gbest.get(nk, INF):
                gbest[nk] = ng
                parent[nk] = key
                cnt += 1
                hh = hs[nleg][w] + rest[nleg] if nleg < nl else 0
                heapq.heappush(pq, (ng + hh, ng, cnt, nk))
    return None


def _record_path(S, R, uid, path, park_end, transit_until=None):
    pinfo = S["pinfo"]
    for i in range(len(path) - 1):
        v, t = path[i]
        w, t2 = path[i + 1]
        if w == v:
            R.add_node(v, t, uid, F_FULL)
            continue
        R.add_node(v, t, uid, F_LO)
        W, ecap, blockers = pinfo[(v, w)]
        R.add_edge((v, w), t, uid, F_HI)
        R.add_node(w, t, uid, F_HI)
        for s in range(t + 1, t2):
            R.add_edge((v, w), s, uid, F_FULL)
            R.add_node(w, s, uid, F_FULL)
    v, t = path[-1]
    if park_end:
        R.perm[v].append((t, uid))
    else:
        R.add_node(v, t, uid, F_FULL)
        R.add_node(v, t + 1, uid, F_FULL)


# --------------------------------------------------------------------------- main planning

def _plan(state, S):
    t_start = time.perf_counter()
    now = state.current_time_step
    D, idx, ncap = S["D"], S["idx"], S["ncap"]
    n = S["n"]
    # --- pods
    pods_by_id = {}
    for p in state.active_pods:
        pods_by_id[p.id] = p
        if p.id not in G["seen"]:
            G["seen"].add(p.id)
            if p.carried_by is None and p.current_node in idx:
                s = idx[p.current_node]
                G["spawn_cnt"][s] = G["spawn_cnt"].get(s, 0) + 1
    queues = {}
    waiting_at = {}
    for p in state.active_pods:
        if p.carried_by is None and p.current_node in idx:
            s = idx[p.current_node]
            dst = idx.get(p.destination_station)
            if dst is not None and D[s][dst] >= INF:
                dst = None
            queues.setdefault(s, []).append((p.entry_time, p.id, dst))
    for s in queues:
        queues[s].sort()
        queues[s] = [(dst, ent, pid) for ent, pid, dst in queues[s]]
        waiting_at[s] = [x[2] for x in queues[s]]

    # --- units
    units = []
    ustate = {}
    for u in sorted(state.drive_units, key=lambda x: x.id):
        if u.current_node not in idx:
            continue
        if u.in_transit and u.transit_destination in idx:
            rem = u.transit_remaining_time
            try:
                k = max(1, int(math.ceil(rem - 1e-9)))
            except Exception:
                k = 1
            pos = idx[u.transit_destination]
            t0 = now + k
            origin = idx[u.current_node]
        else:
            pos = idx[u.current_node]
            t0 = now
            origin = None
            k = 0
        carry = []
        dead = 0
        for pid in u.carrying:
            p = pods_by_id.get(pid)
            if p is None:
                continue
            dst = idx.get(p.destination_station)
            if dst is None or D[pos][dst] >= INF:
                dead += 1
                continue
            carry.append((dst, p.entry_time))
        cap = u.capacity - len(u.carrying)
        rec = {"uid": u.id, "pos": pos, "t": t0, "cap": u.capacity - dead, "carry": carry,
               "free": cap, "transit": u.in_transit, "origin": origin, "k": k}
        units.append(rec)
        ustate[u.id] = rec

    plans = _assign(S, units, queues, now)
    pod_owner = {}
    for uid, blocks in plans.items():
        for b in blocks:
            for s, pods in b["pick"]:
                for p in pods:
                    pod_owner[p[2]] = uid
    G["prev_first"] = {}
    for uid, blocks in plans.items():
        if blocks and blocks[0]["pick"]:
            G["prev_first"][uid] = blocks[0]["pick"][0][0]

    # --- idle positioning (anticipation)
    storages = S["storages"]
    park_goal = {}
    idle = [u for u in units if not plans[u["uid"]]]
    if storages and idle:
        weights = {}
        tot = 0
        for s in storages:
            weights[s] = 1.0 + G["spawn_cnt"].get(s, 0)
            tot += weights[s]
        for s in storages:
            weights[s] /= tot
        nassigned = dict((s, 0) for s in storages)
        # busy units ending their plan near a storage count a bit
        remaining = list(idle)
        prev_park = G.get("prev_park", {})
        while remaining:
            best = None
            for u in remaining:
                for s in storages:
                    pk = S["park"][s]
                    d = D[u["pos"]][pk]
                    if d >= INF:
                        continue
                    val = weights[s] * (0.5 ** nassigned[s]) - 0.004 * d
                    if prev_park.get(u["uid"]) == s:
                        val += 0.01
                    if best is None or val > best[0]:
                        best = (val, u, s)
            if best is None:
                break
            _, u, s = best
            park_goal[u["uid"]] = s
            nassigned[s] += 1
            remaining.remove(u)
        G["prev_park"] = dict(park_goal)

    # --- legs
    uncap_set = frozenset(S["uncap"]) if S["uncap"] else frozenset(range(n))
    legs_of = {}
    for u in units:
        uid = u["uid"]
        blocks = plans[uid]
        legs = []
        if blocks:
            stops = blocks[0]["ev"][3]
            prev = None
            for st in stops:
                if st == prev:
                    continue
                legs.append((frozenset([st]), False))
                prev = st
                if len(legs) >= 2:
                    break
            if legs:
                lastnode = next(iter(legs[-1][0]))
                if ncap[lastnode] is not None:
                    legs.append((uncap_set, True))
        elif uid in park_goal:
            legs.append((frozenset([S["park"][park_goal[uid]]]), True))
        else:
            if ncap[u["pos"]] is not None:
                legs.append((uncap_set, True))
        legs_of[uid] = legs

    # --- priorities
    stuck = G["stuck"]

    def prio(u):
        uid = u["uid"]
        onc = 0 if (not u["transit"] and ncap[u["pos"]] is not None) else 1
        st = stuck.get(uid, 0)
        boost = 0 if st >= 4 else 1
        if u["carry"]:
            cls = 0
            key2 = min(e for d, e in u["carry"])
        elif plans[uid]:
            cls = 1
            key2 = -plans[uid][0]["ev"][0]
        else:
            cls = 2
            key2 = 0
        return (boost, onc, cls, key2, uid)

    order = sorted(units, key=prio)

    # --- reservations
    R = Res(n)
    for u in units:
        uid = u["uid"]
        if u["transit"]:
            o, w = u["origin"], u["pos"]
            p = (o, w)
            if p in S["pinfo"]:
                for s in range(now, u["t"]):
                    R.add_edge(p, s, uid, F_FULL)
            for s in range(now, u["t"]):
                R.add_node(w, s, uid, F_FULL)
        R.add_node(u["pos"], u["t"], uid, F_FULL)

    avoid_base = {}
    for s, pids in waiting_at.items():
        avoid_base[s] = pids

    moves = {}
    deadline_soft = t_start + SOFT_BUDGET
    deadline_hard = t_start + HARD_BUDGET
    cheap = G["game_time"] > GAME_BUDGET
    for u in order:
        uid = u["uid"]
        R.remove_uid_node(u["pos"], u["t"], uid)
        legs = legs_of[uid]
        path = None
        if legs:
            avoid = {}
            for s, pids in avoid_base.items():
                pen = 0.0
                for pid in pids:
                    ow = pod_owner.get(pid)
                    if ow is None:
                        pen = max(pen, 0.5)
                    elif ow != uid:
                        pen = max(pen, 4.0)
                if pen > 0:
                    avoid[s] = pen
            if not cheap and time.perf_counter() < deadline_soft:
                path = _astar(S, R, uid, u["pos"], u["t"], legs, u["free"], avoid, now,
                              deadline_hard)
            if path is None:
                path = _simple_path(S, R, uid, u, legs)
            park_end = legs[-1][1] and path[-1][0] in legs[-1][0]
        else:
            path = [(u["pos"], u["t"])]
            park_end = True
        if len(path) >= 2 and path[1][0] != path[0][0] and path[1][1] < u["t"]:
            path = [(u["pos"], u["t"])]
        _record_path(S, R, uid, path, park_end)
        if _DEBUG:
            G.setdefault('dbg', []).append((now, uid, [(tuple(S['rid'][x] for x in l[0])[:3], l[1]) for l in legs], [(S['rid'][a], b) for a, b in path]))
        if not u["transit"]:
            if len(path) >= 2 and path[1][0] != path[0][0] and path[0][1] == now:
                moves[uid] = S["rid"][path[1][0]]
            else:
                moves[uid] = None
    return moves


def _simple_path(S, R, uid, u, legs):
    """Cheap fallback: one shortest-path hop toward the first leg if currently free."""
    D, out, pinfo, ncap = S["D"], S["out"], S["pinfo"], S["ncap"]
    v, t = u["pos"], u["t"]
    gset = legs[0][0]
    if v in gset:
        return [(v, t)]
    best = None
    for w, W in out[v]:
        dg = min(D[w][g] for g in gset)
        if dg >= INF:
            continue
        c = W + dg
        if best is None or c < best[0]:
            best = (c, w, W)
    if best is None:
        return [(v, t)]
    _, w, W = best
    _, ecap, blockers = pinfo[(v, w)]
    if ecap is not None and R.edge_count(blockers, t, uid, F_HI) >= ecap:
        return [(v, t)]
    if ncap[w] is not None and R.node_count(w, t, uid, F_HI) >= ncap[w]:
        return [(v, t)]
    return [(v, t), (w, t + W)]


# --------------------------------------------------------------------------- validity / fallback

def _valid_now(state, unit, nxt):
    if nxt is None or not isinstance(nxt, int) or isinstance(nxt, bool):
        return False
    if unit.in_transit or nxt == unit.current_node:
        return False
    edge = None
    for e in state.edges:
        if e.connects(unit.current_node, nxt):
            edge = e
            break
    if edge is None:
        return False
    if edge.capacity is not None:
        c = 0
        for x in state.drive_units:
            if x.in_transit and edge.connects(x.current_node, x.transit_destination):
                c += 1
        if c >= edge.capacity:
            return False
    cap = None
    for nd in state.nodes:
        if nd.id == nxt:
            cap = nd.capacity
            break
    if cap is not None:
        c = 0
        for x in state.drive_units:
            if x.in_transit:
                if x.transit_destination == nxt:
                    c += 1
            elif x.current_node == nxt:
                c += 1
        if c >= cap:
            return False
    return True


def _fallback(uid, state):
    unit = None
    for u in state.drive_units:
        if u.id == uid:
            unit = u
    if unit is None or unit.in_transit:
        return None
    S = G.get("static")
    if S is None:
        S = _build_static(state)
    idx, D = S["idx"], S["D"]
    here = idx.get(unit.current_node)
    if here is None:
        return None
    targets = []
    pods = dict((p.id, p) for p in state.active_pods)
    if unit.carrying:
        for pid in unit.carrying:
            p = pods.get(pid)
            if p is not None and p.destination_station in idx:
                targets.append(idx[p.destination_station])
    else:
        for p in state.active_pods:
            if p.carried_by is None and p.current_node in idx:
                targets.append(idx[p.current_node])
    if not targets:
        return None
    g = min(targets, key=lambda x: D[here][x])
    if D[here][g] >= INF or g == here:
        return None
    for w, W in S["out"][here]:
        if W + D[w][g] == D[here][g]:
            r = S["rid"][w]
            if _valid_now(state, unit, r):
                return r
    return None


def _main(uid, state):
    now = state.current_time_step
    sig = G.get("sig")
    newsig = _signature(state)
    if sig != newsig or now < G.get("last_t", -1) or (now == G.get("last_t", -1) and uid <= G.get("last_uid", -1)):
        _reset(newsig)
    G["last_t"] = now
    G["last_uid"] = uid
    if G["static"] is None:
        G["static"] = _build_static(state)
    S = G["static"]
    if G["plan_t"] != now:
        t0 = time.perf_counter()
        # stuck bookkeeping
        for u in state.drive_units:
            if u.in_transit:
                G["stuck"][u.id] = 0
                continue
            lp = G["last_pos"].get(u.id)
            want = G["moves"].get(u.id)
            if lp == u.current_node and want is not None:
                G["stuck"][u.id] = G["stuck"].get(u.id, 0) + 1
            elif lp != u.current_node:
                G["stuck"][u.id] = 0
            G["last_pos"][u.id] = u.current_node
        try:
            G["moves"] = _plan(state, S)
        except Exception:
            if _DEBUG:
                raise
            G["moves"] = {}
            G["plan_failed"] = True
        G["plan_t"] = now
        G["game_time"] += time.perf_counter() - t0
    if uid not in G["moves"]:
        return _fallback(uid, state)
    mv = G["moves"].get(uid)
    if mv is None:
        return None
    unit = None
    for u in state.drive_units:
        if u.id == uid:
            unit = u
            break
    if unit is None:
        return None
    if _valid_now(state, unit, mv):
        return mv
    return None


def drive_unit_next_move(drive_unit_id, state):
    try:
        return _main(drive_unit_id, state)
    except Exception:
        if _DEBUG:
            raise
        try:
            return _fallback(drive_unit_id, state)
        except Exception:
            return None
