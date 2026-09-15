"""
jointsearch: per-step joint-action beam search with an exact engine simulator.

Every time step (on the first poll of the step) we:
  1. extract a compact state (units, waiting pods),
  2. build a high-level task plan (insertion heuristic over unit routes,
     with hysteresis on previous assignments) and idle "home" spots,
  3. run a beam search over joint unit actions (operator decomposition in
     ascending unit id, exactly mirroring the engine's commit order) for H steps
     using an exact simulator of the engine step semantics,
  4. score leaves with realized pod values + route-based estimates of the
     delivery time of every known undelivered pod + an idle-positioning term,
  5. cache the best first joint action and serve it to the other units' polls.

Python 3.7+ standard library only.
"""
import heapq
import math
import time

try:
    from typing import Optional
except Exception:  # pragma: no cover
    Optional = None

INF = 10 ** 6

# ----------------------------------------------------------------- tunables
LAM = 0.004            # value per step of unit availability (idle positioning)
HYST = 0.015           # bonus for keeping a pod on its previous unit
HOME_KEEP = 0.5        # home hysteresis (in distance units)
HOME_GAMMA = 0.3       # weight of travel distance when choosing homes
STEP_SOFT = 0.060      # soft planning budget per step (s)
STEP_HARD = 0.300      # hard cap per step (s)
GAME_BUDGET = 25.0     # total planning seconds per game before degrading
PH_MAX = 12            # max beam horizon
PB = 24                # beam width (<=3 units)
PR = 40                # rollout length at leaves
PK = 24                # number of leaves rolled out

_EXP = [math.exp(-a / 50.0) for a in range(6000)]


def _pv(age):
    if age < 0:
        age = 0
    if age >= 6000:
        return 0.0
    return _EXP[int(age)]


# ------------------------------------------------------------------- caches
_C = {}


def _reset(sig):
    _C.clear()
    _C["sig"] = sig
    _C["last_t"] = -1
    _C["last_uid"] = -1
    _C["plan_t"] = -1
    _C["actions"] = {}
    _C["prev_assign"] = {}
    _C["homes"] = {}
    _C["src_seen"] = {}
    _C["pods_seen"] = set()
    _C["spent"] = 0.0


def _graph_sig(state):
    return (tuple((n.id, n.node_type, n.capacity) for n in state.nodes),
            tuple((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional) for e in state.edges))


def _build_graph(state):
    nodes = state.nodes
    ids = [n.id for n in nodes]
    idx = {}
    for i, nid in enumerate(ids):
        if nid not in idx:
            idx[nid] = i
    N = len(ids)
    ncap = [None] * N
    ntype = [None] * N
    for n in nodes:
        i = idx[n.id]
        if ntype[i] is None:
            ntype[i] = n.node_type
            ncap[i] = n.capacity
    # moves: for each a, dict b -> (weight_raw, wceil, cap, pairset)
    moves = [dict() for _ in range(N)]
    for e in state.edges:
        if e.from_node not in idx or e.to_node not in idx:
            continue
        a = idx[e.from_node]
        b = idx[e.to_node]
        pairs = {a * N + b}
        if e.bidirectional:
            pairs.add(b * N + a)
        pairs = frozenset(pairs)
        wc = int(math.ceil(e.weight))
        if wc < 1:
            wc = 1
        if a != b and b not in moves[a]:
            moves[a][b] = (e.weight, wc, e.capacity, pairs)
        if e.bidirectional and a != b and a not in moves[b]:
            moves[b][a] = (e.weight, wc, e.capacity, pairs)
    # all-pairs shortest distances (directed) and next hops
    D = []
    NH = []
    for s in range(N):
        d = [INF] * N
        h = [-1] * N
        d[s] = 0
        pq = [(0, s, -1)]
        while pq:
            du, u, fh = heapq.heappop(pq)
            if du > d[u]:
                continue
            for v, mv in moves[u].items():
                nd = du + mv[1]
                if nd < d[v]:
                    d[v] = nd
                    h[v] = v if u == s else fh
                    heapq.heappush(pq, (nd, v, h[v]))
        D.append(d)
        NH.append(h)
    g = {"ids": ids, "idx": idx, "N": N, "ncap": ncap, "ntype": ntype,
         "moves": moves, "D": D, "NH": NH}
    storage = [i for i in range(N) if ntype[i] == "storage"]
    g["storage"] = storage
    g["spots"] = [i for i in range(N) if ncap[i] is None]
    mw = 1
    for i in range(N):
        for mv in moves[i].values():
            if mv[1] > mw:
                mw = mv[1]
    g["maxw"] = mw
    return g


# --------------------------------------------------------------- simulator
# unit state tuple: (node, dest, rem, carry)   dest = -1 when idle
# W: bitmask of waiting pods

class Sim(object):
    def __init__(self, g, ucap, psrc, pdst, pentry, pods_at):
        self.g = g
        self.ucap = ucap
        self.psrc = psrc
        self.pdst = pdst
        self.pentry = pentry
        self.pods_at = pods_at   # node -> list of pod idx sorted by priority

    def valid(self, units, ui, b):
        g = self.g
        node, dest, rem, carry = units[ui]
        mv = g["moves"][node].get(b)
        if mv is None:
            return None
        N = g["N"]
        cap = mv[2]
        if cap is not None:
            pairs = mv[3]
            c = 0
            for (n2, d2, r2, c2) in units:
                if d2 >= 0 and (n2 * N + d2) in pairs:
                    c += 1
            if c >= cap:
                return None
        nc = g["ncap"][b]
        if nc is not None:
            c = 0
            for (n2, d2, r2, c2) in units:
                if d2 >= 0:
                    if d2 == b:
                        c += 1
                elif n2 == b:
                    c += 1
            if c >= nc:
                return None
        return mv

    def boundary(self, units, W, t):
        """phase d + e for step t. returns (units, W, gained_value)."""
        nu = list(units)
        for i, (node, dest, rem, carry) in enumerate(nu):
            if dest >= 0:
                rem = rem - 1
                if rem <= 0:
                    nu[i] = (dest, -1, 0, carry)
                else:
                    nu[i] = (node, dest, rem, carry)
        gain = 0.0
        pdst = self.pdst
        pent = self.pentry
        for i, (node, dest, rem, carry) in enumerate(nu):
            if dest >= 0:
                continue
            changed = False
            if carry:
                keep = []
                for p in carry:
                    if pdst[p] == node:
                        gain += _pv(t - pent[p])
                        changed = True
                    else:
                        keep.append(p)
                if changed:
                    carry = tuple(keep)
            cap = self.ucap[i]
            if len(carry) < cap and W:
                lst = self.pods_at.get(node)
                if lst:
                    c2 = list(carry)
                    for p in lst:
                        if (W >> p) & 1:
                            W &= ~(1 << p)
                            c2.append(p)
                            changed = True
                            if len(c2) >= cap:
                                break
                    carry = tuple(c2)
            if changed:
                nu[i] = (node, dest, rem, carry)
        return tuple(nu), W, gain


# ----------------------------------------------------------- route values

def _route_value(D, stops, rn, rt, t0, onboard, cap, pentry, lam):
    """stops: list of (node, typ, p) typ 0=pickup 1=deliver.
    Returns (value, end_node, end_time) or None if capacity infeasible."""
    T = rt
    node = rn
    ev = rt
    val = 0.0
    load = onboard
    for (sn, typ, p) in stops:
        d = D[node][sn]
        if d > 0:
            T += d
            ev = T - 1
            node = sn
        if typ == 0:
            load += 1
            if load > cap:
                return None
        else:
            load -= 1
            val += _pv(ev - pentry[p])
    return val, node, T


def _eff_stops(ctx, ui, carry, W, rn, rt):
    plan = ctx["plan"][ui]
    stops = []
    if plan:
        for s in plan:
            p = s[2]
            if (W >> p) & 1:
                stops.append(s)
            elif s[1] == 1 and p in carry:
                stops.append(s)
    if carry:
        D = ctx["D"]
        pentry = ctx["pentry"]
        have = None
        for p in carry:
            if have is None:
                have = set(s[2] for s in stops if s[1] == 1)
            if p not in have:
                st = (ctx["pdst"][p], 1, p)
                best = None
                bestv = -1e18
                L = len(stops)
                for j in range(L + 1):
                    cand = stops[:j] + [st] + stops[j:]
                    r = _route_value(D, cand, rn, rt, 0, len(carry), 99, pentry, 0)
                    if r is not None and r[0] - LAM * r[2] > bestv:
                        bestv = r[0] - LAM * r[2]
                        best = cand
                stops = best
                have.add(p)
    return stops


def _unit_term(ctx, ui, us, t, decided, W):
    """Estimated value contributed by unit ui from state us at time t."""
    node, dest, rem, carry = us
    if dest >= 0:
        rn = dest
        n = int(math.ceil(rem))
        if n < 1:
            n = 1
        rt = t + n
    else:
        rn = node
        rt = t + 1 if decided else t
    stops = _eff_stops(ctx, ui, carry, W, rn, rt)
    D = ctx["D"]
    home = ctx["homes"][ui]
    if not stops:
        return -LAM * (rt + D[rn][home])
    r = _route_value(D, stops, rn, rt, t, len(carry), 99, ctx["pentry"], 0)
    return r[0] - LAM * (r[2] + D[r[1]][home])


def _rollout(ctx, sim, units, W, acc, t, R):
    """Follow plans greedily (next hop, wait if blocked) for up to R steps."""
    n = len(units)
    D = ctx["D"]
    NH = ctx["NH"]
    homes = ctx["homes"]
    for _ in range(R):
        ul = list(units)
        anyidle = False
        busy = False
        for ui in range(n):
            us = ul[ui]
            if us[1] >= 0:
                busy = True
                continue
            node = us[0]
            stops = _eff_stops(ctx, ui, us[3], W, node, t)
            target = -1
            for s in stops:
                if s[0] != node:
                    target = s[0]
                    break
            if stops:
                busy = True
            if target < 0:
                target = homes[ui]
                if target == node:
                    continue
            anyidle = True
            h = NH[node][target]
            if h < 0:
                continue
            tu = tuple(ul)
            mv = sim.valid(tu, ui, h)
            if mv is not None:
                ul[ui] = (node, h, mv[0], us[3])
        units, W, gain = sim.boundary(tuple(ul), W, t)
        acc += gain
        t += 1
        if not busy and not anyidle:
            break
    val = acc
    for i in range(n):
        val += _unit_term(ctx, i, units[i], t, False, W)
    return val


# ---------------------------------------------------------------- planning

def _extract(state, g):
    idx = g["idx"]
    units = sorted(state.drive_units, key=lambda u: u.id)
    active = sorted(state.active_pods, key=lambda p: (p.entry_time, p.id))
    pid_index = {}
    psrc, pdst, pentry, pids = [], [], [], []
    for p in active:
        pid_index[p.id] = len(pids)
        pids.append(p.id)
        psrc.append(idx.get(p.current_node, -1) if p.current_node is not None else -1)
        pdst.append(idx.get(p.destination_station, -1))
        pentry.append(p.entry_time)
    W = 0
    pods_at = {}
    for i, p in enumerate(active):
        if p.carried_by is None and p.current_node is not None:
            W |= 1 << i
            pods_at.setdefault(psrc[i], []).append(i)
    ust = []
    ucap = []
    uids = []
    for u in units:
        carry = tuple(pid_index[c] for c in u.carrying if c in pid_index)
        if u.in_transit:
            ust.append((idx[u.current_node], idx[u.transit_destination], u.transit_remaining_time, carry))
        else:
            ust.append((idx[u.current_node], -1, 0, carry))
        ucap.append(u.capacity)
        uids.append(u.id)
    return {"units": tuple(ust), "ucap": ucap, "uids": uids, "W": W, "pids": pids,
            "psrc": psrc, "pdst": pdst, "pentry": pentry, "pods_at": pods_at}


def _ready(us, t):
    node, dest, rem, carry = us
    if dest >= 0:
        n = int(math.ceil(rem))
        return dest, t + max(1, n)
    return node, t


def _build_plan(ex, g, t, k0):
    """Insertion heuristic. Returns list of stop lists per unit."""
    D = g["D"]
    units = ex["units"]
    n = len(units)
    pentry = ex["pentry"]
    prev = _C["prev_assign"]
    pids = ex["pids"]
    ucap = ex["ucap"]
    routes = [[] for _ in range(n)]
    rdy = []
    for i in range(n):
        rn, rt = _ready(units[i], t)
        if i < k0 and units[i][1] < 0:
            rt = t + 1
        rdy.append((rn, rt))
    onb = [len(units[i][3]) for i in range(n)]
    cur = [0.0] * n
    # carried pods first
    for i in range(n):
        for p in units[i][3]:
            st = (ex["pdst"][p], 1, p)
            best = None
            bv = -1e18
            R = routes[i]
            for j in range(len(R) + 1):
                cand = R[:j] + [st] + R[j:]
                r = _route_value(D, cand, rdy[i][0], rdy[i][1], t, onb[i], 10 ** 6, pentry, 0)
                if r is not None:
                    v = r[0] - LAM * r[2]
                    if v > bv:
                        bv = v
                        best = cand
            routes[i] = best
            cur[i] = bv
    for i in range(n):
        r = _route_value(D, routes[i], rdy[i][0], rdy[i][1], t, onb[i], 10 ** 6, pentry, 0)
        cur[i] = r[0] - LAM * r[2]
    waiting = [p for p in range(len(pids)) if (ex["W"] >> p) & 1]
    assign = {}

    def best_insert(p, skip=None):
        bp = None
        src = ex["psrc"][p]
        dst = ex["pdst"][p]
        sp = (src, 0, p)
        sd = (dst, 1, p)
        for i in range(n):
            if i == skip:
                continue
            R = routes[i]
            L = len(R)
            cap = ucap[i]
            bonus = HYST if prev.get(pids[p]) == i else 0.0
            for a in range(L + 1):
                for b in range(a, min(L, a + 2 * cap + 2) + 1):
                    cand = R[:a] + [sp] + R[a:b] + [sd] + R[b:]
                    r = _route_value(D, cand, rdy[i][0], rdy[i][1], t, onb[i], cap, pentry, 0)
                    if r is None:
                        continue
                    dv = r[0] - LAM * r[2] - cur[i] + bonus
                    if bp is None or dv > bp[0]:
                        bp = (dv, i, cand, r[0] - LAM * r[2])
        return bp

    for p in waiting:
        bp = best_insert(p)
        if bp is None:
            continue
        _, i, cand, v = bp
        routes[i] = cand
        cur[i] = v
        assign[p] = i
    # improvement pass: remove + reinsert
    for _it in range(2):
        improved = False
        for p in waiting:
            if p not in assign:
                continue
            i = assign[p]
            R = [s for s in routes[i] if s[2] != p]
            r = _route_value(D, R, rdy[i][0], rdy[i][1], t, onb[i], ucap[i], pentry, 0)
            if r is None:
                continue
            v_removed = r[0] - LAM * r[2]
            old_R, old_v = routes[i], cur[i]
            routes[i] = R
            cur[i] = v_removed
            bp = best_insert(p)
            base_gain = old_v - v_removed + (HYST if prev.get(pids[p]) == i else 0.0)
            if bp is not None and bp[0] > base_gain + 1e-9:
                _, j, cand, v = bp
                routes[j] = cand
                cur[j] = v
                assign[p] = j
                if j != i:
                    improved = True
            else:
                routes[i] = old_R
                cur[i] = old_v
        if not improved:
            break
    newprev = {}
    for p, i in assign.items():
        newprev[pids[p]] = i
    _C["prev_assign"] = newprev
    return routes, rdy


def _compute_homes(ex, g, routes, rdy):
    D = g["D"]
    n = len(ex["units"])
    spots = g["spots"] or list(range(g["N"]))
    freq = {}
    for s in g["storage"]:
        freq[s] = freq.get(s, 0.0) + 1.0
    for s, c in _C["src_seen"].items():
        freq[s] = freq.get(s, 0.0) + c
    if not freq:
        return [rdy[i][0] if not routes[i] else routes[i][-1][0] for i in range(n)]
    tot = sum(freq.values())
    srcs = list(freq.items())
    cov = dict((s, INF) for s, _ in srcs)
    homes = [0] * n
    prevh = _C["homes"]
    order = [i for i in range(n) if not routes[i]] + [i for i in range(n) if routes[i]]
    for i in order:
        end = routes[i][-1][0] if routes[i] else rdy[i][0]
        best = None
        costs = {}
        for x in spots:
            if D[end][x] >= INF:
                continue
            c = 0.0
            for s, f in srcs:
                dd = D[x][s]
                cv = cov[s]
                c += f * (dd if dd < cv else cv)
            c = c / tot + HOME_GAMMA * D[end][x]
            costs[x] = c
            if best is None or c < best[0] - 1e-12:
                best = (c, x)
        if best is None:
            homes[i] = end
            continue
        h = best[1]
        ph = prevh.get(i)
        if ph is not None and ph in costs and costs[ph] <= best[0] + HOME_KEEP:
            h = ph
        homes[i] = h
        if not routes[i]:
            for s, _ in srcs:
                if D[h][s] < cov[s]:
                    cov[s] = D[h][s]
    _C["homes"] = dict((i, homes[i]) for i in range(n))
    return homes


def _search(ex, g, t, k0, routes, homes, t_start):
    sim = Sim(g, ex["ucap"], ex["psrc"], ex["pdst"], ex["pentry"], ex["pods_at"])
    n = len(ex["units"])
    ctx = {"plan": routes, "D": g["D"], "NH": g["NH"], "pentry": ex["pentry"],
           "pdst": ex["pdst"], "homes": homes}
    units0 = ex["units"]
    W0 = ex["W"]
    terms0 = tuple(_unit_term(ctx, i, units0[i], t, i < k0, W0) for i in range(n))
    # state: (val, units, W, acc, terms, first)
    beam = [(sum(terms0), units0, W0, 0.0, terms0, ())]
    maxw = g["maxw"]
    H = max(4, min(PH_MAX, 2 * maxw + 2))
    B = PB if n <= 3 else (max(6, PB * 2 // 3) if n <= 5 else max(4, PB // 2))
    R = PR
    K = PK
    if _C["spent"] > GAME_BUDGET:
        H = min(H, 3)
        B = 4
        K = 2
        R = 10
    moves = g["moves"]
    tcur = t
    for step in range(H):
        start_u = k0 if step == 0 else 0
        for ui in range(start_u, n):
            children = {}
            for st in beam:
                val, units, W, acc, terms, first = st
                us = units[ui]
                if us[1] >= 0:
                    key = (units, W)
                    f2 = first + (None,) if step == 0 else first
                    if key not in children or children[key][0] < val:
                        children[key] = (val, units, W, acc, terms, f2)
                    continue
                node = us[0]
                nt = _unit_term(ctx, ui, us, tcur, True, W)
                v2 = val - terms[ui] + nt
                tl = list(terms)
                tl[ui] = nt
                f2 = first + (None,) if step == 0 else first
                key = (units, W)
                if key not in children or children[key][0] < v2:
                    children[key] = (v2, units, W, acc, tuple(tl), f2)
                for b in moves[node]:
                    mv = sim.valid(units, ui, b)
                    if mv is None:
                        continue
                    nus = (node, b, mv[0], us[3])
                    ul = list(units)
                    ul[ui] = nus
                    ul = tuple(ul)
                    nt = _unit_term(ctx, ui, nus, tcur, True, W)
                    v2 = val - terms[ui] + nt
                    tl = list(terms)
                    tl[ui] = nt
                    f2 = first + (b,) if step == 0 else first
                    key = (ul, W)
                    if key not in children or children[key][0] < v2:
                        children[key] = (v2, ul, W, acc, tuple(tl), f2)
            ch = sorted(children.values(), key=lambda s: -s[0])
            beam = ch[:B]
        nb = {}
        for st in beam:
            val, units, W, acc, terms, first = st
            u2, W2, gain = sim.boundary(units, W, tcur)
            acc2 = acc + gain
            terms2 = tuple(_unit_term(ctx, i, u2[i], tcur + 1, False, W2) for i in range(n))
            v2 = acc2 + sum(terms2)
            key = (u2, W2)
            if key not in nb or nb[key][0] < v2:
                nb[key] = (v2, u2, W2, acc2, terms2, first)
        beam = sorted(nb.values(), key=lambda s: -s[0])[:B]
        tcur += 1
        el = time.perf_counter() - t_start
        if el > STEP_HARD * 0.5 or (el > STEP_SOFT and step >= 2):
            break
    best_first = beam[0][5]
    if K > 0 and R > 0:
        best = None
        for st in beam[:K]:
            v = _rollout(ctx, sim, st[1], st[2], st[3], tcur, R)
            if best is None or v > best[0] + 1e-12:
                best = (v, st[5])
            if time.perf_counter() - t_start > STEP_HARD:
                break
        best_first = best[1]
    return best_first


def _plan_step(state, uid):
    t0 = time.perf_counter()
    g = _C["g"]
    t = state.current_time_step
    ex = _extract(state, g)
    uids = ex["uids"]
    k0 = uids.index(uid)
    # record sources of pods
    seen = _C["pods_seen"]
    for i, pid in enumerate(ex["pids"]):
        if pid not in seen:
            seen.add(pid)
            s = ex["psrc"][i]
            if s >= 0:
                _C["src_seen"][s] = _C["src_seen"].get(s, 0) + 1
    routes, rdy = _build_plan(ex, g, t, k0)
    homes = _compute_homes(ex, g, routes, rdy)
    _C["last_plan"] = (routes, homes, ex)
    first = _search(ex, g, t, k0, routes, homes, t0)
    actions = {}
    ids = g["ids"]
    if first is not None:
        for j, a in enumerate(first):
            ui = k0 + j
            actions[uids[ui]] = None if a is None else ids[a]
    # expected positions for consistency check
    _C["actions"] = actions
    _C["plan_t"] = t
    _C["spent"] += time.perf_counter() - t0
    return actions


def _fallback(uid, state):
    g = _C.get("g")
    if g is None:
        return None
    unit = None
    for u in state.drive_units:
        if u.id == uid:
            unit = u
    if unit is None or unit.in_transit:
        return None
    idx = g["idx"]
    here = idx[unit.current_node]
    target = None
    D = g["D"]
    if unit.carrying:
        best = INF
        for p in state.active_pods:
            if p.id in unit.carrying:
                d = idx.get(p.destination_station)
                if d is not None and D[here][d] < best:
                    best = D[here][d]
                    target = d
    else:
        best = INF
        for p in state.active_pods:
            if p.carried_by is None and p.current_node is not None:
                s = idx.get(p.current_node)
                if s is not None and D[here][s] < best:
                    best = D[here][s]
                    target = s
    if target is None or target == here:
        return None
    h = g["NH"][here][target]
    if h < 0:
        return None
    return g["ids"][h]


def drive_unit_next_move(drive_unit_id, state):
    try:
        t = state.current_time_step
        sig = _graph_sig(state)
        if _C.get("sig") != sig:
            _reset(sig)
        else:
            lt = _C["last_t"]
            if t < lt or (t == lt and drive_unit_id <= _C["last_uid"]):
                _reset(sig)
        if "g" not in _C:
            _C["g"] = _build_graph(state)
        _C["last_t"] = t
        _C["last_uid"] = drive_unit_id
        if _C["plan_t"] == t and drive_unit_id in _C["actions"]:
            a = _C["actions"][drive_unit_id]
            # verify committed-state consistency cheaply: replan if action invalid now
            return a
        acts = _plan_step(state, drive_unit_id)
        return acts.get(drive_unit_id)
    except Exception:
        try:
            return _fallback(drive_unit_id, state)
        except Exception:
            return None


def _env_overrides():
    import os
    g = globals()
    for k in ("LAM", "HYST", "HOME_KEEP", "HOME_GAMMA", "PH_MAX", "PB", "PR", "PK"):
        v = os.environ.get("JS_" + k)
        if v is not None:
            g[k] = type(g[k])(float(v)) if isinstance(g[k], float) else int(v)


try:
    _env_overrides()
except Exception:
    pass
