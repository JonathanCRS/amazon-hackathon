"""
whca: event-driven task assignment (rollout over greedy insertion) +
cooperative space-time A* with an exact, slot-ordered reservation table.

Self-contained, Python standard library only (3.7 compatible).
"""

import heapq
import math
import time
from itertools import permutations

try:  # optional type hint import
    from ar_hackathon.models.graph_state import GraphState  # noqa: F401
except Exception:  # pragma: no cover
    GraphState = object

_EXPC = 50.0
_BIG = 10 ** 9

# ----------------------------------------------------------------------------
# module state (reset per game)
# ----------------------------------------------------------------------------
_S = {"sig": None}

# budgets (seconds)
_STEP_SOFT = 0.060
_STEP_HARD = 0.250
_GAME_BUDGET = 40.0


def _steps_for(w):
    """Exact number of engine decrements until arrival for remaining time w."""
    try:
        r = float(w)
    except Exception:
        return 1
    if r <= 0:
        return 1
    k = int(math.ceil(r))
    if k < 1:
        k = 1
    # replicate float decrement semantics exactly
    x = r
    c = 0
    while True:
        x -= 1
        c += 1
        if x <= 0 or c > 100000:
            break
    return c


def _signature(state):
    return (
        tuple((n.id, n.node_type, n.capacity) for n in state.nodes),
        tuple((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional) for e in state.edges),
        tuple(sorted((u.id, u.capacity) for u in state.drive_units)),
    )


def _build_static(state):
    S = {}
    ntype = {}
    ncap = {}
    for n in state.nodes:
        ntype[n.id] = n.node_type
        ncap[n.id] = n.capacity
    first = {}
    for e in state.edges:
        a, b = e.from_node, e.to_node
        if a != b and (a, b) not in first:
            first[(a, b)] = e
        if e.bidirectional and a != b and (b, a) not in first:
            first[(b, a)] = e
    allnodes = set(ntype.keys())
    for (a, b) in first:
        allnodes.add(a)
        allnodes.add(b)
    adj = dict((v, []) for v in allnodes)
    einfo = {}
    for (a, b), e in first.items():
        W = _steps_for(e.weight)
        keys = ((a, b), (b, a)) if e.bidirectional else ((a, b),)
        cap = e.capacity
        adj[a].append((b, W, cap, keys))
        einfo[(a, b)] = (W, cap, keys)
    for v in adj:
        adj[v].sort(key=lambda x: (x[1], x[0]))
    # all pairs shortest (directed)
    dist = {}
    for s in allnodes:
        d = {s: 0}
        pq = [(0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if du > d[u]:
                continue
            for (v, W, _c, _k) in adj[u]:
                nd = du + W
                if nd < d.get(v, _BIG):
                    d[v] = nd
                    heapq.heappush(pq, (nd, v))
        dist[s] = d
    storages = sorted(v for v in ntype if ntype[v] == "storage")
    stations = sorted(v for v in ntype if ntype[v] == "station")
    # parking candidates: storage nodes without capacity limits; else nearest uncapped non-station
    park = [v for v in storages if ncap.get(v) is None]
    if not park:
        cand = [v for v in allnodes if ncap.get(v) is None and ntype.get(v) != "station"]
        for x in storages:
            if cand:
                best = min(cand, key=lambda v: (dist[v].get(x, _BIG) + dist[x].get(v, _BIG), v))
                if best not in park:
                    park.append(best)
    if not park:
        park = [v for v in allnodes if ntype.get(v) != "station"][:1] or sorted(allnodes)[:1]
    S.update(ntype=ntype, ncap=ncap, adj=adj, einfo=einfo, dist=dist, nodes=sorted(allnodes),
             storages=storages, stations=stations, park=park)
    return S


def _new_game(state, sig):
    _S.clear()
    _S["sig"] = sig
    _S["st"] = _build_static(state)
    _S["last_t"] = state.current_time_step
    _S["last_uid"] = -1
    _S["plan_t"] = None
    _S["moves"] = {}
    _S["expect"] = {}
    _S["seen_pods"] = set()
    _S["src_count"] = {}
    _S["stuck"] = {}
    _S["lastpos"] = {}
    _S["game_time"] = 0.0
    _S["prev_target"] = {}
    _S["boost"] = {}


# ----------------------------------------------------------------------------
# reservation table
# ----------------------------------------------------------------------------
class _RT(object):
    __slots__ = ("N", "occ", "perm", "maxs")

    def __init__(self, N):
        self.N = N
        self.occ = {}
        self.perm = {}
        self.maxs = {}

    def add(self, res, s, lo, hi, d=1):
        key = (res, s)
        a = self.occ.get(key)
        if a is None:
            a = [0] * (self.N + 1)
            self.occ[key] = a
        for k in range(lo, hi + 1):
            a[k] += d
        if s > self.maxs.get(res, -1):
            self.maxs[res] = s

    def add_perm(self, res, s, lo):
        self.perm.setdefault(res, []).append((s, lo))

    def free(self, keys, s, lo, hi, cap):
        """True if max over slots lo..hi of summed occupancy <= cap-1."""
        lim = cap - 1
        occ = self.occ
        perm = self.perm
        for k in range(lo, hi + 1):
            c = 0
            for res in keys:
                a = occ.get((res, s))
                if a is not None:
                    c += a[k]
                pl = perm.get(res)
                if pl:
                    for (ps, plo) in pl:
                        if ps < s or (ps == s and plo <= k):
                            c += 1
            if c > lim:
                return False
        return True

    def perm_free(self, res, s, cap):
        lim = cap - 1
        pl = self.perm.get(res, ())
        if len(pl) > lim:
            return False
        mx = self.maxs.get(res, -1)
        N = self.N
        for ss in range(s, mx + 1):
            a = self.occ.get((res, ss))
            if a is None:
                continue
            for k in range(N + 1):
                c = a[k]
                for (ps, plo) in pl:
                    if ps < ss or (ps == ss and plo <= k):
                        c += 1
                if c > lim:
                    return False
        return True


def _hold_actions(rt, st, r, actions, final, sign=1):
    """Insert reservations for a unit of rank r. actions: list of (kind, v, b, s, W)."""
    ncap = st["ncap"]
    einfo = st["einfo"]
    N = rt.N
    for (kind, v, b, s, W) in actions:
        if kind == "w":
            if ncap.get(v) is not None:
                rt.add(v, s, 0, N, sign)
        elif kind == "m":
            if ncap.get(v) is not None:
                rt.add(v, s, 0, r, sign)
            ecap = einfo[(v, b)][1]
            bcap = ncap.get(b)
            if ecap is not None:
                rt.add((v, b), s, r + 1, N, sign)
                for ss in range(s + 1, s + W):
                    rt.add((v, b), ss, 0, N, sign)
            if bcap is not None:
                rt.add(b, s, r + 1, N, sign)
                for ss in range(s + 1, s + W):
                    rt.add(b, ss, 0, N, sign)
        elif kind == "t":  # in transit (already departed), holds edge and dest full [s, s+W)
            ecap = einfo.get((v, b), (0, None))[1]
            bcap = ncap.get(b)
            for ss in range(s, s + W):
                if ecap is not None:
                    rt.add((v, b), ss, 0, N, sign)
                if bcap is not None:
                    rt.add(b, ss, 0, N, sign)
        elif kind == "p":  # pre-hold before own turn
            if ncap.get(v) is not None:
                rt.add(v, s, 0, r, sign)
    if final is not None and sign > 0:
        v, s = final
        if ncap.get(v) is not None:
            rt.add_perm(v, s, 0)


def _astar(rt, st, r, v0, s0, goal, final, pen_nodes, deadline, max_exp=6000, horizon=None):
    """Space-time A*. goal: node id or None (any permanently-free uncapped node).
    Returns (actions, (v, s)) or None."""
    ncap = st["ncap"]
    adj = st["adj"]
    dist = st["dist"]
    N = rt.N
    if goal is not None:
        dg = None
        h0 = dist[v0].get(goal)
        if h0 is None:
            return None
        # reverse distance lookup: dist[x][goal]
        def H(x):
            return dist[x].get(goal, _BIG)
    else:
        h0 = 0

        def H(x):
            return 0
    if horizon is None:
        horizon = 2 * h0 + 60
    tlim = s0 + horizon

    def is_goal(v, s):
        if goal is not None and v != goal:
            return False
        c = ncap.get(v)
        if goal is None:
            if c is not None:
                return rt.perm_free(v, s, c)
            return True
        if final and c is not None:
            return rt.perm_free(v, s, c)
        return True

    start = (v0, s0)
    parent = {start: None}
    gbest = {start: 0}
    pq = [(h0, 0, s0, v0)]
    exp = 0
    while pq:
        f, g, s, v = heapq.heappop(pq)
        key = (v, s)
        if gbest.get(key, _BIG) < g:
            continue
        if is_goal(v, s):
            acts = []
            k = key
            while parent[k] is not None:
                pk, act = parent[k]
                acts.append(act)
                k = pk
            acts.reverse()
            return acts, key
        exp += 1
        if exp > max_exp:
            return None
        if (exp & 255) == 0 and time.perf_counter() > deadline:
            return None
        vc = ncap.get(v)
        # wait
        if s + 1 <= tlim:
            if vc is None or rt.free((v,), s, 0, N, vc):
                nk = (v, s + 1)
                ng = g + 1
                if ng < gbest.get(nk, _BIG):
                    gbest[nk] = ng
                    parent[nk] = (key, ("w", v, v, s, 1))
                    heapq.heappush(pq, (ng + H(v), ng, s + 1, v))
        if vc is not None and not rt.free((v,), s, 0, r, vc):
            continue
        for (b, W, ecap, keys) in adj[v]:
            ns = s + W
            if ns > tlim:
                continue
            hb = H(b)
            if hb >= _BIG:
                continue
            if ecap is not None:
                if not rt.free(keys, s, r + 1, N, ecap):
                    continue
                okk = True
                for ss in range(s + 1, ns):
                    if not rt.free(keys, ss, 0, N, ecap):
                        okk = False
                        break
                if not okk:
                    continue
            bc = ncap.get(b)
            if bc is not None:
                if not rt.free((b,), s, r + 1, N, bc):
                    continue
                okk = True
                for ss in range(s + 1, ns):
                    if not rt.free((b,), ss, 0, N, bc):
                        okk = False
                        break
                if not okk:
                    continue
            ng = g + W
            if pen_nodes and b in pen_nodes and b != goal:
                ng += pen_nodes[b]
            nk = (b, ns)
            if ng < gbest.get(nk, _BIG):
                gbest[nk] = ng
                parent[nk] = (key, ("m", v, b, s, W))
                heapq.heappush(pq, (ng + hb, ng, ns, b))
    return None


# ----------------------------------------------------------------------------
# assignment: greedy insertion + rollout
# ----------------------------------------------------------------------------
def _opt(dist, n, t, cap, carried, X, q):
    """Evaluate a task: optional pickup at X (taking oldest pods from q), then deliver all.
    carried: list of (dest, entry, pid). q: list of (entry, pid, dest).
    Returns (val, perm, T_end, end_node, k, meanD, deliveries) or None."""
    k = 0
    pods = list(carried)
    if X is not None:
        dd = dist[n].get(X)
        if dd is None:
            return None
        t = t + dd
        n = X
        free = cap - len(carried)
        if free <= 0 or not q:
            return None
        new = q[:free]
        k = len(new)
        for (e, pid, dest) in new:
            pods.append((dest, e, pid))
    if not pods:
        return None
    sts = sorted(set(p[0] for p in pods))
    best = None
    if len(sts) <= 4:
        perms = permutations(sts)
    else:
        # nearest neighbour order
        order = []
        cur = n
        rem = set(sts)
        while rem:
            nx = min(rem, key=lambda x: (dist[cur].get(x, _BIG), x))
            order.append(nx)
            rem.discard(nx)
            cur = nx
        perms = [tuple(order)]
    for perm in perms:
        tt = t
        cur = n
        Ds = {}
        ok = True
        for stn in perm:
            dd = dist[cur].get(stn)
            if dd is None:
                ok = False
                break
            tt += dd
            cur = stn
            Ds[stn] = max(tt - 1, 0)
        if not ok:
            continue
        val = 0.0
        sD = 0
        for (dest, e, pid) in pods:
            D = Ds[dest]
            val += math.exp(-(D - e) / _EXPC)
            sD += D
        if best is None or val > best[0] + 1e-12:
            best = (val, perm, tt, cur, k, float(sD) / len(pods), [(p[2], Ds[p[0]]) for p in pods])
    return best


def _greedy(units, avail0, dist, fixed):
    """units: dict uid -> (n, t, cap, carried). avail0: dict X -> list (entry,pid,dest).
    fixed: list of (uid, X) first tasks. Returns (total, tasks)."""
    U = {}
    for uid, (n, t, cap, carried) in units.items():
        U[uid] = [n, t, cap, list(carried)]
    avail = dict((X, list(q)) for X, q in avail0.items() if q)
    tasks = dict((uid, []) for uid in U)
    total = 0.0
    cache = {}

    def commit(uid, X, res):
        u = U[uid]
        val, perm, T, end, k, mD, dl = res
        if X is not None:
            del avail[X][:k]
            for key in [kk for kk in cache if kk[1] == X]:
                del cache[key]
        for key in [kk for kk in cache if kk[0] == uid]:
            del cache[key]
        u[0] = end
        u[1] = T
        u[3] = []
        tasks[uid].append((X, perm, dl, k))
        return val

    for (uid, X) in fixed:
        if uid not in U:
            continue
        u = U[uid]
        res = _opt(dist, u[0], u[1], u[2], u[3], X, avail.get(X, []) if X is not None else None)
        if res is None:
            continue
        total += commit(uid, X, res)

    while True:
        best = None
        for uid in U:
            u = U[uid]
            carried = u[3]
            free = u[2] - len(carried)
            opts = []
            if carried:
                opts.append(None)
            if free > 0:
                for X, q in avail.items():
                    if q:
                        opts.append(X)
            for X in opts:
                ck = (uid, X)
                if ck in cache:
                    res = cache[ck]
                else:
                    res = _opt(dist, u[0], u[1], u[2], carried, X, avail.get(X) if X is not None else None)
                    cache[ck] = res
                if res is None:
                    continue
                key = res[5]
                if best is None or key < best[0] - 1e-9:
                    best = (key, uid, X, res)
        if best is None:
            break
        total += commit(best[1], best[2], best[3])
    return total, tasks


def _assign(units, avail, dist, deadline, prev_first):
    base_total, base_tasks = _greedy(units, avail, dist, [])
    best_total, best_tasks = base_total, base_tasks
    fixed = []
    order = sorted(units.keys(), key=lambda uid: (units[uid][1], uid))
    for uid in order:
        if time.perf_counter() > deadline:
            break
        n, t, cap, carried = units[uid]
        free = cap - len(carried)
        opts = []
        if carried:
            opts.append(None)
        if free > 0:
            for X, q in avail.items():
                if q:
                    opts.append(X)
        cur_first = best_tasks[uid][0][0] if best_tasks.get(uid) else "none"
        if len(opts) <= 1:
            if best_tasks.get(uid):
                fixed.append((uid, cur_first))
            continue
        chosen = cur_first
        for X in opts:
            if X == cur_first:
                continue
            if time.perf_counter() > deadline:
                break
            tot, tk = _greedy(units, avail, dist, fixed + [(uid, X)])
            bonus = 1e-6 if prev_first.get(uid) == X else 0.0
            if tot > best_total + 1e-9 - bonus:
                best_total, best_tasks = tot, tk
                chosen = X
        if chosen != "none":
            fixed.append((uid, chosen))
    return best_total, best_tasks


# ----------------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------------
def _plan(state, caller):
    st = _S["st"]
    dist = st["dist"]
    ncap = st["ncap"]
    t0 = time.perf_counter()
    soft = t0 + _STEP_SOFT
    hard = t0 + _STEP_HARD
    t = state.current_time_step
    units = sorted(state.drive_units, key=lambda u: u.id)
    N = len(units)
    rank = dict((u.id, i) for i, u in enumerate(units))
    podmap = dict((p.id, p) for p in state.active_pods)

    # pod source statistics (for parking)
    for p in state.active_pods:
        if p.id not in _S["seen_pods"]:
            _S["seen_pods"].add(p.id)
            if p.carried_by is None and p.current_node is not None:
                src = p.current_node
            else:
                src = None
            if src is not None:
                _S["src_count"][src] = _S["src_count"].get(src, 0) + 1

    waiting = {}
    for p in state.active_pods:
        if p.carried_by is None and p.current_node is not None:
            waiting.setdefault(p.current_node, []).append((p.entry_time, p.id, p.destination_station))
    for X in waiting:
        waiting[X].sort()

    # unit start states
    starts = {}  # uid -> (node, s0, fixed_actions)
    sim_units = {}
    arrivals = []
    for u in units:
        r = rank[u.id]
        carried = []
        for pid in u.carrying:
            p = podmap.get(pid)
            if p is not None:
                carried.append((p.destination_station, p.entry_time, p.id))
        if u.in_transit:
            k = _steps_for(u.transit_remaining_time)
            dest = u.transit_destination
            starts[u.id] = (dest, t + k, [("t", u.current_node, dest, t, k)])
            arrivals.append((t + k, u.id, dest, carried, u.capacity))
        else:
            if u.id < caller:
                starts[u.id] = (u.current_node, t + 1, [("w", u.current_node, u.current_node, t, 1)])
                sim_units[u.id] = (u.current_node, t + 1, u.capacity, carried)
            else:
                starts[u.id] = (u.current_node, t, [])
                sim_units[u.id] = (u.current_node, t, u.capacity, carried)
    # transit arrivals: deliveries and involuntary pickups (by arrival order)
    avail = dict((X, list(q)) for X, q in waiting.items())
    forced_pick = {}
    arrivals.sort()
    for (ta, uid, dest, carried, cap) in arrivals:
        carried = [c for c in carried if c[0] != dest]
        free = cap - len(carried)
        q = avail.get(dest)
        if free > 0 and q:
            take = q[:free]
            del q[:free]
            for (e, pid, dd) in take:
                carried.append((dd, e, pid))
            forced_pick[uid] = len(take)
        sim_units[uid] = (dest, ta, cap, carried)

    prev_first = _S.get("prev_first", {})
    total, tasks = _assign(sim_units, avail, dist, t0 + _STEP_SOFT * 0.6, prev_first)
    newfirst = {}
    for uid in tasks:
        if tasks[uid]:
            newfirst[uid] = tasks[uid][0][0]
    _S["prev_first"] = newfirst

    # parking targets for units with no tasks / after tasks
    src_count = _S["src_count"]
    park = st["park"]
    totw = 0.0
    wts = {}
    for X in park:
        w = 1.0 + src_count.get(X, 0)
        wts[X] = w
        totw += w
    load = dict((X, 0) for X in park)

    def choose_park(n):
        best = None
        for X in park:
            dd = dist[n].get(X)
            if dd is None:
                continue
            c = dd + 6.0 * load[X] * totw / (wts[X] * max(1, N))
            if best is None or c < best[0]:
                best = (c, X)
        if best is None:
            return None
        load[best[1]] += 1
        return best[1]

    legs = {}
    for u in units:
        uid = u.id
        tk = tasks.get(uid) or []
        lg = []  # list of (goal, free_cap_during_leg)
        cap = u.capacity
        ncar = len(sim_units[uid][3])
        if tk:
            X, perm, dl, k = tk[0]
            if X is not None:
                lg.append((X, cap - ncar))
                ncar += k
            carried_dests = {}
            for (pid, D) in dl:
                pass
            # count carried per station for free-capacity tracking
            dcount = {}
            # rebuild from sim: pods list order unknown -> approximate
            for stn in perm:
                lg.append((stn, cap - ncar))
                dcount[stn] = 1
            ncar = 0
            if len(tk) > 1 and tk[1][0] is not None:
                lg.append((tk[1][0], cap))
                endn = tk[1][0]
            else:
                endn = perm[-1] if perm else (X if X is not None else sim_units[uid][0])
                pk = choose_park(endn)
                if pk is not None:
                    lg.append((pk, cap))
        else:
            pk = choose_park(sim_units[uid][0])
            if pk is not None:
                lg.append((pk, cap - ncar))
        legs[uid] = lg

    # priorities
    stuck = _S["stuck"]

    def prio(u):
        uid = u.id
        onc = (not u.in_transit) and ncap.get(u.current_node) is not None
        carried = sim_units[uid][3]
        tk = tasks.get(uid) or []
        if carried:
            cls = 1
            key = min(c[1] for c in carried)
        elif tk:
            cls = 2
            key = min([D for (_p, D) in tk[0][2]] or [0])
        else:
            cls = 3
            key = 0
        boost = 0 if stuck.get(uid, 0) >= 6 else 1
        return (boost, 0 if onc else 1, cls, key, uid)

    order = sorted(units, key=prio)

    rt = _RT(N)
    # fixed prefixes + pre-holds
    for u in units:
        r = rank[u.id]
        v, s0, fx = starts[u.id]
        _hold_actions(rt, st, r, fx, None)
        if not u.in_transit and u.id >= caller:
            _hold_actions(rt, st, r, [("p", u.current_node, None, t, 0)], None)

    pen_base = {}
    for X, q in waiting.items():
        if q:
            pen_base[X] = 25

    moves = {}
    expect = {}
    for u in order:
        uid = u.id
        r = rank[uid]
        v, s0, fx = starts[uid]
        if not u.in_transit and uid >= caller:
            _hold_actions(rt, st, r, [("p", u.current_node, None, t, 0)], None, sign=-1)
        acts = []
        cur = (v, s0)
        lg = legs[uid]
        ok_all = True
        for i, (goal, freecap) in enumerate(lg):
            if goal == cur[0] and i < len(lg) - 1:
                continue
            final = (i == len(lg) - 1)
            pen = pen_base if freecap > 0 else None
            dl = hard if time.perf_counter() < soft else min(hard, time.perf_counter() + 0.005)
            res = _astar(rt, st, r, cur[0], cur[1], goal, final, pen, dl)
            if res is None:
                ok_all = False
                break
            a2, cur = res
            acts.extend(a2)
        if not ok_all or not lg:
            # escape to any permanently free node
            res = _astar(rt, st, r, cur[0], cur[1], None, True, None,
                         min(hard, time.perf_counter() + 0.01), max_exp=2000, horizon=30)
            if res is not None:
                a2, cur = res
                acts.extend(a2)
        _hold_actions(rt, st, r, acts, cur)
        mv = None
        if not u.in_transit:
            for a in acts:
                if a[3] == s0:
                    if a[0] == "m" and s0 == t:
                        mv = a[2]
                    break
        moves[uid] = mv
    _S["moves"] = moves
    _S["plan_t"] = t
    _S["plan_caller"] = caller
    _S["legs"] = legs
    return moves


def _valid_move(state, unit, b):
    st = _S["st"]
    v = unit.current_node
    info = st["einfo"].get((v, b))
    if info is None:
        return False
    W, ecap, keys = info
    if ecap is not None:
        c = 0
        for x in state.drive_units:
            if x.in_transit and ((x.current_node, x.transit_destination) in keys):
                c += 1
        if c >= ecap:
            return False
    bc = st["ncap"].get(b)
    if bc is not None:
        c = 0
        for x in state.drive_units:
            if x.in_transit:
                if x.transit_destination == b:
                    c += 1
            elif x.current_node == b:
                c += 1
        if c >= bc:
            return False
    return True


def _fallback(uid, state):
    unit = None
    for u in state.drive_units:
        if u.id == uid:
            unit = u
    if unit is None or unit.in_transit:
        return None
    st = _S.get("st")
    if st is None:
        return None
    dist = st["dist"]
    here = unit.current_node
    target = None
    if unit.carrying:
        best = None
        for p in state.active_pods:
            if p.id in unit.carrying:
                d = dist[here].get(p.destination_station, _BIG)
                if best is None or d < best[0]:
                    best = (d, p.destination_station)
        if best:
            target = best[1]
    else:
        best = None
        for p in state.active_pods:
            if p.carried_by is None and p.current_node is not None:
                d = dist[here].get(p.current_node, _BIG)
                if best is None or d < best[0]:
                    best = (d, p.current_node)
        if best:
            target = best[1]
    if target is None or target == here:
        return None
    for (b, W, cap, keys) in st["adj"].get(here, []):
        if dist[b].get(target, _BIG) + W == dist[here].get(target, _BIG):
            if _valid_move(state, unit, b):
                return b
    return None


def drive_unit_next_move(drive_unit_id, state):
    try:
        return _decide(drive_unit_id, state)
    except Exception:
        try:
            return _fallback(drive_unit_id, state)
        except Exception:
            return None


def _decide(uid, state):
    t = state.current_time_step
    sig = _S.get("sig")
    new = False
    if sig is None:
        new = True
    else:
        if t < _S["last_t"] or (t == _S["last_t"] and uid <= _S["last_uid"]):
            new = True
    if not new and t == 0 and not state.delivered_pods and _S.get("last_t", 0) > 0:
        new = True
    if new or _S.get("sig_t") != t:
        s2 = _signature(state)
        if new or s2 != sig:
            _new_game(state, s2)
        _S["sig_t"] = t
    _S["last_t"] = t
    _S["last_uid"] = uid

    unit = None
    for u in state.drive_units:
        if u.id == uid:
            unit = u
            break
    if unit is None or unit.in_transit:
        return None

    # stuck tracking (once per unit per step)
    lp = _S["lastpos"].get(uid)
    if lp is not None and lp[0] == unit.current_node and lp[1] < t:
        _S["stuck"][uid] = _S["stuck"].get(uid, 0) + (t - lp[1])
    elif lp is None or lp[0] != unit.current_node:
        _S["stuck"][uid] = 0
    _S["lastpos"][uid] = (unit.current_node, t)

    need = _S.get("plan_t") != t
    if not need:
        # verify lower-id units did what we expected
        exp = _S.get("expect", {})
        for x in state.drive_units:
            if x.id >= uid:
                continue
            e = exp.get(x.id)
            if e is None:
                continue
            if e != (x.in_transit, x.transit_destination if x.in_transit else x.current_node):
                need = True
                break
    if need:
        tt0 = time.perf_counter()
        moves = _plan(state, uid)
        _S["game_time"] += time.perf_counter() - tt0
        exp = {}
        for x in state.drive_units:
            if x.in_transit:
                exp[x.id] = (True, x.transit_destination)
            else:
                m = moves.get(x.id)
                if m is not None:
                    exp[x.id] = (True, m)
                else:
                    exp[x.id] = (False, x.current_node)
        _S["expect"] = exp
    mv = _S["moves"].get(uid)
    if mv is None:
        return None
    if not _valid_move(state, unit, mv):
        _S["expect"][uid] = (False, unit.current_node)
        return None
    return mv
