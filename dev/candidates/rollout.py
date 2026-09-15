"""
Rollout planner: receding-horizon plan search evaluated by an exact internal
simulator of the game engine.

Plan = per-unit ordered stop list ((0, pod) pickup / (1, pod) delivery) plus a
parking node for idle units.  A fixed low-level policy turns a plan into moves
(shortest-path next hop with congestion-aware waiting/detours).  The planner
reconciles the plan with the observed state every step, and on events runs a
hill-climbing search over relocations / swaps / reorderings of stops, scoring
each candidate by simulating the whole remaining game for known pods.
"""

import heapq
import math
import time
from typing import Optional

INF = 10 ** 9

# ---------------------------------------------------------------- parameters
SOFT_EVENT = 0.12      # search budget on steps with new information (s)
SOFT_IDLE = 0.03       # budget to continue an unconverged search (s)
GAME_BUDGET = 35.0     # total planning seconds per game before degrading
ROLL_STEPS = 260       # rollout horizon cap
PARK_SPREAD = 4        # cost added per extra unit parked at the same storage
NODE_DELAY = 2         # assumed delay when a node is full
STALL_LIMIT = 25
PARK_MODE = "storage"
YIELD = True

# unit record indices
U_ID, U_NODE, U_DEST, U_REM, U_CARRY, U_CAP, U_WAIT = 0, 1, 2, 3, 4, 5, 6
# pod record indices
P_NODE, P_DEST, P_ENTRY, P_CARRIER = 0, 1, 2, 3

_G = {}
DEBUG = None  # set to a dict to cross-check predictions


def _exp(dt):
    return math.exp(-dt / 50.0)


# ============================================================ topology
class Topo(object):
    def __init__(self, state):
        self.nodes = [n.id for n in state.nodes]
        self.ncap = {}
        self.ntype = {}
        for n in state.nodes:
            self.ncap[n.id] = n.capacity
            self.ntype[n.id] = n.node_type
        # first edge connecting a->b (engine uses get_edge = first match)
        self.move = {}  # (a,b) -> (wfloat, wceil, cap, pairs)
        out = {}
        for e in state.edges:
            dirs = [(e.from_node, e.to_node)]
            if e.bidirectional:
                dirs.append((e.to_node, e.from_node))
            pairs = tuple(dirs)
            for (a, b) in dirs:
                if a == b:
                    continue
                if (a, b) not in self.move:
                    self.move[(a, b)] = (e.weight, int(math.ceil(e.weight - 1e-12)) if e.weight >= 1 else 1,
                                         e.capacity, pairs)
                    out.setdefault(a, []).append(b)
        for a in out:
            out[a].sort()
        self.out = out
        radj = {}
        for (a, b), m in self.move.items():
            radj.setdefault(b, []).append((a, m[1]))
        allnodes = set(self.nodes)
        for (a, b) in self.move:
            allnodes.add(a)
            allnodes.add(b)
        self.D = {}
        for g in allnodes:
            d = {g: 0}
            pq = [(0, g)]
            while pq:
                du, u = heapq.heappop(pq)
                if du > d[u]:
                    continue
                for v, w in radj.get(u, ()):
                    nd = du + w
                    if nd < d.get(v, INF):
                        d[v] = nd
                        heapq.heappush(pq, (nd, v))
            self.D[g] = d
        self.storages = [n for n in self.nodes if self.ntype[n] == "storage"]
        self.uncapped = [n for n in self.nodes if self.ncap[n] is None]

    def dist(self, a, b):
        return self.D.get(b, {}).get(a, INF)


# ============================================================ sim state
class Sim(object):
    __slots__ = ("t", "units", "pods", "waiting", "nocc", "pcnt", "value", "uidx")

    def copy(self):
        s = Sim()
        s.t = self.t
        s.units = [[u[0], u[1], u[2], u[3], list(u[4]), u[5], u[6]] for u in self.units]
        s.pods = dict((k, v[:]) for k, v in self.pods.items())
        s.waiting = dict((k, v[:]) for k, v in self.waiting.items() if v)
        s.nocc = dict(self.nocc)
        s.pcnt = dict(self.pcnt)
        s.value = self.value
        s.uidx = self.uidx
        return s


def sim_from_state(state, waits):
    s = Sim()
    s.t = state.current_time_step
    units = []
    for u in sorted(state.drive_units, key=lambda x: x.id):
        units.append([u.id, u.current_node, u.transit_destination if u.in_transit else None,
                      float(u.transit_remaining_time) if u.in_transit else 0.0,
                      list(u.carrying), u.capacity, waits.get(u.id, 0)])
    s.units = units
    s.uidx = dict((u[0], i) for i, u in enumerate(units))
    pods = {}
    waiting = {}
    for p in state.active_pods:
        pods[p.id] = [p.current_node if p.carried_by is None else None, p.destination_station,
                      p.entry_time, p.carried_by]
        if p.carried_by is None:
            waiting.setdefault(p.current_node, []).append(p.id)
    for n in waiting:
        waiting[n].sort(key=lambda pid: (pods[pid][P_ENTRY], pid))
    s.pods = pods
    s.waiting = waiting
    nocc = {}
    pcnt = {}
    for u in units:
        if u[U_DEST] is not None:
            nocc[u[U_DEST]] = nocc.get(u[U_DEST], 0) + 1
            k = (u[U_NODE], u[U_DEST])
            pcnt[k] = pcnt.get(k, 0) + 1
        else:
            nocc[u[U_NODE]] = nocc.get(u[U_NODE], 0) + 1
    s.nocc = nocc
    s.pcnt = pcnt
    s.value = 0.0
    return s


def can_move(T, s, a, b):
    m = T.move.get((a, b))
    if m is None:
        return False
    cap = m[2]
    if cap is not None:
        occ = 0
        for pr in m[3]:
            occ += s.pcnt.get(pr, 0)
        if occ >= cap:
            return False
    nc = T.ncap.get(b)
    if nc is not None and s.nocc.get(b, 0) >= nc:
        return False
    return True


def commit_move(T, s, u, b):
    a = u[U_NODE]
    m = T.move[(a, b)]
    u[U_DEST] = b
    u[U_REM] = float(m[0])
    s.nocc[a] = s.nocc.get(a, 0) - 1
    s.nocc[b] = s.nocc.get(b, 0) + 1
    s.pcnt[(a, b)] = s.pcnt.get((a, b), 0) + 1


def resolve_unit(s, u, events):
    """deliveries then pickups for a non-transit unit (engine order)."""
    node = u[U_NODE]
    car = u[U_CARRY]
    pods = s.pods
    if car:
        keep = []
        for pid in car:
            pd = pods.get(pid)
            if pd is not None and pd[P_DEST] == node:
                s.value += _exp(s.t - pd[P_ENTRY])
                del pods[pid]
                events.append((1, u[U_ID], pid))
            else:
                keep.append(pid)
        if len(keep) != len(car):
            u[U_CARRY] = keep
            car = keep
    if len(car) < u[U_CAP]:
        wl = s.waiting.get(node)
        if wl:
            while wl and len(car) < u[U_CAP]:
                pid = wl.pop(0)
                pd = pods[pid]
                pd[P_NODE] = None
                pd[P_CARRIER] = u[U_ID]
                car.append(pid)
                events.append((0, u[U_ID], pid))
                # delivery at the same node happens only at the next resolve


def advance(s, events):
    arrived = False
    for u in s.units:
        if u[U_DEST] is not None:
            u[U_REM] -= 1.0
            if u[U_REM] <= 0:
                k = (u[U_NODE], u[U_DEST])
                s.pcnt[k] -= 1
                u[U_NODE] = u[U_DEST]
                u[U_DEST] = None
                u[U_REM] = 0.0
                arrived = True
    for u in s.units:
        if u[U_DEST] is None:
            resolve_unit(s, u, events)
    return arrived


# ============================================================ plan helpers
def target_of(s, u, stops):
    onboard = len(u[U_CARRY])
    uid = u[U_ID]
    for k, p in stops:
        pd = s.pods.get(p)
        if pd is None:
            continue
        if k == 0:
            if pd[P_CARRIER] is None and onboard < u[U_CAP]:
                return pd[P_NODE]
        else:
            if pd[P_CARRIER] == uid:
                return pd[P_DEST]
    return None


def route_value(T, s, u, stops):
    """static estimate of delivered value along a stop list; None if infeasible."""
    if u[U_DEST] is not None:
        x = u[U_DEST]
        dep = s.t + int(math.ceil(u[U_REM] - 1e-12))
        present = dep - 1
    else:
        x = u[U_NODE]
        dep = s.t
        present = s.t
    onboard = len(u[U_CARRY])
    cap = u[U_CAP]
    val = 0.0
    D = T.D
    pods = s.pods
    for k, p in stops:
        pd = pods[p]
        n = pd[P_NODE] if k == 0 else pd[P_DEST]
        if n is None:
            n = x
        d = D[n].get(x, INF)
        if d >= INF:
            return None
        if d > 0:
            present = dep + d - 1
            dep = dep + d
            x = n
        if k == 0:
            onboard += 1
            if onboard > cap:
                return None
        else:
            onboard -= 1
            val += _exp(present - pd[P_ENTRY])
    return val


def best_insert(T, s, u, stops, pid, pickup, base=None):
    """insert (0,pid)+(1,pid) (or only (1,pid)) at the best positions."""
    if base is None:
        base = route_value(T, s, u, stops)
        if base is None:
            base = 0.0
    best = None
    L = len(stops)
    if pickup:
        for i in range(L + 1):
            a = stops[:i] + [(0, pid)]
            rest = stops[i:]
            for j in range(len(rest) + 1):
                cand = a + rest[:j] + [(1, pid)] + rest[j:]
                v = route_value(T, s, u, cand)
                if v is not None and (best is None or v > best[0]):
                    best = (v, cand)
    else:
        for j in range(L + 1):
            cand = stops[:j] + [(1, pid)] + stops[j:]
            v = route_value(T, s, u, cand)
            if v is not None and (best is None or v > best[0]):
                best = (v, cand)
    if best is None:
        if pickup:
            cand = stops + [(0, pid), (1, pid)]
        else:
            cand = stops + [(1, pid)]
        return (-1.0 + base, cand)
    return best


def reconcile(T, s, plan):
    """prune stale stops, add deliveries for carried pods, assign unassigned
    waiting pods, maintain parking. Idempotent and deterministic."""
    stops_d, park = plan
    pods = s.pods
    new_stops = {}
    assigned = set()
    for u in s.units:
        uid = u[U_ID]
        old = stops_d.get(uid, ())
        picks = set()
        lst = []
        hasd = set()
        for k, p in old:
            pd = pods.get(p)
            if pd is None:
                continue
            if k == 0:
                if pd[P_CARRIER] is None and p not in assigned:
                    lst.append((0, p))
                    picks.add(p)
                    assigned.add(p)
            else:
                if (pd[P_CARRIER] == uid or (pd[P_CARRIER] is None and p in picks)) and p not in hasd:
                    lst.append((1, p))
                    hasd.add(p)
        for p in picks:
            if p not in hasd:
                lst.append((1, p))
                hasd.add(p)
        for p in u[U_CARRY]:
            if p not in hasd and p in pods:
                lst = best_insert(T, s, u, lst, p, False)[1]
                hasd.add(p)
        new_stops[uid] = lst
    # unassigned waiting pods
    un = [p for p, pd in pods.items() if pd[P_CARRIER] is None and p not in assigned]
    if un:
        un.sort(key=lambda p: (pods[p][P_ENTRY], p))
        for p in un:
            best = None
            for u in s.units:
                uid = u[U_ID]
                lst = new_stops[uid]
                base = route_value(T, s, u, lst)
                if base is None:
                    base = 0.0
                v, cand = best_insert(T, s, u, lst, p, True, base)
                gain = v - base
                if best is None or gain > best[0] + 1e-12:
                    best = (gain, uid, cand)
            new_stops[best[1]] = best[2]
    # parking
    new_park = {}
    load = {}
    idle = []
    for u in s.units:
        uid = u[U_ID]
        if target_of(s, u, new_stops[uid]) is None:
            if uid in park:
                new_park[uid] = park[uid]
                load[park[uid]] = load.get(park[uid], 0) + 1
            else:
                idle.append(u)
    for u in idle:
        new_park[u[U_ID]] = choose_park(T, s, u, load)
    return (new_stops, new_park)


def choose_park(T, s, u, load):
    x = u[U_DEST] if u[U_DEST] is not None else u[U_NODE]
    cands = [n for n in T.storages if T.ncap[n] is None] if PARK_MODE == "storage" else []
    best = None
    for n in cands:
        d = T.dist(x, n)
        if d >= INF:
            continue
        c = d + PARK_SPREAD * load.get(n, 0)
        if best is None or c < best[0]:
            best = (c, n)
    if best is None:
        if T.ncap.get(x) is None:
            n = x
        else:
            n = None
            bd = INF
            for m in T.uncapped:
                d = T.dist(x, m)
                if d < bd:
                    bd, n = d, m
            if n is None:
                n = x
    else:
        n = best[1]
    load[n] = load.get(n, 0) + 1
    return n


# ============================================================ low-level policy
def decide(T, s, u, plan):
    stops_d, park = plan
    g = target_of(s, u, stops_d.get(u[U_ID], ()))
    x = u[U_NODE]
    if g is None:
        g = park.get(u[U_ID])
        if g is None or g == x:
            # never stand on a capacity-limited node while idle
            if T.ncap.get(x) is not None:
                g = None
                bd = INF
                for m in T.uncapped:
                    d = T.dist(x, m)
                    if d < bd:
                        bd, g = d, m
                if g is None or g == x:
                    u[U_WAIT] = 0
                    return None
            else:
                u[U_WAIT] = 0
                return None
    if g == x:
        u[U_WAIT] = 0
        return None
    Dg = T.D.get(g)
    if Dg is None:
        return None
    base = Dg.get(x, INF)
    if base >= INF:
        return None
    outs = T.out.get(x, ())
    best_valid = None
    blocked_delay = None
    for v in outs:
        dv = Dg.get(v, INF)
        if dv >= INF:
            continue
        m = T.move[(x, v)]
        c = m[1] + dv
        if can_move(T, s, x, v):
            if best_valid is None or c < best_valid[0]:
                best_valid = (c, v)
        elif c == base:
            # estimate delay of the blocked shortest move
            dl = edge_delay(T, s, x, v, u[U_WAIT])
            if blocked_delay is None or dl < blocked_delay:
                blocked_delay = dl
    if best_valid is None:
        u[U_WAIT] += 1
        return None
    if best_valid[0] <= base:
        u[U_WAIT] = 0
        return best_valid[1]
    # shortest moves blocked: wait or detour
    if blocked_delay is None or best_valid[0] < base + blocked_delay:
        u[U_WAIT] = 0
        return best_valid[1]
    u[U_WAIT] += 1
    return None


def edge_delay(T, s, x, v, wait):
    m = T.move[(x, v)]
    dl = 0
    cap = m[2]
    if cap is not None:
        occ = 0
        for pr in m[3]:
            occ += s.pcnt.get(pr, 0)
        if occ >= cap:
            rems = []
            for w in s.units:
                if w[U_DEST] is not None and (w[U_NODE], w[U_DEST]) in m[3]:
                    rems.append(int(math.ceil(w[U_REM] - 1e-12)))
            if rems:
                dl = min(rems)
    nc = T.ncap.get(v)
    if nc is not None and s.nocc.get(v, 0) >= nc:
        dl = max(dl, NODE_DELAY + max(0, wait - 3))
    return dl


# ============================================================ rollout
def contested_moves(T, s, plan):
    """capacity-limited edge groups / nodes wanted by tasked standing units."""
    stops_d = plan[0]
    edges = set()
    nodes = set()
    for w in s.units:
        if w[U_DEST] is not None:
            continue
        g = target_of(s, w, stops_d.get(w[U_ID], ()))
        if g is None or g == w[U_NODE]:
            continue
        x = w[U_NODE]
        Dg = T.D.get(g)
        base = Dg.get(x, INF)
        for v in T.out.get(x, ()):
            m = T.move[(x, v)]
            if m[1] + Dg.get(v, INF) == base:
                if m[2] is not None:
                    for pr in m[3]:
                        edges.add(pr)
                if T.ncap.get(v) is not None:
                    nodes.add(v)
    return edges, nodes


def routing_phase(T, s, plan, record=None):
    moved = False
    contested = None
    stops_d = plan[0]
    for u in s.units:
        if u[U_DEST] is None:
            nv = decide(T, s, u, plan)
            if nv is not None and YIELD and target_of(s, u, stops_d.get(u[U_ID], ())) is None:
                if contested is None:
                    contested = contested_moves(T, s, plan)
                m = T.move[(u[U_NODE], nv)]
                if (m[2] is not None and (u[U_NODE], nv) in contested[0]) or nv in contested[1]:
                    if T.ncap.get(u[U_NODE]) is None:
                        nv = None
            if nv is not None and can_move(T, s, u[U_NODE], nv):
                commit_move(T, s, u, nv)
                moved = True
                if record is not None:
                    record[u[U_ID]] = nv
            elif record is not None:
                record[u[U_ID]] = None
    return moved


def needs_reconcile(s, plan, events):
    stops_d = plan[0]
    for kind, uid, pid in events:
        if kind == 1:
            return True
        if (1, pid) not in stops_d.get(uid, ()):
            return True
    return False


def rollout(T, s0, plan, max_steps=ROLL_STEPS):
    s = s0.copy()
    t0 = s.t
    stall = 0
    events = []
    while s.pods and s.t - t0 < max_steps:
        moved = routing_phase(T, s, plan)
        del events[:]
        advance(s, events)
        s.t += 1
        if events and needs_reconcile(s, plan, events):
            plan = reconcile(T, s, plan)
        if not moved and not events:
            intr = False
            for u in s.units:
                if u[U_DEST] is not None:
                    intr = True
                    break
            if not intr:
                stall += 1
                if stall > STALL_LIMIT:
                    break
            else:
                stall = 0
        else:
            stall = 0
    val = s.value
    if s.pods:
        pen = 60 if stall > STALL_LIMIT else 5
        uidx = s.uidx
        for p, pd in s.pods.items():
            if pd[P_CARRIER] is None:
                est = s.t + T.dist(pd[P_NODE], pd[P_DEST]) + pen + 10
            else:
                u = s.units[uidx[pd[P_CARRIER]]]
                x = u[U_DEST] if u[U_DEST] is not None else u[U_NODE]
                est = s.t + T.dist(x, pd[P_DEST]) + pen
            if est >= INF:
                continue
            val += _exp(est - pd[P_ENTRY]) * 0.5
    return val


# ============================================================ search
def plan_copy(plan):
    return (dict((k, list(v)) for k, v in plan[0].items()), dict(plan[1]))


def static_total(T, s, stops_d):
    tot = 0.0
    for u in s.units:
        v = route_value(T, s, u, stops_d.get(u[U_ID], []))
        if v is None:
            v = -5.0
        tot += v
    return tot


def gen_candidates(T, s, plan, deadline):
    stops_d, park = plan
    pods = s.pods
    units = s.units
    uv = {}
    for u in units:
        v = route_value(T, s, u, stops_d.get(u[U_ID], []))
        uv[u[U_ID]] = -5.0 if v is None else v
    cands = []
    owner = {}
    for u in units:
        for k, p in stops_d.get(u[U_ID], []):
            if k == 0:
                owner[p] = u[U_ID]
    # relocations
    for p, a in owner.items():
        ua = units[s.uidx[a]]
        rem_a = [st for st in stops_d[a] if st[1] != p]
        va = route_value(T, s, ua, rem_a)
        if va is None:
            continue
        for ub in units:
            b = ub[U_ID]
            if b == a:
                v, cand = best_insert(T, s, ua, rem_a, p, True, va)
                delta = v - uv[a]
                nd = dict(stops_d)
                nd[a] = cand
            else:
                v, cand = best_insert(T, s, ub, stops_d.get(b, []), p, True, uv[b])
                delta = (va - uv[a]) + (v - uv[b])
                nd = dict(stops_d)
                nd[a] = rem_a
                nd[b] = cand
            if cand == stops_d.get(b):
                continue
            cands.append((delta, nd))
        if time.perf_counter() > deadline:
            break
    # swaps between units
    plist = sorted(owner.keys())
    for i in range(len(plist)):
        for j in range(i + 1, len(plist)):
            p, q = plist[i], plist[j]
            a, b = owner[p], owner[q]
            if a == b:
                continue
            sa = [(k, q if x == p else x) for k, x in stops_d[a]]
            sb = [(k, p if x == q else x) for k, x in stops_d[b]]
            va = route_value(T, s, units[s.uidx[a]], sa)
            vb = route_value(T, s, units[s.uidx[b]], sb)
            if va is None or vb is None:
                continue
            nd = dict(stops_d)
            nd[a] = sa
            nd[b] = sb
            cands.append((va + vb - uv[a] - uv[b], nd))
        if time.perf_counter() > deadline:
            break
    # reorder deliveries of carried pods
    for u in units:
        uid = u[U_ID]
        lst = stops_d.get(uid, [])
        if len(lst) < 2:
            continue
        for p in u[U_CARRY]:
            if (1, p) not in lst:
                continue
            rem = [st for st in lst if st != (1, p)]
            for j in range(len(rem) + 1):
                cand = rem[:j] + [(1, p)] + rem[j:]
                if cand == lst:
                    continue
                v = route_value(T, s, u, cand)
                if v is None:
                    continue
                nd = dict(stops_d)
                nd[uid] = cand
                cands.append((v - uv[uid], nd))
    cands.sort(key=lambda c: -c[0])
    return cands


def search(T, s, plan, deadline):
    base = rollout(T, s, plan)
    while True:
        if time.perf_counter() > deadline:
            return plan, False
        cands = gen_candidates(T, s, plan, deadline)
        improved = False
        seen = set()
        for delta, nd in cands:
            if time.perf_counter() > deadline:
                return plan, False
            key = tuple(sorted((k, tuple(v)) for k, v in nd.items()))
            if key in seen:
                continue
            seen.add(key)
            cand = reconcile(T, s, (nd, plan[1]))
            v = rollout(T, s, cand)
            if v > base + 1e-9:
                plan, base = cand, v
                improved = True
                break
        if not improved:
            return plan, True


# ============================================================ controller
def _reset(state, sig):
    _G.clear()
    _G["sig"] = sig
    _G["T"] = Topo(state)
    _G["plan"] = ({}, {})
    _G["waits"] = {}
    _G["t"] = -1
    _G["last_uid"] = -1
    _G["moves"] = {}
    _G["used"] = 0.0
    _G["known"] = set()
    _G["converged"] = False
    _G["ndeliv"] = 0


def _signature(state):
    return (tuple((n.id, n.node_type, n.capacity) for n in state.nodes),
            tuple((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional) for e in state.edges),
            tuple((u.id, u.capacity) for u in state.drive_units))


def _plan_step(state):
    t_start = time.perf_counter()
    T = _G["T"]
    s = sim_from_state(state, _G["waits"])
    plan = reconcile(T, s, _G["plan"])
    known = set(s.pods.keys())
    newinfo = bool(known - _G["known"])
    _G["known"] = known
    if newinfo:
        _G["converged"] = False
    if not _G["converged"]:
        budget = SOFT_EVENT if newinfo else SOFT_IDLE
        if _G["used"] > GAME_BUDGET:
            budget = 0.002
        try:
            plan, conv = search(T, s, plan, t_start + budget)
            _G["converged"] = conv
        except Exception:
            _G["converged"] = True
    _G["plan"] = plan
    rec = {}
    s2 = s.copy()
    routing_phase(T, s2, plan, rec)
    waits = {}
    for u in s2.units:
        waits[u[U_ID]] = u[U_WAIT]
    _G["waits"] = waits
    _G["moves"] = rec
    if DEBUG is not None:
        ev = []
        advance(s2, ev)
        s2.t += 1
        pred = _G.get("pred")
        if pred is not None and pred[0] == s.t:
            now = [(u[U_NODE], u[U_DEST], round(u[U_REM], 6), sorted(p for p in u[U_CARRY] if p in pred[2])) for u in s.units]
            if now != pred[1]:
                DEBUG["mismatch"] = DEBUG.get("mismatch", 0) + 1
                if DEBUG.get("verbose"):
                    print("MISMATCH t=%d" % s.t, now, pred[1])
        DEBUG["checks"] = DEBUG.get("checks", 0) + 1
        _G["pred"] = (s2.t, [(u[U_NODE], u[U_DEST], round(u[U_REM], 6), sorted(u[U_CARRY])) for u in s2.units], set(s2.pods.keys()) | set(p for u in s2.units for p in u[U_CARRY]))
    _G["used"] += time.perf_counter() - t_start


def _fallback(drive_unit_id, state):
    try:
        T = _G.get("T")
        if T is None:
            T = Topo(state)
        unit = None
        for u in state.drive_units:
            if u.id == drive_unit_id:
                unit = u
        if unit is None:
            return None
        x = unit.current_node
        targets = []
        if unit.carrying:
            ids = set(unit.carrying)
            targets = [p.destination_station for p in state.active_pods if p.id in ids]
        else:
            targets = [p.current_node for p in state.active_pods if p.carried_by is None]
        best = None
        for g in targets:
            d = T.dist(x, g)
            if d < INF and d > 0 and (best is None or d < best[0]):
                best = (d, g)
        if best is None:
            return None
        g = best[1]
        Dg = T.D[g]
        bv = None
        for v in T.out.get(x, ()):
            c = T.move[(x, v)][1] + Dg.get(v, INF)
            if bv is None or c < bv[0]:
                bv = (c, v)
        return bv[1] if bv else None
    except Exception:
        return None


def drive_unit_next_move(drive_unit_id, state):
    try:
        t = state.current_time_step
        ndeliv = len(state.delivered_pods)
        newgame = ("T" not in _G or t < _G["t"] or (t == _G["t"] and drive_unit_id <= _G["last_uid"])
                   or ndeliv < _G["ndeliv"])
        if newgame or t != _G["t"]:
            sig = _signature(state)
            if newgame or sig != _G.get("sig"):
                _reset(state, sig)
            _G["t"] = t
            _G["last_uid"] = -1
            _G["ndeliv"] = ndeliv
            _plan_step(state)
        _G["last_uid"] = drive_unit_id
        mv = _G["moves"]
        if drive_unit_id in mv:
            return mv[drive_unit_id]
        return _fallback(drive_unit_id, state)
    except Exception:
        return _fallback(drive_unit_id, state)
