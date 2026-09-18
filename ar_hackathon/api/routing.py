import heapq
from itertools import permutations
from typing import Dict, List, Optional, Tuple
from ar_hackathon.models.graph_state import GraphState

AGE_BONUS = 0.15
DETOUR_SLACK = 2.20
CONGESTION_PENALTY = 2.0
MAX_TSP_STOPS = 4
MAX_ASSIGN_PODS = 60

INF = float('inf')

_cache: Dict[str, object] = {}


def _signature(state: GraphState) -> Tuple:
    return (
        len(state.nodes),
        len(state.edges),
        tuple(sorted((e.from_node, e.to_node, e.weight, e.bidirectional)
                     for e in state.edges)),
    )


def _graph(state: GraphState):
    sig = _signature(state)
    if _cache.get("sig") != sig:
        adj: Dict[int, List[Tuple[int, float]]] = {n.id: [] for n in state.nodes}
        for e in state.edges:
            adj.setdefault(e.from_node, []).append((e.to_node, e.weight))
            if e.bidirectional:
                adj.setdefault(e.to_node, []).append((e.from_node, e.weight))
        _cache["sig"] = sig
        _cache["adj"] = adj
        _cache["sp"] = {}
    return _cache["adj"]


def _paths_from(state: GraphState, source: int):
    adj = _graph(state)
    sp = _cache["sp"]
    cached = sp.get(source)
    if cached is not None:
        return cached

    dist = {source: 0.0}
    first: Dict[int, int] = {}
    heap = [(0.0, source)]
    seen = set()
    while heap:
        cost, node = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        for nbr, w in adj.get(node, ()):
            nd = cost + w
            if nd < dist.get(nbr, INF):
                dist[nbr] = nd
                first[nbr] = nbr if node == source else first[node]
                heapq.heappush(heap, (nd, nbr))

    sp[source] = (dist, first)
    return dist, first


def _dist(state: GraphState, a: int, b: int) -> float:
    if a == b:
        return 0.0
    return _paths_from(state, a)[0].get(b, INF)


class _Traffic:
    __slots__ = ("edge_cap", "edge_occ", "node_cap", "node_occ", "congested")

    def __init__(self, state: GraphState):
        self.edge_cap: Dict[Tuple[int, int], Optional[int]] = {}
        self.edge_occ: Dict[Tuple[int, int], int] = {}
        self.node_cap: Dict[int, Optional[int]] = {}
        self.node_occ: Dict[int, int] = {}

        for e in state.edges:
            key = self._key(e.from_node, e.to_node)
            self.edge_cap[key] = e.capacity
            if not e.bidirectional:
                prev = self.edge_cap.get(key)
                if prev is not None and e.capacity is not None:
                    self.edge_cap[key] = min(prev, e.capacity)

        for n in state.nodes:
            self.node_cap[n.id] = n.capacity

        for u in state.drive_units:
            if u.in_transit:
                key = self._key(u.current_node, u.transit_destination)
                self.edge_occ[key] = self.edge_occ.get(key, 0) + 1
                dest = u.transit_destination
            else:
                dest = u.current_node
            self.node_occ[dest] = self.node_occ.get(dest, 0) + 1

        self.congested = False
        for key, occ in self.edge_occ.items():
            cap = self.edge_cap.get(key)
            if cap is not None and occ >= cap:
                self.congested = True
                break
        if not self.congested:
            for node_id, occ in self.node_occ.items():
                cap = self.node_cap.get(node_id)
                if cap is not None and occ >= cap:
                    self.congested = True
                    break

    @staticmethod
    def _key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    def edge_full(self, a: int, b: int) -> bool:
        key = self._key(a, b)
        cap = self.edge_cap.get(key)
        return cap is not None and self.edge_occ.get(key, 0) >= cap

    def node_full(self, node_id: int) -> bool:
        cap = self.node_cap.get(node_id)
        return cap is not None and self.node_occ.get(node_id, 0) >= cap

    def can_enter(self, state: GraphState, unit, nxt: int) -> bool:
        if nxt is None or nxt == unit.current_node:
            return False
        if state.get_edge(unit.current_node, nxt) is None:
            return False
        if self.edge_full(unit.current_node, nxt):
            return False
        return not self.node_full(nxt)


def _dynamic_hop(state: GraphState, traffic: _Traffic, start: int, target: int,
                 avoid_first: Optional[int] = None) -> Optional[int]:
    if start == target:
        return None
    adj = _graph(state)

    dist = {start: 0.0}
    first: Dict[int, int] = {}
    heap = [(0.0, start)]
    seen = set()

    while heap:
        cost, node = heapq.heappop(heap)
        if node in seen:
            continue
        seen.add(node)
        if node == target:
            break
        for nbr, w in adj.get(node, ()):
            if nbr in seen:
                continue
            if nbr != target and traffic.node_full(nbr):
                continue
            if node == start:
                if avoid_first is not None and nbr == avoid_first:
                    continue
                if traffic.node_full(nbr) and nbr != target:
                    continue
            step = w
            if traffic.edge_full(node, nbr):
                step += CONGESTION_PENALTY
            nd = cost + step
            if nd < dist.get(nbr, INF):
                dist[nbr] = nd
                first[nbr] = nbr if node == start else first[node]
                heapq.heappush(heap, (nd, nbr))

    return first.get(target)


def _route(state: GraphState, traffic: _Traffic, unit, target: int) -> Optional[int]:
    here = unit.current_node
    if target == here:
        return None

    if not traffic.congested:
        hop = _paths_from(state, here)[1].get(target)
        if hop is not None and traffic.can_enter(state, unit, hop):
            return hop

    hop = _dynamic_hop(state, traffic, here, target)
    if hop is not None and traffic.can_enter(state, unit, hop):
        return hop

    if hop is not None and hop == target and traffic.node_full(target) \
            and not traffic.edge_full(here, hop):
        return None

    alt = _dynamic_hop(state, traffic, here, target, avoid_first=hop)
    if alt is not None and traffic.can_enter(state, unit, alt):
        return alt

    return None


def _anchor(unit) -> Tuple[int, float]:
    if unit.in_transit:
        return unit.transit_destination, float(unit.transit_remaining_time)
    return unit.current_node, 0.0


def _assign(state: GraphState) -> Dict[int, str]:
    now = state.current_time_step
    if _cache.get("assign_step") == now:
        return _cache["assign"]

    waiting = [p for p in state.active_pods
               if p.carried_by is None and p.current_node is not None]
    if not waiting:
        _cache["assign_step"] = now
        _cache["assign"] = {}
        return {}

    candidates = []
    for unit in state.drive_units:
        if unit.capacity - len(unit.carrying) <= 0:
            continue
        anchor, eta = _anchor(unit)
        dist = _paths_from(state, anchor)[0]

        pods = waiting
        if len(pods) > MAX_ASSIGN_PODS:
            pods = sorted(pods, key=lambda p: dist.get(p.current_node, INF))
            pods = pods[:MAX_ASSIGN_PODS]

        for pod in pods:
            travel = dist.get(pod.current_node, INF)
            if travel == INF:
                continue
            cost = eta + travel - AGE_BONUS * (now - pod.entry_time)
            candidates.append((cost, unit.id, pod.id))

    candidates.sort()
    assignment: Dict[int, str] = {}
    used_units, used_pods = set(), set()
    for _, unit_id, pod_id in candidates:
        if unit_id in used_units or pod_id in used_pods:
            continue
        assignment[unit_id] = pod_id
        used_units.add(unit_id)
        used_pods.add(pod_id)

    _cache["assign_step"] = now
    _cache["assign"] = assignment
    return assignment


def _next_dropoff(state: GraphState, unit) -> Optional[int]:
    dests = []
    for pod_id in unit.carrying:
        pod = state.get_pod(pod_id)
        if pod is not None:
            dests.append(pod.destination_station)
    dests = list(dict.fromkeys(dests))
    if not dests:
        return None
    if len(dests) == 1:
        return dests[0]

    here = unit.current_node
    if len(dests) <= MAX_TSP_STOPS:
        best_order, best_cost = None, INF
        for order in permutations(dests):
            cost, node, ok = 0.0, here, True
            for stop in order:
                leg = _dist(state, node, stop)
                if leg == INF:
                    ok = False
                    break
                cost += leg
                node = stop
            if ok and cost < best_cost:
                best_cost, best_order = cost, order
        if best_order is not None:
            return best_order[0]

    return min(dests, key=lambda s: _dist(state, here, s))


def _vacate(state: GraphState, traffic: _Traffic, unit) -> Optional[int]:
    if traffic.node_cap.get(unit.current_node) is None:
        return None

    best, best_key = None, None
    for nbr, w in _graph(state).get(unit.current_node, ()):
        if not traffic.can_enter(state, unit, nbr):
            continue
        key = (0 if traffic.node_cap.get(nbr) is None else 1, w)
        if best_key is None or key < best_key:
            best_key, best = key, nbr
    return best


def _reposition(state: GraphState, unit) -> Optional[int]:
    storages = [n.id for n in state.nodes if n.node_type == "storage"]
    if not storages or unit.current_node in storages:
        return None

    claimed = {u.current_node for u in state.drive_units
               if u.id != unit.id and not u.in_transit and not u.carrying}
    options = [s for s in storages if s not in claimed] or storages

    dist = _paths_from(state, unit.current_node)[0]
    target = min(options, key=lambda s: dist.get(s, INF))
    return target if dist.get(target, INF) != INF else None


def _decide(drive_unit_id: int, state: GraphState) -> Optional[int]:
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    traffic = _Traffic(state)
    here = unit.current_node

    dropoff = _next_dropoff(state, unit) if unit.carrying else None

    pickup = None
    pod_id = _assign(state).get(drive_unit_id)
    if pod_id is not None:
        pod = state.get_pod(pod_id)
        if pod is not None and pod.current_node is not None:
            pickup = pod.current_node

    if dropoff is not None and pickup is not None:
        direct = _dist(state, here, dropoff)
        detour = _dist(state, here, pickup) + _dist(state, pickup, dropoff)
        target = pickup if detour <= direct * DETOUR_SLACK else dropoff
    elif dropoff is not None:
        target = dropoff
    elif pickup is not None:
        target = pickup
    else:
        target = None

    if target is None:
        step_off = _vacate(state, traffic, unit)
        if step_off is not None:
            return step_off
        target = _reposition(state, unit)
        if target is None:
            return None
    elif target == here:
        return None

    return _route(state, traffic, unit, target)


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    try:
        return _decide(drive_unit_id, state)
    except Exception:
        try:
            unit = state.get_drive_unit(drive_unit_id)
            if unit is None or unit.in_transit:
                return None
            target = None
            if unit.carrying:
                pod = state.get_pod(unit.carrying[0])
                if pod is not None:
                    target = pod.destination_station
            else:
                best = INF
                for pod in state.active_pods:
                    if pod.carried_by is None and pod.current_node is not None:
                        d = _dist(state, unit.current_node, pod.current_node)
                        if d < best:
                            best, target = d, pod.current_node
            if target is None or target == unit.current_node:
                return None
            return _paths_from(state, unit.current_node)[1].get(target)
        except Exception:
            return None