"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name:
Email address:
*******************
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ar_hackathon.models.graph_state import GraphState
from ar_hackathon.models.drive_unit import DriveUnit
from ar_hackathon.utils.routing_utils import is_valid_move

INF = float("inf")

# Topology cache (static for a test case).
_adj: Dict[int, List[Tuple[int, float]]] = {}
_distances: Dict[int, Dict[int, float]] = {}
_node_types: Dict[int, str] = {}
_node_caps: Dict[int, Optional[int]] = {}
_graph_signature: Optional[Tuple] = None

# Persistent exclusive assignment: pod_id -> unit_id
_pod_owner: Dict[str, int] = {}


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    Args:
        drive_unit_id: ID of the drive unit being routed
        state: GraphState object containing the current state of the floor

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
                      at the current node
    """
    _ensure_graph(state)
    _refresh_assignments(state)

    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    goal = _select_goal(unit, state)
    if goal is None:
        return _idle_move(unit, state)

    if goal == unit.current_node:
        return None

    hop = _first_hop(unit.current_node, goal, state, unit)
    if hop is not None and is_valid_move(state, unit, hop):
        return hop
    return None


def _ensure_graph(state: GraphState) -> None:
    global _adj, _distances, _node_types, _node_caps, _graph_signature, _pod_owner

    signature = (
        tuple(sorted(n.id for n in state.nodes)),
        tuple(sorted((e.from_node, e.to_node, e.weight, e.bidirectional) for e in state.edges)),
    )
    if signature == _graph_signature and _adj:
        return

    _pod_owner = {}
    _graph_signature = signature
    _adj = defaultdict(list)
    _node_types = {n.id: n.node_type for n in state.nodes}
    _node_caps = {n.id: n.capacity for n in state.nodes}

    for edge in state.edges:
        _adj[edge.from_node].append((edge.to_node, float(edge.weight)))
        if edge.bidirectional:
            _adj[edge.to_node].append((edge.from_node, float(edge.weight)))

    _distances = {}
    for node in state.nodes:
        _distances[node.id] = _dijkstra(node.id, blocked_edges=set(), blocked_nodes=set())


def _dijkstra(
    start: int,
    blocked_edges: Set[Tuple[int, int]],
    blocked_nodes: Set[int],
) -> Dict[int, float]:
    dist = {start: 0.0}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist.get(u, INF):
            continue
        for v, w in _adj.get(u, []):
            if (u, v) in blocked_edges:
                continue
            if v in blocked_nodes and v != start:
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _dist(a: int, b: int) -> float:
    if a == b:
        return 0.0
    return _distances.get(a, {}).get(b, INF)


def _refresh_assignments(state: GraphState) -> None:
    """Rebuild exclusive pod-to-unit claims from the live snapshot."""
    global _pod_owner

    active_ids = {pod.id for pod in state.active_pods}
    _pod_owner = {pid: uid for pid, uid in _pod_owner.items() if pid in active_ids}

    claimed_waiting: Set[str] = set()

    for pod in state.active_pods:
        if pod.carried_by is not None:
            _pod_owner[pod.id] = pod.carried_by
            claimed_waiting.add(pod.id)

    # Units already carrying keep those pods; remaining slots can claim waiting pods.
    unit_load: Dict[int, int] = {u.id: len(u.carrying) for u in state.drive_units}
    unit_cap: Dict[int, int] = {u.id: u.capacity for u in state.drive_units}

    # Drop stale claims that would overflow a unit or point at a carried-by-other pod.
    for pod in state.active_pods:
        if pod.carried_by is not None:
            continue
        owner = _pod_owner.get(pod.id)
        if owner is None:
            continue
        unit = state.get_drive_unit(owner)
        if unit is None or unit_load[owner] >= unit_cap[owner]:
            _pod_owner.pop(pod.id, None)
        else:
            unit_load[owner] += 1
            claimed_waiting.add(pod.id)

    waiting = [
        pod for pod in state.active_pods
        if pod.carried_by is None and pod.id not in claimed_waiting and pod.current_node is not None
    ]

    candidates: List[Tuple[float, int, str, int]] = []
    for pod in waiting:
        for unit in state.drive_units:
            if unit_load[unit.id] >= unit_cap[unit.id]:
                continue
            if unit.in_transit:
                pos = unit.transit_destination
            else:
                pos = unit.current_node
            d_pick = _dist(pos, pod.current_node)
            d_del = _dist(pod.current_node, pod.destination_station)
            # Older pods (smaller entry_time) get priority via the sort key.
            cost = d_pick + d_del
            candidates.append((cost, pod.entry_time, pod.id, unit.id))

    candidates.sort()
    assigned_pods: Set[str] = set(claimed_waiting)
    for cost, entry_time, pod_id, unit_id in candidates:
        if pod_id in assigned_pods:
            continue
        if unit_load[unit_id] >= unit_cap[unit_id]:
            continue
        _pod_owner[pod_id] = unit_id
        unit_load[unit_id] += 1
        assigned_pods.add(pod_id)


def _pods_for_unit(unit_id: int, state: GraphState) -> List:
    return [pod for pod in state.active_pods if _pod_owner.get(pod.id) == unit_id]


def _station_is_full(state: GraphState, station_id: int) -> bool:
    node = state.get_node(station_id)
    if node is None or node.capacity is None:
        return False
    return state.node_occupancy(station_id) >= node.capacity


def _is_at_capacity_node(state: GraphState, node_id: int) -> bool:
    node = state.get_node(node_id)
    if node is None or node.capacity is None:
        return False
    return state.node_occupancy(node_id) >= node.capacity


def _edge_is_full(state: GraphState, frm: int, to: int) -> bool:
    edge = state.get_edge(frm, to)
    if edge is None or edge.capacity is None:
        return False
    return state.edge_occupancy(frm, to) >= edge.capacity


def _holding_node(station_id: int, unit: DriveUnit, state: GraphState) -> Optional[int]:
    """Neighbor of a full station to wait at until the dock frees."""
    neighbors = list(state.neighbors(station_id))
    best = None
    best_score = INF
    for n in neighbors:
        if _is_at_capacity_node(state, n) and n != unit.current_node:
            continue
        # Prefer nodes that this unit can actually reach, closer first.
        d = _dist(unit.current_node, n)
        # Slightly prefer travel/storage over other stations.
        penalty = 0.0 if _node_types.get(n) != "station" else 50.0
        score = d + penalty
        if score < best_score:
            best_score = score
            best = n
    return best


def _escape_nodes(state: GraphState) -> Set[int]:
    """Neighbors that a unit on a capacity-limited station may need in order to leave."""
    reserved: Set[int] = set()
    for node in state.nodes:
        if node.node_type != "station" or node.capacity is None:
            continue
        occupants = [
            u for u in state.drive_units
            if (not u.in_transit and u.current_node == node.id)
            or (u.in_transit and u.transit_destination == node.id)
        ]
        if not occupants:
            continue
        neigh = state.neighbors(node.id)
        if len(neigh) == 1:
            reserved.add(neigh[0])
    return reserved


def _should_batch_pickup(unit: DriveUnit, pickup_node: int, carried_stations: List[int]) -> bool:
    """Pick up another pod before delivering if it is cheaper than a second trip."""
    here = unit.current_node
    if not carried_stations:
        return True
    next_station = min(carried_stations, key=lambda s: _dist(here, s))
    go_now = _dist(here, next_station)
    extra = _dist(here, pickup_node) + _dist(pickup_node, next_station)
    # Also compare against delivering then coming back for the extra pod.
    come_back = go_now + _dist(next_station, pickup_node)
    return extra <= come_back


def _select_goal(unit: DriveUnit, state: GraphState) -> Optional[int]:
    assigned = _pods_for_unit(unit.id, state)
    waiting = [p for p in assigned if p.carried_by is None and p.current_node is not None]
    carried = [p for p in assigned if p.carried_by == unit.id]
    # Also include anything physically carried even if assignment lagged.
    carried_ids = set(unit.carrying)
    for pod in state.active_pods:
        if pod.id in carried_ids and pod not in carried:
            carried.append(pod)

    stations = [p.destination_station for p in carried]

    # Batch: fetch another waiting assigned pod before heading to a station.
    if waiting and unit.has_capacity and (not carried or _should_batch_pickup(unit, waiting[0].current_node, stations)):
        # Prefer the closest waiting assigned pod.
        waiting.sort(key=lambda p: (_dist(unit.current_node, p.current_node), p.entry_time, p.id))
        target = waiting[0].current_node
        return target

    if carried:
        # Nearest-neighbor station tour among carried pods.
        stations_unique = list(dict.fromkeys(stations))
        stations_unique.sort(key=lambda s: (_dist(unit.current_node, s), s))
        station = stations_unique[0]
        if _station_is_full(state, station) and unit.current_node != station:
            hold = _holding_node(station, unit, state)
            if hold is not None:
                return hold
            return None
        return station

    return None


def _storage_nodes() -> List[int]:
    return [nid for nid, t in _node_types.items() if t == "storage"]


def _idle_move(unit: DriveUnit, state: GraphState) -> Optional[int]:
    node = state.get_node(unit.current_node)
    # Vacate limited docks so the next unit can unload.
    if node is not None and node.node_type == "station" and node.capacity is not None:
        hop = _leave_node(unit, state)
        if hop is not None:
            return hop

    storages = _storage_nodes()
    if not storages:
        return None
    goal = min(storages, key=lambda s: (_dist(unit.current_node, s), s))
    if goal == unit.current_node:
        return None
    hop = _first_hop(unit.current_node, goal, state, unit)
    if hop is not None and is_valid_move(state, unit, hop):
        return hop
    return None


def _leave_node(unit: DriveUnit, state: GraphState) -> Optional[int]:
    best = None
    best_w = INF
    for n in state.neighbors(unit.current_node):
        if not is_valid_move(state, unit, n):
            continue
        edge = state.get_edge(unit.current_node, n)
        w = float(edge.weight) if edge is not None else INF
        # Prefer leaving toward storage / travel, not another station.
        if _node_types.get(n) == "station":
            w += 100.0
        if w < best_w:
            best_w = w
            best = n
    return best


def _blocked_resources(state: GraphState, unit: DriveUnit, goal: int) -> Tuple[Set[Tuple[int, int]], Set[int]]:
    blocked_edges: Set[Tuple[int, int]] = set()
    blocked_nodes: Set[int] = set()
    escape = _escape_nodes(state)

    for edge in state.edges:
        if edge.capacity is None:
            continue
        occ = state.edge_occupancy(edge.from_node, edge.to_node)
        if occ >= edge.capacity:
            blocked_edges.add((edge.from_node, edge.to_node))
            if edge.bidirectional:
                blocked_edges.add((edge.to_node, edge.from_node))

    for node in state.nodes:
        if node.capacity is None:
            continue
        if state.node_occupancy(node.id) >= node.capacity:
            # Still allow the goal if we are already there; otherwise block entry.
            if node.id != goal:
                blocked_nodes.add(node.id)

    # Keep a hole on the unique exit of a limited station we do not occupy.
    here = unit.current_node
    for n in escape:
        if n == goal or n == here:
            continue
        node = state.get_node(n)
        if node is not None and node.capacity is not None:
            if state.node_occupancy(n) >= max(0, node.capacity - 1):
                blocked_nodes.add(n)
        else:
            # Unlimited node: only block if we would sit on the sole exit while
            # someone is docked and we are not the one leaving.
            docked_elsewhere = any(
                (not u.in_transit and u.current_node != here
                 and state.get_node(u.current_node) is not None
                 and state.get_node(u.current_node).node_type == "station"
                 and state.get_node(u.current_node).capacity is not None
                 and n in state.neighbors(u.current_node)
                 and len(state.neighbors(u.current_node)) == 1)
                for u in state.drive_units if u.id != unit.id
            )
            if docked_elsewhere and state.node_occupancy(n) >= 1 and n != here:
                # Don't pile onto the only exit if someone is already there.
                pass

    return blocked_edges, blocked_nodes


def _first_hop(start: int, goal: int, state: GraphState, unit: DriveUnit) -> Optional[int]:
    if start == goal:
        return None

    blocked_edges, blocked_nodes = _blocked_resources(state, unit, goal)

    # Dijkstra with parent pointers.
    dist = {start: 0.0}
    parent: Dict[int, Optional[int]] = {start: None}
    heap = [(0.0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist.get(u, INF):
            continue
        if u == goal:
            break
        for v, w in _adj.get(u, []):
            if (u, v) in blocked_edges:
                continue
            if v in blocked_nodes:
                continue
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))

    if goal not in parent:
        # Retry ignoring currently-full distant edges/nodes except the immediate hop,
        # so we still walk toward a goal behind a temporary jam. Immediate full
        # resources stay blocked via is_valid_move.
        dist = {start: 0.0}
        parent = {start: None}
        heap = [(0.0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d != dist.get(u, INF):
                continue
            if u == goal:
                break
            for v, w in _adj.get(u, []):
                # Only skip if this hop is an immediately full resource.
                if u == start and (u, v) in blocked_edges:
                    continue
                if u == start and v in blocked_nodes:
                    continue
                nd = d + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(heap, (nd, v))
        if goal not in parent:
            return None

    # Walk back to the first hop.
    cur = goal
    while parent.get(cur) is not None and parent[cur] != start:
        cur = parent[cur]
    if parent.get(cur) != start:
        return None
    return cur
