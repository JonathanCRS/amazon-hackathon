"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: Boys+Megan
Email address: meganong1@gmail.com, jonathncrsinaga@gmail.com, jeffersonabrahamdermawan@gmail.com
*******************
"""

import heapq
from typing import Dict, Optional
from ar_hackathon.models.graph_state import GraphState


def _shortest_path_next_hop(state: GraphState, start: int, target: int) -> Optional[int]:
    if start == target:
        return None

    dist: Dict[int, float] = {start: 0}
    prev: Dict[int, int] = {}
    visited = set()

    heap = [(0, start)]

    while heap:
        d, node = heapq.heappop(heap)

        if node in visited:
            continue
        visited.add(node)

        if node == target:
            break

        for neighbor in state.neighbors(node):
            if neighbor in visited:
                continue
            edge = state.get_edge(node, neighbor)
            if edge is None:
                continue
            new_dist = d + edge.weight
            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(heap, (new_dist, neighbor))

    if target not in prev and target != start:
        # No path found
        return None

    # Walk the prev chain back from target to start to find the first hop.
    node = target
    while prev.get(node) != start:
        if node not in prev:
            return None
        node = prev[node]

    return node


def _choose_target(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Decide what node this drive unit should currently be heading toward.

    If it's carrying a pod, head to that pod's destination station.
    Otherwise, head to the nearest unclaimed waiting pod (by node, ignoring
    other drive units already en route to it -- fine for Level 1's single
    drive unit case).
    """
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None:
        return None

    if unit.carrying:
        pod = state.get_pod(unit.carrying[0])
        if pod is not None:
            return pod.destination_station
        return None

    best_target = None
    best_dist = float('inf')
    for pod in state.active_pods:
        if pod.carried_by is not None or pod.current_node is None:
            continue
        dist = _path_distance(state, unit.current_node, pod.current_node)
        if dist is not None and dist < best_dist:
            best_dist = dist
            best_target = pod.current_node

    return best_target


def _path_distance(state: GraphState, start: int, target: int) -> Optional[float]:
    """Total weighted distance from start to target, or None if unreachable."""
    if start == target:
        return 0

    dist: Dict[int, float] = {start: 0}
    visited = set()
    heap = [(0, start)]

    while heap:
        d, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            return d
        for neighbor in state.neighbors(node):
            if neighbor in visited:
                continue
            edge = state.get_edge(node, neighbor)
            if edge is None:
                continue
            new_dist = d + edge.weight
            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    return None


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    target = _choose_target(drive_unit_id, state)
    if target is None:
        return None

    return _shortest_path_next_hop(state, unit.current_node, target)