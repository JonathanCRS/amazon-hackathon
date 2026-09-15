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
    """
    Run Dijkstra's algorithm to find the full shortest path, 
    but only return the very first step of that path.
    """
    # If we are already at the target, we don't need to move
    if start == target:
        return None

    # Track the shortest distance to each node from the start
    dist: Dict[int, float] = {start: 0}
    
    # Track the "breadcrumb trail" (which node we came from to reach a node)
    prev: Dict[int, int] = {}
    visited = set()

    # Min-heap prioritizes the paths with the lowest distance cost so far
    heap = [(0, start)]

    while heap:
        d, node = heapq.heappop(heap)

        # Skip nodes we have already fully explored
        if node in visited:
            continue
        visited.add(node)

        # Stop exploring if we reached our target destination
        if node == target:
            break

        # Look at all connected nodes (aisles)
        for neighbor in state.neighbors(node):
            if neighbor in visited:
                continue
                
            edge = state.get_edge(node, neighbor)
            if edge is None:
                continue
                
            # Calculate the cost to move to this neighbor
            new_dist = d + edge.weight
            
            # If this is the fastest way we've found to this neighbor so far, save it
            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                prev[neighbor] = node # Drop a breadcrumb
                heapq.heappush(heap, (new_dist, neighbor))

    # If we explored everything and never reached the target, it's unreachable
    if target not in prev and target != start:
        return None

    # Backtrack along the breadcrumb trail from the target back to the start
    node = target
    while prev.get(node) != start:
        if node not in prev:
            return None
        node = prev[node]

    # Return the first node we step onto right after the start node
    return node


def _choose_target(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Decide what node this drive unit should currently be heading toward.
    """
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None:
        return None

    # SCENARIO 1: The robot is currently carrying a pod
    if unit.carrying:
        pod = state.get_pod(unit.carrying[0])
        if pod is not None:
            # The goal is to drop it off at its required station
            return pod.destination_station
        return None

    # SCENARIO 2: The robot is empty and needs to pick up a pod
    best_target = None
    best_dist = float('inf')
    
    # Check every active pod on the floor
    for pod in state.active_pods:
        # Ignore pods that are already being carried or haven't spawned in a valid spot
        if pod.carried_by is not None or pod.current_node is None:
            continue
            
        # Use our distance measuring function to see how far away this pod is
        dist = _path_distance(state, unit.current_node, pod.current_node)
        
        # If this is the closest pod we've found so far, update our best target
        if dist is not None and dist < best_dist:
            best_dist = dist
            best_target = pod.current_node

    return best_target


def _path_distance(state: GraphState, start: int, target: int) -> Optional[float]:
    """
    Acts as a measuring tape. Calculates the total weighted distance from 
    start to target using Dijkstra's algorithm.
    """
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
        
        # As soon as we reach the target, return the total distance accumulated
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

    # Return None if no path exists
    return None


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    The main orchestrator. Called by the engine once per idle unit per time step.
    """
    unit = state.get_drive_unit(drive_unit_id)
    
    # Do nothing if the unit doesn't exist or is currently moving between squares
    if unit is None or unit.in_transit:
        return None

    # Step 1: Figure out our final destination (a drop-off station or a waiting pod)
    target = _choose_target(drive_unit_id, state)
    if target is None:
        return None

    # Step 2: Figure out the immediate next square to step onto to get closer to that target
    return _shortest_path_next_hop(state, unit.current_node, target)