# Warehouse Robot Routing | Amazon Robotics Hackathon 2026

### Submitted to Amazon Robotics Hackathon @ UBC
### 🏆 2nd place out of 50 teams

## Problem statement

Warehouse drive units must pick up inventory pods and deliver them to pick stations. The floor is a graph: aisles take different amounts of time to cross, and aisles and stations have capacity limits. Several robots may need the same route or station at once. Each pod's score falls the longer it waits for delivery; an undelivered pod earns no points.

## What it does

Our routing algorithm assigns waiting pods to drive units, guides units to pickups and stations, and adapts when traffic blocks a route. It can make pickup detours when a unit has spare carrying capacity, plan the order of multiple deliveries, and move idle units away from limited-capacity locations. The simulator performs pickups and drop-offs automatically when a unit arrives.

## How we built it

At each simulation step, the engine calls `drive_unit_next_move(unit_id, state)` for every idle drive unit. The function returns an adjacent node to enter or `None` to wait. The routing logic uses the full floor state to make four decisions:

1. **Assign work across the fleet.** For each unit with free carrying capacity and each waiting pod, estimate travel from the unit's current position or its destination if already in transit. Favor nearby pods, give older pods a small priority bonus, and greedily assign distinct pods to units. Share assignments across calls in the same time step.
2. **Choose a destination.** Head toward a carried pod's delivery station. When there is room for another pod, take a pickup detour if the extra travel stays within a configured limit. For up to four distinct delivery stations, check possible visit orders and choose the shortest.
3. **Find the next move.** Cache Dijkstra shortest paths for open routes. When traffic interferes, search again with aisle congestion and node occupancy in mind, choose a feasible alternate first move, or wait if entry is blocked.
4. **Keep the floor moving.** Move an idle unit off a capacity-limited node when possible, then reposition it near an unclaimed storage location.

The routing logic uses the Python standard library. The supplied simulation and visualization tools have additional dependencies listed in `setup.py`.

## Challenges we ran into

- **Shortest paths can become traffic jams.** A narrow aisle may be the fastest route on paper but block another robot. The router checks current aisle and node occupancy and looks for alternate moves when the direct route is unavailable.
- **Several units can chase the same pod.** Assigning pods across the fleet once per time step avoids sending multiple units after one pickup and accounts for units that are still in transit.
- **Station space and carrying capacity complicate delivery order.** A station may have only one available slot, while a unit may carry multiple pods. The router chooses delivery order, limits pickup detours, and moves idle units away from constrained nodes.

## Results on the included practice cases

The routing algorithm delivered **24 of 24 pods** across all six provided scenarios, compared with **16 of 24** for the bundled basic driver. Its mean per-case score was **84.20/100**, compared with **58.66/100** for the basic driver. 

## Run it locally

Requires Python 3.7 or newer. From the repository root:

```bash
python3 -m pip install -e .

# Run our router on a sample scenario.
python3 scripts/run_game.py test_cases/level3/test_case_6.json

# Compare against the bundled basic driver.
python3 scripts/run_game.py test_cases/level3/test_case_6.json --driver basic

# Generate an interactive HTML replay.
python3 scripts/visualize.py test_cases/level3/test_case_6.json --format html
```

The runner prints the score, delivered-pod count, average delivery time, and elapsed simulation steps. The replay is written to `visualization_output/animation.html`. Other scenarios are under `test_cases/level1/`, `test_cases/level2/`, and `test_cases/level3/`.

## Key files

| Path | Purpose |
| --- | --- |
| [`ar_hackathon/api/routing.py`](ar_hackathon/api/routing.py) | Our fleet assignment, routing, traffic handling, and delivery decisions |
| [`ar_hackathon/engine/game_engine.py`](ar_hackathon/engine/game_engine.py) | Supplied simulator and scoring logic |
| [`ar_hackathon/examples/basic_driver.py`](ar_hackathon/examples/basic_driver.py) | Supplied comparison driver |
| [`scripts/run_game.py`](scripts/run_game.py) | Command-line simulation runner |
| [`scripts/visualize.py`](scripts/visualize.py) | Interactive simulation replay generator |
