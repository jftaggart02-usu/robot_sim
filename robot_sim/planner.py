"""
planner.py — RRT* motion planner (pure functions).

The planner takes a start position, goal position, obstacle list, and
workspace bounds, and returns a :class:`~robot_sim.types.Path` of
waypoints that avoids all obstacles.

Algorithm: RRT* (Karaman & Frazzoli 2011).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from robot_sim.obstacles import segment_collides_with_any
from robot_sim.types import Path, PolygonObstacle, VehicleState, Waypoint


# ---------------------------------------------------------------------------
# Internal tree node (private to this module)
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    x: float
    y: float
    parent: Optional[int] = None  # index into the node list
    cost: float = 0.0             # cost from root


# ---------------------------------------------------------------------------
# Helper math
# ---------------------------------------------------------------------------

def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def _nearest_index(nodes: List[_Node], x: float, y: float) -> int:
    """Return index of the node closest to (x, y)."""
    return min(range(len(nodes)), key=lambda i: _dist(nodes[i].x, nodes[i].y, x, y))


def _steer(
    from_node: _Node, tx: float, ty: float, step_size: float
) -> Tuple[float, float]:
    """Return a new point on the segment from_node → (tx, ty) at most *step_size* away."""
    d = _dist(from_node.x, from_node.y, tx, ty)
    if d <= step_size:
        return tx, ty
    ratio = step_size / d
    return from_node.x + ratio * (tx - from_node.x), from_node.y + ratio * (ty - from_node.y)


def _near_indices(
    nodes: List[_Node], x: float, y: float, radius: float
) -> List[int]:
    """Return indices of all nodes within *radius* of (x, y)."""
    return [i for i, n in enumerate(nodes) if _dist(n.x, n.y, x, y) <= radius]


def _collision_free(
    x1: float, y1: float, x2: float, y2: float, obstacles: List[PolygonObstacle]
) -> bool:
    """Return True if the segment (x1,y1)→(x2,y2) is free of obstacles."""
    return not segment_collides_with_any((x1, y1), (x2, y2), obstacles)


def _extract_path(nodes: List[_Node], goal_idx: int) -> Path:
    """Walk parent pointers from *goal_idx* back to root and return a Path."""
    indices: List[int] = []
    idx: Optional[int] = goal_idx
    while idx is not None:
        indices.append(idx)
        idx = nodes[idx].parent
    indices.reverse()
    waypoints = [Waypoint(x=nodes[i].x, y=nodes[i].y) for i in indices]
    return Path(waypoints=waypoints)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan(
    initial_state: VehicleState,
    goal: Tuple[float, float],
    obstacles: List[PolygonObstacle],
    bounds: Tuple[float, float, float, float],
    max_iter: int = 3000,
    step_size: float = 0.5,
    goal_bias: float = 0.1,
    neighbor_radius: float = 1.5,
    seed: Optional[int] = None,
) -> Optional[Path]:
    """Run RRT* and return a collision-free :class:`Path`, or None if planning fails.

    Parameters
    ----------
    initial_state:
        Start state of the vehicle (x, y, theta, v).
    goal:
        ``(x, y)`` goal position.
    obstacles:
        List of polygon obstacles the path must avoid.
    bounds:
        ``(x_min, x_max, y_min, y_max)`` workspace bounds for random sampling.
    max_iter:
        Maximum number of RRT* iterations.
    step_size:
        Maximum branch length at each extension step  (m).
    goal_bias:
        Probability of sampling the goal position directly.
    neighbor_radius:
        Radius for the neighborhood rewire search  (m).
    seed:
        Optional random seed for reproducibility.
    """
    rng = random.Random(seed)

    x_min, x_max, y_min, y_max = bounds
    gx, gy = goal

    # Initialise tree with start node.
    nodes: List[_Node] = [_Node(x=initial_state.x, y=initial_state.y, cost=0.0)]
    goal_idx: Optional[int] = None

    for _ in range(max_iter):
        # --- sample ---
        if rng.random() < goal_bias:
            sx, sy = gx, gy
        else:
            sx = rng.uniform(x_min, x_max)
            sy = rng.uniform(y_min, y_max)

        # --- nearest ---
        near_idx = _nearest_index(nodes, sx, sy)
        near_node = nodes[near_idx]

        # --- steer ---
        nx, ny = _steer(near_node, sx, sy, step_size)

        # --- collision check for new edge ---
        if not _collision_free(near_node.x, near_node.y, nx, ny, obstacles):
            continue

        # --- choose parent with minimum cost among neighbors ---
        neighbors = _near_indices(nodes, nx, ny, neighbor_radius)
        best_parent = near_idx
        best_cost = near_node.cost + _dist(near_node.x, near_node.y, nx, ny)

        for ni in neighbors:
            n = nodes[ni]
            c = n.cost + _dist(n.x, n.y, nx, ny)
            if c < best_cost and _collision_free(n.x, n.y, nx, ny, obstacles):
                best_parent = ni
                best_cost = c

        # --- add new node ---
        new_idx = len(nodes)
        nodes.append(_Node(x=nx, y=ny, parent=best_parent, cost=best_cost))

        # --- rewire: update neighbors if routing through new node is cheaper ---
        for ni in neighbors:
            n = nodes[ni]
            c = best_cost + _dist(nx, ny, n.x, n.y)
            if c < n.cost and _collision_free(nx, ny, n.x, n.y, obstacles):
                n.parent = new_idx
                n.cost = c

        # --- check if we can connect to goal ---
        if _dist(nx, ny, gx, gy) <= step_size:
            if _collision_free(nx, ny, gx, gy, obstacles):
                goal_cost = best_cost + _dist(nx, ny, gx, gy)
                if goal_idx is None:
                    goal_idx = len(nodes)
                    nodes.append(_Node(x=gx, y=gy, parent=new_idx, cost=goal_cost))
                elif goal_cost < nodes[goal_idx].cost:
                    nodes[goal_idx].parent = new_idx
                    nodes[goal_idx].cost = goal_cost

    if goal_idx is None:
        return None

    return _extract_path(nodes, goal_idx)
