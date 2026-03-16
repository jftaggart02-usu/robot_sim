"""
trajectory.py — time-indexed trajectory generation (pure functions).

Converts a :class:`~robot_sim.types.Path` of 2-D waypoints into a
:class:`~robot_sim.types.Trajectory` by assigning travel times based on
a constant cruise speed and linearly interpolating heading from each
segment direction.

No path smoothing is applied; the trajectory follows the raw RRT* path.
"""

from __future__ import annotations

import math
from typing import List

from robot_sim.types import Path, Trajectory, TrajectoryPoint


def _segment_heading(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return the heading (rad) pointing from (x1,y1) to (x2,y2)."""
    return math.atan2(y2 - y1, x2 - x1)


def _angle_diff(a: float, b: float) -> float:
    """Shortest signed angular difference a - b, wrapped to (−π, π]."""
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


def build_trajectory(path: Path, cruise_speed: float = 1.5) -> Trajectory:
    """Convert a :class:`Path` into a time-indexed :class:`Trajectory`.

    Parameters
    ----------
    path:
        Sequence of 2-D waypoints from the motion planner.
    cruise_speed:
        Constant travel speed used to assign timestamps  (m/s).

    Returns
    -------
    Trajectory
        Time-indexed desired states.  The first point is at t=0 with the
        speed set to 0; the vehicle accelerates to *cruise_speed* once
        moving.  The last point also has v=0 (goal stop).
    """
    waypoints = path.waypoints
    if len(waypoints) == 0:
        return Trajectory(points=[])

    points: List[TrajectoryPoint] = []
    t = 0.0

    for i, wp in enumerate(waypoints):
        if i == 0:
            # Determine initial heading from direction to next waypoint.
            if len(waypoints) > 1:
                heading = _segment_heading(wp.x, wp.y, waypoints[1].x, waypoints[1].y)
            else:
                heading = 0.0
            speed = 0.0
            points.append(TrajectoryPoint(t=t, x=wp.x, y=wp.y, theta=heading, v=speed))
            continue

        prev = waypoints[i - 1]
        dist = math.hypot(wp.x - prev.x, wp.y - prev.y)

        # Assign heading from this segment direction.
        heading = _segment_heading(prev.x, prev.y, wp.x, wp.y)

        # Travel at cruise speed; slow to 0 at the last waypoint.
        speed = 0.0 if i == len(waypoints) - 1 else cruise_speed
        dt = dist / cruise_speed if cruise_speed > 0 else 0.0
        t += dt

        points.append(TrajectoryPoint(t=t, x=wp.x, y=wp.y, theta=heading, v=speed))

    return Trajectory(points=points)


def sample_trajectory(trajectory: Trajectory, t: float) -> TrajectoryPoint:
    """Return the desired state at time *t* by interpolating the trajectory.

    Before the trajectory start the first point is returned; after the
    end the last point is returned.
    """
    pts = trajectory.points
    if not pts:
        raise ValueError("Trajectory is empty.")

    if t <= pts[0].t:
        return pts[0]
    if t >= pts[-1].t:
        return pts[-1]

    # Binary-search for the surrounding segment.
    lo, hi = 0, len(pts) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if pts[mid].t <= t:
            lo = mid
        else:
            hi = mid

    p0, p1 = pts[lo], pts[hi]
    dt = p1.t - p0.t
    alpha = (t - p0.t) / dt if dt > 0 else 0.0

    # Interpolate position and speed linearly; wrap-interpolate heading.
    x = p0.x + alpha * (p1.x - p0.x)
    y = p0.y + alpha * (p1.y - p0.y)
    v = p0.v + alpha * (p1.v - p0.v)
    theta = p0.theta + alpha * _angle_diff(p1.theta, p0.theta)

    return TrajectoryPoint(t=t, x=x, y=y, theta=theta, v=v)
