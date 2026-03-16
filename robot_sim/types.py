"""
types.py — all shared dataclasses for the robot simulation framework.

Dataclasses carry data only.  Pure functions live in their respective
modules (planner.py, controller.py, dynamics.py, …).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Geometry / environment
# ---------------------------------------------------------------------------

@dataclass
class PolygonObstacle:
    """A convex or concave filled polygon in 2-D space."""
    vertices: List[Tuple[float, float]]
    """Ordered list of (x, y) vertices describing the polygon boundary."""


# ---------------------------------------------------------------------------
# Vehicle state and control
# ---------------------------------------------------------------------------

@dataclass
class VehicleState:
    """Full state of a unicycle-model vehicle."""
    x: float       # position x  (m)
    y: float       # position y  (m)
    theta: float   # heading     (rad, measured from +x axis)
    v: float       # speed       (m/s)


@dataclass
class ControlInput:
    """Control input for a unicycle-model vehicle."""
    a: float       # linear acceleration  (m/s²)
    omega: float   # angular velocity     (rad/s)


# ---------------------------------------------------------------------------
# Motion plan
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """A 2-D position waypoint produced by the motion planner."""
    x: float
    y: float


@dataclass
class Path:
    """An ordered sequence of waypoints from start to goal."""
    waypoints: List[Waypoint]


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    """A desired vehicle state at a specific simulation time."""
    t: float        # time since trajectory start  (s)
    x: float        # desired position x            (m)
    y: float        # desired position y            (m)
    theta: float    # desired heading               (rad)
    v: float        # desired speed                 (m/s)


@dataclass
class Trajectory:
    """A time-indexed sequence of desired vehicle states."""
    points: List[TrajectoryPoint]


# ---------------------------------------------------------------------------
# PID controller
# ---------------------------------------------------------------------------

@dataclass
class PIDGains:
    """Proportional / integral / derivative gains."""
    kp: float
    ki: float
    kd: float


@dataclass
class PIDControllerConfig:
    """Gains for the two independent PID channels used by the unicycle PID."""
    heading_gains: PIDGains   # controls omega (angular velocity)
    speed_gains: PIDGains     # controls a     (linear acceleration)


@dataclass
class PIDState:
    """Mutable integrator / derivative state for a single PID channel."""
    integral: float = 0.0
    prev_error: float = 0.0


@dataclass
class PIDControllerState:
    """Combined PID state for heading and speed channels."""
    heading: PIDState = field(default_factory=PIDState)
    speed: PIDState = field(default_factory=PIDState)


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """Top-level simulation configuration provided by the user."""
    initial_state: VehicleState
    goal: Tuple[float, float]          # (x, y) goal position
    obstacles: List[PolygonObstacle]
    bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    dt: float = 0.05                   # simulation timestep  (s)
    max_time: float = 60.0             # maximum simulation duration  (s)
    goal_tolerance: float = 0.3        # distance to consider goal reached  (m)
    cruise_speed: float = 1.5          # nominal travel speed along trajectory (m/s)
    max_speed: float = 3.0             # speed saturation limit  (m/s)
    max_accel: float = 2.0             # acceleration saturation limit (m/s²)
    max_omega: float = 1.5             # angular-velocity saturation limit (rad/s)

    # RRT* planner parameters
    rrt_max_iter: int = 3000
    rrt_step_size: float = 0.5
    rrt_goal_bias: float = 0.1         # probability of sampling the goal directly
    rrt_neighbor_radius: float = 1.5   # rewire neighborhood radius  (m)

    # PID controller
    pid: PIDControllerConfig = field(
        default_factory=lambda: PIDControllerConfig(
            heading_gains=PIDGains(kp=2.5, ki=0.0, kd=0.3),
            speed_gains=PIDGains(kp=1.5, ki=0.1, kd=0.05),
        )
    )
