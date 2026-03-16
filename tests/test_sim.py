"""
tests/test_sim.py — unit tests for all simulation modules.

Tests are written with plain pytest and the stdlib.  No external testing
libraries beyond pytest are required.
"""

from __future__ import annotations

import math

import pytest

from robot_sim.controller import compute_control
from robot_sim.dynamics import step as dynamics_step
from robot_sim.obstacles import (
    point_in_any_obstacle,
    point_in_obstacle,
    segment_collides_with_any,
    segment_collides_with_obstacle,
)
from robot_sim.planner import plan
from robot_sim.trajectory import build_trajectory, sample_trajectory
from robot_sim.types import (
    ControlInput,
    PIDControllerConfig,
    PIDControllerState,
    PIDGains,
    PolygonObstacle,
    SimConfig,
    TrajectoryPoint,
    VehicleState,
    Waypoint,
    Path,
)


# ---------------------------------------------------------------------------
# Obstacles
# ---------------------------------------------------------------------------

SQUARE_OBS = PolygonObstacle(vertices=[(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)])


def test_point_inside_obstacle():
    assert point_in_obstacle(2.0, 2.0, SQUARE_OBS)


def test_point_outside_obstacle():
    assert not point_in_obstacle(0.0, 0.0, SQUARE_OBS)


def test_point_in_any_obstacle_true():
    assert point_in_any_obstacle(2.0, 2.0, [SQUARE_OBS])


def test_point_in_any_obstacle_false():
    assert not point_in_any_obstacle(5.0, 5.0, [SQUARE_OBS])


def test_segment_collides_with_obstacle():
    # Segment crosses straight through the square.
    assert segment_collides_with_obstacle((0.0, 2.0), (4.0, 2.0), SQUARE_OBS)


def test_segment_no_collision():
    assert not segment_collides_with_obstacle((0.0, 0.0), (0.5, 0.5), SQUARE_OBS)


def test_segment_collides_with_any():
    assert segment_collides_with_any((0.0, 2.0), (4.0, 2.0), [SQUARE_OBS])


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def test_dynamics_zero_control():
    """With zero acceleration and zero omega the vehicle keeps moving straight."""
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=1.0)
    control = ControlInput(a=0.0, omega=0.0)
    new_state = dynamics_step(state, control, dt=1.0)
    assert abs(new_state.x - 1.0) < 1e-9
    assert abs(new_state.y - 0.0) < 1e-9
    assert abs(new_state.theta - 0.0) < 1e-9
    assert abs(new_state.v - 1.0) < 1e-9


def test_dynamics_acceleration():
    """Positive acceleration increases speed."""
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    control = ControlInput(a=1.0, omega=0.0)
    new_state = dynamics_step(state, control, dt=1.0)
    assert new_state.v > 0.0


def test_dynamics_speed_saturation():
    """Speed should not exceed max_speed."""
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=2.9)
    control = ControlInput(a=10.0, omega=0.0)
    new_state = dynamics_step(state, control, dt=1.0, max_speed=3.0)
    assert new_state.v <= 3.0


def test_dynamics_turning():
    """Non-zero omega changes heading."""
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=1.0)
    control = ControlInput(a=0.0, omega=1.0)
    new_state = dynamics_step(state, control, dt=1.0)
    assert abs(new_state.theta - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def test_build_trajectory_single_waypoint():
    path = Path(waypoints=[Waypoint(x=0.0, y=0.0)])
    traj = build_trajectory(path, cruise_speed=1.0)
    assert len(traj.points) == 1
    assert traj.points[0].t == 0.0


def test_build_trajectory_two_waypoints():
    path = Path(waypoints=[Waypoint(x=0.0, y=0.0), Waypoint(x=3.0, y=0.0)])
    traj = build_trajectory(path, cruise_speed=1.5)
    assert len(traj.points) == 2
    # Travel time = 3 m / 1.5 m/s = 2 s
    assert abs(traj.points[1].t - 2.0) < 1e-9
    # Start speed = 0, end speed = 0 (last waypoint)
    assert traj.points[0].v == 0.0
    assert traj.points[1].v == 0.0


def test_sample_trajectory_before_start():
    path = Path(waypoints=[Waypoint(0.0, 0.0), Waypoint(2.0, 0.0)])
    traj = build_trajectory(path, cruise_speed=1.0)
    pt = sample_trajectory(traj, -1.0)
    assert pt.x == traj.points[0].x


def test_sample_trajectory_after_end():
    path = Path(waypoints=[Waypoint(0.0, 0.0), Waypoint(2.0, 0.0)])
    traj = build_trajectory(path, cruise_speed=1.0)
    pt = sample_trajectory(traj, 1000.0)
    assert pt.x == traj.points[-1].x


def test_sample_trajectory_midpoint():
    path = Path(waypoints=[Waypoint(0.0, 0.0), Waypoint(4.0, 0.0)])
    traj = build_trajectory(path, cruise_speed=2.0)
    # Total time = 4/2 = 2 s; midpoint at t=1 should be at x=2
    pt = sample_trajectory(traj, 1.0)
    assert abs(pt.x - 2.0) < 1e-6


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

_PID_CFG = PIDControllerConfig(
    heading_gains=PIDGains(kp=2.0, ki=0.0, kd=0.0),
    speed_gains=PIDGains(kp=1.5, ki=0.0, kd=0.0),
)


def test_controller_heading_error_produces_omega():
    """When vehicle is pointing away from the desired direction, omega != 0."""
    desired = TrajectoryPoint(t=0.0, x=1.0, y=0.0, theta=0.0, v=1.0)
    current = VehicleState(x=0.0, y=0.0, theta=math.pi / 2, v=1.0)
    pid_state = PIDControllerState()
    control, _ = compute_control(desired, current, _PID_CFG, pid_state, dt=0.1)
    assert control.omega != 0.0


def test_controller_no_error_zero_output():
    """When vehicle matches desired state (within proximity), outputs should be near zero."""
    desired = TrajectoryPoint(t=0.0, x=0.0, y=0.0, theta=0.0, v=1.5)
    current = VehicleState(x=0.0, y=0.0, theta=0.0, v=1.5)
    pid_state = PIDControllerState()
    control, _ = compute_control(desired, current, _PID_CFG, pid_state, dt=0.1)
    assert abs(control.omega) < 1e-9
    assert abs(control.a) < 1e-9


# ---------------------------------------------------------------------------
# Planner (integration test — slow)
# ---------------------------------------------------------------------------

def test_planner_finds_path_open_space():
    """RRT* should find a path in an obstacle-free environment."""
    start = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    goal = (5.0, 5.0)
    path = plan(start, goal, obstacles=[], bounds=(0.0, 10.0, 0.0, 10.0),
                max_iter=500, step_size=1.0, seed=0)
    assert path is not None
    assert len(path.waypoints) >= 2
    # Last waypoint should be at the goal.
    last = path.waypoints[-1]
    assert math.hypot(last.x - goal[0], last.y - goal[1]) < 0.5


def test_planner_avoids_obstacles():
    """All path segments should avoid a blocking obstacle."""
    start = VehicleState(x=0.0, y=5.0, theta=0.0, v=0.0)
    goal = (10.0, 5.0)
    # Tall wall blocking the direct route.
    wall = PolygonObstacle(vertices=[(4.5, 0.0), (5.5, 0.0), (5.5, 9.0), (4.5, 9.0)])
    path = plan(start, goal, obstacles=[wall], bounds=(0.0, 10.0, 0.0, 10.0),
                max_iter=2000, step_size=0.5, seed=7)
    assert path is not None
    # Verify no segment intersects the wall.
    for i in range(len(path.waypoints) - 1):
        w1, w2 = path.waypoints[i], path.waypoints[i + 1]
        assert not segment_collides_with_obstacle((w1.x, w1.y), (w2.x, w2.y), wall)
