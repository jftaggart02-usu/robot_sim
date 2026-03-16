"""
tests/test_sim.py — unit tests for all simulation modules.

Tests are written with plain pytest and the stdlib.  No external testing
libraries beyond pytest are required.
"""

from __future__ import annotations

import math
import os
import tempfile

import matplotlib
matplotlib.use("Agg")  # headless rendering for all tests

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
    MultiRobotSimConfig,
    PIDControllerConfig,
    PIDControllerState,
    PIDGains,
    PolygonObstacle,
    RobotConfig,
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


# ---------------------------------------------------------------------------
# Visualizer / animation
# ---------------------------------------------------------------------------

def _make_small_sim():
    """Return a minimal (config, path, trajectory, history) for visualizer tests."""
    path = Path(waypoints=[Waypoint(0.0, 0.0), Waypoint(2.0, 0.0)])
    traj = build_trajectory(path, cruise_speed=1.0)
    config = SimConfig(
        initial_state=VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0),
        goal=(2.0, 0.0),
        obstacles=[],
        bounds=(0.0, 5.0, 0.0, 5.0),
    )
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    history = []
    for i in range(5):
        desired = traj.points[0]
        history.append((desired, state))
    return config, path, traj, history


def test_animate_display_creates_gif():
    """animate_display should write a non-empty GIF file."""
    from robot_sim.visualizer import animate_display

    config, path, traj, history = _make_small_sim()
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        out = f.name
    try:
        animate_display(config, path, traj, history, filepath=out, fps=5, step=1)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
    finally:
        if os.path.exists(out):
            os.unlink(out)


def test_animate_display_empty_history():
    """animate_display should silently return when history is empty."""
    from robot_sim.visualizer import animate_display

    config, path, traj, _ = _make_small_sim()
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        out = f.name
    os.unlink(out)  # ensure it doesn't exist yet
    try:
        animate_display(config, path, traj, [], filepath=out, fps=5)
        # File should NOT be created for empty history.
        assert not os.path.exists(out)
    finally:
        if os.path.exists(out):
            os.unlink(out)


# ---------------------------------------------------------------------------
# Multi-robot types
# ---------------------------------------------------------------------------

def test_robot_config_defaults():
    """RobotConfig should have sensible default values."""
    state = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    robot = RobotConfig(initial_state=state, goal=(5.0, 5.0))
    assert robot.label == ""
    assert robot.color == "red"
    assert robot.cruise_speed == 1.5
    assert robot.goal_tolerance == 0.3
    assert robot.goal_color is None


def test_multi_robot_sim_config_holds_robots():
    """MultiRobotSimConfig should store all robots and shared environment."""
    robots = [
        RobotConfig(
            initial_state=VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0),
            goal=(5.0, 5.0),
            label="R0",
            color="blue",
        ),
        RobotConfig(
            initial_state=VehicleState(x=5.0, y=0.0, theta=math.pi, v=0.0),
            goal=(0.0, 5.0),
            label="R1",
            color="green",
        ),
    ]
    config = MultiRobotSimConfig(
        robots=robots,
        obstacles=[SQUARE_OBS],
        bounds=(0.0, 10.0, 0.0, 10.0),
        dt=0.05,
        max_time=30.0,
    )
    assert len(config.robots) == 2
    assert config.robots[0].label == "R0"
    assert config.robots[1].label == "R1"
    assert config.dt == 0.05
    assert config.max_time == 30.0


# ---------------------------------------------------------------------------
# Multi-robot visualizer
# ---------------------------------------------------------------------------

def _make_multi_sim():
    """Return a minimal multi-robot (config, paths, trajectories, history) for tests."""
    path_a = Path(waypoints=[Waypoint(0.0, 0.0), Waypoint(2.0, 0.0)])
    path_b = Path(waypoints=[Waypoint(2.0, 2.0), Waypoint(0.0, 2.0)])
    traj_a = build_trajectory(path_a, cruise_speed=1.0)
    traj_b = build_trajectory(path_b, cruise_speed=1.0)

    robots = [
        RobotConfig(
            initial_state=VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0),
            goal=(2.0, 0.0),
            label="R0",
            color="blue",
        ),
        RobotConfig(
            initial_state=VehicleState(x=2.0, y=2.0, theta=math.pi, v=0.0),
            goal=(0.0, 2.0),
            label="R1",
            color="green",
        ),
    ]
    config = MultiRobotSimConfig(
        robots=robots,
        obstacles=[],
        bounds=(0.0, 5.0, 0.0, 5.0),
    )

    state_a = VehicleState(x=0.0, y=0.0, theta=0.0, v=0.0)
    state_b = VehicleState(x=2.0, y=2.0, theta=math.pi, v=0.0)
    history = [
        [(traj_a.points[0], state_a) for _ in range(5)],
        [(traj_b.points[0], state_b) for _ in range(5)],
    ]
    return config, [path_a, path_b], [traj_a, traj_b], history


def test_init_multi_display():
    """init_multi_display should create a MultiRobotDisplayState with one artist group per robot."""
    from robot_sim.visualizer import (
        MultiRobotDisplayState,
        close_multi_display,
        init_multi_display,
    )

    config, paths, trajs, _ = _make_multi_sim()
    mds = init_multi_display(config, paths, trajs, interactive=False)
    assert isinstance(mds, MultiRobotDisplayState)
    assert len(mds.robots) == len(config.robots)
    close_multi_display(mds)


def test_update_multi_display():
    """update_multi_display should update all robot artists without error."""
    from robot_sim.visualizer import (
        close_multi_display,
        init_multi_display,
        update_multi_display,
    )

    config, paths, trajs, history = _make_multi_sim()
    mds = init_multi_display(config, paths, trajs, interactive=False)
    desired = [h[0][0] for h in history]
    vehicles = [h[0][1] for h in history]
    update_multi_display(mds, desired, vehicles, interactive=False)
    # Trail should have grown.
    assert len(mds.robots[0].trail_x) == 1
    assert len(mds.robots[1].trail_x) == 1
    close_multi_display(mds)


def test_animate_multi_display_creates_gif():
    """animate_multi_display should write a non-empty GIF file."""
    from robot_sim.visualizer import animate_multi_display

    config, paths, trajs, history = _make_multi_sim()
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        out = f.name
    try:
        animate_multi_display(config, paths, trajs, history, filepath=out, fps=5, step=1)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
    finally:
        if os.path.exists(out):
            os.unlink(out)


def test_animate_multi_display_empty_history():
    """animate_multi_display should silently return when history is empty."""
    from robot_sim.visualizer import animate_multi_display

    config, paths, trajs, _ = _make_multi_sim()
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        out = f.name
    os.unlink(out)
    try:
        animate_multi_display(config, paths, trajs, [[], []], filepath=out, fps=5)
        assert not os.path.exists(out)
    finally:
        if os.path.exists(out):
            os.unlink(out)
