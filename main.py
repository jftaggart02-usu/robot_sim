"""
main.py — runnable end-to-end robot simulation example.

Usage
-----
    python main.py

The script:
  1. Builds a :class:`~robot_sim.types.SimConfig` with a start state,
     goal, and several polygon obstacles.
  2. Runs RRT* to find a collision-free path.
  3. Converts the path to a time-indexed trajectory.
  4. Enters the main simulation loop:
       • sample desired state from trajectory
       • compute PID control from state error
       • advance unicycle dynamics one step
       • update the real-time matplotlib display
  5. Saves a final screenshot to ``sim_result.png``.

Every major concern is handled by a dedicated module so that any
component (planner, controller, dynamics, visualizer) can be swapped
independently.
"""

from __future__ import annotations

import math
import sys

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless execution.

from robot_sim.controller import compute_control
from robot_sim.dynamics import step as dynamics_step
from robot_sim.planner import plan
from robot_sim.trajectory import build_trajectory, sample_trajectory
from robot_sim.types import (
    PIDControllerState,
    PolygonObstacle,
    SimConfig,
    VehicleState,
)
from robot_sim.visualizer import (
    close_display,
    init_display,
    save_display,
    update_display,
)


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

def build_config() -> SimConfig:
    """Return the example simulation configuration."""
    initial_state = VehicleState(x=0.5, y=0.5, theta=0.0, v=0.0)
    goal = (9.5, 9.5)

    obstacles = [
        PolygonObstacle(vertices=[(2.0, 1.0), (3.5, 1.0), (3.5, 4.5), (2.0, 4.5)]),
        PolygonObstacle(vertices=[(5.0, 3.0), (7.0, 3.0), (7.0, 5.0), (5.0, 5.0)]),
        PolygonObstacle(vertices=[(1.5, 6.0), (4.0, 6.0), (4.0, 8.0), (1.5, 8.0)]),
        PolygonObstacle(vertices=[(6.5, 6.5), (8.5, 6.5), (8.5, 9.0), (6.5, 9.0)]),
    ]

    return SimConfig(
        initial_state=initial_state,
        goal=goal,
        obstacles=obstacles,
        bounds=(0.0, 10.0, 0.0, 10.0),
        dt=0.05,
        max_time=120.0,
        goal_tolerance=0.4,
        cruise_speed=1.5,
        max_speed=3.0,
        max_accel=2.0,
        max_omega=1.5,
        rrt_max_iter=5000,
        rrt_step_size=0.5,
        rrt_goal_bias=0.1,
        rrt_neighbor_radius=1.5,
    )


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run_simulation(config: SimConfig, save_path: str = "sim_result.png") -> None:
    """Execute the full simulation pipeline."""

    # --- 1. Motion planning ---
    print("Running RRT* planner …")
    path = plan(
        initial_state=config.initial_state,
        goal=config.goal,
        obstacles=config.obstacles,
        bounds=config.bounds,
        max_iter=config.rrt_max_iter,
        step_size=config.rrt_step_size,
        goal_bias=config.rrt_goal_bias,
        neighbor_radius=config.rrt_neighbor_radius,
        seed=42,
    )
    if path is None:
        print("ERROR: RRT* failed to find a path. Increase max_iter or check the scenario.")
        sys.exit(1)
    print(f"  Path found with {len(path.waypoints)} waypoints.")

    # --- 2. Trajectory generation ---
    trajectory = build_trajectory(path, cruise_speed=config.cruise_speed)
    print(f"  Trajectory has {len(trajectory.points)} points, "
          f"duration = {trajectory.points[-1].t:.1f} s")

    # --- 3. Visualizer initialisation ---
    display = init_display(config, path, trajectory, interactive=False)

    # --- 4. Main simulation loop ---
    state = config.initial_state
    pid_state = PIDControllerState()
    t = 0.0
    step_count = 0

    print("Simulating …")
    while t <= config.max_time:
        desired = sample_trajectory(trajectory, t)

        control, pid_state = compute_control(
            desired=desired,
            current=state,
            pid_config=config.pid,
            pid_state=pid_state,
            dt=config.dt,
            max_omega=config.max_omega,
            max_accel=config.max_accel,
        )

        state = dynamics_step(
            state=state,
            control=control,
            dt=config.dt,
            max_speed=config.max_speed,
            max_accel=config.max_accel,
            max_omega=config.max_omega,
        )

        # Redraw every 5 steps to reduce overhead.
        if step_count % 5 == 0:
            update_display(display, desired, state, interactive=False)

        t += config.dt
        step_count += 1

        # Check goal reached.
        dist_to_goal = math.hypot(state.x - config.goal[0], state.y - config.goal[1])
        if dist_to_goal < config.goal_tolerance:
            print(f"  Goal reached at t = {t:.2f} s  (step {step_count})")
            break
    else:
        print(f"  Simulation ended at max_time = {config.max_time} s")

    # Final display update with the terminal state.
    update_display(display, sample_trajectory(trajectory, t), state, interactive=False)

    # --- 5. Save result ---
    save_display(display, save_path)
    print(f"  Saved visualisation → {save_path}")
    close_display(display)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = build_config()
    run_simulation(config, save_path="sim_result.png")
