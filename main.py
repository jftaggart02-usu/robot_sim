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

import argparse
import math
import sys
from typing import List, Tuple

# Set the matplotlib backend before any pyplot import.
# Parse --interactive with a minimal pre-parser so that argparse semantics
# are used (rather than a raw sys.argv string check) and the backend is
# correctly chosen before matplotlib.pyplot is imported.
import matplotlib
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--interactive", action="store_true")
_pre_args, _ = _pre.parse_known_args()
if not _pre_args.interactive:
    matplotlib.use("Agg")  # headless rendering unless a live window is requested

from robot_sim.controller import compute_control
from robot_sim.dynamics import step as dynamics_step
from robot_sim.planner import plan
from robot_sim.trajectory import build_trajectory, sample_trajectory
from robot_sim.types import (
    MultiRobotSimConfig,
    PIDControllerState,
    PolygonObstacle,
    RobotConfig,
    SimConfig,
    TrajectoryPoint,
    VehicleState,
)
from robot_sim.visualizer import (
    animate_display,
    animate_multi_display,
    close_display,
    close_multi_display,
    init_display,
    init_multi_display,
    save_display,
    save_multi_display,
    update_display,
    update_multi_display,
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

def run_simulation(
    config: SimConfig,
    save_path: str = "sim_result.png",
    animate_path: str = "sim_animation.gif",
    interactive: bool = False,
) -> None:
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
    display = init_display(config, path, trajectory, interactive=interactive)

    # --- 4. Main simulation loop ---
    state = config.initial_state
    pid_state = PIDControllerState()
    t = 0.0
    step_count = 0
    history: List[Tuple[TrajectoryPoint, VehicleState]] = []

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

        history.append((desired, state))

        # Redraw every 5 steps to reduce overhead.
        if step_count % 5 == 0:
            update_display(display, desired, state, interactive=interactive)

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
    final_desired = sample_trajectory(trajectory, t)
    update_display(display, final_desired, state, interactive=interactive)

    # --- 5. Save static result ---
    save_display(display, save_path)
    print(f"  Saved visualisation → {save_path}")
    close_display(display)

    # --- 6. Post-simulation animation ---
    if animate_path:
        print("Generating animation …")
        animate_display(config, path, trajectory, history, filepath=animate_path)
        print(f"  Saved animation → {animate_path}")


# ---------------------------------------------------------------------------
# Multi-robot scenario
# ---------------------------------------------------------------------------

def build_multi_robot_config() -> MultiRobotSimConfig:
    """Return an example multi-robot simulation configuration.

    Three robots start at different corners and navigate to opposite goals,
    sharing the same obstacle field.
    """
    obstacles = [
        PolygonObstacle(vertices=[(2.0, 1.0), (3.5, 1.0), (3.5, 4.5), (2.0, 4.5)]),
        PolygonObstacle(vertices=[(5.0, 3.0), (7.0, 3.0), (7.0, 5.0), (5.0, 5.0)]),
        PolygonObstacle(vertices=[(1.5, 6.0), (4.0, 6.0), (4.0, 8.0), (1.5, 8.0)]),
        PolygonObstacle(vertices=[(6.5, 6.5), (8.5, 6.5), (8.5, 9.0), (6.5, 9.0)]),
    ]

    robots = [
        RobotConfig(
            initial_state=VehicleState(x=0.5, y=0.5, theta=0.0, v=0.0),
            goal=(9.5, 9.5),
            label="Robot A",
            color="royalblue",
            rrt_max_iter=5000,
            rrt_seed=42,
        ),
        RobotConfig(
            initial_state=VehicleState(x=6.5, y=1.0, theta=math.pi/2, v=0.0),
            goal=(0.0, 6.5),
            label="Robot B",
            color="crimson",
            rrt_max_iter=5000,
            rrt_seed=45,
        ),
        RobotConfig(
            initial_state=VehicleState(x=0.5, y=9.5, theta=-math.pi / 2, v=0.0),
            goal=(9.5, 0.5),
            label="Robot C",
            color="forestgreen",
            rrt_max_iter=5000,
            rrt_seed=44,
        ),
    ]

    return MultiRobotSimConfig(
        robots=robots,
        obstacles=obstacles,
        bounds=(0.0, 10.0, 0.0, 10.0),
        dt=0.05,
        max_time=10.0,
    )


def run_multi_simulation(
    config: MultiRobotSimConfig,
    save_path: str = "sim_multi_result.png",
    animate_path: str = "sim_multi_animation.gif",
    interactive: bool = False,
) -> None:
    """Execute the full multi-robot simulation pipeline.

    Each robot is planned, given a trajectory, and stepped simultaneously
    through the same shared environment at every tick.

    Parameters
    ----------
    config:
        Multi-robot simulation configuration.
    save_path:
        Output path for the final static PNG image.
    animate_path:
        Output path for the GIF animation (set to ``""`` to skip).
    interactive:
        When True, show a live matplotlib window during simulation.
    """
    n = len(config.robots)

    # --- 1. Motion planning (one RRT* per robot) ---
    paths = []
    for i, robot in enumerate(config.robots):
        print(f"Running RRT* planner for {robot.label or f'Robot {i}'} …")
        path = plan(
            initial_state=robot.initial_state,
            goal=robot.goal,
            obstacles=config.obstacles,
            bounds=config.bounds,
            max_iter=robot.rrt_max_iter,
            step_size=robot.rrt_step_size,
            goal_bias=robot.rrt_goal_bias,
            neighbor_radius=robot.rrt_neighbor_radius,
            seed=robot.rrt_seed if robot.rrt_seed is not None else 42 + i,
        )
        if path is None:
            print(
                f"ERROR: RRT* failed for {robot.label or f'Robot {i}'}. "
                "Increase max_iter or check the scenario."
            )
            sys.exit(1)
        print(f"  Path found with {len(path.waypoints)} waypoints.")
        paths.append(path)

    # --- 2. Trajectory generation ---
    trajectories = []
    for robot, path in zip(config.robots, paths):
        traj = build_trajectory(path, cruise_speed=robot.cruise_speed)
        print(
            f"  {robot.label or 'Robot'}: trajectory has {len(traj.points)} points, "
            f"duration = {traj.points[-1].t:.1f} s"
        )
        trajectories.append(traj)

    # --- 3. Visualiser initialisation ---
    display = init_multi_display(config, paths, trajectories, interactive=interactive)

    # --- 4. Main simulation loop ---
    states = [r.initial_state for r in config.robots]
    pid_states = [PIDControllerState() for _ in range(n)]
    reached = [False] * n
    t = 0.0
    step_count = 0
    # history[i] is the per-robot list of (desired, vehicle) pairs
    history: List[List[Tuple[TrajectoryPoint, VehicleState]]] = [[] for _ in range(n)]

    print("Simulating …")
    while t <= config.max_time:
        desired_states = [
            sample_trajectory(trajectories[i], t) for i in range(n)
        ]

        new_states = []
        new_pid_states = []
        for i, robot in enumerate(config.robots):
            control, new_pid = compute_control(
                desired=desired_states[i],
                current=states[i],
                pid_config=robot.pid,
                pid_state=pid_states[i],
                dt=config.dt,
                max_omega=robot.max_omega,
                max_accel=robot.max_accel,
            )
            new_state = dynamics_step(
                state=states[i],
                control=control,
                dt=config.dt,
                max_speed=robot.max_speed,
                max_accel=robot.max_accel,
                max_omega=robot.max_omega,
            )
            new_states.append(new_state)
            new_pid_states.append(new_pid)
            history[i].append((desired_states[i], new_state))

        states = new_states
        pid_states = new_pid_states

        if step_count % 5 == 0:
            update_multi_display(display, desired_states, states, interactive=interactive)

        t += config.dt
        step_count += 1

        # Check which robots have reached their goals.
        for i, robot in enumerate(config.robots):
            if not reached[i]:
                dist = math.hypot(
                    states[i].x - robot.goal[0], states[i].y - robot.goal[1]
                )
                if dist < robot.goal_tolerance:
                    reached[i] = True
                    print(
                        f"  {robot.label or f'Robot {i}'} reached goal at "
                        f"t = {t:.2f} s  (step {step_count})"
                    )

        if all(reached):
            print("  All robots reached their goals.")
            break
    else:
        print(f"  Simulation ended at max_time = {config.max_time} s")

    # Final display update.
    final_desired = [sample_trajectory(trajectories[i], t) for i in range(n)]
    update_multi_display(display, final_desired, states, interactive=interactive)

    # --- 5. Save static result ---
    save_multi_display(display, save_path)
    print(f"  Saved visualisation → {save_path}")
    close_multi_display(display)

    # --- 6. Post-simulation animation ---
    if animate_path:
        print("Generating animation …")
        animate_multi_display(
            config, paths, trajectories, history, filepath=animate_path
        )
        print(f"  Saved animation → {animate_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot simulation demo.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show a live matplotlib window during simulation.",
    )
    parser.add_argument(
        "--multi-robot",
        action="store_true",
        help="Run the multi-robot scenario instead of the single-robot scenario.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        metavar="FILE",
        help=(
            "Output path for the final static image "
            "(default: sim_result.png for single-robot, "
            "sim_multi_result.png for multi-robot)."
        ),
    )
    anim_group = parser.add_mutually_exclusive_group()
    anim_group.add_argument(
        "--animate",
        nargs="?",
        const="__default__",
        default="__default__",
        metavar="FILE",
        help=(
            "Save a post-simulation animation to FILE "
            "(default: sim_animation.gif / sim_multi_animation.gif)."
        ),
    )
    anim_group.add_argument(
        "--no-animate",
        action="store_const",
        dest="animate",
        const=None,
        help="Skip saving the animation.",
    )
    args = parser.parse_args()

    if args.multi_robot:
        anim = (
            "sim_multi_animation.gif"
            if args.animate == "__default__"
            else args.animate or ""
        )
        multi_config = build_multi_robot_config()
        run_multi_simulation(
            multi_config,
            save_path=args.save_path or "sim_multi_result.png",
            animate_path=anim,
            interactive=args.interactive,
        )
    else:
        anim = (
            "sim_animation.gif"
            if args.animate == "__default__"
            else args.animate or ""
        )
        single_config = build_config()
        run_simulation(
            single_config,
            save_path=args.save_path or "sim_result.png",
            animate_path=anim,
            interactive=args.interactive,
        )
