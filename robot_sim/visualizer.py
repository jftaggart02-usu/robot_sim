"""
visualizer.py — matplotlib-based real-time simulation display (pure functions).

The display shows:
  • Polygon obstacles (filled grey)
  • RRT* path         (dashed blue line through waypoints)
  • Time-indexed trajectory (solid cyan line)
  • Current desired state   (green circle + heading arrow)
  • Current vehicle state   (red triangle pointing in heading direction)
  • Goal marker             (gold star)

``init_display`` sets up the figure/axes and returns a :class:`DisplayState`.
``update_display`` redraws the dynamic elements each simulation step.
``save_display`` saves the current figure to a file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.text import Annotation

from robot_sim.types import (
    MultiRobotSimConfig,
    Path,
    SimConfig,
    Trajectory,
    TrajectoryPoint,
    VehicleState,
)


# ---------------------------------------------------------------------------
# Display state (carries all matplotlib artist references)
# ---------------------------------------------------------------------------

@dataclass
class DisplayState:
    """Holds all matplotlib figure, axes, and artist references for the simulation display."""

    fig: Figure
    ax: Axes
    # Static artists (drawn once)
    obstacle_patches: List[MplPolygon] = field(default_factory=list)
    path_line: Optional[Line2D] = None
    traj_line: Optional[Line2D] = None
    goal_marker: Optional[Line2D] = None
    # Dynamic artists (updated each step)
    desired_marker: Optional[Line2D] = None
    desired_arrow: Optional[Annotation] = None
    vehicle_marker: Optional[Line2D] = None
    vehicle_arrow: Optional[Annotation] = None
    vehicle_trail: Optional[Line2D] = None
    # Trail history
    trail_x: List[float] = field(default_factory=list)
    trail_y: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_display(
    config: SimConfig,
    path: Path,
    trajectory: Trajectory,
    interactive: bool = True,
) -> DisplayState:
    """Create and return a :class:`DisplayState` with all static elements drawn.

    Parameters
    ----------
    config:
        Simulation configuration (provides bounds, goal, obstacles).
    path:
        RRT* path waypoints.
    trajectory:
        Time-indexed trajectory.
    interactive:
        When True (default), use interactive matplotlib mode (``plt.ion()``)
        for real-time updates.  Set to False for static / headless rendering.
    """
    if interactive:
        if "TkAgg" in matplotlib.rcsetup.all_backends:
            matplotlib.use("TkAgg")
        plt.ion()

    fig, ax = plt.subplots(figsize=(9, 9))
    x_min, x_max, y_min, y_max = config.bounds
    margin = 0.5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title("Robot Simulation")
    ax.grid(True, linestyle="--", alpha=0.4)

    ds = DisplayState(fig=fig, ax=ax)

    # --- Obstacles ---
    for obs in config.obstacles:
        patch = MplPolygon(
            obs.vertices, closed=True, facecolor="dimgrey", edgecolor="black", alpha=0.7, zorder=2
        )
        ax.add_patch(patch)
        ds.obstacle_patches.append(patch)

    # --- RRT* path ---
    if path.waypoints:
        px = [w.x for w in path.waypoints]
        py = [w.y for w in path.waypoints]
        (line,) = ax.plot(px, py, "b--", linewidth=1.2, alpha=0.6, label="RRT* path", zorder=3)
        ds.path_line = line

    # --- Trajectory ---
    if trajectory.points:
        tx = [p.x for p in trajectory.points]
        ty = [p.y for p in trajectory.points]
        (line,) = ax.plot(tx, ty, "c-", linewidth=1.5, alpha=0.7, label="Trajectory", zorder=4)
        ds.traj_line = line

    # --- Goal ---
    gx, gy = config.goal
    (marker,) = ax.plot(gx, gy, "*", color="gold", markersize=18, label="Goal", zorder=6)
    ds.goal_marker = marker

    # --- Vehicle trail placeholder ---
    (trail,) = ax.plot(
        [], [], "-", color="salmon", linewidth=1.0, alpha=0.6, label="Vehicle trail", zorder=5
    )
    ds.vehicle_trail = trail

    # --- Desired state placeholder ---
    (dm,) = ax.plot([], [], "o", color="limegreen", markersize=10, label="Desired state", zorder=7)
    ds.desired_marker = dm

    # --- Vehicle state placeholder ---
    (vm,) = ax.plot([], [], "^", color="red", markersize=12, label="Vehicle", zorder=8)
    ds.vehicle_marker = vm

    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    if interactive:
        plt.pause(0.001)

    return ds


# ---------------------------------------------------------------------------
# Per-step update
# ---------------------------------------------------------------------------

def update_display(
    ds: DisplayState,
    desired: TrajectoryPoint,
    vehicle: VehicleState,
    pause: float = 0.001,
    interactive: bool = True,
) -> None:
    """Redraw dynamic elements for the current simulation step.

    Parameters
    ----------
    ds:
        :class:`DisplayState` returned by :func:`init_display`.
    desired:
        Current desired state from the trajectory.
    vehicle:
        Current actual vehicle state.
    pause:
        Matplotlib pause duration  (s) — controls animation speed.
    interactive:
        Pass False to skip ``plt.pause()`` (useful for batch rendering).
    """
    arrow_len = 0.4

    # --- Update desired marker ---
    if ds.desired_marker is not None:
        ds.desired_marker.set_data([desired.x], [desired.y])

    # Remove old desired arrow and redraw.
    if ds.desired_arrow is not None:
        ds.desired_arrow.remove()
        ds.desired_arrow = None
    dx = arrow_len * math.cos(desired.theta)
    dy = arrow_len * math.sin(desired.theta)
    ds.desired_arrow = ds.ax.annotate(
        "",
        xy=(desired.x + dx, desired.y + dy),
        xytext=(desired.x, desired.y),
        arrowprops={"arrowstyle": "->", "color": "limegreen", "lw": 1.5},
        zorder=7,
    )

    # --- Update vehicle marker ---
    if ds.vehicle_marker is not None:
        ds.vehicle_marker.set_data([vehicle.x], [vehicle.y])

    if ds.vehicle_arrow is not None:
        ds.vehicle_arrow.remove()
        ds.vehicle_arrow = None
    vx = arrow_len * math.cos(vehicle.theta)
    vy = arrow_len * math.sin(vehicle.theta)
    ds.vehicle_arrow = ds.ax.annotate(
        "",
        xy=(vehicle.x + vx, vehicle.y + vy),
        xytext=(vehicle.x, vehicle.y),
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 2.0},
        zorder=8,
    )

    # --- Update trail ---
    ds.trail_x.append(vehicle.x)
    ds.trail_y.append(vehicle.y)
    if ds.vehicle_trail is not None:
        ds.vehicle_trail.set_data(ds.trail_x, ds.trail_y)

    if interactive:
        ds.fig.canvas.draw_idle()
        plt.pause(pause)


# ---------------------------------------------------------------------------
# Save / finalise
# ---------------------------------------------------------------------------

def save_display(ds: DisplayState, filepath: str) -> None:
    """Save the current figure to *filepath* (PNG, PDF, SVG, …)."""
    ds.fig.savefig(filepath, dpi=150, bbox_inches="tight")


def close_display(ds: DisplayState) -> None:
    """Close the matplotlib figure."""
    plt.close(ds.fig)


# ---------------------------------------------------------------------------
# Post-simulation animation
# ---------------------------------------------------------------------------

def animate_display(
    config: SimConfig,
    path: Path,
    trajectory: Trajectory,
    history: List[Tuple[TrajectoryPoint, VehicleState]],
    filepath: str = "sim_animation.gif",
    fps: int = 20,
    step: int = 1,
) -> None:
    """Create and save a post-simulation animation from recorded states.

    Parameters
    ----------
    config, path, trajectory:
        Passed directly to :func:`init_display` to draw static elements.
    history:
        Ordered list of ``(desired, vehicle)`` state pairs recorded during
        the simulation.  Every *step*-th entry becomes one animation frame.
    filepath:
        Output file.  The format is inferred from the extension
        (``.gif`` requires *Pillow*; ``.mp4`` requires *ffmpeg*).
    fps:
        Playback frame rate (frames per second).
    step:
        Sub-sampling stride.  ``step=1`` uses every recorded state;
        ``step=5`` uses every fifth state, reducing file size.
    """
    if not history:
        return

    ds = init_display(config, path, trajectory, interactive=False)
    frames = history[::step]

    def _update(frame_data: Tuple[TrajectoryPoint, VehicleState]) -> List[Line2D]:
        desired, vehicle = frame_data
        update_display(ds, desired, vehicle, interactive=False)
        return []

    anim = FuncAnimation(
        ds.fig,
        _update,
        frames=frames,
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )

    anim.save(filepath, fps=fps)
    close_display(ds)


# ---------------------------------------------------------------------------
# Multi-robot display
# ---------------------------------------------------------------------------

@dataclass
class _RobotArtists:
    """Per-robot matplotlib artist group used inside :class:`MultiRobotDisplayState`."""

    path_line: Optional[Line2D] = None
    traj_line: Optional[Line2D] = None
    goal_marker: Optional[Line2D] = None
    desired_marker: Optional[Line2D] = None
    desired_arrow: Optional[Annotation] = None
    vehicle_marker: Optional[Line2D] = None
    vehicle_arrow: Optional[Annotation] = None
    vehicle_trail: Optional[Line2D] = None
    trail_x: List[float] = field(default_factory=list)
    trail_y: List[float] = field(default_factory=list)


# #APPLE
@dataclass
class MultiRobotDisplayState:
    """Holds figure/axes and per-robot artist groups for the multi-robot display."""

    fig: Figure
    ax: Axes
    obstacle_patches: List[MplPolygon] = field(default_factory=list)
    robots: List[_RobotArtists] = field(default_factory=list)


def init_multi_display(
    config: MultiRobotSimConfig,
    paths: List[Path],
    trajectories: List[Trajectory],
    interactive: bool = True,
) -> MultiRobotDisplayState:
    """Create and return a :class:`MultiRobotDisplayState` with all static elements drawn.

    Parameters
    ----------
    config:
        Multi-robot simulation configuration (provides bounds, obstacles, robots).
    paths:
        RRT* path for each robot (same order as ``config.robots``).
    trajectories:
        Time-indexed trajectory for each robot (same order as ``config.robots``).
    interactive:
        When True, use interactive matplotlib mode for real-time updates.
    """
    if interactive:
        if "TkAgg" in matplotlib.rcsetup.all_backends:
            matplotlib.use("TkAgg")
        plt.ion()

    fig, ax = plt.subplots(figsize=(9, 9))
    x_min, x_max, y_min, y_max = config.bounds
    margin = 0.5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (m)")
    ax.set_ylabel("y  (m)")
    ax.set_title(f"Robot Simulation ({len(config.robots)} robots)")
    ax.grid(True, linestyle="--", alpha=0.4)

    mds = MultiRobotDisplayState(fig=fig, ax=ax)

    # --- Obstacles (drawn once, shared by all robots) ---
    for obs in config.obstacles:
        patch = MplPolygon(
            obs.vertices, closed=True, facecolor="dimgrey", edgecolor="black",
            alpha=0.7, zorder=2,
        )
        ax.add_patch(patch)
        mds.obstacle_patches.append(patch)

    legend_handles: List[Line2D] = []

    for i, (robot_cfg, path, traj) in enumerate(
        zip(config.robots, paths, trajectories)
    ):
        color = robot_cfg.color
        goal_color = robot_cfg.goal_color if robot_cfg.goal_color is not None else color
        label = robot_cfg.label or f"Robot {i}"
        arts = _RobotArtists()

        # RRT* path
        if path.waypoints:
            px = [w.x for w in path.waypoints]
            py = [w.y for w in path.waypoints]
            (line,) = ax.plot(
                px, py, "--", color=color, linewidth=1.2, alpha=0.5, zorder=3,
            )
            arts.path_line = line

        # Trajectory
        if traj.points:
            tx = [p.x for p in traj.points]
            ty = [p.y for p in traj.points]
            (line,) = ax.plot(
                tx, ty, "-", color=color, linewidth=1.5, alpha=0.4, zorder=4,
            )
            arts.traj_line = line

        # Goal
        gx, gy = robot_cfg.goal
        (gm,) = ax.plot(
            gx, gy, "*", color=goal_color, markersize=16, zorder=6,
        )
        arts.goal_marker = gm

        # Vehicle trail placeholder
        (trail,) = ax.plot(
            [], [], "-", color=color, linewidth=1.0, alpha=0.6, zorder=5,
        )
        arts.vehicle_trail = trail

        # Desired-state placeholder
        (dm,) = ax.plot([], [], "o", color="limegreen", markersize=8, zorder=7)
        arts.desired_marker = dm

        # Vehicle-state placeholder — add a legend entry per robot
        (vm,) = ax.plot(
            [], [], "^", color=color, markersize=12, label=label, zorder=8,
        )
        arts.vehicle_marker = vm
        legend_handles.append(vm)

        mds.robots.append(arts)

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    fig.tight_layout()

    if interactive:
        plt.pause(0.001)

    return mds


def update_multi_display(
    mds: MultiRobotDisplayState,
    desired_states: List[TrajectoryPoint],
    vehicle_states: List[VehicleState],
    pause: float = 0.001,
    interactive: bool = True,
) -> None:
    """Redraw dynamic elements for all robots at the current simulation step.

    Parameters
    ----------
    mds:
        :class:`MultiRobotDisplayState` returned by :func:`init_multi_display`.
    desired_states:
        Desired state for each robot (same order as ``mds.robots``).
    vehicle_states:
        Actual state for each robot (same order as ``mds.robots``).
    pause:
        Matplotlib pause duration  (s).
    interactive:
        Pass False to skip ``plt.pause()`` (useful for batch rendering).
    """
    arrow_len = 0.4

    for arts, desired, vehicle in zip(mds.robots, desired_states, vehicle_states):
        # Desired marker
        if arts.desired_marker is not None:
            arts.desired_marker.set_data([desired.x], [desired.y])

        if arts.desired_arrow is not None:
            arts.desired_arrow.remove()
            arts.desired_arrow = None
        ddx = arrow_len * math.cos(desired.theta)
        ddy = arrow_len * math.sin(desired.theta)
        arts.desired_arrow = mds.ax.annotate(
            "",
            xy=(desired.x + ddx, desired.y + ddy),
            xytext=(desired.x, desired.y),
            arrowprops={"arrowstyle": "->", "color": "limegreen", "lw": 1.2},
            zorder=7,
        )

        # Vehicle marker
        if arts.vehicle_marker is not None:
            color = arts.vehicle_marker.get_color()
            arts.vehicle_marker.set_data([vehicle.x], [vehicle.y])
        else:
            color = "red"

        if arts.vehicle_arrow is not None:
            arts.vehicle_arrow.remove()
            arts.vehicle_arrow = None
        vdx = arrow_len * math.cos(vehicle.theta)
        vdy = arrow_len * math.sin(vehicle.theta)
        arts.vehicle_arrow = mds.ax.annotate(
            "",
            xy=(vehicle.x + vdx, vehicle.y + vdy),
            xytext=(vehicle.x, vehicle.y),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0},
            zorder=8,
        )

        # Trail
        arts.trail_x.append(vehicle.x)
        arts.trail_y.append(vehicle.y)
        if arts.vehicle_trail is not None:
            arts.vehicle_trail.set_data(arts.trail_x, arts.trail_y)

    if interactive:
        mds.fig.canvas.draw_idle()
        plt.pause(pause)


def save_multi_display(mds: MultiRobotDisplayState, filepath: str) -> None:
    """Save the multi-robot figure to *filepath* (PNG, PDF, SVG, …)."""
    mds.fig.savefig(filepath, dpi=150, bbox_inches="tight")


def close_multi_display(mds: MultiRobotDisplayState) -> None:
    """Close the multi-robot matplotlib figure."""
    plt.close(mds.fig)


def animate_multi_display(
    config: MultiRobotSimConfig,
    paths: List[Path],
    trajectories: List[Trajectory],
    history: List[List[Tuple[TrajectoryPoint, VehicleState]]],
    filepath: str = "sim_multi_animation.gif",
    fps: int = 20,
    step: int = 1,
) -> None:
    """Create and save a post-simulation animation from recorded multi-robot states.

    Parameters
    ----------
    config, paths, trajectories:
        Passed to :func:`init_multi_display` to draw static elements.
    history:
        ``history[i]`` is the ordered list of ``(desired, vehicle)`` pairs
        recorded for robot *i* during the simulation.
    filepath:
        Output file (``.gif`` requires *Pillow*; ``.mp4`` requires *ffmpeg*).
    fps:
        Playback frame rate.
    step:
        Sub-sampling stride (``step=1`` uses every frame).
    """
    if not history or not history[0]:
        return

    mds = init_multi_display(config, paths, trajectories, interactive=False)
    n_frames = min(len(h) for h in history)
    frame_indices = range(0, n_frames, step)

    def _update(frame_idx: int) -> List[Line2D]:
        desired_states = [history[r][frame_idx][0] for r in range(len(history))]
        vehicle_states = [history[r][frame_idx][1] for r in range(len(history))]
        update_multi_display(mds, desired_states, vehicle_states, interactive=False)
        return []

    anim = FuncAnimation(
        mds.fig,
        _update,
        frames=list(frame_indices),
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )

    anim.save(filepath, fps=fps)
    close_multi_display(mds)
