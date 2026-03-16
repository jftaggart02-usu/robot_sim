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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow, Polygon as MplPolygon

from robot_sim.types import (
    Path,
    PolygonObstacle,
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
    fig: Figure
    ax: Axes
    # Static artists (drawn once)
    obstacle_patches: List[MplPolygon] = field(default_factory=list)
    path_line: Optional[Line2D] = None
    traj_line: Optional[Line2D] = None
    goal_marker: Optional[Line2D] = None
    # Dynamic artists (updated each step)
    desired_marker: Optional[Line2D] = None
    desired_arrow: Optional[FancyArrow] = None
    vehicle_marker: Optional[Line2D] = None
    vehicle_arrow: Optional[FancyArrow] = None
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
        matplotlib.use("TkAgg") if "TkAgg" in matplotlib.rcsetup.all_backends else None
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
    (trail,) = ax.plot([], [], "-", color="salmon", linewidth=1.0, alpha=0.6, label="Vehicle trail", zorder=5)
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
        arrowprops=dict(arrowstyle="->", color="limegreen", lw=1.5),
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
        arrowprops=dict(arrowstyle="->", color="red", lw=2.0),
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
    from matplotlib.animation import FuncAnimation

    if not history:
        return

    ds = init_display(config, path, trajectory, interactive=False)
    frames = history[::step]

    def _update(frame_data: Tuple[TrajectoryPoint, VehicleState]) -> None:
        desired, vehicle = frame_data
        update_display(ds, desired, vehicle, interactive=False)

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
