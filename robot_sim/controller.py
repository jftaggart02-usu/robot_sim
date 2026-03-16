"""
controller.py — PID controller for unicycle-model vehicles (pure functions).

Two independent PID channels are used:
  * Heading channel → produces omega (angular velocity, rad/s)
  * Speed  channel  → produces a    (linear acceleration, m/s²)

Because pure functions cannot hold internal state, the caller is
responsible for threading the :class:`~robot_sim.types.PIDControllerState`
forward through time.
"""

from __future__ import annotations

import math
from typing import Tuple

from robot_sim.types import (
    ControlInput,
    PIDControllerConfig,
    PIDControllerState,
    PIDGains,
    PIDState,
    TrajectoryPoint,
    VehicleState,
)


# ---------------------------------------------------------------------------
# Single-channel PID update
# ---------------------------------------------------------------------------

def _pid_step(
    gains: PIDGains,
    state: PIDState,
    error: float,
    dt: float,
    output_limit: float,
) -> Tuple[float, PIDState]:
    """Compute one PID step.

    Parameters
    ----------
    gains:
        Proportional / integral / derivative gains.
    state:
        Previous integrator and last-error values.
    error:
        Current error signal.
    dt:
        Timestep  (s).
    output_limit:
        Symmetric saturation limit applied to the output.

    Returns
    -------
    output:
        Saturated PID output.
    new_state:
        Updated :class:`PIDState` to pass to the next call.
    """
    integral = state.integral + error * dt
    derivative = (error - state.prev_error) / dt if dt > 0 else 0.0

    output = gains.kp * error + gains.ki * integral + gains.kd * derivative
    output = max(-output_limit, min(output_limit, output))

    # Anti-windup: clamp integrator if output is saturated.
    if abs(output) >= output_limit:
        integral = state.integral  # don't accumulate further

    new_state = PIDState(integral=integral, prev_error=error)
    return output, new_state


# ---------------------------------------------------------------------------
# Heading error helper
# ---------------------------------------------------------------------------

def _heading_error(desired: float, current: float) -> float:
    """Shortest signed angular error desired − current, wrapped to (−π, π]."""
    err = desired - current
    while err > math.pi:
        err -= 2 * math.pi
    while err < -math.pi:
        err += 2 * math.pi
    return err


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_control(
    desired: TrajectoryPoint,
    current: VehicleState,
    pid_config: PIDControllerConfig,
    pid_state: PIDControllerState,
    dt: float,
    max_omega: float = 1.5,
    max_accel: float = 2.0,
) -> Tuple[ControlInput, PIDControllerState]:
    """Compute a :class:`ControlInput` from the error between *desired* and *current* state.

    Parameters
    ----------
    desired:
        Desired vehicle state at the current time step (from the trajectory).
    current:
        Actual vehicle state.
    pid_config:
        Heading and speed PID gains.
    pid_state:
        Previous PID integrator state.
    dt:
        Simulation timestep  (s).
    max_omega:
        Saturation limit for angular velocity output  (rad/s).
    max_accel:
        Saturation limit for linear acceleration output  (m/s²).

    Returns
    -------
    control:
        :class:`ControlInput` to apply to the dynamics model.
    new_pid_state:
        Updated :class:`PIDControllerState` for the next call.
    """
    # The heading controller steers toward the desired position first.
    # Once close to the desired position, it tracks the desired heading.
    dx = desired.x - current.x
    dy = desired.y - current.y
    dist = math.hypot(dx, dy)

    if dist > 0.1:
        # Point toward the desired position.
        target_heading = math.atan2(dy, dx)
    else:
        # Close enough — track the trajectory heading directly.
        target_heading = desired.theta

    h_err = _heading_error(target_heading, current.theta)
    s_err = desired.v - current.v

    omega, new_h_state = _pid_step(
        pid_config.heading_gains, pid_state.heading, h_err, dt, max_omega
    )
    accel, new_s_state = _pid_step(
        pid_config.speed_gains, pid_state.speed, s_err, dt, max_accel
    )

    new_pid_state = PIDControllerState(heading=new_h_state, speed=new_s_state)
    return ControlInput(a=accel, omega=omega), new_pid_state
