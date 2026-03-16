"""
dynamics.py — unicycle-model dynamics (pure functions).

State  : (x, y, θ, v)
Input  : (a, ω)  — linear acceleration and angular velocity

Continuous equations of motion
    ẋ = v · cos(θ)
    ẏ = v · sin(θ)
    θ̇ = ω
    v̇ = a

Integration is performed with a simple forward-Euler step.
"""

from __future__ import annotations

import math

from robot_sim.types import ControlInput, VehicleState


def step(
    state: VehicleState,
    control: ControlInput,
    dt: float,
    max_speed: float = 3.0,
    max_accel: float = 2.0,
    max_omega: float = 1.5,
) -> VehicleState:
    """Advance the vehicle state by one timestep using forward Euler integration.

    Parameters
    ----------
    state:
        Current vehicle state ``(x, y, θ, v)``.
    control:
        Control input ``(a, ω)`` to apply.
    dt:
        Integration timestep  (s).
    max_speed:
        Speed is clamped to ``[−max_speed, max_speed]``.
    max_accel:
        Acceleration magnitude is clamped before integration.
    max_omega:
        Angular-velocity magnitude is clamped before integration.

    Returns
    -------
    VehicleState
        New vehicle state after the timestep.
    """
    # Saturate inputs.
    a = max(-max_accel, min(max_accel, control.a))
    omega = max(-max_omega, min(max_omega, control.omega))

    # Integrate velocity.
    v_new = state.v + a * dt
    v_new = max(-max_speed, min(max_speed, v_new))

    # Use average speed for position integration (trapezoidal rule for v).
    v_avg = 0.5 * (state.v + v_new)

    # Integrate heading and position.
    theta_new = state.theta + omega * dt
    x_new = state.x + v_avg * math.cos(state.theta) * dt
    y_new = state.y + v_avg * math.sin(state.theta) * dt

    return VehicleState(x=x_new, y=y_new, theta=theta_new, v=v_new)
