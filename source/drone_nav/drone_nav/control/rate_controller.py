"""Body-rate controller for collective-thrust + body-rate (CTBR) actions (PLAN.md §4.1 v2). Plain torch.

The policy commands body rates; this proportional loop turns the rate error into body torques, every physics
substep. Scaling the gain by the inertia makes it a first-order response with time constant 1/kp, whatever the
airframe, and clamping the torque keeps the same control authority as the thrust + torque action.
"""

from __future__ import annotations

import torch


def rate_command(actions: torch.Tensor, max_rates: tuple[float, float, float]) -> torch.Tensor:
    """(n, 3) body-rate setpoints (rad/s) from the policy's rate actions in [-1, 1]."""
    return actions.clamp(-1.0, 1.0) * torch.tensor(max_rates, dtype=actions.dtype, device=actions.device)


def rate_controller_torque(
    rate_cmd: torch.Tensor,
    ang_vel_b: torch.Tensor,
    inertia_diag: torch.Tensor,
    kp: tuple[float, float, float],
    max_torque: float,
) -> torch.Tensor:
    """(n, 3) body torques: tau = I kp (omega_cmd - omega), each axis clamped to +-max_torque.

    inertia_diag is (3,) or (n, 3), the body's principal inertia (kg m^2). The gyroscopic term omega x I omega
    is left out: for a 30 g quadrotor at a few rad/s it is orders of magnitude below the control torque.
    """
    gains = torch.tensor(kp, dtype=rate_cmd.dtype, device=rate_cmd.device)
    torque = inertia_diag * gains * (rate_cmd - ang_vel_b)
    return torque.clamp(-max_torque, max_torque)
