"""CPU tests for the CTBR body-rate controller (drone_nav/control/rate_controller.py)."""

import pytest
import torch
from drone_nav.control.rate_controller import rate_command, rate_controller_torque

INERTIA = torch.tensor([1.4e-5, 1.4e-5, 2.2e-5])  # roughly a Crazyflie 2.x, kg m^2
KP = (25.0, 25.0, 15.0)
MAX_TORQUE = 0.01  # N m, the thrust + torque action's authority (moment_scale)


def test_rate_command_scales_and_clamps():
    cmd = rate_command(torch.tensor([[1.0, -0.5, 2.0]]), (6.0, 6.0, 3.0))
    torch.testing.assert_close(cmd, torch.tensor([[6.0, -3.0, 3.0]]))  # 2.0 clamped to 1.0


def test_no_torque_at_the_setpoint_and_sign_follows_the_error():
    w = torch.tensor([[1.0, -2.0, 0.5]])
    torch.testing.assert_close(rate_controller_torque(w, w, INERTIA, KP, MAX_TORQUE), torch.zeros(1, 3))
    tau = rate_controller_torque(torch.zeros(1, 3), w, INERTIA, KP, MAX_TORQUE)
    assert torch.all(torch.sign(tau) == -torch.sign(w))


def test_torque_is_clamped():
    tau = rate_controller_torque(torch.tensor([[1e6, -1e6, 0.0]]), torch.zeros(1, 3), INERTIA, KP, MAX_TORQUE)
    torch.testing.assert_close(tau, torch.tensor([[MAX_TORQUE, -MAX_TORQUE, 0.0]]))


@pytest.mark.parametrize("inertia_scale", [1.0, 1.2, 0.8])
def test_closed_loop_tracks_a_step_without_overshoot(inertia_scale):
    """Integrate omega' = tau / I at the env's 100 Hz physics rate. The loop must settle within 0.25 s with no
    overshoot, also when the true inertia differs from the one the controller assumes (model mismatch)."""
    dt, cmd = 0.01, torch.tensor([[3.0, -3.0, 1.5]])
    true_inertia = INERTIA * inertia_scale
    w = torch.zeros(1, 3)
    peak = torch.zeros(1, 3)
    for _ in range(25):
        tau = rate_controller_torque(cmd, w, INERTIA, KP, MAX_TORQUE)
        w = w + tau / true_inertia * dt
        peak = torch.maximum(peak, w.abs())
    assert torch.all((w - cmd).abs() < 0.05 * cmd.abs())
    assert torch.all(peak <= cmd.abs() * 1.001)
