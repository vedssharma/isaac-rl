"""Domain randomization for Phase 4 (PLAN.md §5.5): pure-torch pieces the goal task (and every task built on it)
uses when env.randomization.* is on. Everything is drawn per env at each reset; with every setting at 0 the task is
exactly the nominal one and draws no extra random numbers, so earlier runs reproduce.
"""

from __future__ import annotations

import math

import torch

# Noise std of each of the goal task's 18 observation inputs at obs_noise_scale = 1: body linear velocity (m/s),
# body angular velocity (rad/s), projected gravity, last action (exact: the policy's own command), goal direction
# (unit vector), goal distance (m) and altitude (m). Roughly a small drone's state estimator.
OBS_NOISE_STD = (0.1,) * 3 + (0.05,) * 3 + (0.02,) * 3 + (0.0,) * 4 + (0.02,) * 3 + (0.1, 0.05)


def sample_scale(n: int, spread: float, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """(n,) factors uniform in [1 - spread, 1 + spread]."""
    u = torch.rand(n, device=device, generator=generator)
    return 1.0 + (2.0 * u - 1.0) * spread


def lag_alpha(step_dt: float, tau: torch.Tensor) -> torch.Tensor:
    """Per-step blend of a first-order lag with time constant tau (s): 1 - exp(-dt / tau); 1 (no lag) where tau = 0."""
    safe = tau.clamp(min=1e-9)
    return torch.where(tau > 0, 1.0 - torch.exp(-step_dt / safe), torch.ones_like(tau))


def first_order_lag(actual: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """One step of actual -> target with per-env blend alpha (n,), for (n, ...) tensors."""
    a = alpha.reshape(-1, *([1] * (target.dim() - 1)))
    return actual + a * (target - actual)


def ou_step(x: torch.Tensor, step_dt: float, tau: float, std: float, noise: torch.Tensor) -> torch.Tensor:
    """One Euler step of an Ornstein-Uhlenbeck process with correlation time tau and stationary std `std`."""
    return x - x * (step_dt / tau) + std * math.sqrt(2.0 * step_dt / tau) * noise


def horizontal_wind(n: int, speed_max: float, device, generator: torch.Generator | None = None) -> torch.Tensor:
    """(n, 3) steady wind in the world frame: a random horizontal direction, speed uniform in [0, speed_max]."""
    angle = torch.rand(n, device=device, generator=generator) * (2.0 * math.pi)
    speed = torch.rand(n, device=device, generator=generator) * speed_max
    return torch.stack([speed * torch.cos(angle), speed * torch.sin(angle), torch.zeros_like(speed)], dim=-1)


def world_to_body(v_w: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(n, 3) world-frame vectors expressed in the body frame of (n, 4) quaternions (w, x, y, z): q^-1 v q."""
    w, xyz = quat_wxyz[:, :1], -quat_wxyz[:, 1:]  # the conjugate rotates world -> body
    t = 2.0 * torch.cross(xyz, v_w, dim=-1)
    return v_w + w * t + torch.cross(xyz, t, dim=-1)


class ActionDelay:
    """Per-env control latency: actions come out `steps` control steps after they go in (0 = no delay)."""

    def __init__(self, n: int, dim: int, max_steps: int, device):
        self.size = max_steps + 1
        self.buf = torch.zeros(self.size, n, dim, device=device)
        self.ptr = 0

    def push(self, actions: torch.Tensor) -> None:
        self.ptr = (self.ptr + 1) % self.size
        self.buf[self.ptr] = actions

    def delayed(self, steps: torch.Tensor) -> torch.Tensor:
        """(n, dim): each env's action from steps[env] pushes ago (the latest push is 0)."""
        idx = (self.ptr - steps) % self.size
        return self.buf[idx, torch.arange(self.buf.shape[1], device=self.buf.device)]

    def reset(self, env_ids: torch.Tensor, value: float = 0.0) -> None:
        self.buf[:, env_ids] = value
