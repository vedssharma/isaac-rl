"""The Phase 3 depth student (PLAN.md §4.5) and the pieces of DAgger distillation that don't need Isaac Sim.

The student reads Drone-Nav-Depth-v0's "policy" observation: the goal task's 18 inputs (velocities, gravity, last
action, goal, altitude) followed by a depth image scaled to [0, 1], flattened row-major. A CNN encodes the image to
a 64-d latent; the 18 inputs go through a running normalizer (like RSL-RL's empirical normalization); an MLP maps
both to the 4 CTBR actions. scripts/distill.py trains it to imitate the frozen Phase 2 teacher (DAgger), and
scripts/play.py evaluates its checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import nn

CHECKPOINT_KIND = "depth_student"


class RunningNorm(nn.Module):
    """Normalizes inputs by a running mean and variance over everything seen while `update` is called."""

    def __init__(self, dim: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Merge a batch (n, dim) into the statistics (Chan et al.'s parallel variance)."""
        n = x.shape[0]
        if n == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        total = self.count + n
        w = (n / total).to(x.dtype)
        delta = batch_mean - self.mean
        self.mean += delta * w
        self.var.copy_(self.var * (1 - w) + batch_var * w + delta**2 * w * (1 - w))
        self.count.fill_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + self.eps)


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class DepthStudent(nn.Module):
    """CNN depth encoder + MLP. forward() takes the env's "policy" observation (n, proprio_dim + H * W)."""

    def __init__(
        self,
        proprio_dim: int = 18,
        image_hw: tuple[int, int] = (48, 64),
        latent_dim: int = 64,
        hidden_dims: tuple[int, ...] = (256, 256, 128),
        num_actions: int = 4,
        channels: tuple[int, ...] = (32, 64, 64, 128),
    ):
        super().__init__()
        self.config = {
            "proprio_dim": proprio_dim,
            "image_hw": list(image_hw),
            "latent_dim": latent_dim,
            "hidden_dims": list(hidden_dims),
            "num_actions": num_actions,
            "channels": list(channels),
        }
        self.proprio_dim = proprio_dim
        self.image_hw = tuple(image_hw)
        convs: list[nn.Module] = []
        in_ch = 1
        for i, ch in enumerate(channels):
            k = 5 if i == 0 else 3
            convs += [nn.Conv2d(in_ch, ch, k, stride=2, padding=k // 2), nn.ELU()]
            in_ch = ch
        self.cnn = nn.Sequential(*convs, nn.Flatten())
        with torch.no_grad():
            flat = self.cnn(torch.zeros(1, 1, *self.image_hw)).shape[1]
        self.encoder = nn.Sequential(self.cnn, nn.Linear(flat, latent_dim), nn.ELU())
        self.normalizer = RunningNorm(proprio_dim)
        self.mlp = _mlp([proprio_dim + latent_dim, *hidden_dims, num_actions])

    @property
    def obs_dim(self) -> int:
        return self.proprio_dim + self.image_hw[0] * self.image_hw[1]

    def split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(proprio (n, P), image (n, 1, H, W)) from the flat observation."""
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"expected {self.obs_dim} inputs ({self.proprio_dim} + image), got {obs.shape[-1]}")
        return obs[:, : self.proprio_dim], obs[:, self.proprio_dim :].reshape(-1, 1, *self.image_hw)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio, image = self.split(obs)
        return self.mlp(torch.cat([self.normalizer(proprio), self.encoder(image)], dim=-1))

    def update_normalizer(self, obs: torch.Tensor) -> None:
        self.normalizer.update(obs[:, : self.proprio_dim])

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """No memory yet (a GRU comes later); for play.py's loop, which resets recurrent policies."""


def checkpoint(student: DepthStudent, iteration: int, extra: dict | None = None) -> dict:
    return {
        "kind": CHECKPOINT_KIND,
        "config": dict(student.config),
        "model_state_dict": student.state_dict(),
        "iter": iteration,
        **(extra or {}),
    }


def is_student_checkpoint(ckpt: dict) -> bool:
    return isinstance(ckpt, dict) and ckpt.get("kind") == CHECKPOINT_KIND


def load_student(ckpt: dict, device: str | torch.device = "cpu") -> DepthStudent:
    if not is_student_checkpoint(ckpt):
        raise ValueError("not a depth-student checkpoint")
    cfg = ckpt["config"]
    student = DepthStudent(
        proprio_dim=cfg["proprio_dim"],
        image_hw=tuple(cfg["image_hw"]),
        latent_dim=cfg["latent_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        num_actions=cfg["num_actions"],
        channels=tuple(cfg["channels"]),
    )
    student.load_state_dict(ckpt["model_state_dict"])
    return student.to(device).eval()


# -- DAgger


def teacher_fraction(iteration: int, decay_iterations: int) -> float:
    """beta: the share of actions the teacher executes at this iteration, falling linearly from 1 to 0 over
    decay_iterations (0: the student acts from the start)."""
    if decay_iterations <= 0:
        return 0.0
    return max(0.0, 1.0 - iteration / decay_iterations)


def mix_actions(
    teacher: torch.Tensor, student: torch.Tensor, beta: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Each env executes the teacher's action with probability beta, else the student's."""
    if beta <= 0.0:
        return student
    if beta >= 1.0:
        return teacher
    pick = torch.rand(teacher.shape[0], generator=generator, device=teacher.device) < beta
    return torch.where(pick[:, None], teacher, student)


def minibatches(
    n: int, num_minibatches: int, device: str | torch.device, generator: torch.Generator | None = None
) -> Iterator[torch.Tensor]:
    """Index tensors covering a random permutation of range(n) in num_minibatches nearly equal parts."""
    perm = torch.randperm(n, generator=generator, device="cpu").to(device)
    yield from torch.tensor_split(perm, num_minibatches)


def imitation_loss(student_actions: torch.Tensor, teacher_actions: torch.Tensor) -> torch.Tensor:
    """MSE to the teacher's actions as the env would execute them (clamped to [-1, 1], like the env's actions)."""
    return torch.mean((student_actions - teacher_actions.clamp(-1.0, 1.0)) ** 2)
