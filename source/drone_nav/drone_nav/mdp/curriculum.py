"""Per-env difficulty curriculum for Drone-Nav-Obstacles-v0 (PLAN.md §5.3), in plain torch so it runs in CPU tests.

Each env remembers whether its last `window` episodes reached the goal. Once the window is full, the env moves up a
difficulty level if at least `up_rate` of them succeeded, and down a level if fewer than `down_rate` did. Moving
clears the window, so the next decision is about the new level only. Time-outs and crashes both count as failures.

Isaac Lab's TerrainImporter.update_env_origins sends an env that passes the top level to a random level. Here it
stays at the top instead: Phase 2 is judged on the hard levels, and the per-level fractions stay readable. Moving
down still catches an env whose skills slip.
"""

from __future__ import annotations

import torch


class SuccessCurriculum:
    def __init__(
        self,
        num_envs: int,
        num_levels: int,
        window: int,
        up_rate: float,
        down_rate: float,
        device: torch.device | str = "cpu",
    ):
        if not 0.0 <= down_rate < up_rate <= 1.0:
            raise ValueError(f"need 0 <= down_rate < up_rate <= 1, got down {down_rate}, up {up_rate}")
        if window < 1:
            raise ValueError(f"window must be at least 1 episode, got {window}")
        self.num_levels = num_levels
        self.window = window
        self.up_rate = up_rate
        self.down_rate = down_rate
        self._history = torch.zeros(num_envs, window, dtype=torch.bool, device=device)  # ring buffer of outcomes
        self._count = torch.zeros(num_envs, dtype=torch.long, device=device)  # episodes recorded since last move

    def update(self, env_ids: torch.Tensor, success: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        """Record one finished episode for each of env_ids (success: bool, same length) and return their new levels
        (levels: their current ones). Envs that move start a fresh window."""
        slot = self._count[env_ids] % self.window
        self._history[env_ids, slot] = success
        self._count[env_ids] += 1

        full = self._count[env_ids] >= self.window
        rate = self._history[env_ids].float().mean(dim=1)
        up = full & (rate >= self.up_rate) & (levels < self.num_levels - 1)
        down = full & (rate < self.down_rate) & (levels > 0)
        new_levels = levels + up.long() - down.long()

        moved = env_ids[up | down]
        self._count[moved] = 0
        self._history[moved] = False
        return new_levels


def level_fractions(levels: torch.Tensor, num_levels: int) -> torch.Tensor:
    """(num_levels,) fraction of envs on each level: the difficulty histogram PLAN.md §5.3 asks to log."""
    return torch.bincount(levels, minlength=num_levels).float() / levels.numel()
