"""Goal-reaching MDP pieces for Drone-Nav-Goal-v0, in plain torch (no Isaac Lab), so they run in CPU tests.

PLAN.md §4: goal sampling (§4.4), the goal part of the observation (§4.2), terminations (§4.4) and the
reward terms of §4.3 without the obstacle term. The env in tasks/goal/goal_env.py feeds them Isaac Lab state.
"""

from __future__ import annotations

import math

import torch

# Reward terms, in the order they are logged (Episode_Reward/<name>).
REWARD_TERMS = ("progress", "goal", "crash", "smooth", "ang_rate", "upright", "time")
# Why an episode ended (Episode_Termination/<name>). "goal" and the crash reasons are terminal; "time_out"
# is a truncation, so PPO bootstraps through it instead of treating it as a death.
CRASH_REASONS = ("too_low", "too_high", "flipped", "off_course")


def sample_goal_offsets(
    n: int,
    min_dist: float,
    max_dist: float,
    z_range: tuple[float, float],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(n, 3) goal positions relative to the env origin: a uniformly random heading, a horizontal distance in
    [min_dist, max_dist] and an altitude in z_range (absolute, since env origins are on the ground)."""
    heading = torch.rand(n, device=device, generator=generator) * (2 * math.pi)
    dist = min_dist + torch.rand(n, device=device, generator=generator) * (max_dist - min_dist)
    z = z_range[0] + torch.rand(n, device=device, generator=generator) * (z_range[1] - z_range[0])
    return torch.stack([dist * torch.cos(heading), dist * torch.sin(heading), z], dim=-1)


def goal_observation(goal_pos_b: torch.Tensor, dist_clip: float) -> torch.Tensor:
    """(n, 4): unit direction to the goal in the body frame, then distance clipped to dist_clip and scaled to
    [0, 1]. Splitting direction from distance keeps the inputs bounded for goals 20 m away (PLAN.md §4.2)."""
    dist = goal_pos_b.norm(dim=-1, keepdim=True)
    direction = goal_pos_b / dist.clamp(min=1e-6)
    return torch.cat([direction, dist.clamp(max=dist_clip) / dist_clip], dim=-1)


def up_z(projected_gravity_b: torch.Tensor) -> torch.Tensor:
    """z component of the body's up axis in the world frame: 1 when level, < 0 when upside down."""
    return -projected_gravity_b[:, 2]


def compute_terminations(
    dist_to_goal: torch.Tensor,
    start_dist_to_goal: torch.Tensor,
    altitude: torch.Tensor,
    upz: torch.Tensor,
    *,
    goal_radius: float,
    min_altitude: float,
    max_altitude: float,
    off_course_margin: float,
) -> dict[str, torch.Tensor]:
    """Boolean (n,) tensors: "goal" plus one per crash reason. A crash takes precedence over reaching the goal."""
    crashes = {
        "too_low": altitude < min_altitude,
        "too_high": altitude > max_altitude,
        "flipped": upz < 0.0,
        # Flying away from the goal (e.g. a policy that learned to leave the map) ends the episode.
        "off_course": dist_to_goal > start_dist_to_goal + off_course_margin,
    }
    crashed = torch.stack(list(crashes.values())).any(dim=0)
    return {"goal": (dist_to_goal < goal_radius) & ~crashed, **crashes}


def compute_reward_terms(
    *,
    prev_dist: torch.Tensor,
    dist: torch.Tensor,
    reached: torch.Tensor,
    crashed: torch.Tensor,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    ang_vel_b: torch.Tensor,
    upz: torch.Tensor,
    step_dt: float,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """The §4.3 terms for one env step, already weighted and signed (sum them for the reward).

    Progress is in metres closed this step, and the sparse terms are per event, so neither is scaled by
    step_dt. The rest are rates (per second), so they are multiplied by step_dt and an episode's total doesn't
    depend on the control frequency.
    """
    return {
        "progress": weights["progress"] * (prev_dist - dist),
        "goal": weights["goal"] * reached.float(),
        "crash": -weights["crash"] * crashed.float(),
        "smooth": -weights["smooth"] * (action - prev_action).square().sum(dim=-1) * step_dt,
        "ang_rate": -weights["ang_rate"] * ang_vel_b.square().sum(dim=-1) * step_dt,
        "upright": -weights["upright"] * (1.0 - upz) * step_dt,
        "time": -weights["time"] * torch.ones_like(dist) * step_dt,
    }
