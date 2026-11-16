"""Obstacle MDP pieces for Drone-Nav-Obstacles-v0, in plain torch (no Isaac Lab), so they run in CPU tests.

PLAN.md Phase 2: the privileged ray-cast "lidar ring" observation (§4.2), collision termination (§4.4), the
proximity reward term (§4.3), and spawn / goal sampling inside the obstacle terrain's start and goal strips. The env
in tasks/obstacles/obstacles_env.py feeds them Isaac Lab's RayCaster data.

Obstacles in the terrain are vertical prisms taller than the flight ceiling (drone_nav/terrain/obstacle_field.py),
so the horizontal channel of the ring sees every obstacle at the drone's own altitude. Collision and clearance use
that channel; the tilted channels add the ground and the obstacles' extent for the observation.
"""

from __future__ import annotations

import math

import torch


def lidar_num_horizontal(horizontal_fov_range: tuple[float, float], horizontal_res: float) -> int:
    """Rays per channel in Isaac Lab's LidarPatternCfg (isaaclab v2.3.2, patterns.lidar_pattern): ceil(fov / res)
    + 1 angles, minus the duplicate end point when the field of view is a full 360 degrees."""
    fov = horizontal_fov_range[1] - horizontal_fov_range[0]
    n = math.ceil(fov / horizontal_res) + 1
    return n - 1 if abs(abs(fov) - 360.0) < 1e-6 else n


def lidar_channel_angles(channels: int, vertical_fov_range: tuple[float, float]) -> torch.Tensor:
    """Elevation (degrees) of each channel, in the order the ray caster returns them (as lidar_pattern)."""
    return torch.linspace(vertical_fov_range[0], vertical_fov_range[1], channels)


def horizontal_channel(channels: int, vertical_fov_range: tuple[float, float]) -> int:
    """Index of the 0-degree channel. The env needs one: collisions and clearance are measured on it."""
    angles = lidar_channel_angles(channels, vertical_fov_range)
    idx = int(angles.abs().argmin())
    if abs(float(angles[idx])) > 1e-6:
        raise ValueError(f"the lidar needs a 0-degree channel; its channels are at {angles.tolist()} degrees")
    return idx


def ray_distances(hits_w: torch.Tensor, origin_w: torch.Tensor, max_range: float) -> torch.Tensor:
    """(n, rays) distance from the sensor to each hit, clipped to max_range. The ray caster reports a miss (nothing
    within max_distance) as an inf hit position; that becomes max_range."""
    dist = torch.linalg.norm(hits_w - origin_w.unsqueeze(1), dim=-1)
    return torch.nan_to_num(dist, nan=max_range, posinf=max_range).clamp(max=max_range)


def lidar_observation(dist: torch.Tensor, max_range: float) -> torch.Tensor:
    """(n, rays) in [0, 1]: 0 is touching, 1 is nothing within max_range."""
    return dist / max_range


def min_horizontal_distance(dist: torch.Tensor, channels: int, channel: int) -> torch.Tensor:
    """(n,) the closest hit on the horizontal channel. Rays come channel by channel (lidar_pattern's meshgrid)."""
    return dist.view(dist.shape[0], channels, -1)[:, channel].min(dim=-1).values


def collided(min_dist: torch.Tensor, collision_radius: float) -> torch.Tensor:
    """(n,) True when an obstacle is within collision_radius of the drone's centre, horizontally."""
    return min_dist < collision_radius


def proximity_penalty(
    min_dist: torch.Tensor, collision_radius: float, sigma: float, weight: float, step_dt: float
) -> torch.Tensor:
    """PLAN.md §4.3 soft clearance shaping, -w exp(-clearance / sigma) per second, where clearance is the gap left
    between the drone's collision radius and the closest obstacle. It is -w per second at the point of collision and
    falls off over a few sigma. Uses only the closest hit, so a big obstacle and a thin pole at the same distance
    cost the same."""
    clearance = (min_dist - collision_radius).clamp(min=0.0)
    return -weight * torch.exp(-clearance / sigma) * step_dt


def sample_in_box(
    n: int,
    box: tuple[tuple[float, float], tuple[float, float]],
    z_range: tuple[float, float],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(n, 3) uniform points: x and y in box = ((xmin, xmax), (ymin, ymax)), relative to the env origin, and an
    absolute altitude in z_range (env origins are on the ground)."""
    lows = torch.tensor([box[0][0], box[1][0], z_range[0]], device=device)
    highs = torch.tensor([box[0][1], box[1][1], z_range[1]], device=device)
    return lows + torch.rand(n, 3, device=device, generator=generator) * (highs - lows)
