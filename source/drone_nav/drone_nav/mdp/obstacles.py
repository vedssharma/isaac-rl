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


def lidar_horizontal_angles(horizontal_fov_range: tuple[float, float], horizontal_res: float) -> torch.Tensor:
    """Angle (degrees, relative to the drone's heading) of each ray in a channel, in the ray caster's order (as
    Isaac Lab's lidar_pattern: evenly spaced over the field of view, the duplicate end dropped for 360 degrees)."""
    n = lidar_num_horizontal(horizontal_fov_range, horizontal_res)
    fov = horizontal_fov_range[1] - horizontal_fov_range[0]
    full = abs(abs(fov) - 360.0) < 1e-6
    return torch.linspace(horizontal_fov_range[0], horizontal_fov_range[1], n + 1 if full else n)[:n]


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


def clearance_speed_limit(
    min_dist: torch.Tensor, collision_radius: float, cruise: float, near: float, slow_distance: float
) -> torch.Tensor:
    """(n,) per-env speed limit for the over-speed term: `cruise` once the closest obstacle is at least
    slow_distance outside the collision radius, falling linearly to `near` at contact. slow_distance <= 0 turns it
    off (cruise everywhere). Like the proximity term it uses the closest horizontal hit in any direction, so the
    drone also slows when it passes an obstacle side-on, not only when it heads at one."""
    if slow_distance <= 0.0:
        return torch.full_like(min_dist, cruise)
    clearance = (min_dist - collision_radius).clamp(min=0.0)
    return near + (cruise - near) * (clearance / slow_distance).clamp(max=1.0)


def sample_in_box(
    n: int,
    box: tuple[tuple[float, float], tuple[float, float]],
    z_range: tuple[float, float],
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """(n, 3) uniform points: x and y in box = ((xmin, xmax), (ymin, ymax)), relative to the env origin, and an
    absolute altitude in z_range (env origins are on the ground)."""
    # float32 whatever the bounds are: numpy float64 bounds would otherwise make a float64 tensor, which Isaac Lab's
    # float32 buffers refuse.
    lows = torch.tensor([float(box[0][0]), float(box[1][0]), float(z_range[0])], dtype=torch.float32, device=device)
    highs = torch.tensor([float(box[0][1]), float(box[1][1]), float(z_range[1])], dtype=torch.float32, device=device)
    return lows + torch.rand(n, 3, device=device, generator=generator) * (highs - lows)
