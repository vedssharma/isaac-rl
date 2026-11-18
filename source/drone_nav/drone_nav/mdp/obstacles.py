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


def yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """(n,) heading (radians) of (n, 4) quaternions in Isaac Lab's (w, x, y, z) order."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_rotate(directions: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """(n, rays, 3): the pattern's ray directions (rays, 3) turned about the vertical axis by each env's heading
    (n,). Used to make the lidar ring follow the drone's yaw (Isaac Lab's "yaw" alignment turns only the rays'
    start points, not their directions)."""
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    x, y, z = directions[None, :, 0], directions[None, :, 1], directions[None, :, 2]
    return torch.stack([c * x - s * y, s * x + c * y, z.expand_as(c * x)], dim=-1)


def footprint_distances(
    point: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    half_extents: torch.Tensor,
    yaws: torch.Tensor,
    is_round: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact distance from each env's point (n, 2) to each of its m obstacle footprints, and the footprint's nearest
    point: (n, m) and (n, m, 2). Footprints are circles (is_round: centers, radii) or rotated rectangles (centers,
    half_extents (n, m, 2), yaws). 0 inside a footprint; inf where valid is False (padding)."""
    rel = point[:, None, :] - centers  # (n, m, 2)
    # Circles.
    norm = rel.norm(dim=-1).clamp(min=1e-9)
    d_round = (norm - radii).clamp(min=0.0)
    q_round = centers + rel / norm[..., None] * radii[..., None]
    q_round = torch.where((norm > radii)[..., None], q_round, point[:, None, :].expand_as(q_round))
    # Rectangles: into the box frame, clamp to the box, back out.
    c, s = torch.cos(yaws), torch.sin(yaws)
    local = torch.stack([c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], dim=-1)
    clamped = torch.maximum(torch.minimum(local, half_extents), -half_extents)
    d_box = (local - clamped).norm(dim=-1)
    q_box = centers + torch.stack(
        [c * clamped[..., 0] - s * clamped[..., 1], s * clamped[..., 0] + c * clamped[..., 1]], dim=-1
    )
    dist = torch.where(is_round, d_round, d_box)
    nearest = torch.where(is_round[..., None], q_round, q_box)
    return torch.where(valid, dist, torch.full_like(dist, float("inf"))), nearest


def nearest_obstacle_observation(
    point: torch.Tensor,
    yaw: torch.Tensor,
    dist: torch.Tensor,
    nearest: torch.Tensor,
    k: int,
    max_range: float,
) -> torch.Tensor:
    """(n, 3k) privileged observation of the k closest footprints (PLAN.md §4.2's "K nearest obstacles"): for each,
    the unit direction to its nearest point in the drone's body-yaw frame (2), then distance / max_range clipped to 1
    (1). Closest first. A footprint beyond max_range (or padding) reads as direction (0, 0), distance 1."""
    d, idx = torch.topk(dist, k, dim=1, largest=False)
    q = torch.gather(nearest, 1, idx[..., None].expand(-1, -1, 2))
    vec = q - point[:, None, :]
    c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    body = torch.stack([c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]], dim=-1)
    unit = body / body.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    far = ~(d < max_range)  # also catches inf (padding)
    unit = torch.where(far[..., None], torch.zeros_like(unit), unit)
    scaled = torch.where(far, torch.ones_like(d), d / max_range)
    return torch.cat([unit, scaled[..., None]], dim=-1).reshape(point.shape[0], 3 * k)


def closest_by_kind(dist: torch.Tensor, kinds: torch.Tensor, num_kinds: int) -> torch.Tensor:
    """(n, num_kinds) each env's distance to its closest footprint of each kind, from footprint_distances' (n, m)
    distances and the footprints' kind indices (n, m) (-1 for padding). inf where an env's tile has none."""
    match = kinds[..., None] == torch.arange(num_kinds, device=kinds.device)  # (n, m, k)
    return torch.where(match, dist[..., None], torch.full_like(match, float("inf"), dtype=dist.dtype)).amin(dim=1)
