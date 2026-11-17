"""CPU tests for Drone-Nav-Obstacles-v0's MDP math (drone_nav/mdp/obstacles.py)."""

import math

import pytest
import torch
from drone_nav.mdp import obstacles as mdp

# The env's default ring (PLAN.md §4.2): 3 channels at -15 / 0 / +15 degrees, 36 rays each, 5 m range.
CHANNELS, VFOV, HFOV, HRES, RANGE = 3, (-15.0, 15.0), (-180.0, 180.0), 10.0, 5.0
COLLISION_RADIUS = 0.1


def test_ray_counts_match_isaac_lab_lidar_pattern():
    assert mdp.lidar_num_horizontal(HFOV, HRES) == 36  # 360 degrees: the duplicate end ray is dropped
    assert mdp.lidar_num_horizontal((-90.0, 90.0), 5.0) == 37  # a partial fan keeps both ends
    assert mdp.lidar_num_horizontal((0.0, 360.0), 7.0) == 52  # ceil(360 / 7) + 1 - 1


def test_horizontal_channel():
    assert mdp.horizontal_channel(3, (-15.0, 15.0)) == 1
    assert mdp.horizontal_channel(1, (0.0, 0.0)) == 0
    assert mdp.horizontal_channel(4, (-30.0, 0.0)) == 3
    with pytest.raises(ValueError):
        mdp.horizontal_channel(2, (-15.0, 15.0))  # -15 and +15 only


def test_ray_distances_clip_misses_to_max_range():
    origin = torch.tensor([[1.0, 2.0, 1.0]])
    hits = torch.tensor([[[4.0, 6.0, 1.0], [float("inf")] * 3, [1.0, 2.0, -9.0]]])
    dist = mdp.ray_distances(hits, origin, RANGE)
    torch.testing.assert_close(dist, torch.tensor([[5.0, RANGE, RANGE]]))
    torch.testing.assert_close(mdp.lidar_observation(dist, RANGE), torch.tensor([[1.0, 1.0, 1.0]]))


def test_min_horizontal_distance_uses_only_the_horizontal_channel():
    n_h = mdp.lidar_num_horizontal(HFOV, HRES)
    dist = torch.full((2, CHANNELS * n_h), RANGE)
    dist[0, 0 * n_h + 3] = 0.05  # the -15 degree channel (e.g. the ground below): ignored
    dist[0, 1 * n_h + 7] = 0.8
    dist[1, 1 * n_h + 0] = 0.3
    torch.testing.assert_close(mdp.min_horizontal_distance(dist, CHANNELS, 1), torch.tensor([0.8, 0.3]))


def ring_distances_to_pole(center_dist: float, bearing: float, radius: float) -> torch.Tensor:
    """Exact horizontal-ring distances from a drone at the origin to one vertical pole (a circle), as the ray
    caster would report them on the horizontal channel."""
    n_h = mdp.lidar_num_horizontal(HFOV, HRES)
    angles = torch.deg2rad(torch.linspace(HFOV[0], HFOV[1], n_h + 1)[:-1]).double()
    u = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    c = torch.tensor([math.cos(bearing), math.sin(bearing)], dtype=torch.double) * center_dist
    along = u @ c
    perp2 = (c * c).sum() - along**2
    hit = (along > 0) & (perp2 <= radius**2)
    t = along - torch.sqrt((radius**2 - perp2).clamp(min=0))
    return torch.where(hit, t, torch.full_like(t, RANGE)).clamp(max=RANGE).float()


@pytest.mark.parametrize("radius", [0.03, 0.06, 0.5])  # thinnest pole, thickest pole, a big cylinder
def test_ten_degree_ring_never_misses_a_collision(radius):
    """At every bearing, an obstacle whose surface is inside the collision radius is seen by some ray of the 10
    degree ring, closer than the collision radius. (Far away a thin pole can fall between rays; up close it can't.)
    And a ray never reports less than the true distance, so nothing outside the radius counts as a collision."""
    for bearing in torch.linspace(0, 2 * math.pi, 721)[:-1].tolist():
        for surface in (0.02, 0.06, 0.095):
            ring = ring_distances_to_pole(surface + radius, bearing, radius)
            assert mdp.collided(ring.min().unsqueeze(0), COLLISION_RADIUS).item(), (bearing, surface)
        ring = ring_distances_to_pole(0.15 + radius, bearing, radius)
        assert ring.min() >= 0.15 - 1e-5
        assert not mdp.collided(ring.min().unsqueeze(0), COLLISION_RADIUS).item()


def test_proximity_penalty():
    d = torch.tensor([0.05, COLLISION_RADIUS, COLLISION_RADIUS + 0.3, RANGE])
    p = mdp.proximity_penalty(d, COLLISION_RADIUS, sigma=0.3, weight=1.0, step_dt=0.02)
    # Full weight per second at (or inside) the collision radius, e^-1 one sigma further out, ~0 far away.
    torch.testing.assert_close(p[:3], torch.tensor([-0.02, -0.02, -0.02 * math.exp(-1.0)]))
    assert -1e-8 < p[3] <= 0.0
    assert torch.all(p[1:] - p[:-1] >= 0)  # never cheaper to be closer


def test_clearance_speed_limit():
    cruise, near, slow = 3.0, 1.5, 1.0
    d = torch.tensor([0.05, COLLISION_RADIUS, COLLISION_RADIUS + 0.5, COLLISION_RADIUS + slow, RANGE])
    limit = mdp.clearance_speed_limit(d, COLLISION_RADIUS, cruise, near, slow)
    # near_speed_limit at (or inside) contact, halfway at half slow_distance, cruise from slow_distance out.
    torch.testing.assert_close(limit, torch.tensor([1.5, 1.5, 2.25, 3.0, 3.0]))
    assert torch.all(limit[1:] - limit[:-1] >= 0)  # never a higher limit closer in


def test_clearance_speed_limit_off():
    d = torch.tensor([0.05, 0.5, RANGE])
    limit = mdp.clearance_speed_limit(d, COLLISION_RADIUS, 3.0, 1.5, slow_distance=0.0)
    torch.testing.assert_close(limit, torch.full((3,), 3.0))


def test_sample_in_box():
    g = torch.Generator().manual_seed(0)
    pts = mdp.sample_in_box(5000, ((-0.9, 0.9), (-4.7, 4.7)), (1.0, 1.5), generator=g)
    assert pts.shape == (5000, 3)
    for axis, (lo, hi) in enumerate([(-0.9, 0.9), (-4.7, 4.7), (1.0, 1.5)]):
        assert pts[:, axis].min() >= lo and pts[:, axis].max() <= hi
        assert pts[:, axis].max() - pts[:, axis].min() > 0.95 * (hi - lo)  # covers the box


def test_sample_in_box_is_float32_even_from_numpy_bounds():
    """The env's strip bounds come from numpy; Isaac Lab's buffers are float32 and refuse float64 writes."""
    import numpy as np

    box = ((np.float64(-0.5), np.float64(0.5)), (np.float64(-4.3), np.float64(4.3)))
    assert mdp.sample_in_box(4, box, (np.float64(1.0), 1.5)).dtype == torch.float32


def test_lidar_horizontal_angles_match_the_pattern():
    a = mdp.lidar_horizontal_angles(HFOV, HRES)
    assert len(a) == mdp.lidar_num_horizontal(HFOV, HRES) == 36
    assert a[0] == -180.0 and a[-1] == pytest.approx(170.0)
    assert len(mdp.lidar_horizontal_angles(HFOV, 5.0)) == 72
    b = mdp.lidar_horizontal_angles((-90.0, 90.0), 5.0)
    assert len(b) == 37 and b[-1] == 90.0  # a partial fan keeps both ends
