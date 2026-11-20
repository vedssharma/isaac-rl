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


def quat_from_yaw(yaw):
    """(w, x, y, z) for a pure heading."""
    return torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], -1)


def test_yaw_from_quat():
    yaws = torch.tensor([0.0, 0.5, -2.0, 3.0])
    torch.testing.assert_close(mdp.yaw_from_quat(quat_from_yaw(yaws)), yaws)
    # A pitched and rolled drone keeps its heading: compose yaw 0.7 with a 20-degree pitch about the body y axis.
    pitch = math.radians(20.0)
    q_pitch = torch.tensor([[math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]])
    q_yaw = quat_from_yaw(torch.tensor([0.7]))
    w1, x1, y1, z1 = q_yaw[0]
    w2, x2, y2, z2 = q_pitch[0]
    q = torch.tensor([[w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                       w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]])  # fmt: skip
    torch.testing.assert_close(mdp.yaw_from_quat(q), torch.tensor([0.7]))


def test_yaw_rotate_turns_each_ray_by_the_heading():
    """With the ring following yaw, the ray at pattern angle theta on a drone with heading psi points at world
    bearing theta + psi, and the channel's elevation (z) is unchanged."""
    theta = torch.deg2rad(mdp.lidar_horizontal_angles(HFOV, HRES))
    elev = math.radians(15.0)
    dirs = torch.stack(
        [math.cos(elev) * torch.cos(theta), math.cos(elev) * torch.sin(theta), torch.full_like(theta, math.sin(elev))],
        dim=-1,
    )
    yaw = torch.tensor([0.0, 0.8, -2.5])
    out = mdp.yaw_rotate(dirs, yaw)
    assert out.shape == (3, len(theta), 3)
    bearing = torch.atan2(out[..., 1], out[..., 0])
    expected = torch.atan2(torch.sin(theta[None] + yaw[:, None]), torch.cos(theta[None] + yaw[:, None]))
    torch.testing.assert_close(bearing, expected)
    torch.testing.assert_close(out[..., 2], dirs[None, :, 2].expand(3, -1))
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(3, len(theta)))


def test_footprint_distances_match_the_numpy_geometry():
    """The GPU distances must agree with obstacle_field's own footprint geometry, on real layouts (cylinders, poles,
    rotated boxes, walls and side walls) at random points, including points inside footprints."""
    import numpy as np
    from drone_nav.terrain import obstacle_field as of

    p = of.ObstacleFieldParams()
    tiles = {(0, i): of.generate_layout(p, d, of.layout_rng(i, d)) for i, d in enumerate((0.35, 0.65, 0.95))}
    packed = {k: torch.as_tensor(v) for k, v in of.pack_tiles(tiles, 1, 3).items()}
    rng = np.random.default_rng(0)
    for (_, col), layout in tiles.items():
        pts = np.stack([rng.uniform(0, p.size[0], 300), rng.uniform(0, p.size[1], 300)], axis=1)
        footprints = of.layout_footprints(layout)
        tile = {k: v[0, col][None].expand(len(pts), *v[0, col].shape) for k, v in packed.items()}
        dist, nearest = mdp.footprint_distances(torch.as_tensor(pts, dtype=torch.float32), **tile)
        for j, pt in enumerate(pts):
            for i, o in enumerate(footprints):
                d_ref, q_ref = of._point_footprint_distance(o, pt)
                assert abs(dist[j, i].item() - d_ref) < 1e-4, (o, pt)
                if d_ref > 1e-6:
                    assert np.allclose(nearest[j, i].numpy(), q_ref, atol=1e-4)
            assert torch.isinf(dist[j, len(footprints):]).all()  # padding


def test_nearest_obstacle_observation():
    point = torch.tensor([[0.0, 0.0]])
    dist = torch.tensor([[2.0, 0.5, float("inf"), 6.0]])
    nearest = torch.tensor([[[2.0, 0.0], [0.0, 0.5], [0.0, 0.0], [6.0, 0.0]]])
    obs = mdp.nearest_obstacle_observation(point, torch.tensor([math.pi / 2]), dist, nearest, k=3, max_range=5.0)
    assert obs.shape == (1, 9)
    # Closest first: (0, 0.5) is straight ahead for a drone facing +y; (2, 0) is to its right (body -y).
    torch.testing.assert_close(obs[0, 0:3], torch.tensor([1.0, 0.0, 0.1]), atol=1e-6, rtol=0)
    torch.testing.assert_close(obs[0, 3:6], torch.tensor([0.0, -1.0, 0.4]), atol=1e-6, rtol=0)
    # The third closest is 6 m away, beyond max_range: reads as "nothing there".
    torch.testing.assert_close(obs[0, 6:9], torch.tensor([0.0, 0.0, 1.0]))


def test_closest_by_kind():
    inf = float("inf")
    dist = torch.tensor([[0.5, 0.2, 0.9, inf], [1.0, 3.0, 0.4, 2.0]])
    kinds = torch.tensor([[0, 2, 0, -1], [2, 2, 1, -1]])
    out = mdp.closest_by_kind(dist, kinds, 4)
    torch.testing.assert_close(out, torch.tensor([[0.5, inf, 0.2, inf], [inf, 0.4, 1.0, inf]]))


def test_heading_travel_angle():
    yaw = torch.tensor([0.0, 0.0, math.pi / 2, math.pi, -3.0])
    vel = torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 3.0], [1.0, 0.0], [-1.0, 0.0]])
    expected = torch.tensor([0.0, math.pi / 2, 0.0, math.pi, math.pi - 3.0])  # -3 rad vs pi: wraps to 0.14
    torch.testing.assert_close(mdp.heading_travel_angle(yaw, vel), expected, atol=1e-6, rtol=0)


def test_heading_penalty():
    angle = torch.tensor([0.0, math.pi / 2, math.pi, math.pi, math.pi])
    speed = torch.tensor([3.0, 3.0, 3.0, 0.5, 0.0])
    r = mdp.heading_penalty(angle, speed, full_speed=1.0, weight=2.0, step_dt=0.02)
    # Facing travel: free. Sideways: -w per second. Backwards: -2w per second at full speed, half that at half speed.
    torch.testing.assert_close(r, torch.tensor([0.0, -0.04, -0.08, -0.04, 0.0]), atol=1e-7, rtol=0)
    assert (mdp.heading_penalty(angle, speed, 1.0, 0.0, 0.02) == 0).all()  # weight 0: off
