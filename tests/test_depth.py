"""CPU tests for the Phase 3 depth-camera math (drone_nav/sensors/depth.py)."""

import math

import torch
from drone_nav.sensors import depth as dm

# The env's camera (tasks/depth/depth_env.py): 64 x 48, 90-degree horizontal FOV, square pixels.
W, H, FOCAL, H_AP = 64, 48, 10.0, 20.0
V_AP = H_AP * H / W


def test_fov_and_focal_length():
    assert dm.horizontal_fov_deg(H_AP, FOCAL) == 90.0
    assert dm.focal_length_px(W, H_AP, FOCAL) == dm.focal_length_px(H, V_AP, FOCAL) == 32.0  # square pixels


def test_column_azimuths_match_the_pinhole_pattern():
    az = dm.column_azimuths(W, H_AP, FOCAL)
    assert az.shape == (W,)
    assert (az[:-1] > az[1:]).all()  # column 0 is on the left (positive azimuth), decreasing to the right
    torch.testing.assert_close(az, -az.flip(0))  # symmetric about the optical axis
    # Pixel centres: the outermost column is half a pixel inside the 45-degree edge.
    assert math.isclose(az[0].item(), math.atan(31.5 / 32.0))
    assert math.isclose(az[W // 2].item(), -math.atan(0.5 / 32.0))
    el = dm.row_elevations(H, V_AP, FOCAL)
    assert (el[:-1] > el[1:]).all() and math.isclose(el[0].item(), math.atan(23.5 / 32.0))  # row 0 is the top


def test_depth_edges():
    d = torch.full((1, 4, 5), 3.0)
    d[0, :, 3:] = 1.0  # a near object on the right
    e = dm.depth_edges(d, 0.5)
    assert e[0, :, 2:4].all() and not e[0, :, :2].any() and not e[0, :, 4].any()


def test_noise_free_settings_change_nothing_but_clipping():
    d = torch.rand(3, H, W) * 8.0
    out = dm.add_depth_noise(d, 6.0, 0.0, 0.0, 0.0, 0.0, 0.3)
    torch.testing.assert_close(out, d.clamp(max=6.0))


def test_noise_grows_with_range_and_dropout_reads_max_range():
    g = torch.Generator().manual_seed(0)
    near, far = torch.full((1, 200, 200), 1.0), torch.full((1, 200, 200), 5.0)
    kw = dict(max_range=6.0, gaussian_std=0.01, range_std=0.005, dropout_prob=0.0, edge_dropout_prob=0.0,
              edge_threshold=0.3)  # fmt: skip
    sn = (dm.add_depth_noise(near, generator=g, **kw) - near).std().item()
    sf = (dm.add_depth_noise(far, generator=g, **kw) - far).std().item()
    assert math.isclose(sn, 0.015, rel_tol=0.05) and math.isclose(sf, 0.01 + 0.005 * 25, rel_tol=0.05)
    kw.update(gaussian_std=0.0, range_std=0.0, dropout_prob=0.1)
    out = dm.add_depth_noise(near, generator=g, **kw)
    assert math.isclose((out == 6.0).float().mean().item(), 0.1, abs_tol=0.01)
    assert ((out == 1.0) | (out == 6.0)).all()


def test_edge_pixels_drop_out_more():
    g = torch.Generator().manual_seed(1)
    d = torch.full((50, 8, 8), 4.0)
    d[:, :, 4:] = 1.0
    out = dm.add_depth_noise(d, 6.0, 0.0, 0.0, 0.0, 0.5, 0.3, generator=g)
    edges = dm.depth_edges(d, 0.3)
    assert math.isclose((out[edges] == 6.0).float().mean().item(), 0.5, abs_tol=0.05)
    assert not (out[~edges] == 6.0).any()


def test_max_range_pixels_stay_at_max_range():
    g = torch.Generator().manual_seed(2)
    d = torch.full((1, 10, 10), 6.0)
    out = dm.add_depth_noise(d, 6.0, 0.05, 0.05, 0.0, 0.0, 0.3, generator=g)
    assert (out == 6.0).all()


def test_depth_observation():
    d = torch.tensor([[[0.0, 3.0], [6.0, 9.0]]])
    torch.testing.assert_close(dm.depth_observation(d, 6.0), torch.tensor([[0.0, 0.5, 1.0, 1.0]]))


# -- the camera replay that scripts/check_depth.py compares the real camera against

from drone_nav.terrain import obstacle_field as of  # noqa: E402


def _rays():
    f = dm.focal_length_px(W, H_AP, FOCAL)
    col_y = -(torch.arange(W) + 0.5 - W / 2).double().numpy() / f
    row_z = -(torch.arange(H) + 0.5 - H / 2).double().numpy() / f
    return col_y, row_z


def test_camera_replay_wall_ahead_and_ground():
    import numpy as np

    col_y, row_z = _rays()
    wall = of.Obstacle("wall", (4.0, 0.0), half_extents=(0.1, 10.0))  # its face at x = 3.9, spanning the view
    depth, src = of.camera_depth_replay([wall], (0.0, 0.0, 1.0), 0.0, col_y, row_z, 5.0, 6.0)
    assert depth.shape == (H, W)
    # A plane facing the camera has the same image-plane depth in every pixel that sees it...
    assert np.allclose(depth[src == 0], 3.9)
    # ...which is every pixel except the bottom rows, where the ground (1 m below) is closer: z / -row_z < 3.9.
    with np.errstate(divide="ignore"):
        ground_rows = (row_z < 0) & (1.0 / -row_z < 3.9)
    assert (src[ground_rows] == -1).all() and (src[~ground_rows] == 0).all()
    np.testing.assert_allclose(depth[ground_rows][:, 0], 1.0 / -row_z[ground_rows])


def test_camera_replay_follows_yaw_and_left_is_column_zero():
    import numpy as np

    col_y, row_z = _rays()
    pole = of.Obstacle("pole", (0.0, 3.0), radius=0.05)  # 3 m along +y
    mid = [H // 2 - 1, H // 2]
    # Facing +x, +y is 90 degrees to the left: outside the 90-degree FOV.
    _, src = of.camera_depth_replay([pole], (0.0, 0.0, 1.0), 0.0, col_y, row_z, 5.0, 6.0)
    assert not (src == 0).any()
    # Facing +y: dead ahead, in the middle columns, at 2.95 m.
    depth, src = of.camera_depth_replay([pole], (0.0, 0.0, 1.0), np.pi / 2, col_y, row_z, 5.0, 6.0)
    cols = np.nonzero((src[mid] == 0).any(axis=0))[0]
    assert set(cols) <= {W // 2 - 1, W // 2} and len(cols) > 0
    # The middle columns are half a pixel (az) off the axis, so their rays hit the pole off-centre: along the ray,
    # t = 3 cos(az) - sqrt(r^2 - (3 sin az)^2), and the image-plane depth is t cos(az).
    az = abs(np.arctan(col_y[W // 2]))
    t = 3.0 * np.cos(az) - np.sqrt(0.05**2 - (3.0 * np.sin(az)) ** 2)
    assert np.isclose(depth[mid][:, cols].min(), t * np.cos(az), atol=1e-9)
    # Facing +y rotated 30 degrees to the right (yaw 60 degrees): the pole is 30 degrees left, in the left half.
    _, src = of.camera_depth_replay([pole], (0.0, 0.0, 1.0), np.pi / 3, col_y, row_z, 5.0, 6.0)
    cols = np.nonzero((src[mid] == 0).any(axis=0))[0]
    assert len(cols) > 0 and cols.max() < W // 2


def test_camera_replay_sees_over_a_short_obstacle_only_above_its_top():
    import numpy as np

    col_y, row_z = _rays()
    box = of.Obstacle("box", (3.0, 0.0), half_extents=(0.5, 5.0))  # face at x = 2.5, 0.5 m tall in this test
    depth, src = of.camera_depth_replay([box], (0.0, 0.0, 1.0), 0.0, col_y, row_z, 0.5, 6.0)
    # Rows whose ray is below 0.5 m at x = 2.5 see the box; rows above it see past it (nothing, or max range).
    sees = 1.0 + row_z * 2.5 <= 0.5
    assert (src[sees & (1.0 / np.maximum(-row_z, 1e-9) > 2.5)] == 0).all()
    assert (src[~sees] != 0).all()
    assert (depth <= 6.0).all() and (depth[src == -2] == 6.0).all()


def test_pitch_quat_tilts_the_view_down():
    from drone_nav.mdp import randomization as rnd

    q = dm.pitch_quat(torch.tensor([0.0, 0.2]))
    forward = torch.tensor([[1.0, 0.0, 0.0]] * 2)
    # world_to_body applies q^-1; the camera's forward axis in the parent frame is q applied, i.e. q^-1 of -angle.
    body = rnd.world_to_body(forward, torch.stack([q[:, 0], -q[:, 1], -q[:, 2], -q[:, 3]], dim=-1))
    torch.testing.assert_close(body[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(body[1], torch.tensor([math.cos(0.2), 0.0, -math.sin(0.2)]), atol=1e-6, rtol=0)


def test_pitched_replay_matches_level_at_zero_and_sees_the_ground_sooner_when_tilted_down():
    import numpy as np

    col_y, row_z = _rays()
    p = of.ObstacleFieldParams()
    layout = of.generate_layout(p, 0.55, of.layout_rng(4, 0.55))
    fps = of.layout_footprints(layout)
    pos, yaw = (3.0, 5.0, 1.2), 0.3
    level, src = of.camera_depth_replay(fps, pos, yaw, col_y, row_z, 5.0, 6.0)
    tiny, src_t = of._camera_depth_replay_pitched(fps, pos, yaw, col_y, row_z, 5.0, 6.0, 1e-9)
    np.testing.assert_allclose(tiny, level, atol=1e-6)  # the general path agrees with the level one
    down, src_d = of.camera_depth_replay(fps, pos, yaw, col_y, row_z, 5.0, 6.0, pitch=np.radians(10))
    up, src_u = of.camera_depth_replay(fps, pos, yaw, col_y, row_z, 5.0, 6.0, pitch=np.radians(-10))
    assert (src_d == -1).sum() > (src == -1).sum() > (src_u == -1).sum()  # more ground in view when tilted down


def test_ood_variants_still_generate_passable_layouts():
    import numpy as np

    p = of.ObstacleFieldParams()
    assert of.ood_overrides(p) == {}
    for kw in ({"density_scale": 1.5}, {"mix": "poles"}, {"mix": "walls"}, {"unseen_sizes": True}):
        q = of.ObstacleFieldParams(**{**p.__dict__, **of.ood_overrides(p, **kw)})
        for seed in range(5):
            layout = of.generate_layout(q, 0.55, of.layout_rng(seed, 0.55))
            assert of.is_passable(layout)
        if kw.get("mix") == "poles":
            assert all(o.kind == "pole" for o in layout.obstacles)
        if kw.get("unseen_sizes"):
            assert all(o.radius <= 0.03 for o in layout.obstacles if o.kind == "pole")
            assert all(o.radius >= 0.5 for o in layout.obstacles if o.kind == "cylinder")
    dense = of.ObstacleFieldParams(**{**p.__dict__, **of.ood_overrides(p, density_scale=1.5)})
    n_dense = np.mean([len(of.generate_layout(dense, 0.55, of.layout_rng(s, 0.55)).obstacles) for s in range(5)])
    n_base = np.mean([len(of.generate_layout(p, 0.55, of.layout_rng(s, 0.55)).obstacles) for s in range(5)])
    assert n_dense > 1.25 * n_base
    import pytest

    with pytest.raises(ValueError):
        of.ood_overrides(p, mix="spheres")
