"""CPU tests for the obstacle-field terrain generator (drone_nav/terrain/obstacle_field.py), PLAN.md Phase 2.

The generator's guarantees: empty start and goal strips, gaps of at least min_gap everywhere, doors at least
min_gap wide, a flyable path from start to goal, and meshes that stay inside the footprints those checks use.
"""

import ast
import itertools
import math
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from drone_nav.terrain import obstacle_field as of

P = of.ObstacleFieldParams()
SEEDS = range(8)
# Mid-level difficulties, as Isaac Lab's curriculum draws them: row i is in [i/10, (i+1)/10).
LEVELS = [(lvl + 0.5) / 10 for lvl in range(10)]


def layout(seed, difficulty, params=P):
    return of.generate_layout(params, difficulty, of.layout_rng(seed, difficulty))


def all_layouts():
    return [layout(s, d) for s, d in itertools.product(SEEDS, LEVELS)]


@pytest.fixture(scope="module")
def layouts():
    return all_layouts()


# -- geometry helpers


def square(cx, cy, half=0.5, yaw=0.0):
    return of.Obstacle("box", (cx, cy), half_extents=(half, half), yaw=yaw)


def test_polygon_distance_apart_touching_and_overlapping():
    assert of.footprint_distance(square(0, 0), square(2, 0)) == pytest.approx(1.0)
    assert of.footprint_distance(square(0, 0), square(1, 0)) == pytest.approx(0.0)
    assert of.footprint_distance(square(0, 0), square(0.5, 0.5)) == 0.0
    # Diagonal: corner (0.5, 0.5) to corner (1.5, 1.5).
    assert of.footprint_distance(square(0, 0), square(2, 2)) == pytest.approx(math.sqrt(2))


def test_polygon_distance_sees_a_plus_sign_overlap():
    """Two thin crossing rectangles: no vertex of either is inside the other, but they overlap."""
    a = of.Obstacle("box", (0, 0), half_extents=(2.0, 0.1))
    b = of.Obstacle("box", (0, 0), half_extents=(0.1, 2.0))
    assert of.footprint_distance(a, b) == 0.0


def test_rotated_box_and_circle_distance():
    box = square(0, 0, half=0.5, yaw=math.pi / 4)  # a diamond with corners 0.707 m out along the axes
    circle = of.Obstacle("cylinder", (2.0, 0.0), radius=0.5)
    assert of.footprint_distance(box, circle) == pytest.approx(2.0 - math.sqrt(0.5) - 0.5)
    assert of.footprint_distance(circle, box) == of.footprint_distance(box, circle)
    # Facing an edge rather than a corner, and a circle centred inside the box.
    assert of.footprint_distance(square(0, 0), of.Obstacle("pole", (0.0, 1.5), radius=0.05)) == pytest.approx(0.95)
    assert of.footprint_distance(square(0, 0), of.Obstacle("pole", (0.1, 0.1), radius=0.05)) == 0.0


def test_circle_circle_distance_is_exact():
    a = of.Obstacle("cylinder", (0, 0), radius=0.3)
    b = of.Obstacle("pole", (1, 0), radius=0.05)
    assert of.footprint_distance(a, b) == pytest.approx(0.65)


# -- layout guarantees


def test_strips_and_buffers_are_empty(layouts):
    x0, x1 = P.obstacle_zone_x
    assert x0 == P.start_zone + P.zone_buffer and x1 == P.size[0] - P.goal_zone - P.zone_buffer
    for lay in layouts:
        for o in lay.obstacles:
            xmin, xmax, _, _ = o.bounds()
            assert xmin >= x0 - 1e-9 and xmax <= x1 + 1e-9, (lay.difficulty, o)


def test_every_gap_is_at_least_min_gap(layouts):
    for lay in layouts:
        for a, b in itertools.combinations(lay.obstacles, 2):
            if a.kind == b.kind == "wall" and a.center[0] == b.center[0]:
                continue  # the two halves of one wall: the door between them is checked below
            assert of.footprint_distance(a, b) >= lay.min_gap - 1e-9, (lay.difficulty, a, b)


def test_free_obstacles_keep_min_gap_from_side_walls(layouts):
    y0, y1 = P.inner_y
    for lay in layouts:
        for o in lay.obstacles:
            if o.kind != "wall":
                _, _, ymin, ymax = o.bounds()
                assert ymin >= y0 + lay.min_gap - 1e-9 and ymax <= y1 - lay.min_gap + 1e-9


def test_walls_span_the_corridor_with_a_wide_enough_door(layouts):
    y0, y1 = P.inner_y
    seen = 0
    for lay in layouts:
        segments = [o for o in lay.obstacles if o.kind == "wall"]
        assert len(segments) == 2 * lay.walls
        for x in {s.center[0] for s in segments}:
            lo, hi = sorted((s for s in segments if s.center[0] == x), key=lambda s: s.center[1])
            _, _, lo_ymin, lo_ymax = lo.bounds()
            _, _, hi_ymin, hi_ymax = hi.bounds()
            assert lo_ymin == pytest.approx(y0) and hi_ymax == pytest.approx(y1)  # wall to wall
            assert hi_ymin - lo_ymax >= lay.min_gap - 1e-9  # the door
            assert min(lo_ymax - y0, y1 - hi_ymin) >= P.min_wall_segment - 1e-9
            seen += 1
    assert seen > 0, "no walls in any layout: the test checked nothing"


def test_every_layout_is_passable(layouts):
    assert all(of.is_passable(lay) for lay in layouts)


def test_a_closed_wall_or_a_too_narrow_door_is_not_passable():
    y0, y1 = P.inner_y
    closed = of.Obstacle("wall", (12.0, (y0 + y1) / 2), half_extents=(0.1, (y1 - y0) / 2))
    assert not of.is_passable(of.Layout([closed], 1.0, 0.6, 1, 1, P))

    def door_layout(width):
        mid = (y0 + y1) / 2
        lo, hi = mid - width / 2, mid + width / 2
        segs = [
            of.Obstacle("wall", (12.0, (y0 + lo) / 2), half_extents=(0.1, (lo - y0) / 2)),
            of.Obstacle("wall", (12.0, (hi + y1) / 2), half_extents=(0.1, (y1 - hi) / 2)),
        ]
        return of.Layout(segs, 1.0, 0.6, 1, 1, P)

    assert not of.is_passable(door_layout(0.1))  # narrower than the drone (0.2 m across)
    assert of.is_passable(door_layout(0.6))


def test_density_and_gap_follow_difficulty(layouts):
    by_level = {d: [lay for lay in layouts if lay.difficulty == d] for d in LEVELS}
    counts = [np.mean([lay.placed for lay in by_level[d]]) for d in LEVELS]
    gaps = [by_level[d][0].min_gap for d in LEVELS]
    assert all(np.diff(counts) >= 0) and counts[-1] > counts[0] + 10
    assert all(np.diff(gaps) < 0)
    assert gaps[-1] >= 2 * 2 * 0.065  # PLAN.md §5.1: at least twice the Crazyflie's span, even at the hardest
    assert layout(0, 0.0).obstacles == []  # difficulty 0 is an empty corridor
    # The default density fits: at most one requested obstacle is ever skipped, and almost never that.
    assert all(lay.requested - 1 <= lay.placed <= lay.requested for lay in layouts)
    assert sum(lay.placed for lay in layouts) >= 0.99 * sum(lay.requested for lay in layouts)


def test_walls_only_appear_at_higher_difficulty(layouts):
    assert all(lay.walls == 0 for lay in layouts if lay.difficulty < 0.2)
    assert any(lay.walls > 0 for lay in layouts if lay.difficulty > 0.5)


def test_layouts_are_reproducible_and_seeds_differ():
    a, b = layout(3, 0.55), layout(3, 0.55)
    assert a.obstacles == b.obstacles
    assert layout(4, 0.55).obstacles != a.obstacles
    assert layout(3, 0.56).obstacles != a.obstacles
    assert of.TRAIN_SEED not in of.TEST_SEEDS


def test_gap_must_exceed_the_drone():
    params = of.ObstacleFieldParams(min_gap_range=(1.0, 0.15))
    with pytest.raises(ValueError):
        layout(0, 1.0, params)


# -- spawn / goal regions and meshes


def test_spawn_and_goal_regions_are_inside_the_strips():
    (sx, sy), (gx, gy) = of.spawn_and_goal_regions(P)
    ox, oy, oz = of.terrain_origin(P)
    assert oz == 0.0
    assert 0 < sx[0] + ox < sx[1] + ox < P.start_zone
    assert P.size[0] - P.goal_zone < gx[0] + ox < gx[1] + ox < P.size[0]
    y0, y1 = P.inner_y
    assert y0 < sy[0] + oy < sy[1] + oy < y1 and sy == gy
    # Start to goal is at least the obstacle zone's length.
    assert gx[0] - sx[1] >= P.obstacle_zone_x[1] - P.obstacle_zone_x[0]


def test_meshes_stay_inside_their_footprints():
    """The gap and passability checks use the footprints, so the collision meshes must not stick out of them."""
    lay = layout(5, 0.95)
    meshes = of.layout_meshes(lay)
    ground, rest = meshes[0], meshes[1:]
    assert np.allclose(ground.vertices[:, 2], 0.0)
    walls, obstacle_meshes = rest[:2], rest[2:]
    assert len(obstacle_meshes) == len(lay.obstacles)
    for mesh, o in zip(obstacle_meshes, lay.obstacles):
        z = mesh.vertices[:, 2]
        assert z.min() == pytest.approx(0.0) and z.max() == pytest.approx(P.obstacle_height)
        xy = mesh.vertices[:, :2]
        if o.is_round:
            assert np.linalg.norm(xy - np.asarray(o.center), axis=1).max() <= o.radius + 1e-9
        else:
            poly = o.polygon()
            dist = of._point_segment_distances(xy, poly, np.roll(poly, -1, axis=0)).min(axis=1)
            assert (of._inside_convex(xy, poly) | (dist < 1e-9)).all()
    y0, y1 = P.inner_y
    for wall in walls:
        ys = wall.vertices[:, 1]
        assert ys.max() <= y0 + 1e-9 or ys.min() >= y1 - 1e-9  # side walls stay outside the corridor


def test_isaac_lab_sub_terrain_contract():
    """obstacle_field_terrain(difficulty, cfg) -> (meshes, origin), as Isaac Lab's TerrainGenerator calls it: it
    copies the cfg, sets cfg.seed, then concatenates the meshes and shifts the origin in place."""
    import trimesh

    cfg = SimpleNamespace(**{f.name: getattr(P, f.name) for f in fields(P)}, seed=0)
    meshes, origin = of.obstacle_field_terrain(0.75, cfg)
    mesh = trimesh.util.concatenate(meshes)
    lo, hi = mesh.bounds
    assert np.allclose(lo, [0.0, 0.0, 0.0]) and np.allclose(hi, [*P.size, P.obstacle_height])
    origin += np.array([-P.size[0] / 2, -P.size[1] / 2, 0.0])  # what Isaac Lab does; needs a float array
    assert origin.shape == (3,)
    # Same seed and difficulty: same tile (Isaac Lab caching relies on it too).
    again, _ = of.obstacle_field_terrain(0.75, cfg)
    assert np.allclose(trimesh.util.concatenate(again).vertices, mesh.vertices)


def test_isaac_lab_cfg_has_the_same_fields_as_the_params():
    """obstacle_field_cfg.py needs Isaac Lab, so read its field names from the source instead of importing it."""
    src = Path(of.__file__).with_name("obstacle_field_cfg.py").read_text()
    cls = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef) and n.name == "ObstacleFieldTerrainCfg"
    )
    cfg_fields = {s.target.id for s in cls.body if isinstance(s, ast.AnnAssign)}
    assert cfg_fields == {f.name for f in fields(of.ObstacleFieldParams)}


def test_spawn_and_goal_regions_are_plain_floats():
    for region in of.spawn_and_goal_regions(P):
        for lo_hi in region:
            assert all(type(v) is float for v in lo_hi)


# -- collision attribution


def build_like_isaac_lab(num_rows, num_cols, seed=0):
    """Replay Isaac Lab v2.3.2's curriculum generator loop: column by column, rows in order, one uniform draw per
    tile from default_rng(seed), difficulty (row + u) / num_rows. Returns {(row, col): difficulty}."""
    rng = np.random.default_rng(seed)
    cfg = SimpleNamespace(**{f.name: getattr(P, f.name) for f in fields(P)}, seed=seed)
    built = {}
    for col in range(num_cols):
        for row in range(num_rows):
            d = (row + rng.uniform()) / num_rows
            of.obstacle_field_terrain(d, cfg)
            built[(row, col)] = d
    return built


def test_tile_layouts_recovers_each_tiles_layout():
    of.GENERATED_LAYOUTS.clear()
    built = build_like_isaac_lab(num_rows=10, num_cols=3)
    tiles = of.tile_layouts(of.GENERATED_LAYOUTS, 10, 3)
    assert set(tiles) == set(built)
    for key, difficulty in built.items():
        assert tiles[key].difficulty == difficulty
    # A layout built earlier in the process (another env) is ignored: only the last rows x cols count.
    of.GENERATED_LAYOUTS.insert(0, of.GENERATED_LAYOUTS[-1])
    assert of.tile_layouts(of.GENERATED_LAYOUTS, 10, 3)[(0, 0)].difficulty == built[(0, 0)]
    with pytest.raises(ValueError):
        of.tile_layouts(of.GENERATED_LAYOUTS[-5:], 10, 3)  # too few


def test_nearest_obstacle():
    y0, y1 = P.inner_y
    cyl = of.Obstacle("cylinder", (10.0, 5.0), radius=0.3)
    box = of.Obstacle("box", (14.0, 5.0), half_extents=(0.5, 0.5), yaw=math.pi / 4)  # corners 0.707 m out on the axes
    lay = of.Layout([cyl, box], 0.5, 1.0, 2, 0, P)

    kind, d, q = of.nearest_obstacle(lay, (10.0, 5.5))
    assert kind == "cylinder" and d == pytest.approx(0.2) and np.allclose(q, [10.0, 5.3])
    kind, d, q = of.nearest_obstacle(lay, (12.0, 5.0))
    assert kind == "box" and d == pytest.approx(2.0 - math.sqrt(0.5)) and np.allclose(q, [14.0 - math.sqrt(0.5), 5.0])
    assert of.nearest_obstacle(lay, (14.0, 5.1))[:2] == ("box", 0.0)  # inside
    kind, d, q = of.nearest_obstacle(lay, (3.0, y0 + 0.07))
    assert kind == "side_wall" and d == pytest.approx(0.07) and np.allclose(q, [3.0, y0])
    assert of.nearest_obstacle(lay, (3.0, y1 - 0.2))[0] == "side_wall"


def test_nearest_obstacle_agrees_with_the_occupancy_grid():
    """Cells the grid marks as within r of something must have nearest_obstacle distance <= r, and vice versa."""
    lay = layout(2, 0.95)
    r = 0.25
    blocked, xs, ys = of.occupancy_grid(lay, r)
    rng = np.random.default_rng(0)
    for i, j in zip(rng.integers(0, len(xs), 400), rng.integers(0, len(ys), 400)):
        d = of.nearest_obstacle(lay, (xs[i], ys[j]))[1]
        if abs(d - r) < 1e-6:
            continue  # exactly on the threshold: the grid uses < for side walls and <= for obstacles
        assert (d <= r) == bool(blocked[i, j]), (xs[i], ys[j], d)


def test_kind_counts_count_a_wall_once():
    lay = layout(2, 0.95)
    counts = of.kind_counts(lay)
    assert counts["wall"] == lay.walls
    assert sum(counts.values()) == lay.placed


# -- lidar replay

RING_10 = np.deg2rad(np.linspace(-180.0, 180.0, 37)[:-1])  # the env's 10-degree ring, relative to the heading


def analytic_ring(position, yaw, circles, max_range=5.0):
    """Independent reference: each ray's distance to the nearest of `circles` [(cx, cy, r)], clipped to max_range."""
    out = np.full(len(RING_10), max_range)
    for i, off in enumerate(RING_10):
        u = np.array([math.cos(yaw + off), math.sin(yaw + off)])
        for cx, cy, r in circles:
            d = np.array([cx, cy]) - np.asarray(position)
            along = d @ u
            perp2 = d @ d - along**2
            if along > 0 and perp2 <= r * r:
                out[i] = min(out[i], along - math.sqrt(r * r - perp2))
    return out


def test_ray_hit_distances():
    cyl = of.Obstacle("cylinder", (2.0, 0.0), radius=0.5)
    sq = of.Obstacle("box", (2.0, 0.0), half_extents=(0.5, 0.5))
    origins = np.array([[0.0, 0.0], [2.0, 0.1]])
    angles = np.array([[0.0, math.pi / 2, math.pi], [0.0, 0.0, 0.0]])
    d = of.ray_hit_distances(cyl, origins, angles)
    assert d[0, 0] == pytest.approx(1.5) and np.isinf(d[0, 1]) and np.isinf(d[0, 2])
    assert np.all(d[1] == 0.0)  # starting inside
    d = of.ray_hit_distances(sq, origins, angles)
    assert d[0, 0] == pytest.approx(1.5) and np.isinf(d[0, 1]) and np.isinf(d[0, 2]) and np.all(d[1] == 0.0)
    # From (1, 1) at -60 degrees: the ray drops to the top face's height (y = 0.5) at x = 1.29, still left of the
    # square (x >= 1.5), so it hits the left face x = 1.5 instead, at t = 0.5 / cos(60 deg) = 1.0 (y = 0.134).
    t = of.ray_hit_distances(sq, np.array([[1.0, 1.0]]), np.array([[-math.pi / 3]]))[0, 0]
    assert t == pytest.approx(1.0)
    # A rotated box (diamond, corners 0.707 m out): straight at its left corner from the origin.
    diamond = of.Obstacle("box", (2.0, 0.0), half_extents=(0.5, 0.5), yaw=math.pi / 4)
    t = of.ray_hit_distances(diamond, np.zeros((1, 2)), np.zeros((1, 1)))[0, 0]
    assert t == pytest.approx(2.0 - math.sqrt(0.5))


def fly_at(circle_target, circles, yaw, speed=3.0, dt=0.02, end_surface=0.07):
    """Positions flying along +x at the target circle's centre line until end_surface from it, with the ring's
    readings at each step."""
    cx, cy, r = circle_target
    xs = np.arange(0.0, cx - r - end_surface + 1e-9, speed * dt)
    positions = np.stack([xs, np.full_like(xs, cy)], axis=1)
    readings = np.stack([analytic_ring(p, yaw, circles) for p in positions])
    return positions, np.full(len(xs), yaw), readings


def test_a_thin_pole_between_rays_is_seen_only_up_close():
    """Heading 5 degrees, so the 10-degree ring's rays are at +-5 degrees either side of a pole straight ahead. A
    3 cm pole is first crossed by a ray once its centre is within 0.03 / sin(5 deg) = 0.344 m (surface 0.314 m)."""
    pole = of.Obstacle("pole", (3.0, 0.0), radius=0.03)
    pos, yaw, readings = fly_at((3.0, 0.0, 0.03), [(3.0, 0.0, 0.03)], math.radians(5.0))
    vis = of.obstacle_visibility(pole, pos, yaw, RING_10, readings, max_range=5.0)
    summary = of.visibility_summary(vis, 0.02)
    assert summary["never_returned"] == 0.0
    assert 0.25 <= summary["first_return_dist"] <= 0.3142 + 1e-9
    assert summary["between_rays_frac"] > 0.85  # almost the whole approach: invisible
    assert summary["occluded_frac"] == 0.0
    assert summary["first_return_s"] < 0.1


def test_a_thick_cylinder_is_seen_all_the_way():
    cyl = of.Obstacle("cylinder", (3.0, 0.0), radius=0.5)
    pos, yaw, readings = fly_at((3.0, 0.0, 0.5), [(3.0, 0.0, 0.5)], math.radians(5.0))
    vis = of.obstacle_visibility(cyl, pos, yaw, RING_10, readings, max_range=5.0)
    assert vis["returned"].all()
    summary = of.visibility_summary(vis, 0.02)
    assert summary["between_rays_frac"] == 0.0 and summary["first_return_dist"] == pytest.approx(2.5)


def test_a_pole_behind_another_obstacle_is_occluded_not_between_rays():
    pole = of.Obstacle("pole", (3.0, 0.0), radius=0.06)
    pos = np.zeros((5, 2))
    readings = np.stack([analytic_ring(p, 0.0, [(3.0, 0.0, 0.06), (1.5, 0.0, 0.2)]) for p in pos])
    vis = of.obstacle_visibility(pole, pos, np.zeros(5), RING_10, readings, max_range=5.0)
    assert not vis["returned"].any() and vis["occluded"].all() and not vis["between_rays"].any()
    assert of.visibility_summary(vis, 0.02)["never_returned"] == 1.0


def test_nearest_footprint_returns_the_obstacle():
    cyl = of.Obstacle("cylinder", (10.0, 5.0), radius=0.3)
    lay = of.Layout([cyl], 0.5, 1.0, 1, 0, P)
    kind, d, _, o = of.nearest_footprint(lay, (10.0, 5.5))
    assert kind == "cylinder" and o is cyl and d == pytest.approx(0.2)
    assert of.nearest_footprint(lay, (3.0, P.inner_y[0] + 0.05))[3] is None  # side wall
