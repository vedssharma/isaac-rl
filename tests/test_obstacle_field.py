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
