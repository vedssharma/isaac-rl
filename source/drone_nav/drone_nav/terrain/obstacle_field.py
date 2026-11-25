"""Obstacle-field sub-terrain for Drone-Nav-Obstacles-v0 (PLAN.md §5.1, option A of §5.2).

Layout logic is plain numpy and meshes are plain trimesh, with no Isaac Lab imports, so all of it runs in CPU tests.
obstacle_field_cfg.py plugs `obstacle_field_terrain` into Isaac Lab's terrain generator, which calls it once per
tile with a difficulty in [0, 1] and bakes every tile into one static mesh. That one mesh gives collisions through
the terrain collider and fast ray casting through RayCaster on any GPU.

A tile is a corridor along +x, `size = (L, W)` metres:

    x = 0                                                                       x = L
    | start strip | buffer |        obstacle zone        | buffer | goal strip |
    side walls along both long edges (optional)

The drone spawns in the start strip and its goal is in the goal strip, about L - 2 m away. Obstacles are vertical
prisms from the ground up to `obstacle_height`, which is above the env's altitude limit, so the drone can only go
around them. That makes the layout a 2-D problem about the footprints.

Obstacle types (PLAN.md §5.1): cylinders (trees, pillars), boxes at any yaw (buildings, crates), thin poles, and
walls across the corridor with one door. Difficulty sets the density, the minimum gap and the mix of types, by
linear interpolation between the `*_range` / `mix_*` endpoints. Isaac Lab's curriculum gives tile row i a
difficulty in [i/rows, (i+1)/rows), so with 10 rows, row i is difficulty level i.

What every layout guarantees (tests/test_obstacle_field.py checks each one):
- the start strip, the goal strip and the buffers next to them are empty;
- every two footprints are at least `min_gap` apart, free-standing obstacles are at least `min_gap` from the side
  walls, and every wall door is at least `min_gap` wide;
- a disk of radius `drone_radius` can get from the start strip to the goal strip. The generator checks this with a
  grid flood fill and draws a new layout if the check fails.

Held-out test maps are tiles from a generator seed that training never uses (PLAN.md §5.1): a tile's layout depends
only on (seed, difficulty), see `layout_rng`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np

KINDS = ("cylinder", "box", "pole", "wall")
# Terrain generator seeds. Evaluation tiles come from TEST_SEEDS and are never used in training.
TRAIN_SEED = 0
TEST_SEEDS = (1001, 1002, 1003)
# Sides of a cylinder mesh. Its vertices are on the circle, so the mesh is inside the footprint the gap checks use.
_CIRCLE_SIDES = 16


@dataclass
class ObstacleFieldParams:
    """Generator settings. ObstacleFieldTerrainCfg (Isaac Lab) has the same fields, with the same defaults."""

    size: tuple[float, float] = (24.0, 10.0)  # (length along x, width along y), m
    start_zone: float = 2.0  # depth of the empty spawn strip at x = 0, m
    goal_zone: float = 2.0  # depth of the empty goal strip at x = L, m
    zone_buffer: float = 1.0  # extra empty depth between each strip and the obstacle zone, m
    obstacle_height: float = 5.0  # m; above the env's 4 m altitude limit, so nothing can be flown over
    side_walls: bool = True
    side_wall_thickness: float = 0.2  # m, inside the tile

    # Curriculum: (value at difficulty 0, value at difficulty 1), linear in between.
    # obstacles per 100 m^2 of obstacle zone. 12 at difficulty 1 is 20 obstacles in the default 18 x 9.6 m zone,
    # about the most that still fits at that difficulty's gap (99.98% of requested obstacles get placed).
    density_range: tuple[float, float] = (0.0, 12.0)
    min_gap_range: tuple[float, float] = (2.0, 0.6)  # m; at least 2x the Crazyflie's span even at difficulty 1
    # Relative weights of KINDS (cylinder, box, pole, wall).
    mix_easy: tuple[float, float, float, float] = (0.6, 0.3, 0.1, 0.0)
    mix_hard: tuple[float, float, float, float] = (0.35, 0.3, 0.2, 0.15)

    # Shapes, uniform in (low, high).
    cylinder_radius: tuple[float, float] = (0.15, 0.5)  # m
    pole_radius: tuple[float, float] = (0.03, 0.06)  # m
    box_side: tuple[float, float] = (0.4, 1.5)  # m, each side
    wall_thickness: float = 0.2  # m
    door_width: tuple[float, float] = (0.8, 2.0)  # m, raised to min_gap when it is below
    min_wall_segment: float = 0.5  # m: the door stays this far from the side walls

    # Generation.
    drone_radius: float = 0.1  # m, for the passability check; Crazyflie 2.x prop tips are at about 0.065 m
    grid_resolution: float = 0.1  # m, passability grid
    max_attempts: int = 200  # random placements tried per obstacle before skipping it (rarely needed: see tests)
    max_layout_tries: int = 10  # whole layouts tried before giving up on passability

    @classmethod
    def from_cfg(cls, cfg) -> ObstacleFieldParams:
        """Read the same-named attributes from any config object (e.g. ObstacleFieldTerrainCfg)."""
        return cls(**{f.name: getattr(cfg, f.name) for f in fields(cls)})

    # -- derived geometry (tile frame: x in [0, L], y in [0, W])

    @property
    def inner_y(self) -> tuple[float, float]:
        """y range between the side walls (the whole width without them)."""
        t = self.side_wall_thickness if self.side_walls else 0.0
        return t, self.size[1] - t

    @property
    def obstacle_zone_x(self) -> tuple[float, float]:
        return self.start_zone + self.zone_buffer, self.size[0] - self.goal_zone - self.zone_buffer


def lerp(endpoints, t: float) -> float:
    return float(endpoints[0] + (endpoints[1] - endpoints[0]) * t)


@dataclass(frozen=True)
class Obstacle:
    """One footprint. Round kinds (cylinder, pole) use `radius`; box and wall segments use `half_extents` and
    `yaw`. A wall across the corridor is two "wall" segments, one each side of its door."""

    kind: str
    center: tuple[float, float]
    radius: float = 0.0
    half_extents: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0

    @property
    def is_round(self) -> bool:
        return self.kind in ("cylinder", "pole")

    def polygon(self) -> np.ndarray:
        """(4, 2) corners of a box or wall segment, counter-clockwise. Round kinds have none."""
        if self.is_round:
            raise ValueError(f"a {self.kind} footprint is a circle, not a polygon")
        c = np.asarray(self.center, dtype=np.float64)
        hx, hy = self.half_extents
        corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
        cos, sin = math.cos(self.yaw), math.sin(self.yaw)
        return c + corners @ np.array([[cos, sin], [-sin, cos]])

    def bounds(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the footprint (exact for circles)."""
        if self.is_round:
            x, y = self.center
            return x - self.radius, x + self.radius, y - self.radius, y + self.radius
        p = self.polygon()
        return p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()


@dataclass
class Layout:
    obstacles: list[Obstacle]
    difficulty: float
    min_gap: float
    requested: int  # obstacles the density asked for; fewer are placed when the zone is too full
    walls: int  # walls across the corridor (each is two "wall" segments in `obstacles`)
    params: ObstacleFieldParams = field(repr=False)

    @property
    def placed(self) -> int:
        return len(self.obstacles) - self.walls  # a wall's two segments count as one obstacle


# -- geometry


def _point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(m, k) distances from each of m points to each of k segments a[i]-b[i]."""
    ab = b - a
    ap = points[:, None, :] - a[None]
    t = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1)


def _inside_convex(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(m,) True where a point is inside (or on) the counter-clockwise convex polygon."""
    a, b = poly, np.roll(poly, -1, axis=0)
    cross = (b - a)[None, :, 0] * (points[:, None, 1] - a[None, :, 1]) - (b - a)[None, :, 1] * (
        points[:, None, 0] - a[None, :, 0]
    )
    return (cross >= -1e-12).all(axis=1)


def polygon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Distance between two convex polygons, 0 if they overlap."""
    # Edges can cross with no vertex inside the other polygon (a plus sign of two thin rectangles), so check both.
    if _inside_convex(p, q).any() or _inside_convex(q, p).any() or _edges_cross(p, q):
        return 0.0
    # Apart, the closest pair is always a vertex of one polygon and an edge of the other.
    d_pq = _point_segment_distances(p, q, np.roll(q, -1, axis=0)).min()
    d_qp = _point_segment_distances(q, p, np.roll(p, -1, axis=0)).min()
    return float(min(d_pq, d_qp))


def _edges_cross(p: np.ndarray, q: np.ndarray) -> bool:
    a1, a2 = p, np.roll(p, -1, axis=0)
    b1, b2 = q, np.roll(q, -1, axis=0)

    def orient(o, s, t):  # sign of cross(s - o, t - o), broadcast over (len(p), len(q))
        u, v = s - o, t - o
        return np.sign(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])

    A1, A2, B1, B2 = a1[:, None], a2[:, None], b1[None], b2[None]
    d1, d2 = orient(B1, B2, A1), orient(B1, B2, A2)
    d3, d4 = orient(A1, A2, B1), orient(A1, A2, B2)
    return bool(((d1 * d2 < 0) & (d3 * d4 < 0)).any())


def circle_polygon_distance(center, radius: float, poly: np.ndarray) -> float:
    """Distance between a circle and a convex polygon, 0 if they overlap."""
    c = np.asarray(center, dtype=np.float64)[None]
    if _inside_convex(c, poly)[0]:
        return 0.0
    d = _point_segment_distances(c, poly, np.roll(poly, -1, axis=0)).min()
    return max(float(d) - radius, 0.0)


def footprint_distance(a: Obstacle, b: Obstacle) -> float:
    """Exact distance between two footprints (circles or rotated rectangles), 0 if they overlap."""
    if a.is_round and b.is_round:
        return max(math.dist(a.center, b.center) - a.radius - b.radius, 0.0)
    if a.is_round or b.is_round:
        circle, box = (a, b) if a.is_round else (b, a)
        return circle_polygon_distance(circle.center, circle.radius, box.polygon())
    return polygon_distance(a.polygon(), b.polygon())


def _bounding_radius(o: Obstacle) -> float:
    return o.radius if o.is_round else math.hypot(*o.half_extents)


def _clear_of(candidate: Obstacle, placed: list[Obstacle], gap: float) -> bool:
    cr = _bounding_radius(candidate)
    for o in placed:
        # Cheap accept: bounding circles already far enough apart.
        if math.dist(candidate.center, o.center) - cr - _bounding_radius(o) >= gap:
            continue
        if footprint_distance(candidate, o) < gap:
            return False
    return True


# -- layout


def layout_rng(seed: int | None, difficulty: float) -> np.random.Generator:
    """The tile's random generator, from the terrain generator's seed and the tile's difficulty.

    Isaac Lab sets `cfg.seed` (the TerrainGeneratorCfg seed, possibly None) and draws a different continuous
    difficulty for every tile, so (seed, difficulty) identifies a tile. With no seed, fall back to numpy's global
    state, as Isaac Lab itself does.
    """
    bits = int(np.float64(difficulty).view(np.uint64))
    base = int(seed) if seed is not None else int(np.random.randint(2**31))
    return np.random.default_rng([base, bits])


def _sample_wall(p: ObstacleFieldParams, gap: float, rng: np.random.Generator) -> list[Obstacle]:
    """Two segments across the corridor, one each side of a door at least `gap` wide."""
    (x0, x1), (y0, y1) = p.obstacle_zone_x, p.inner_y
    t = p.wall_thickness
    x = rng.uniform(x0 + t / 2, x1 - t / 2)
    door = max(rng.uniform(*p.door_width), gap)
    door = min(door, (y1 - y0) - 2 * p.min_wall_segment)
    lo = rng.uniform(y0 + p.min_wall_segment, y1 - p.min_wall_segment - door)  # door's lower edge
    hi = lo + door
    return [
        Obstacle("wall", (x, (y0 + lo) / 2), half_extents=(t / 2, (lo - y0) / 2)),
        Obstacle("wall", (x, (hi + y1) / 2), half_extents=(t / 2, (y1 - hi) / 2)),
    ]


def _sample_free_obstacle(kind: str, p: ObstacleFieldParams, rng: np.random.Generator) -> Obstacle:
    (x0, x1), (y0, y1) = p.obstacle_zone_x, p.inner_y
    center = (rng.uniform(x0, x1), rng.uniform(y0, y1))
    if kind == "cylinder":
        return Obstacle(kind, center, radius=rng.uniform(*p.cylinder_radius))
    if kind == "pole":
        return Obstacle(kind, center, radius=rng.uniform(*p.pole_radius))
    half = (rng.uniform(*p.box_side) / 2, rng.uniform(*p.box_side) / 2)
    return Obstacle(kind, center, half_extents=half, yaw=rng.uniform(0.0, math.pi))


def _fits_zone(o: Obstacle, p: ObstacleFieldParams, gap: float) -> bool:
    """A free-standing obstacle stays inside the obstacle zone and at least `gap` from the side walls."""
    (x0, x1), (y0, y1) = p.obstacle_zone_x, p.inner_y
    xmin, xmax, ymin, ymax = o.bounds()
    return xmin >= x0 and xmax <= x1 and ymin >= y0 + gap and ymax <= y1 - gap


def _draw_layout(p: ObstacleFieldParams, difficulty: float, rng: np.random.Generator) -> Layout:
    gap = lerp(p.min_gap_range, difficulty)
    (x0, x1), (y0, y1) = p.obstacle_zone_x, p.inner_y
    requested = round(lerp(p.density_range, difficulty) * (x1 - x0) * (y1 - y0) / 100.0)
    mix = np.array([lerp(pair, difficulty) for pair in zip(p.mix_easy, p.mix_hard)])
    kinds = [KINDS[i] for i in rng.choice(len(KINDS), size=requested, p=mix / mix.sum())]
    kinds.sort(key=lambda k: k != "wall")  # walls first: they are the hardest to fit

    placed: list[Obstacle] = []
    walls = 0
    for kind in kinds:
        for _ in range(p.max_attempts):
            if kind == "wall":
                candidate = _sample_wall(p, gap, rng)
            else:
                candidate = [_sample_free_obstacle(kind, p, rng)]
                if not _fits_zone(candidate[0], p, gap):
                    continue
            if all(_clear_of(c, placed, gap) for c in candidate):
                placed += candidate
                walls += kind == "wall"
                break
    return Layout(placed, difficulty, gap, requested, walls, p)


def generate_layout(p: ObstacleFieldParams, difficulty: float, rng: np.random.Generator) -> Layout:
    """A random layout at this difficulty that a disk of radius `drone_radius` can fly through."""
    if lerp(p.min_gap_range, difficulty) <= 2 * p.drone_radius:
        raise ValueError(f"min gap at difficulty {difficulty:.2f} must exceed the drone's diameter")
    for _ in range(p.max_layout_tries):
        layout = _draw_layout(p, difficulty, rng)
        if is_passable(layout):
            return layout
    raise RuntimeError(f"no passable layout in {p.max_layout_tries} tries at difficulty {difficulty:.3f}")


# -- passability


def occupancy_grid(layout: Layout, inflate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(blocked, xs, ys): blocked[i, j] is True when the cell centred at (xs[i], ys[j]) is within `inflate` of an
    obstacle or a side wall (the tile's long edges when there are no walls), i.e. a disk of that radius centred
    there would touch something or leave the tile."""
    p = layout.params
    res = p.grid_resolution
    xs = np.arange(res / 2, p.size[0], res)
    ys = np.arange(res / 2, p.size[1], res)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    y0, y1 = p.inner_y
    blocked = (gy - y0 < inflate) | (y1 - gy < inflate)
    for o in layout.obstacles:
        xmin, xmax, ymin, ymax = o.bounds()
        ix = (xs >= xmin - inflate) & (xs <= xmax + inflate)
        iy = (ys >= ymin - inflate) & (ys <= ymax + inflate)
        if not ix.any() or not iy.any():
            continue
        sub = np.stack([gx[np.ix_(ix, iy)].ravel(), gy[np.ix_(ix, iy)].ravel()], axis=-1)
        if o.is_round:
            hit = np.linalg.norm(sub - np.asarray(o.center), axis=-1) <= o.radius + inflate
        else:
            poly = o.polygon()
            dist = _point_segment_distances(sub, poly, np.roll(poly, -1, axis=0)).min(axis=1)
            hit = _inside_convex(sub, poly) | (dist <= inflate)
        blocked[np.ix_(ix, iy)] |= hit.reshape(ix.sum(), iy.sum())
    return blocked, xs, ys


def is_passable(layout: Layout) -> bool:
    """True if a disk of radius drone_radius can get from the start strip to the goal strip (4-connected flood
    fill on the occupancy grid)."""
    p = layout.params
    blocked, xs, _ = occupancy_grid(layout, p.drone_radius)
    free = ~blocked
    reached = free & (xs <= p.start_zone)[:, None]
    goal = (xs >= p.size[0] - p.goal_zone)[:, None]
    while True:
        if (reached & goal).any():
            return True
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= free
        if (grown == reached).all():
            return False
        reached = grown


# -- spawn / goal regions and meshes


def terrain_origin(p: ObstacleFieldParams) -> np.ndarray:
    """The tile's origin (tile frame): on the ground, in the middle of the start strip. Isaac Lab puts each env
    there, and moves it with the tile, so offsets from it stay valid."""
    return np.array([p.start_zone / 2, p.size[1] / 2, 0.0])


def spawn_and_goal_regions(p: ObstacleFieldParams, margin: float | None = None):
    """((xmin, xmax), (ymin, ymax)) for spawning and for goals, relative to `terrain_origin`. Both strips are
    shrunk by `margin` (default: drone_radius) from their edges and from the side walls."""
    m = p.drone_radius if margin is None else margin
    ox, oy, _ = (float(v) for v in terrain_origin(p))  # plain floats, not numpy float64
    y0, y1 = p.inner_y
    ys = (y0 + m - oy, y1 - m - oy)
    start = ((m - ox, p.start_zone - m - ox), ys)
    goal = ((p.size[0] - p.goal_zone + m - ox, p.size[0] - m - ox), ys)
    return start, goal


def layout_meshes(layout: Layout) -> list:
    """trimesh meshes in the tile frame: the ground, the side walls and every obstacle."""
    import trimesh

    p = layout.params
    L, W = p.size
    h = p.obstacle_height
    ground = trimesh.Trimesh(
        vertices=np.array([[L, W, 0.0], [L, 0.0, 0.0], [0.0, W, 0.0], [0.0, 0.0, 0.0]]),
        faces=np.array([[1, 0, 2], [2, 3, 1]]),
    )
    meshes = [ground]

    def box(cx, cy, hx, hy, yaw=0.0):
        transform = trimesh.transformations.rotation_matrix(yaw, (0, 0, 1))
        transform[:3, 3] = (cx, cy, h / 2)
        return trimesh.creation.box(extents=(2 * hx, 2 * hy, h), transform=transform)

    if p.side_walls:
        t = p.side_wall_thickness
        meshes += [box(L / 2, t / 2, L / 2, t / 2), box(L / 2, W - t / 2, L / 2, t / 2)]
    for o in layout.obstacles:
        if o.is_round:
            transform = np.eye(4)
            transform[:3, 3] = (*o.center, h / 2)
            meshes.append(trimesh.creation.cylinder(o.radius, h, sections=_CIRCLE_SIDES, transform=transform))
        else:
            meshes.append(box(*o.center, *o.half_extents, o.yaw))
    return meshes


def obstacle_field_terrain(difficulty: float, cfg) -> tuple[list, np.ndarray]:
    """Isaac Lab sub-terrain function: (meshes in the tile frame, origin). `cfg` is an ObstacleFieldTerrainCfg;
    Isaac Lab's generator has set `cfg.seed`."""
    p = ObstacleFieldParams.from_cfg(cfg)
    layout = generate_layout(p, difficulty, layout_rng(getattr(cfg, "seed", None), difficulty))
    GENERATED_LAYOUTS.append(layout)
    return layout_meshes(layout), terrain_origin(p)


# -- collision attribution (evaluation): which obstacle a drone hit

# Every layout obstacle_field_terrain has built in this process, in order. Isaac Lab builds each tile once while
# the env starts, so the env can recover which layout each tile has (tile_layouts).
GENERATED_LAYOUTS: list[Layout] = []
# What a collision can be attributed to: an obstacle kind, or the corridor's side walls.
ATTRIBUTION_KINDS = (*KINDS, "side_wall")


def tile_layouts(layouts: list[Layout], num_rows: int, num_cols: int) -> dict[tuple[int, int], Layout]:
    """{(row, col): layout} for the last num_rows x num_cols layouts built.

    Isaac Lab v2.3.2's curriculum generator (TerrainGenerator._generate_curriculum_terrains) visits tiles column by
    column, rows in order within a column, and gives row r a difficulty in [r, r + 1) / num_rows. So a layout's row
    follows from its difficulty, and its column from how many layouts that row already had.
    """
    recent = layouts[-num_rows * num_cols :]
    if len(recent) != num_rows * num_cols:
        raise ValueError(f"expected {num_rows * num_cols} generated layouts, found {len(recent)}")
    out: dict[tuple[int, int], Layout] = {}
    next_col = [0] * num_rows
    for layout in recent:
        row = min(int(layout.difficulty * num_rows), num_rows - 1)
        col = next_col[row]
        if col >= num_cols:
            raise ValueError(f"row {row} got more than {num_cols} layouts: not the generator order assumed here")
        next_col[row] += 1
        out[(row, col)] = layout
    return out


def _closest_on_segments(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray]:
    """Distance from point p to the nearest of segments a[i]-b[i], and that nearest point."""
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0.0, 1.0)
    pts = a + t[:, None] * ab
    d = np.linalg.norm(pts - p, axis=-1)
    i = int(d.argmin())
    return float(d[i]), pts[i]


def _point_footprint_distance(o: Obstacle, p: np.ndarray) -> tuple[float, np.ndarray]:
    """Distance from point p to footprint o (0 inside), and the nearest point of o."""
    if o.is_round:
        c = np.asarray(o.center, dtype=np.float64)
        v = p - c
        n = float(np.linalg.norm(v))
        return (max(n - o.radius, 0.0), c + v / n * o.radius) if n > 1e-12 else (0.0, p)
    poly = o.polygon()
    if _inside_convex(p[None], poly)[0]:
        return 0.0, p
    return _closest_on_segments(p, poly, np.roll(poly, -1, axis=0))


def nearest_footprint(layout: Layout, point) -> tuple[str, float, np.ndarray, Obstacle | None]:
    """(kind, distance, nearest point, obstacle) of the footprint closest to `point` (x, y in the tile frame). The
    side walls count as kind "side_wall", with obstacle None (the tile's long edges when there are no side walls).
    Distance 0 means inside the footprint."""
    p = np.asarray(point, dtype=np.float64)[:2]
    y0, y1 = layout.params.inner_y
    if p[1] - y0 <= y1 - p[1]:
        best = ("side_wall", float(p[1] - y0), np.array([p[0], y0]), None)
    else:
        best = ("side_wall", float(y1 - p[1]), np.array([p[0], y1]), None)
    for o in layout.obstacles:
        d, q = _point_footprint_distance(o, p)
        if d < best[1]:
            best = (o.kind, d, q, o)
    return best


def nearest_obstacle(layout: Layout, point) -> tuple[str, float, np.ndarray]:
    """(kind, distance, nearest point): nearest_footprint without the obstacle."""
    return nearest_footprint(layout, point)[:3]


def kind_counts(layout: Layout) -> dict[str, int]:
    """How many obstacles of each kind a layout has (a wall counts once, not per segment)."""
    counts = {k: 0 for k in KINDS}
    for o in layout.obstacles:
        counts[o.kind] += 1
    counts["wall"] //= 2
    return counts


# -- lidar replay (evaluation): did the ring ever see the obstacle a drone hit?


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def ray_hit_distances(o: Obstacle, origins: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """(L, R) distance along each horizontal ray to footprint o: ray r of step l starts at origins[l] (L, 2) and
    points at world angle angles[l, r] (radians). inf where the ray misses o; 0 where the origin is inside it."""
    origins = np.asarray(origins, dtype=np.float64)
    u = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (L, R, 2)
    org = origins[:, None, :]  # (L, 1, 2)
    if o.is_round:
        w = org - np.asarray(o.center, dtype=np.float64)
        b = (u * w).sum(-1)
        cc = (w * w).sum(-1) - o.radius**2  # < 0: the origin is inside the circle
        disc = b * b - cc
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        t = np.where(cc <= 0.0, 0.0, t)
        return np.where((disc >= 0.0) & (t >= 0.0), t, np.inf)
    poly = o.polygon()
    best = np.full(u.shape[:2], np.inf)
    for a, e in zip(poly, np.roll(poly, -1, axis=0) - poly):
        denom = _cross(u, e)
        ao = a - org
        with np.errstate(divide="ignore", invalid="ignore"):
            t = _cross(ao, e) / denom
            s_ = _cross(ao, u) / denom
        ok = (np.abs(denom) > 1e-12) & (t >= 0.0) & (s_ >= 0.0) & (s_ <= 1.0)
        best = np.minimum(best, np.where(ok, t, np.inf))
    inside = _inside_convex(origins, poly)
    best[inside] = 0.0
    return best


def obstacle_visibility(
    o: Obstacle,
    positions: np.ndarray,
    yaws: np.ndarray,
    ray_offsets: np.ndarray,
    readings: np.ndarray,
    max_range: float,
    tol: float = 0.05,
) -> dict[str, np.ndarray]:
    """Replay a horizontal lidar trace against one obstacle. positions (L, 2) are the drone's tile-frame positions,
    yaws (L,) its headings, ray_offsets (R,) the ring's ray angles relative to the heading (radians; the ring
    follows yaw only), readings (L, R) the distances the ring reported. Per step (L,):

    - surface: the drone's distance to the obstacle's footprint;
    - returned: some ray reported this obstacle (its reading matches its exact distance to it, within tol);
    - between_rays: the obstacle is within max_range but no ray's path crosses it at all;
    - occluded: some ray's path crosses it within max_range, but every such ray reported something closer.
    """
    positions = np.asarray(positions, dtype=np.float64)
    angles = np.asarray(yaws, dtype=np.float64)[:, None] + np.asarray(ray_offsets, dtype=np.float64)[None, :]
    expected = ray_hit_distances(o, positions, angles)
    crosses = expected <= max_range
    returned = (crosses & (np.abs(np.asarray(readings) - expected) <= tol)).any(axis=1)
    surface = np.array([_point_footprint_distance(o, p)[0] for p in positions])
    in_range = surface <= max_range
    return {
        "surface": surface,
        "in_range": in_range,
        "returned": returned,
        "between_rays": in_range & ~crosses.any(axis=1),
        "occluded": crosses.any(axis=1) & ~returned,
    }


def visibility_summary(vis: dict[str, np.ndarray], step_dt: float) -> dict[str, float]:
    """Collision-level numbers from obstacle_visibility over a trace that ends at the collision:
    never_returned (1/0); first_return_s, how long before the collision a ray first reported the obstacle (capped
    by the trace length; nan if never); first_return_dist, the drone's distance to it then (nan if never);
    between_rays_frac and occluded_frac, the share of in-range steps it was missed each way."""
    returned = vis["returned"]
    steps = len(returned)
    in_range = vis["in_range"]
    if returned.any():
        first = int(np.argmax(returned))
        first_s, first_d = (steps - 1 - first) * step_dt, float(vis["surface"][first])
    else:
        first_s, first_d = float("nan"), float("nan")
    n_in = max(int(in_range.sum()), 1)
    return {
        "never_returned": float(not returned.any()),
        "first_return_s": first_s,
        "first_return_dist": first_d,
        "between_rays_frac": float(vis["between_rays"].sum() / n_in),
        "occluded_frac": float(vis["occluded"].sum() / n_in),
    }


# -- depth-camera replay (Phase 3): what a level pinhole camera should see in a tile


def camera_depth_replay(
    footprints: list[Obstacle],
    position,
    yaw: float,
    col_y: np.ndarray,
    row_z: np.ndarray,
    obstacle_height: float,
    max_range: float,
    pitch: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact distance-to-image-plane image (H, W) of a pinhole camera at tile-frame `position` (x, y, z above
    the ground plane), looking along world heading `yaw`, tilted `pitch` radians nose-down (0: level).

    Pixel (r, c)'s ray has camera-frame direction (1, col_y[c], row_z[r]): x forward, y left, z up, as Isaac Lab's
    pinhole pattern builds them, so the ray reaches the image-plane depth s at s * (1, y, z). Footprints are prisms
    from the ground to obstacle_height; the ground is the plane z = 0 everywhere. Also returns the source of each
    pixel: the footprint's index, -1 for the ground, -2 for nothing within max_range (depth = max_range).
    """
    col_y = np.asarray(col_y, dtype=np.float64)
    row_z = np.asarray(row_z, dtype=np.float64)
    if pitch != 0.0:
        return _camera_depth_replay_pitched(footprints, position, yaw, col_y, row_z, obstacle_height, max_range, pitch)
    x, y, z = (float(v) for v in position)
    angles = yaw + np.arctan2(col_y, 1.0)  # (W,) world heading of each column
    horiz = np.sqrt(1.0 + col_y**2)  # horizontal length of the ray per unit depth
    shape = (len(row_z), len(col_y))
    depth = np.full(shape, np.inf)
    source = np.full(shape, -2, dtype=np.int64)
    for i, o in enumerate(footprints):
        t = ray_hit_distances(o, np.array([[x, y]]), angles[None])[0] / horiz  # (W,) depth of the wall hit
        hit_z = z + row_z[:, None] * t[None, :]
        ok = np.isfinite(t)[None, :] & (hit_z >= 0.0) & (hit_z <= obstacle_height)
        d = np.where(ok, np.broadcast_to(t[None, :], shape), np.inf)
        closer = d < depth
        depth = np.where(closer, d, depth)
        source = np.where(closer, i, source)
    with np.errstate(divide="ignore"):
        ground = np.where(row_z < 0.0, z / -row_z, np.inf)[:, None]  # (H, 1)
    closer = ground < depth
    depth = np.where(closer, ground, depth)
    source = np.where(closer, -1, source)
    beyond = depth >= max_range
    return np.where(beyond, max_range, depth), np.where(beyond, -2, source)


def _camera_depth_replay_pitched(
    footprints: list[Obstacle], position, yaw: float, col_y: np.ndarray, row_z: np.ndarray, obstacle_height: float,
    max_range: float, pitch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """camera_depth_replay for a pitched camera: every pixel's ray has its own heading and elevation."""
    x, y, z = (float(v) for v in position)
    yy, zz = np.meshgrid(col_y, row_z)  # (H, W) camera-frame ray (1, y, z)
    c, s_ = math.cos(pitch), math.sin(pitch)
    fx, fz = c + s_ * zz, -s_ + c * zz  # rotated about the camera's y axis (positive pitch: x towards -z)
    cy, sy = math.cos(yaw), math.sin(yaw)
    dx, dy = cy * fx - sy * yy, sy * fx + cy * yy  # then by the heading, in the horizontal plane
    horiz = np.hypot(dx, dy)  # horizontal length of the ray per unit image-plane depth
    angles = np.arctan2(dy, dx).reshape(1, -1)
    shape = yy.shape
    depth = np.full(shape, np.inf)
    source = np.full(shape, -2, dtype=np.int64)
    for i, o in enumerate(footprints):
        t = ray_hit_distances(o, np.array([[x, y]]), angles)[0].reshape(shape) / horiz
        hit_z = z + fz * t
        ok = np.isfinite(t) & (hit_z >= 0.0) & (hit_z <= obstacle_height)
        closer = ok & (t < depth)
        depth = np.where(closer, t, depth)
        source = np.where(closer, i, source)
    with np.errstate(divide="ignore"):
        ground = np.where(fz < 0.0, z / -fz, np.inf)
    closer = ground < depth
    depth = np.where(closer, ground, depth)
    source = np.where(closer, -1, source)
    beyond = depth >= max_range
    return np.where(beyond, max_range, depth), np.where(beyond, -2, source)


# -- out-of-distribution maps (Phase 4): variants of the training generator, applied before the terrain is built

# Obstacle mixes (cylinder, box, pole, wall weights), used for both ends of the curriculum.
OOD_MIXES = {"poles": (0.0, 0.0, 1.0, 0.0), "walls": (0.25, 0.25, 0.1, 0.4)}
# Sizes outside every training range: bigger cylinders and boxes, thinner poles.
UNSEEN_SIZES = {"cylinder_radius": (0.5, 0.9), "box_side": (1.5, 2.5), "pole_radius": (0.02, 0.03)}


def ood_overrides(
    p: ObstacleFieldParams, density_scale: float = 1.0, mix: str = "", unseen_sizes: bool = False
) -> dict:
    """Field values that turn generator settings `p` into an out-of-distribution variant (empty: unchanged)."""
    out: dict = {}
    if density_scale != 1.0:
        out["density_range"] = tuple(d * density_scale for d in p.density_range)
    if mix:
        if mix not in OOD_MIXES:
            raise ValueError(f"terrain mix must be one of {sorted(OOD_MIXES)}, not {mix!r}")
        out["mix_easy"] = out["mix_hard"] = OOD_MIXES[mix]
    if unseen_sizes:
        out.update(UNSEEN_SIZES)
    return out


# -- near misses (evaluation): how did the drone meet the obstacle it hit?

# Times before a collision at which near_miss_summary reports straight_line_miss.
NEAR_MISS_TIMES_MS = (0, 250, 500)
NEAR_MISS_METRICS = ("radius", "speed", "approach_deg", *(f"line_miss_{t}ms" for t in NEAR_MISS_TIMES_MS))


def straight_line_miss(o: Obstacle, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    """(L,) how far the drone's centre would pass from round footprint o's surface if it flew on in a straight line
    from positions[l] (L, 2) along velocities[l] (L, 2): the half-line's closest approach to o's centre, minus the
    radius. Negative when the line goes through the obstacle (-radius: straight through its centre). A drone that's
    moving away from o (or hovering) passes closest where it is now."""
    if not o.is_round:
        raise ValueError(f"straight_line_miss needs a round footprint, not a {o.kind}")
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    w = np.asarray(o.center, dtype=np.float64) - positions  # (L, 2) drone -> centre
    speed = np.linalg.norm(velocities, axis=-1)
    u = velocities / np.maximum(speed, 1e-9)[:, None]
    along = (w * u).sum(-1)
    perp = np.abs(_cross(u, w))
    closest = np.where((along > 0.0) & (speed > 1e-6), perp, np.linalg.norm(w, axis=-1))
    return closest - o.radius


def near_miss_summary(
    o: Obstacle, positions: np.ndarray, velocities: np.ndarray, step_dt: float
) -> dict[str, float]:
    """Collision-level numbers for the obstacle o a drone hit, from its trace: tile-frame positions (L, 2) and
    horizontal velocities (L, 2), oldest first, ending at the collision step.

    - radius: o's radius (nan unless round);
    - speed: horizontal speed at the collision;
    - approach_deg: angle between the velocity and the direction to o's nearest point at the collision (0: flying
      straight at it; 90: moving along its surface);
    - line_miss_<t>ms: straight_line_miss t ms before the collision (nan unless round, or if the trace is
      shorter than that).
    """
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    out = dict.fromkeys(NEAR_MISS_METRICS, float("nan"))
    v = velocities[-1]
    speed = float(np.linalg.norm(v))
    out["speed"] = speed
    _, q = _point_footprint_distance(o, positions[-1])
    to = q - positions[-1]
    if speed > 1e-6 and np.linalg.norm(to) > 1e-9:
        cos = float(v @ to) / (speed * float(np.linalg.norm(to)))
        out["approach_deg"] = math.degrees(math.acos(min(max(cos, -1.0), 1.0)))
    if o.is_round:
        out["radius"] = o.radius
        miss = straight_line_miss(o, positions, velocities)
        for t in NEAR_MISS_TIMES_MS:
            back = round(t / 1000.0 / step_dt)
            if back < len(positions):
                out[f"line_miss_{t}ms"] = float(miss[len(positions) - 1 - back])
    return out


# -- privileged observation: every tile's footprints as padded arrays


def layout_footprints(layout: Layout) -> list[Obstacle]:
    """The layout's obstacles plus the corridor's side walls (as two boxes inside the tile), when it has them."""
    p = layout.params
    out = list(layout.obstacles)
    if p.side_walls:
        L, W, t = p.size[0], p.size[1], p.side_wall_thickness
        out += [
            Obstacle("side_wall", (L / 2, t / 2), half_extents=(L / 2, t / 2)),
            Obstacle("side_wall", (L / 2, W - t / 2), half_extents=(L / 2, t / 2)),
        ]
    return out


def pack_tiles(
    tiles: dict[tuple[int, int], Layout], num_rows: int, num_cols: int, min_slots: int = 0
) -> dict[str, np.ndarray]:
    """Every tile's footprints (layout_footprints) as arrays of shape (num_rows, num_cols, m, ...), padded to the
    largest tile (and to at least min_slots): centers (.., 2), radii, half_extents (.., 2), yaws, is_round, valid.
    Tile frame."""
    m = max(min_slots, max(len(layout_footprints(lay)) for lay in tiles.values()))
    shape = (num_rows, num_cols, m)
    packed = {
        "centers": np.zeros((*shape, 2), dtype=np.float32),
        "radii": np.zeros(shape, dtype=np.float32),
        "half_extents": np.zeros((*shape, 2), dtype=np.float32),
        "yaws": np.zeros(shape, dtype=np.float32),
        "is_round": np.zeros(shape, dtype=bool),
        "valid": np.zeros(shape, dtype=bool),
    }
    for (r, c), layout in tiles.items():
        for i, o in enumerate(layout_footprints(layout)):
            packed["centers"][r, c, i] = o.center
            packed["radii"][r, c, i] = o.radius
            packed["half_extents"][r, c, i] = o.half_extents
            packed["yaws"][r, c, i] = o.yaw
            packed["is_round"][r, c, i] = o.is_round
            packed["valid"][r, c, i] = True
    return packed


def pack_tile_kinds(tiles: dict[tuple[int, int], Layout], num_rows: int, num_cols: int, slots: int) -> np.ndarray:
    """(num_rows, num_cols, slots) index into ATTRIBUTION_KINDS of each footprint, in pack_tiles' order (pass its
    slot count); -1 for padding."""
    kinds = np.full((num_rows, num_cols, slots), -1, dtype=np.int64)
    for (r, c), layout in tiles.items():
        for i, o in enumerate(layout_footprints(layout)):
            kinds[r, c, i] = ATTRIBUTION_KINDS.index(o.kind)
    return kinds
