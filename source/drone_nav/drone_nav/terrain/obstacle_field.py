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
    ox, oy, _ = terrain_origin(p)
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
    return layout_meshes(layout), terrain_origin(p)
