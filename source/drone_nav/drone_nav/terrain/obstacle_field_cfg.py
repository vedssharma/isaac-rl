"""Isaac Lab configs for the obstacle-field terrain (drone_nav/terrain/obstacle_field.py). Needs Isaac Lab.

ObstacleFieldTerrainCfg repeats ObstacleFieldParams' fields so Hydra can override each one
(env.terrain.terrain_generator.sub_terrains.obstacles.<name>=...). tests/test_obstacle_field.py checks that the two
field lists match, since this file can't be imported without Isaac Lab.
"""

from __future__ import annotations

from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass

from .obstacle_field import TRAIN_SEED, ObstacleFieldParams, obstacle_field_terrain

_D = ObstacleFieldParams()


@configclass
class ObstacleFieldTerrainCfg(SubTerrainBaseCfg):
    """One corridor tile of obstacles. See ObstacleFieldParams for what each field means."""

    function = obstacle_field_terrain

    size: tuple[float, float] = _D.size
    start_zone: float = _D.start_zone
    goal_zone: float = _D.goal_zone
    zone_buffer: float = _D.zone_buffer
    obstacle_height: float = _D.obstacle_height
    side_walls: bool = _D.side_walls
    side_wall_thickness: float = _D.side_wall_thickness

    density_range: tuple[float, float] = _D.density_range
    min_gap_range: tuple[float, float] = _D.min_gap_range
    mix_easy: tuple[float, float, float, float] = _D.mix_easy
    mix_hard: tuple[float, float, float, float] = _D.mix_hard

    cylinder_radius: tuple[float, float] = _D.cylinder_radius
    pole_radius: tuple[float, float] = _D.pole_radius
    box_side: tuple[float, float] = _D.box_side
    wall_thickness: float = _D.wall_thickness
    door_width: tuple[float, float] = _D.door_width
    min_wall_segment: float = _D.min_wall_segment

    drone_radius: float = _D.drone_radius
    grid_resolution: float = _D.grid_resolution
    max_attempts: int = _D.max_attempts
    max_layout_tries: int = _D.max_layout_tries


# Training terrain: 10 rows of difficulty (row i is level i, PLAN.md §5.1) x 20 random tiles per level.
# With curriculum=True, Isaac Lab gives row i a difficulty in [i/10, (i+1)/10) and lets the env move each robot
# between rows. For held-out evaluation maps, use .replace(seed=s) with s from obstacle_field.TEST_SEEDS.
OBSTACLE_FIELD_GENERATOR_CFG = TerrainGeneratorCfg(
    seed=TRAIN_SEED,
    curriculum=True,
    size=_D.size,
    border_width=0.0,
    num_rows=10,
    num_cols=20,
    use_cache=False,
    sub_terrains={"obstacles": ObstacleFieldTerrainCfg()},
)
