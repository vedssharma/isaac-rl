"""Drone-Nav-Obstacles-v0: fly through a procedural obstacle field to a goal 20-24 m away (PLAN.md Phase 2).

The Phase 1 CTBR goal task (tasks/goal/goal_env.py, same robot, actions, speed cap and rewards) on the obstacle
terrain (drone_nav/terrain/obstacle_field.py), plus:

- spawns in the tile's start strip and goals in its goal strip, instead of a random heading from the origin;
- a privileged ray-cast "lidar ring" in the observation (PLAN.md §4.2, the Phase 2 teacher's sensing): 3 channels
  at -15 / 0 / +15 degrees x 36 rays, 5 m range, kept level with the horizon (it follows the drone's yaw only);
- a collision termination when the horizontal channel sees an obstacle within collision_radius of the drone's
  centre (it counts as a crash), and a proximity reward term on the same distance.

The math is in drone_nav/mdp/obstacles.py. No curriculum yet: envs start on random difficulty levels and stay
there. eval_terrain_level puts every env on one level, for evaluation; held-out maps come from a terrain seed in
obstacle_field.TEST_SEEDS (Hydra: env.terrain.terrain_generator.seed=1001).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from drone_nav.mdp import obstacles as obs_mdp
from drone_nav.tasks.goal.goal_env import DroneNavGoalCTBREnvCfg, DroneNavGoalEnv, RewardWeightsCfg
from drone_nav.terrain.obstacle_field import ObstacleFieldParams, spawn_and_goal_regions
from drone_nav.terrain.obstacle_field_cfg import OBSTACLE_FIELD_GENERATOR_CFG

# The goal task's observation, which this one extends with the lidar ring.
GOAL_OBS_DIM = DroneNavGoalCTBREnvCfg().observation_space


@configclass
class ObstacleRewardWeightsCfg(RewardWeightsCfg):
    proximity: float = 1.0  # per second, x exp(-clearance / proximity_sigma); 0 turns proximity shaping off


@configclass
class DroneNavObstaclesEnvCfg(DroneNavGoalCTBREnvCfg):
    # Goal-task observation (18) + one value per lidar ray (3 x 36). The env recomputes this from `lidar`, so
    # Hydra overrides of the ring's resolution just work.
    observation_space = GOAL_OBS_DIM + 3 * 36

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBSTACLE_FIELD_GENERATOR_CFG,
        max_init_terrain_level=None,  # every difficulty level from the start (no curriculum yet)
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.52, 0.48)),
        debug_vis=False,
    )

    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        mesh_prim_paths=["/World/ground"],  # the terrain: one static mesh with every obstacle baked in
        ray_alignment="yaw",  # level with the horizon whatever the drone's roll and pitch
        pattern_cfg=patterns.LidarPatternCfg(
            channels=3, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=10.0
        ),
        max_distance=5.0,
        debug_vis=False,
    )
    # Collision when an obstacle is this close to the drone's centre, horizontally. Crazyflie 2.x prop tips are
    # about 0.07 m out; 0.1 m is also the disk the terrain generator guarantees a path for.
    collision_radius = 0.1
    proximity_sigma = 0.3  # m: the proximity penalty is e^-1 of its maximum this far outside collision_radius

    # Spawn and goal points stay this far inside the start and goal strips (from their edges and the side walls).
    strip_margin = 0.5
    # Evaluation: put every env on this difficulty row (0-9) instead of random ones. None for training.
    eval_terrain_level: int | None = None

    rewards: ObstacleRewardWeightsCfg = ObstacleRewardWeightsCfg()


class DroneNavObstaclesEnv(DroneNavGoalEnv):
    cfg: DroneNavObstaclesEnvCfg

    def __init__(self, cfg: DroneNavObstaclesEnvCfg, render_mode: str | None = None, **kwargs):
        pattern = cfg.lidar.pattern_cfg
        self._lidar_channels = pattern.channels
        self._horizontal_channel = obs_mdp.horizontal_channel(pattern.channels, pattern.vertical_fov_range)
        n_rays = pattern.channels * obs_mdp.lidar_num_horizontal(pattern.horizontal_fov_range, pattern.horizontal_res)
        cfg.observation_space = GOAL_OBS_DIM + n_rays

        # The terrain generator gives every tile its own size, so the strips come from the generator's.
        generator = cfg.terrain.terrain_generator
        tile = ObstacleFieldParams.from_cfg(generator.sub_terrains["obstacles"])
        tile.size = tuple(generator.size)
        self._spawn_box, self._goal_box = spawn_and_goal_regions(tile, margin=cfg.strip_margin)

        super().__init__(cfg, render_mode, **kwargs)

        n, dev = self.num_envs, self.device
        self._max_range = self.cfg.lidar.max_distance
        self._lidar_dist = torch.full((n, n_rays), self._max_range, device=dev)
        self._min_dist = torch.full((n,), self._max_range, device=dev)  # closest obstacle, horizontally, this step
        self._min_dist_episode = torch.full((n,), self._max_range, device=dev)
        self._episode_sums["proximity"] = torch.zeros(n, device=dev)

        if self.cfg.eval_terrain_level is not None:
            t = self._terrain
            level = self.cfg.eval_terrain_level
            if not 0 <= level < t.terrain_origins.shape[0]:
                raise ValueError(f"eval_terrain_level must be in [0, {t.terrain_origins.shape[0]}), not {level}")
            t.terrain_levels[:] = level
            t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]

    def _add_sensors(self):
        self._lidar = RayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self._lidar

    def _read_lidar(self) -> torch.Tensor:
        data = self._lidar.data
        return obs_mdp.ray_distances(data.ray_hits_w, data.pos_w, self._max_range)

    # -- MDP

    def _get_observations(self) -> dict:
        obs = super()._get_observations()["policy"]
        # Read again rather than reuse _get_dones' reading: envs that just reset have moved since.
        lidar = obs_mdp.lidar_observation(self._read_lidar(), self._max_range)
        return {"policy": torch.cat([obs, lidar], dim=-1)}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, time_out = super()._get_dones()
        self._lidar_dist = self._read_lidar()
        self._min_dist = obs_mdp.min_horizontal_distance(
            self._lidar_dist, self._lidar_channels, self._horizontal_channel
        )
        self._min_dist_episode = torch.minimum(self._min_dist_episode, self._min_dist)
        hit = obs_mdp.collided(self._min_dist, self.cfg.collision_radius)
        # A collision is a crash: it ends the episode, takes the crash penalty and overrides reaching the goal.
        self._terms["collision"] = hit
        self._terms["goal"] = self._terms["goal"] & ~hit
        return terminated | hit, time_out

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        proximity = obs_mdp.proximity_penalty(
            self._min_dist,
            self.cfg.collision_radius,
            self.cfg.proximity_sigma,
            self._weights["proximity"],
            self.step_dt,
        )
        self._episode_sums["proximity"] += proximity
        return reward + proximity

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self._min_dist_episode[env_ids] = self._max_range

    def _sample_goal_offsets(self, n: int) -> torch.Tensor:
        """Anywhere in the tile's goal strip, at the goal task's altitudes."""
        return obs_mdp.sample_in_box(n, self._goal_box, self.cfg.goal_z_range, device=self.device)

    def _sample_spawn_offsets(self, n: int) -> torch.Tensor:
        """Anywhere in the tile's start strip."""
        return obs_mdp.sample_in_box(n, self._spawn_box, self.cfg.spawn_z_range, device=self.device)

    def _log_episodes(self, env_ids: torch.Tensor) -> None:
        super()._log_episodes(env_ids)
        ended = env_ids[self.episode_length_buf[env_ids] > 0]  # as in super: it wrote a fresh log only if any
        log = self.extras.get("log")
        if len(ended) == 0 or log is None:
            return
        # Closest an obstacle came to the drone's centre during the episode (horizontal, capped at the lidar range).
        log["Metrics/min_obstacle_distance"] = self._min_dist_episode[ended]
        log["Curriculum/terrain_level"] = self._terrain.terrain_levels[ended].float()
