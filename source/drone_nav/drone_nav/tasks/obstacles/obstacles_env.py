"""Drone-Nav-Obstacles-v0: fly through a procedural obstacle field to a goal 20-24 m away (PLAN.md Phase 2).

The Phase 1 CTBR goal task (tasks/goal/goal_env.py, same robot, actions, speed cap and rewards) on the obstacle
terrain (drone_nav/terrain/obstacle_field.py), plus:

- spawns in the tile's start strip and goals in its goal strip, instead of a random heading from the origin;
- a privileged ray-cast "lidar ring" in the observation (PLAN.md §4.2, the Phase 2 teacher's sensing): 3 channels
  at -15 / 0 / +15 degrees x 36 rays, 5 m range, level with the horizon and FIXED IN THE WORLD: ray i always
  points at the same compass bearing, whatever the drone's heading (see the lidar cfg's ray_alignment note);
- a collision termination when the horizontal channel sees an obstacle within collision_radius of the drone's
  centre (it counts as a crash), and a proximity reward term on the same distance;
- a curriculum (PLAN.md §5.3): every env starts on level 0 and moves up or down a level on its recent success rate.

The math is in drone_nav/mdp/obstacles.py and drone_nav/mdp/curriculum.py. eval_terrain_level puts every env on one
level (and turns the curriculum off), for evaluation; held-out maps come from a terrain seed in
obstacle_field.TEST_SEEDS (Hydra: env.terrain.terrain_generator.seed=1001).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from drone_nav.mdp import obstacles as obs_mdp
from drone_nav.mdp.curriculum import SuccessCurriculum, level_fractions
from drone_nav.tasks.goal.goal_env import DroneNavGoalCTBREnvCfg, DroneNavGoalEnv, RewardWeightsCfg
from drone_nav.terrain import obstacle_field as of
from drone_nav.terrain.obstacle_field import ObstacleFieldParams, spawn_and_goal_regions
from drone_nav.terrain.obstacle_field_cfg import OBSTACLE_FIELD_GENERATOR_CFG

# of.visibility_summary's fields, logged per obstacle kind as Trace/<kind>_<field>.
TRACE_METRICS = ("never_returned", "first_return_s", "first_return_dist", "between_rays_frac", "occluded_frac")

# The goal task's observation, which this one extends with the lidar ring.
GOAL_OBS_DIM = DroneNavGoalCTBREnvCfg().observation_space


@configclass
class ObstacleRewardWeightsCfg(RewardWeightsCfg):
    proximity: float = 1.0  # per second, x exp(-clearance / proximity_sigma); 0.0 turns proximity shaping off


@configclass
class DroneNavObstaclesEnvCfg(DroneNavGoalCTBREnvCfg):
    # Goal-task observation (18) + one value per lidar ray (3 x 36). The env recomputes this from `lidar`, so
    # Hydra overrides of the ring's resolution just work.
    observation_space = GOAL_OBS_DIM + 3 * 36

    # No cloning in Fabric (the goal task uses it): Fabric clones never reach the USD stage, and Isaac Lab sensors
    # count their envs by matching USD prims, so the lidar would see only env_0 and index out of bounds on reset.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=OBSTACLE_FIELD_GENERATOR_CFG,
        # Envs start on random rows 0..this: 0 for the curriculum, which then moves each env up and down; 9 (every
        # level) for training without it. An int, not None: Isaac Lab's Hydra overrides must match the default's
        # type, so a None default can't be set to a level.
        max_init_terrain_level=0,
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
        # Isaac Lab v2.3.2's "yaw" alignment turns the rays' start points with the drone's yaw but NOT their
        # directions (RayCaster._update_ray_infos: ray_directions_w = self.ray_directions). Our rays all start at
        # the drone's centre, so the ring is level and world-fixed: ray i always points at world bearing
        # horizontal angle i. The observation is therefore world-frame while the goal and velocities are
        # body-frame. Found 2026-11-17 by the lidar replay (docs/phase2-obstacles.md §11).
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=3, vertical_fov_range=(-15.0, 15.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=10.0
        ),
        max_distance=5.0,
        debug_vis=False,
    )
    # Make the ring follow the drone's heading (body-yaw frame) instead of staying world-fixed: before every read,
    # the pattern's ray directions are turned by each drone's yaw. The lidar observation then shares the body frame
    # of the goal and velocities. Off by default, so runs before 2026-11-17 reproduce (docs/phase2-obstacles.md §12).
    lidar_follows_yaw: bool = False

    # Collision when an obstacle is this close to the drone's centre, horizontally. Crazyflie 2.x prop tips are
    # about 0.07 m out; 0.1 m is also the disk the terrain generator guarantees a path for.
    collision_radius = 0.1
    proximity_sigma = 0.3  # m: the proximity penalty is e^-1 of its maximum this far outside collision_radius
    # Slow down near obstacles: the over-speed term's limit falls linearly from speed_limit, once the closest obstacle
    # is slow_distance outside collision_radius, to near_speed_limit at contact (mdp.clearance_speed_limit). The
    # policy cruises about 0.25 m/s above whichever limit applies. slow_distance=0.0 (the default) turns it off.
    slow_distance: float = 0.0  # m of clearance
    near_speed_limit: float = 1.5  # m/s

    # Spawn and goal points stay this far inside the start and goal strips (from their edges and the side walls).
    strip_margin = 0.5
    # Evaluation: put every env on this difficulty row (0-9) instead of random ones. -1 (off) for training; an int
    # rather than None so that Hydra can override it (see max_init_terrain_level).
    eval_terrain_level: int = -1

    # Curriculum (PLAN.md §5.3, drone_nav/mdp/curriculum.py): once an env has `curriculum_window` episodes at its
    # level, it moves up if at least curriculum_up_rate of them reached the goal and down if fewer than
    # curriculum_down_rate did. Off during evaluation (eval_terrain_level >= 0). Ablation (b), no curriculum:
    # env.curriculum=False env.terrain.max_init_terrain_level=9.
    curriculum: bool = True
    curriculum_window: int = 5
    curriculum_up_rate: float = 0.8
    curriculum_down_rate: float = 0.4

    # Evaluation diagnostic: attribute each collision to the obstacle hit (its kind, from the tile's layout), with the
    # drone's speed, closing speed and warning time, logged as Collision/* and Exposure/*. Costs a CPU lookup per
    # collision, so it's off for training.
    attribute_collisions: bool = False
    # Warning time = how long the closest horizontal lidar reading had been within this distance before a collision.
    warning_distance: float = 1.0
    # With attribute_collisions, also replay the last trace_steps steps (1 s at the 50 Hz control rate) of the
    # horizontal lidar ring against the obstacle hit: was it ever seen, and from how far? Logged as Trace/*.
    trace_steps: int = 50

    rewards: ObstacleRewardWeightsCfg = ObstacleRewardWeightsCfg()


class DroneNavObstaclesEnv(DroneNavGoalEnv):
    cfg: DroneNavObstaclesEnvCfg

    def __init__(self, cfg: DroneNavObstaclesEnvCfg, render_mode: str | None = None, **kwargs):
        pattern = cfg.lidar.pattern_cfg
        self._lidar_channels = pattern.channels
        self._horizontal_channel = obs_mdp.horizontal_channel(pattern.channels, pattern.vertical_fov_range)
        n_rays = pattern.channels * obs_mdp.lidar_num_horizontal(pattern.horizontal_fov_range, pattern.horizontal_res)
        cfg.observation_space = GOAL_OBS_DIM + n_rays

        if cfg.lidar_follows_yaw and cfg.lidar.ray_alignment != "yaw":
            raise ValueError("lidar_follows_yaw turns the ray directions itself; it needs lidar.ray_alignment='yaw'")
        if cfg.slow_distance > 0.0 and not 0.0 < cfg.near_speed_limit <= cfg.speed_limit:
            raise ValueError(
                f"near_speed_limit must be in (0, speed_limit={cfg.speed_limit}], not {cfg.near_speed_limit}"
            )

        # The terrain generator gives every tile its own size, so the strips come from the generator's.
        generator = cfg.terrain.terrain_generator
        tile = ObstacleFieldParams.from_cfg(generator.sub_terrains["obstacles"])
        tile.size = tuple(generator.size)
        self._spawn_box, self._goal_box = spawn_and_goal_regions(tile, margin=cfg.strip_margin)
        self._tile_params = tile

        super().__init__(cfg, render_mode, **kwargs)
        if self._lidar.num_instances != self.num_envs:
            raise RuntimeError(
                f"the lidar found {self._lidar.num_instances} drones for {self.num_envs} envs; are the envs cloned in "
                "Fabric (scene.clone_in_fabric)? Sensors only see clones on the USD stage."
            )

        n, dev = self.num_envs, self.device
        # The pattern's ray directions (identical for every env at start-up), turned per env when the ring follows yaw.
        self._lidar_base_dirs = self._lidar.ray_directions[0].clone()
        self._max_range = self.cfg.lidar.max_distance
        self._lidar_dist = torch.full((n, n_rays), self._max_range, device=dev)
        self._min_dist = torch.full((n,), self._max_range, device=dev)  # closest obstacle, horizontally, this step
        self._min_dist_episode = torch.full((n,), self._max_range, device=dev)
        self._episode_sums["proximity"] = torch.zeros(n, device=dev)

        if self.cfg.eval_terrain_level >= 0:
            t = self._terrain
            level = self.cfg.eval_terrain_level
            if not 0 <= level < t.terrain_origins.shape[0]:
                raise ValueError(f"eval_terrain_level must be in [0, {t.terrain_origins.shape[0]}), not {level}")
            t.terrain_levels[:] = level
            t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]

        self._num_levels = self._terrain.terrain_origins.shape[0]
        self._curriculum = None
        if self.cfg.curriculum and self.cfg.eval_terrain_level < 0:
            self._curriculum = SuccessCurriculum(
                n,
                self._num_levels,
                self.cfg.curriculum_window,
                self.cfg.curriculum_up_rate,
                self.cfg.curriculum_down_rate,
                device=dev,
            )
        # The level each env's current episode is played on (the terrain's level can change at the episode's end).
        self._episode_level = self._terrain.terrain_levels.clone()

        # How long the closest obstacle has been within warning_distance (collision attribution's warning time).
        self._near_time = torch.zeros(n, device=dev)
        # Speed while something is within warning_distance, summed over the episode (Metrics/speed_near_obstacles).
        self._near_speed_sum = torch.zeros(n, device=dev)
        self._near_steps = torch.zeros(n, device=dev)
        self._tile_layouts = None
        if self.cfg.attribute_collisions:
            self._setup_attribution()

    def _setup_attribution(self):
        """Look up every tile's layout (the terrain function recorded them as Isaac Lab built the tiles), and keep
        per-env buffers describing the collision that ended the current episode, if any."""
        n, dev = self.num_envs, self.device
        rows, cols = self._terrain.terrain_origins.shape[:2]
        self._tile_layouts = of.tile_layouts(of.GENERATED_LAYOUTS, rows, cols)
        self._terrain_origins_np = self._terrain.terrain_origins.cpu().numpy()
        self._origin_in_tile = of.terrain_origin(self._tile_params)[:2]
        counts = torch.zeros(rows, cols, len(of.KINDS))
        for (r, c), layout in self._tile_layouts.items():
            kc = of.kind_counts(layout)
            counts[r, c] = torch.tensor([kc[k] for k in of.KINDS], dtype=torch.float)
        self._kind_counts = counts.to(dev)  # obstacles of each kind per tile, to normalize collision rates
        self._hit_kind = torch.full((n,), -1, dtype=torch.long, device=dev)  # index into of.ATTRIBUTION_KINDS
        self._hit_dist = torch.zeros(n, device=dev)  # drone centre to the attributed footprint (sanity check)
        self._hit_speed = torch.zeros(n, device=dev)  # horizontal speed at the collision
        self._hit_closing = torch.zeros(n, device=dev)  # speed towards the obstacle hit
        self._hit_warning = torch.zeros(n, device=dev)  # how long something had been within warning_distance

        # Lidar replay: a ring buffer of the horizontal channel's readings, the drone's position and heading, written
        # every step (all envs at the same slot), and how many valid steps each env has since its reset.
        k = self.cfg.trace_steps
        pattern = self.cfg.lidar.pattern_cfg
        angles = obs_mdp.lidar_horizontal_angles(pattern.horizontal_fov_range, pattern.horizontal_res)
        self._ray_offsets = np.deg2rad(angles.numpy().astype(np.float64))
        # Which way the ring's ray 0 points: with the drone's yaw only for "base" alignment; for "yaw" and "world",
        # Isaac Lab doesn't rotate ray directions at all (see the lidar cfg), so the ring is world-fixed.
        self._ring_follows_yaw = self.cfg.lidar_follows_yaw or self.cfg.lidar.ray_alignment == "base"
        self._trace_read = torch.zeros(n, k, len(self._ray_offsets), device=dev)
        self._trace_pos = torch.zeros(n, k, 2, device=dev)
        self._trace_yaw = torch.zeros(n, k, device=dev)
        self._trace_len = torch.zeros(n, dtype=torch.long, device=dev)
        self._trace_ptr = 0
        # Per collision (nan otherwise): of.visibility_summary of the obstacle hit.
        self._vis = {name: torch.full((n,), float("nan"), device=dev) for name in TRACE_METRICS}

    def _record_trace(self) -> None:
        n = self.num_envs
        horizontal = self._lidar_dist.view(n, self._lidar_channels, -1)[:, self._horizontal_channel]
        q = self._robot.data.root_quat_w  # (w, x, y, z)
        yaw = obs_mdp.yaw_from_quat(q)
        self._trace_read[:, self._trace_ptr] = horizontal
        self._trace_pos[:, self._trace_ptr] = self._robot.data.root_pos_w[:, :2]
        self._trace_yaw[:, self._trace_ptr] = yaw
        self._trace_ptr = (self._trace_ptr + 1) % self.cfg.trace_steps
        self._trace_len = (self._trace_len + 1).clamp(max=self.cfg.trace_steps)

    def _attribute(self, ids: torch.Tensor) -> None:
        """Fill the collision buffers for envs `ids`, which just collided."""
        pos = self._robot.data.root_pos_w[ids, :2].cpu().numpy()
        vel = self._robot.data.root_lin_vel_w[ids, :2].cpu().numpy()
        rows = self._episode_level[ids].cpu().numpy()
        cols = self._terrain.terrain_types[ids].cpu().numpy()
        # The last trace_len steps of each collided env's trace, oldest first (the current step was just recorded).
        size = self.cfg.trace_steps
        lengths = self._trace_len[ids].cpu().numpy()
        trace_read = self._trace_read[ids].cpu().numpy()
        trace_pos = self._trace_pos[ids].cpu().numpy()
        trace_yaw = self._trace_yaw[ids].cpu().numpy()
        kinds, dists, closing = [], [], []
        vis_rows = {name: [] for name in TRACE_METRICS}
        for k in range(len(ids)):
            r, c = int(rows[k]), int(cols[k])
            shift = self._origin_in_tile - self._terrain_origins_np[r, c, :2]  # world -> tile frame
            p_tile = pos[k] + shift
            kind, dist, nearest, obstacle = of.nearest_footprint(self._tile_layouts[(r, c)], p_tile)
            to = nearest - p_tile
            norm = float(np.linalg.norm(to))
            kinds.append(of.ATTRIBUTION_KINDS.index(kind))
            dists.append(dist)
            closing.append(float(vel[k] @ to) / norm if norm > 1e-9 else float(np.linalg.norm(vel[k])))
            summary = dict.fromkeys(TRACE_METRICS, float("nan"))
            if obstacle is not None and lengths[k] > 0:
                order = (self._trace_ptr - lengths[k] + np.arange(lengths[k])) % size
                vis = of.obstacle_visibility(
                    obstacle,
                    trace_pos[k, order] + shift,
                    trace_yaw[k, order] if self._ring_follows_yaw else np.zeros(lengths[k]),
                    self._ray_offsets,
                    trace_read[k, order],
                    self._max_range,
                )
                summary = of.visibility_summary(vis, self.step_dt)
            for name in TRACE_METRICS:
                vis_rows[name].append(summary[name])
        dev = self.device
        for name in TRACE_METRICS:
            self._vis[name][ids] = torch.tensor(vis_rows[name], dtype=torch.float, device=dev)
        self._hit_kind[ids] = torch.tensor(kinds, dtype=torch.long, device=dev)
        self._hit_dist[ids] = torch.tensor(dists, dtype=torch.float, device=dev)
        self._hit_speed[ids] = torch.linalg.norm(self._robot.data.root_lin_vel_w[ids, :2], dim=1)
        self._hit_closing[ids] = torch.tensor(closing, dtype=torch.float, device=dev)
        self._hit_warning[ids] = self._near_time[ids]

    def _add_sensors(self):
        self._lidar = RayCaster(self.cfg.lidar)
        self.scene.sensors["lidar"] = self._lidar

    def _read_lidar(self) -> torch.Tensor:
        if self.cfg.lidar_follows_yaw:
            # Turn every env's rays by its current heading, then recompute: the sensor would otherwise return the
            # hits it cast (lazily) with the previous directions.
            yaw = obs_mdp.yaw_from_quat(self._robot.data.root_quat_w)
            self._lidar.ray_directions[:] = obs_mdp.yaw_rotate(self._lidar_base_dirs, yaw)
            self._lidar.update(dt=0.0, force_recompute=True)
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
        near = self._min_dist < self.cfg.warning_distance
        self._near_time = torch.where(near, self._near_time + self.step_dt, torch.zeros_like(self._near_time))
        speed = torch.linalg.norm(self._robot.data.root_lin_vel_w[:, :2], dim=1)
        self._near_speed_sum += speed * near.float()
        self._near_steps += near.float()
        if self._tile_layouts is not None:
            self._record_trace()
            if hit.any():
                self._attribute(hit.nonzero(as_tuple=False).squeeze(-1))
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

    def _speed_limit(self) -> float | torch.Tensor:
        # _get_dones (which reads the lidar into _min_dist) runs before _get_rewards in DirectRLEnv.step.
        if self.cfg.slow_distance <= 0.0:
            return self.cfg.speed_limit
        return obs_mdp.clearance_speed_limit(
            self._min_dist,
            self.cfg.collision_radius,
            self.cfg.speed_limit,
            self.cfg.near_speed_limit,
            self.cfg.slow_distance,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        ended = env_ids[self.episode_length_buf[env_ids] > 0]  # not the first reset of a fresh env
        if self._curriculum is not None and len(ended) > 0:
            # Move envs between levels before super() respawns them, so they start on their new tile.
            t = self._terrain
            success = self._terms["goal"][ended]
            t.terrain_levels[ended] = self._curriculum.update(ended, success, t.terrain_levels[ended])
            t.env_origins[ended] = t.terrain_origins[t.terrain_levels[ended], t.terrain_types[ended]]
        super()._reset_idx(env_ids)  # logs the ended episodes (at _episode_level), then respawns
        self._episode_level[env_ids] = self._terrain.terrain_levels[env_ids]
        self._min_dist_episode[env_ids] = self._max_range
        self._near_time[env_ids] = 0.0
        self._near_speed_sum[env_ids] = 0.0
        self._near_steps[env_ids] = 0.0
        if self._tile_layouts is not None:
            self._hit_kind[env_ids] = -1
            self._trace_len[env_ids] = 0
            for value in self._vis.values():
                value[env_ids] = float("nan")

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
        # Mean horizontal speed while an obstacle was within warning_distance, over episodes that came that close.
        steps = self._near_steps[ended]
        log["Metrics/speed_near_obstacles"] = (self._near_speed_sum[ended] / steps.clamp(min=1.0))[steps > 0]
        log["Curriculum/terrain_level"] = self._episode_level[ended].float()  # the level each episode was played on
        # Difficulty histogram over all envs (PLAN.md §5.3: "your best progress chart").
        for level, fraction in enumerate(level_fractions(self._terrain.terrain_levels, self._num_levels)):
            log[f"Curriculum/level_{level}"] = fraction
        if self._tile_layouts is not None:
            # Per ended episode: 1 if it ended in a collision with this kind (the mean is that kind's rate), and how
            # many of each kind its tile had. Speeds and times only over the episodes that collided.
            kind = self._hit_kind[ended]
            for j, name in enumerate(of.ATTRIBUTION_KINDS):
                log[f"Collision/{name}"] = (kind == j).float()
            rows, cols = self._episode_level[ended], self._terrain.terrain_types[ended]
            for j, name in enumerate(of.KINDS):
                log[f"Exposure/{name}"] = self._kind_counts[rows, cols, j]
            collided = kind >= 0
            log["Collision/speed"] = self._hit_speed[ended][collided]
            log["Collision/closing_speed"] = self._hit_closing[ended][collided]
            log["Collision/warning_time_s"] = self._hit_warning[ended][collided]
            log["Collision/attribution_distance"] = self._hit_dist[ended][collided]
            # Lidar replay per obstacle kind, over that kind's collisions: never_returned is a rate; the rest are
            # means over the collisions where they're defined (first_return_* only where the obstacle was seen).
            for j, name in enumerate(of.KINDS):
                of_kind = kind == j
                for metric in TRACE_METRICS:
                    values = self._vis[metric][ended][of_kind]
                    log[f"Trace/{name}_{metric}"] = values[~torch.isnan(values)]
