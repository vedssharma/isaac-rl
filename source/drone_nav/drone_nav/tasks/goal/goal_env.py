"""Drone-Nav-Goal-v0: fly a Crazyflie to a goal 10-20 m away, no obstacles (PLAN.md Phase 1).

Direct workflow, starting from Isaac Lab's Isaac-Quadcopter-Direct-v0 (v2.3.2, BSD-3-Clause): same robot,
same thrust + body-torque action (PLAN.md §4.1 v1), but with far goals, a progress reward, goal-reached and
crash terminations, and per-episode success metrics. The math is in drone_nav/mdp/goal.py.
"""

from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms

from drone_nav.mdp import goal as mdp

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import SPHERE_MARKER_CFG  # isort: skip


@configclass
class RewardWeightsCfg:
    """PLAN.md §4.3 weights (Hydra: env.rewards.<name>=<value>). See mdp.compute_reward_terms for units."""

    progress: float = 1.0  # per metre closed toward the goal
    goal: float = 10.0  # once, on reaching the goal
    crash: float = 10.0  # once, on any crash termination
    smooth: float = 0.05  # per second, x ||a_t - a_{t-1}||^2
    ang_rate: float = 0.01  # per second, x ||omega||^2 (rad/s)
    upright: float = 0.5  # per second, x (1 - up_z)
    time: float = 0.1  # per second


@configclass
class DroneNavGoalEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 2
    action_space = 4
    # lin vel (3) + ang vel (3) + gravity (3) + last action (4) + goal direction (3) + goal distance (1) + altitude (1)
    observation_space = 18
    state_space = 0
    debug_vis = True

    # simulation (as Isaac-Quadcopter-Direct-v0)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # Drones fly far outside their 2.5 m cell. That's fine: collisions between envs are filtered.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )
    # Videos (--video) follow env 0's drone instead of showing the whole grid from far away.
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root", asset_name="robot", env_index=0, eye=(-2.5, -2.5, 1.5), lookat=(0.0, 0.0, 0.0)
    )

    # robot and action scaling (as Isaac-Quadcopter-Direct-v0)
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # task (PLAN.md §4.4)
    goal_min_dist = 10.0  # horizontal, m
    goal_max_dist = 20.0
    goal_z_range = (0.75, 2.5)  # altitude, m
    goal_radius = 0.5  # reaching it ends the episode as a success
    spawn_z_range = (1.0, 1.5)
    spawn_lin_vel_std = 0.2  # m/s, per axis
    min_altitude = 0.1
    max_altitude = 4.0
    off_course_margin = 5.0  # crash if this much further from the goal than at the start
    obs_goal_dist_clip = 10.0

    rewards: RewardWeightsCfg = RewardWeightsCfg()


class DroneNavGoalEnv(DirectRLEnv):
    cfg: DroneNavGoalEnvCfg

    def __init__(self, cfg: DroneNavGoalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(n, action_dim, device=dev)
        self._prev_actions = torch.zeros(n, action_dim, device=dev)
        self._thrust = torch.zeros(n, 1, 3, device=dev)
        self._moment = torch.zeros(n, 1, 3, device=dev)

        self._goal_pos_w = torch.zeros(n, 3, device=dev)
        self._dist = torch.zeros(n, device=dev)  # to the goal, this step
        self._prev_dist = torch.zeros(n, device=dev)
        self._start_dist = torch.zeros(n, device=dev)
        self._prev_pos_w = torch.zeros(n, 3, device=dev)
        self._path_len = torch.zeros(n, device=dev)
        self._terms = {k: torch.zeros(n, dtype=torch.bool, device=dev) for k in ("goal", *mdp.CRASH_REASONS)}
        self._episode_sums = {k: torch.zeros(n, device=dev) for k in mdp.REWARD_TERMS}

        self._body_id = self._robot.find_bodies("body")[0]
        robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        gravity = torch.tensor(self.sim.cfg.gravity, device=dev).norm()
        self._robot_weight = (robot_mass * gravity).item()
        self._weights = self.cfg.rewards.to_dict()

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -- action

    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id, forces=self._thrust, torques=self._moment
        )

    # -- MDP

    def _get_observations(self) -> dict:
        data = self._robot.data
        goal_pos_b, _ = subtract_frame_transforms(data.root_pos_w, data.root_quat_w, self._goal_pos_w)
        obs = torch.cat(
            [
                data.root_lin_vel_b,
                data.root_ang_vel_b,
                data.projected_gravity_b,
                self._actions,
                mdp.goal_observation(goal_pos_b, self.cfg.obs_goal_dist_clip),
                data.root_pos_w[:, 2:3],  # altitude: the ground plane is at z = 0
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # DirectRLEnv never clears extras, so a step without resets would hand RSL-RL the last step's episode
        # log again. Clear it here; _reset_idx writes a fresh one when episodes end.
        self.extras.pop("log", None)
        pos = self._robot.data.root_pos_w
        self._dist = torch.linalg.norm(self._goal_pos_w - pos, dim=1)
        self._path_len += torch.linalg.norm(pos - self._prev_pos_w, dim=1)
        self._prev_pos_w = pos.clone()
        self._terms = mdp.compute_terminations(
            self._dist,
            self._start_dist,
            pos[:, 2],
            mdp.up_z(self._robot.data.projected_gravity_b),
            goal_radius=self.cfg.goal_radius,
            min_altitude=self.cfg.min_altitude,
            max_altitude=self.cfg.max_altitude,
            off_course_margin=self.cfg.off_course_margin,
        )
        terminated = torch.stack(list(self._terms.values())).any(dim=0)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        crashed = torch.stack([self._terms[k] for k in mdp.CRASH_REASONS]).any(dim=0)
        terms = mdp.compute_reward_terms(
            prev_dist=self._prev_dist,
            dist=self._dist,
            reached=self._terms["goal"],
            crashed=crashed,
            action=self._actions,
            prev_action=self._prev_actions,
            ang_vel_b=self._robot.data.root_ang_vel_b,
            upz=mdp.up_z(self._robot.data.projected_gravity_b),
            step_dt=self.step_dt,
            weights=self._weights,
        )
        self._prev_dist = self._dist.clone()
        for key, value in terms.items():
            self._episode_sums[key] += value
        return torch.stack(list(terms.values())).sum(dim=0)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._log_episodes(env_ids)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        n = len(env_ids)
        origins = self._terrain.env_origins[env_ids]
        self._actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
        for value in self._episode_sums.values():
            value[env_ids] = 0.0

        # New goal: random heading, 10-20 m away horizontally (PLAN.md §4.4)
        self._goal_pos_w[env_ids] = origins + mdp.sample_goal_offsets(
            n, self.cfg.goal_min_dist, self.cfg.goal_max_dist, self.cfg.goal_z_range, device=self.device
        )
        # Spawn above the env origin with a random yaw and a small random velocity
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = origins
        z_lo, z_hi = self.cfg.spawn_z_range
        root_state[:, 2] = torch.empty(n, device=self.device).uniform_(z_lo, z_hi)
        yaw = torch.empty(n, device=self.device).uniform_(-torch.pi, torch.pi)
        zeros = torch.zeros_like(yaw)
        root_state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root_state[:, 7:10] = torch.randn(n, 3, device=self.device) * self.cfg.spawn_lin_vel_std
        root_state[:, 10:13] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids], self._robot.data.default_joint_vel[env_ids], None, env_ids
        )

        start_dist = torch.linalg.norm(self._goal_pos_w[env_ids] - root_state[:, :3], dim=1)
        self._start_dist[env_ids] = start_dist
        self._prev_dist[env_ids] = start_dist
        self._dist[env_ids] = start_dist
        self._prev_pos_w[env_ids] = root_state[:, :3]
        self._path_len[env_ids] = 0.0

    def _log_episodes(self, env_ids: torch.Tensor) -> None:
        """Per-episode values for the episodes ending now. RSL-RL averages every element of each tensor over
        the iteration, so a 0/1 vector logs an exact rate (e.g. Metrics/success_rate)."""
        ended = env_ids[self.episode_length_buf[env_ids] > 0]  # skip the initial reset of fresh envs
        if len(ended) == 0:
            return
        reached = self._terms["goal"][ended]
        terminated = self.reset_terminated[ended]
        steps = self.episode_length_buf[ended].float()
        log = {f"Episode_Reward/{k}": self._episode_sums[k][ended] for k in mdp.REWARD_TERMS}
        log.update({f"Episode_Termination/{k}": self._terms[k][ended].float() for k in self._terms})
        log["Episode_Termination/time_out"] = (self.reset_time_outs[ended] & ~terminated).float()
        log["Metrics/success_rate"] = reached.float()
        log["Metrics/final_distance_to_goal"] = self._dist[ended]
        log["Metrics/mean_speed"] = self._path_len[ended] / (steps * self.step_dt)
        # Only over successful episodes; may be empty for an iteration (then RSL-RL logs nan).
        log["Metrics/time_to_goal_s"] = steps[reached] * self.step_dt
        log["Metrics/path_efficiency"] = self._start_dist[ended][reached] / self._path_len[ended][reached].clamp(
            min=1e-6
        )
        self.extras["log"] = log

    def close(self):
        # DirectRLEnv.close() deletes the scene before the viewport camera controller, which only unsubscribes
        # its per-frame callback in __del__. Following a robot (our viewer cfg), that callback then hits the
        # deleted scene and prints an AttributeError at the end of every run. Unsubscribe first.
        controller = getattr(self, "viewport_camera_controller", None)
        handle = getattr(controller, "_viewport_camera_update_handle", None)
        if handle is not None:
            handle.unsubscribe()
            controller._viewport_camera_update_handle = None
        super().close()

    # -- debug visualization

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = self.cfg.goal_radius
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._goal_pos_w)
