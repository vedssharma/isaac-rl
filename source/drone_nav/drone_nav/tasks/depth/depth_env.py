"""Drone-Nav-Depth-v0: Drone-Nav-Obstacles-v0 seen through a forward-facing depth camera (PLAN.md Phase 3).

The world, rewards, terminations and curriculum are the obstacle task's. What changes is the observation:

- "policy" (the student's, PLAN.md §4.2): the goal task's 18 inputs (velocities, gravity, last action, goal,
  altitude) followed by a depth image from a ray-cast pinhole camera on the drone's nose, 64 x 48 pixels, 90 degrees
  horizontal FOV, 6 m range, with a stereo-like noise model (drone_nav/sensors/depth.py), scaled to [0, 1] and
  flattened row-major.
- "teacher": the obstacle task's privileged observation (lidar ring and, by default, the 4 nearest obstacles),
  unchanged, so a frozen Phase 2 teacher can label the student's states (scripts/distill.py).

Isaac Lab's RayCasterCamera casts against the terrain mesh with Warp (no RT cores), and unlike the lidar's "yaw"
alignment it applies the camera's full orientation to its rays, so the image pitches and rolls with the drone.

The defaults match the Phase 2 teacher (docs/phase2-obstacles.md §18): the mild slowdown and the 4 nearest
obstacles. It was trained at collision_radius=0.15; the student keeps the evaluation rule, 0.1 m.
"""

from __future__ import annotations

import torch
from isaaclab.sensors import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.utils import configclass

from drone_nav.sensors import depth as depth_mdp
from drone_nav.tasks.obstacles.obstacles_env import GOAL_OBS_DIM, DroneNavObstaclesEnv, DroneNavObstaclesEnvCfg

# Pinhole intrinsics (Isaac Lab units: cm, but only ratios matter): 90-degree horizontal FOV, square pixels.
DEPTH_WIDTH, DEPTH_HEIGHT = 64, 48
FOCAL_LENGTH, HORIZONTAL_APERTURE = 10.0, 20.0


@configclass
class DroneNavDepthEnvCfg(DroneNavObstaclesEnvCfg):
    observation_space = GOAL_OBS_DIM + DEPTH_WIDTH * DEPTH_HEIGHT  # recomputed from depth_camera by the env

    # The Phase 2 teacher's settings, so its "teacher" observation is the one it was trained on.
    slow_distance: float = 1.0
    near_speed_limit: float = 1.5
    nearest_obstacles_obs: int = 4

    depth_camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        mesh_prim_paths=["/World/ground"],
        # On the nose, looking along the body's +x axis, level with the body ("world" convention: x forward, z up).
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.04, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=FOCAL_LENGTH,
            horizontal_aperture=HORIZONTAL_APERTURE,
            width=DEPTH_WIDTH,
            height=DEPTH_HEIGHT,
        ),
        max_distance=6.0,
        depth_clipping_behavior="max",  # beyond range (and misses) read max_distance
        debug_vis=False,
    )

    # Depth noise (drone_nav/sensors/depth.py add_depth_noise). depth_noise=False gives clean depth (scripts/
    # check_depth.py uses it to check the camera against the layouts' exact geometry).
    depth_noise: bool = True
    depth_gaussian_std: float = 0.01  # m
    depth_range_std: float = 0.005  # m per m^2: std grows with depth^2, 0.19 m at 6 m
    depth_dropout_prob: float = 0.01
    depth_edge_dropout_prob: float = 0.3
    depth_edge_threshold: float = 0.3  # m
    # Phase 4: multiplies the noise model's Gaussian std, range-dependent std and both dropout probabilities.
    depth_noise_scale: float = 1.0


class DroneNavDepthEnv(DroneNavObstaclesEnv):
    cfg: DroneNavDepthEnvCfg

    def __init__(self, cfg: DroneNavDepthEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if self._camera.num_instances != self.num_envs:
            raise RuntimeError(f"the depth camera found {self._camera.num_instances} drones for {self.num_envs} envs")

    def _policy_obs_dim(self, cfg: DroneNavDepthEnvCfg) -> int:
        pattern = cfg.depth_camera.pattern_cfg
        return GOAL_OBS_DIM + pattern.width * pattern.height

    def _add_sensors(self):
        super()._add_sensors()  # the lidar ring: collisions, rewards and the teacher's observation use it
        self._camera = RayCasterCamera(self.cfg.depth_camera)
        self.scene.sensors["depth_camera"] = self._camera

    @property
    def depth_image_shape(self) -> tuple[int, int]:
        pattern = self.cfg.depth_camera.pattern_cfg
        return pattern.height, pattern.width

    def _record_image_shape(self) -> tuple[int, int] | None:
        self._record_image_range = self.cfg.depth_camera.max_distance
        return self.depth_image_shape

    def _record_image(self) -> torch.Tensor:
        """The clean depth image (noise is random, so it would only blur the record), 0-255 over the camera range."""
        scaled = self.clean_depth() / self.cfg.depth_camera.max_distance * 255.0
        return scaled.round().clamp(0, 255).to(torch.uint8)

    def clean_depth(self) -> torch.Tensor:
        """(n, H, W) noise-free distance to the image plane in metres, capped at the camera's range."""
        return self._camera.data.output["distance_to_image_plane"][..., 0]

    def _read_depth(self) -> torch.Tensor:
        depth = self.clean_depth()
        if not self.cfg.depth_noise:
            return depth
        k = self.cfg.depth_noise_scale
        return depth_mdp.add_depth_noise(
            depth,
            self.cfg.depth_camera.max_distance,
            self.cfg.depth_gaussian_std * k,
            self.cfg.depth_range_std * k,
            min(self.cfg.depth_dropout_prob * k, 1.0),
            min(self.cfg.depth_edge_dropout_prob * k, 1.0),
            self.cfg.depth_edge_threshold,
        )

    def _get_observations(self) -> dict:
        teacher = super()._get_observations()["policy"]
        depth = depth_mdp.depth_observation(self._read_depth(), self.cfg.depth_camera.max_distance)
        return {"policy": torch.cat([teacher[:, :GOAL_OBS_DIM], depth], dim=-1), "teacher": teacher}
