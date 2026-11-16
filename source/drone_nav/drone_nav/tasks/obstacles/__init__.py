"""Drone-Nav-Obstacles-v0: reach a goal through a procedural obstacle field, privileged lidar ring (PLAN.md Phase 2)."""

import gymnasium as gym

gym.register(
    id="Drone-Nav-Obstacles-v0",
    entry_point=f"{__name__}.obstacles_env:DroneNavObstaclesEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.obstacles_env:DroneNavObstaclesEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DroneNavObstaclesPPORunnerCfg",
    },
)
