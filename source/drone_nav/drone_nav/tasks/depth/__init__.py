"""Drone-Nav-Depth-v0: the obstacle task seen through a forward depth camera (PLAN.md Phase 3)."""

import gymnasium as gym

gym.register(
    id="Drone-Nav-Depth-v0",
    entry_point=f"{__name__}.depth_env:DroneNavDepthEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.depth_env:DroneNavDepthEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DroneNavDepthPPORunnerCfg",
    },
)
