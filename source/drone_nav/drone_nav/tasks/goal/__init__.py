"""Drone-Nav-Goal-v0: reach a goal 10-20 m away, no obstacles (PLAN.md Phase 1)."""

import gymnasium as gym

gym.register(
    id="Drone-Nav-Goal-v0",
    entry_point=f"{__name__}.goal_env:DroneNavGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goal_env:DroneNavGoalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DroneNavGoalPPORunnerCfg",
    },
)

gym.register(
    id="Drone-Nav-Goal-CTBR-v0",
    entry_point=f"{__name__}.goal_env:DroneNavGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.goal_env:DroneNavGoalCTBREnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:DroneNavGoalCTBRPPORunnerCfg",
    },
)
