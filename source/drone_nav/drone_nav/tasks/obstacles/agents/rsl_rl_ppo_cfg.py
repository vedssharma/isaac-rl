"""PPO for Drone-Nav-Obstacles-v0: the Phase 1 CTBR settings (PLAN.md §4.5), trained longer."""

from isaaclab.utils import configclass

from drone_nav.tasks.goal.agents.rsl_rl_ppo_cfg import DroneNavGoalCTBRPPORunnerCfg


@configclass
class DroneNavObstaclesPPORunnerCfg(DroneNavGoalCTBRPPORunnerCfg):
    # CTBR passed 98% on open goals by iteration ~22; obstacles take longer. A first guess, to revisit after the
    # first run.
    max_iterations = 1500
    save_interval = 100
    experiment_name = "drone_nav_obstacles"
