"""PPO for Drone-Nav-Depth-v0. A placeholder for PLAN.md Phase 3's Option B (end-to-end PPO from depth): RSL-RL 3.1.2
has no CNN policy, so this flattens the image into the MLP. Option A (distillation) uses scripts/distill.py."""

from isaaclab.utils import configclass

from drone_nav.tasks.obstacles.agents.rsl_rl_ppo_cfg import DroneNavObstaclesPPORunnerCfg


@configclass
class DroneNavDepthPPORunnerCfg(DroneNavObstaclesPPORunnerCfg):
    experiment_name = "drone_nav_depth"
