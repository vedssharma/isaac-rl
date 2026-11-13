# isaac-rl

Reinforcement learning for quadrotor navigation through cluttered obstacle fields, built on
NVIDIA Isaac Sim / Isaac Lab, with experiments run on NVIDIA Brev GPU instances.

See [PLAN.md](PLAN.md) for the full project plan and roadmap.

## Getting started

1. Accounts, API keys and the Brev CLI: follow [docs/setup.md](docs/setup.md), then run `python scripts/check_env.py`.
2. Isaac Sim / Isaac Lab on a Brev GPU instance: [scripts/brev/install_isaac.sh](scripts/brev/install_isaac.sh).
3. Train with [scripts/train.py](scripts/train.py) (see "Training runs" in [docs/setup.md](docs/setup.md)), or from
   your laptop on a fresh GPU instance with [scripts/brev/launch.py](scripts/brev/launch.py) (§7 there).

## Progress notes

- [Phase 0 wrap-up: our own `train.py` and the first drone videos](docs/phase0-train-wrapper-and-video.md)
- [Phase 1: our own environment, `Drone-Nav-Goal-v0`](docs/phase1-goal-env.md): 100% success on goals 10–20 m away
