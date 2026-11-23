# Phase 4: robustness

PLAN.md Phase 4 asks whether the policy survives the mismatch between the simulator and a real drone: mass and
inertia, thrust (motor constant), motor lag, drag, wind, control latency, sensor noise. **Exit test: success drops by
less than 10% under full randomization compared with nominal.** This note records the parts as they land.

Status:

- [x] Domain randomization in the env, all off by default (§1)
- [x] The Phase 3 student measured under each factor and all together (§2): **65.1% under full randomization against
  94.75% nominal, a 31% drop.** Mass and wind cause most of it
- [ ] Train under randomization
- [ ] Camera pitch randomization; out-of-distribution maps (higher density, new obstacle kinds)

**The policy under test** is the Phase 3 result (docs/phase3-depth-student.md §10.1): the GRU depth student, seed 42
([uke1cqz8](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/uke1cqz8)), 94.75% / 5.2% on the 60
held-out level-5 layouts. It was trained with none of this.

## 1. Domain randomization

`env.randomization.*` (`RandomizationCfg` in `tasks/goal/goal_env.py`, so the goal, obstacle and depth tasks all have
it). The math is in `drone_nav/mdp/randomization.py`, pure torch with CPU tests (commit `bae779f`). Each env draws new
values at every reset:

| Setting | What it does |
|---|---|
| `mass_spread` | mass × U(1 − s, 1 + s), with the inertia scaled the same; thrust commands and the rate loop keep using the nominal values |
| `inertia_spread` | inertia × a further U(1 − s, 1 + s) |
| `thrust_spread` | true thrust per commanded thrust × U(1 − s, 1 + s): motor-constant mismatch |
| `motor_tau_max` | thrust and torques follow their commands with a first-order lag, τ ~ U(0, max) s |
| `drag_spread` | drag coefficient × U(1 − s, 1 + s) |
| `latency_max_steps` | 0 … N control steps (20 ms each) between a command and the motors; the observation keeps the policy's own last command |
| `wind_speed_max`, `gust_std`, `gust_tau` | a steady horizontal wind (random direction, speed ~ U(0, max)) plus Ornstein–Uhlenbeck gusts, acting through the drag model as the drone's speed relative to the air |
| `obs_noise_scale` | Gaussian noise on the 18 goal and motion inputs: 0.1 m/s on velocity, 0.05 rad/s on body rates, 0.02 on gravity and goal direction, 0.1 m on goal distance, 0.05 m on altitude (× scale) |
| `env.depth_noise_scale` (depth task) | multiplies the depth noise model's Gaussian, range-dependent and dropout terms |

Every setting at 0 is the nominal task and draws no extra random numbers: the nominal evaluation below reproduces
§10's 94.75% exactly. Not yet randomized: camera pitch (it needs per-env access to Isaac Lab's camera offset).

`launch.py --script play` now also takes `--eval NAME:overrides`, to evaluate one checkpoint under several conditions
on one box (CPU-tested). It saves a box start-up of about 5 minutes per condition.

## 2. The Phase 3 student under randomization

`p4-robust-gru-s42` ($1.03, one box): the student on the 60 held-out level-5 layouts, 2000 episodes per condition, at
PLAN.md §5.5's levels. Each factor alone, then all of them together:

| Condition | Settings | Success | Drop (relative) | Collisions | Too high / too low / flipped | Time-out |
|---|---|---:|---:|---:|---|---:|
| Nominal | | **94.75%** | | 5.2% | 0 / 0 / 0% | 0.05% |
| Thrust | ±15% | 94.65% | 0.1% | 5.1% | 0.15 / 0.05 / 0% | 0.05% |
| Motor lag | τ ≤ 50 ms | 94.25% | 0.5% | 5.45% | 0 / 0.05 / 0% | 0.25% |
| Drag | ±30% | 95.1% | −0.4% | 4.8% | 0 / 0.05 / 0% | 0.05% |
| Latency | 0–40 ms | 93.9% | 0.9% | 5.8% | 0 / 0.05 / 0% | 0.25% |
| Observation noise | × 1 | 95.5% | −0.8% | 4.5% | 0 / 0 / 0% | 0% |
| Depth noise | × 2 | 94.0% | 0.8% | 5.85% | 0 / 0 / 0% | 0.15% |
| **Mass and inertia** | ±20% each | **87.0%** | **8.2%** | 4.9% | **6.6** / 0.1 / 0.85% | 0.55% |
| **Wind** | ≤ 2 m/s + 0.5 m/s gusts | **83.5%** | **11.9%** | **15.9%** | 0.25 / 0.25 / 0% | 0.1% |
| **Full** | all of the above | **65.1%** | **31.3%** | **20.1%** | **9.0 / 4.5 / 1.3%** | 0% |

1. **The student is already robust to most of the list:** thrust, motor lag, drag, latency, observation noise and
   doubled depth noise each cost under 1%. The rate loop and a few steps of latency are within what it handles.
2. **Mass is the first problem, through the altitude channel.** Thrust is commanded relative to the *nominal* weight,
   so a 20% lighter drone gets 25% more lift than it expects and drifts up through the 4 m ceiling (6.6% of
   episodes); a heavier one sinks (too low). Thrust ±15% does much less because its range is narrower, and it
   doesn't come with an inertia change (the 0.85% flips are the inertia part). The policy has never had to work out
   a weight it doesn't know.
3. **Wind is the second:** a steady 0–2 m/s wind with gusts pushes the drone into obstacles (collisions 5.2% →
   15.9%). The student flies by sight with no notion of drift, and 2 m/s is two thirds of its 3 m/s cruise.
4. **Together they compound:** 65.1%, with every failure mode up at once. The exit test (< 10% drop) is not met.

**Next: train under randomization.** The standard fix is to make these factors part of training:
- retrain the teacher with `env.randomization.*` on, at these levels;
- distil the GRU student under the same randomization. Its memory can infer the drone's weight and the wind from
  how the drone responds, which a feed-forward policy can't.

Then re-run this table. Mass and wind are the ones to watch.
