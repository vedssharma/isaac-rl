# Phase 4: robustness

PLAN.md Phase 4 asks whether the policy survives the mismatch between the simulator and a real drone: mass and
inertia, thrust (motor constant), motor lag, drag, wind, control latency, sensor noise. **Exit test: success drops by
less than 10% under full randomization compared with nominal.** This note records the parts as they land.

Status:

- [x] Domain randomization in the env, all off by default (§1)
- [x] The Phase 3 student measured under each factor and all together (§2): **65.1% under full randomization against
  94.75% nominal, a 31% drop.** Mass and wind cause most of it
- [x] Train under randomization (§3–5): training the teacher from scratch under full randomization failed (§3);
  fine-tuning the nominal teacher under it worked (§4). **Its GRU student drops 5.5% under full randomization
  (93.15% → 88.06%): the exit test is met, on one seed** (§5)
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
   episodes). Heavier drones mostly cope (too low: 0.1%); the policy has more thrust in reserve than it uses. Thrust ±15% does much less because its range is narrower, and it
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

## 3. Training the teacher under randomization from scratch: it doesn't learn

`p4-teacher-dr-s42` ([02o13kdu](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/02o13kdu), commit
`179cc45`, $0.56). The Phase 3 teacher's settings: heading and soft-ceiling terms, 4 nearest obstacles, 0.15 m
training margin, mild slowdown. Plus every randomization at §2's full levels, trained from scratch for 1500
iterations.

| Iteration | Success | Too low | Flipped | Mean curriculum level |
|---:|---:|---:|---:|---:|
| 84 | 92% | 0% | 4.6% | 0 |
| 443 | 61% | 17% | 7.6% | 2.7 |
| 1438 | 48% | 23% | 6.3% | 3.1 |

The same teacher without randomization (`vam5p6dd`) reached level 8.6 and never fell. This one stalls at about level 3
from iteration 400 on, with a quarter of its episodes ending on the ground. Held out at level 5 it reaches 19.8%
nominal and 19.5% fully randomized, with 46% too low. Learning to fly, to avoid obstacles, and to cope with every
mismatch at full strength, all from scratch, is more than PPO manages in this budget. It never gets good enough to
climb the curriculum. A student distillation from it was started and stopped after 85 iterations ($0.36).

Stopping it showed two launcher gaps:
- A launcher started in the background from a non-interactive shell ignores SIGINT, so Ctrl-C-style cleanup didn't
  run. Stopping the remote job by hand made the launcher finish normally.
- The launcher's own interrupt only signalled `train.py` and `play.py`, not `distill.py`. That's fixed (`4404f4b`).

The run had exited 0 without a verified checkpoint, so the launcher kept its box, as designed; I deleted it.

## 4. Fine-tuning the nominal teacher under randomization: it works

`train.py --init_wandb_run <run>` (`4404f4b`) starts from another run's final checkpoint: weights, optimizer, empirical
normalization and iteration count, in a new run and log directory.

`p4-teacher-dr-ft-s42` ([tgjesjd2](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/tgjesjd2), $0.47): the
nominal soft-ceiling teacher (`vam5p6dd`, docs/phase3-depth-student.md §9) fine-tuned for 1000 more iterations under
full randomization. The curriculum restarts at level 0. Falls drop from 36% in the very first iteration to under 4%
within about 120 iterations, and to about 1% by iteration 1800. The mean level is 4.9 by iteration 2000 (of 2500) and
still climbing.

| Held-out level 5 (0.1 m rule) | Nominal | Full randomization | Relative drop |
|---|---:|---:|---:|
| Fine-tuned teacher | **93.3%** | **81.35%** | 12.8% |
| Starting point, `vam5p6dd` (§9 of Phase 3) | 92.7% | | |

Fine-tuning costs it nothing at nominal (93.3% against 92.7%). Under full randomization it keeps 81%: collisions
15.3%, flipped 2.15%, too low 0.95%. The teacher, with its lidar and no memory, doesn't quite meet the exit test, but
it's a teacher that can fly in these conditions.

## 5. The GRU student under randomization: the exit test is met (one seed)

`p4-distill-gru-dr-s42` ([0g9bvtcl](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/0g9bvtcl), $1.26):
the Phase 3 GRU recipe (256 units, 48-step windows, 500 iterations, β decaying over 50) distilled from the fine-tuned
teacher. Training ran under full randomization plus `env.depth_noise_scale=2.0`, so the teacher labels states under
the dynamics actually in play, and the student's memory can infer the weight and the wind. Then the same ten
conditions as §2 (`p4-robust-gru-dr-s42`, $0.98):

| Condition | Phase 3 student (§2) | **Randomization-trained student** | Drop from nominal | Collisions | Other failures |
|---|---:|---:|---:|---:|---|
| Nominal | 94.75% | **93.15%** | | 6.65% | flipped 0.2% |
| Mass and inertia ±20% | 87.0% | **92.95%** | 0.2% | 7.0% | |
| Thrust ±15% | 94.65% | 93.6% | −0.5% | 6.2% | |
| Motor lag ≤ 50 ms | 94.25% | 92.5% | 0.7% | 7.3% | |
| Drag ±30% | 95.1% | 92.65% | 0.5% | 6.95% | |
| Latency 0–40 ms | 93.9% | 92.15% | 1.1% | 7.7% | |
| Wind ≤ 2 m/s + gusts | 83.5% | **90.85%** | 2.5% | 9.0% | |
| Observation noise | 95.5% | 92.9% | 0.3% | 7.0% | |
| Depth noise × 2 | 94.0% | 95.2% | −2.2% | 4.7% | |
| **Full** | **65.1%** (31.3% drop) | **88.06%** | **5.5%** | 10.8% | too low 0.35%, flipped 0.5%, time-out 0.25% |

1. **The exit test is met on this seed:** 88.06% under full randomization against 93.15% nominal, a 5.5% relative
   drop (the target is < 10%). The Phase 3 student lost 31.3% under the same conditions.
2. **Mass and wind, the two big losses, are mostly gone:** mass 8.2% → 0.2%, wind 11.9% → 2.5%. Climbing out the
   top under mass (6.6%) is now 0. What's left under full randomization is mostly collisions (10.8%, about 6.5 points
   from obstacles other than poles), i.e. wind pushing the drone into things.
3. **The cost is 1.6 points at nominal** (93.15% against 94.75%), mostly more pole hits (5.15% against 4.6%).
4. **Depth noise × 2 scores higher than nominal** (95.2%). The student was trained at that noise level, so the
   nominal evaluation (× 1) is the slightly unfamiliar one. Training at a randomized noise level would remove the
   asymmetry.
5. One seed. PLAN.md §7 asks for three before a claim.

Cost of §3–5: $0.56 + $0.36 (the stopped student) + $0.47 + $1.26 + $0.98 = $3.63.

**Next:**
1. Seeds 1 and 2 of §4–5: fine-tune the seed-1 and seed-2 soft-ceiling teachers (`56j6n22n`, `arrqs6y5`), distil,
   and run the table. About $2.70 per seed.
2. Then the rest of Phase 4: camera pitch randomization and out-of-distribution maps.

