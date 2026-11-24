# Phase 4: robustness

PLAN.md Phase 4 asks whether the policy survives the mismatch between the simulator and a real drone: mass and
inertia, thrust (motor constant), motor lag, drag, wind, control latency, sensor noise. **Exit test: success drops by
less than 10% under full randomization compared with nominal.** This note records the parts as they land.

Status:

- [x] Domain randomization in the env, all off by default (§1)
- [x] The Phase 3 student measured under each factor and all together (§2): **65.1% under full randomization against
  94.75% nominal, a 31% drop.** Mass and wind cause most of it
- [x] Train under randomization (§3–5): training the teacher from scratch under full randomization failed (§3);
  fine-tuning the nominal teacher under it worked (§4). Its GRU student drops 5.5% under full randomization on seed 42
  (§5). **Over 3 seeds (§5.1) the mean drop is 9.7%, just inside the 10% target, but seed 1 drops 16.1%:** wind is
  the weak point
- [x] Stronger training wind and longer teacher fine-tuning (§6): the students cope with wind consistently, and
  fully randomized success is steadier (84.9% ± 1.3), but nominal success falls to 88.3% ± 6.2. Not adopted: §5.1's
  setup stays the Phase 4 result
- [x] Camera pitch randomization and out-of-distribution maps (§7): pitch up to ±10° costs nothing; with pitch
  added to "full", the mean drop is 10.7%, just over the target. Density, unseen sizes and poles-only cost 5–8%;
  wall-heavy maps 15%

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

### 5.1 Three seeds: met on average, not on every seed

Seeds 1 and 2 of §4–5, run end to end by a small pipeline script (fine-tune the teacher, distil the student, run the
table, each starting when the previous step verified). Their teachers were fine-tuned from the seed-1 and seed-2
soft-ceiling teachers (`56j6n22n`, `arrqs6y5`):

| Run | Seed 1 | Seed 2 |
|---|---|---|
| Fine-tuned teacher | [dsevgp9o](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/dsevgp9o) | [6nbcsoii](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/6nbcsoii) |
| GRU student | [bmylysop](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/bmylysop) | [jegy71of](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/jegy71of) |

**Teachers** (held-out level 5, 0.1 m rule):

| Teacher | Nominal | Full randomization | Relative drop |
|---|---:|---:|---:|
| s42 (§4) | 93.3% | 81.35% | 12.8% |
| s1 | 93.8% | 71.95% | 23.3% |
| s2 | 90.0% | 78.0% | 13.3% |

**Students:**

| Condition | s42 | s1 | s2 | **Mean ± SD** | Mean relative drop (per seed) |
|---|---:|---:|---:|---:|---|
| Nominal | 93.15 | 93.7 | 91.25 | **92.7 ± 1.3** | |
| Mass and inertia ±20% | 92.95 | 92.15 | 91.2 | 92.1 | 0.6% (0.2 / 1.7 / 0.1) |
| Thrust ±15% | 93.6 | 92.2 | 91.5 | 92.4 | 0.3% |
| Motor lag ≤ 50 ms | 92.5 | 92.3 | 91.3 | 92.0 | 0.7% |
| Drag ±30% | 92.65 | 93.5 | 91.25 | 92.5 | 0.3% |
| Latency 0–40 ms | 92.15 | 91.65 | 89.65 | 91.2 | 1.7% |
| **Wind ≤ 2 m/s + gusts** | 90.85 | **83.1** | 88.05 | 87.3 ± 3.9 | **5.8%** (2.5 / **11.3** / 3.5) |
| Observation noise | 92.9 | 92.75 | 91.45 | 92.4 | 0.4% |
| Depth noise × 2 | 95.2 | 94.15 | 93.95 | 94.4 | −1.9% |
| **Full** | 88.06 | **78.6** | 84.3 | **83.6 ± 4.8** | **9.7%** (5.5 / **16.1** / 7.7) |

All in % of episodes. Under full randomization: collisions 10.8 / 18.2 / 11.1%, too high 0.05 / 1.5 / 1.25%, flipped
0.5 / 0.35 / 1.65%.

1. **The exit test is met on average, narrowly:** a 9.7% mean relative drop against the < 10% target (83.6% fully
   randomized against 92.7% nominal). It's met on two seeds of three (5.5%, 7.7%); **seed 1 fails at 16.1%**.
2. **Every factor except wind is solved on every seed:** mass, thrust, motor lag, drag, latency and both noises cost
   under 2% each.
3. **Wind is what's left,** and it's seed-dependent: 2.5%, 11.3% and 3.5%. The seed-1 student collides with obstacles
   in 18% of fully randomized episodes. Its fine-tuned teacher was also the weakest under randomization (23% drop),
   so, as in Phase 3, the student inherits its teacher's weakness.
4. **The robustness costs about 2 points at nominal:** 92.7% ± 1.3 against Phase 3's 94.6% ± 0.4.

**Verdict:** Phase 4's exit is met on average but isn't robust across seeds. The remaining work is wind. The
cheapest things to try:
- **Train with stronger wind than the test** (e.g. up to 3 m/s), so 2 m/s sits inside the training range rather than
  at its edge.
- **Fine-tune the teachers longer.** 1000 iterations left the seed-42 curriculum still climbing (§4).

Cost of seeds 1 and 2: teachers $0.76 + $0.45, students $1.98 + $1.96, tables $0.80 + $0.77, $6.72 in all.

## 6. Fixing wind: stronger training wind, longer fine-tuning

§5.1 left wind as the weak point, varying by seed. Two changes, on all three seeds at once:
- **Train with stronger wind than the test:** wind up to 3 m/s with 0.75 m/s gusts, for teacher and student. The
  tests stay at 2 m/s + 0.5 m/s, now inside the training range rather than at its edge.
- **Fine-tune the teachers longer:** 2000 iterations instead of 1000. §4's seed-42 teacher was still climbing the
  curriculum at 1000.

Everything else is as in §4–5. Same pipeline script, run names tagged `-w3`.

| Run | Seed 42 | Seed 1 | Seed 2 |
|---|---|---|---|
| Fine-tuned teacher | [relovls4](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/relovls4) | [nlsrwcf1](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/nlsrwcf1) | [g9ope2wo](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/g9ope2wo) |
| GRU student | [0fbiefxc](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/0fbiefxc) | [hzxsqsor](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/hzxsqsor) | [9l44nfl0](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/9l44nfl0) |

**Teachers** (held-out level 5; the "full" evaluation uses the 2 m/s test wind):

| Teacher | Nominal | Full | Relative drop | §5.1's teacher (nominal / full / drop) |
|---|---:|---:|---:|---|
| s42 | 92.3% | 82.7% | 10.4% | 93.3 / 81.35 / 12.8% |
| s1 | 92.8% | 74.2% | 20.1% | 93.8 / 71.95 / 23.3% |
| s2 | **85.9%** | 78.35% | 8.8% | 90.0 / 78.0 / 13.3% |

**Students:**

| Held-out level 5 | §5.1 (2 m/s, 1000 it.): s42 / s1 / s2 | Mean ± SD | **Stronger wind: s42 / s1 / s2** | **Mean ± SD** |
|---|---|---:|---|---:|
| **Nominal** | 93.15 / 93.7 / 91.25 | **92.7 ± 1.3** | 89.55 / 93.8 / **81.7** | **88.3 ± 6.2** |
| Mass, thrust, motor lag, drag, latency, obs noise | | 91.2–92.5 | | 87.4–88.6 |
| Wind 2 m/s + gusts | 90.85 / 83.1 / 88.05 | 87.3 ± 3.9 | 87.0 / 88.9 / 87.9 | **88.0 ± 0.95** |
| **Full** | 88.06 / 78.6 / 84.3 | 83.6 ± 4.8 | 85.9 / 83.4 / 85.3 | **84.9 ± 1.3** |
| Mean relative drop (per seed) | | 9.7% (5.5 / 16.1 / 7.7) | | 3.6% (4.1 / 11.1 / −4.5) |

All in % of episodes. Seed 2 at nominal: 10.2% time-outs, 8.1% collisions. Seed 42 at nominal: 10.3% collisions.

1. **Wind is evened out.** Under 2 m/s wind every seed is now at 87–89% (SD 0.95, against 3.9), and the weak seed 1
   improves under full randomization (78.6% → 83.4%). Fully randomized success is slightly higher and much steadier:
   84.9% ± 1.3 against 83.6% ± 4.8.
2. **But nominal success falls:** 92.7% → 88.3%, with seed 2 at 81.7%. Its teacher had already lost 4 points at
   nominal (90.0% → 85.9%), and the student inherited it, with 10% of calm episodes timing out. Seed 42's student
   collides more at nominal (10.3% against 6.65%).
3. **That makes the relative drop misleading here.** A 3.6% mean drop looks like a big improvement on 9.7%, but much
   of it comes from a worse nominal, not a better randomized result. Seed 2 even does better fully randomized than
   nominal. Seed 1, whose nominal held up, still fails the test (11.1%).
4. **Averaged over nominal and fully randomized,** the §5.1 setup is better (88.2% against 86.6%).

**Decision:** keep §5.1's setup (2 m/s training wind, 1000 fine-tuning iterations) as the Phase 4 result:
- 92.7% nominal, 83.6% fully randomized, 9.7% mean drop;
- the exit test met on average, on two seeds of three.

The stronger-wind variant shows the trade-off rather than fixing it. Ideas not yet tried:
- keep calm conditions in training (e.g. half the episodes windless), so the policy doesn't lose its calm-air
  behaviour;
- give the teacher the wind as a privileged observation, so its labels compensate for wind precisely and the GRU
  student only has to infer it.

Cost: $12.86 (teachers $1.13 + $1.23 + $1.81, students $1.46 + $1.90 + $1.29, tables $1.37 + $1.02 + $1.65). Two
runs' prices are computed by hand, because the launcher's price lookup failed. The tables took 35–57 minutes of box
time against about 22 in §5.1, mostly the launcher's follow stalls adding idle time.

## 7. Camera pitch and out-of-distribution maps

The last two Phase 4 items in PLAN.md: camera pitch in the randomization list (§5.5), and the out-of-distribution
evaluation ("higher density, new primitive types, more noise"). Commit `f869480`.

**Camera pitch:** `env.camera_pitch_max_deg` (depth task) draws each env's camera mounting pitch at every reset from
U(−max, max) degrees, positive nose-down. It writes the rows of Isaac Lab's `RayCasterCamera._offset_quat`: one
(w, x, y, z) row per env, in the "world" convention, applied on top of the body pose at every update.
`camera_depth_replay` now takes a pitch, with per-pixel rays, and a CPU test checks that the pitched path reproduces
the level one at zero pitch.

On the GPU, `check_depth.py --camera_pitch_max_deg 10` (`p4-check-depth-pitch`, 256 cameras pitched −9.8° to +10°)
agrees with exact geometry on **99.97%** of pixels, as the level check did (Phase 3 §2). Ground pixels agree on
99.998%, which also confirms the sign.

**Out-of-distribution maps:** scalar env settings that change the terrain generator before it builds
(`obstacle_field.ood_overrides`; a CPU test checks every variant still generates passable layouts):
- `env.terrain_density_scale`: obstacle density × s at every difficulty;
- `env.terrain_mix`: `poles` (only poles) or `walls` (wall-heavy: 40% walls);
- `env.terrain_unseen_sizes`: sizes outside every training range. Cylinders 0.5–0.9 m in radius (training ≤ 0.5),
  boxes 1.5–2.5 m a side (≤ 1.5), poles 2–3 cm in radius (≥ 3 cm).

**Evaluation** (`p4-ood-gru-dr-s{42,1,2}`, one box per student, $1.36): §5.1's three randomization-trained students
(`0g9bvtcl`, `bmylysop`, `jegy71of`), 60 held-out layouts, 2000 episodes per condition, level 5 unless stated:

| Condition | Success s42 / s1 / s2 | Mean | Drop from nominal | Mean collisions (poles) | Time-out |
|---|---|---:|---:|---:|---:|
| Nominal | 93.2 / 93.7 / 91.3 | **92.7** | | 6.3 (4.8) | 0.9 |
| Camera pitch ±5° | 93.1 / 93.3 / 91.5 | 92.6 | 0.1% | 6.2 (4.7) | 1.0 |
| Camera pitch ±10° | 93.1 / 92.8 / 90.8 | 92.2 | 0.5% | 6.5 (4.8) | 1.2 |
| **Full randomization + pitch ±5°** | 87.1 / 77.2 / 84.0 | **82.75** | **10.7%** (6.5 / 17.6 / 8.0) | 14.0 (4.3) | 0.9 |
| Depth noise × 3, observation noise × 2 | 96.2 / 93.5 / 95.5 | 95.1 | −2.6% | 3.9 (2.4) | 0.9 |
| Sizes outside training | 86.7 / 89.6 / 87.1 | 87.8 | 5.3% | 11.3 (9.1) | 0.9 |
| Density × 1.5 | 86.8 / 87.7 / 84.7 | 86.4 | 6.8% | 11.5 (9.2) | 2.0 |
| Poles only | 84.3 / 86.5 / 86.3 | 85.7 | 7.5% | 14.2 (13.7) | 0 |
| Wall-heavy | 85.3 / 80.3 / 71.3 | 79.0 | 14.9% | 12.5 (6.0) | 8.5 |
| Level 9 (densest training level, 0.6 m gaps) | 51.9 / 65.1 / 54.9 | 57.3 | 38.2% | 32.5 (13.2) | 10.2 |

1. **Camera pitch is a non-issue:** up to ±10° costs at most 0.5%, though the students never trained with it. A 90°
   wide view from a 3 m/s drone barely changes with a few degrees of tilt.
2. **With pitch in the "full" condition, the exit test is right at the line:** a 10.7% mean drop (9.7% without
   pitch, §5.1). Seeds 42 and 2 pass (6.5%, 8.0%); seed 1 fails (17.6%), through wind as before.
3. **Noise is not a problem in either direction:** three times the depth noise and twice the observation noise
   *improves* success (95.1%), and pole collisions fall (2.4%). The students trained at twice the depth noise, so
   more noise is closer to their training than nominal is. That it helps beyond that suggests heavier noise makes
   the policy more cautious near thin obstacles. Untested.
4. **Out-of-distribution maps cost 5–8%, through poles:**
   - Thinner-than-training poles (unseen sizes), denser maps (more poles) and poles-only maps all move collisions
     towards poles (9–14%). Bigger cylinders and boxes are handled.
   - Poles remain the student's weak point, as in every phase since 2.
5. **Walls are the weak spot out of distribution:** wall-heavy maps cost 15% (and 22% on seed 2), mostly time-outs
   (8.5%). That's the old door-finding problem (Phase 3 §8): with more walls, the GRU's memory of where the door was
   isn't always enough.
6. **Level 9 is out of reach:** 57%. Its 0.6 m gaps and 12 obstacles per 100 m² were in training, but the students
   ended training with only 70–90% of envs reaching level 9, at 55–90% success there (§5).

**Phase 4 verdict:**
- **The exit test is met narrowly without camera pitch** (9.7% mean drop) **and missed narrowly with it** (10.7%).
  Both rest on two seeds of three, with seed 1 failing through wind.
- **Robust:** mass, inertia, thrust, motor lag, drag, latency, sensor noise and camera pitch each cost under 2%, where
  the Phase 3 student lost 8% to mass and 12% to wind.
- **Not solved:** wind at full strength, poles, and wall-heavy layouts.

Moving obstacles (PLAN.md's optional item) are not attempted.

Cost of §7: $1.36 for the evaluations, plus the pitch check (its launcher was killed when the app quit, before it
could account the cost: about 4 minutes up, plus a Nebius delete, roughly $0.20–0.55).

