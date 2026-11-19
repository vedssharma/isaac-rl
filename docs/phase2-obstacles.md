# Phase 2: obstacles with privileged sensing

PLAN.md Phase 2 builds `Drone-Nav-Obstacles-v0`: the CTBR goal-reaching task from Phase 1, through a procedural
obstacle field, with a ray-cast "lidar ring" as the observation and a curriculum over difficulty. This note
records the parts as they land.

Status:

- [x] Obstacle terrain generator and its CPU tests (§1)
- [x] `Drone-Nav-Obstacles-v0` (§2): spawning in the strips, lidar ring, collision termination, proximity reward.
  Passed a 30-iteration smoke run on an L40S (§3).
- [x] Curriculum (§4)
- [x] First training run and held-out evaluation (§5): 88.1% success at level 5 on held-out maps, one seed.
  The exit criterion (≥ 90% success, < 5% collisions) is **not met yet**.
- [x] Proximity-shaping sweep (§6): no effect on level-5 success or collisions. Shaping isn't the bottleneck.
- [x] No-poles diagnostic (§7): poles cause about a third of level-5 collisions, but most come from other obstacles.
- [x] 5° lidar ring (§8): overall level-5 collisions stay about 10–12%.
- [x] Collision attribution (§9): walls and poles cause about two thirds of collisions. Cylinders and boxes are
  handled well.
- [x] Slowing down near obstacles, evaluated on 60 held-out layouts (§10): the mild slowdown reaches 89.8% /
  10.2% (baseline 85.5% / 13.4%), one seed. Poles are now the largest single cause.
- [x] Three seeds of the mild slowdown vs the baseline (§10.2): 89.7% / 10.0% against 86.4% / 12.6%. Poles
  (5.8%) are now the entire gap to the < 5% collision target.
- [x] Lidar replay of collisions (§11): poles fall between rays 65% of the time and are first seen 0.45 s before
  impact. It also found the ring is world-fixed, not yaw-following (a bug in our assumptions about Isaac Lab).
- [x] Ring that follows the drone's yaw (§12): no improvement on one seed (84.6% / 14.6% against 89.8% / 10.2%).
  The world-fixed ring stays.
- [x] Privileged nearest-obstacle observation (§13): poles 5.4% → 4.5%, overall a wash (89.3% / 8.8%). Pole
  collisions aren't mainly about sensing or information.
- [x] Pole near misses (§15): most pole hits are grazes, not head-on crashes. The policy passes poles within
  0.2 m three to six times as often as any other obstacle, even with exact positions. Next: clearance pressure.
- [x] Training with a 0.15 m collision margin (§16): halves cylinder hits, but poles are unchanged (6.1%) and so is
  the overall rate (89.8% / 9.9%). The policy keeps more room from large obstacles, not from poles.
- [x] Margin and nearest-obstacle observation together (§17): 93.8% / 5.65% on seed 42, but **91.0% / 7.5% over
  3 seeds** (§17.1). Collisions −2.6 points against the slowdown alone; the < 5% target is still not met.
- [ ] Ablations

## 1. Obstacle terrain

[`drone_nav/terrain/obstacle_field.py`](../source/drone_nav/drone_nav/terrain/obstacle_field.py) generates one tile
of Isaac Lab's terrain generator (PLAN.md §5.2 option A). Isaac Lab bakes all tiles into one static mesh, which
gives collisions through the terrain collider and fast ray casting (`RayCaster`) on any GPU.
[`obstacle_field_cfg.py`](../source/drone_nav/drone_nav/terrain/obstacle_field_cfg.py) holds the Isaac Lab configs:
`ObstacleFieldTerrainCfg` for one tile, and `OBSTACLE_FIELD_GENERATOR_CFG`, which has 10 rows of difficulty × 20
tiles each.

**A tile** is a 24 × 10 m corridor along +x with side walls. The drone starts in a 2 m strip at one end and its
goal is in a 2 m strip at the other, 20–24 m away (Phase 1 goals were 10–20 m). Obstacles fill the 18 m zone in
between, with 1 m left empty next to each strip:

- **cylinders** (trees and pillars, radius 0.15–0.5 m), **boxes** at any yaw (0.4–1.5 m sides), **thin poles**
  (3–6 cm, hard to see in depth images later), and **walls across the corridor** with one door.
- Every obstacle is a prism from the ground up to 5 m, above the env's 4 m altitude limit, so the drone has to go
  around it and can't fly over. That also turns every check into a 2-D problem about footprints.

**Difficulty** (0–1, row *i* of the curriculum is level *i*) sets three things by linear interpolation:

| | Level 0 | Level 5 | Level 9 |
|---|---:|---:|---:|
| Density (per 100 m² of obstacle zone) | 0.6 | 6.6 | 11.4 |
| Obstacles in the zone | 1 | 11 | 20 |
| Minimum gap | 1.93 m | 1.23 m | 0.67 m |
| Mix (cylinder / box / pole / wall) | 59 / 30 / 11 / 1 % | 46 / 30 / 16 / 8 % | 36 / 30 / 20 / 14 % |

Even the hardest gap is about 5× the Crazyflie's 13 cm span (PLAN.md asks for at least 2×).

![One tile per difficulty level, seen from above](images/obstacle_field_levels.svg)

*One training tile per level (seed 0). The start strip is blue and the goal strip red. Regenerate it with
`python scripts/preview_obstacle_field.py`, or `--seed 1001` for a held-out map.*

**Guarantees.** Every layout has all of these, and
[`tests/test_obstacle_field.py`](../tests/test_obstacle_field.py) checks each one on 80 layouts (8 seeds × 10
levels):

1. The start strip, the goal strip and the buffers next to them are empty.
2. Every two footprints are at least the minimum gap apart (exact distances between circles and rotated
   rectangles). Free-standing obstacles also keep that gap from the side walls, and every door is at least that
   wide.
3. A disk of radius 0.1 m (the Crazyflie plus a margin) can get from the start strip to the goal strip. The
   generator checks this with a flood fill on a 10 cm grid, and draws a new layout if it fails.
4. The collision meshes lie inside the footprints that checks 1–3 use. Cylinder meshes are 16-sided polygons
   inscribed in their circle.

Placement is random with rejection: up to 200 tries per obstacle, then that obstacle is skipped. At the default
densities, 99.98% of requested obstacles get placed, and a layout never ends up more than one short. The
density range tops out at the point where that stops being true. A tile takes about 6 ms to generate, so the
200 training tiles add about a second to start-up.

**Held-out maps.** A tile's layout depends only on the generator seed and the tile's difficulty. Training uses seed
`TRAIN_SEED = 0`, and evaluation uses `TEST_SEEDS = (1001, 1002, 1003)`, so the evaluation maps never appear in
training (PLAN.md §5.1).

**Not in this version:** horizontal bars and floating spheres (PLAN.md §5.1). Both need real 3-D checks, because
the drone could fly under or over them. Obstacles also never move; moving ones need PLAN.md §5.2 option B.

## 2. The env: `Drone-Nav-Obstacles-v0`

[`tasks/obstacles/obstacles_env.py`](../source/drone_nav/drone_nav/tasks/obstacles/obstacles_env.py) subclasses
the Phase 1 CTBR env, so the robot, the actions, the speed cap and every Phase 1 reward term stay the same (CTBR
was the Phase 1 decision). The obstacle-specific math is plain torch in
[`mdp/obstacles.py`](../source/drone_nav/drone_nav/mdp/obstacles.py), tested in
[`tests/test_obstacles_mdp.py`](../tests/test_obstacles_mdp.py). What changes:

- **Terrain and resets.** The flat plane becomes the obstacle terrain (§1), 10 levels × 20 tiles. Each env sits on
  one tile. It spawns anywhere in the start strip and its goal is anywhere in the goal strip, both at least 0.5 m
  from the strip edges and the side walls. Yaw is random, as in Phase 1.
- **Lidar ring** (PLAN.md §4.2, the teacher's privileged sensing). An Isaac Lab `RayCaster` on the drone's body
  casts 3 channels (−15°, 0°, +15°) × 36 rays (every 10°) against the terrain mesh, out to 5 m. It stays level
  with the horizon, so pitching forward to accelerate doesn't point it at the ground. ~~It turns with the drone's
  yaw.~~ **Correction (§11):** it doesn't. Isaac Lab's `ray_alignment="yaw"` turns only the rays' start points, so
  the ring is fixed in the world: the lidar part of the observation is world-frame, while the goal and velocities
  are body-frame. Each ray's distance / 5 m is appended to the Phase 1 observation: 18 + 108 = 126 inputs.
- **Collision** ends the episode as a crash, with the usual crash penalty, when the horizontal channel sees an
  obstacle within 0.1 m of the drone's centre. Obstacles are vertical and taller than the flight ceiling, so the
  horizontal channel always sees what the drone could hit. A test checks that 10° spacing can't miss one:
  at every bearing, even a 3 cm pole inside 0.1 m is hit by some ray, closer than 0.1 m. Further out a thin pole
  can fall between two rays; from about 0.35 m in it can't. Ground and ceiling are still the Phase 1 altitude
  limits.
- **Proximity reward** (PLAN.md §4.3): −1 × exp(−clearance / 0.3 m) per second, where clearance is the closest
  horizontal hit minus the 0.1 m collision radius. That costs 1 per second at contact, 0.37 at 0.3 m clearance and
  about 0 past 1 m. Progress at the 3.25 m/s cruise speed earns about 3.25 per second, so brushing past an obstacle
  is cheaper than a long detour. `env.rewards.proximity=0.0` is ablation (a). Write it as a float: Isaac Lab
  rejects a Hydra override whose type differs from the default's, and a `None` default can't be overridden at all.
- **New metrics**: `Episode_Termination/collision`, `Episode_Reward/proximity`, `Metrics/min_obstacle_distance`
  (closest an obstacle came during the episode) and `Curriculum/terrain_level`.

**Curriculum**: see §4. The smoke run below predates it; it put envs on random levels 0–2 that never changed.

**Evaluation.** `env.eval_terrain_level=5` puts every env on level 5 (the Phase 2 exit criterion), and
`env.terrain.terrain_generator.seed=1001` swaps in held-out maps:

```
launch.py --run-name p2-obstacles-s42 --eval eval_heldout_l5:env.eval_terrain_level=5,env.terrain.terrain_generator.seed=1001 \
  -- --task Drone-Nav-Obstacles-v0 --num_envs 4096 --seed 42
```

**Smoke run first.** This is the first code that touches Isaac Sim since §1, and none of it has run there yet. A
short run on easy levels should show that the terrain loads, the rays hit it, and collisions get counted, before
a full 1500-iteration run:

```
launch.py --run-name p2-obstacles-smoke-s42 -- --task Drone-Nav-Obstacles-v0 --num_envs 1024 --seed 42 \
  --max_iterations 30 env.terrain.max_init_terrain_level=2
```

What to look for: start-up without errors and a non-zero `Episode_Termination/collision`.
`Metrics/min_obstacle_distance` should sit below 5 m on levels with obstacles.

## 3. Smoke run

`p2-obstacles-smoke-s42`, the command above: W&B run
[sed6nd3l](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/sed6nd3l), commit `9689689`, 1024 envs on
levels 0–2, 30 iterations. It ran end to end, and the launcher verified the checkpoint on W&B and HF. On the L40S
it does about 63k env steps/s (0.4 s per iteration), so 1500 iterations at 4096 envs should take roughly 30–40
minutes. The 200 terrain tiles generate in 3.4 s.

| | Iter 1 | Iter 10 | Iter 20 | Iter 29 |
|---|---:|---:|---:|---:|
| Mean reward | −10.2 | −10.0 | −6.9 | −1.0 |
| Episode length (steps) | 44 | 124 | 239 | 398 |
| Mean speed (m/s) | 0.15 | 1.34 | 1.58 | 2.17 |
| Closest obstacle per episode (m) | 1.84 | 1.56 | 0.93 | 0.44 |
| Episodes ending flipped / too low | 79% / 3% | 10% / 79% | 9% / 4% | 13% / 4% |
| Episodes ending in a collision | 7% | 12% | 49% | 35% |
| Success | 0% | 0% | 0% | 0% |

The policy learns to stay up, then to fly forward, and by iteration 20 it reaches the obstacles, so collisions take
over as the main way episodes end. No goal is reached in 30 iterations, which is too early to read anything into.

It took four launches to get here, each failing a little later than the one before. Each fix is its own commit:

1. **Hydra overrides must match the default's type.** `max_init_terrain_level` defaulted to `None`, so `=2` was
   rejected. It's now an int (9), and `eval_terrain_level` is -1 when off.
2. **Sensors don't see Fabric clones.** The goal task clones envs in Fabric. Isaac Lab sensors count envs from the
   USD stage, so the lidar sized itself for env_0 only, and the first reset hit a CUDA index-out-of-bounds assert.
   This task now clones on the USD stage, and the env refuses to start if the lidar and env counts disagree.
3. **float64 goal offsets.** The strip bounds were numpy float64, so the sampled goals were too, and Isaac Lab's
   float32 buffers refused them.

A separate bug turned up along the way: `launch.py` printed its whole start script, API keys included, to the
terminal. Every earlier launch did too. That's fixed and tested, and the keys should be rotated.

## 4. Curriculum

PLAN.md §5.3, in [`mdp/curriculum.py`](../source/drone_nav/drone_nav/mdp/curriculum.py) (tests:
[`tests/test_curriculum.py`](../tests/test_curriculum.py)). Every env starts on level 0. Each env keeps a window of
its last 5 episodes; once the window is full, the env moves up a level if at least 80% of them reached the goal,
and down a level if fewer than 40% did. Crashes and time-outs both count as failures. A move clears the window, so
the next decision is based only on episodes at the new level. The env moves before it respawns, so its next episode
is on the new tile.

One deliberate difference from Isaac Lab's own `TerrainImporter.update_env_origins`, which sends an env that passes
the top level to a random level: here it stays at level 9. Phase 2 is judged on the hard levels, and moving down
still catches an env that starts failing.

Rough pace: an episode is 5–10 s and a 1500-iteration run gives each env about 720 s of flying, so roughly 70–140
episodes. At 5 episodes per move, an env can climb from level 0 to 9 in about 45 episodes if it never fails.

**Logged:** `Curriculum/terrain_level` (the level each finished episode was played on) and
`Curriculum/level_0` … `level_9`, the fraction of envs on each level. That's the difficulty histogram PLAN.md calls
the best progress chart.

**Settings** (Hydra, under `env.`): `curriculum`, `curriculum_window`, `curriculum_up_rate` and
`curriculum_down_rate`. Write the rates as floats. Evaluation (`eval_terrain_level` ≥ 0) turns the curriculum off.
Ablation (b), no curriculum, is `env.curriculum=False env.terrain.max_init_terrain_level=9`: envs on random levels
that never change.

Not done: PLAN.md §5.3 also suggests a curriculum on goal distance and on the proximity penalty's size. Goal
distance is fixed by the tile (20–24 m), and the proximity weight stays constant for now.

## 5. First training run

`p2-obstacles-s42`: W&B run [x7x2fl6t](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/x7x2fl6t),
commit `8ea755b`, 4096 envs, 1500 iterations, seed 42, with the curriculum. Training took 10 minutes on the L40S
(0.4 s per iteration, about 240k env steps/s; 1024 envs ran at the same 0.4 s, so the smoke run left the GPU mostly
idle). The launcher then ran the held-out evaluation on the same box.

**Curriculum.** Every env starts on level 0 and climbs. Fractions of envs per level:

| Iteration | Mostly on | Envs on level 9 | Training success | Collisions | Mean speed |
|---:|---|---:|---:|---:|---:|
| 100 | level 1 (93%) | 0% | 95% | 5% | 3.25 m/s |
| 250 | level 3 (92%) | 0% | 91% | 9% | 3.16 m/s |
| 500 | level 6 (74%), level 5 (19%) | 0% | 82% | 16% | 3.04 m/s |
| 750 | levels 7–9 (89%) | 19% | 67% | 29% | 2.98 m/s |
| 1000 | levels 8–9 (85%) | 60% | 75% | 21% | 3.00 m/s |
| 1499 | level 9 (84%) | 84% | 83% | 15% | 3.05 m/s |

Training success is measured at whatever level each env is on, so it dips while envs climb and recovers once most of
them settle at level 9. By the end the policy reaches the goal on 83% of level-9 episodes, the densest level with
the narrowest gaps. Almost all failures are collisions.

**Held-out evaluation, level 5** (PLAN.md's Phase 2 exit test). 1001 deterministic episodes, every env on level 5,
on maps from test seed 1001 that training never saw:

| | Held-out level 5 | Exit criterion |
|---|---:|---:|
| Success | **88.1%** (95% CI 86.0–90.0%) | ≥ 90% |
| Collision | **11.4%** (95% CI 9.6–13.5%) | < 5% |
| Too high | 0.5% | |
| Time-out, flipped, too low, off course | 0% | |
| Time to goal | 7.4 s | |
| Path efficiency | 0.92 | |
| Mean speed | 3.22 m/s | |
| Closest obstacle per episode | 0.29 m (mean) | |

So the first run comes close on success and clearly misses on collisions. That's one seed, so it's a baseline, not a
result.

**What the numbers suggest.** These are hypotheses for the next runs, not conclusions:

- **It flies close.** The mean closest pass is 0.29 m from the drone's centre, against a 0.1 m collision radius. The
  proximity term costs only −0.5 per episode against +20.7 for progress. At the 0.3 m sigma it barely changes the
  trade-off, so a stronger or wider proximity term (weight or `proximity_sigma`) is the obvious first knob.
- **It doesn't slow down near obstacles.** The mean speed is 3.2 m/s, the same as open-field cruising. Nothing in
  the reward asks for slowing down, and the speed penalty only starts above 3 m/s.
- **Held-out maps vs. level 5 itself.** The policy ended training almost entirely on level 9 and was never tested on
  training-seed level-5 maps. An evaluation there would show whether the 88% comes from new maps or from level 5
  itself (for example forgetting easier layouts, though level 5 should be easier than level 9).
- **0.5% too high:** a few episodes climb above 4 m. Obstacles are 5 m tall, so climbing never pays off; worth
  watching in a video.

Next: a training-seed level-5 evaluation, a proximity-shaping sweep, then 3 seeds of whatever wins. The PLAN.md
ablations (no proximity, no curriculum, number of rays) come after that.

## 6. Proximity-shaping sweep

§5 left the policy colliding on about 11% of level-5 episodes, with its closest pass at 0.29 m, so the first
suspect was the proximity term (−w exp(−clearance / σ), baseline w = 1.0, σ = 0.3 m). Three variants, otherwise
identical to §5 (seed 42, 4096 envs, 1500 iterations, curriculum on). Each was evaluated at level 5 on held-out maps
(test seed 1001) and on the training maps (seed 0). The baseline got the training-map evaluation afterwards, from
its saved checkpoint.

| Run | w | σ | Held-out L5: success / collisions | Training maps L5: success / collisions | Closest pass (held-out) |
|---|---:|---:|---:|---:|---:|
| `p2-obstacles-s42` ([x7x2fl6t](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/x7x2fl6t)) | 1.0 | 0.3 | 88.1% / 11.4% | 86.8% / 12.9% | 0.29 m |
| `p2-prox-w3-s42` ([s3wi4bgh](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/s3wi4bgh)) | 3.0 | 0.3 | 81.6% / 17.9% | 86.8% / 12.2% | 0.28 m |
| `p2-prox-sig06-s42` ([lbzgjbn9](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/lbzgjbn9)) | 1.0 | 0.6 | 87.8% / 12.1% | 87.1% / 12.6% | 0.30 m |
| `p2-prox-w3-sig06-s42` ([t9os0rrz](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/t9os0rrz)) | 3.0 | 0.6 | 87.1% / 10.7% | 87.1% / 12.2% | 0.34 m |

Each cell is 1000 episodes; binomial 95% CIs are about ±2 points. All four runs climbed the curriculum the same
way (78–84% of envs on level 9 at the end) and fly at the same 3.1–3.2 m/s.

What it shows:

1. **The proximity term doesn't change level-5 performance.** On the training maps all four settings land at
   86.8–87.1% success and 12.2–12.9% collisions, which is noise. Tripling the weight and doubling the width
   (about 4× the penalty at 0.2 m clearance) moves the closest pass from 0.29 to 0.34 m and nothing else.
   Stronger shaping reduces how long the drone flies near obstacles: the w = 3 runs pay less proximity penalty
   in total than the baseline does at w = 1. But the closest pass and the collision rate barely move.
2. **New maps aren't the problem either.** Each policy scores about the same on held-out and training maps. The
   exception, `w3` at 81.6% held-out vs 86.8% on training maps, is best explained by the next point.
3. **The evaluation set is small.** A level has 20 tiles, so 1000 episodes cover just 20 layouts. Two runs can
   differ by a few points on one map set simply because one layout is harder, and the binomial CI doesn't
   include that map-to-map spread. Future evaluations should cover more layouts: all three test seeds, 60 tiles.
4. Evaluations now run with 1024 envs (`--num_envs=1024` in the `--eval` spec). That took one from 19 minutes to
   under 2. The baseline's held-out number in §5 used 256 envs.

**Working hypothesis: the drone doesn't see thin poles in time.** Poles are 6–12 cm across and the ring's rays are
10° apart, so a pole falls between rays until it is within about 0.35–0.7 m, which is 0.1–0.2 s at 3 m/s. Collisions
that come from sudden surprises would explain why shaping, which acts on obstacles the drone can see, changes so
little. It's untested. The cheap check is to evaluate an existing checkpoint at level 5 with poles removed from the
mix. If collisions drop sharply, the fix is sensing (a finer horizontal ring, which is also PLAN.md's ablation (c)),
not reward.

## 7. Diagnostic: level 5 without poles

To test §6's hypothesis, I evaluated the §5 checkpoint
([x7x2fl6t](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/x7x2fl6t)) at level 5 on the training maps
with poles removed from the mix (pole weight 0 in `mix_easy` and `mix_hard`, so other obstacles fill their slots).
The comparison is the same checkpoint on the same seed with poles (§6). No retraining.

| Level 5, training maps | With poles | Without poles |
|---|---:|---:|
| Success | 86.8% (95% CI 84.6–88.8%) | **90.2%** (95% CI 88.2–91.9%) |
| Collisions | 12.9% (95% CI 11.0–15.1%) | **8.6%** (95% CI 7.0–10.5%) |
| Too high | 0.1% | 1.0% |
| Closest pass | 0.30 m | 0.30 m |
| Mean speed / path efficiency | 3.22 m/s / 0.94 | 3.23 m/s / 0.92 |

Removing poles also changes the random draws, so the layouts differ, and a level has only 20 tiles. The size of the
drop is therefore approximate, but its direction is clear.

- **Poles cause more than their share of collisions.** They are about 16% of level-5 obstacles, but removing them
  cuts collisions by about a third (12.9% → 8.6%), even though other obstacles fill the freed slots. Per obstacle, a
  pole is roughly 2.5× as likely as anything else to cause a collision. That fits the 10° ring seeing them late.
- **But poles aren't the main cause.** With none at all, 8.6% of episodes still end in a collision, well above the
  5% target. Most collisions come from cylinders, boxes or walls.
- Too-high crashes rise to 1%. The mix without poles has relatively more walls (about 10% instead of 8%), and a
  few episodes try to climb over one; it never pays off, since walls are 5 m and the ceiling is 4 m.

**Next candidates:**

1. **A finer lidar ring**, which is PLAN.md ablation (c). A 5° ring (72 × 3 = 216 rays) would see a 6 cm pole from
   about 0.7 m instead of 0.35 m, and box corners and wall ends earlier. This tests the sensing hypothesis
   causally, with retraining.
2. **Recording which obstacle type each collision hits**, to explain the remaining 8.6% instead of guessing.
3. **Slowing down near obstacles.** The policy cruises at 3.2 m/s through level 5, the same as in open space.

## 8. A finer lidar ring (5°)

`p2-lidar5deg-s42` ([ckba9w5s](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/ckba9w5s)) is §5 with
`env.lidar.pattern_cfg.horizontal_res=5.0`: 72 rays per channel instead of 36, so 216 rays and a 234-input
observation. Training time per iteration was unchanged (0.41 s). Everything else is identical: seed 42, 4096 envs,
1500 iterations, curriculum on. By the end, 83% of envs were on level 9 with 86% training success and 12%
collisions there, against 83% / 15% for the 10° baseline.

| Level 5 | 10° ring (§5–7) | 5° ring |
|---|---:|---:|
| Training maps: success / collisions | 86.8% / 12.9% | 86.1% / 11.4% |
| Training maps, no poles: success / collisions | 90.2% / 8.6% | 87.1% / 11.2% |
| **Collisions caused by poles** (difference) | **4.3 points** | **0.2 points** |
| Held-out maps (seed 1001): success / collisions | 88.1% / 11.4% | 79.9% / 19.6% |
| Closest pass / mean speed (training maps) | 0.30 m / 3.22 m/s | 0.30 m / 3.29 m/s |
| Too high (training maps) | 0.1% | 2.5% |

1. ~~**The finer ring does what it was meant to do for poles.**~~ **Corrected by §9:** measured directly, the
   5° policy hits poles exactly as often as the 10° one (4.1% of episodes each). The no-poles comparison in this
   table changes the layouts along with the mix, and on 20 tiles that change swamped the effect. It can't isolate
   one obstacle type; §9's attribution can.
2. **But the total doesn't fall.** On the training maps with poles, collisions are 11.4% against 12.9%: within noise,
   and far from the 5% target. Collisions with cylinders, boxes and walls are, if anything, a bit higher than the 10°
   policy's (11.2% vs 8.6% without poles). With one seed per ring, that could be policy-to-policy variation.
3. **The held-out number dropped (79.9%), as `w3`'s did in §6.** The training-map numbers are level, so this is most
   likely the 20-layout held-out set again, not worse generalization. Two of four comparisons have now swung 6–8
   points on those 20 tiles. The held-out level-5 number isn't a reliable way to compare runs until it covers more
   layouts.
4. The 5° policy climbs out the top more often (2.5% of training-map episodes). It never gets over a 5 m obstacle
   under the 4 m ceiling.

**Where Phase 2 stands.** Two hypotheses, weak proximity shaping and coarse sensing, have each been tested with
retraining. Neither moved level-5 collisions off about 11–13%. Rather than guess at a third, the next steps should
measure:

- **Attribute each collision:** obstacle type, approach speed and angle, and how long the obstacle had been visible.
  That takes some code (the env knows the hit ray, not which obstacle it hit), but it replaces guessing.
- **Measure the noise:** 3 seeds of the baseline, to know how much two runs differ by chance. Every comparison so far
  is one seed.
- **Evaluate on more layouts:** all three test seeds (60 tiles) instead of one.

**Infrastructure fixed along the way:** `launch.py --eval` now passes the training run's `env.`/`agent.`
overrides to `play.py` (`da6f9ed`). Without that, the 5° checkpoint wouldn't load into the default 10° env. `--eval`
specs can also hold Hydra lists (`e79e070`).

## 9. Collision attribution

To stop guessing, the env can now attribute each collision to the obstacle it hit (`env.attribute_collisions=True`,
evaluation only). The terrain function records every layout it builds. `tile_layouts` maps them back to (row,
column) using Isaac Lab's generation order, and at a collision `nearest_obstacle` finds the footprint the drone is
closest to, counting the corridor's side walls. Each episode logs which kind it hit and how many of each kind its tile
had (exposure). Collisions also log the drone's speed, its closing speed on the obstacle, and its warning time (how
long the closest horizontal lidar reading had been within 1 m). Sanity check: at the moment of collision, drones are
on average 0.080 m from the attributed footprint, inside the 0.1 m collision radius, so the tile lookup is right.

Both checkpoints, level 5, training maps, 2000 episodes each (233 and 201 collisions):

| Hit | 10° ring: % of episodes | 5° ring: % of episodes | Per tile | 10°: rate per obstacle | 5°: rate per obstacle |
|---|---:|---:|---:|---:|---:|
| wall (with a door) | 4.2% | 2.6% | 0.92 | 4.6% | 2.8% |
| pole | 4.1% | 4.1% | 1.6 | 2.6% | 2.7% |
| cylinder | 1.8% | 1.8% | 5.3 | 0.34% | 0.34% |
| box | 1.1% | 1.3% | 3.6 | 0.30% | 0.36% |
| corridor side wall | 0.3% | 0.1% | | | |
| **total** | **11.6%** | **10.0%** | | | |

At the collision: speed 3.0 / 3.2 m/s, closing speed 1.7 / 1.5 m/s, warning time 0.41 / 0.39 s (10° / 5°).

1. **Walls and poles cause about two thirds of collisions** (8.3 of 11.6 points at 10°, 6.7 of 10.0 at 5°), though
   they're only about 2.5 of a level-5 tile's 11.5 obstacles. Per obstacle, each is 8–15× as dangerous as a cylinder
   or box.
2. **Cylinders and boxes, most of the clutter, are handled well**, at about 0.3% per obstacle with either ring.
3. **The 5° ring doesn't help with poles** (4.1% both), which corrects §8. The only difference is walls (4.2% →
   2.6%), on one seed per ring.
4. **Collisions happen at full cruise speed**, 3.0–3.2 m/s, with only about 0.4 s between something coming within
   1 m and the crash. The policy doesn't slow down for the hard cases.

What walls and poles have in common: both are thin in the direction the drone meets them. A pole is 6–12 cm across.
A wall's door edges are its 0.2 m-thick ends, which the drone has to pass within about half a metre of to get
through a 1.2 m door. Both also need a late, sharp correction at 3 m/s. Two things to try next:

- **Slow down near obstacles.** Lower `speed_limit` (for example 2 m/s), or tie the speed penalty to the closest
  lidar reading. That buys reaction time for exactly these cases, at some cost in time to goal.
- **Look at wall collisions directly.** A video of a few would show whether they happen at the door edges or
  because the drone misses the door. A finer attribution could also split segment ends from faces.

The held-out set is also still 20 layouts per evaluation (§6); any decision-making run should cover all three test
seeds.

## 10. Slowing down near obstacles, and a 60-layout evaluation

§9 found that collisions happen at full cruise speed with about 0.4 s of warning, mostly on walls and poles. This
section adds a speed limit that depends on clearance, so the policy can still cruise in open space, and an
evaluation set large enough to compare runs. Results are in §10.1.

**Clearance-based speed limit.** The over-speed term (Phase 1: −2 × (speed − limit)² per second above the limit)
now takes a per-env limit. With `env.slow_distance` > 0, the limit falls linearly from `speed_limit` (3 m/s) once
the closest horizontal lidar hit is `slow_distance` outside the 0.1 m collision radius, down to
`env.near_speed_limit` at contact ([`clearance_speed_limit`](../source/drone_nav/drone_nav/mdp/obstacles.py)).
Since progress pays 1 per metre, the best speed is always about 0.25 m/s above whichever limit applies (the Phase 1
derivation), so the policy should settle at roughly limit + 0.25 near obstacles too. `slow_distance=0.0` (the
default) turns it off, which keeps every earlier run reproducible. Like the proximity term, it uses the closest hit
in any direction, so passing a wall side-on also slows the drone; the cost is some time to goal.

A new metric, `Metrics/speed_near_obstacles`, is the mean horizontal speed while something is within
`warning_distance` (1 m), over episodes that came that close. It shows directly whether the policy slows down. For
the §5 baseline it should read close to the 3.0–3.2 m/s collision speeds from §9.

**60 held-out layouts in one evaluation.** Isaac Lab builds curriculum tiles column by column from the generator
seed, so `env.terrain.terrain_generator.num_cols=60` with seed 1001 gives level 5 sixty tiles: the 20 in today's
held-out set, plus 40 new ones. That's one Isaac Sim start instead of three seeds to pool, and
`attribute_collisions` works with any column count. 2000 episodes over 60 tiles is about 33 per tile. Spec:

```
--eval-episodes 2000 --eval eval_heldout60_l5:env.eval_terrain_level=5,env.terrain.terrain_generator.seed=1001,\
env.terrain.terrain_generator.num_cols=60,env.attribute_collisions=True,--num_envs=1024
```

**Planned runs** (seed 42, 4096 envs, 1500 iterations, curriculum on, each with the evaluation above):

| Run | Overrides | Question |
|---|---|---|
| `p2-slow-n15-d10-s42` | `env.slow_distance=1.0 env.near_speed_limit=1.5` | Does slowing near obstacles cut collisions? |
| `p2-slow-n10-d15-s42` | `env.slow_distance=1.5 env.near_speed_limit=1.0` | A stronger slowdown, starting further out |
| `p2-speed2-s42` | `env.speed_limit=2.0` | Is a slower drone everywhere just as good? |

Plus the §5 baseline checkpoint ([x7x2fl6t](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/x7x2fl6t))
on the same 60 layouts, with `launch.py --script play`, so every run is compared on the same maps. What to read:
held-out collisions (target < 5%), success (≥ 90%), the wall and pole rows of the attribution, time to goal and
`Metrics/speed_near_obstacles`. If a variant clearly wins, the next step is 3 seeds of it and of the baseline
(PLAN.md ablation (d)).

### 10.1 Results

Commit `cd702c4`, 3 training runs plus the §5 baseline checkpoint. Each was evaluated on the same 60 held-out level-5
layouts (2000 episodes, 1024 envs, collision attribution on). One seed each, so the binomial CIs (about ±1.5
points) don't include seed-to-seed spread.

| Level 5, 60 held-out layouts | Baseline (§5) | Mild slowdown | Strong slowdown | 2 m/s everywhere |
|---|---:|---:|---:|---:|
| Run | [x7x2fl6t](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/x7x2fl6t) | [mh19cygo](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/mh19cygo) | [i7i3du3y](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/i7i3du3y) | [zdta1m8u](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/zdta1m8u) |
| Overrides | | `slow_distance=1.0`, `near_speed_limit=1.5` | `slow_distance=1.5`, `near_speed_limit=1.0` | `speed_limit=2.0` |
| **Success** | 85.5% (83.8–86.9) | **89.8%** (88.4–91.1) | 17.1% | 85.7% (84.1–87.1) |
| **Collisions** | 13.4% (12.0–15.0) | **10.2%** (8.9–11.6) | 17.3% | 14.2% (12.7–15.8) |
| … with poles | 5.1% | 5.4% | 4.5% | 6.5% |
| … with cylinders | 4.2% | 2.2% | 5.0% | 3.6% |
| … with walls | 2.0% | 1.5% | 3.1% | 3.1% |
| … with boxes | 2.0% | 1.0% | 4.4% | 1.0% |
| Other endings | 1.2% too high | 0.05% too high | 47% flipped, 14% time-out, 5% too low | 0.2% too low |
| Speed near obstacles | 3.26 m/s | 2.75 m/s | 2.06 m/s | 2.22 m/s |
| Speed / warning time at a collision | 3.08 m/s / 0.45 s | 2.54 m/s / 0.53 s | 1.94 m/s / 1.16 s | 2.21 m/s / 0.70 s |
| Time to goal | 7.3 s | 8.2 s | 9.0 s | 10.6 s |
| End of training: envs on level 9 | 84% | 84% | 0% | 71% |

1. **The bigger evaluation set is harder than the old one.** The baseline scores 85.5% / 13.4% on 60 layouts,
   against 88.1% / 11.4% on the original 20. Cylinders cause many more collisions here (4.2% of episodes against
   1.8% on the training maps in §9), so the 20-tile sets were light on hard cylinder layouts. Every earlier
   held-out number should be read against 85.5%, not 88.1%.
2. **The mild slowdown is the best policy so far:** 89.8% success, 0.2 points short of the exit criterion, with
   collisions down 3.2 points from the baseline (CIs don't overlap). The drop comes mostly from cylinders (2.0
   points), then boxes (1.0) and walls (0.5); pole collisions rose slightly (0.3). The cost is 0.9 s more to the goal (8.2 s against 7.3 s). Collisions are still double the 5%
   target.
3. **Slowing down everywhere doesn't help.** At 2 m/s the drone hits obstacles more slowly and with more warning
   (0.70 s), yet collisions are 14.2%, the same as the baseline, and it takes 3.3 s longer. What helps is going
   fast in the open and slowing near clutter, not speed itself. With one seed per run, though, part of the
   difference between these two could be seed noise.
4. **A slowdown that's too steep breaks training.** Braking from 3 to 1 m/s over 1.5 m flipped the drone in 32%
   of training episodes by iteration 700. Its curriculum stayed around levels 2–4 to the end, and at level 5 it flips in 47% of
   episodes. Hard braking near obstacles, at CTBR's body-rate limits, tips it over.
5. **Poles are now the largest single cause**, at 4.5–6.5% of episodes in every run, whatever the speed. Even at
   2 m/s with 0.7 s of warning, the pole rate doesn't fall. A slowdown can't fix what the drone doesn't see:
   a 6 cm pole falls between 10° rays until it is about 0.35 m away, and the warning time measures the closest
   reading of anything, not of the pole. §9 found the 5° ring didn't help poles either, so the next pole
   experiment should look at what the drone sees before a pole hit, not just the ring's resolution.

**Next:**

- **3 seeds of the mild slowdown and of the baseline** on these 60 layouts. A 3.2-point gain on one seed each is
  promising, not settled.
- **Tune around the mild slowdown**, for example `near_speed_limit=2.0` (gentler) and `slow_distance=0.7`. The
  strong variant shows there's a cliff not far beyond this setting.
- **Look at pole collisions directly:** record the horizontal lidar readings for the 1 s before a pole hit (does
  any ray see the pole before contact?), and a few videos.

### 10.2 Three seeds: the mild slowdown against the baseline

Seeds 1 and 2 of both the §5 baseline and the mild slowdown (`slow_distance=1.0`, `near_speed_limit=1.5`), each
trained and evaluated exactly like §10.1's seed-42 runs: 60 held-out level-5 layouts, 2000 episodes, attribution on.
W&B: baseline [jnt5cv07](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/jnt5cv07) (s1),
[qvfplndy](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/qvfplndy) (s2); slowdown
[up0ct5qd](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/up0ct5qd) (s1),
[xn4qaayp](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/xn4qaayp) (s2).

| Level 5, 60 held-out layouts | Baseline: s42 / s1 / s2 | Mean ± SD | Slowdown: s42 / s1 / s2 | Mean ± SD |
|---|---|---:|---|---:|
| **Success** | 85.5 / 85.4 / 88.3 | **86.4 ± 1.7** | 89.8 / 89.9 / 89.4 | **89.7 ± 0.3** |
| **Collisions** | 13.4 / 12.8 / 11.6 | **12.6 ± 0.9** | 10.2 / 9.6 / 10.4 | **10.0 ± 0.4** |
| … poles | 5.1 / 5.4 / 6.2 | 5.6 | 5.4 / 6.0 / 6.0 | 5.8 |
| … cylinders | 4.2 / 3.7 / 1.6 | 3.2 | 2.2 / 1.8 / 3.3 | 2.4 |
| … walls | 2.0 / 1.7 / 2.1 | 1.9 | 1.5 / 1.1 / 0.3 | 1.0 |
| … boxes | 2.0 / 1.8 / 1.6 | 1.8 | 1.0 / 0.7 / 0.7 | 0.8 |
| Too high | 1.2 / 1.6 / 0.0 | 0.9 | 0.05 / 0 / 0 | 0.0 |
| Speed near obstacles (m/s) | 3.26 / 3.25 / 3.28 | 3.26 | 2.75 / 2.81 / 2.77 | 2.78 |
| Speed at a collision (m/s) | | 3.01 | | 2.54 |
| Time to goal (s) | | 7.4 | | 8.2 |

All in % of episodes unless stated. SD is across seeds.

1. **The mild slowdown is a real improvement.** Every slowdown seed beats every baseline seed: +3.3 points of
   success and −2.6 points of collisions on average. Welch's t is 3.4 (success) and 4.4 (collisions), on about 2–3
   degrees of freedom. With 3 seeds per arm that's strong but not conclusive; no rank test on 3 against 3 can get
   below p = 0.1. It is also more consistent: success varies by 0.3 points across seeds, against 1.7 for the
   baseline. The cost is 0.8 s more to the goal (8.2 s against 7.4 s).
2. **The gain comes from everything except poles.** Walls drop from 1.9% to 1.0%, boxes from 1.8% to 0.8%, cylinders
   from 3.2% to 2.4%, and climbing out the top from 0.9% to 0. Poles stay at about 5.7% in all six runs.
3. **Poles are now the entire gap to the exit criterion.** The slowdown's non-pole collisions add up to 4.2% of
   episodes, already under the 5% target. Its pole collisions (5.8%) are more than everything else combined. Phase
   2's next step is poles specifically, not more general shaping.
4. Success (89.7%) is 0.3 points under the 90% target. Fixing poles would clear it too.

**On poles, what we know:** they cause the same share of collisions with a 10° or 5° ring (§9), at 3 m/s or 2 m/s
(§10.1), with or without a slowdown. So it isn't ray spacing or reaction time alone. Leads to follow:

- **What does the drone see before a pole hit?** Log the horizontal lidar readings along the bearing of the hit pole
  for the last second before a collision. If no ray ever returns the pole, it's a sensing gap; if rays see it and
  the policy flies on, it's the policy.
- **The ring's vertical coverage:** the lidar is level, and poles are vertical, so this shouldn't matter, but it's
  worth confirming that the hit happens at the drone's own height.
- **Collision radius against pole size:** a pole is 3–6 cm in radius. Passing at 0.1 m from its axis counts as a
  collision, which is only 4–7 cm from its surface. That's a tighter margin than around cylinders, which the
  policy may not have learned to respect. Not a reason to change the criterion, but worth knowing.

## 11. What the drone sees before a collision, and a world-fixed ring

`env.attribute_collisions=True` now also replays the last second of the horizontal lidar ring at each collision
(`trace_steps=50` at the 50 Hz control rate). For every step it computes each ray's exact distance to the obstacle
that was hit (`ray_hit_distances`), and from that whether the ring returned that obstacle, whether it fell between
rays, or whether something closer hid it. Results are logged per obstacle kind as `Trace/<kind>_*`. Tests check the
replay against independent analytic ray casts.

**A bug the replay found.** The first run of the replay (with the same policy) reported that about half of the
cylinder and box collisions had never been returned by any ray. That's impossible, since a collision is detected
precisely because a ray returned something within 0.1 m. The replay assumed the rays turn with the drone's heading,
as §2 said. They don't. Isaac Lab v2.3.2's `ray_alignment="yaw"` rotates the rays' *start points* by the yaw but
leaves their *directions* alone (`RayCaster._update_ray_infos`: `ray_directions_w = self.ray_directions`). Our rays
all start at the drone's centre, so **the ring has always been fixed in the world**: ray *i* points at the same
compass bearing whatever the drone's heading.

- Unaffected: collision detection, the proximity term and every distance metric (`min_obstacle_distance`,
  `Collision/*`) take the minimum over the whole 360° ring, which doesn't depend on direction. All earlier
  collision, attribution and success numbers stand.
- Affected: the policy's observation. The lidar part is world-frame, while the goal direction and velocities are
  body-frame, and nothing in the observation gives the heading. The policy can only infer how the two frames
  relate indirectly, for example from the goal direction (goals are always roughly towards +x). That it reaches
  about 90% anyway is somewhat surprising.

The replay now uses world bearings, and §2 and the env's docstring are corrected.

**Result.** The mild-slowdown policy (seed 42, [mh19cygo](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/mh19cygo))
on the 60 held-out level-5 layouts, 2000 episodes; it reproduces §10.1 exactly (89.8% / 10.15%):

| Collisions with… | n | Never returned | Between rays (% of in-range steps) | First returned before impact | Distance then |
|---|---:|---:|---:|---:|---:|
| **pole** | 108 | 0% | **65%** | **0.45 s** | **1.38 m** |
| cylinder | 43 | 0% | 2% | ≥ 0.96 s (the whole window) | 2.38 m |
| box | 20 | 0% | 0% | ≥ 0.98 s | 2.37 m |
| wall | 31 | 0% | 0% | ≥ 0.98 s | 1.55 m |

1. **Pole collisions are a sensing problem.** Within range, a pole falls between the 10° rays on 65% of steps, so it
   flickers in and out of the observation. A ray first returns it on average 0.45 s and 1.4 m before impact.
   Every other obstacle is returned for the whole last second, from more than 2 m away (walls 1.6 m).
2. Every hit obstacle was returned at least once, and none was hidden by another obstacle. So it's late, flickering
   detection, not complete blindness. The MLP policy has no memory, so a pole that drops between rays is gone
   from its input until it reappears.
3. **The world-fixed ring makes this harder.** §9 found that a 5° ring alone didn't reduce pole collisions. A
   likely reason: the policy doesn't know its heading, so it can't map a single return at a fixed compass bearing
   to a body-relative dodge. Large obstacles survive this because they cover many rays continuously.

**Next:**

1. **Fix the ring's frame**, then retrain and compare on these 60 layouts. Either rotate the ray directions with the
   drone's yaw every step (set the sensor's per-env `ray_directions` before each read), or give the policy its
   heading (sin and cos of yaw) as two extra observations. The first is the cleaner observation. Behind a flag, so
   earlier runs stay reproducible.
2. **Then poles, with the frame fixed:** the 5° ring again, or memory (a GRU, PLAN.md Phase 3), so a pole that
   drops between rays isn't forgotten. The teacher may also use privileged geometry directly (PLAN.md §4.2's "K
   nearest obstacles" option).

## 12. A ring that turns with the drone

§11 found that the lidar ring is fixed in the world. `env.lidar_follows_yaw=True` (commit `5c9bb93`) makes it
follow the drone's heading instead. Before every read, the env turns the pattern's ray directions by each drone's
yaw (`yaw_rotate`), writes them into the sensor's per-env `ray_directions`, and forces a recompute. The ring stays
level, and the lidar observation then shares the body-yaw frame of the goal and velocities. The observation size is
unchanged (126). The flag is off by default, so every earlier run reproduces, and it needs
`ray_alignment="yaw"`. CPU tests check that a ray at pattern angle θ on heading ψ points at world bearing θ + ψ.

**Planned run:** `p2-slow-yawring-s42`, the mild slowdown (`slow_distance=1.0`, `near_speed_limit=1.5`) plus
`lidar_follows_yaw=True`, seed 42, evaluated on the 60 held-out level-5 layouts with attribution and the lidar
replay on. That makes it directly comparable with §10's seed-42 mild slowdown (89.8% / 10.2%, poles 5.4%) and §11's
replay. If it helps, seeds 1 and 2 next, then poles (5° ring or memory) on top.

### 12.1 Result: no improvement (one seed)

`p2-slow-yawring-s42` ([9k3d559o](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/9k3d559o)), against
the same settings with the world-fixed ring
([mh19cygo](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/mh19cygo), §10), both seed 42:

| | World-fixed ring | Ring follows yaw |
|---|---:|---:|
| Training at iteration 1000: success / collisions (envs on level 9) | 64% / 25% (57%) | 80% / 18% (74%) |
| End of training: success / collisions (envs on level 9) | 87.3% / 10.3% (84%) | 81.7% / 15.8% (87%) |
| **Held-out level 5 (60 layouts): success / collisions** | **89.8% / 10.2%** | **84.6% / 14.6%** |
| … poles / cylinders / boxes / walls | 5.4 / 2.2 / 1.0 / 1.5% | 8.0 / 4.3 / 1.6 / 0.7% |
| Time-out / flipped | 0 / 0% | 0.45 / 0.15% |
| Poles: between rays / first returned before impact | 65% / 0.45 s at 1.4 m | 56% / 0.85 s at 2.6 m |
| Speed near obstacles / time to goal | 2.75 m/s / 8.2 s | 2.70 m/s / 8.6 s |

1. **Turning the ring with the drone didn't help.** It was ahead midway through training, but finished worse at level
   9 and is 5 points worse at held-out level 5. That's one seed, against a world-fixed setting whose three seeds
   landed within 0.3 points of each other (§10.2), so it's a real signal against the change, though not proof.
2. **The frame mismatch was not what limited the policy.** A plausible reason the world-fixed ring works: it gives
   an absolute reference. The corridor's side walls always show up on the same rays, so the policy can work out its
   heading from the ring itself, not only from the goal direction.
3. **Poles get worse even though they're seen earlier.** With the ring following yaw, the hit pole is first returned
   0.85 s and 2.6 m before impact, about twice the world-fixed figures, yet pole collisions rise from 5.4% to 8.0%.
   Seeing a pole earlier, as a flickering single ray, doesn't by itself lead to avoiding it. That points more towards
   memory (keeping a pole in mind between returns) or a privileged nearest-obstacle observation than towards more
   rays.
4. Cost: $2.45, not the expected $1.10. The evaluation took 21 minutes instead of 1–5, which suggests the forced
   ray recompute costs more in `play.py` than it did in training (iteration time there was unchanged at 0.40 s),
   and the Nebius delete took 21 minutes. Not investigated.

**Decision:** keep `lidar_follows_yaw` off (the default). The world-fixed ring stays the baseline. For poles, the
next candidates are memory (a GRU policy) or giving the privileged teacher the nearest obstacles' geometry directly
(PLAN.md §4.2).

## 13. Privileged nearest-obstacle observation

§11–12 left poles looking like a sensing problem: they fall between rays and are first seen late. To test that
directly, the teacher can now be given exact obstacle geometry (PLAN.md §4.2's "K nearest obstacles" option).
`env.nearest_obstacles_obs=K` (commit `b9871a8`) packs every tile's footprints, including the side walls, into GPU
tensors from the recorded layouts. Each step it computes every drone's exact distance and nearest point to them
(`footprint_distances`), and appends, for the K closest, the unit direction in the body-yaw frame and distance /
5 m. That's 3K inputs after the lidar, which stays. The flag is off by default. Tests cross-check the torch distances
against the numpy footprint geometry on real layouts, every kind of footprint, points inside included. With K = 4
the policy has 138 inputs and training speed is unchanged (0.37 s per iteration).

`p2-slow-nearest4-s42` ([lnb0fe9q](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/lnb0fe9q)): the mild
slowdown plus K = 4, seed 42, against the same run without it ([mh19cygo](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/mh19cygo)):

| | Mild slowdown | + 4 nearest obstacles |
|---|---:|---:|
| Training at iteration 1000: success / collisions (envs on level 9) | 64% / 25% (57%) | 85% / 10% (68%) |
| End of training: success / collisions (envs on level 9) | 87.3% / 10.3% (84%) | 87.5% / 9.4% (90%) |
| **Held-out level 5 (60 layouts): success / collisions** | **89.8% / 10.2%** | **89.3% / 8.8%** |
| … poles / cylinders / walls / boxes / side walls | 5.4 / 2.2 / 1.5 / 1.0 / 0.1% | 4.5 / 2.4 / 0.9 / 0.8 / 0.2% |
| Time-out / flipped | 0 / 0% | 1.4 / 0.5% |
| Closest pass / speed near obstacles / time to goal | 0.34 m / 2.75 m/s / 8.2 s | 0.36 m / 2.70 m/s / 8.3 s |

1. **Exact obstacle positions barely move pole collisions:** 5.4% → 4.5%, with the pole's position in the
   observation every step whenever it's among the 4 nearest. So pole collisions aren't mainly a sensing problem
   either. Given perfect information, the policy still passes poles close enough to come within 0.1 m of their
   surface.
2. Overall it's a wash on one seed. Collisions fall 1.4 points (poles and walls), but time-outs (1.4%) and flips
   (0.5%) appear for the first time, so success is unchanged. The observation does speed up learning: the best
   mid-training numbers of any run, and the most envs on level 9 at the end.
3. The exit criterion is still not met (8.8% collisions against < 5%; 89.3% success against ≥ 90%).

**What's left for poles**, now that both sensing (§8, §11–12) and information (this section) have been ruled out
as the main cause, is how the drone passes them:

- **Near misses:** the collision rule fires when anything comes within 0.1 m of the drone's centre, and prop tips
  are at about 0.07 m. A 3–6 cm pole passed at 7–10 cm counts as a hit. An evaluation that records each pole hit's
  closest approach and heading relative to the pole would show whether these are genuine head-on hits or grazes.
- **Clearance pressure:** the proximity term is weak for thin obstacles (−1 × exp(−clearance / 0.3 m) per second
  for a brief pass). Shaping that's stronger near the collision radius, or a slightly larger margin in training
  than in evaluation, would push the policy to give poles more room.
- **Control limits at speed:** a late sideways correction at 2.7 m/s within CTBR's rate limits may simply not be
  possible. Lower speed near poles specifically, rather than everywhere, would test that.

## 14. Why some runs cost twice as much: Kit's shutdown hangs

The last two runs cost $2.45 and $2.96 against about $1 before, almost all of it idle box time. Timing every
launch from three clocks (`run_remote`'s start stamp; the Hugging Face commit each script makes as its last real
step; and the launcher's local log) shows where it goes:

| Run | Training: HF upload → process exit | Evaluation: HF upload → process exit |
|---|---:|---:|
| 6 runs on 2026-11-17 (slowdown s1/s2, baseline s1/s2, trace evaluations) | 5–40 s | 5–40 s |
| `p2-slow-yawring-s42` | 7.6 min | 21 min |
| `p2-slow-nearest4-s42` | 13.7 min | 37 min |

The training and evaluation work itself was quick: 2000 evaluation episodes take about 40 s on a warm box. On
some boxes, though, the process didn't exit for 8–37 minutes after it had saved everything, at the end of both
training and evaluation. After the upload, the only thing left is `simulation_app.close()`, Kit's shutdown. No ssh
failures were logged, so the launcher was polling normally; the remote `exit_code` file just wasn't written until
the process ended, and the box billed the whole time. It happened on Nebius and Crusoe alike, and with the lidar
replay off (training), so it isn't the provider or that code. It looks like a nondeterministic hang in Kit's
shutdown.

**Fix** (`scripts/kit_exit.py`): `train.py` and `play.py` now end with `kit_exit.close_app(simulation_app)`. It
closes Kit normally, but if that hasn't finished within 60 s it prints `[WARN] Kit shutdown still running after
60 s; exiting without it` and exits with code 0. Code 0 is correct: the checkpoint, W&B summary, HF upload and
results JSON are all written before this point. Tests check both paths, a hung close and a normal one, in a
subprocess. Limitation: if the hang holds Python's interpreter lock in native code, the watchdog thread can't
run, and `run_remote.sh`'s outer `timeout` remains the backstop. The warning line in a future run's log will show
the watchdog doing its job.

One stall isn't explained by this. On 2026-11-16, an evaluation that crashed at start-up (exiting through
`os._exit`, not Kit's shutdown) still took the launcher 18 minutes to notice. Worth watching for.

## 15. How the drone meets poles: near misses

§13 ruled out sensing and information as the main cause of pole collisions, which left *how the drone passes
poles*. `env.attribute_collisions=True` now also logs `NearMiss/*` (commit `b7eafdd`):

- **Per collision**, from the lidar replay's 1 s trace (`of.near_miss_summary`): the obstacle's radius, the speed,
  the approach angle (between the velocity and the direction to the obstacle's nearest point; 0° is flying straight
  at it, 90° is moving along its surface), and the **straight-line miss** 0, 250 and 500 ms before impact. The
  straight-line miss is how far the drone's centre would have passed from the surface if it had flown straight
  on from there. It is negative when the line goes through the obstacle. Round obstacles only.
- **Per episode**, from exact footprint distances every step: the closest the drone came to each kind of obstacle,
  over episodes that came within 1 m of one (`NearMiss/<kind>_closest`). This puts the collisions next to the
  passes that didn't hit.

`play.py` keeps the raw `NearMiss/*` values in the results JSON. Both policies were evaluated with no retraining, on
the 60 held-out level-5 layouts, 2000 episodes each, and reproduce their earlier numbers exactly: the mild
slowdown (`mh19cygo`, §10; 89.8% / 10.2%, 108 pole hits as in §11) and the same plus 4 nearest obstacles
(`lnb0fe9q`, §13; 89.3% / 8.8%, 89 pole hits). The two evaluations cost about $0.70 each.

**Pole collisions** (mild slowdown; the nearest-4 policy is within a few points on every row):

| | 500 ms before | 250 ms before | At impact |
|---|---:|---:|---:|
| Straight line goes through the pole | 26% | 30% | 20% |
| … clears the surface by 0–5 cm | 24% | 36% | 44% |
| … clears it by 5–10 cm | 15% | 20% | 36% |
| … clears it by ≥ 10 cm | 35% | 14% | 0% |
| Median straight-line miss | 5.0 cm | 3.1 cm | 3.6 cm |

At impact: approach angle median 41° (29% under 30°, 30% over 60°), speed 2.7 m/s, pole radius 3–6 cm (median
4.4 cm). For comparison, at 250 ms before
impact, 79% of cylinder hits are on a line straight through the cylinder: those really are crashes into it.

**Closest pass per episode**, over episodes that came within 1 m of that kind (mild slowdown):

| | Passes | < 0.1 m (hit) | 0.1–0.15 m | 0.15–0.2 m | **Within 0.2 m** | Hit, given within 0.2 m | Median |
|---|---:|---:|---:|---:|---:|---:|---:|
| **pole** | 1045 | 10.3% | 3.8% | 5.5% | **19.6%** | 53% | 0.45 m |
| cylinder | 1747 | 2.5% | 1.3% | 2.1% | 5.8% | 43% | 0.51 m |
| box | 1551 | 1.3% | 1.3% | 0.8% | 3.4% | 38% | 0.56 m |
| wall | 1361 | 2.5% | 2.3% | 3.5% | 8.2% | 30% | 0.45 m |
| + 4 nearest: pole | 1065 | 8.4% | 2.6% | 5.0% | 16.0% | 52% | 0.47 m |

1. **Most pole collisions are grazes, not crashes.** At impact, 80% of pole hits are on a line that would clear
   the pole's surface, by 3.6 cm at the median. The drone's centre is within 10 cm of the surface (the collision
   rule), but its path goes past the pole. Only about a quarter are head-on: on a line through the pole 250–500 ms
   out, never corrected in time. Cylinder hits look the opposite way: most are on a line straight through it.
2. **The policy passes poles tight on purpose.** Of the episodes that pass a pole, 19.6% come within 0.2 m of its
   surface, against 3.4–8.2% for the other kinds. Within that band, a pole pass is only a little more likely to
   end in a hit (53% against 30–43%). So the excess comes from how often the drone cuts close to poles, not from
   poles being harder to clear once it does. A thin pole costs almost no detour to shave past, and the reward
   barely charges for a close pass (the proximity term is −1 × exp(−clearance / 0.3 m) per second, for a pass
   lasting a few hundredths of a second).
3. **Perfect information doesn't change this.** With the 4 nearest obstacles' exact geometry in its observation,
   the policy still comes within 0.2 m on 16% of pole passes, with the same graze-dominated pattern. That matches
   §13: the policy knows where the pole is and chooses to pass it closely.
4. A third of pole hits (35%) were on a line that would clear the pole by ≥ 10 cm 500 ms out, and drifted in.
   Plausibly the drone was turning back towards the goal or around a neighbouring obstacle. That fits the same
   explanation: the policy is content with a few centimetres of margin.
5. Kit shut down normally on both boxes (no `Kit shutdown still running` warning), so §14's watchdog wasn't needed.

**Next: clearance pressure, aimed at grazes.** In order:

- **A training margin larger than the evaluation's:** `collision_radius=0.15` in training, 0.1 m in evaluation.
  Grazes at 0.1–0.15 m become training crashes, which is exactly the band that's over-represented for poles. It
  needs `env.collision_radius=0.1` passed to the evaluation by hand, since `--eval` reuses the training overrides.
  `collision_radius` also offsets the proximity term, so this shifts the shaping by 5 cm as well.
- **A proximity term that rises steeply near the collision radius,** for example a shorter `proximity_sigma` with a
  larger weight. §6's sweep only went the other way (σ = 0.6 m, weight 3, or both), which spreads the
  penalty further out instead of concentrating it on near misses.
- Slowing down near poles specifically addresses only the head-on quarter, so it comes after these.

## 16. A larger collision margin in training

§15's proposal: train with `collision_radius=0.15` and evaluate at 0.1 m, so the 0.1–0.15 m grazes that are
over-represented for poles become training crashes. `collision_radius` also offsets the proximity term and the
slowdown's clearance, so both shaping terms move out by 5 cm with it.

`p2-slow-margin15-s42` ([8z4i2c6p](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/8z4i2c6p), commit
`1c2e99e`): the mild slowdown, seed 42, 4096 envs, 1500 iterations, curriculum on, `env.collision_radius=0.15`. It
was evaluated on the 60 held-out level-5 layouts (2000 episodes, attribution on) at the usual 0.1 m rule
(`eval_heldout60_l5`) and at its own 0.15 m (`eval_heldout60_l5_r15`). It's compared with the same settings at 0.1 m
([mh19cygo](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/mh19cygo), §10 and §15):

| | Trained at 0.1 m | Trained at 0.15 m |
|---|---:|---:|
| Training at iteration 1000: success / collisions (envs on level 9) | 64% / 25% (57%) | 69% / 26% (42%) |
| End of training: success / collisions (envs on level 9), at its own margin | 87.3% / 10.3% (84%) | 79.7% / 15.3% (74%) |
| **Held-out level 5 at 0.1 m: success / collisions** | **89.8% / 10.2%** | **89.8% / 9.9%** |
| … poles / cylinders / walls / boxes / side walls | 5.4 / 2.2 / 1.6 / 1.0 / 0.1% | **6.1** / **1.0** / 1.5 / 0.9 / 0.4% |
| Held-out level 5 at 0.15 m: success / collisions | | 83.9% / 15.9% |
| Closest pass (mean) / speed near obstacles / time to goal | 0.34 m / 2.75 m/s / 8.2 s | 0.36 m / 2.70 m/s / 8.3 s |

Closest pass per episode, held-out at 0.1 m, over episodes that came within 1 m of that kind (as in §15):

| Within 0.2 m of… | Trained at 0.1 m | Trained at 0.15 m | Median pass, 0.1 m → 0.15 m |
|---|---:|---:|---:|
| **pole** | 19.6% | **22.1%** | 0.45 → 0.44 m |
| cylinder | 5.8% | 4.9% | 0.51 → 0.57 m |
| box | 3.4% | 2.7% | 0.56 → 0.64 m |
| wall | 8.2% | 6.2% | 0.45 → 0.46 m |

1. **No overall gain.** Held-out success and collisions are the same as at 0.1 m (89.8% / 9.9% against 89.8% /
   10.2%, CIs almost identical). The exit criterion is still not met.
2. **The margin moved the policy away from large obstacles, not from poles.** Cylinder hits halve (2.2% → 1.0%), and
   the median pass widens by 6 cm for cylinders and 8 cm for boxes. For poles nothing moves: hits 5.4% → 6.1%,
   passes within 0.2 m 19.6% → 22.1%, median pass unchanged. On one seed and about 1050 pole passes, the pole
   differences are within noise (about 1.5 standard errors), so read it as "no effect on poles", not "worse".
3. **Pole hits still look the same:** 72% are grazes at impact (80% at 0.1 m), a third are on a line through the
   pole 250 ms out, and the speed at a pole hit is a little lower (2.5 against 2.7 m/s).
4. So the policy *can* trade path length for clearance, and does it for the obstacles it sees well. A training
   penalty on 0.1–0.15 m passes doesn't make it give poles more room. The simplest reading is that it doesn't know
   precisely where a pole's surface is when it passes. Within range, a pole falls between the 10° rays 65% of the
   time (§11). When it is returned, the single ray only places it within about ±5° (±9 cm at 1 m), which is as
   large as the margin being asked for. That doesn't square with §13, where exact geometry of the 4 nearest
   obstacles barely helped (16% of pole passes within 0.2 m). That experiment gave information without extra
   pressure, and this one gives pressure without extra information.

**Infrastructure.** The launcher's verify step failed on this run, so the evaluations had to be launched
separately (play-only, about $0.70 extra). W&B's run summary was missing `final_checkpoint`, `hf_url` and
`training_time_s`. The `wandb-summary.json` the client uploaded had all three, and the checkpoint was on W&B and
HF. That looks like a server-side merge race on the W&B server. The run's summary was repaired by hand, and
`launch.py` now restores missing keys from that file before verifying (`0ed0406`). Kit shut down normally on all
three boxes. Cost: $0.95 training + $0.55 + $0.15 evaluations.

**Next: information and pressure together.** Train with the 0.15 m margin *and* `nearest_obstacles_obs=4` (§13), one
seed, same evaluation. If pole passes within 0.2 m drop well below 16–22%, poles need both, and the Phase 2 teacher
should carry the privileged geometry. If they don't, it's time to stop tuning the teacher for poles: either accept
the 89.8% / ~10% teacher and start Phase 3 (whose GRU is the other lead for poles), or relax the exit criterion for
poles explicitly.

## 17. Information and pressure together: 0.15 m margin plus the 4 nearest obstacles

§13 gave the policy exact obstacle geometry without extra pressure to keep clear (pole passes within 0.2 m: 16%).
§16 added the pressure without the information (22%). This run does both: `p2-slow-margin15-nearest4-s42`
([wn2l0g9q](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/wn2l0g9q), commit `3d4264b`). That's the mild
slowdown, `env.collision_radius=0.15`, `env.nearest_obstacles_obs=4`, seed 42, 4096 envs, 1500 iterations,
curriculum on. Evaluation as in §16, on the 60 held-out level-5 layouts at the usual 0.1 m rule.

| Held-out level 5, at 0.1 m | Slowdown (§10) | + 4 nearest (§13) | + 0.15 m margin (§16) | **+ both** |
|---|---:|---:|---:|---:|
| **Success** | 89.8% | 89.3% | 89.8% | **93.8%** (CI 92.7–94.8) |
| **Collisions** | 10.2% | 8.8% | 9.9% | **5.65%** (CI 4.7–6.7) |
| … poles | 5.4% | 4.5% | 6.1% | **3.75%** |
| … cylinders | 2.2% | 2.4% | 1.0% | 1.35% |
| … walls | 1.6% | 0.9% | 1.5% | **0.15%** |
| … boxes / side walls | 1.0 / 0.05% | 0.8 / 0.25% | 0.9 / 0.35% | 0.4 / 0% |
| Time-out / flipped | 0 / 0% | 1.4 / 0.5% | 0 / 0.3% | 0.25 / 0.1% |
| Pole passes within 0.2 m (median pass) | 19.6% (0.45 m) | 16.0% (0.47 m) | 22.1% (0.44 m) | **13.9% (0.53 m)** |
| Wall passes within 0.2 m | 8.2% | 3.8% | 6.2% | 1.3% |
| Closest pass (mean) / time to goal / path efficiency | 0.34 m / 8.2 s / 0.90 | 0.36 m / 8.3 s / 0.90 | 0.36 m / 8.3 s / 0.90 | 0.40 m / 8.5 s / 0.89 |

Training, at its own 0.15 m rule: 67% / 26% at iteration 1000 (56% of envs on level 9), and **90.7% / 8.3%** at the
end (87% on level 9), the best end-of-training numbers of any run, even under the stricter rule. At 0.15 m on the
held-out maps it scores 90.4% / 9.2%.

1. **Together they work, though neither did alone.** Collisions almost halve (10.2% → 5.65%) and success rises 4
   points (89.8% → 93.8%). The confidence intervals don't overlap the slowdown's on either. Information alone or
   pressure alone each moved the overall rate by at most 1.4 points. The effect is far more than additive.
2. **The policy now gives poles room.** Pole passes within 0.2 m drop to 13.9%, and the median pole pass widens by
   8 cm (0.45 → 0.53 m). With geometry it can rely on, the margin pressure finally reaches poles. Wall collisions
   nearly disappear too (1.6% → 0.15%).
3. **What's left of poles changed shape.** 69% of the remaining pole hits were on a line clearing the pole by ≥ 10 cm
   500 ms before impact (35% for the slowdown), and they happen slower (1.9 against 2.7 m/s). So most of what's
   left is drifting into a pole while manoeuvring, not shaving past one.
4. **The cost is small:** 0.2 s more to the goal, path efficiency 0.89 against 0.90.
5. **Exit criterion:** success (93.8% ≥ 90%) is met; collisions (5.65% against < 5%) aren't quite, and the CI
   (4.7–6.7%) straddles the target. That's one seed. §10.2's seeds differed by up to about a point on
   collisions, so this needs seeds 1 and 2 before it counts.

**What it means for the plan.** The Phase 2 teacher is privileged by design (PLAN.md §4.2), so it should carry the
nearest-obstacle geometry, which the Phase 3 depth student will learn to do without. One thing to keep in mind for
distillation: the student gets the lidar/depth view only, which §11–16 showed doesn't place poles precisely. The
student will need memory (the GRU) or more resolution to match the teacher on poles.

**Infrastructure.** W&B dropped the same three summary keys again (two runs out of two today, so it's systematic,
not a one-off). The launcher's new restore step (§16) caught it and the evaluations ran. But the first evaluation
started 2 s after the restore, still read the old summary, and skipped its Hugging Face upload. `play.py` now reads
`final_checkpoint` and `hf_url` through `scripts/wandb_summary.py`, which falls back to the uploaded
`wandb-summary.json` (`846b24c`). That evaluation was re-run play-only and reproduced 93.80% / 5.65% exactly. Kit
shut down normally on both boxes. Cost: $0.99 training with both evaluations, plus $0.55 for the re-run.

**Next:** seeds 1 and 2 of this configuration, with the same evaluation, to check against the exit criterion (§17.1).

### 17.1 Three seeds: seed 42 was the lucky one

Seeds 1 and 2 of the same configuration, trained and evaluated exactly like seed 42. W&B:
[z5m5nb1s](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/z5m5nb1s) (s1),
[l6ijehy3](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/l6ijehy3) (s2); commit `deafff1`.

| Level 5, 60 held-out layouts, at 0.1 m | s42 / s1 / s2 | **Mean ± SD** | Slowdown alone (§10.2), mean ± SD |
|---|---|---:|---:|
| **Success** | 93.8 / 88.1 / 91.2 | **91.0 ± 2.9** | 89.7 ± 0.3 |
| **Collisions** | 5.65 / 8.4 / 8.4 | **7.5 ± 1.6** | 10.0 ± 0.4 |
| … poles | 3.75 / 5.0 / 4.55 | 4.4 | 5.8 |
| … cylinders | 1.35 / 1.05 / 1.35 | 1.25 | 2.4 |
| … walls | 0.15 / 1.25 / 1.0 | 0.8 | 1.0 |
| … boxes | 0.4 / 0.85 / 1.2 | 0.8 | 0.8 |
| Flipped / too high / time-out | 0.1 / 2.7 / 0.2; 0 / 0.8 / 0; 0.25 / 0 / 0.15 | 1.0 / 0.3 / 0.1 | 0 / 0 / 0 |
| Pole passes within 0.2 m | 13.9 / 17.2 / 15.7 | 15.6 | 19.6 (s42 only) |
| Time to goal (s) | 8.46 / 8.35 / 8.50 | 8.4 | 8.2 |
| End of training, at 0.15 m: success / collisions | 90.7/8.3, 93.8/6.2, 86.9/13.1 | | |

All in % of episodes unless stated. SD is across seeds.

1. **The gain is real but about half what seed 42 suggested.** Collisions fall 2.6 points against the slowdown
   alone (7.5% against 10.0%; Welch's t = 2.7). Every seed is below every slowdown seed. Success rises only 1.3
   points on average (t = 0.8), because seed 1 trades collisions for flips.
2. **Poles improve on every seed** (4.4% against 5.8%) and cylinders halve, as in §16. Poles are still the largest
   cause, and pole passes within 0.2 m stay well above the other kinds (15.6%).
3. **It's much less consistent across seeds.** The slowdown alone landed within 0.3 points of 89.7% on all three
   seeds; this varies from 88% to 94%. Seed 1 flips on 2.7% of episodes and climbs out the top on 0.8%, failure
   modes no slowdown seed showed. The extra observation and the stricter margin make training less stable.
4. **Exit criterion still not met.** The mean success is 91.0% (≥ 90% met on two seeds of three), but collisions
   average 7.5% against < 5%, and no seed is under 5%.

**Where Phase 2 stands.** Over three seeds each, the teacher has gone from 86.4% / 12.6% (§10.2 baseline) to
91.0% / 7.5%. The remaining collisions are mostly poles, then a spread of everything else. Options:

- **Keep tuning the teacher:** a larger margin (0.2 m), longer training (3000 iterations), or a steeper proximity
  term near the collision radius. Each costs about $1.50 per three seeds. None is
  obviously the one that gets to < 5%.
- **Accept this teacher and start Phase 3.** The teacher's job is to be good enough to distill; 91% / 7.5% from
  privileged geometry is a reasonable target for a depth student whose own exit bar is ≥ 85% success. Seed 1's flips
  argue for distilling from seed 42 or seed 2.
- **Relax the < 5% collision criterion** for the teacher explicitly and record why (poles 3–6 cm thick against a
  10 cm collision radius are the hardest case, and the student will face them with less information).

**Infrastructure.** W&B dropped the same summary keys on both runs, and both launchers restored them; both
evaluations uploaded their results. Seed 1's launcher finished normally ($0.49). Seed 2's launcher was killed with
the Claude Code session right after it sent `brev delete`, and the delete never took effect. The box sat RUNNING
for about 15 hours (about $33) until `reap.py` found and deleted it the next morning. The fix that exists is
procedural (memory note: wait for every launcher's teardown before a session ends, and run `reap.py` at the start
of each session). A launcher that confirms the delete before it exits, or a box-side idle shutdown, would be the
code fix.

## 18. Closing Phase 2: the teacher for Phase 3

Phase 2 ends without meeting its collision criterion (< 5%; the best configuration averages 7.5% over three seeds),
and that is now a recorded decision rather than an open item: the 3–6 cm poles against a 10 cm collision radius are
the residual, and §15–17 showed they need exact geometry plus pressure to improve at all. Success (≥ 90%) is met on
average. Phase 3 starts from this teacher rather than spending more on it.

**The Phase 3 teacher:** `p2-slow-margin15-nearest4-s42`
([wn2l0g9q](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/wn2l0g9q), `model_1499.pt`): 93.8% / 5.65% on
the 60 held-out level-5 layouts, the best of the three seeds and free of seed 1's flips. Its observation is the
goal task's 18 inputs, the 108-ray lidar ring and the 4 nearest obstacles (138), with the mild slowdown
(`slow_distance=1.0`, `near_speed_limit=1.5`); it was trained at `collision_radius=0.15` and is evaluated at 0.1 m.
Seed 2 ([l6ijehy3](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/l6ijehy3), 91.2% / 8.4%) is the
fallback.

**Not done:** PLAN.md's Phase 2 ablations (no proximity shaping, no curriculum, number of rays, 3 seeds each). §6
(proximity sweep) and §8 (5° ring) cover parts of them on one seed; the rest move to Phase 5's benchmark table.

