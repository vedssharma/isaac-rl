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
  casts 3 channels (−15°, 0°, +15°) × 36 rays (every 10°) against the terrain mesh, out to 5 m. It turns with
  the drone's yaw but stays level with the horizon, so pitching forward to accelerate doesn't point it at the
  ground. Each ray's distance / 5 m is appended to the Phase 1 observation: 18 + 108 = 126 inputs.
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
