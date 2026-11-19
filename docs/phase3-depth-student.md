# Phase 3: a depth-based student

PLAN.md Phase 3 trains a policy that sees the obstacle field only through a forward depth camera, by distilling the
Phase 2 teacher (Option A, DAgger) and, for comparison, by PPO from depth (Option B). This note records the parts as
they land.

Status:

- [x] `Drone-Nav-Depth-v0`: a ray-cast depth camera, a noise model, and the teacher's observation alongside (§1)
- [x] The camera checked against exact geometry on 1024 drones: 99.97% of pixels agree (§2)
- [x] CNN student and `scripts/distill.py` (DAgger from the frozen teacher) (§3)
- [x] First distillation: **3.6% success** on held-out level 5. The teacher flies backwards: its direction of
  travel is inside a forward camera's view 1.9% of the time (§4)
- [ ] A teacher that faces where it flies, then distil again (§5)
- [ ] Option B: PPO from depth with an asymmetric critic
- [ ] GRU memory
- **Exit (G3):** ≥ 85% success at difficulty 5 from depth only.

**The teacher** (docs/phase2-obstacles.md §18): `p2-slow-margin15-nearest4-s42`
([wn2l0g9q](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/wn2l0g9q), `model_1499.pt`), 93.8% / 5.65% on
the 60 held-out level-5 layouts. Its observation is the goal task's 18 inputs, the 108-ray lidar ring and the 4 nearest
obstacles (138).

## 1. The env: `Drone-Nav-Depth-v0`

`source/drone_nav/drone_nav/tasks/depth/depth_env.py` (commit `aca604d`). It's `Drone-Nav-Obstacles-v0` with the same
world, rewards, terminations and curriculum, and a different observation:

| Group | Contents | Size |
|---|---|---:|
| `policy` (the student) | the goal task's 18 inputs (body velocities, gravity, last action, goal direction and distance, altitude), then the depth image scaled to [0, 1] (1 = nothing within 6 m), row-major | 18 + 3072 = 3090 |
| `teacher` | the obstacle task's privileged observation, unchanged: goal (18), lidar ring (108), 4 nearest obstacles (12) | 138 |

The defaults are the teacher's settings (mild slowdown, `nearest_obstacles_obs=4`), so the `teacher` group is exactly
what it was trained on. The collision rule is the evaluation one, 0.1 m.

**The camera** is Isaac Lab's `RayCasterCamera`: Warp ray casts against the terrain mesh, so it needs no RT cores
(PLAN.md §3.1). It's a 64 × 48 pinhole with a 90° horizontal and 74° vertical field of view (focal length 10,
aperture 20; square pixels, 32 px focal length), on the nose 4 cm ahead of the centre, looking along the body's +x
axis. It returns distance to the image plane, capped at 6 m. Unlike the lidar ring's "yaw" alignment (Phase 2 §11),
`RayCasterCamera` applies the camera's full orientation to its rays, so the image pitches, rolls and yaws with the
drone.

**Noise** (`drone_nav/sensors/depth.py`, PLAN.md §5.4), roughly a stereo camera:

- Gaussian noise with std 0.01 m + 0.005 m × depth², which is 1.5 cm at 1 m and 19 cm at 6 m;
- 1% of pixels drop out (read max range, "no return");
- pixels on a depth edge (a jump of more than 0.3 m to a neighbour) drop out with probability 0.3, as stereo
  matching fails at occlusion boundaries;
- everything clipped to [0, 6 m].

`env.depth_noise=False` turns it off. CPU tests cover the pinhole geometry, the noise statistics and the observation.

**Throughput:** `env.step` takes 18 ms with 1024 envs (56k env-steps/s) on an L40S, including the camera's 3.1M
rays per step. Training throughput at 4096 envs is still to be measured, with the student network in the loop.

## 2. Checking the camera against exact geometry

A depth image that is mirrored, in the wrong frame, or off by the camera offset would still look plausible, so
`scripts/check_depth.py` compares it with ground truth. `obstacle_field.camera_depth_replay` computes the exact
image of a level camera in a tile: obstacles are prisms from the ground to 5 m, the ground is the plane z = 0, and
pixel (r, c)'s ray is (1, y_c, z_r) in the camera frame, as Isaac Lab's pinhole pattern builds it (pixel centres,
row 0 at the top, column 0 on the left). For a vertical surface, a pixel's image-plane depth is the horizontal hit
distance divided by sqrt(1 + y_c²). CPU tests check the replay on walls, the ground, poles off the optical axis,
headings and short obstacles.

On a GPU box, the script builds the depth env with noise off, on level 5 with 1024 envs. It resets, takes one
zero-action step (the drones are then level, at random headings), and compares every pixel of every camera with
the replay from that camera's own pose (`launch.py --script check_depth`).

**First run** (`p3-check-depth-v1`): 98.1% agreement. Obstacles matched (boxes, walls and side walls 99.9–100%, poles
99.7%, cylinders 99.4%). The misses were all "camera sees something where the replay sees nothing", in 181 of 1024
envs. The images showed why: drones in the start strip facing backwards look into the neighbouring tile, which the
replay didn't include. The camera was right.

**Second run** (`p3-check-depth-v2`, commit `11fd051`), with the 8 neighbouring tiles' footprints moved into the
tile's frame. All 1024 cameras were level after the step:

| Expected to see | Pixels | Agree (± 3 cm + 1%) | Camera nearer | Camera farther |
|---|---:|---:|---:|---:|
| side wall | 1,142,704 | 99.996% | 0 | 51 |
| ground | 856,585 | 99.999% | 5 | 0 |
| nothing within 6 m | 809,549 | 99.982% | 142 | 0 |
| wall | 145,056 | 99.992% | 8 | 4 |
| box | 123,842 | 99.952% | 21 | 39 |
| cylinder | 65,332 | 99.302% | 0 | 456 |
| pole | 2,660 | 97.406% | 0 | 69 |
| **all** | 3,145,728 | **99.975%** | | |

The worst camera agrees on 98.9% of its pixels; 447k pixels (14%) see into a neighbouring tile.

1. **The camera is right:** frame, orientation (no mirroring), offset, intrinsics and range all match exact geometry.
2. **Round obstacles read slightly farther at their silhouettes,** and only there: every cylinder and pole miss is
   "farther". The terrain meshes a circle as a 16-sided polygon inside it (`_CIRCLE_SIDES`), so a ray that just
   clips the true circle can miss the mesh and see what's behind. That's a property of the world, not the camera,
   and the teacher's lidar sees the same mesh. For a 3–6 cm pole the polygon is only 0.6–1.2 mm smaller in radius, but at 64
   pixels across 90° a pole 2 m away is one or two pixels wide, so its edges are a large share of its pixels.
3. **Thin poles are tiny in the image:** 2,660 pole pixels in 1024 images against 65k for cylinders. The student
   will have even less to go on for poles than the lidar ring had (Phase 2 §11), which is worth remembering when its
   pole collisions are compared with the teacher's.
4. With noise on, the share of max-range pixels rises from 25.7% to 29.8% (dropout plus edge dropout).

Cost: $0.56 and $0.19 for the two checks (the first box's delete took 13 minutes).

**Next:** the CNN student and `scripts/distill.py`. RSL-RL 3.1.2 (on the boxes) has a `DistillationRunner`, but its
`StudentTeacher` policy is MLP-only and is chosen by `eval(class_name)` in RSL-RL's own namespace. So the DAgger
loop will be our own script, as PLAN.md §8 already names it:

- roll out the student, take the frozen teacher's action mean on the same states as the label, and train with MSE;
- a small CNN encoder (64-d latent) on the depth image, concatenated with the 18 goal/proprioception inputs, into
  an MLP; a GRU comes after;
- load the teacher from its RSL-RL checkpoint, including its observation normalizer.


## 3. The student and DAgger distillation

**The student** (`drone_nav/policies/depth_student.py`, 0.35M parameters) has three parts:
- **CNN encoder:** four stride-2 convolutions (32, 64, 64, 128 channels; 5×5 then 3×3, ELU) turn the 48×64 depth
  image into a 64-d latent.
- **Normalizer:** a running normalizer on the 18 goal and motion inputs, like RSL-RL's empirical normalization.
- **MLP:** 256, 256, 128 → the 4 CTBR actions.

It is pure torch, with CPU tests for the normalizer, the observation split (row-major image), checkpoints and a
small end-to-end fit.

**Distillation** (`scripts/distill.py`) is DAgger:
- Each iteration rolls out 24 steps in every env.
- The executed action is the teacher's with probability β, else the student's. β falls linearly from 1 to 0 over
  the first 100 iterations.
- Every visited state is labelled with the frozen teacher's action. RSL-RL computes it from the env's `teacher`
  observation group (`obs_groups = {"policy": ["teacher"], ...}`), so the teacher runs exactly as in training and
  evaluation.
- The student is fitted to the labels clamped to [−1, 1] with MSE: 2 epochs of 4 minibatches per rollout, Adam
  at 5e-4, gradient clipping at 1.0.
- The curriculum follows whoever flies.

RSL-RL 3.1.2's own `DistillationRunner` couldn't be used: its `StudentTeacher` is MLP-only. W&B, checkpoints, the
Hugging Face upload and the summary keys work as in `train.py`, so `launch.py --script distill` verifies the run.
`play.py` recognises a student checkpoint and evaluates it like any policy.

**Smoke run** (`p3-distill-smoke-s42`, 1024 envs, 30 iterations, $0.35): the whole pipeline ran, from teacher
loading through training, verification and evaluation of the student on held-out maps.

**First run** (`p3-distill-s42`,
[oadiizgu](https://forge.coreweave.com/wandb/vedssharma1/isaac-rl/runs/oadiizgu), commit `7cc0553`): 4096 envs,
1000 iterations, 2.1 s per iteration (47k samples/s), 35 minutes, $1.37.

| Iteration | Teacher share β | Imitation loss | Training success / collisions | Envs mostly on |
|---:|---:|---:|---:|---|
| 50 | 0.50 | 0.004 | 100% / 0% | level 0 |
| 100 | 0.00 | 0.048 | 87% / 13% | level 1 |
| 250 | 0 | 0.063 | 50% / 51% | levels 1–3 |
| 500 | 0 | 0.064 | 51% / 49% | levels 1–3 |
| 999 | 0 | 0.062 | 48% / 52% | levels 1–3 |

Held-out level 5 (60 layouts, 2000 episodes, 0.1 m rule): **3.6% success, 96.4% collisions**. By obstacle:
walls 40.8%, boxes 27.5%, cylinders 23.4%, poles 3.6%, side walls 1.1%. Mean speed 2.7 m/s; collisions come
0.46 s after something is first within 1 m.

The student learns the teacher well while the teacher is flying (loss 0.004). Once it flies alone, the loss
settles 15 times higher and it never gets past level 3, where the teacher is at nearly 100%. That is a student that
can't see what the teacher reacts to, not one that lacks training: nothing improved over the last 750 iterations.

## 4. Why: the teacher flies backwards

The teacher's lidar ring is fixed in the world (Phase 2 §11), and its nearest-obstacle input covers every
direction. So its heading doesn't matter to it, and nothing in its reward asks it to face where it flies. Drones
spawn with a random heading, and a quadrotor flies in any direction without turning. The student's camera only
covers 90° ahead.

To measure it, `Drone-Nav-Obstacles-v0` now logs, over the steps a drone moves faster than 0.5 m/s:
- `Metrics/heading_travel_angle_deg`: the angle between its heading (body +x, where the camera points) and its
  direction of travel;
- `Metrics/travel_in_camera_fov`: the share of those steps the direction of travel is within ±45° of the heading,
  i.e. inside the camera's view.

(Commit `536c7f2`, with a CPU test; the yaw convention is the one the camera check and the lidar code already use.)

The teacher on the 60 held-out level-5 layouts (`p3-teacher-heading-eval-s42`, $0.18; it reproduces 93.8% / 5.65%
exactly):

| | Teacher |
|---|---:|
| Mean angle between heading and travel | **122°** |
| Moving steps with the direction of travel in a forward camera's view | **1.9%** |

A random heading held still would give 90° and 25%. The teacher does worse than that: it mostly flies with its tail
forward. A forward camera on this teacher sees where it's going on about 1 step in 50, so the student is asked
to copy reactions to obstacles it cannot see. That is consistent with everything in §3: it imitates well while the
teacher drives, then stalls at about 50% on easy levels and crashes into walls (whose doors it never sees) most of
all. No student architecture fixes this, and neither does a GRU, since the camera never sees the obstacle in the
first place.

## 5. Next: a teacher that faces where it flies

The fix belongs in the teacher (PLAN.md's teacher is privileged, but it still has to be imitable from the student's
viewpoint). Proposal:

1. **A heading term in the obstacle task's reward:** `−w × (1 − cos(heading_travel_angle))` per second while moving,
   off by default so Phase 2 runs reproduce. Possibly also spawn facing the goal (± some noise) so the policy
   doesn't have to learn a 180° turn first.
2. **Retrain the teacher** with the §17 settings (0.15 m margin, 4 nearest obstacles, mild slowdown) plus the
   heading term. One seed first, about $1. Check `travel_in_camera_fov` (aim for > 90%) and that success and
   collisions stay near 93.8% / 5.65%. The heading term may cost some of either.
3. **Distil again** into the same student and compare with §3's 3.6%.

The alternative, a 360° camera rig for the student, would sidestep the problem but drops PLAN.md's single forward
camera, and real small drones don't carry one.
