# Phase 3: a depth-based student

PLAN.md Phase 3 trains a policy that sees the obstacle field only through a forward depth camera, by distilling the
Phase 2 teacher (Option A, DAgger) and, for comparison, by PPO from depth (Option B). This note records the parts as
they land.

Status:

- [x] `Drone-Nav-Depth-v0`: a ray-cast depth camera, a noise model, and the teacher's observation alongside (§1)
- [x] The camera checked against exact geometry on 1024 drones: 99.97% of pixels agree (§2)
- [ ] CNN student and `scripts/distill.py` (DAgger from the frozen teacher)
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
