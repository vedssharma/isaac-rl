---
library_name: pytorch
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - robotics
  - quadrotor
  - drone
  - isaac-lab
  - isaac-sim
  - depth
  - imitation-learning
  - dagger
  - ppo
  - sim-only
model-index:
  - name: isaac-rl depth student (GRU, randomized)
    results:
      - task:
          type: reinforcement-learning
          name: Quadrotor navigation through clutter from depth
        dataset:
          type: drone-nav-obstacles-heldout
          name: Drone-Nav-Depth-v0 held-out maps, level 5 (test seed 1001)
        metrics:
          - type: success_rate
            name: Success rate (%), nominal, mean of 3 seeds
            value: 92.9
          - type: success_rate
            name: Success rate (%), full domain randomization, mean of 3 seeds
            value: 81.9
  - name: isaac-rl depth student (GRU)
    results:
      - task:
          type: reinforcement-learning
          name: Quadrotor navigation through clutter from depth
        dataset:
          type: drone-nav-obstacles-heldout
          name: Drone-Nav-Depth-v0 held-out maps, level 5 (test seed 1001)
        metrics:
          - type: success_rate
            name: Success rate (%), nominal, mean of 3 seeds
            value: 94.5
          - type: success_rate
            name: Success rate (%), full domain randomization, mean of 3 seeds
            value: 54.2
---

# isaac-rl: a quadrotor that flies through clutter from depth alone

Policies for a Crazyflie-sized quadrotor that flies to a goal 20–24 m away through a randomized field of
cylinders, boxes, thin poles and walls with doors. They see only a 64 × 48 depth camera and their own motion
sensors. Everything was trained and evaluated in simulation (NVIDIA Isaac Sim / Isaac Lab 2.3.2); nothing here has
flown a real drone.

Code, training details and the phase-by-phase notes: **[github.com/vedssharma/isaac-rl](https://github.com/vedssharma/isaac-rl)**.

![The randomized GRU student on a held-out level-5 map, one frame every 2 s; top right: the depth image it flies from](media/p5-final-l5.png)

*The recommended checkpoint on a held-out level-5 map, one frame every 2 s. Top right: the depth image it sees
(shown with near = bright). Videos: [nominal](media/p5-final-l5-depth-pip.mp4),
[full randomization](media/p5-final-l5-randomized-depth-pip.mp4).*

## Which checkpoint to use

| Policy | Use it for | Checkpoint (seed 42) |
|---|---|---|
| **Depth student, GRU, randomized** (recommended) | robustness to mass, thrust, motor lag, latency, wind and sensor noise | `runs/drone_nav_depth_distill/2026-11-23_20-42-56_p4-distill-gru-dr-s42/model_499.pt` |
| Depth student, GRU | the best nominal (unrandomized) numbers | `runs/drone_nav_depth_distill/2026-11-23_16-56-07_p3-distill-gru-alt-s42/model_499.pt` |
| Teacher (privileged) | not deployable: it reads a 360° lidar ring and the true positions of the 4 nearest obstacles | `runs/drone_nav_obstacles/2026-11-22_23-40-34_p3-teacher-heading-alt-s42/model_1499.pt` |

Every checkpoint for all three seeds is listed under [Checkpoints](#checkpoints). Each run folder also holds the
training parameters (`params/`) and the evaluation results (`play/*.json`).

## Results

Success rate (%) on held-out maps (60 layouts per level never seen in training, 1000 episodes per cell and seed),
mean over 3 seeds ± 95% CI (Student's t). Difficulty runs from level 0 to 9; level 9 has 0.6 m gaps and 12
obstacles per 100 m². "L5 randomized": level 5 with every training randomization on at once, and depth noise × 2 for the depth
policies.

| Policy | L1 | L3 | L5 | L7 | L9 | L5 randomized |
|---|---:|---:|---:|---:|---:|---:|
| Teacher (privileged) | 98.6 ± 2.1 | 97.1 ± 1.0 | 93.4 ± 2.5 | 81.4 ± 5.0 | 58.1 ± 15.5 | 50.8 ± 8.6 |
| Depth student, feed-forward | 95.5 ± 15.0 | 97.6 ± 1.2 | 92.9 ± 6.3 | 88.2 ± 8.1 | 69.5 ± 31.1 | 41.7 ± 28.9 |
| Depth student, GRU | 99.7 ± 0.5 | 98.4 ± 0.7 | 94.5 ± 0.7 | 87.8 ± 4.4 | 70.4 ± 9.7 | 54.2 ± 18.9 |
| Depth student, GRU, randomized | 99.0 ± 2.5 | 97.8 ± 2.4 | 92.9 ± 3.5 | 84.5 ± 4.7 | 61.2 ± 25.5 | 81.9 ± 16.9 |

Collision rates and the W&B runs: [benchmark.md](https://github.com/vedssharma/isaac-rl/blob/main/docs/benchmark.md).
With 3 seeds the intervals mostly show seed-to-seed differences.

## How they were trained

1. **Teacher:** PPO (RSL-RL), 4096 parallel environments, a 10-level obstacle curriculum. It sees a 360° lidar ring
   and the 4 nearest obstacles. Reward terms for facing the direction of travel and for staying below 3 m make its
   behaviour imitable from a forward camera.
2. **Students:** distilled from the frozen teacher with DAgger (the student flies; the teacher labels its actions).
   A CNN encodes the depth image to 64 numbers; with the 18 motion inputs it feeds a GRU (256) and an MLP
   (256-256-128). The GRU is trained with truncated backpropagation through time over 48-step windows. About 0.8 M
   parameters.
3. **Randomized student:** the teacher is fine-tuned under domain randomization and the student is re-distilled
   under the same randomization: mass and inertia ± 20%, thrust ± 15%, motor lag up to 50 ms, drag ± 30%, 0–40 ms
   latency, a steady wind up to 2 m/s with gusts, observation noise, and depth noise × 2. Camera pitch was not
   randomized in training; a ± 5° pitch on top of everything else costs about one more point
   ([Phase 4 §7](https://github.com/vedssharma/isaac-rl/blob/main/docs/phase4-robustness.md)).

Total compute: about 130 launches on single rented L40S GPUs, roughly $140.

## Interface

- **Control:** 50 Hz. Physics at 100 Hz.
- **Input:** one vector of 18 + 3072 numbers.
  - Body-frame linear velocity (3), angular velocity (3) and gravity direction (3).
  - The previous action (4).
  - The goal as a body-frame unit direction (3) and a distance clipped and scaled to [0, 1] (1).
  - Altitude in m (1).
  - The depth image, 48 × 64, row-major, metres / 6 clipped to [0, 1]: 1 means nothing within 6 m. The camera has a
    90° horizontal field of view and faces forward.
- **Output:** 4 actions in [-1, 1]: collective thrust (−1 = 0, +1 = 1.9 × weight) and roll, pitch and yaw rates
  (± 6, 6, 3 rad/s), tracked by a rate controller.

```python
# pip install torch huggingface_hub
# git clone https://github.com/vedssharma/isaac-rl && pip install -e isaac-rl/source/drone_nav   (no Isaac Sim needed)
import torch
from huggingface_hub import hf_hub_download
from drone_nav.policies.depth_student import load_student

path = hf_hub_download(
    "vedssharma/isaac-rl-drone-nav",
    "runs/drone_nav_depth_distill/2026-11-23_20-42-56_p4-distill-gru-dr-s42/model_499.pt",
)
student = load_student(torch.load(path, map_location="cpu", weights_only=False))  # a trusted file: it unpickles

obs = torch.zeros(1, 18 + 48 * 64)  # fill in as above
student.reset()                     # clears the GRU memory; reset(dones) clears it per environment
with torch.no_grad():
    action = student.act(obs)       # (1, 4), one call per 20 ms step
```

To evaluate in simulation, use `scripts/play.py` in the GitHub repo (an RTX GPU with Isaac Lab 2.3).

## Limitations

- **Simulation only.** The depth camera is a ray-caster with a stereo-like noise model, not a real sensor, and the
  vehicle is a rigid body with first-order motors. No sim-to-real transfer has been tested.
- **Dense clutter is not solved:** 61–70% at level 9.
- **Thin poles** (3–6 cm) cause most collisions. They are 1–3 pixels wide in the depth image a second before
  impact.
- **Out-of-distribution layouts** cost the most: pole-heavy fields and wall-heavy layouts.
  ([Phase 4 §7](https://github.com/vedssharma/isaac-rl/blob/main/docs/phase4-robustness.md))
- **Wind** is the weakest single randomization for the randomized student.
- Static obstacles only; flat ground; the goal position is given.
- Not for controlling a real aircraft.

## Checkpoints

All under `runs/` in this repository; W&B run ids are on the W&B server the project used.

| Policy | Seed | Folder | Checkpoint |
|---|---|---|---|
| Teacher (privileged) | 42 | `drone_nav_obstacles/2026-11-22_23-40-34_p3-teacher-heading-alt-s42` | `model_1499.pt` |
| Teacher (privileged) | 1 | `drone_nav_obstacles/2026-11-23_00-46-32_p3-teacher-heading-alt-s1` | `model_1499.pt` |
| Teacher (privileged) | 2 | `drone_nav_obstacles/2026-11-23_00-47-06_p3-teacher-heading-alt-s2` | `model_1499.pt` |
| Depth student, feed-forward | 42 | `drone_nav_depth_distill/2026-11-20_18-47-18_p3-distill-heading-s42` | `model_999.pt` |
| Depth student, feed-forward | 1 | `drone_nav_depth_distill/2026-11-20_20-07-32_p3-distill-heading-s1` | `model_999.pt` |
| Depth student, feed-forward | 2 | `drone_nav_depth_distill/2026-11-20_20-24-52_p3-distill-heading-s2` | `model_999.pt` |
| Depth student, GRU | 42 | `drone_nav_depth_distill/2026-11-23_16-56-07_p3-distill-gru-alt-s42` | `model_499.pt` |
| Depth student, GRU | 1 | `drone_nav_depth_distill/2026-11-23_17-52-14_p3-distill-gru-alt-s1` | `model_499.pt` |
| Depth student, GRU | 2 | `drone_nav_depth_distill/2026-11-23_17-59-22_p3-distill-gru-alt-s2` | `model_499.pt` |
| Depth student, GRU, randomized | 42 | `drone_nav_depth_distill/2026-11-23_20-42-56_p4-distill-gru-dr-s42` | `model_499.pt` |
| Depth student, GRU, randomized | 1 | `drone_nav_depth_distill/2026-11-24_00-01-06_p4-distill-gru-dr-s1` | `model_499.pt` |
| Depth student, GRU, randomized | 2 | `drone_nav_depth_distill/2026-11-24_00-01-35_p4-distill-gru-dr-s2` | `model_499.pt` |

The other folders under `runs/` are earlier experiments from the phase notes.

The teacher checkpoints are RSL-RL PPO checkpoints (`model_state_dict` holds actor and critic). The students are this
project's format (`kind: depth_student`), loaded with `load_student` as above.

No license has been chosen yet.
