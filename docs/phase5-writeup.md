# Phase 5: benchmark and write-up

PLAN.md Phase 5: a final benchmark table across policies × difficulty levels × seeds (mean ± 95% CI), write-up
videos (chase cam + depth view picture-in-picture), and the README, a W&B report and a model card.

Status:

- [x] Benchmark: 4 policies × 3 seeds × 6 conditions, 72 evaluations (§1, [benchmark.md](benchmark.md))
- [x] Video tooling: a chase camera that follows the heading, and the depth view as a picture-in-picture (§2)
- [x] Write-up videos of the final policy (§2)
- [x] README ([../README.md](../README.md))
- [x] Model card on Hugging Face and a W&B report (§3)

## 1. The benchmark

**Policies**, three seeds each (the W&B runs are listed in [benchmark.md](benchmark.md)):

| Policy | What it is | From |
|---|---|---|
| Teacher (privileged) | lidar ring + 4 nearest obstacles; heading and soft-ceiling terms | Phase 3 §9 |
| Depth student, feed-forward | CNN + MLP, distilled from the heading-only teacher | Phase 3 §6.1 |
| Depth student, GRU | CNN + GRU, distilled from the teacher above | Phase 3 §10.1 (the Phase 3 result) |
| Depth student, GRU, randomized | the same, teacher and student trained under domain randomization | Phase 4 §5.1 (the Phase 4 result) |

**Conditions:**
- held-out levels 1, 3, 5, 7 and 9: test seed 1001, 60 layouts per level that training never saw;
- level 5 under full randomization (Phase 4 §2's levels; the depth policies also get depth noise × 2).

1000 episodes per cell and seed, which is PLAN.md §7's protocol. One box per checkpoint ran its six conditions
(`p5-bench-*`, at most four at a time); `scripts/benchmark_table.py` collects the results from Hugging Face and
computes the mean over seeds ± a 95% CI from Student's t (t = 4.30 for 3 seeds).

**Success (%)**:

| Policy | L1 | L3 | L5 | L7 | L9 | L5 randomized |
|---|---:|---:|---:|---:|---:|---:|
| Teacher (privileged) | 98.6 ± 2.1 | 97.1 ± 1.0 | 93.4 ± 2.5 | 81.4 ± 5.0 | 58.1 ± 15.5 | 50.8 ± 8.6 |
| Depth student, feed-forward | 95.5 ± 15.0 | 97.6 ± 1.2 | 92.9 ± 6.3 | 88.2 ± 8.1 | 69.5 ± 31.1 | 41.7 ± 28.9 |
| Depth student, GRU | 99.7 ± 0.5 | 98.4 ± 0.7 | 94.5 ± 0.7 | 87.8 ± 4.4 | 70.4 ± 9.7 | 54.2 ± 18.9 |
| Depth student, GRU, randomized | 99.0 ± 2.5 | 97.8 ± 2.4 | 92.9 ± 3.5 | 84.5 ± 4.7 | 61.2 ± 25.5 | 81.9 ± 16.9 |

Collision rates are in [benchmark.md](benchmark.md).

1. **The depth students match the teacher at levels 1–5 and beat it at 7–9.** At level 7 the teacher manages 81.4%
   and the students 84.5–88.2%; at level 9, 58.1% against 61.2–70.4%. This is despite the students seeing 90° ahead
   where the teacher sees 360° around.
2. **The GRU student is the most consistent:** ± 0.7 at level 5 against ± 6.3 for the feed-forward student.
3. **Randomization training is what makes a policy robust:** 81.9% at level 5 fully randomized, against 41.7–54.2% for
   every policy trained without it (the teacher included). It costs 1.6 points at nominal level 5 (92.9% against
   94.5%).
4. **Wide intervals are seed-to-seed differences, and they're real.** The feed-forward student's L1 interval comes from
   one seed (s42) climbing out of the arena in 11.2% of level-1 episodes, a habit copied from its teacher (Phase 3 §8).
   Level 9 varies a lot by seed for every policy.
5. **The level-5 numbers agree with the phase notes** within their noise. These use 1000 episodes; the phase notes
   used 2000. For example, the randomized GRU students get 81.9% fully randomized here against 83.6% in Phase 4 §5.1.
6. **Not solved:** level 9 (0.6 m gaps, 12 obstacles per 100 m²) for any policy, and thin poles, which are the main
   cause of collisions throughout.

Cost: about $6 for the 12 benchmark boxes.

## 2. Write-up videos

Phase 3 §7's first video showed two problems. The chase camera sat at a fixed world offset and ended up inside
obstacles, and it showed nothing of what the policy sees. Both are fixed (commit `46af0fb`):

- **`play.py --chase_heading`** keeps Isaac Lab's viewport camera behind and above env 0's drone along its smoothed
  heading. It updates the camera controller's eye and look-at offsets every step. The space just behind the drone is
  space it has just flown through, so it's free.
- **Depth frames:** with `--video` on the depth task, `play.py` saves env 0's depth observation after every recorded
  step (`videos/<name>/depth_env0.npz`, uploaded with the video).
- **`scripts/compose_video.py`** (local, ffmpeg) overlays the depth image, near = bright, as a framed
  picture-in-picture in the top-right corner, aligned frame by frame. A CPU test composes a synthetic clip and checks
  the inset.

Getting the recording made took four attempts:
1. A box that Brev listed RUNNING never accepted ssh. I deleted it.
2. Brev had no capacity for either rendering-capable instance type.
3. `git fetch` hung while the laptop slept.
4. The launch whose network calls came back to life created a box just as I stopped it, so that box was orphaned and
   I deleted it.

The fifth launch worked (Nebius timed out in DEPLOYING and the launcher fell back to Crusoe). It recorded the Phase 4
policy, the seed-42 randomized GRU student (`0g9bvtcl`), with `--chase_heading`, 1500 frames at 50 fps (30 s) per clip:

| Clip | Condition | Contact sheet |
|---|---|---|
| `p5-final-l5-depth-pip.mp4` | held-out level 5, nominal | [p5-final-l5.png](images/p5-final-l5.png) |
| `p5-final-l5-randomized-depth-pip.mp4` | held-out level 5, full randomization, depth noise × 2 | [p5-final-l5-randomized.png](images/p5-final-l5-randomized.png) |

The randomized recording run's evaluation scored 88.0% over its 100 episodes (95% CI 80.2–93.0), in line with the
benchmark's 81.9 ± 16.9. In both clips the camera stays behind the drone and never ends up inside an obstacle, and the
inset shows what the policy reacts to: near obstacles bright, the far field dark, and the dropout specks of the noise
model (denser in the randomized clip). The contact sheets in `docs/images/` show one frame every 2 s.

The composed MP4s (6–7 MB each) are not in git (`*.mp4` is ignored). The raw videos and depth frames are on Hugging
Face next to the student's checkpoint (`runs/drone_nav_depth_distill/2026-11-23_20-42-56_p4-distill-gru-dr-s42/play/`,
W&B run `86lv5vaj`); `scripts/compose_video.py` rebuilds the composed ones from them.

Cost: about $1.02 for the recording box.

## 3. Model card and W&B report

- **Model card:** [https://huggingface.co/vedssharma/isaac-rl-drone-nav](https://huggingface.co/vedssharma/isaac-rl-drone-nav), from [model-card.md](model-card.md) (uploaded as the repo's `README.md`). It says
  which checkpoint to use (the randomized GRU student), the input and output layout, a loading snippet (checked on
  CPU: `drone_nav` installs without Isaac Sim), the benchmark, the limitations and every benchmark checkpoint. The
  contact sheets and the two composed videos are in the repo's `media/`. The repo has no license yet, and the card
  says so.
- **W&B report:** [the report](https://forge.coreweave.com/vedssharma1/isaac-rl/reports/isaac-rl:-a-quadrotor-that-flies-through-clutter-from-depth-alone--VmlldzoxODM2NDEwMQ==), built by `scripts/wandb_report.py` from the same run list as
  `benchmark_table.py`. It has the summary, the contact sheet, the benchmark table with per-run bar charts for each
  condition, the teacher's and the students' training curves, and the lessons. Each run set is filtered by run ID,
  checked to select exactly its policy's three seeds (one earlier failed `p4-distill-gru-dr-s42` run shares a name).
  The report is visible to whoever can see the W&B project; it is not public.
