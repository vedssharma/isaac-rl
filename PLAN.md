# Project Plan: Learning Drone Navigation Through Clutter with Isaac Lab

**One-line goal:** train a quadrotor policy, entirely in simulation with NVIDIA Isaac Sim / Isaac Lab,
that flies from a start point to a goal through a dense, randomized obstacle field without crashing.
It starts from full state and ends with only onboard depth sensing. Training runs on NVIDIA Brev GPU instances.

This plan is written for someone **new to RL**. Each phase teaches one new concept, and each phase has
a clear "done" test, so you always have a working result to build on.

---

## 1. Goals

| # | Goal | How you know it's done |
|---|------|------------------------|
| G1 | Learn applied RL: PPO, reward design, curricula, domain randomization, teacher→student distillation | You can explain why each reward term exists and show an ablation that proves it |
| G2 | Build a custom Isaac Lab environment, not just run a built-in one | `Drone-Nav-Obstacles-v0` is registered and trains from one command |
| G3 | A policy that navigates clutter from **depth images only** | ≥ 85% success on held-out obstacle maps at medium density |
| G4 | Reproducible, cheap cloud experiments | One command starts a Brev GPU run; checkpoints and metrics outlive the instance |
| G5 | A portfolio-quality write-up | README with videos, learning curves, ablation tables |

**Non-goals for v1:** real hardware deployment, multi-drone swarms, and photorealistic RGB perception.
These are listed as stretch goals in §10.

---

## 2. Tech stack

| Layer | Choice | Why |
|-------|--------|-----|
| Simulator | **Isaac Sim** (via pip) | GPU physics with thousands of parallel environments |
| RL framework | **Isaac Lab**, using a pinned **stable 2.3.x** release | Ships a quadcopter example (`Isaac-Quadcopter-Direct-v0`, Crazyflie), sensors, terrain tools and RL wrappers. Isaac Lab 3.0 (on Isaac Sim 6.x) is a major rewrite with breaking changes and was still in beta in mid-2026. Start on stable, and migrate only once 3.0 is GA. |
| RL library | **RSL-RL** (PPO), with **skrl** as a second option | RSL-RL is fast, simple and the default in Isaac Lab examples. skrl is useful later to compare algorithms (SAC, etc.) |
| Environment style | Isaac Lab **Direct** workflow (`DirectRLEnv`) | Easier to read and debug than the manager-based workflow when learning. Rewards and observations live in one file. |
| Config | Isaac Lab `@configclass` + Hydra CLI overrides | Every experiment is fully described by its config and command line |
| Tracking | **Weights & Biases** (TensorBoard as fallback) | Run boxes are deleted after each run, so metrics have to go off-box live |
| Artifacts | W&B Artifacts or a **Hugging Face Hub** model repo | Checkpoints and videos survive after the instance is deleted |
| Compute | **NVIDIA Brev GPU instances** | See §3 |
| Local dev | Any laptop, no GPU needed | Write code locally and run it on Brev. Use `pytest` on CPU-only logic such as reward math and map generation. |

> **Version pinning rule:** pin exact versions of Isaac Sim, Isaac Lab, RSL-RL and torch in
> `requirements.lock` / the base Docker image, and record the `brev` CLI version you use. Isaac Lab changes APIs between minor releases.

---

## 3. Compute: running on NVIDIA Brev

[Brev](https://brev.nvidia.com) rents GPU **virtual machines** from several cloud providers through one account
and one CLI (`brev`). An instance is a normal Ubuntu VM with NVIDIA drivers, Docker and SSH, and open outbound
internet, so `pypi.nvidia.com`, `nvcr.io`, W&B and HF all work with no workarounds. Billing is prepaid credits,
charged hourly while an instance runs.

### 3.1 Important hardware constraint: pick the GPU by observation type

Isaac Sim's **RTX rendering** (RTX cameras, `TiledCamera`) needs a GPU with **RT cores**.
H100/H200 (and A100) have none, so RTX camera rendering is **not supported** on them. Physics-only
training runs fine on them.

Brev's catalog includes **L40S, L40, RTX 6000 Ada, RTX PRO Server 6000, A6000, L4, A10G, RTX 5090** (all with
RT cores) and **H100, H200, A100, B200** (no RT cores). Availability depends on the provider and region, so check
with `brev search --gpu-name <name>`. So:

| Workload | GPU | Notes |
|----------|-----|-------|
| State-based training (Phases 1–2, teacher policy) | Any of them | No rendering. Run `--headless`. |
| Ray-cast depth (`RayCaster` / `RayCasterCamera`) | Any of them | Warp ray casting runs on CUDA cores and needs no RT cores. **This is the main perception path in this plan** (see §5.4). |
| RTX depth/RGB cameras (`TiledCamera`) | **RT-core cards only** (L40S, RTX PRO 6000, RTX 6000 Ada, …) | Run `--headless --enable_cameras`. |
| Video recording for the write-up | RT-core card | `--video` needs rendering |

**Default choice:** the **L40S** (48 GB, RT cores, runs everything, and is NVIDIA's reference cloud GPU for
Isaac Sim). Use the **RTX PRO Server 6000** when you want more VRAM, an **L4 / A10G** for cheap debugging, and
H100s only for large state-based sweeps, if they cost less per run.

### 3.2 One dev box, disposable run boxes

Brev instances are VMs that can be **stopped**. Stopping releases the GPU (no compute charge, a small storage
charge) and keeps the disk, including installed packages and Docker images. Deleting loses everything. Restarting a
stopped instance needs the same GPU type to be free in that region, so it can fail when capacity is tight. Use this:

1. **A dev box** (`isaac-rl-dev`, one L40S): set up once, used by hand for debugging, quick runs and videos.
   **Stop it whenever you're not using it.** Keep code in git and results on W&B/HF, not only on its disk.
2. **A base Docker image, built once per Isaac Lab version:** Isaac Sim + Isaac Lab + this repo's dependencies,
   built from `scripts/brev/install_isaac.sh` (or `FROM nvcr.io/nvidia/isaac-lab:<tag>`) and pushed to a private
   registry (GHCR), e.g. `ghcr.io/<you>/isaac-rl-base:<isaaclab-version>`. Run a short warm-up inside it so the
   **shader/kit caches** are baked in (the first Isaac Sim start can take several minutes). This is what makes new
   run boxes start fast.
3. **One run = one run box.** A local launcher (`scripts/brev/launch.py`) wraps the `brev` CLI
   (there is no Python SDK):
   - `brev create isaac-rl-<exp>-s<seed>-<id> --gpu-name <gpu> --startup-script @scripts/brev/bootstrap.sh`
     (pulls the base image)
   - passes secrets (`WANDB_API_KEY`, `HF_TOKEN`) as environment variables at run time, never baked into the image
   - checks out this repo **at a specific git SHA**, so every run is reproducible
   - `brev exec` starts `train.py` inside the container under `nohup`, writing logs to a file
   - streams logs back, and polls for completion or a timeout
   - on exit (including errors and Ctrl-C), checks that the final checkpoint was uploaded, then
     **`brev delete`s the instance**
4. **Upload continuously:** checkpoints go to W&B or HF Hub every N iterations (not only at the end), so a
   crash or a lost instance costs minutes, not hours. `train.py --resume <artifact>` continues a run.
5. **Sweeps = fan-out:** `launch.py --sweep configs/sweeps/reward_ablation.yaml` starts one run box
   per (config × seed), up to a concurrency cap.
6. **Cost guards:** prepaid credits are the hard cap (top up only what you plan to spend); a hard `max_runtime`
   per run in the launcher; `scripts/brev/reap.py` that lists instances named `isaac-rl-*` (not the dev box) and
   deletes any older than their max runtime; and a cost line per run in W&B (GPU hours × rate).

> **Check before coding the launcher:** the `brev` CLI is young and changes quickly. Pin and record its version,
> sign in non-interactively with `brev login --api-key "$BREV_API_KEY"` followed by `brev refresh`, and note
> that `brev create` can report success before the instance is ready for `brev exec`, so poll `brev ls` first.

### 3.3 Rough throughput and budget

These are planning figures only. Measure your own numbers in Phase 0.

| Phase | Envs | Approx. wall-clock per run | Runs (incl. seeds/ablations) |
|-------|------|---------------------------|------------------------------|
| 1 Hover/goal | 4096 | 5–15 min | ~10 |
| 2 State-based obstacles | 4096 | 30–90 min | ~30 |
| 3 Depth student | 512–2048 | 1–4 h | ~15 |
| 4 Robustness | 2048–4096 | 1–3 h | ~15 |

That totals roughly **60–120 GPU-hours** for the whole project. Multiply by your Brev rate (L40S is usually
the cheapest RT-core option) to get a budget. Buy only that much credit up front.

---

## 4. Problem formulation (the MDP)

### 4.1 Robot and action space
- **Robot:** Crazyflie 2.x model shipped with Isaac Lab (`CRAZYFLIE_CFG`). It is small and well-modeled,
  and it is a real, cheap platform if you ever want sim-to-real.
- **Action (v1):** collective thrust + body-frame torques (this matches the built-in quadcopter env, so it's a known-good start).
- **Action (v2, recommended for navigation):** **collective thrust + body rates (CTBR)**, tracked by a simple
  low-level PD rate controller in the env. CTBR is the standard interface in learning-based agile flight.
  It is more robust to model mismatch and easier to learn than raw torques.
- Actions are normalized to [-1, 1], then scaled. Add a **first-order motor lag** model in Phase 4.

### 4.2 Observations (they grow by phase)
| Group | Contents | Phase |
|-------|----------|-------|
| Proprioception | linear velocity (body), angular velocity, gravity vector in body frame (or rotation matrix first two columns), last action | 1+ |
| Goal | goal position relative to drone, in body frame (clipped distance + unit direction) | 1+ |
| Privileged obstacles | 1-D ring of **ray-cast distances** (e.g. 36 horizontal × 3 elevation rays, max range 5 m), *or* K nearest obstacle positions/radii | 2 (teacher) |
| Depth | 64×48 (or 80×60) depth image from a forward-facing camera, clipped to 0–6 m, normalized, with noise | 3+ (student) |

### 4.3 Reward (start simple, add terms one at a time)
```
r = + w_prog   * (d_goal[t-1] - d_goal[t])          # progress toward goal (dense, main driver)
    + w_goal   * 1[reached goal]                    # sparse success bonus
    - w_col    * 1[collision]                       # crash penalty (episode ends)
    - w_prox   * Σ exp(-d_obs_i / σ)                # soft obstacle clearance shaping (Phase 2+)
    - w_smooth * ||a_t - a_{t-1}||²                 # action smoothness
    - w_rate   * ||ω||²                             # discourage spinning
    - w_up     * (1 - up_z)                         # stay upright-ish
    - w_time                                        # small per-step cost → fly efficiently
```
Log **every term separately** to W&B. This is the most useful debugging tool in RL.

### 4.4 Termination and reset
- **Terminate:** collision (contact sensor force > threshold, or ray distance < drone radius), flip
  (`up_z < 0`), leaving bounds, altitude out of [0.2 m, ceiling], goal reached.
- **Truncate:** time limit (e.g. 20 s). Handle truncation as a time-out, not a death, so PPO bootstraps correctly.
- **Reset:** new random start/goal pair on opposite sides of the map, at least *L* meters apart, with a collision-free
  spawn check. Randomize the initial yaw and a small initial velocity.

### 4.5 Algorithm
- **PPO** (RSL-RL), actor-critic MLP `[256, 256, 128]`, ELU, ~24 steps per env per iteration,
  4096 envs, adaptive KL learning rate, observation normalization.
- **Asymmetric actor-critic** from Phase 3 on: the critic gets privileged state, the actor gets only realistic observations.
- Phase 3 student: a small **CNN depth encoder → latent (64-d)**, concatenated with proprioception and goal, then an MLP
  (optionally a GRU to remember obstacles that left the field of view).

---

## 5. Environment design

### 5.1 Obstacle world
Build obstacles **procedurally** so every environment instance and every reset can look different:

- **Primitives:** vertical cylinders (trees/pillars), boxes (buildings/crates), thin poles, horizontal
  bars, wall segments with gaps (window-like openings), and later floating spheres.
- **Parameters:** density (obstacles / 100 m²), size distribution, min gap width (≥ 2× drone span at the start),
  arena size (e.g. 30 × 10 × 4 m corridor), clutter height range.
- **Difficulty levels 0–9** map to (density, min-gap, primitive mix). These drive the curriculum.
- **Held-out test maps:** a fixed set of seeds that are **never used in training**, used for all evaluation.

### 5.2 How to put obstacles into Isaac Lab (two options; try A first)
- **A. Bake obstacles into the terrain mesh** (recommended). Write a custom `SubTerrainBaseCfg` generator that
  outputs a ground plane plus obstacle meshes (trimesh) in a grid of tiles. Each tile is one difficulty
  level and variant. Envs are assigned to tiles, and the curriculum moves each env to harder tiles. Why this works well:
  - Isaac Lab's `RayCaster` / `RayCasterCamera` ray-casts against **static meshes** very fast with Warp,
    which gives you cheap depth on any GPU (including H100).
  - Collisions come for free from the terrain collider.
  - This is the same pattern Isaac Lab uses for legged-robot rough-terrain curricula, so there are good examples to copy.
- **B. Spawn rigid-body obstacle prims per env** (`RigidObjectCollection` / multi-asset spawners). Needed later for
  **moving** obstacles (Phase 4 stretch). It is more flexible but heavier. Ray casting against moving meshes
  needs the multi-mesh ray caster available in your Isaac Lab version, or RTX cameras.

### 5.3 Curriculum
- Per-env difficulty level: **move up** if the env reached the goal on ≥ 80% of its last N episodes, **move down** if < 40%.
- Also apply a curriculum to the goal distance and the size of the clearance-penalty term.
- Log the difficulty histogram over time. It is your best progress chart.

### 5.4 Perception path
1. **Phase 2 (teacher):** a low-dimensional ray-cast "lidar ring". It is cheap, privileged and noise-free.
2. **Phase 3 (student):** a `RayCasterCamera` that outputs a **depth image** (pinhole model, ~90° FOV). It needs no RT cores
   and runs fast for thousands of envs. Add realistic depth noise: Gaussian + range-dependent noise, random pixel dropout,
   edge artifacts, and a max-range cutoff.
3. **Optional Phase 3b:** switch to `TiledCamera` (RTX depth) on an RTX GPU and compare. It is a more realistic renderer,
   but much slower. It's a good ablation for your write-up.

### 5.5 Domain randomization (Phase 4)
Randomize mass (±20%), inertia, thrust-to-weight / motor constant (±15%), motor time constant, drag,
**wind gusts** (an external force process), control latency (0–40 ms), observation noise, depth noise level,
and camera pitch (±5°).

---

## 6. Phased roadmap

Each phase ends with a **demo video + W&B report**. Don't move on until the exit criteria pass.

### Phase 0 — Setup and RL basics (week 1)
- [ ] Read: *Spinning Up* (intro + PPO), and Sutton & Barto ch. 1–3 & 13 (skim)
- [ ] Create a Brev account, add a limited amount of credit, create an API key
- [ ] Set up the dev box and build the base image (§3.2). Record start-up time, steps/s and VRAM for 1k/4k/8k envs
- [ ] Run the built-in `Isaac-Quadcopter-Direct-v0` with RSL-RL headless on Brev; get W&B logging working
- [ ] Write `launch.py` v0: create instance → run → upload → delete
- **Exit:** the built-in quadcopter task trains to convergence from one local command, and the checkpoint is on W&B/HF.

### Phase 1 — Own environment, goal reaching, no obstacles (week 2)
- [ ] Scaffold `drone_nav/` as an Isaac Lab extension; register `Drone-Nav-Goal-v0` (Direct workflow)
- [ ] Implement observations, the reward from §4.3 (no obstacle terms yet), terminations and resets
- [ ] `play.py` that loads a checkpoint, records a video and prints metrics
- [ ] Switch action space to CTBR + rate controller; compare against thrust+torques (your first ablation)
- **Exit:** ≥ 98% of episodes reach random goals 10–20 m away; smooth flight in the video.

### Phase 2 — Obstacles with privileged sensing (weeks 3–5)
- [ ] Procedural obstacle terrain generator (§5.1–5.2A) + unit tests (gaps ≥ min width, spawn/goal free)
- [ ] Ray-cast lidar-ring observation; collision detection; proximity reward
- [ ] Curriculum (§5.3); evaluation script on held-out maps
- [ ] Ablations: (a) no proximity shaping, (b) no curriculum, (c) number of rays, (d) 3 seeds each
- **Exit (teacher):** ≥ 90% success, < 5% collisions at difficulty 5 on held-out maps.

### Phase 3 — Depth-based student (weeks 6–8)
- [ ] `RayCasterCamera` depth + noise model; CNN encoder
- [ ] **Option A — Teacher→student distillation (DAgger):** roll out the student, label actions with the frozen
      teacher, and train with supervised loss. This is usually fastest and most stable.
- [ ] **Option B — End-to-end PPO from depth** with an asymmetric critic. Compare against A; it's a good write-up result.
- [ ] Add a GRU memory; test on maps with obstacles behind corners / outside the FOV
- **Exit (G3):** ≥ 85% success at difficulty 5 from depth only.

### Phase 4 — Robustness and difficulty (weeks 9–10)
- [ ] Domain randomization (§5.5), wind, latency, motor lag
- [ ] Out-of-distribution evaluation: higher density, new primitive types, more noise
- [ ] Optional: moving obstacles (spawn option B)
- **Exit:** < 10% drop in success under full randomization compared with nominal.

### Phase 5 — Evaluation and write-up (weeks 11–12)
- [ ] Final benchmark table across policies × difficulty levels × seeds (mean ± 95% CI)
- [ ] Videos: chase cam + depth-view picture-in-picture, rendered on an RTX GPU
- [ ] README write-up, W&B report, and model card on HF Hub

---

## 7. Experiment management

- **Every run is identified by:** git SHA + config hash + seed + Isaac Lab version. All of these are logged to W&B config.
- **Naming:** `p{phase}-{short-desc}-s{seed}`, e.g. `p2-no-prox-shaping-s1`.
- **Seeds:** at least 3 for any claim in the write-up.
- **Metrics (logged in training and in eval):**
  - success rate, collision rate, timeout rate
  - time-to-goal, path length / straight-line distance (path efficiency)
  - min obstacle clearance, mean speed, action jerk
  - curriculum level histogram, per-reward-term means
  - steps/s, GPU utilization, run cost
- **Evaluation protocol:** a fixed set of 1,000 held-out (map, start, goal) triples per difficulty level, run with a
  deterministic policy. Use the same protocol for every model so numbers are comparable.

---

## 8. Proposed repository layout

```
isaac-rl/
├── PLAN.md
├── README.md
├── pyproject.toml / requirements.lock
├── source/drone_nav/                 # Isaac Lab extension (pip install -e)
│   └── drone_nav/
│       ├── tasks/
│       │   ├── goal/                 # Phase 1 env + cfg
│       │   └── obstacles/            # Phase 2–4 env + cfg
│       ├── terrain/obstacle_field.py # procedural obstacle sub-terrain generator
│       ├── control/rate_controller.py
│       ├── sensors/depth_noise.py
│       ├── rewards.py
│       └── agents/rsl_rl_ppo_cfg.py
├── scripts/
│   ├── train.py  play.py  eval.py  distill.py
│   └── brev/
│       ├── install_isaac.sh          # Isaac Sim + Isaac Lab install (dev box + base image)
│       ├── Dockerfile                # base image, pushed to GHCR
│       ├── bootstrap.sh              # run-box startup script: pull base image
│       ├── launch.py                 # single run + sweeps (wraps the brev CLI)
│       └── reap.py                   # delete orphaned run boxes
├── configs/
│   ├── exp/                          # one YAML per named experiment
│   └── sweeps/
├── tests/                            # CPU-only: map gen, reward math, curriculum logic
└── docs/                             # results, figures, write-up
```

---

## 9. Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Isaac Sim install/start-up problems on the instance (EGL, Vulkan, drivers) | Fix them once in the base image; add a `smoke_test.py` (10 iterations) that runs at the start of every run box |
| RTX cameras unavailable on H100 | Use ray-cast depth as the default; use RTX cards only for RTX-camera ablations and videos |
| Reward hacking (e.g. hovering forever, flying around the map edges) | Per-term logging, a time penalty, bounds termination, and **watch videos every phase** |
| Policy never finds the goal in dense clutter | Curriculum from empty to dense; dense progress reward; privileged teacher first |
| Lost work when an instance dies or can't restart (no capacity) | Continuous checkpoint upload; `--resume`; launcher verifies uploads before deleting; code always in git |
| Runaway cost (a forgotten dev box or run box) | Prepaid credits only, stop the dev box after each session, max runtime per run, reaper script |
| API drift (Isaac Lab 2.x → 3.0, `brev` CLI) | Pin versions; isolate Isaac Lab–specific code; migrate in a separate branch |
| Sim-to-sim gap (ray-cast vs RTX depth) | Phase 3b ablation; noise model on depth |

---

## 10. Stretch goals

- **Sim-to-real:** deploy the CTBR policy to a real Crazyflie (with a Flow deck / an external depth sensor)
- **Drone racing:** a gate-passing task with lap-time reward
- **Moving obstacles** and other agents; **multi-drone** coordination (MAPPO)
- **Newton physics backend / Isaac Lab 3.0** migration and a speed comparison
- **Algorithm comparison:** PPO vs SAC vs DreamerV3-style world model on the same task

---

## 11. Reading list

- OpenAI *Spinning Up in Deep RL*; Schulman et al., *Proximal Policy Optimization* (2017)
- Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL* (2021) — the parallel-env + curriculum recipe this plan follows
- Loquercio et al., *Learning High-Speed Flight in the Wild* (2021) — depth-based navigation via privileged-expert imitation
- Kaufmann et al., *Champion-level drone racing using deep RL* (Nature, 2023)
- Kulkarni et al., *Aerial Gym* and Xu et al., *OmniDrones* — open-source GPU drone RL simulators (good reference code)
- Isaac Lab docs: Direct workflow tutorial, `RayCaster` / camera sensors, terrain generator, RSL-RL integration

---

## 12. First week: concrete next steps

1. Create Brev + W&B (+ HF) accounts; put the API keys in a local `.env` (git-ignored). Add a small amount of
   Brev credit to start.
2. Create the dev box (`brev create isaac-rl-dev --gpu-name L40S`), run `scripts/brev/install_isaac.sh` on it,
   and run `Isaac-Quadcopter-Direct-v0` headless. Write down every command that worked. Stop the box after.
3. Turn those commands into `scripts/brev/Dockerfile`, build the base image and push it to GHCR.
4. Write `launch.py` v0 and reproduce step 2 on a fresh run box from one local command.
5. Scaffold `source/drone_nav` and start Phase 1.
