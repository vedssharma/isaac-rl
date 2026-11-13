# Setup: accounts, keys and the dev box

These are steps 1 and 2 of [PLAN.md §12](../PLAN.md#12-first-week-concrete-next-steps). Compute runs on
[NVIDIA Brev](https://brev.nvidia.com) GPU instances (see PLAN.md §3).

## 1. Create the accounts

| Service | What it's for | Sign up | Get the key |
|---------|---------------|---------|-------------|
| **NVIDIA Brev** | GPU instances (dev box + training runs) | https://brev.nvidia.com | Your org → **API keys** → create a key with **Read & Write** access |
| **Weights & Biases** | Live metrics and checkpoint artifacts | https://wandb.ai/site | https://wandb.ai/authorize |
| **Hugging Face** | Model repo for checkpoints, videos and the model card | https://huggingface.co/join | https://huggingface.co/settings/tokens (fine-grained, write access to your model repo) |

## 2. Limit spending before anything else

Brev bills prepaid credits by the hour while an instance runs. The whole project is budgeted at roughly
60–120 GPU-hours (PLAN.md §3.3).

- **Brev:** add only the credit you plan to spend for now. That's your hard cap. Note the hourly rate for the
  L40S (default) and the H100 so we can log cost per run later.
- **Stopped instances** don't pay for the GPU (only a little storage), so `brev stop` the dev box after every
  session. Deleted instances cost nothing.
- **W&B and HF:** the free tiers are enough for this project.

## 3. Install the Brev CLI and sign in

```bash
# macOS
brew install brevdev/homebrew-brev/brev
# Linux / WSL
bash -c "$(curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh)"

brev login --api-key "<your BREV_API_KEY>"   # or plain `brev login` to sign in through the browser
brev refresh                                  # writes the SSH config that brev shell/exec/copy use
```

## 4. Create the HF model repo

```bash
hf auth login            # or: export HF_TOKEN=...
hf repos create <your-hf-username>/isaac-rl-drone-nav --private
```

## 5. Fill in `.env` and check it

```bash
cp .env.example .env     # .env is git-ignored
$EDITOR .env             # paste the keys

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
python scripts/check_env.py
```

`check_env.py` checks that the `brev` CLI is installed and signed in, makes one read-only call each to W&B
and HF, and prints `All good.` when everything works. It never prints a full secret.

## Rules for secrets

- Keys live only in your local `.env`. Pass `WANDB_API_KEY` / `HF_TOKEN` to an instance as environment
  variables when you run something; never bake them into a Docker image, and never commit `.env`.
- Your Brev API key stays on your laptop. Instances never need it.
- If a key leaks, revoke it in that service's dashboard and make a new one.

## 6. Step 2: the dev box

Create one L40S instance, install Isaac Sim + Isaac Lab on it, and run the built-in quadcopter task:

```bash
brev search --gpu-name L40S                  # check availability and price first
brev create isaac-rl-dev --gpu-name L40S
brev shell isaac-rl-dev                      # opens a shell on the instance
```

Then, on the instance:

```bash
cd ~/workspace
git clone https://github.com/vedssharma/isaac-rl.git && cd isaac-rl
bash scripts/brev/install_isaac.sh           # prints the smoke-test command at the end
```

Write down anything you had to change. Those notes become the base-image `Dockerfile` in step 3.

### Training runs

Start every run with this repo's [`scripts/train.py`](../scripts/train.py), not Isaac Lab's copy. It takes the
same arguments, plus `--hf_upload none` to skip the Hugging Face upload. It also:

- finishes the W&B run before Isaac Sim exits. Isaac Lab's script doesn't, so its runs stay "running" and lose
  the final checkpoint.
- records the git SHA, package versions and GPU in `params/run_meta.yaml` and in the W&B config.
- uploads the final checkpoint, `params/`, git diffs and any `--video` clips to `HF_REPO_ID` under
  `runs/<experiment>/<run>/`, and logs the clips to W&B.
- writes logs to `logs/rsl_rl/<experiment>/<timestamp>_<run_name>/` in this repo.

On the instance, export the same variables as your `.env` (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`,
`HF_TOKEN`, `HF_REPO_ID`), then:

```bash
source ~/isaac-venv/bin/activate
python scripts/train.py --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 4096 \
    --seed 42 --run_name p0-quadcopter-baseline-s42 --logger wandb
```

Name runs `p{phase}-{short-desc}-s{seed}` (PLAN.md §7).

**When you're done, stop it** from your laptop: `brev stop isaac-rl-dev`. The disk (including the install) is
kept, and `brev start isaac-rl-dev` brings it back. A restart can fail if no L40S is free in that region at the
moment; your data waits until one is. Keep code in git and results on W&B/HF so nothing important lives only
on the box.

Pinned versions (Isaac Lab needs Python 3.11 for 2.3.x):

| Package | Version |
|---------|---------|
| isaaclab | 2.3.2.post1 |
| isaacsim | 5.1.0 |
| torch / torchvision | 2.7.0 / 0.22.0 (CUDA 12.8) |
| wandb | 0.31.0 |
| huggingface-hub | 2.4.0 |
| brev CLI | record the version from `brev --version` |

## 7. Launching runs from your laptop

[`scripts/brev/launch.py`](../scripts/brev/launch.py) runs one training job on a fresh Brev instance and
deletes the instance when it's done. Nothing has to be set up on the instance first:

```bash
source .venv/bin/activate
git push                                   # the instance clones the commit you're on from GitHub
python scripts/brev/launch.py --run-name p0-quadcopter-launch-s42 --quiet -- \
    --task Isaac-Quadcopter-Direct-v0 --num_envs 4096 --seed 42
```

Everything after `--` goes to `scripts/train.py`. The launcher adds `--headless`, `--logger wandb` and
`--run_name`. One run:

1. **Preflight:** the keys in `.env` are set, `brev` is signed in, there are no uncommitted changes, and the
   commit is pushed.
2. **Instance:** `brev create isaac-rl-<run name>-<UTC yymmddHHMM>`, trying each type in `--types` in order
   (default: Nebius L40S, then Crusoe L40S, then AWS `g6e.2xlarge`). Each type gets its own attempt of up to
   `--boot-timeout-min` (12). An instance that goes to `FAILURE` is deleted and the next type is tried.
3. **Job:** over ssh, clones this repo at that exact commit into `~/launch/isaac-rl` and starts
   [`scripts/brev/run_remote.sh`](../scripts/brev/run_remote.sh) in the background. It runs
   `install_isaac.sh` if Isaac Lab isn't installed yet (about 5 minutes on a 16-vCPU box), then `train.py`.
   The W&B and HF keys are sent as environment variables over ssh and never written to the instance's disk.
4. **Log:** streamed to the terminal (`--quiet` shows only progress lines) and saved in full to
   `logs/launch/<run>/run.log`.
5. **Check:** the W&B run must be `finished`, and the final checkpoint must be in the W&B run's files and on
   HF. If that can't be confirmed, the run's `train.py` log directory (checkpoints included) is copied to
   `logs/launch/<run>/run_dir/` before the instance goes.
6. **Teardown:** `brev delete`, then waits until the instance is really gone, even after an error or Ctrl-C.
   Box time and cost go into `logs/launch/<run>/launch.json` and into the W&B run summary as `launch/*`.

The exit code is 0 only when training succeeded and the checkpoint was verified.

| Option | What it does |
|---|---|
| `--box isaac-rl-video` | Reuse an existing instance (starts it if stopped, skips the install, stops it at the end) |
| `--types a,b,c` | Instance types to try, in order (`brev search --gpu-name L40S` lists them). Env: `BREV_TYPES` |
| `--max-setup-min 40` / `--max-train-min 120` | Time limits for the install and for `train.py`. The training limit includes Kit's start-up (about 2 min on a new box, which builds its shader caches). At the limit, train.py gets SIGINT: it stops at the next env step, saves `model_<iteration>.pt`, ends the W&B run as failed, the run directory is copied back, and the instance is deleted. Ctrl-C on the laptop does the same. |
| `--boot-timeout-min 12` | How long each instance type gets to reach `RUNNING` |
| `--keep` | Leave the instance running at the end (for debugging; remember to delete it) |
| `--ref <commit>` | Run another pushed commit instead of `HEAD` |
| `--dry-run` | Preflight checks only |

The instance is kept only in one case: training succeeded, but the checkpoint couldn't be verified on W&B/HF
*or* copied back. It would otherwise be lost. The launcher then prints the `brev delete` command.

**Brev behaviour the launcher works around** (seen while testing it, 2026-11-13):

- A new instance runs its own `apt-get` for the first minute or so. `install_isaac.sh` waits for the dpkg lock.
- `brev create --type a,b,c` falls back only when the create call itself fails. An instance can be accepted
  and then go to `FAILURE`, so the launcher tries one type at a time and watches the status.
- `brev start` on a stopped instance can block for several minutes, and `brev stop`/`delete` are refused
  while it is `STARTING`. The launcher polls the status and retries the teardown.
- The Brev API sometimes answers `resource exhausted` (no capacity for `create`, and for a minute even
  `brev ls`). `brev ls` is retried with backoff, and each cleanup step runs even if another one fails.
- Nebius takes about 10 minutes to finish a delete or a stop. The launcher waits for it before exiting, so
  "box time" in `launch.json` includes that wait.

**If the laptop sleeps or loses its network mid-run,** the instance keeps running and billing.
[`scripts/brev/reap.py`](../scripts/brev/reap.py) lists run boxes (only names that end in the launcher's
timestamp, never the dev box) and deletes the ones older than `--max-age-hours` (default 6):

```bash
python scripts/brev/reap.py            # list
python scripts/brev/reap.py --delete   # delete the stale ones
```

Tests for the launcher's pure logic run on the laptop: `pip install -r requirements-local.txt && pytest tests`.
