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
