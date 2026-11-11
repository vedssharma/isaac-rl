# Setup: accounts and API keys

This is step 1 of [PLAN.md §12](../PLAN.md#12-first-week-concrete-next-steps). It takes about 20 minutes,
and only you can do it, because each account needs your own sign-up and payment details.

## 1. Create the accounts

| Service | What it's for | Sign up | Get the key |
|---------|---------------|---------|-------------|
| **Daytona** | GPU sandboxes for every training run | https://app.daytona.io | Dashboard → **Keys** → Create Key (sandbox + snapshot permissions) |
| **Weights & Biases** | Live metrics and checkpoint artifacts | https://wandb.ai/site | https://wandb.ai/authorize |
| **Hugging Face** | Model repo for checkpoints, videos and the model card | https://huggingface.co/join | https://huggingface.co/settings/tokens (fine-grained, write access to your model repo) |

## 2. Set a spending cap before anything else

Do this before starting your first GPU sandbox. The whole project is budgeted at roughly
60–120 GPU-hours (PLAN.md §3.3).

- **Daytona:** set a monthly limit in the billing settings, and check that your account has GPU access
  (GPU sandboxes can require a request or a higher tier). Note your hourly rate for the RTX Pro 6000 and the
  H100 so we can log cost per run later.
- **W&B and HF:** the free tiers are enough for this project.

## 3. Create the HF model repo

```bash
hf auth login            # or: export HF_TOKEN=...
hf repos create <your-hf-username>/isaac-rl-drone-nav --private
```

## 4. Fill in `.env` and check it

```bash
cp .env.example .env     # .env is git-ignored
$EDITOR .env             # paste the keys

python -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt
python scripts/check_env.py
```

`check_env.py` makes one read-only call to each service and prints `All good.` when every key works.
It never prints a full secret.

## Rules for secrets

- Keys live only in your local `.env` and are injected into sandboxes as environment variables at launch.
- Never bake a key into a Daytona snapshot or image, and never commit `.env`.
- If a key leaks, revoke it in that service's dashboard and make a new one.

## Next: step 2

### Daytona Tier 1/2 network limits

On Daytona Tier 1 and 2, sandboxes can only reach the hosts on
[Daytona's allow-list](https://github.com/daytonaio/sandbox-network-whitelist/blob/main/whitelist.yaml).
PyPI, PyTorch, GitHub, Hugging Face and AWS S3 (path-style URLs) are on it. NVIDIA's package index
(`pypi.nvidia.com`), `nvcr.io` and W&B are not. Tier 3 (a $500 top-up) lifts this. Until then:

| Blocked | Workaround |
|---------|-----------|
| Isaac Sim wheels on `pypi.nvidia.com` | A GitHub Action copies them once into a private HF repo; the install script pulls them from there |
| Isaac asset URL (virtual-hosted S3) | The install script uses the path-style S3 URL for the same bucket (`--kit_args` in the train command) |
| W&B logging | Log to TensorBoard and `hf upload` the logs and checkpoints to your HF repo |

**One-time wheel mirror** (takes a while; the wheels total several GB):

1. GitHub repo → Settings → Secrets and variables → Actions → add secret `HF_TOKEN` (write access).
2. Actions → **Mirror Isaac Sim wheels to Hugging Face** → Run workflow, with a repo name such as
   `<hf-username>/isaacsim-wheels`. It creates the repo as private. Keep it private: the wheels are under
   NVIDIA's license and are mirrored only for this project's own use.

### Install in a sandbox

Start one GPU sandbox by hand from the `daytona-gpu` snapshot (it is always ephemeral, so set auto-stop to 0 or
high and delete it yourself when done). Then, inside it:

```bash
git clone https://github.com/vedssharma/isaac-rl.git && cd isaac-rl
export HF_TOKEN=...                                   # read access is enough here
export ISAACSIM_WHEELS_REPO=<hf-username>/isaacsim-wheels
bash scripts/daytona/install_isaac.sh
```

Write down anything you had to change. Those notes become `build_snapshot.py` in step 3.

Pinned versions (Isaac Lab needs Python 3.11 for 2.3.x):

| Package | Version |
|---------|---------|
| isaaclab | 2.3.2.post1 |
| isaacsim | 5.1.0 |
| torch / torchvision | 2.7.0 / 0.22.0 (CUDA 12.8) |
| daytona (SDK) | 0.231.1 |
| wandb | 0.31.0 |
| huggingface-hub | 2.4.0 |
