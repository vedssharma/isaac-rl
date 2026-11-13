#!/usr/bin/env bash
# Install the pinned Isaac Sim + Isaac Lab stack on an NVIDIA Brev GPU instance
# (PLAN.md §12 step 2). Run it by hand on the dev box first; the commands that
# work become scripts/brev/Dockerfile for the base image (§12 step 3).
#
# Brev instances are Ubuntu VMs with NVIDIA drivers and open internet, so this
# installs straight from pypi.nvidia.com. Isaac Lab 2.3.x needs Python 3.11
# exactly, so we create a uv-managed 3.11 venv instead of using the system Python.
# Safe to rerun: the venv and uv's download cache are reused.
set -euo pipefail

ISAACLAB_VERSION="2.3.2.post1"   # pip package
ISAACLAB_GIT_TAG="v2.3.2"        # repo tag, for the train/play scripts
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
VENV="${VENV:-$HOME/isaac-venv}"
WORKDIR="${WORKDIR:-$HOME/workspace}"   # /home/ubuntu/workspace persists across brev stop/start

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
# Isaac Sim wheels are several GB; be patient with slow or flaky downloads.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

mkdir -p "$WORKDIR"

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== System libs (EGL/Vulkan/X for headless Kit) =="
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
# A freshly booted instance may still be running its own apt-get (Brev's setup); wait for the lock.
APT=(apt-get -o DPkg::Lock::Timeout=900)
$SUDO "${APT[@]}" update -qq
$SUDO "${APT[@]}" install -y -qq git cmake build-essential libglu1-mesa libegl1 libvulkan1 libxt6 libxrandr2 >/dev/null

echo "== Python 3.11 venv via uv =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install --upgrade pip

echo "== torch (CUDA 12.8 wheels) =="
uv pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX"

echo "== Isaac Sim + Isaac Lab =="
# Real isaacsim-* wheels live only on pypi.nvidia.com (PyPI has placeholder sdists).
uv pip install "isaaclab[isaacsim,all]==${ISAACLAB_VERSION}" --extra-index-url https://pypi.nvidia.com

echo "== Isaac Lab repo (for scripts/reinforcement_learning/*) =="
if [ ! -d "$WORKDIR/IsaacLab" ]; then
  git clone --depth 1 --branch "$ISAACLAB_GIT_TAG" https://github.com/isaac-sim/IsaacLab.git "$WORKDIR/IsaacLab"
fi
# The isaaclab pip wheel doesn't make isaaclab_rl / isaaclab_tasks importable on
# their own, and the repo's train.py needs them. Install the repo's extensions
# (editable) with RSL-RL, the standard setup for running Isaac Lab's scripts.
(cd "$WORKDIR/IsaacLab" && ./isaaclab.sh --install rsl_rl)

echo "== Experiment tracking =="
uv pip install "wandb==0.31.0"

echo "== Versions =="
python - <<'PY'
import importlib.metadata as md
for p in ["isaacsim", "isaaclab", "isaaclab_rl", "isaaclab_tasks", "isaaclab_assets", "rsl-rl-lib", "wandb", "torch"]:
    try:
        print(f"{p:16s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p:16s} (not installed)")
import torch; print("cuda ok:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

cat <<MSG

Installed. Smoke test (first start builds shader caches and can take several minutes):

  source $VENV/bin/activate
  export PYTHONUNBUFFERED=1   # Kit hard-exits on close and drops buffered stdout when logging to a file
  cd $WORKDIR/IsaacLab
  python scripts/reinforcement_learning/rsl_rl/train.py \\
      --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 4096 --max_iterations 10

Then a full run with this repo's scripts/train.py, which logs to W&B, finishes the W&B run
cleanly and uploads the final checkpoint to HF. Export WANDB_API_KEY, WANDB_ENTITY,
WANDB_PROJECT, HF_TOKEN and HF_REPO_ID on the instance first (the names used in .env):

  cd <this repo>
  python scripts/train.py --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 4096 \\
      --seed 42 --run_name p0-quadcopter-baseline-s42 --logger wandb

When you're done: run 'brev stop isaac-rl-dev' on your laptop so the GPU stops billing.
MSG
