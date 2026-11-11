#!/usr/bin/env bash
# Install the pinned Isaac Sim + Isaac Lab stack inside a Daytona GPU sandbox
# (PLAN.md §12 step 2). Run it by hand in a fresh sandbox started from the
# `daytona-gpu` snapshot, note anything you had to change, and that becomes
# the basis for scripts/daytona/build_snapshot.py (§12 step 3).
#
# Untested first draft: Isaac Lab 2.3.x needs Python 3.11 exactly, so we
# create a uv-managed 3.11 venv instead of trusting the image's Python.
set -euo pipefail

ISAACLAB_VERSION="2.3.2.post1"   # pip package
ISAACLAB_GIT_TAG="v2.3.2"        # repo tag, for the train/play scripts
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
VENV="${VENV:-$HOME/isaac-venv}"
WORKDIR="${WORKDIR:-$HOME}"

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== System libs (EGL/Vulkan/X for headless Kit) =="
if command -v apt-get >/dev/null; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq
  $SUDO apt-get install -y -qq git cmake build-essential libglu1-mesa libegl1 libvulkan1 libxt6 libxrandr2 >/dev/null
fi

echo "== Python 3.11 venv via uv =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install --upgrade pip

echo "== torch (CUDA 12.8 wheels) =="
uv pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX"

echo "== Isaac Sim + Isaac Lab =="
uv pip install "isaaclab[isaacsim,all]==${ISAACLAB_VERSION}" --extra-index-url https://pypi.nvidia.com

echo "== Isaac Lab repo (for scripts/reinforcement_learning/*) =="
if [ ! -d "$WORKDIR/IsaacLab" ]; then
  git clone --depth 1 --branch "$ISAACLAB_GIT_TAG" https://github.com/isaac-sim/IsaacLab.git "$WORKDIR/IsaacLab"
fi

echo "== Versions =="
python - <<'PY'
import importlib.metadata as md
for p in ["isaacsim", "isaaclab", "isaaclab_rl", "rsl-rl-lib", "torch"]:
    try:
        print(f"{p:12s} {md.version(p)}")
    except md.PackageNotFoundError:
        print(f"{p:12s} (not installed)")
import torch; print("cuda ok:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

cat <<MSG

Installed. Smoke test (first start builds shader caches and can take several minutes):

  source $VENV/bin/activate
  cd $WORKDIR/IsaacLab
  python scripts/reinforcement_learning/rsl_rl/train.py \\
      --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 4096 --max_iterations 10

Then a full run with W&B (needs WANDB_API_KEY in the environment):

  python scripts/reinforcement_learning/rsl_rl/train.py \\
      --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 4096 \\
      --logger wandb --log_project_name "\${WANDB_PROJECT:-isaac-rl}"
MSG
