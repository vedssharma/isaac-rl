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

# Daytona Tier 1/2 sandboxes can't reach pypi.nvidia.com or NVIDIA's default
# asset URL. Set ISAACSIM_WHEELS_REPO to the private HF repo filled by
# .github/workflows/mirror-isaacsim-wheels.yml (needs HF_TOKEN with read access)
# to install Isaac Sim from there. Leave it empty on networks that can reach NVIDIA.
ISAACSIM_WHEELS_REPO="${ISAACSIM_WHEELS_REPO:-}"
WHEELS_DIR="${WHEELS_DIR:-$HOME/isaacsim-wheels}"
# Same NVIDIA asset bucket as Isaac Sim's default, but in the path-style form
# that Daytona's allow-list (*.us-west-2.amazonaws.com) permits.
ISAAC_ASSET_ROOT="${ISAAC_ASSET_ROOT:-https://s3.us-west-2.amazonaws.com/omniverse-content-production/Assets/Isaac/5.1}"

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
# Isaac Sim wheels are large (several GB in total) and pypi.nvidia.com can reset
# connections mid-download. Be patient and retry; uv caches finished wheels, so
# rerunning this script resumes where it stopped.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"

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
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install --upgrade pip

echo "== torch (CUDA 12.8 wheels) =="
uv pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX"

echo "== Isaac Sim + Isaac Lab =="
if [ -n "$ISAACSIM_WHEELS_REPO" ]; then
  # Mirror mode: real isaacsim-* wheels come from the private HF repo; PyPI only
  # has placeholder sdists for them, so forbid building those from source.
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: ISAACSIM_WHEELS_REPO is set but HF_TOKEN isn't (needed to read the private repo)." >&2
    exit 1
  fi
  uv pip install "huggingface-hub==2.4.0"
  hf download "$ISAACSIM_WHEELS_REPO" --include "*.whl" --local-dir "$WHEELS_DIR"
  NO_BUILD_ARGS=()
  for whl in "$WHEELS_DIR"/*.whl; do
    [ -e "$whl" ] || { echo "ERROR: no wheels downloaded from $ISAACSIM_WHEELS_REPO" >&2; exit 1; }
    name="$(basename "$whl")"; name="${name%%-*}"
    NO_BUILD_ARGS+=(--only-binary "${name//_/-}")
  done
  uv pip install "isaaclab[isaacsim,all]==${ISAACLAB_VERSION}" \
    --find-links "$WHEELS_DIR" --index-strategy unsafe-best-match "${NO_BUILD_ARGS[@]}"
else
  # Direct mode: real isaacsim-* wheels live only on pypi.nvidia.com.
  if ! curl -sS --max-time 30 -o /dev/null https://pypi.nvidia.com/isaacsim-core/; then
    echo "ERROR: cannot reach https://pypi.nvidia.com (Daytona Tier 1/2 blocks it)." >&2
    echo "Set ISAACSIM_WHEELS_REPO to your HF wheel mirror instead; see docs/setup.md." >&2
    exit 1
  fi
  uv pip install "isaaclab[isaacsim,all]==${ISAACLAB_VERSION}" --extra-index-url https://pypi.nvidia.com
fi

echo "== Isaac asset server =="
if curl -sSfI --max-time 30 -o /dev/null "$ISAAC_ASSET_ROOT/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"; then
  echo "reachable: $ISAAC_ASSET_ROOT"
else
  echo "WARNING: cannot reach $ISAAC_ASSET_ROOT; the quadcopter task won't find its robot model." >&2
fi

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
      --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 4096 --max_iterations 10 \\
      --kit_args "--/persistent/isaac/asset_root/cloud=$ISAAC_ASSET_ROOT"

Logs go to TensorBoard files under $WORKDIR/IsaacLab/logs/rsl_rl/. Daytona Tier 1/2
blocks W&B, so copy anything you want to keep to Hugging Face before the sandbox stops:

  hf upload "\$HF_REPO_ID" logs/rsl_rl runs/smoke-test
MSG
