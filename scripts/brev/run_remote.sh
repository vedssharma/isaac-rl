#!/usr/bin/env bash
# Runs ON the Brev instance, started in the background by scripts/brev/launch.py from a clone of this repo
# checked out at the run's git SHA. Installs Isaac Sim / Isaac Lab if they aren't there yet, then runs
# scripts/train.py, or $RUN_SCRIPT (e.g. scripts/play.py). Secrets (WANDB_*, HF_*) arrive as environment
# variables; nothing is written to disk.
#
#   [RUN_SCRIPT=scripts/play.py] bash scripts/brev/run_remote.sh <run dir> <max minutes> <script args...>
#
# Writes <run dir>/phase (setup | train | done) and, when finished, <run dir>/exit_code:
#   0 = training succeeded, 124 = hit the time limit, 90 = install failed, anything else = train.py's code.
set -uo pipefail

RUN_DIR="$1"; MAX_TRAIN_MIN="$2"; shift 2
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${VENV:-$HOME/isaac-venv}"
mkdir -p "$RUN_DIR"

# exit_code is written last: once it exists, the launcher reads the log one final time and stops.
finish() { echo "[run_remote] exit code $1"; echo done > "$RUN_DIR/phase"; echo "$1" > "$RUN_DIR/exit_code"; exit 0; }

export PYTHONUNBUFFERED=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES

echo setup > "$RUN_DIR/phase"
echo "[run_remote] $(date -u +%FT%TZ) repo $(git -C "$REPO_DIR" rev-parse HEAD) on $(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

# Skip the install when the pinned stack is already there (a reused dev box). install_isaac.sh is safe to rerun.
installed() {
  [ -x "$VENV/bin/python" ] && "$VENV/bin/python" - <<'PY' 2>/dev/null
import importlib.metadata as md
assert md.version("isaaclab") == "2.3.2.post1"
for p in ("isaaclab_rl", "isaaclab_tasks", "rsl-rl-lib", "wandb"):
    md.version(p)
PY
}
if installed; then
  echo "[run_remote] Isaac Lab already installed in $VENV"
else
  echo "[run_remote] installing Isaac Sim + Isaac Lab ($(date -u +%T))"
  bash "$REPO_DIR/scripts/brev/install_isaac.sh" || finish 90
  echo "[run_remote] install finished ($(date -u +%T))"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
cd "$REPO_DIR"
echo train > "$RUN_DIR/phase"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/train.py}"
echo "[run_remote] $(date -u +%FT%TZ) python $RUN_SCRIPT $*"
# On the time limit, SIGINT lets train.py mark the W&B run failed and flush it; SIGKILL 3 minutes later if not.
timeout --signal=INT --kill-after=180 "${MAX_TRAIN_MIN}m" python "$RUN_SCRIPT" "$@"
finish $?
