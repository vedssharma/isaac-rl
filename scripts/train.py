# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an Isaac Lab task with RSL-RL. This is the entry point for every run in this repo.

Based on IsaacLab v2.3.2 scripts/reinforcement_learning/rsl_rl/train.py, with these changes:

- W&B is finished *before* the simulator closes. Kit hard-exits on close, which used to kill wandb
  before it uploaded the last metrics and the final checkpoint, leaving the run "running"/"crashed".
- Reproducibility metadata (this repo's git SHA, package versions, GPU, command line) goes to
  params/run_meta.yaml and to the W&B config under "run_meta".
- The final checkpoint, params/, git diffs and --video clips are uploaded to the HF model repo in HF_REPO_ID,
  and the clips are logged to W&B.
- The W&B entity and project come from WANDB_ENTITY / WANDB_PROJECT (the same names as .env).
- Logs go to <this repo>/logs/rsl_rl/<experiment>/<timestamp>_<run_name>, whatever the working directory.
- A failed run exits non-zero, so a launcher can tell it apart from a finished one.

Usage, on the Brev box with ~/isaac-venv active (run names follow PLAN.md §7: p{phase}-{desc}-s{seed}):

    python scripts/train.py --task Isaac-Quadcopter-Direct-v0 --headless --num_envs 4096 \
        --seed 42 --run_name p0-quadcopter-baseline-s42 --logger wandb

Hydra overrides work as in Isaac Lab, e.g. `agent.max_iterations=300 env.episode_length_s=20`.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

# Kit hard-exits on close and drops whatever is still buffered, so a finished run can look like a
# silent crash in a log file. Flush every line.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

ORIG_ARGV = list(sys.argv)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--hf_upload",
    choices=("final", "none"),
    default="final",
    help="'final' uploads the final checkpoint, params/ and git diffs to HF_REPO_ID after training; 'none' skips it.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install it, run:"
        f"\n\n\tpip install rsl-rl-lib=={RSL_RL_VERSION}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import platform
import signal
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Only one process uploads and writes metadata in multi-GPU runs.
IS_RANK_ZERO = int(os.environ.get("RANK", "0")) == 0
VERSIONED_PACKAGES = ("isaacsim", "isaaclab", "isaaclab_rl", "isaaclab_tasks", "rsl-rl-lib", "torch", "wandb")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def apply_env_defaults(args: argparse.Namespace) -> None:
    """Map the .env variable names onto what RSL-RL reads."""
    # RSL-RL's W&B writer takes the entity from WANDB_USERNAME, not WANDB_ENTITY.
    if os.environ.get("WANDB_ENTITY"):
        os.environ["WANDB_USERNAME"] = os.environ["WANDB_ENTITY"]
    # Otherwise RSL-RL logs to the agent config's default project, "isaaclab".
    if args.log_project_name is None and os.environ.get("WANDB_PROJECT"):
        args.log_project_name = os.environ["WANDB_PROJECT"]


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def collect_run_meta(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> dict:
    """Everything needed to say exactly what produced a run (PLAN.md §7)."""
    status = _git(REPO_ROOT, "status", "--porcelain")
    packages = {}
    for name in VERSIONED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "git_sha": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "git_branch": _git(REPO_ROOT, "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "packages": packages,
        "task": args_cli.task,
        "seed": agent_cfg.seed,
        "num_envs": env_cfg.scene.num_envs,
        "max_iterations": agent_cfg.max_iterations,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "hostname": platform.node(),
        "command": " ".join(ORIG_ARGV),
    }


def upload_to_hf(log_dir: str, final_ckpt: str, experiment_name: str) -> str | None:
    """Upload the final checkpoint, params/, git diffs and any training videos.

    Never raises: a failed upload must not lose the run.
    """
    repo_id, token = os.environ.get("HF_REPO_ID"), os.environ.get("HF_TOKEN")
    if not repo_id or not token:
        print("[WARN] HF_REPO_ID or HF_TOKEN is not set; skipping the Hugging Face upload.")
        return None
    path_in_repo = f"runs/{experiment_name}/{os.path.basename(log_dir)}"
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=log_dir,
            path_in_repo=path_in_repo,
            allow_patterns=[os.path.basename(final_ckpt), "params/*", "git/*", "videos/*/*.mp4"],
            commit_message=f"{os.path.basename(log_dir)}: final checkpoint",
        )
    except Exception:
        print(f"[WARN] Hugging Face upload failed; the checkpoint is still in {log_dir}:")
        traceback.print_exc()
        return None
    url = f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
    print(f"[INFO] Uploaded final checkpoint to {url}")
    return url


def finish_logging(runner: OnPolicyRunner, exit_code: int, summary: dict, videos: list[str]) -> None:
    """Flush and close the run's logger while the process is still alive."""
    writer = getattr(runner, "writer", None)
    if writer is None:
        return
    if runner.logger_type == "wandb":
        import wandb

        if wandb.run is not None:
            if videos:
                # RSL-RL doesn't log videos. Clips are named rl-video-step-<env step>.mp4; log them after the
                # last training iteration (W&B drops rows logged at an earlier step).
                wandb.log(
                    {f"Video/train/{Path(v).stem}": wandb.Video(v, format="mp4") for v in videos},
                    step=runner.current_learning_iteration + 1,
                )
            wandb.run.summary.update(summary)
            # Blocks until the history and every wandb.save()'d file (including the final checkpoint) are uploaded.
            wandb.finish(exit_code=exit_code)
    elif runner.logger_type == "neptune":
        writer.stop()
    writer.close()  # flush the TensorBoard event file


# Set by the SIGINT/SIGTERM handler; checked at every env step (see stop_on_signal).
STOP_SIGNAL: list[str] = []


def _record_stop_signal(signum, frame):
    STOP_SIGNAL.append(signal.Signals(signum).name)


def stop_on_signal(env: RslRlVecEnvWrapper) -> None:
    """Make a Ctrl-C / SIGTERM stop training cleanly, so the `finally` in main() finishes the W&B run.

    Kit installs its own SIGINT handler, which shuts the app down without unwinding Python. Raising from a
    Python signal handler doesn't work either: most of a training step runs inside Kit's update loop, and an
    exception raised in one of its Python callbacks is caught and logged by Kit, so training carries on. So
    the handler only records the signal, and the next env.step(), called from RSL-RL's rollout loop in plain
    Python, raises KeyboardInterrupt.
    """
    signal.signal(signal.SIGINT, _record_stop_signal)
    signal.signal(signal.SIGTERM, _record_stop_signal)
    step = env.step

    def step_or_stop(actions):
        if STOP_SIGNAL:
            raise KeyboardInterrupt(f"received {STOP_SIGNAL[0]}")
        return step(actions)

    env.step = step_or_stop


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    apply_env_defaults(args_cli)
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = str(REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    elif args_cli.export_io_descriptors:
        logger.warning("IO descriptors are only supported for manager based RL environments. None will be exported.")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state of this repo to logs. Isaac Lab is recorded by package version in run_meta: inside Kit,
    # isaaclab_tasks loads from the copy bundled in the isaaclab wheel, not from the git checkout.
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    if IS_RANK_ZERO:
        run_meta = collect_run_meta(env_cfg, agent_cfg)
        dump_yaml(os.path.join(log_dir, "params", "run_meta.yaml"), run_meta)
        # learn() creates the logger lazily (a no-op if it already exists). Create it now so the
        # metadata is in the W&B config from the start, even if training crashes.
        runner._prepare_logging_writer()
        if runner.logger_type == "wandb":
            import wandb

            wandb.config.update({"run_meta": run_meta})

    # run training
    exit_code = 1
    stop_on_signal(env)  # after Kit has started, so our handlers replace Kit's
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
        exit_code = 0
    except KeyboardInterrupt as e:
        print(f"[INFO] Training stopped at iteration {runner.current_learning_iteration}: {e}")
        if IS_RANK_ZERO:
            # Keep the progress since the last periodic save, so the run can be resumed (--resume).
            try:
                runner.save(os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt"))
            except Exception:
                traceback.print_exc()
        raise
    finally:
        train_time = round(time.time() - start_time, 2)
        print(f"Training time: {train_time} seconds")
        if IS_RANK_ZERO:
            # learn() saves model_<last iteration>.pt when it finishes
            final_ckpt = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
            summary = {"training_time_s": train_time}
            if exit_code == 0 and os.path.isfile(final_ckpt):
                summary["final_checkpoint"] = os.path.basename(final_ckpt)
                if args_cli.hf_upload == "final":
                    hf_url = upload_to_hf(log_dir, final_ckpt, agent_cfg.experiment_name)
                    if hf_url:
                        summary["hf_url"] = hf_url
            # Only finished clips: one still recording is written when the env closes, after this.
            videos = sorted(str(p) for p in Path(log_dir, "videos", "train").glob("*.mp4"))
            finish_logging(runner, exit_code, summary, videos)

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        # Hydra reports errors itself and raises SystemExit(1); anything else gets a traceback here.
        code = e.code if isinstance(e, SystemExit) else 1
        if not isinstance(e, SystemExit):
            traceback.print_exc()
        if code not in (0, None):
            # Exit now with the error code: Kit's shutdown hard-exits the process and may not keep it.
            os._exit(code if isinstance(code, int) else 1)
    # close sim app
    simulation_app.close()
