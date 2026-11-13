# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained RSL-RL policy: deterministic rollouts, episode metrics and a chase-cam video.

Based on IsaacLab v2.3.2 scripts/reinforcement_learning/rsl_rl/play.py, with these changes:

- The checkpoint comes from a W&B run (--wandb_run <id>): its summary's final_checkpoint, the file
  train.py recorded, instead of a guess from a directory listing. Or pass --checkpoint <path>.
- Runs until --episodes episodes have finished (PLAN.md §7: 1000), then prints success / crash / time-out
  rates with 95% confidence intervals, time to goal, path efficiency and speed, from the env's
  per-episode log (extras["log"]). The initial episode-length spread used for training is undone, so every
  evaluated episode gets the full time limit.
- --video records env 0 for --video_length steps. Drone-Nav tasks point the viewer at env 0's drone.
- Results go to logs/play/<run>/eval.json. With --wandb_run they are also written to that W&B run's
  summary (eval/*), and the video and eval.json are uploaded next to its checkpoint on Hugging Face.

    python scripts/play.py --task Drone-Nav-Goal-v0 --headless --wandb_run <run id> --video
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RL agent trained with RSL-RL.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=12345, help="Seed (different from training by default).")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--wandb_run", type=str, help="W&B run id (or entity/project/id) whose final checkpoint to load.")
src.add_argument("--checkpoint_path", type=str, help="A local checkpoint file instead.")
parser.add_argument("--episodes", type=int, default=1000, help="Evaluate until this many episodes have ended.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video of env 0.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the video (in env steps).")
parser.add_argument("--hf_upload", choices=("yes", "no"), default="yes", help="Upload eval.json and the video to HF.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os
import re
import traceback
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "drone_nav"))
import drone_nav.tasks  # noqa: E402, F401

REPO_ROOT = Path(__file__).resolve().parents[1]


def fetch_wandb_checkpoint(run_ref: str) -> tuple[Path, object]:
    """Download the W&B run's final checkpoint (summary final_checkpoint, else the highest model_<N>.pt)."""
    import wandb

    if run_ref.count("/") != 2:
        run_ref = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{run_ref}"
    run = wandb.Api(timeout=60).run(run_ref)
    name = run.summary.get("final_checkpoint")
    if not name:
        ckpts = [f.name for f in run.files() if re.fullmatch(r"model_\d+\.pt", f.name)]
        if not ckpts:
            raise SystemExit(f"W&B run {run_ref} has no model_<N>.pt files")
        name = max(ckpts, key=lambda n: int(n[6:-3]))
        print(f"[WARN] {run_ref} has no final_checkpoint in its summary (unfinished run?); using {name}")
    dest = REPO_ROOT / "logs" / "play" / f"{run.name}-{run.id}"
    run.file(name).download(root=str(dest), replace=True)
    print(f"[INFO] Downloaded {name} from {run.url}")
    return dest / name, run


def rate_ci(successes: int, n: int) -> tuple[float, float, float]:
    """Rate with a 95% Wilson score interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z, p = 1.96, successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, max(0.0, centre - half), min(1.0, centre + half)


def summarize(logs: dict[str, list[torch.Tensor]], n_target: int) -> dict:
    """Turn the per-episode tensors from extras["log"] into rates (with CIs) and means."""
    cat = {k: torch.cat(v).float().cpu() for k, v in logs.items() if v}
    n = len(cat.get("Metrics/success_rate", []))
    out: dict = {"episodes": n, "episodes_requested": n_target}
    for key, values in sorted(cat.items()):
        if key.startswith("Episode_Termination/") or key == "Metrics/success_rate":
            p, lo, hi = rate_ci(int(values.sum().item()), len(values))
            out[key] = {"rate": p, "ci95": [lo, hi]}
        else:
            out[key] = {"mean": values.mean().item() if len(values) else float("nan"), "n": len(values)}
    return out


def print_summary(summary: dict) -> None:
    print(f"\n==== Evaluation: {summary['episodes']} episodes ====")
    for key, value in summary.items():
        if isinstance(value, dict) and "rate" in value:
            lo, hi = value["ci95"]
            print(f"  {key:<40} {100 * value['rate']:6.2f} %   (95% CI {100 * lo:.2f}-{100 * hi:.2f})")
        elif isinstance(value, dict):
            print(f"  {key:<40} {value['mean']:9.3f}     (n={value['n']})")


def upload_results(out_dir: Path, run) -> None:
    """eval.json and videos next to the run's checkpoint on HF (from the run's hf_url); never raises."""
    repo_id, token, hf_url = os.environ.get("HF_REPO_ID"), os.environ.get("HF_TOKEN"), run.summary.get("hf_url")
    if not (repo_id and token and hf_url):
        print("[WARN] No HF_REPO_ID / HF_TOKEN / hf_url in the run summary; skipping the HF upload.")
        return
    path_in_repo = hf_url.split("/tree/main/", 1)[-1] + "/play"
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(out_dir),
            path_in_repo=path_in_repo,
            allow_patterns=["eval.json", "videos/*/*.mp4"],
            commit_message=f"{run.name}: evaluation",
        )
        print(f"[INFO] Uploaded the evaluation to https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
    except Exception:
        print("[WARN] HF upload of the evaluation failed:")
        traceback.print_exc()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    run = None
    if args_cli.wandb_run:
        ckpt, run = fetch_wandb_checkpoint(args_cli.wandb_run)
    else:
        ckpt = Path(args_cli.checkpoint_path).resolve()
    out_dir = ckpt.parent if run is not None else REPO_ROOT / "logs" / "play" / ckpt.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(out_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(out_dir / "videos" / "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            name_prefix=f"play-{ckpt.stem}",
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # The training env spreads initial episode lengths at random; give every evaluated episode the full limit.
    env.unwrapped.episode_length_buf[:] = 0

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(ckpt))
    policy = runner.get_inference_policy(device=env.unwrapped.device)  # deterministic: the action mean
    print(f"[INFO] Evaluating {ckpt} on {args_cli.task} with {args_cli.num_envs} envs, seed {agent_cfg.seed}")

    logs: dict[str, list[torch.Tensor]] = {}
    episodes = 0
    obs = env.get_observations()
    steps = 0
    # Keep going until the video is complete too, even if the episodes are done first.
    video_steps = args_cli.video_length if args_cli.video else 0
    while simulation_app.is_running() and (episodes < args_cli.episodes or steps < video_steps):
        with torch.inference_mode():
            obs, _, dones, extras = env.step(policy(obs))
            runner.alg.policy.reset(dones)
        steps += 1
        log = extras.get("log")
        if log and episodes < args_cli.episodes:
            for key, value in log.items():
                logs.setdefault(key, []).append(torch.as_tensor(value).reshape(-1))
            episodes += len(log.get("Metrics/success_rate", []))
        if steps % 500 == 0:
            print(f"[INFO] step {steps}: {episodes}/{args_cli.episodes} episodes")

    summary = summarize(logs, args_cli.episodes)
    summary.update({"checkpoint": str(ckpt.name), "task": args_cli.task, "seed": agent_cfg.seed, "steps": steps})
    if run is not None:
        summary["wandb_run"] = run.path if isinstance(run.path, str) else "/".join(run.path)
    print_summary(summary)
    (out_dir / "eval.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[INFO] Wrote {out_dir / 'eval.json'}")

    env.close()  # finishes writing a video that is still recording
    if run is not None:
        flat = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                flat[f"eval/{key}"] = value.get("rate", value.get("mean"))
        flat["eval/episodes"] = summary["episodes"]
        run.summary.update(flat)
        print(f"[INFO] Wrote eval/* to the W&B summary of {run.url}")
        if args_cli.hf_upload == "yes":
            upload_results(out_dir, run)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        code = e.code if isinstance(e, SystemExit) else 1
        if not isinstance(e, SystemExit):
            traceback.print_exc()
        if code not in (0, None):
            os._exit(code if isinstance(code, int) else 1)
    simulation_app.close()
