"""Distil the Phase 2 teacher into the depth student with DAgger (PLAN.md Phase 3, Option A).

    python scripts/distill.py --task Drone-Nav-Depth-v0 --headless --num_envs 4096 --seed 42 \
        --teacher_run wn2l0g9q --run_name p3-distill-s42 --logger wandb [env.* Hydra overrides]

Every iteration rolls out num_steps_per_env steps in all envs. The executed action is the teacher's with
probability beta (falling from 1 to 0 over --teacher_decay_iterations, so early episodes get far enough to be worth
learning from), else the student's. Every visited state is labelled with the frozen teacher's action, computed by
RSL-RL from the env's "teacher" observation, and the student (drone_nav/policies/depth_student.py) is fitted to the
labels with MSE for --epochs passes over the rollout. The curriculum follows whoever flies, so the student climbs
the levels as it improves.

Like train.py: W&B logging (entity and project from WANDB_ENTITY / WANDB_PROJECT; the run id from WANDB_RUN_ID when
the launcher sets it), checkpoints model_<iteration>.pt in logs/distill/<timestamp>_<run_name>/ and on W&B, the final
one uploaded to Hugging Face, and final_checkpoint / hf_url / training_time_s in the W&B summary, so launch.py can
verify the run and play.py can evaluate the student (launch.py --script distill --eval ...).
"""

import argparse
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

import kit_exit  # isort: skip
import wandb_summary  # isort: skip

ORIG_ARGV = list(sys.argv)

parser = argparse.ArgumentParser(description="Distil the Phase 2 teacher into the depth student (DAgger).")
parser.add_argument("--task", type=str, default="Drone-Nav-Depth-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="the teacher's RSL-RL config")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--epochs", type=int, default=2, help="passes over each rollout")
parser.add_argument("--num_minibatches", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=5e-4)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--teacher_decay_iterations", type=int, default=100, help="beta falls from 1 to 0 over these")
parser.add_argument("--teacher_run", type=str, default="wn2l0g9q", help="W&B run id (or entity/project/id)")
parser.add_argument("--teacher_checkpoint", type=str, default=None, help="file in that run (default: its final)")
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--logger", type=str, default="wandb", choices=("wandb", "none"))
parser.add_argument("--hf_upload", choices=("final", "none"), default="final")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "drone_nav"))
import drone_nav.tasks  # noqa: E402, F401
from drone_nav.policies import depth_student as ds  # noqa: E402
from drone_nav.tasks.obstacles.obstacles_env import GOAL_OBS_DIM  # noqa: E402

EXPERIMENT = "drone_nav_depth_distill"
# Isaac Lab's hydra_task_config wrapper drops main()'s return value, so main() leaves its exit code here.
EXIT_CODE = [1]
VERSIONED_PACKAGES = ("isaacsim", "isaaclab", "isaaclab_rl", "rsl-rl-lib", "torch", "wandb")


def git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def fetch_teacher(run_ref: str, name: str | None, dest: Path) -> tuple[Path, str]:
    """Download the teacher's checkpoint from its W&B run; returns (local path, "entity/project/id:file")."""
    import wandb

    if run_ref.count("/") != 2:
        run_ref = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{run_ref}"
    run = wandb.Api(timeout=60).run(run_ref)
    name = name or wandb_summary.summary_value(run, "final_checkpoint")
    if not name:
        raise SystemExit(f"W&B run {run_ref} has no final_checkpoint; pass --teacher_checkpoint")
    run.file(name).download(root=str(dest), replace=True)
    print(f"[INFO] Teacher: {name} from {run.url}")
    return dest / name, f"{run_ref}:{name}"


def upload_to_hf(log_dir: Path, final_ckpt: Path) -> str | None:
    """The final checkpoint and params/ to HF_REPO_ID; never raises."""
    repo_id, token = os.environ.get("HF_REPO_ID"), os.environ.get("HF_TOKEN")
    if not repo_id or not token:
        print("[WARN] HF_REPO_ID or HF_TOKEN is not set; skipping the Hugging Face upload.")
        return None
    path_in_repo = f"runs/{EXPERIMENT}/{log_dir.name}"
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(log_dir),
            path_in_repo=path_in_repo,
            allow_patterns=[final_ckpt.name, "params/*"],
            commit_message=f"{log_dir.name}: final checkpoint",
        )
    except Exception:  # noqa: BLE001
        print(f"[WARN] Hugging Face upload failed; the checkpoint is still in {log_dir}:")
        traceback.print_exc()
        return None
    url = f"https://huggingface.co/{repo_id}/tree/main/{path_in_repo}"
    print(f"[INFO] Uploaded final checkpoint to {url}")
    return url


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    run_name = args_cli.run_name or "distill"
    log_dir = REPO_ROOT / "logs" / "distill" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))
    device = env.unwrapped.device

    # The teacher, exactly as RSL-RL runs it, reading the env's "teacher" observation group.
    teacher_path, teacher_ref = fetch_teacher(args_cli.teacher_run, args_cli.teacher_checkpoint, log_dir / "teacher")
    teacher_cfg = agent_cfg.to_dict()
    teacher_cfg["obs_groups"] = {"policy": ["teacher"], "critic": ["teacher"]}
    teacher_runner = OnPolicyRunner(env, teacher_cfg, log_dir=None, device=agent_cfg.device)
    teacher_runner.load(str(teacher_path))
    teacher = teacher_runner.get_inference_policy(device=device)

    student = ds.DepthStudent(proprio_dim=GOAL_OBS_DIM, image_hw=env.unwrapped.depth_image_shape).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args_cli.learning_rate)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[INFO] Student: {n_params / 1e6:.2f}M parameters, config {student.config}")

    packages = {}
    for name in VERSIONED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    run_meta = {
        "git_sha": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "packages": packages,
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": env.num_envs,
        "teacher": teacher_ref,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "command": " ".join(ORIG_ARGV),
    }
    distill_cfg = {k: getattr(args_cli, k) for k in (
        "max_iterations", "num_steps_per_env", "epochs", "num_minibatches", "learning_rate", "max_grad_norm",
        "teacher_decay_iterations", "seed")}  # fmt: skip
    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    (log_dir / "params" / "distill.json").write_text(
        json.dumps({"distill": distill_cfg, "student": student.config, "run_meta": run_meta}, indent=2) + "\n"
    )

    wandb = None
    if args_cli.logger == "wandb":
        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=f"{log_dir.name}",
            dir=str(log_dir),
            config={"distill": distill_cfg, "student": student.config, "run_meta": run_meta},
        )

    def save(iteration: int) -> Path:
        path = log_dir / f"model_{iteration}.pt"
        torch.save(ds.checkpoint(student, iteration, {"teacher": teacher_ref, "distill": distill_cfg}), path)
        if wandb is not None:
            wandb.save(str(path), base_path=str(log_dir), policy="now")
        return path

    # As RSL-RL does: spread the first episodes' lengths so resets don't all happen at once.
    env.unwrapped.episode_length_buf = torch.randint_like(
        env.unwrapped.episode_length_buf, high=int(env.unwrapped.max_episode_length)
    )
    obs = env.get_observations()
    n, T = env.num_envs, args_cli.num_steps_per_env
    obs_buf = torch.zeros(T, n, student.obs_dim, device=device)
    label_buf = torch.zeros(T, n, env.num_actions, device=device)

    start = time.time()
    exit_code, iteration, final_ckpt = 1, 0, None
    try:
        for iteration in range(args_cli.max_iterations):
            t0 = time.time()
            beta = ds.teacher_fraction(iteration, args_cli.teacher_decay_iterations)
            episode_logs: dict[str, list[torch.Tensor]] = {}
            student.eval()
            # no_grad, not inference_mode: the normalizer's buffers are updated here and used in backprop later.
            with torch.no_grad():
                for t in range(T):
                    policy_obs = obs["policy"]
                    teacher_actions = teacher(obs)
                    student.update_normalizer(policy_obs)
                    student_actions = student(policy_obs)
                    obs_buf[t] = policy_obs
                    label_buf[t] = teacher_actions
                    obs, _, _, extras = env.step(ds.mix_actions(teacher_actions, student_actions, beta))
                    for key, value in (extras.get("log") or {}).items():
                        value = torch.as_tensor(value, device=device).float().reshape(-1)
                        episode_logs.setdefault(key, []).append(value)
            t_collect = time.time() - t0

            student.train()
            x, y = obs_buf.reshape(T * n, -1), label_buf.reshape(T * n, -1)
            losses = []
            for _ in range(args_cli.epochs):
                for idx in ds.minibatches(T * n, args_cli.num_minibatches, device):
                    loss = ds.imitation_loss(student(x[idx]), y[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args_cli.max_grad_norm)
                    optimizer.step()
                    losses.append(loss.item())
            t_iter = time.time() - t0

            stats = {
                "Loss/imitation": sum(losses) / len(losses),
                "Distill/teacher_fraction": beta,
                "Perf/collection_time": t_collect,
                "Perf/learning_time": t_iter - t_collect,
                "Perf/total_fps": T * n / t_iter,
            }
            for key, values in episode_logs.items():
                values = torch.cat(values)
                if len(values):
                    stats[f"Episode/{key}"] = values.mean().item()
            if wandb is not None:
                wandb.log(stats, step=iteration)
            print(
                f"Learning iteration {iteration}/{args_cli.max_iterations}  loss {stats['Loss/imitation']:.4f}  "
                f"beta {beta:.2f}  success {stats.get('Episode/Metrics/success_rate', float('nan')):.3f}  "
                f"collision {stats.get('Episode/Episode_Termination/collision', float('nan')):.3f}  "
                f"level9 {stats.get('Episode/Curriculum/level_9', float('nan')):.2f}  "
                f"Iteration time {t_iter:.2f}s"
            )
            if args_cli.save_interval and iteration % args_cli.save_interval == 0:
                save(iteration)
        final_ckpt = save(iteration)
        exit_code = 0
    except KeyboardInterrupt:
        print(f"[INFO] Distillation stopped at iteration {iteration}")
        save(iteration)
    finally:
        train_time = round(time.time() - start, 2)
        print(f"Training time: {train_time} seconds")
        if wandb is not None:
            summary = {"training_time_s": train_time}
            if exit_code == 0 and final_ckpt is not None:
                summary["final_checkpoint"] = final_ckpt.name
                if args_cli.hf_upload == "final":
                    hf_url = upload_to_hf(log_dir, final_ckpt)
                    if hf_url:
                        summary["hf_url"] = hf_url
            wandb.run.summary.update(summary)
            wandb.finish(exit_code=exit_code)
    env.close()
    EXIT_CODE[0] = exit_code


if __name__ == "__main__":
    try:
        main()
        code = EXIT_CODE[0]
    except BaseException as e:  # noqa: BLE001
        code = e.code if isinstance(e, SystemExit) and isinstance(e.code, int) else 1
        if not isinstance(e, SystemExit):
            traceback.print_exc()
    if code:
        os._exit(code)
    kit_exit.close_app(simulation_app)
