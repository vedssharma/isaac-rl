"""Check Drone-Nav-Depth-v0's depth camera against exact geometry, on a GPU box (PLAN.md Phase 3).

    python scripts/check_depth.py --headless [--num_envs 1024] [--level 5] [--name check_depth]

Builds the depth env with noise off, resets it and takes one zero-action step (the drones are level, at random
headings). Then, for every level camera, it compares the whole image with obstacle_field.camera_depth_replay: the
exact image of that tile's recorded layout (footprints as prisms, ground plane at z = 0) from the camera's own pose.
It reports the share of pixels that agree, by what each pixel should see (pole, cylinder, box, wall, side wall,
ground, nothing within range), and whether disagreeing pixels read nearer or farther. The replay includes the 8
neighbouring tiles, which a camera near a tile's end looks into. It also checks the observation layout,
times env.step, and saves camera | expected | difference images (plus a noisy one) for a few envs.

Results go to logs/check_depth/<name>/ (report.json, PNGs) and, with HF_TOKEN and HF_REPO_ID set, to
runs/drone_nav_depth/checks/<name>/ on Hugging Face. Exits 0 if at least --min_agreement of the pixels agree, 2
otherwise.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the depth camera against exact geometry.")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--level", type=int, default=5, help="difficulty row every env is put on")
parser.add_argument("--timing_steps", type=int, default=200)
parser.add_argument("--images", type=int, default=8, help="envs to save images of")
parser.add_argument("--tol", type=float, default=0.03, help="m, plus 1%% of the depth")
parser.add_argument("--min_agreement", type=float, default=0.99)
parser.add_argument("--camera_pitch_max_deg", type=float, default=0.0, help="randomize the camera's pitch (Phase 4)")
parser.add_argument("--name", default="check_depth")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

import kit_exit  # noqa: E402  # isort: skip

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "source" / "drone_nav"))
from drone_nav.sensors import depth as dm  # noqa: E402
from drone_nav.tasks.depth.depth_env import DroneNavDepthEnv, DroneNavDepthEnvCfg  # noqa: E402
from drone_nav.tasks.obstacles.obstacles_env import GOAL_OBS_DIM  # noqa: E402
from drone_nav.terrain import obstacle_field as of  # noqa: E402


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def save_images(path: Path, panels: list[np.ndarray], max_range: float, scale: int = 4) -> None:
    """One row of grayscale panels (H, W) in metres, near = dark, scaled up."""
    from PIL import Image

    row = np.concatenate([np.pad(p, ((0, 0), (0, 2)), constant_values=max_range) for p in panels], axis=1)
    img = (np.clip(row / max_range, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(path)


def upload(out_dir: Path, name: str) -> None:
    repo_id, token = os.environ.get("HF_REPO_ID"), os.environ.get("HF_TOKEN")
    if not (repo_id and token):
        print("[WARN] HF_REPO_ID / HF_TOKEN not set; not uploading")
        return
    try:
        from huggingface_hub import HfApi

        path = f"runs/drone_nav_depth/checks/{name}"
        HfApi(token=token).upload_folder(repo_id=repo_id, folder_path=str(out_dir), path_in_repo=path)
        print(f"[INFO] Uploaded the check to https://huggingface.co/{repo_id}/tree/main/{path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] HF upload failed: {e}")


def main() -> int:
    cfg = DroneNavDepthEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device or cfg.sim.device
    cfg.depth_noise = False
    cfg.eval_terrain_level = args_cli.level
    cfg.camera_pitch_max_deg = args_cli.camera_pitch_max_deg
    env = DroneNavDepthEnv(cfg)
    out_dir = REPO_ROOT / "logs" / "check_depth" / args_cli.name
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"num_envs": env.num_envs, "level": args_cli.level, "tol_m": args_cli.tol}

    obs, _ = env.reset()
    n, dev = env.num_envs, env.device
    H, W = env.depth_image_shape
    max_range = cfg.depth_camera.max_distance
    policy, teacher = obs["policy"], obs["teacher"]
    report["obs"] = {
        "policy_shape": list(policy.shape),
        "teacher_shape": list(teacher.shape),
        "policy_matches_teacher_goal_part": bool(torch.equal(policy[:, :GOAL_OBS_DIM], teacher[:, :GOAL_OBS_DIM])),
        "depth_part_in_0_1": bool(((policy[:, GOAL_OBS_DIM:] >= 0) & (policy[:, GOAL_OBS_DIM:] <= 1)).all()),
    }
    print(f"[INFO] observations: {report['obs']}")

    env.step(torch.zeros(n, 4, device=dev))
    depth = env.clean_depth().cpu().numpy().astype(np.float64)
    cam = env._camera.data
    pos = cam.pos_w.cpu().numpy().astype(np.float64)
    # Level drones (from the body's attitude: the camera may be pitched on purpose), at the body's heading.
    roll, pitch, yaw = (wrap(a) for a in euler_xyz_from_quat(env._robot.data.root_quat_w))
    level = ((roll.abs() < math.radians(1.0)) & (pitch.abs() < math.radians(1.0))).cpu().numpy()
    cam_pitch = env._camera_pitch.cpu().numpy().astype(np.float64)
    report["camera_pitch_deg_range"] = [float(np.degrees(cam_pitch.min())), float(np.degrees(cam_pitch.max()))]
    yaw = yaw.cpu().numpy().astype(np.float64)
    t = env._terrain
    rows, cols = t.terrain_levels.cpu().numpy(), t.terrain_types.cpu().numpy()
    env_origins = t.env_origins.cpu().numpy().astype(np.float64)
    terrain_origins = t.terrain_origins.cpu().numpy().astype(np.float64)
    origin_in_tile = np.asarray(env._origin_in_tile, dtype=np.float64)
    layouts = env._layouts()

    pattern = cfg.depth_camera.pattern_cfg
    f = dm.focal_length_px(W, pattern.horizontal_aperture, pattern.focal_length)
    col_y = -(np.arange(W) + 0.5 - W / 2) / f
    row_z = -(np.arange(H) + 0.5 - H / 2) / f  # square pixels (vertical_aperture None)

    agree, total, nearer, farther = Counter(), Counter(), Counter(), Counter()
    per_env = []
    saved = 0
    neighbour_pixels = 0
    for i in np.nonzero(level)[0]:
        r, c = int(rows[i]), int(cols[i])
        layout = layouts[(r, c)]
        # This tile's footprints, then its 8 neighbours' moved into this tile's frame: a camera near a tile's end
        # looks into the next tile (the range, 6 m, is shorter than a tile, so no further).
        footprints = of.layout_footprints(layout)
        own = len(footprints)
        for r2 in range(max(r - 1, 0), min(r + 2, terrain_origins.shape[0])):
            for c2 in range(max(c - 1, 0), min(c + 2, terrain_origins.shape[1])):
                if (r2, c2) == (r, c):
                    continue
                dx, dy = terrain_origins[r2, c2, :2] - terrain_origins[r, c, :2]
                footprints += [
                    dataclasses.replace(o, center=(o.center[0] + dx, o.center[1] + dy))
                    for o in of.layout_footprints(layouts[(r2, c2)])
                ]
        p_tile = pos[i, :2] - terrain_origins[r, c, :2] + origin_in_tile[:2]
        z = pos[i, 2] - env_origins[i, 2]
        exp, src = of.camera_depth_replay(
            footprints, (*p_tile, z), yaw[i], col_y, row_z, layout.params.obstacle_height, max_range, cam_pitch[i]
        )
        diff = depth[i] - exp
        tol = args_cli.tol + 0.01 * exp
        ok = np.abs(diff) <= tol
        names = np.array([o.kind for o in footprints] + ["ground", "nothing"])
        kinds = names[np.where(src >= 0, src, np.where(src == -1, len(footprints), len(footprints) + 1))]
        for k in np.unique(kinds):
            m = kinds == k
            total[k] += int(m.sum())
            agree[k] += int((ok & m).sum())
            nearer[k] += int((~ok & m & (diff < 0)).sum())
            farther[k] += int((~ok & m & (diff > 0)).sum())
        neighbour_pixels += int((src >= own).sum())
        per_env.append(float(ok.mean()))
        if saved < args_cli.images:
            save_images(out_dir / f"env{i:04d}.png", [depth[i], exp, np.abs(diff)], max_range)
            np.savez_compressed(out_dir / f"env{i:04d}.npz", camera=depth[i], expected=exp, source=src)
            saved += 1

    agreement = sum(agree.values()) / max(sum(total.values()), 1)
    report["geometry"] = {
        "level_cameras": int(level.sum()),
        "agreement": agreement,
        "by_expected_source": {
            k: {"pixels": total[k], "agree": agree[k] / total[k], "nearer": nearer[k], "farther": farther[k]}
            for k in sorted(total)
        },
        "pixels_seeing_a_neighbouring_tile": neighbour_pixels,
        "envs_below_95pct": int(sum(a < 0.95 for a in per_env)),
        "worst_envs": sorted(per_env)[:5],
    }
    print(f"[INFO] geometry: {json.dumps(report['geometry'], indent=1)}")

    # A noisy image of the same envs, for the record.
    env.cfg.depth_noise = True
    noisy = env._read_depth().cpu().numpy()
    env.cfg.depth_noise = False
    for i in np.nonzero(level)[0][: args_cli.images]:
        save_images(out_dir / f"env{i:04d}_noisy.png", [depth[i], noisy[i]], max_range)
    report["noise"] = {"max_range_fraction_clean": float((depth >= max_range).mean()),
                       "max_range_fraction_noisy": float((noisy >= max_range).mean())}  # fmt: skip

    actions = torch.zeros(n, 4, device=dev)
    for _ in range(10):
        env.step(actions)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args_cli.timing_steps):
        env.step(actions)
    torch.cuda.synchronize()
    step_s = (time.perf_counter() - start) / args_cli.timing_steps
    report["timing"] = {"ms_per_step": 1000 * step_s, "env_steps_per_s": n / step_s}
    print(f"[INFO] timing: {report['timing']}")

    ok = agreement >= args_cli.min_agreement and report["obs"]["policy_matches_teacher_goal_part"]
    report["passed"] = bool(ok)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[INFO] {'PASSED' if ok else 'FAILED'}: {100 * agreement:.2f}% of pixels agree")
    upload(out_dir, args_cli.name)
    env.close()
    return 0 if ok else 2


if __name__ == "__main__":
    code = main()
    if code:
        os._exit(code)
    kit_exit.close_app(simulation_app)
