"""Draw crashed episodes recorded by play.py with env.record_failures=K (no Isaac Sim needed; matplotlib).

    python scripts/plot_failures.py <dir with *.npz> [more dirs] [--out <dir>] [--last_s 2.0]

For each episode it writes <name>.png:

- top: the tile from above, with its obstacle footprints, the path coloured by time, arrows for the heading (where
  the depth camera looks) every 0.5 s, the start, the goal and where it ended;
- middle: altitude (with the ceiling), horizontal speed and the closest lidar reading over time;
- bottom: the depth images of the last --last_s seconds (near = dark), when the episode has them.

It also prints one line per episode: termination, obstacle hit, duration, peak altitude, where along the corridor it
ended, and its speed then.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

KIND_COLORS = {"pole": "#d62728", "cylinder": "#1f77b4", "box": "#8c564b", "wall": "#7f7f7f", "side_wall": "#bbbbbb"}


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        rec = {k: z[k] for k in z.files if k != "meta"}
        rec["meta"] = json.loads(str(z["meta"]))
    return rec


def yaw_of(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def summary(rec: dict) -> dict:
    meta, pos, vel = rec["meta"], rec["pos"], rec["vel"]
    length_x = meta["tile_size"][0]
    return {
        "termination": meta["termination"],
        "hit": meta.get("hit_kind"),
        "duration_s": round(meta["steps"] * meta["step_dt"], 2),
        "peak_altitude": round(float(pos[:, 2].max()), 2),
        "end_x_frac": round(float(pos[-1, 0] / length_x), 2),
        "end_speed": round(float(np.linalg.norm(vel[-1, :2])), 2),
        "climb_rate_last_s": round(float(vel[-int(1 / meta["step_dt"]) :, 2].mean()), 2),
    }


def plot(rec: dict, out: Path, last_s: float = 2.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    meta, pos, vel = rec["meta"], rec["pos"], rec["vel"]
    dt, steps = meta["step_dt"], meta["steps"]
    t = np.arange(steps) * dt
    images = rec.get("image")
    n_frames = 0 if images is None else 8
    fig = plt.figure(figsize=(16, 9 if n_frames else 7))
    grid = fig.add_gridspec(3 if n_frames else 2, max(n_frames, 1), height_ratios=[3, 2, 1.6][: 3 if n_frames else 2])

    ax = fig.add_subplot(grid[0, :])
    for fp in meta["footprints"]:
        color = KIND_COLORS.get(fp["kind"], "k")
        if fp["kind"] in ("cylinder", "pole"):
            ax.add_patch(Circle(fp["center"], fp["radius"], color=color, alpha=0.8))
        else:
            hx, hy = fp["half_extents"]
            c, s = math.cos(fp["yaw"]), math.sin(fp["yaw"])
            corners = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]]) @ np.array([[c, s], [-s, c]])
            ax.add_patch(Polygon(corners + np.asarray(fp["center"]), color=color, alpha=0.8))
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=t, cmap="viridis", s=4, zorder=3)
    yaw = yaw_of(rec["quat"])
    every = max(int(0.5 / dt), 1)
    ax.quiver(pos[::every, 0], pos[::every, 1], np.cos(yaw[::every]), np.sin(yaw[::every]), angles="xy",
              scale_units="xy", scale=2.5, width=0.002, color="k", zorder=4)  # fmt: skip
    ax.plot(*pos[0, :2], "go", ms=8, zorder=5, label="start")
    ax.plot(*rec["goal"][:2], "*", color="gold", mec="k", ms=14, zorder=5, label="goal")
    ax.plot(*pos[-1, :2], "rx", ms=12, mew=3, zorder=5, label=meta["termination"])
    L, W = meta["tile_size"]
    ax.set_xlim(-1, L + 1)
    ax.set_ylim(-0.5, W + 0.5)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.015, label="time (s)")
    s = summary(rec)
    ax.set_title(
        f"{meta['termination']}" + (f" ({s['hit']})" if s["hit"] else "") + f" after {s['duration_s']} s, tile "
        f"{tuple(meta['tile'])}, env {meta['env']}; peak altitude {s['peak_altitude']} m, end speed {s['end_speed']} m/s"
    )

    ax2 = fig.add_subplot(grid[1, :])
    ax2.plot(t, pos[:, 2], label="altitude (m)")
    ax2.axhline(meta["max_altitude"], color="r", ls="--", lw=1, label="ceiling")
    ax2.plot(t, np.linalg.norm(vel[:, :2], axis=1), label="horizontal speed (m/s)")
    ax2.plot(t, rec["min_dist"], label="closest lidar reading (m)")
    ax2.axhline(meta["collision_radius"], color="k", ls=":", lw=1, label="collision radius")
    if n_frames:
        ax2.axvspan(max(t[-1] - last_s, 0), t[-1], color="0.9", zorder=0)
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(0, max(meta["max_altitude"] + 0.5, 5.0))
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper left", fontsize=8, ncol=5)

    if n_frames:
        idx = np.linspace(max(steps - 1 - int(last_s / dt), 0), steps - 1, n_frames).round().astype(int)
        for j, i in enumerate(idx):
            a = fig.add_subplot(grid[2, j])
            a.imshow(images[i], cmap="gray", vmin=0, vmax=255)
            a.set_title(f"{t[i] - t[-1]:+.2f} s", fontsize=8)
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="where to write the PNGs (default: next to the npz)")
    parser.add_argument("--last_s", type=float, default=2.0)
    args = parser.parse_args()
    for d in args.dirs:
        for path in sorted(d.glob("*.npz")):
            rec = load(path)
            out = (args.out or path.parent) / f"{path.stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plot(rec, out, args.last_s)
            print(path.stem, json.dumps(summary(rec)))


if __name__ == "__main__":
    main()
