"""Compose a write-up video: a play.py chase-cam recording with what the policy saw, env 0's depth image, as a
picture-in-picture (PLAN.md Phase 5). Runs locally; needs ffmpeg.

    python scripts/play.py ... --video --chase_heading          # on a GPU box: the .mp4 and depth_env0.npz
    python scripts/compose_video.py <video.mp4> <depth_env0.npz> -o <out.mp4> [--scale 4]

The depth image is drawn near = bright (1 - depth / range), scaled up with square pixels, framed, in the top-right
corner. The two streams are aligned frame by frame (frame k of both is the state after step k) and the output stops
at the shorter one.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def load_depth(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path) as z:
        return z["frames"].astype(np.uint8), float(z["fps"])


def ffmpeg_command(
    video: Path, out: Path, width: int, height: int, fps: float, scale: int, margin: int = 20
) -> list[str]:
    pip = (
        f"[1:v]negate,scale={width * scale}:{height * scale}:flags=neighbor,"
        f"pad=iw+6:ih+6:3:3:color=white[pip];"
        f"[0:v][pip]overlay=main_w-overlay_w-{margin}:{margin}:shortest=1"
    )
    return [
        "ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
        "-filter_complex", pip, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out),
    ]  # fmt: skip


def compose(video: Path, depth: Path, out: Path, scale: int = 4) -> None:
    frames, fps = load_depth(depth)
    _, height, width = frames.shape
    cmd = ffmpeg_command(video, out, width, height, fps, scale)
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("depth", type=Path, help="depth_env0.npz written by play.py --video on a depth task")
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=4, help="depth pixels per output pixel (default 4)")
    args = parser.parse_args()
    compose(args.video, args.depth, args.out, args.scale)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
