"""compose_video.py on a synthetic recording (skipped without ffmpeg)."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compose_video  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def frame_count(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True,
    )  # fmt: skip
    return int(out.stdout.strip())


def test_compose_puts_the_depth_view_in_the_top_right(tmp_path):
    video = tmp_path / "chase.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=640x360:r=50:d=1.2",
                    "-pix_fmt", "yuv420p", str(video)], check=True)  # fmt: skip
    frames = np.zeros((50, 48, 64), dtype=np.uint8)  # 1 s of depth: everything at 0 = near, drawn white
    np.savez_compressed(tmp_path / "depth_env0.npz", frames=frames, dones=np.zeros(50, bool), fps=50.0)
    out = tmp_path / "out.mp4"
    compose_video.compose(video, tmp_path / "depth_env0.npz", out, scale=2)
    assert frame_count(out) == 50  # stops at the shorter stream
    # One decoded frame: the picture-in-picture (128 x 96 plus a 3 px frame) is bright, the rest is black.
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1", "-f", "rawvideo",
                          "-pix_fmt", "gray", "pipe:1"], capture_output=True, check=True).stdout  # fmt: skip
    img = np.frombuffer(raw, dtype=np.uint8).reshape(360, 640)
    assert img[20 + 50, 640 - 20 - 60] > 200  # inside the inset
    assert img[300, 100] < 30  # elsewhere
