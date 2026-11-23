"""CPU test for scripts/plot_failures.py on a synthetic crashed episode saved the way play.py saves them."""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from drone_nav.terrain import obstacle_field as of

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
plot_failures = pytest.importorskip("plot_failures")
pytest.importorskip("matplotlib")


def fake_record(with_image: bool) -> dict:
    p = of.ObstacleFieldParams()
    layout = of.generate_layout(p, 0.55, of.layout_rng(3, 0.55))
    steps, dt = 300, 0.02
    t = np.arange(steps) * dt
    pos = np.stack([1.0 + 3.0 * t, np.full(steps, p.size[1] / 2), 1.2 + 0.5 * t], axis=1).astype(np.float32)
    vel = np.tile(np.array([3.0, 0.0, 0.5], dtype=np.float32), (steps, 1))
    rec = {
        "pos": pos,
        "quat": np.tile(np.array([1.0, 0, 0, 0], dtype=np.float32), (steps, 1)),
        "vel": vel,
        "action": np.zeros((steps, 4), dtype=np.float32),
        "min_dist": np.full(steps, 2.0, dtype=np.float32),
        "goal": np.array([p.size[0] - 1.0, p.size[1] / 2, 1.5], dtype=np.float32),
        "meta": {
            "termination": "too_high", "hit_kind": None, "tile": [5, 3], "env": 17, "steps": steps, "step_dt": dt,
            "min_altitude": 0.1, "max_altitude": 4.0, "collision_radius": 0.1, "image_max_range": 6.0,
            "tile_size": list(p.size),
            "footprints": [dataclasses.asdict(o) for o in of.layout_footprints(layout)],
        },  # fmt: skip
    }
    if with_image:
        rec["image"] = np.random.default_rng(0).integers(0, 256, (steps, 48, 64), dtype=np.uint8)
    return rec


@pytest.mark.parametrize("with_image", [True, False])
def test_plot_a_saved_failure(tmp_path, with_image):
    rec = fake_record(with_image)
    path = tmp_path / "000_too_high.npz"
    arrays = {k: v for k, v in rec.items() if k != "meta"}
    np.savez_compressed(path, meta=json.dumps(rec["meta"]), **arrays)  # as play.py does
    loaded = plot_failures.load(path)
    assert loaded["meta"]["termination"] == "too_high" and len(loaded["meta"]["footprints"]) > 3
    s = plot_failures.summary(loaded)
    assert s["duration_s"] == 6.0 and s["peak_altitude"] == pytest.approx(1.2 + 0.5 * 299 * 0.02, abs=0.01)
    assert s["climb_rate_last_s"] == 0.5
    out = tmp_path / "000_too_high.png"
    plot_failures.plot(loaded, out)
    assert out.stat().st_size > 10_000
