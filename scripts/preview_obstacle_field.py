"""Draw obstacle-field tiles, one per difficulty level, as a top-down SVG. Runs on a laptop (numpy only).

    python scripts/preview_obstacle_field.py                      # docs/images/obstacle_field_levels.svg
    python scripts/preview_obstacle_field.py --seed 1001 --out /tmp/test_maps.svg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "drone_nav"))

from drone_nav.terrain import obstacle_field as of  # noqa: E402

COLORS = {"cylinder": "#2f7d4f", "pole": "#b8860b", "box": "#7a5c3e", "wall": "#555c66"}
PX = 22  # pixels per metre


def tile_svg(layout: of.Layout, y_offset: float, label: str) -> list[str]:
    p = layout.params
    L, W = p.size
    # SVG y grows downward; flip so +y is up, as in the env.
    X = lambda x: 90 + x * PX  # noqa: E731
    Y = lambda y: y_offset + (W - y) * PX  # noqa: E731
    out = [f'<text x="8" y="{Y(W / 2) + 4:.1f}" font-size="12">{label}</text>']
    out.append(f'<rect x="{X(0)}" y="{Y(W)}" width="{L * PX}" height="{W * PX}" fill="#f4f1ea" stroke="#999"/>')
    for x0, x1, color in ((0, p.start_zone, "#cfe3ff"), (L - p.goal_zone, L, "#ffd9cf")):
        out.append(f'<rect x="{X(x0)}" y="{Y(W)}" width="{(x1 - x0) * PX}" height="{W * PX}" fill="{color}"/>')
    if p.side_walls:
        t = p.side_wall_thickness
        for y in (0.0, W - t):
            out.append(f'<rect x="{X(0)}" y="{Y(y + t)}" width="{L * PX}" height="{t * PX}" fill="#333"/>')
    for o in layout.obstacles:
        if o.is_round:
            r = max(o.radius * PX, 1.5)  # poles are 3-6 cm: keep them visible
            cx, cy = X(o.center[0]), Y(o.center[1])
            out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{COLORS[o.kind]}"/>')
        else:
            pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in o.polygon())
            out.append(f'<polygon points="{pts}" fill="{COLORS[o.kind]}"/>')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=of.TRAIN_SEED)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "images" / "obstacle_field_levels.svg")
    args = ap.parse_args()

    p = of.ObstacleFieldParams()
    L, W = p.size
    row_h = W * PX + 14
    body = []
    for level in range(10):
        d = (level + 0.5) / 10  # the middle of row `level` in Isaac Lab's curriculum
        lay = of.generate_layout(p, d, of.layout_rng(args.seed, d))
        label = f"level {level}"
        body += tile_svg(lay, 10 + level * row_h, label)
        body.append(
            f'<text x="{90 + L * PX + 10}" y="{10 + level * row_h + W * PX / 2 + 4:.1f}" font-size="11">'
            f"{lay.placed} obstacles incl. {lay.walls} wall{'' if lay.walls == 1 else 's'}, "
            f"gap {lay.min_gap:.2f} m</text>"
        )
    width, height = 90 + L * PX + 230, 10 + 10 * row_h
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        *body,
        "</svg>",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(svg) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
