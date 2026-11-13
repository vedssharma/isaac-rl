#!/usr/bin/env python3
"""List, and optionally delete, run boxes that outlived their launcher (PLAN.md §3.2 cost guard).

    python scripts/brev/reap.py                  # list run boxes and their age
    python scripts/brev/reap.py --delete         # delete the ones older than --max-age-hours

Only instances named like launch.py's run boxes (isaac-rl-<run>-<UTC yymmddHHMM>) are considered, so a
hand-made instance such as the dev box (isaac-rl-video) is never touched. launch.py deletes its box itself;
this is for when the laptop slept, lost its network or was closed mid-run.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from brevcli import brev, list_instances, parse_run_box


def stale_run_boxes(instances: list[dict], max_age_hours: float, now: datetime) -> list[tuple[dict, float, bool]]:
    """(instance, age in hours, older than the limit) for every launcher-made instance."""
    rows = []
    for inst in instances:
        parsed = parse_run_box(inst.get("name", ""))
        if parsed is None:
            continue
        age = (now - parsed[1]).total_seconds() / 3600
        rows.append((inst, age, age > max_age_hours))
    return rows


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-age-hours", type=float, default=6, help="older run boxes are stale (default 6)")
    p.add_argument("--delete", action="store_true", help="delete the stale ones (default: only list them)")
    args = p.parse_args(argv)

    rows = stale_run_boxes(list_instances(), args.max_age_hours, datetime.now(UTC))
    if not rows:
        print("No run boxes.")
        return 0
    for inst, age, stale in rows:
        mark = "STALE" if stale else "ok   "
        print(f"{mark} {inst['name']:<60} {inst.get('status', '?'):<10} {age:5.1f} h  {inst.get('instance_type', '')}")
    stale = [inst["name"] for inst, _, is_stale in rows if is_stale]
    if not stale:
        return 0
    if not args.delete:
        print(f"\n{len(stale)} stale; rerun with --delete to delete them.")
        return 0
    failed = 0
    for name in stale:
        result = brev("delete", name, check=False, timeout=300)
        print(f"{'deleted' if result.returncode == 0 else 'FAILED to delete'} {name}")
        failed += result.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
