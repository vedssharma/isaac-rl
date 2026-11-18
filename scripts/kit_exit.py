"""Close Isaac Sim (Kit) at the end of train.py / play.py, without letting a hung shutdown keep a paid box up.

On some Brev boxes, Kit's shutdown (simulation_app.close()) hung for 8-37 minutes after a run had already saved
everything (checkpoint, W&B, Hugging Face, results json): seen on 2026-11-17/18 in 2 of 8 launches, at the end of
both training and evaluation, on Nebius and Crusoe alike. The launcher only tears the box down once the process
exits, so the box sat idle and billing (docs/phase2-obstacles.md §14).

close_app() still closes Kit normally, but a watchdog hard-exits with code 0 if that takes longer than timeout_s.
Code 0 is right: everything the run produces is written before this is called; only Kit's cleanup is cut short.
"""

from __future__ import annotations

import os
import sys
import threading
import time


def close_app(app, timeout_s: float = 60.0) -> None:
    """app.close(), or os._exit(0) after timeout_s if it hasn't returned (or hard-exited the process) by then."""
    sys.stdout.flush()
    sys.stderr.flush()
    start = time.monotonic()

    def _give_up() -> None:
        print(f"[WARN] Kit shutdown still running after {timeout_s:.0f} s; exiting without it", flush=True)
        sys.stderr.flush()
        os._exit(0)

    watchdog = threading.Timer(timeout_s, _give_up)
    watchdog.daemon = True
    watchdog.start()
    app.close()
    watchdog.cancel()
    if time.monotonic() - start > 0.8 * timeout_s:
        print(f"[WARN] Kit shutdown took {time.monotonic() - start:.0f} s", flush=True)
