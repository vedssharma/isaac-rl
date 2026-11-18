"""CPU tests for scripts/kit_exit.py: a hung Kit shutdown must not keep the process (and the paid box) alive."""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run(close_seconds: float, timeout_s: float) -> tuple[int, float, str]:
    code = f"""
import sys, time
sys.path.insert(0, {str(SCRIPTS)!r})
import kit_exit
class App:
    def close(self):
        time.sleep({close_seconds})
kit_exit.close_app(App(), timeout_s={timeout_s})
print("returned normally", flush=True)
"""
    start = time.monotonic()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    return p.returncode, time.monotonic() - start, p.stdout


def test_a_hung_shutdown_exits_with_0_after_the_timeout():
    code, took, out = run(close_seconds=20.0, timeout_s=0.5)
    assert code == 0
    assert took < 10.0  # far less than the 20 s "hang"
    assert "exiting without it" in out and "returned normally" not in out


def test_a_normal_shutdown_returns():
    code, took, out = run(close_seconds=0.05, timeout_s=5.0)
    assert code == 0 and "returned normally" in out and "WARN" not in out
