"""Thin wrappers around the `brev` CLI and `ssh`, shared by launch.py and reap.py.

Brev has no Python SDK, so everything goes through the CLI. Plain `ssh <instance>` (using the config that
`brev refresh` writes) is more reliable than `brev exec` for scripted commands.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import UTC, datetime

# Run boxes are named isaac-rl-<run name>-<UTC yymmddHHMM>. reap.py only touches names with that suffix,
# so a hand-made box such as the dev box (isaac-rl-video) is never reaped.
RUN_BOX_PREFIX = "isaac-rl-"
STAMP_FORMAT = "%y%m%d%H%M"
RUN_BOX_RE = re.compile(rf"^{RUN_BOX_PREFIX}(?P<run>[a-z0-9-]+)-(?P<stamp>\d{{10}})$")
RUN_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME_LEN = 60

SSH_OPTS = ["-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30"]


class BrevError(RuntimeError):
    pass


def run_box_name(run_name: str, now: datetime | None = None) -> str:
    """The instance name for one run: isaac-rl-<run_name>-<UTC yymmddHHMM>."""
    if not RUN_NAME_RE.match(run_name):
        raise ValueError(f"run name {run_name!r} must be lowercase letters, digits and single hyphens")
    now = now or datetime.now(UTC)
    name = f"{RUN_BOX_PREFIX}{run_name}-{now.strftime(STAMP_FORMAT)}"
    if len(name) > MAX_NAME_LEN:
        raise ValueError(f"instance name {name!r} is longer than {MAX_NAME_LEN} characters; shorten the run name")
    return name


def parse_run_box(name: str) -> tuple[str, datetime] | None:
    """(run name, UTC creation time) for a launcher-made instance name, else None."""
    m = RUN_BOX_RE.match(name)
    if not m:
        return None
    try:
        created = datetime.strptime(m["stamp"], STAMP_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None
    return m["run"], created


def brev(*args: str, timeout: float = 120, check: bool = True) -> subprocess.CompletedProcess:
    # --no-check-latest stops the "a new version is available" banner from mixing into parsed output.
    result = subprocess.run(
        ["brev", *args, "--no-check-latest"],
        capture_output=True,
        text=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if check and result.returncode != 0:
        raise BrevError(
            f"brev {' '.join(args)} failed ({result.returncode}): {(result.stderr or result.stdout)[-800:]}"
        )
    return result


def _json_from(output: str):
    # Tolerate any banner text before the JSON document.
    start = min((i for i in (output.find("["), output.find("{")) if i >= 0), default=-1)
    if start < 0:
        raise BrevError(f"no JSON in brev output: {output[:300]!r}")
    return json.loads(output[start:])


def list_instances(retries: int = 6) -> list[dict]:
    """`brev ls --json`: dicts with name, id, status, build_status, shell_status, instance_type, gpu.

    Retries with backoff: the Brev API sometimes answers "resource exhausted" for a minute or so, and the
    launcher's cleanup depends on this call.
    """
    for attempt in range(retries):
        try:
            out = brev("ls", "--json", timeout=60).stdout
            break
        except (BrevError, subprocess.TimeoutExpired):
            if attempt == retries - 1:
                raise
            time.sleep(min(5 * 2**attempt, 60))
    if not out.strip() or "no instances" in out.lower():
        return []
    data = _json_from(out)
    return data if isinstance(data, list) else []


def get_instance(name: str) -> dict | None:
    return next((i for i in list_instances() if i.get("name") == name), None)


def price_per_hour(instance_type: str) -> float | None:
    """Hourly price of an instance type from `brev search`, or None if it isn't listed right now."""
    try:
        rows = _json_from(brev("search", "--json", timeout=180).stdout)
    except (BrevError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None
    return next((r.get("price_per_hour") for r in rows if r.get("type") == instance_type), None)


def ssh(host: str, command: str, *, input: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    """Run one command on an instance. Never raises on a non-zero exit; check returncode."""
    try:
        return subprocess.run(
            ["ssh", *SSH_OPTS, host, command],
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=None if input is not None else subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, 255, "", f"ssh timed out after {timeout}s")
