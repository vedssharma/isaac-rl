#!/usr/bin/env python3
"""Run one training job on a Brev GPU instance from your laptop (PLAN.md §3.2).

    python scripts/brev/launch.py --run-name p0-quadcopter-launch-s42 -- \\
        --task Isaac-Quadcopter-Direct-v0 --num_envs 4096 --seed 42

Everything after `--` goes to scripts/train.py. The launcher adds `--headless`, `--logger wandb` and
`--run_name <run name>` itself. One run is:

1. Preflight: keys in .env, `brev` signed in, the commit to run is pushed to GitHub.
2. `brev create isaac-rl-<run name>-<UTC yymmddHHMM>`, trying each of --types in order.
   With `--box <name>` it instead starts (if needed) and reuses an existing instance, such as the dev box.
3. Over ssh: clone this repo at that exact commit and start scripts/brev/run_remote.sh in the background.
   It installs Isaac Sim / Isaac Lab if needed (about 5 min on a 16-vCPU box) and runs train.py.
   Secrets go over ssh as environment variables and are never written to the instance's disk.
4. Streams the remote log to the terminal and to logs/launch/<run>/run.log, until training ends or the
   time limits are hit.
5. Checks that the W&B run is "finished" and that the final checkpoint is on W&B and on Hugging Face.
   If it can't confirm that, it copies the run's log directory (checkpoints included) back to the laptop.
6. Deletes the instance (or stops a reused --box), even after an error or Ctrl-C, and records the box
   time and cost in logs/launch/<run>/launch.json and in the W&B run summary under launch/*.

Exit code: 0 only if training succeeded and the checkpoint was verified.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import string
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import brevcli
from brevcli import BrevError, brev, get_instance, ssh

REPO_ROOT = Path(__file__).resolve().parents[2]
# The first type that has capacity wins. Nebius L40S came up in ~3 min and installed in ~5 min (Phase 0).
DEFAULT_TYPES = "gpu-l40s-d.1gpu-16vcpu-96gb,l40s-48gb.1x,g6e.2xlarge"
REQUIRED_ENV = ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HF_TOKEN", "HF_REPO_ID")
OPTIONAL_ENV = ("WANDB_BASE_URL",)  # a self-hosted / dedicated W&B server; unset means wandb.ai
REMOTE_ROOT = "$HOME/launch"  # on the instance: launch/isaac-rl (the clone) and launch/runs/<run>/
POLL_S = 20
# --quiet prints only lines containing one of these; the full log is always saved locally.
QUIET_MARKERS = (
    "[run_remote]", "Learning iteration", "Mean reward", "Mean episode length", "Iteration time",
    "Training time", "[INFO]", "[WARN]", "Error", "Traceback", "wandb:",
)  # fmt: skip


def log(msg: str) -> None:
    print(f"[launch {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- arguments


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        i = argv.index("--")
        ours, train_args = argv[:i], argv[i + 1 :]
    else:
        ours, train_args = argv, []
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", required=True, help="p{phase}-{desc}-s{seed} (PLAN.md §7); lowercase and hyphens")
    p.add_argument("--box", help="reuse this existing instance (started if stopped, stopped again at the end)")
    p.add_argument(
        "--types", default=os.environ.get("BREV_TYPES", DEFAULT_TYPES), help="instance types to try, in order"
    )
    p.add_argument("--ref", default="HEAD", help="git commit/branch to run; must be pushed (default: HEAD)")
    p.add_argument(
        "--allow-dirty", action="store_true", help="run even with uncommitted changes (they are NOT included)"
    )
    p.add_argument("--boot-timeout-min", type=float, default=12, help="per instance type, to reach RUNNING (12)")
    p.add_argument("--max-setup-min", type=float, default=40, help="limit for the install (default 40)")
    p.add_argument("--max-train-min", type=float, default=120, help="limit for train.py (default 120)")
    p.add_argument("--keep", action="store_true", help="don't delete/stop the instance at the end")
    p.add_argument("--quiet", action="store_true", help="print only progress lines; the full log is still saved")
    p.add_argument("--dry-run", action="store_true", help="run the preflight checks and print the plan, nothing else")
    p.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    args = p.parse_args(ours)
    return args, train_args


def flag_value(args: list[str], flag: str) -> str | None:
    """The value of `--flag v` or `--flag=v` in an argument list."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def build_train_args(train_args: list[str], run_name: str) -> list[str]:
    """Add the flags every launched run needs, and refuse ones that would break verification."""
    args = list(train_args)

    def value_of(flag: str) -> str | None:
        return flag_value(args, flag)

    if value_of("--task") is None:
        raise SystemExit("train.py needs --task, e.g.: launch.py --run-name ... -- --task Isaac-Quadcopter-Direct-v0")
    if value_of("--run_name") not in (None, run_name):
        raise SystemExit("don't pass --run_name to train.py; the launcher sets it from --run-name")
    if value_of("--logger") not in (None, "wandb"):
        raise SystemExit("launched runs log to W&B (the launcher verifies the run there); drop --logger")
    if "--headless" not in args:
        args.append("--headless")
    if value_of("--logger") is None:
        args += ["--logger", "wandb"]
    if value_of("--run_name") is None:
        args += ["--run_name", run_name]
    return args


# ---------------------------------------------------------------- preflight


def load_env(env_file: str) -> None:
    path = Path(env_file)
    if path.exists():
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k, "").strip()]
    if missing:
        raise SystemExit(f"missing in {env_file} / the environment: {', '.join(missing)} (see .env.example)")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def resolve_commit(ref: str, allow_dirty: bool) -> tuple[str, str]:
    """(sha, https clone URL). The instance clones from GitHub, so the commit must be pushed."""
    if git("status", "--porcelain") and not allow_dirty:
        raise SystemExit("uncommitted changes; commit and push them (or pass --allow-dirty to run without them)")
    sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
    git("fetch", "--quiet", "origin")
    if not git("branch", "-r", "--contains", sha):
        raise SystemExit(f"{sha[:10]} isn't on GitHub yet; push it first (the instance clones from origin)")
    url = git("remote", "get-url", "origin")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    return sha, url


# ---------------------------------------------------------------- the run


class Launch:
    def __init__(self, args: argparse.Namespace, train_args: list[str], sha: str, clone_url: str):
        self.args = args
        self.train_args = train_args
        self.sha = sha
        self.clone_url = clone_url
        now = datetime.now(UTC)
        self.reuse = bool(args.box)
        self.name = args.box or brevcli.run_box_name(args.run_name, now)
        self.tag = f"{args.run_name}-{now.strftime(brevcli.STAMP_FORMAT)}"  # unique per launch
        self.remote_run_dir = f"{REMOTE_ROOT}/runs/{self.tag}"
        self.local_dir = REPO_ROOT / "logs" / "launch" / self.tag
        self.wandb_run_id = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        self.box_up_at: float | None = None
        self.started = False
        self.failed_boxes: list[str] = []  # instances from failed create attempts, already deleted
        self.exit_code: int | None = None
        self.record: dict = {
            "run_name": args.run_name,
            "instance": self.name,
            "reused_box": self.reuse,
            "git_sha": sha,
            "train_args": train_args,
            "wandb_run_id": self.wandb_run_id,
            "launched_at": now.isoformat(timespec="seconds"),
        }

    def save_record(self) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        (self.local_dir / "launch.json").write_text(json.dumps(self.record, indent=2) + "\n")

    # -- 2. instance

    def provision(self) -> None:
        if self.reuse:
            inst = get_instance(self.name)
            if inst is None:
                raise SystemExit(f"no instance named {self.name} (brev ls)")
            self.box_up_at = time.time()
            if inst.get("status") != "RUNNING":
                log(f"starting {self.name} (status {inst.get('status')})")
                # `brev start` can block for a long time (or exit early) while the provider finds a GPU;
                # either way the status poll below decides.
                try:
                    brev("start", self.name, timeout=120, check=False)
                except subprocess.TimeoutExpired:
                    pass
            inst = self.wait_running(self.args.max_setup_min * 60)
            if inst is None:
                raise BrevError(f"{self.name} didn't reach RUNNING")
        else:
            inst = self.create()
        self.record["instance"] = self.name
        self.record["instance_type"] = inst.get("instance_type")
        self.record["gpu"] = inst.get("gpu")
        log(f"{self.name} is RUNNING ({inst.get('instance_type')}, {inst.get('gpu')}); waiting for ssh")
        brev("refresh", timeout=180)  # writes this instance's entry in ~/.brev/ssh_config
        deadline = time.time() + 10 * 60
        while ssh(self.name, "true", timeout=60).returncode != 0:
            if time.time() > deadline:
                raise BrevError(f"no ssh to {self.name} after 10 min")
            time.sleep(POLL_S)
        log("ssh is up")

    def create(self) -> dict:
        """Try each instance type in turn, one instance at a time, until one reaches RUNNING.

        `brev create --type a,b,c` only falls back when the create call itself fails. An instance that is
        accepted and then goes to FAILURE (seen on Nebius) would otherwise be waited on until the time limit.
        """
        types = [t.strip() for t in self.args.types.split(",") if t.strip()]
        per_type_s = self.args.boot_timeout_min * 60
        tried: set[str] = set()
        for i, itype in enumerate(types):
            # Each attempt gets its own name: a failed instance can take 10+ minutes to finish deleting.
            name = brevcli.run_box_name(self.args.run_name)
            while name in tried or get_instance(name) is not None:
                time.sleep(15)
                name = brevcli.run_box_name(self.args.run_name)
            tried.add(name)
            self.name = name
            log(f"creating {name} as {itype} ({i + 1}/{len(types)})")
            if self.box_up_at is None:
                self.box_up_at = time.time()  # billing can start before create returns
            result = brev(
                "create", name, "--type", itype, "--timeout", str(int(per_type_s)),
                timeout=per_type_s + 300, check=False,
            )  # fmt: skip
            if result.returncode == 0:
                inst = self.wait_running(per_type_s)
                if inst is not None:
                    return inst
            else:
                log(f"brev create failed: {(result.stderr or result.stdout).strip()[-300:]}")
            self.failed_boxes.append(name)
            if get_instance(name) is not None:
                log(f"deleting the failed instance {name}")
                brev("delete", name, timeout=300, check=False)
        raise BrevError(f"none of the instance types came up: {', '.join(types)}")

    def wait_running(self, timeout_s: float) -> dict | None:
        """The instance once it is RUNNING; None if it fails, disappears or runs out of time."""
        deadline = time.time() + timeout_s
        last = None
        while time.time() < deadline:
            inst = get_instance(self.name)
            status = inst.get("status") if inst else None
            if status != last:
                log(f"{self.name}: {status}")
                last = status
            if status == "RUNNING":
                return inst
            if status in (None, "FAILURE", "DELETING") or (status == "STOPPING" and not self.reuse):
                return None
            time.sleep(POLL_S)
        log(f"{self.name} not RUNNING after {timeout_s / 60:.0f} min")
        return None

    # -- 3. start the job

    def start_remote(self) -> None:
        q = shlex.quote
        env = {k: os.environ[k] for k in REQUIRED_ENV + OPTIONAL_ENV if os.environ.get(k)}
        env["WANDB_RUN_ID"] = self.wandb_run_id  # so we can find and verify the run on W&B afterwards
        exports = "\n".join(f"export {k}={q(v)}" for k, v in env.items())
        clone = f"{REMOTE_ROOT}/isaac-rl"
        script = f"""set -euo pipefail
{exports}
command -v git >/dev/null || {{ sudo apt-get -o DPkg::Lock::Timeout=900 update -qq && sudo apt-get -o DPkg::Lock::Timeout=900 install -y -qq git; }}
if pgrep -f '[s]cripts/train.py' >/dev/null; then echo 'another train.py is already running here' >&2; exit 4; fi
mkdir -p {self.remote_run_dir}
[ -d {clone}/.git ] || git clone --quiet {q(self.clone_url)} {clone}
git -C {clone} fetch --quiet origin
git -C {clone} checkout --quiet --force --detach {self.sha}
git -C {clone} clean -fdq
cd {clone}
nohup setsid bash scripts/brev/run_remote.sh {self.remote_run_dir} {int(self.args.max_train_min)} \\
    {" ".join(q(a) for a in self.train_args)} > {self.remote_run_dir}/run.log 2>&1 < /dev/null &
echo "started pid $!"
"""
        result = ssh(self.name, "bash -s", input=script, timeout=300)
        if result.returncode != 0:
            raise BrevError(f"starting the job failed ({result.returncode}): {result.stderr[-800:]}")
        self.started = True
        self.record["started_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        self.save_record()
        log(f"{result.stdout.strip()} on {self.name}; W&B run id {self.wandb_run_id}")

    # -- 4. follow

    def follow(self) -> int | None:
        """Stream the log until run_remote.sh writes exit_code. Returns it, or None if the limits are hit."""
        local_log = self.local_dir / "run.log"
        offset = 0
        failures = 0
        # Setup and training have separate limits; the remote `timeout` enforces the training one, this is a backstop.
        deadline = time.time() + (self.args.max_setup_min + self.args.max_train_min + 10) * 60
        cmd = (
            f'cd {self.remote_run_dir} && echo "$(cat phase 2>/dev/null)/$(cat exit_code 2>/dev/null)" '
            f"&& tail -c +{{start}} run.log 2>/dev/null"
        )
        with open(local_log, "ab") as out:
            while True:
                try:
                    r = subprocess.run(
                        ["ssh", *brevcli.SSH_OPTS, self.name, cmd.format(start=offset + 1)],
                        capture_output=True, timeout=120, stdin=subprocess.DEVNULL, check=False,
                    )  # fmt: skip
                    ok = r.returncode == 0
                except subprocess.TimeoutExpired:
                    ok = False
                if not ok:
                    failures += 1
                    if failures % 5 == 0:
                        status = (get_instance(self.name) or {}).get("status")
                        log(f"ssh failed {failures} times in a row; instance status {status}")
                        if status != "RUNNING":
                            raise BrevError(f"{self.name} is {status}; giving up on this run")
                else:
                    failures = 0
                    head, _, chunk = r.stdout.partition(b"\n")
                    phase, _, code = head.decode().strip().partition("/")
                    if chunk:
                        offset += len(chunk)
                        out.write(chunk)
                        out.flush()
                        self.echo(chunk.decode(errors="replace"))
                    if code.strip():
                        self.record["remote_phase"] = phase
                        return int(code)
                if time.time() > deadline:
                    log("time limit reached; stopping training")
                    return None
                time.sleep(POLL_S)

    def echo(self, text: str) -> None:
        if not self.args.quiet:
            sys.stdout.write(text)
        else:
            for line in text.splitlines(keepends=True):
                if any(m in line for m in QUIET_MARKERS):
                    sys.stdout.write(line)
        sys.stdout.flush()

    def interrupt_remote(self, wait_s: float = 180) -> None:
        """SIGINT train.py so it marks the W&B run failed and uploads what it has, then wait for it to exit."""
        if not self.started:
            return
        # Signal only run_remote.sh's `timeout`: it passes SIGINT on to train.py once (and SIGKILLs it if it
        # hangs). Signalling python directly as well would interrupt its W&B cleanup a second time.
        ssh(self.name, "pkill -INT -f '^timeout .*[s]cripts/train.py' || true", timeout=60)
        end = time.time() + wait_s
        while time.time() < end:
            r = ssh(self.name, f"cat {self.remote_run_dir}/exit_code", timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                self.exit_code = int(r.stdout.strip())
                return
            time.sleep(10)

    # -- 5. verify

    def verify(self) -> bool:
        """The W&B run finished, and the final checkpoint is on W&B and (unless upload was off) on HF."""
        import wandb
        from huggingface_hub import HfApi

        path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{self.wandb_run_id}"
        try:
            run = wandb.Api(timeout=60).run(path)
        except Exception as e:  # noqa: BLE001
            log(f"verify: W&B run {path} not found: {e}")
            return False
        self.record["wandb_url"] = run.url
        ckpt, hf_url = run.summary.get("final_checkpoint"), run.summary.get("hf_url")
        self.record["final_checkpoint"], self.record["hf_url"] = ckpt, hf_url
        problems = []
        if run.state != "finished":
            problems.append(f"W&B run state is {run.state!r}")
        if not ckpt:
            problems.append("no final_checkpoint in the W&B summary")
        elif not any(f.name == ckpt for f in run.files()):
            problems.append(f"{ckpt} isn't in the W&B run's files")
        if flag_value(self.train_args, "--hf_upload") == "none":
            pass  # upload turned off on purpose
        elif not hf_url:
            problems.append("no hf_url in the W&B summary (HF upload failed?)")
        elif ckpt:
            path_in_repo = hf_url.split("/tree/main/", 1)[-1]
            api = HfApi(token=os.environ["HF_TOKEN"])
            if not api.file_exists(os.environ["HF_REPO_ID"], f"{path_in_repo}/{ckpt}", repo_type="model"):
                problems.append(f"{path_in_repo}/{ckpt} isn't on HF")
        for p in problems:
            log(f"verify: {p}")
        if not problems:
            log(f"verified: {ckpt} is on W&B ({run.url}) and HF ({hf_url})")
        return not problems

    def pull_run_dir(self) -> bool:
        """Copy the run's train.py log dir (checkpoints, params, videos) back to logs/launch/<run>/."""
        run_name = shlex.quote(self.args.run_name)
        find = f"ls -dt {REMOTE_ROOT}/isaac-rl/logs/rsl_rl/*/*_{run_name} 2>/dev/null | head -1"
        r = ssh(self.name, find, timeout=60)
        remote = r.stdout.strip()
        if r.returncode != 0 or not remote:
            log("pull: no train.py log dir found on the instance")
            return False
        dest = self.local_dir / "run_dir"
        dest.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(
            f"ssh {' '.join(brevcli.SSH_OPTS)} {shlex.quote(self.name)} "
            f"'tar czf - -C {shlex.quote(remote)} .' | tar xzf - -C {shlex.quote(str(dest))}",
            shell=True, capture_output=True, text=True, timeout=1800, check=False,
        )  # fmt: skip
        if tar.returncode != 0:
            log(f"pull failed: {tar.stderr[-400:]}")
            return False
        log(f"copied {remote} to {dest}")
        return True

    # -- 6. teardown

    def teardown(self, keep_for_safety: bool) -> None:
        if self.args.keep or keep_for_safety:
            why = "--keep" if self.args.keep else "the checkpoint couldn't be verified or copied back"
            log(f"leaving {self.name} running ({why}). It bills until you run: brev delete {self.name}")
            self.record["teardown"] = "kept"
            return
        if get_instance(self.name) is None:
            self.record["teardown"] = "already gone"
            return
        action, gone = ("stop", "STOPPED") if self.reuse else ("delete", None)
        # An instance that is still STARTING refuses `brev stop`/`delete`; keep trying until it accepts.
        end = time.time() + 25 * 60
        while True:
            log(f"brev {action} {self.name}")
            try:
                result = brev(action, self.name, timeout=300, check=False)
                ok, err = result.returncode == 0, (result.stderr or result.stdout).strip()[-300:]
            except subprocess.TimeoutExpired:
                ok, err = False, "timed out"
            if ok:
                break
            status = (get_instance(self.name) or {}).get("status")
            if time.time() > end:
                log(f"brev {action} still failing (status {status}); RUN IT BY HAND: brev {action} {self.name}")
                self.record["teardown"] = f"{action} failed"
                return
            log(f"brev {action} failed while {status} ({err}); retrying in 30 s")
            time.sleep(30)
        self.record["teardown_issued_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        # Nebius can sit in STOPPING/DELETING for 10+ minutes; confirm before saying it's done.
        end = time.time() + 25 * 60
        while time.time() < end:
            inst = get_instance(self.name)
            status = inst.get("status") if inst else None
            if status == gone:
                log(f"{self.name} is {gone or 'deleted'}")
                self.record["teardown"] = "stopped" if self.reuse else "deleted"
                return
            time.sleep(30)
        log(f"{self.name} is still {status} after 25 min; check `brev ls` later")
        self.record["teardown"] = f"{action} pending ({status})"

    def check_failed_boxes(self) -> None:
        for name in self.failed_boxes:
            inst = get_instance(name)
            if inst is not None and inst.get("status") != "DELETING":
                log(f"failed attempt {name} is still {inst.get('status')}; deleting it again")
                brev("delete", name, timeout=300, check=False)
        self.record["failed_attempts"] = self.failed_boxes

    def account(self) -> None:
        """Box time × hourly price into launch.json and the W&B summary (launch/*)."""
        if self.box_up_at is None:
            return
        hours = (time.time() - self.box_up_at) / 3600
        price = brevcli.price_per_hour(self.record.get("instance_type") or "")
        self.record["box_hours"] = round(hours, 3)
        self.record["price_per_hour"] = price
        self.record["cost_usd"] = round(hours * price, 2) if price is not None else None
        log(f"box time {hours * 60:.0f} min" + (f", about ${hours * price:.2f}" if price else ""))
        if not self.started:
            return  # train.py never ran, so there is no W&B run to annotate
        try:
            import wandb

            run = wandb.Api(timeout=60).run(
                f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/{self.wandb_run_id}"
            )
            keys = ("instance", "instance_type", "box_hours", "price_per_hour", "cost_usd", "git_sha")
            run.summary.update({f"launch/{k}": self.record.get(k) for k in keys})
        except Exception as e:  # noqa: BLE001
            log(f"couldn't add launch/* to the W&B summary: {e}")

    def safely(self, what: str, fn, *args, default=None, **kwargs):
        """Call fn; on any error, log it and return `default` instead of raising."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            log(f"{what} failed: {type(e).__name__}: {e}")
            return default

    def run(self) -> int:
        verified = pulled = False
        try:
            self.provision()
            self.start_remote()
            self.exit_code = self.follow()
            if self.exit_code is None:
                self.interrupt_remote()
        except KeyboardInterrupt:
            log("Ctrl-C: stopping training and cleaning up (Ctrl-C again skips the cleanup)")
            self.interrupt_remote()
        except (BrevError, subprocess.SubprocessError, OSError) as e:
            log(f"error: {e}")
        finally:
            # Each step is isolated: a failure in one (e.g. the Brev API refusing calls for a minute) must not
            # skip the teardown or the record.
            self.record["exit_code"] = self.exit_code
            if self.exit_code == 0:
                verified = bool(self.safely("verify", self.verify))
            if not verified and self.started and self.safely("instance lookup", get_instance, self.name):
                pulled = bool(self.safely("copy back", self.pull_run_dir))
            self.record["verified"] = verified
            self.record["pulled_run_dir"] = pulled
            self.save_record()
            # Keep the box only if training worked but its checkpoint exists nowhere else.
            keep = self.exit_code == 0 and not verified and not pulled
            if self.safely("teardown", self.teardown, keep_for_safety=keep, default=False) is False:
                self.record["teardown"] = "error"
                log(f"!! couldn't confirm {self.name} is gone. Check `brev ls` and run: brev delete {self.name}")
            self.safely("failed-attempt cleanup", self.check_failed_boxes)
            self.safely("cost accounting", self.account)
            self.record["ended_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            self.save_record()
        log(f"exit code {self.exit_code}, verified {verified}; record in {self.local_dir / 'launch.json'}")
        return 0 if self.exit_code == 0 and verified else 1


def main(argv: list[str]) -> int:
    args, train_args = parse_args(argv)
    try:
        brevcli.run_box_name(args.run_name)  # validates the name early
    except ValueError as e:
        raise SystemExit(str(e))
    train_args = build_train_args(train_args, args.run_name)
    load_env(args.env_file)
    sha, url = resolve_commit(args.ref, args.allow_dirty)
    brev("ls")  # signed in?
    launch = Launch(args, train_args, sha, url)
    log(f"run {launch.tag}: {sha[:10]} on {launch.name}{' (reused)' if launch.reuse else ''}")
    log(f"train.py {' '.join(train_args)}")
    if args.dry_run:
        log("dry run; nothing was created")
        return 0
    return launch.run()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
