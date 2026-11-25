"""CPU-only tests for the Brev launcher's pure logic (no brev, ssh or network calls)."""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "brev"))

import brevcli
import launch
import reap

NOW = datetime(2026, 11, 13, 15, 30, tzinfo=UTC)


def test_run_box_name_round_trips():
    name = brevcli.run_box_name("p0-quadcopter-launch-s42", NOW)
    assert name == "isaac-rl-p0-quadcopter-launch-s42-2611131530"
    assert brevcli.parse_run_box(name) == ("p0-quadcopter-launch-s42", NOW)


@pytest.mark.parametrize("bad", ["P0-Caps", "under_score", "trailing-", "-leading", "dou--ble", "", "x" * 60])
def test_run_box_name_rejects_bad_run_names(bad):
    with pytest.raises(ValueError):
        brevcli.run_box_name(bad, NOW)


@pytest.mark.parametrize("name", ["isaac-rl-video", "isaac-rl-dev", "other-p0-x-2611131530", "isaac-rl-p0-261113153"])
def test_hand_made_instances_are_not_run_boxes(name):
    assert brevcli.parse_run_box(name) is None


def test_reap_only_flags_old_run_boxes():
    instances = [
        {"name": "isaac-rl-video", "status": "STOPPED"},
        {"name": brevcli.run_box_name("p0-old-s1", NOW - timedelta(hours=7)), "status": "RUNNING"},
        {"name": brevcli.run_box_name("p0-new-s1", NOW - timedelta(hours=1)), "status": "RUNNING"},
    ]
    rows = reap.stale_run_boxes(instances, max_age_hours=6, now=NOW)
    assert [(i["name"].split("-")[3], stale) for i, _, stale in rows] == [("old", True), ("new", False)]


def test_build_train_args_adds_required_flags():
    args = launch.build_train_args(["--task", "Isaac-Quadcopter-Direct-v0", "--seed", "42"], "p0-x-s42")
    assert args == [
        "--task", "Isaac-Quadcopter-Direct-v0", "--seed", "42",
        "--headless", "--logger", "wandb", "--run_name", "p0-x-s42",
    ]  # fmt: skip


def test_build_train_args_keeps_matching_flags():
    given = ["--task=T", "--headless", "--logger=wandb", "--run_name", "p0-x-s42"]
    assert launch.build_train_args(given, "p0-x-s42") == given


@pytest.mark.parametrize(
    "given",
    [
        ["--num_envs", "16"],  # no task
        ["--task", "T", "--run_name", "other"],
        ["--task", "T", "--logger", "tensorboard"],
    ],
)
def test_build_train_args_refuses(given):
    with pytest.raises(SystemExit):
        launch.build_train_args(given, "p0-x-s42")


def test_parse_args_splits_on_double_dash():
    args, train = launch.parse_args(["--run-name", "p0-x-s1", "--quiet", "--", "--task", "T", "--quiet"])
    assert args.run_name == "p0-x-s1" and args.quiet
    assert train == ["--task", "T", "--quiet"]


def test_flag_value():
    assert launch.flag_value(["--hf_upload", "none"], "--hf_upload") == "none"
    assert launch.flag_value(["--hf_upload=none"], "--hf_upload") == "none"
    assert launch.flag_value(["--hf_upload"], "--hf_upload") is None


def test_create_falls_back_when_an_instance_goes_to_failure(monkeypatch):
    """Nebius once accepted a create and then left the instance in FAILURE; the next type must be tried."""
    states: dict[str, list[str]] = {}  # instance name -> statuses it will report, last one sticks
    calls = []

    def fake_brev(*args, **kwargs):
        calls.append(args)
        if args[0] == "create":
            name, itype = args[1], args[3]
            states[name] = ["DEPLOYING", "FAILURE"] if itype == "bad-type" else ["DEPLOYING", "RUNNING"]
        elif args[0] == "delete":
            states.pop(args[1], None)
        return launch.subprocess.CompletedProcess(args, 0, "", "")

    def fake_get_instance(name):
        if name not in states:
            return None
        seq = states[name]
        status = seq.pop(0) if len(seq) > 1 else seq[0]
        return {"name": name, "status": status, "instance_type": "t", "gpu": "L40S"}

    stamps = iter(range(10))
    monkeypatch.setattr(launch, "brev", fake_brev)
    monkeypatch.setattr(launch, "get_instance", fake_get_instance)
    monkeypatch.setattr(launch.time, "sleep", lambda s: None)
    monkeypatch.setattr(launch.brevcli, "run_box_name", lambda run, now=None: f"isaac-rl-{run}-261113150{next(stamps)}")

    args, _ = launch.parse_args(["--run-name", "p0-x-s1", "--types", "bad-type,good-type"])
    lnch = launch.Launch(args, launch.build_train_args(["--task", "T"], "p0-x-s1"), "sha", "url")
    inst = lnch.create()

    assert inst["status"] == "RUNNING"
    assert [c[3] for c in calls if c[0] == "create"] == ["bad-type", "good-type"]
    assert lnch.failed_boxes == ["isaac-rl-p0-x-s1-2611131501"]
    assert ("delete", "isaac-rl-p0-x-s1-2611131501") in [c[:2] for c in calls]
    assert lnch.name == "isaac-rl-p0-x-s1-2611131502"


def test_cleanup_survives_a_brev_api_outage(monkeypatch, tmp_path, capsys):
    """If every brev call fails during cleanup, run() must still finish, record it, and say what to delete."""

    def broken(*args, **kwargs):
        raise launch.BrevError("brev ls --json failed (1): resource exhausted")

    def failing_provision():
        raise launch.BrevError("brev create failed")

    monkeypatch.setattr(launch, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(launch, "get_instance", broken)
    monkeypatch.setattr(launch, "brev", broken)
    monkeypatch.setattr(launch.brevcli, "price_per_hour", lambda t: None)

    args, _ = launch.parse_args(["--run-name", "p0-x-s1"])
    lnch = launch.Launch(args, launch.build_train_args(["--task", "T"], "p0-x-s1"), "sha", "url")
    monkeypatch.setattr(lnch, "provision", failing_provision)
    lnch.box_up_at = 0.0

    assert lnch.run() == 1
    assert lnch.record["teardown"] == "error"
    assert f"brev delete {lnch.name}" in capsys.readouterr().out
    assert (tmp_path / "logs" / "launch" / lnch.tag / "launch.json").exists()


def test_play_args_need_a_task_and_run_headless():
    assert launch.build_play_args(["--task", "T", "--wandb_run", "abc"]) == [
        "--task",
        "T",
        "--wandb_run",
        "abc",
        "--headless",
    ]
    with pytest.raises(SystemExit):
        launch.build_play_args(["--wandb_run", "abc"])


def test_script_flag_defaults_to_train():
    assert launch.parse_args(["--run-name", "p1-x-s1"])[0].script == "train"
    assert launch.parse_args(["--run-name", "p1-x-s1", "--script", "play"])[0].script == "play"


def test_parse_eval_spec():
    assert launch.parse_eval_spec("eval") == ("eval", [])
    assert launch.parse_eval_spec("eval_mass1.2:env.mass_scale=1.2") == ("eval_mass1.2", ["env.mass_scale=1.2"])
    assert launch.parse_eval_spec("x:env.a=1,env.b=2") == ("x", ["env.a=1", "env.b=2"])
    # Hydra lists keep their commas; play.py flags pass straight through.
    assert launch.parse_eval_spec("x:--num_envs=1024,env.mix=[0.6,0.3,0.0,0.0],env.b=2") == (
        "x",
        ["--num_envs=1024", "env.mix=[0.6,0.3,0.0,0.0]", "env.b=2"],
    )
    for bad in ("", ":env.a=1", "bad name", "a/b"):
        with pytest.raises(SystemExit):
            launch.parse_eval_spec(bad)


def test_eval_flags_parse_and_repeat():
    args, _ = launch.parse_args(["--run-name", "p1-x-s1", "--eval", "eval", "--eval", "eval_m:env.mass_scale=1.2"])
    assert args.eval == ["eval", "eval_m:env.mass_scale=1.2"]
    assert args.eval_episodes == 1000


def test_start_remote_never_prints_secrets(monkeypatch, tmp_path, capsys):
    """The start script carries the API keys to the box over ssh's stdin; nothing the launcher prints may contain
    them. (A variable-shadowing bug once logged the whole script, keys included.)"""
    secrets = {"WANDB_API_KEY": "wandb-secret-123", "HF_TOKEN": "hf-secret-456"}
    for k in launch.REQUIRED_ENV:
        monkeypatch.setenv(k, secrets.get(k, f"value-of-{k}"))
    sent = {}

    def fake_ssh(host, command, *, input=None, timeout=120):
        sent["input"] = input
        return launch.subprocess.CompletedProcess(["ssh"], 0, "started pid 4242\n", "")

    monkeypatch.setattr(launch, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(launch, "ssh", fake_ssh)
    args, _ = launch.parse_args(["--run-name", "p0-x-s1"])
    lnch = launch.Launch(args, launch.build_train_args(["--task", "T"], "p0-x-s1"), "sha", "url")
    lnch.start_remote()

    out = capsys.readouterr().out
    assert all(v in sent["input"] for v in secrets.values())  # the box still gets them
    assert not any(v in out for v in secrets.values())
    assert "train.py: started pid 4242" in out
    assert "RUN_SCRIPT=scripts/train.py" in sent["input"]


def test_preflight_survives_a_brief_brev_api_outage(monkeypatch, capsys):
    """Several launchers starting together can hit the Brev API's rate limit; the preflight must retry, not die."""
    calls = []

    def flaky_brev(*args, **kwargs):
        calls.append(args)
        if len(calls) <= 2:
            raise launch.BrevError("brev ls failed (1): resource exhausted")
        return launch.subprocess.CompletedProcess(["brev"], 0, "[]", "")

    monkeypatch.setattr(launch.brevcli, "brev", flaky_brev)
    monkeypatch.setattr(launch.brevcli.time, "sleep", lambda s: None)
    monkeypatch.setattr(launch, "load_env", lambda path: None)
    monkeypatch.setattr(launch, "resolve_commit", lambda ref, dirty: ("0" * 40, "url"))
    assert launch.main(["--run-name", "p0-x-s1", "--dry-run", "--", "--task", "T"]) == 0
    assert len(calls) == 3


def test_evals_inherit_the_training_overrides():
    """An eval must run the env the policy was trained in: a 5-degree lidar run's checkpoint (234 inputs) failed to
    load in the default 10-degree env (126). The eval's own overrides win where they share a key."""
    train = launch.build_train_args(
        ["--task", "T", "--num_envs", "4096", "env.lidar.pattern_cfg.horizontal_res=5.0", "env.eval_terrain_level=2",
         "agent.seed=7"],
        "p2-x-s1",
    )  # fmt: skip
    got = launch.eval_play_overrides(train, ["--num_envs=1024", "env.eval_terrain_level=5"])
    assert got == [
        "env.lidar.pattern_cfg.horizontal_res=5.0",
        "agent.seed=7",
        "--num_envs=1024",
        "env.eval_terrain_level=5",
    ]
    assert launch.eval_play_overrides(["--task", "T", "--run_name", "a=b"], []) == []  # flag values aren't overrides


def test_hydra_key():
    assert launch.hydra_key("env.a.b=1") == "env.a.b"
    assert launch.hydra_key("+agent.x=[1,2]") == "agent.x"
    for not_override in ("--num_envs=1024", "T", "a=b", "env.a"):
        assert launch.hydra_key(not_override) is None


class _FakeRun:
    """A W&B run whose server summary lost keys that its wandb-summary.json file still has."""

    def __init__(self, summary: dict, saved: dict | None):
        self.summary, self._saved = dict(summary), saved

    def file(self, name):
        run = self

        class _File:
            def download(self, root, replace):
                if run._saved is None:
                    raise FileNotFoundError(name)
                Path(root, name).write_text(launch.json.dumps(run._saved))

        return _File()


def test_restore_summary_keys_from_the_summary_file():
    saved = {"final_checkpoint": "model_1499.pt", "hf_url": "https://hf/x", "training_time_s": 695.0}
    run = _FakeRun({"training_time_s": 1.0}, saved)
    keys = ("final_checkpoint", "hf_url", "training_time_s")
    assert launch.restore_summary_keys(run, keys) == ["final_checkpoint", "hf_url"]
    assert run.summary == {"final_checkpoint": "model_1499.pt", "hf_url": "https://hf/x", "training_time_s": 1.0}
    assert launch.restore_summary_keys(run, keys) == []  # nothing missing: no download
    # No file, or the file lacks them too: nothing restored, and no exception.
    assert launch.restore_summary_keys(_FakeRun({}, None), keys) == []
    assert launch.restore_summary_keys(_FakeRun({}, {"other": 1}), keys) == []


def test_summary_value_falls_back_to_the_summary_file():
    import wandb_summary

    saved = {"hf_url": "https://hf/x", "final_checkpoint": "x"}
    run = _FakeRun({"hf_url": None, "final_checkpoint": "model_9.pt"}, saved)
    assert wandb_summary.summary_value(run, "final_checkpoint") == "model_9.pt"  # the live summary wins
    assert wandb_summary.summary_value(run, "hf_url") == "https://hf/x"
    assert wandb_summary.summary_value(run, "missing") is None
    assert wandb_summary.summary_value(_FakeRun({}, None), "hf_url") is None  # no file: None, no exception


def test_unreachable_box_gives_up_after_the_limit_and_resets_on_contact():
    u = launch.Unreachable(limit_s=20 * 60)
    u.report(True, 0.0)
    assert not u.exceeded(0.0) and u.seconds(0.0) == 0.0
    # ssh starts failing at t = 100 s: the clock runs from the first failure, not from each one.
    for t in range(100, 100 + 20 * 60 + 1, 140):  # a failed poll every ~140 s (120 s timeout + 20 s sleep)
        u.report(False, float(t))
    assert u.seconds(1300.0) == 1200.0 and not u.exceeded(1300.0)
    assert u.exceeded(1301.0)
    # One answer resets it.
    u.report(True, 1302.0)
    assert not u.exceeded(5000.0)
    u.report(False, 5000.0)
    assert not u.exceeded(5000.0 + 20 * 60) and u.exceeded(5000.0 + 20 * 60 + 1)


def test_max_unreachable_min_default():
    args, _ = launch.parse_args(["--run-name", "p3-x-s1", "--", "--task", "T"])
    assert args.max_unreachable_min == 20


def test_follow_gives_up_on_a_box_that_stays_running_but_never_answers(monkeypatch, tmp_path):
    """The 2026-11-20 hang: every ssh poll times out while Brev keeps listing the box RUNNING. follow() must give up
    after --max-unreachable-min, long before its overall deadline (setup + train limits)."""
    import subprocess
    from types import SimpleNamespace

    clock = {"t": 0.0}
    monkeypatch.setattr(launch.time, "time", lambda: clock["t"])
    monkeypatch.setattr(launch.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))

    def hung_ssh(*args, **kwargs):
        clock["t"] += kwargs.get("timeout", 120)
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 120))

    monkeypatch.setattr(launch.subprocess, "run", hung_ssh)
    monkeypatch.setattr(launch, "get_instance", lambda name: {"status": "RUNNING"})
    monkeypatch.setattr(launch, "log", lambda msg: None)
    run = object.__new__(launch.Launch)
    run.args = SimpleNamespace(max_setup_min=40, max_train_min=120, max_unreachable_min=20, quiet=True)
    run.name, run.remote_run_dir, run.local_dir, run.record = "isaac-rl-x", "/r", tmp_path, {}
    with pytest.raises(launch.BrevError, match="hasn't answered ssh"):
        run.follow()
    assert 20 * 60 < clock["t"] < 25 * 60  # not the 170-minute deadline


def test_evals_after_a_play_run_use_its_checkpoint_and_overrides(tmp_path):
    from types import SimpleNamespace

    run = object.__new__(launch.Launch)
    run.args = SimpleNamespace(script="play", eval=["eval_mass:env.randomization.mass_spread=0.2"], eval_episodes=500)
    run.train_args = ["--task", "Drone-Nav-Depth-v0", "--wandb_run", "abc123", "--eval_name", "nominal",
                      "env.eval_terrain_level=5", "--headless"]  # fmt: skip
    run.wandb_run_id, run.remote_run_dir, run.local_dir, run.record = "launcher-id", "/r", tmp_path, {}
    started = []
    run.start_remote = lambda script, args, run_dir: started.append((script, args))
    run.follow = lambda run_dir, log_name: 0
    run.save_record = lambda: None
    launch_log = launch.log
    launch.log = lambda msg: None
    try:
        run.run_evals()
    finally:
        launch.log = launch_log
    (script, args), = started
    assert script == "play" and args[args.index("--wandb_run") + 1] == "abc123"  # the evaluated run, not the launcher's
    assert "env.eval_terrain_level=5" in args and "env.randomization.mass_spread=0.2" in args
    assert args[args.index("--eval_name") + 1] == "eval_mass" and run.record["evals"] == {"eval_mass": 0}


def test_benchmark_mean_ci():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import benchmark_table as bt

    m, h = bt.mean_ci([90.0, 92.0, 94.0])
    assert m == 92.0 and abs(h - 4.303 * 2.0 / 3**0.5) < 1e-9  # sd 2, t(2) = 4.303
    assert bt.mean_ci([5.0]) == (5.0, 0.0)
    assert bt.cell([90.0, 92.0, 94.0]) == "92.0 ± 5.0"
