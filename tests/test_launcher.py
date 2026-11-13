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
