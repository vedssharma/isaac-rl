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
