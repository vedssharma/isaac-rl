"""CPU tests for Drone-Nav-Goal-v0's MDP math (drone_nav/mdp/goal.py)."""

import math

import pytest
import torch
from drone_nav.mdp import goal as mdp

WEIGHTS = {
    "progress": 1.0,
    "goal": 10.0,
    "crash": 10.0,
    "smooth": 0.05,
    "ang_rate": 0.01,
    "upright": 0.5,
    "time": 0.1,
    "overspeed": 2.0,
}
LIMIT = 3.0
DT = 0.02


def test_goal_offsets_respect_distance_and_altitude_ranges():
    g = torch.Generator().manual_seed(0)
    goals = mdp.sample_goal_offsets(10_000, 10.0, 20.0, (0.75, 2.5), generator=g)
    horiz = goals[:, :2].norm(dim=-1)
    assert goals.shape == (10_000, 3)
    assert horiz.min() >= 10.0 - 1e-4 and horiz.max() <= 20.0 + 1e-4
    assert goals[:, 2].min() >= 0.75 and goals[:, 2].max() <= 2.5
    # Headings cover all four quadrants roughly equally.
    quadrant = (goals[:, 0] > 0).long() * 2 + (goals[:, 1] > 0).long()
    counts = torch.bincount(quadrant, minlength=4).float() / 10_000
    assert torch.all((counts - 0.25).abs() < 0.03)


def test_goal_observation_is_unit_direction_plus_clipped_distance():
    obs = mdp.goal_observation(torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 20.0], [0.0, 0.0, 0.0]]), dist_clip=10.0)
    torch.testing.assert_close(obs[0], torch.tensor([0.6, 0.8, 0.0, 0.5]))
    torch.testing.assert_close(obs[1], torch.tensor([0.0, 0.0, 1.0, 1.0]))  # 20 m clipped to 10 m
    assert torch.isfinite(obs[2]).all()  # at the goal: no division by zero


def test_up_z():
    gravity_b = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(mdp.up_z(gravity_b), torch.tensor([1.0, -1.0, 0.0]))


def terminations(dist, start, alt, upz):
    t = lambda v: torch.tensor(v, dtype=torch.float)
    return mdp.compute_terminations(
        t(dist), t(start), t(alt), t(upz), goal_radius=0.5, min_altitude=0.1, max_altitude=4.0, off_course_margin=5.0
    )


@pytest.mark.parametrize(
    ("dist", "alt", "upz", "start", "expected"),
    [
        (0.4, 1.0, 1.0, 15.0, "goal"),
        (5.0, 0.05, 1.0, 15.0, "too_low"),
        (5.0, 4.5, 1.0, 15.0, "too_high"),
        (5.0, 1.0, -0.2, 15.0, "flipped"),
        (20.1, 1.0, 1.0, 15.0, "off_course"),
        (5.0, 1.0, 1.0, 15.0, None),
        (0.4, 0.05, 1.0, 15.0, "too_low"),  # a crash at the goal is not a success
    ],
)
def test_terminations(dist, alt, upz, start, expected):
    t = terminations([dist], [start], [alt], [upz])
    fired = [k for k, v in t.items() if v.item()]
    assert fired == ([expected] if expected else [])


def rewards(**overrides):
    n = 1
    args = {
        "prev_dist": torch.tensor([10.0]),
        "dist": torch.tensor([9.9]),
        "reached": torch.tensor([False]),
        "crashed": torch.tensor([False]),
        "action": torch.zeros(n, 4),
        "prev_action": torch.zeros(n, 4),
        "ang_vel_b": torch.zeros(n, 3),
        "upz": torch.tensor([1.0]),
        "speed": torch.tensor([2.0]),
        "speed_limit": LIMIT,
        "step_dt": DT,
        "weights": WEIGHTS,
    }
    args.update(overrides)
    return mdp.compute_reward_terms(**args)


def test_reward_terms_are_all_present_and_signed():
    terms = rewards(
        reached=torch.tensor([True]),
        crashed=torch.tensor([True]),
        action=torch.ones(1, 4),
        ang_vel_b=torch.ones(1, 3),
        upz=torch.tensor([0.5]),
        speed=torch.tensor([4.0]),
    )
    assert tuple(terms) == mdp.REWARD_TERMS
    assert terms["progress"].item() == pytest.approx(0.1, rel=1e-5)  # float32
    assert terms["goal"].item() == 10.0
    assert terms["crash"].item() == -10.0
    assert terms["smooth"].item() == pytest.approx(-0.05 * 4 * DT)
    assert terms["ang_rate"].item() == pytest.approx(-0.01 * 3 * DT)
    assert terms["upright"].item() == pytest.approx(-0.5 * 0.5 * DT)
    assert terms["time"].item() == pytest.approx(-0.1 * DT)
    assert terms["overspeed"].item() == pytest.approx(-2.0 * 1.0**2 * DT)  # 1 m/s over the limit


def test_progress_sums_to_distance_closed():
    """Telescoping: an episode's progress reward is (start distance - final distance) x weight."""
    path = torch.linspace(15.0, 0.4, 500)
    total = sum(rewards(prev_dist=path[i : i + 1], dist=path[i + 1 : i + 2])["progress"].item() for i in range(499))
    assert total == pytest.approx(15.0 - 0.4, abs=1e-3)


def test_flying_to_the_goal_beats_hovering_and_crashing():
    """Over a 20 s episode, flying 15 m to the goal in 10 s must out-earn hovering and an early crash."""
    steps_fly, steps_ep = 500, 1000

    def episode(prog_per_step, steps, reached=False, crashed=False):
        r = rewards(prev_dist=torch.tensor([prog_per_step]), dist=torch.tensor([0.0]))
        dense = sum(v.item() for k, v in r.items() if k not in ("goal", "crash")) * steps
        return dense + 10.0 * reached - 10.0 * crashed

    fly = episode(15.0 / steps_fly, steps_fly, reached=True)
    hover = episode(0.0, steps_ep)
    crash = episode(0.0, 50, crashed=True)
    assert fly > hover > crash
    assert math.isclose(fly, 15.0 + 10.0 - 0.1 * 10.0, rel_tol=1e-3)


def test_no_overspeed_penalty_at_or_below_the_limit():
    for v in (0.0, 2.9, LIMIT):
        assert rewards(speed=torch.tensor([v]))["overspeed"].item() == 0.0


def test_best_cruise_speed_is_just_above_the_limit():
    """Per second, flying straight at speed v earns v (progress) minus the over-speed penalty. Its maximum is at
    v = limit + 1 / (2 x weight): 3.25 m/s with the default weights, far below the 10 m/s of the uncapped baseline."""
    speeds = torch.linspace(0.0, 10.0, 1001)

    def per_second(v):
        terms = rewards(prev_dist=torch.tensor([v * DT]), dist=torch.tensor([0.0]), speed=torch.tensor([v]))
        return (terms["progress"] + terms["overspeed"]).item() / DT

    best = speeds[torch.tensor([per_second(v.item()) for v in speeds]).argmax()].item()
    assert best == pytest.approx(LIMIT + 1 / (2 * WEIGHTS["overspeed"]), abs=0.02)


def test_drag_coefficient_gives_the_requested_top_speed():
    weight, ttw, v_top = 0.27, 1.9, 5.0
    k = mdp.linear_drag_coefficient(weight, ttw, v_top)
    # At v_top, drag balances the largest horizontal thrust that still holds altitude.
    max_horizontal = math.sqrt((ttw * weight) ** 2 - weight**2)
    assert k * v_top == pytest.approx(max_horizontal)
    drag = mdp.drag_force(torch.tensor([[v_top, 0.0, 0.0]]), k)
    torch.testing.assert_close(drag, torch.tensor([[-max_horizontal, 0.0, 0.0]]))


def test_drag_opposes_velocity_in_any_frame():
    v = torch.tensor([[1.0, -2.0, 0.5]])
    torch.testing.assert_close(mdp.drag_force(v, 0.1), -0.1 * v)
    torch.testing.assert_close(mdp.drag_force(torch.zeros(1, 3), 0.1), torch.zeros(1, 3))
