"""CPU tests for Phase 4's domain randomization pieces (drone_nav/mdp/randomization.py)."""

import math

import torch
from drone_nav.mdp import randomization as rnd


def test_sample_scale_range():
    g = torch.Generator().manual_seed(0)
    s = rnd.sample_scale(10_000, 0.2, "cpu", generator=g)
    assert s.min() >= 0.8 and s.max() <= 1.2 and abs(s.mean().item() - 1.0) < 0.01
    assert (rnd.sample_scale(5, 0.0, "cpu") == 1.0).all()


def test_first_order_lag_reaches_63_percent_after_one_time_constant():
    dt, tau = 0.01, torch.tensor([0.03, 0.0])
    alpha = rnd.lag_alpha(dt, tau)
    assert alpha[1] == 1.0  # tau 0: no lag
    x = torch.zeros(2, 3)
    for _ in range(3):  # 30 ms
        x = rnd.first_order_lag(x, torch.ones(2, 3), alpha)
    assert math.isclose(x[0, 0].item(), 1 - math.exp(-1), rel_tol=1e-5)
    assert (x[1] == 1.0).all()


def test_ou_gusts_have_the_requested_std_and_correlation_time():
    g = torch.Generator().manual_seed(0)
    x = torch.zeros(4000)
    xs = []
    for i in range(3000):  # 60 s at 50 Hz
        x = rnd.ou_step(x, 0.02, 2.0, 0.5, torch.randn(4000, generator=g))
        if i >= 1000:
            xs.append(x)
    xs = torch.stack(xs)
    assert abs(xs.std().item() - 0.5) < 0.03
    lag = 100  # 2 s = one correlation time: autocorrelation about e^-1
    corr = (xs[:-lag] * xs[lag:]).mean() / xs.var()
    assert abs(corr.item() - math.exp(-1)) < 0.05


def test_horizontal_wind():
    g = torch.Generator().manual_seed(0)
    w = rnd.horizontal_wind(10_000, 3.0, "cpu", generator=g)
    speed = w[:, :2].norm(dim=-1)
    assert (w[:, 2] == 0).all() and speed.max() <= 3.0 and abs(speed.mean().item() - 1.5) < 0.05
    assert abs(w[:, 0].mean().item()) < 0.05 and abs(w[:, 1].mean().item()) < 0.05  # no preferred direction


def test_world_to_body():
    # Yawed 90 degrees left: world +x is the body's -y; world +y is the body's +x.
    q = torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]] * 2)
    v = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(rnd.world_to_body(v, q), expected, atol=1e-6, rtol=0)
    ident, u = torch.tensor([[1.0, 0.0, 0.0, 0.0]]), torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(rnd.world_to_body(u, ident), u)


def test_action_delay():
    d = rnd.ActionDelay(n=3, dim=1, max_steps=2, device="cpu")
    steps = torch.tensor([0, 1, 2])
    outs = []
    for k in range(1, 5):
        d.push(torch.full((3, 1), float(k)))
        outs.append(d.delayed(steps)[:, 0].tolist())
    # env 0 sees each action at once; env 1 one step late; env 2 two steps late (zeros before the first actions).
    assert outs == [[1, 0, 0], [2, 1, 0], [3, 2, 1], [4, 3, 2]]
    d.reset(torch.tensor([2]))
    d.push(torch.full((3, 1), 5.0))
    assert d.delayed(steps)[:, 0].tolist() == [5, 4, 0]


def test_obs_noise_std_covers_the_goal_observation():
    assert len(rnd.OBS_NOISE_STD) == 18 and rnd.OBS_NOISE_STD[9:13] == (0.0,) * 4  # last action is exact
