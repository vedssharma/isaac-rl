"""CPU tests for the per-env success curriculum (drone_nav/mdp/curriculum.py), PLAN.md §5.3."""

import pytest
import torch
from drone_nav.mdp.curriculum import SuccessCurriculum, level_fractions

ALL = torch.arange(4)


def make(**kw):
    args = dict(num_envs=4, num_levels=10, window=5, up_rate=0.8, down_rate=0.4)
    args.update(kw)
    return SuccessCurriculum(**args)


def run(cur, outcomes, levels):
    """Feed one episode per env per row of outcomes; return the levels after each row."""
    history = []
    for row in outcomes:
        levels = cur.update(ALL, torch.tensor(row), levels)
        history.append(levels.tolist())
    return history


def test_no_move_until_the_window_is_full():
    cur = make()
    levels = torch.full((4,), 3)
    hist = run(cur, [[True, False, True, False]] * 4, levels)
    assert all(h == [3, 3, 3, 3] for h in hist)


def test_moves_up_at_80_percent_and_down_below_40():
    cur = make()
    levels = torch.full((4,), 3)
    # Per env over 5 episodes: 5/5, 4/5 (= 80%: up), 2/5 (= 40%: stays), 1/5 (20%: down).
    outcomes = [
        [True, True, True, True],
        [True, True, True, False],
        [True, True, False, False],
        [True, False, False, False],
        [True, True, False, False],
    ]
    assert run(cur, outcomes, levels)[-1] == [4, 4, 3, 2]


def test_moving_clears_the_window():
    """After a move the env needs a full new window at the new level before it moves again."""
    cur = make()
    levels = torch.full((4,), 3)
    hist = run(cur, [[True] * 4] * 9, levels)
    assert hist[4] == [4] * 4  # up after 5 successes
    assert all(h == [4] * 4 for h in hist[5:9])  # 4 more successes at level 4: window not full yet
    assert run(cur, [[True] * 4], torch.tensor(hist[-1]))[-1] == [5] * 4  # the 5th one moves it again


def test_window_slides_without_a_move():
    """With no move, only the last `window` episodes count: old failures drop out."""
    cur = make(num_envs=1)
    levels = torch.tensor([3])
    one = torch.arange(1)
    for s in [False, False, True, True, True]:  # 3/5: stays
        levels = cur.update(one, torch.tensor([s]), levels)
    assert levels.tolist() == [3]
    for s in [True, True]:  # window is now F,T,T,T,T... then T,T,T,T,T
        levels = cur.update(one, torch.tensor([s]), levels)
    assert levels.tolist() == [4]


def test_levels_stay_in_range():
    cur = make(num_envs=2)
    two = torch.arange(2)
    levels = torch.tensor([9, 0])
    for _ in range(5):
        levels = cur.update(two, torch.tensor([True, False]), levels)
    assert levels.tolist() == [9, 0]  # top stays at the top (not sent to a random level); bottom stays at 0


def test_only_the_given_envs_are_touched():
    cur = make()
    some = torch.tensor([1, 3])
    levels = torch.tensor([3, 3])
    for _ in range(5):
        levels = cur.update(some, torch.tensor([True, True]), levels)
    assert levels.tolist() == [4, 4]
    # Envs 0 and 2 recorded nothing, so a single episode can't move them.
    assert cur.update(torch.tensor([0, 2]), torch.tensor([True, True]), torch.tensor([3, 3])).tolist() == [3, 3]


@pytest.mark.parametrize(("down", "up", "window"), [(0.8, 0.4, 5), (0.4, 0.4, 5), (0.4, 1.2, 5), (0.4, 0.8, 0)])
def test_rejects_bad_settings(down, up, window):
    with pytest.raises(ValueError):
        make(down_rate=down, up_rate=up, window=window)


def test_level_fractions():
    fr = level_fractions(torch.tensor([0, 0, 3, 9]), 10)
    assert fr.shape == (10,) and fr.sum().item() == pytest.approx(1.0)
    assert fr[0] == 0.5 and fr[3] == 0.25 and fr[9] == 0.25
