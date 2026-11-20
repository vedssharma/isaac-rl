"""CPU tests for the Phase 3 depth student and DAgger helpers (drone_nav/policies/depth_student.py)."""

import pytest
import torch
from drone_nav.policies import depth_student as ds

P, H, W = 18, 48, 64


def test_running_norm_matches_the_batch_statistics():
    torch.manual_seed(0)
    x = torch.randn(10_000, 3) * torch.tensor([1.0, 5.0, 0.1]) + torch.tensor([2.0, -1.0, 0.0])
    norm = ds.RunningNorm(3, eps=0.0)
    for chunk in torch.split(x, 777):  # uneven batches merge to the same result
        norm.update(chunk)
    torch.testing.assert_close(norm.mean, x.mean(0), atol=1e-5, rtol=0)
    torch.testing.assert_close(norm.var, x.var(0, unbiased=False), atol=1e-4, rtol=1e-4)
    out = norm(x)
    torch.testing.assert_close(out.mean(0), torch.zeros(3), atol=1e-4, rtol=0)
    torch.testing.assert_close(out.std(0, unbiased=False), torch.ones(3), atol=1e-3, rtol=0)


def test_student_shapes_and_split():
    student = ds.DepthStudent()
    obs = torch.rand(5, P + H * W)
    assert student(obs).shape == (5, 4)
    proprio, image = student.split(obs)
    assert proprio.shape == (5, P) and image.shape == (5, 1, H, W)
    # Row-major: pixel (r, c) of env 0 is input P + r * W + c.
    assert image[0, 0, 2, 3] == obs[0, P + 2 * W + 3]
    with pytest.raises(ValueError):
        student(torch.rand(5, P + H * W - 1))


def test_student_uses_the_image():
    torch.manual_seed(0)
    student = ds.DepthStudent()
    obs = torch.rand(2, P + H * W)
    other = obs.clone()
    other[:, P:] = torch.rand(2, H * W)
    assert not torch.allclose(student(obs), student(other))


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    student = ds.DepthStudent(latent_dim=32, hidden_dims=(64, 64))
    student.update_normalizer(torch.randn(100, P + H * W))
    path = tmp_path / "student.pt"
    torch.save(ds.checkpoint(student, 7, {"teacher": "run/abc"}), path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ds.is_student_checkpoint(ckpt) and ckpt["iter"] == 7 and ckpt["teacher"] == "run/abc"
    loaded = ds.load_student(ckpt)
    obs = torch.rand(3, P + H * W)
    torch.testing.assert_close(loaded(obs), student.eval()(obs))
    assert not ds.is_student_checkpoint({"model_state_dict": {}})  # an RSL-RL checkpoint


def test_teacher_fraction_schedule():
    assert ds.teacher_fraction(0, 100) == 1.0
    assert ds.teacher_fraction(50, 100) == 0.5
    assert ds.teacher_fraction(100, 100) == ds.teacher_fraction(500, 100) == 0.0
    assert ds.teacher_fraction(0, 0) == 0.0


def test_mix_actions():
    g = torch.Generator().manual_seed(0)
    t, s = torch.ones(10_000, 4), -torch.ones(10_000, 4)
    assert torch.equal(ds.mix_actions(t, s, 1.0), t) and torch.equal(ds.mix_actions(t, s, 0.0), s)
    m = ds.mix_actions(t, s, 0.3, generator=g)
    assert (m == m[:, :1]).all()  # whole actions, not mixed per dimension
    assert abs((m[:, 0] == 1).float().mean().item() - 0.3) < 0.02


def test_minibatches_cover_every_sample_once():
    g = torch.Generator().manual_seed(0)
    parts = list(ds.minibatches(103, 4, "cpu", generator=g))
    assert len(parts) == 4 and sorted(torch.cat(parts).tolist()) == list(range(103))


def test_imitation_loss_targets_the_executed_action():
    s = torch.tensor([[1.0, 0.0, -1.0, 0.5]])
    t = torch.tensor([[1.7, 0.0, -3.0, 0.5]])  # the env clamps to [-1, 1]
    assert ds.imitation_loss(s, t).item() == 0.0


def test_a_few_steps_of_imitation_fit_a_fixed_teacher():
    """The whole training step on CPU: the student should drive the loss well down on a small fixed dataset."""
    torch.manual_seed(0)
    student = ds.DepthStudent(latent_dim=16, hidden_dims=(64, 64), channels=(8, 16, 16, 16))
    obs = torch.rand(256, P + H * W)
    target = torch.tanh(obs[:, :4] * 2 - 1 + obs[:, P : P + 4])  # depends on proprio and image
    student.update_normalizer(obs)
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)
    first = ds.imitation_loss(student(obs), target).item()
    for _ in range(300):
        for idx in ds.minibatches(len(obs), 4, "cpu"):
            loss = ds.imitation_loss(student(obs[idx]), target[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    assert ds.imitation_loss(student(obs), target).item() < 0.1 * first
