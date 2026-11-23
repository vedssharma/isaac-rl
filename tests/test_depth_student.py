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


# -- the recurrent student


def small_recurrent(**kw):
    return ds.RecurrentDepthStudent(latent_dim=8, hidden_dims=(32,), channels=(4, 8, 8, 8), gru_hidden=16, **kw)


def test_recurrent_step_by_step_matches_the_sequence_with_resets():
    torch.manual_seed(0)
    student = small_recurrent()
    T, B = 6, 3
    obs = torch.rand(T, B, P + H * W)
    starts = torch.zeros(T, B, dtype=torch.bool)
    starts[3, 1] = True  # env 1 starts a new episode at step 3
    h0 = torch.randn(B, 16)
    seq = student.forward_sequence(obs, h0, starts)
    h, outs = h0, []
    for t in range(T):
        h = torch.where(starts[t][:, None], torch.zeros_like(h), h)
        a, h = student.step(obs[t], h)
        outs.append(a)
    torch.testing.assert_close(seq, torch.stack(outs))
    # The reset really cuts the memory: env 1 from step 3 on doesn't depend on what came before.
    other = obs.clone()
    other[:3, 1] = torch.rand(3, P + H * W)
    seq2 = student.forward_sequence(other, h0, starts)
    torch.testing.assert_close(seq2[3:, 1], seq[3:, 1])
    assert not torch.allclose(seq2[:3, 1], seq[:3, 1])


def test_recurrent_act_keeps_memory_and_reset_clears_it():
    torch.manual_seed(0)
    student = small_recurrent()
    a, b = torch.rand(2, P + H * W), torch.rand(2, P + H * W)
    first = student.act(a)
    second = student.act(b)
    fresh = small_recurrent()
    fresh.load_state_dict(student.state_dict())
    assert not torch.allclose(second, fresh.act(b))  # b after a differs from b alone
    student.reset(torch.tensor([True, False]))
    third = student.act(a)
    fresh.reset()  # forget everything
    torch.testing.assert_close(third[0], fresh.act(a)[0])  # env 0 starts over
    assert first.shape == (2, 4)
    with pytest.raises(TypeError):
        student(a)


def test_recurrent_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    student = small_recurrent()
    student.update_normalizer(torch.randn(50, P + H * W))
    torch.save(ds.checkpoint(student, 3), tmp_path / "s.pt")
    loaded = ds.load_student(torch.load(tmp_path / "s.pt", map_location="cpu", weights_only=False))
    assert isinstance(loaded, ds.RecurrentDepthStudent) and loaded.gru_hidden == 16
    obs, starts = torch.rand(4, 2, P + H * W), torch.zeros(4, 2, dtype=torch.bool)
    h0 = loaded.initial_state(2)
    expected = student.eval().forward_sequence(obs, h0, starts)
    torch.testing.assert_close(loaded.forward_sequence(obs, h0, starts), expected)
    assert not ds.load_student(ds.checkpoint(ds.DepthStudent(), 0)).is_recurrent


def test_only_the_recurrent_student_can_use_a_cue_it_no_longer_sees():
    """A cue in the first frame of each sequence (a bright or a dark image) decides the target action at every later
    step, when the image is blank. With memory the student learns it; without, the best it can do is the mean."""
    torch.manual_seed(0)
    T, B = 6, 64
    cue = torch.randint(0, 2, (B,)).float()
    obs = torch.zeros(T, B, P + H * W)
    obs[0, :, P:] = cue[:, None]
    target = (cue * 2 - 1)[None, :, None].expand(T, B, 4) * 0.5
    starts = torch.zeros(T, B, dtype=torch.bool)
    starts[0] = True
    gru = small_recurrent()
    opt = torch.optim.Adam(gru.parameters(), lr=3e-3)
    for _ in range(150):
        out = gru.forward_sequence(obs, gru.initial_state(B), starts)
        loss = ds.imitation_loss(out[1:], target[1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.02  # remembered
    mlp = ds.DepthStudent(latent_dim=8, hidden_dims=(32,), channels=(4, 8, 8, 8))
    opt = torch.optim.Adam(mlp.parameters(), lr=3e-3)
    for _ in range(150):
        loss = ds.imitation_loss(mlp(obs[1:].reshape(-1, P + H * W)), target[1:].reshape(-1, 4))
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() > 0.2  # the blank frames carry no cue: about 0.25, the variance of the target
