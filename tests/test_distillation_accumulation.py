"""Gradient accumulation: chunk boundaries, tail flushing, and step-size invariance."""

import pytest
import torch

from helpers import fill_storage, make_algorithm, make_recurrent_student


def _grads(policy):
    return {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}


def test_tail_chunk_is_flushed_when_rollout_is_not_a_multiple_of_gradient_length():
    alg = make_algorithm(num_steps=10, gradient_length=4, flush_tail=True)
    fill_storage(alg)

    _, stats = alg.update(0)

    # 4 + 4 + 2: the two leftover steps get their own optimizer step
    assert stats["optimizer_steps"].item() == 3
    assert stats["tail_chunk_size"].item() == 2


def test_without_flush_tail_the_leftover_steps_produce_no_gradient():
    """Documents the failure mode this flag exists to prevent: 20% of the rollout is logged
    into the loss but silently never backpropagated."""
    alg = make_algorithm(num_steps=10, gradient_length=4, flush_tail=False)
    fill_storage(alg)

    _, stats = alg.update(0)

    assert stats["optimizer_steps"].item() == 2
    assert stats["tail_chunk_size"].item() == 0


def test_exact_multiple_leaves_no_tail():
    alg = make_algorithm(num_steps=12, gradient_length=4)
    fill_storage(alg)

    _, stats = alg.update(0)

    assert stats["optimizer_steps"].item() == 3
    assert stats["tail_chunk_size"].item() == 0


def test_summing_a_chunk_scales_the_gradient_by_the_chunk_length():
    """Chunk-mean keeps the effective step size independent of gradient_length; chunk-sum does
    not. Both algorithms see identical weights and data, so the ratio must be exactly G."""
    num_steps = 6
    weights = make_recurrent_student(seed=3).state_dict()

    def run(normalize):
        policy = make_recurrent_student(seed=3)
        policy.load_state_dict(weights)
        alg = make_algorithm(
            policy=policy,
            num_steps=num_steps,
            gradient_length=num_steps,
            normalize_accumulated_loss=normalize,
            max_grad_norm=None,  # do not let clipping hide the difference
        )
        fill_storage(alg, seed=7)
        alg.update(0)
        return _grads(policy)

    mean_grads = run(True)
    sum_grads = run(False)

    assert mean_grads, "expected the student to receive gradients"
    for name, mean_grad in mean_grads.items():
        assert torch.allclose(sum_grads[name], mean_grad * num_steps, atol=1e-5), name


def test_logged_behavior_loss_covers_every_timestep_including_the_tail():
    alg = make_algorithm(num_steps=10, gradient_length=4, flush_tail=True)
    storage = fill_storage(alg)

    # Reference: mean per-step loss over the whole rollout, computed before any weight update.
    policy = make_recurrent_student(seed=0)
    policy.load_state_dict(alg.actor_critic.state_dict())
    policy.set_actor_hidden_state(None)
    with torch.no_grad():
        step_losses = []
        for obs, label, dones in storage.iter_timesteps():
            step_losses.append(alg._behavior_loss(policy.act_inference(obs), label))
            policy.mask_actor_hidden_state(dones)
        expected = torch.stack(step_losses).mean()

    losses, _ = alg.update(0)

    # The first chunk is computed with the initial weights, later chunks with updated ones, so
    # this is a coarse check that nothing was dropped rather than an exact equality.
    assert losses["behavior_loss"].item() == pytest.approx(expected.item(), rel=0.5)
    assert losses["behavior_loss"].item() > 0


def test_storage_is_cleared_after_update():
    alg = make_algorithm(num_steps=8, gradient_length=4)
    fill_storage(alg)
    assert len(alg.storage) == 8

    alg.update(0)

    assert len(alg.storage) == 0


def test_lr_scheduler_steps_once_per_optimizer_step_by_default():
    alg = make_algorithm(
        num_steps=12,
        gradient_length=4,
        lr_scheduler_class_name="StepLR",
        lr_scheduler=dict(step_size=1, gamma=0.5),
        learning_rate=1.0,
        lr_scheduler_step_unit="optimizer_step",
    )
    fill_storage(alg)

    alg.update(0)

    # three optimizer steps -> 1.0 * 0.5^3
    assert alg.learning_rate == pytest.approx(0.125)


def test_lr_scheduler_can_step_once_per_update_instead():
    alg = make_algorithm(
        num_steps=12,
        gradient_length=4,
        lr_scheduler_class_name="StepLR",
        lr_scheduler=dict(step_size=1, gamma=0.5),
        learning_rate=1.0,
        lr_scheduler_step_unit="update",
    )
    fill_storage(alg)

    alg.update(0)

    assert alg.learning_rate == pytest.approx(0.5)


def test_invalid_scheduler_step_unit_is_rejected():
    with pytest.raises(ValueError, match="lr_scheduler_step_unit"):
        make_algorithm(lr_scheduler_step_unit="epoch")
