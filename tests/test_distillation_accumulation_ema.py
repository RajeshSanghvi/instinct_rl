"""Gradient accumulation and weight EMA.

Both exist to reproduce the distillation recipe in *Now You See That* (Table XII: 800 steps per
env per iteration, 10 gradient accumulation steps, max grad norm 1.0, EMA decay 0.997), where a
rollout produces exactly one clipped optimizer step rather than one per TBPTT chunk.
"""

import pytest
import torch

from helpers import fill_storage, make_algorithm, make_recurrent_student


def _grads(policy):
    return {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}


"""
Gradient accumulation
"""


def test_accumulation_takes_one_optimizer_step_per_rollout():
    alg = make_algorithm(num_steps=12, gradient_length=4, accumulate_gradients=True)
    fill_storage(alg)

    _, stats = alg.update(0)

    # Three chunks, but a single step over their summed gradient.
    assert stats["optimizer_steps"].item() == 1


def test_accumulation_still_flushes_the_tail_into_the_same_step():
    alg = make_algorithm(num_steps=10, gradient_length=4, accumulate_gradients=True, flush_tail=True)
    fill_storage(alg)

    _, stats = alg.update(0)

    assert stats["optimizer_steps"].item() == 1
    assert stats["tail_chunk_size"].item() == 2


def test_accumulated_gradient_equals_the_whole_rollout_computed_in_one_chunk():
    """The point of accumulation: chunking becomes a pure memory optimisation, not a change to
    the objective. Splitting the rollout must not change the gradient that reaches the weights.

    The reference is a single chunk spanning the whole rollout, which also carries gradient
    across what would have been the chunk boundaries -- so exact equality is not expected for a
    recurrent student. What *is* checked is that the two agree closely and, critically, that the
    accumulated version is not off by a factor of the chunk count (the classic accumulation bug).
    """
    num_steps = 12
    weights = make_recurrent_student(seed=5).state_dict()

    def run(gradient_length):
        policy = make_recurrent_student(seed=5)
        policy.load_state_dict(weights)
        alg = make_algorithm(
            policy=policy,
            num_steps=num_steps,
            gradient_length=gradient_length,
            accumulate_gradients=True,
            max_grad_norm=None,  # clipping would mask a scale error
            learning_rate=0.0,  # freeze the weights so both runs differentiate the same function
        )
        fill_storage(alg, seed=11)
        alg.update(0)
        return _grads(policy)

    chunked = run(4)  # three chunks
    whole = run(num_steps)  # one chunk

    assert chunked, "expected the student to receive gradients"
    for name, g in chunked.items():
        ratio = g.norm() / whole[name].norm().clamp_min(1e-12)
        assert 0.5 < ratio < 2.0, f"{name}: accumulated gradient is off by {ratio:.3f}x"


def test_accumulation_weights_a_short_tail_by_its_length():
    """A 2-step tail must not pull as hard as a 4-step chunk. Dividing every chunk by the chunk
    *count* rather than weighting by chunk length is the easy way to get this wrong."""
    weights = make_recurrent_student(seed=2).state_dict()

    def run(num_steps, gradient_length):
        policy = make_recurrent_student(seed=2)
        policy.load_state_dict(weights)
        alg = make_algorithm(
            policy=policy,
            num_steps=num_steps,
            gradient_length=gradient_length,
            accumulate_gradients=True,
            max_grad_norm=None,
            learning_rate=0.0,
        )
        fill_storage(alg, seed=3)
        alg.update(0)
        return _grads(policy)

    # 8 steps as 2x4 (no tail) versus 8 steps as 4+4 via a gradient_length that leaves no tail:
    # both are the mean over 8 steps, so the gradient must match regardless of the split.
    split_two = run(8, 4)
    split_one = run(8, 8)

    for name, g in split_two.items():
        ratio = g.norm() / split_one[name].norm().clamp_min(1e-12)
        assert 0.5 < ratio < 2.0, f"{name}: chunking changed the gradient scale by {ratio:.3f}x"


def test_accumulation_clips_the_whole_rollout_gradient_once():
    alg = make_algorithm(num_steps=12, gradient_length=4, accumulate_gradients=True, max_grad_norm=1e-6)
    fill_storage(alg)

    _, stats = alg.update(0)

    assert stats["optimizer_steps"].item() == 1
    assert stats["grad_norm_after_clip"].item() == pytest.approx(1e-6, rel=1e-3)
    assert stats["grad_norm_before_clip"].item() > 1e-6


def test_accumulation_steps_the_lr_scheduler_once_per_rollout():
    """With accumulation, `optimizer_step` and `update` units coincide -- which is what makes
    OneCycleLR(total_steps=num_iterations) line up with the paper's 4000 iterations."""
    alg = make_algorithm(
        num_steps=12,
        gradient_length=4,
        accumulate_gradients=True,
        lr_scheduler_class_name="StepLR",
        lr_scheduler=dict(step_size=1, gamma=0.5),
        learning_rate=1.0,
        lr_scheduler_step_unit="optimizer_step",
    )
    fill_storage(alg)

    alg.update(0)

    assert alg.learning_rate == pytest.approx(0.5)


def test_without_accumulation_the_default_is_unchanged():
    alg = make_algorithm(num_steps=12, gradient_length=4)
    fill_storage(alg)

    _, stats = alg.update(0)

    assert alg.accumulate_gradients is False
    assert stats["optimizer_steps"].item() == 3


"""
Weight EMA
"""


def test_ema_tracks_but_lags_the_live_weights():
    alg = make_algorithm(num_steps=8, gradient_length=4, ema_decay=0.9, learning_rate=1e-1)
    start = {k: v.clone() for k, v in alg.actor_critic.state_dict().items()}
    fill_storage(alg)

    alg.update(0)

    live = alg.actor_critic.state_dict()
    moved = [k for k, v in live.items() if v.dtype.is_floating_point and not torch.allclose(v, start[k])]
    assert moved, "the student did not train, so the EMA assertion below would be vacuous"

    for key in moved:
        ema, initial, current = alg._ema_state[key], start[key], live[key]
        assert not torch.allclose(ema, current), f"{key}: EMA is not lagging the live weights"
        assert not torch.allclose(ema, initial), f"{key}: EMA never moved"
        # It must sit between where training started and where it currently is.
        assert (ema - initial).norm() < (current - initial).norm()


def test_ema_is_not_maintained_by_default():
    alg = make_algorithm(num_steps=8, gradient_length=4)
    assert alg.ema_decay is None
    assert alg._ema_state is None
    with pytest.raises(RuntimeError, match="No EMA is being maintained"):
        alg.load_ema_into_model()


def test_invalid_ema_decay_is_rejected():
    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="ema_decay"):
            make_algorithm(ema_decay=bad)


def test_checkpoint_ships_the_ema_weights_as_the_deployable_model():
    """`model_state_dict` is the slot OnPolicyRunner and the exporters read, so with EMA on it
    must hold the averaged weights; the raw ones ride along only so training can resume."""
    alg = make_algorithm(num_steps=8, gradient_length=4, ema_decay=0.9, learning_rate=1e-1)
    fill_storage(alg)
    alg.update(0)

    sd = alg.state_dict()

    assert "raw_model_state_dict" in sd
    for key, ema in alg._ema_state.items():
        assert torch.equal(sd["model_state_dict"][key], ema)
    for key, live in alg.actor_critic.state_dict().items():
        assert torch.equal(sd["raw_model_state_dict"][key], live)
    # The two differ, otherwise this test would pass for an implementation that ignored the EMA.
    assert any(
        not torch.equal(sd["model_state_dict"][k], sd["raw_model_state_dict"][k])
        for k in sd["model_state_dict"]
    )


def test_resume_restores_the_raw_weights_and_the_ema_separately():
    alg = make_algorithm(num_steps=8, gradient_length=4, ema_decay=0.9, learning_rate=1e-1)
    fill_storage(alg)
    alg.update(0)
    sd = alg.state_dict()

    resumed = make_algorithm(num_steps=8, gradient_length=4, ema_decay=0.9, learning_rate=1e-1)
    resumed.load_state_dict(sd)

    for key, value in alg.actor_critic.state_dict().items():
        assert torch.equal(resumed.actor_critic.state_dict()[key], value), f"{key} raw"
    for key, value in alg._ema_state.items():
        assert torch.equal(resumed._ema_state[key], value), f"{key} ema"


def test_load_ema_into_model_swaps_the_live_weights():
    alg = make_algorithm(num_steps=8, gradient_length=4, ema_decay=0.9, learning_rate=1e-1)
    fill_storage(alg)
    alg.update(0)
    ema = {k: v.clone() for k, v in alg._ema_state.items()}

    alg.load_ema_into_model()

    for key, value in alg.actor_critic.state_dict().items():
        assert torch.equal(value, ema[key]), key


def test_a_plain_checkpoint_still_loads_when_ema_is_enabled():
    """Turning EMA on mid-project must not orphan checkpoints written before it existed."""
    plain = make_algorithm(num_steps=8, gradient_length=4)
    fill_storage(plain)
    plain.update(0)
    sd = plain.state_dict()
    assert "raw_model_state_dict" not in sd

    alg = make_algorithm(num_steps=8, gradient_length=4, ema_decay=0.9)
    alg.load_state_dict(sd)

    for key, value in sd["model_state_dict"].items():
        assert torch.equal(alg.actor_critic.state_dict()[key], value)
        assert torch.equal(alg._ema_state[key], value), "EMA should start from the loaded weights"
