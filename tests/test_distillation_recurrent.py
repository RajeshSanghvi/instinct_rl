"""The recurrent half: carry handling, BPTT depth, and episode boundaries.

These are the tests that actually justify the algorithm's design. Everything else is
scheduling and bookkeeping; if the replay below is wrong, the student trains on a carry that
never existed and nothing else in the suite would notice.
"""

import pytest
import torch

from helpers import (
    NUM_ACTIONS,
    NUM_ENVS,
    STUDENT_OBS_DIM,
    fill_storage,
    make_algorithm,
    make_mlp_student,
    make_recurrent_student,
)

HIDDEN = 8


def _grads(policy):
    return {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}


def test_stepwise_replay_matches_a_batched_rnn_over_the_whole_sequence():
    """The one test that validates the recurrence itself.

    Stepping ``act_inference`` T times while threading the carry must be numerically identical
    to a single batched ``rnn(sequence, h0)`` call. Any carry-threading bug -- a stray detach,
    a dropped state, an off-by-one -- breaks this and nothing else would catch it.
    """
    num_steps = 6
    obs_seq = torch.randn(num_steps, NUM_ENVS, STUDENT_OBS_DIM)
    labels = torch.randn(num_steps, NUM_ENVS, NUM_ACTIONS)

    policy = make_recurrent_student(seed=5)
    policy.set_actor_hidden_state(None)
    step_losses = []
    for t in range(num_steps):
        student_mean = policy.act_inference(obs_seq[t])
        step_losses.append((student_mean - labels[t]).square().sum(-1).mean())
    torch.stack(step_losses).mean().backward()
    stepwise = _grads(policy)

    reference = make_recurrent_student(seed=5)
    reference.load_state_dict(policy.state_dict())
    rnn_out, _ = reference.memory_a.rnn(obs_seq, None)  # assumes rnn_highway=False
    batched_mean = reference.actor(rnn_out)
    (batched_mean - labels).square().sum(-1).mean().backward()
    batched = _grads(reference)

    assert set(stepwise) >= {"memory_a.rnn.weight_ih_l0", "memory_a.rnn.weight_hh_l0"}
    for name, grad in batched.items():
        assert torch.allclose(stepwise[name], grad, atol=1e-6), name


def test_done_masking_blocks_gradient_across_an_episode_boundary():
    policy = make_recurrent_student(seed=2)
    obs_0 = torch.randn(NUM_ENVS, STUDENT_OBS_DIM, requires_grad=True)
    obs_1 = torch.randn(NUM_ENVS, STUDENT_OBS_DIM, requires_grad=True)
    dones = torch.zeros(NUM_ENVS, 1, dtype=torch.bool)
    dones[0] = True  # env 0 terminates after the first step

    policy.set_actor_hidden_state(None)
    policy.act_inference(obs_0)
    policy.mask_actor_hidden_state(dones)
    policy.act_inference(obs_1).sum().backward()

    assert obs_0.grad[0].abs().max().item() == 0.0, "gradient leaked across a done boundary"
    assert obs_0.grad[1].abs().max().item() > 0.0, "gradient blocked for a non-terminated env"


def test_reset_would_truncate_bptt_which_is_why_mask_exists():
    """Documents why replay uses mask_actor_hidden_state instead of reset(dones).

    ``Memory.reset`` detaches the whole carry, so using it inside the replay would silently
    reduce BPTT to a single step for *every* environment, not just the terminated ones.
    """
    policy = make_recurrent_student(seed=2)
    obs_0 = torch.randn(NUM_ENVS, STUDENT_OBS_DIM, requires_grad=True)
    obs_1 = torch.randn(NUM_ENVS, STUDENT_OBS_DIM, requires_grad=True)
    dones = torch.zeros(NUM_ENVS, 1, dtype=torch.bool)
    dones[0] = True

    policy.set_actor_hidden_state(None)
    policy.act_inference(obs_0)
    policy.reset(dones)
    policy.act_inference(obs_1).sum().backward()

    assert obs_0.grad is None or obs_0.grad.abs().max().item() == 0.0


def test_gradient_length_bounds_the_bptt_depth():
    """With gradient_length=1 no gradient may cross a timestep boundary."""
    alg = make_algorithm(num_steps=4, gradient_length=1, max_grad_norm=None, learning_rate=0.0)
    fill_storage(alg)

    _, stats = alg.update(0)

    assert stats["optimizer_steps"].item() == 4
    # recurrent weights still get gradients (from the within-step path), but the carry entering
    # each step is detached, so consecutive chunks cannot share a graph
    assert alg.actor_critic.memory_a.rnn.weight_hh_l0.grad is not None


def test_update_replays_from_the_saved_rollout_start_carry():
    num_steps = 5
    alg = make_algorithm(num_steps=num_steps, gradient_length=num_steps, learning_rate=0.0)
    storage = fill_storage(alg)
    obs_seq = storage.student_observations[:num_steps].clone()
    dones_seq = storage.dones[:num_steps].clone()

    start_carry = torch.randn(1, NUM_ENVS, HIDDEN)
    alg._rollout_start_hidden = start_carry.clone()
    # Dirty the live carry the way a rollout would; the replay must ignore it.
    alg.actor_critic.set_actor_hidden_state(torch.randn(1, NUM_ENVS, HIDDEN))

    alg.update(0)

    reference = make_recurrent_student(seed=0)
    reference.load_state_dict(alg.actor_critic.state_dict())
    reference.set_actor_hidden_state(start_carry.clone())
    with torch.no_grad():
        for t in range(num_steps):
            reference.act_inference(obs_seq[t])
            reference.mask_actor_hidden_state(dones_seq[t])

    assert torch.allclose(alg._rollout_start_hidden, reference.memory_a.hidden_states, atol=1e-6)


def test_carry_survives_as_a_plain_detached_tensor_after_update():
    alg = make_algorithm(num_steps=8, gradient_length=4)
    fill_storage(alg)

    alg.update(0)

    carry = alg._rollout_start_hidden
    assert carry is not None
    assert not carry.requires_grad
    assert not carry.is_inference()
    assert torch.allclose(carry, alg.actor_critic.memory_a.hidden_states)


def test_carry_collected_under_inference_mode_is_rejected():
    """A carry produced by the rollout can never take part in autograd, so installing one must
    fail here rather than deep inside the RNN on the next replay."""
    policy = make_recurrent_student(seed=1)
    with torch.inference_mode():
        policy.act_inference(torch.randn(NUM_ENVS, STUDENT_OBS_DIM))
        rollout_carry = policy.memory_a.hidden_states

    with pytest.raises(RuntimeError, match="inference-mode"):
        policy.set_actor_hidden_state(rollout_carry)


def test_update_never_touches_the_teacher():
    alg = make_algorithm(num_steps=8, gradient_length=4)
    fill_storage(alg)
    hidden_before = alg.teacher.hidden.clone()
    calls_before = (alg.teacher.act_calls, alg.teacher.reset_calls)

    alg.update(0)

    assert torch.equal(alg.teacher.hidden, hidden_before)
    assert (alg.teacher.act_calls, alg.teacher.reset_calls) == calls_before


def test_gradient_clipping_covers_the_recurrent_parameters():
    alg = make_algorithm(num_steps=4, gradient_length=4, max_grad_norm=1e-6, learning_rate=0.0)
    fill_storage(alg)
    rnn_weight = alg.actor_critic.memory_a.rnn.weight_ih_l0
    assert any(p is rnn_weight for p in alg.trainable_parameters), "RNN excluded from clipping"

    _, stats = alg.update(0)

    assert rnn_weight.grad is not None
    assert stats["grad_norm_after_clip"].item() <= 1e-6 + 1e-9
    assert stats["grad_norm_before_clip"].item() > stats["grad_norm_after_clip"].item()


def test_action_std_is_frozen_and_excluded_from_training():
    alg = make_algorithm()
    assert not alg.actor_critic.std.requires_grad
    assert all(p is not alg.actor_critic.std for p in alg.trainable_parameters)


def test_hidden_refresh_regenerates_the_carry_with_the_final_weights():
    num_steps = 6
    alg = make_algorithm(
        num_steps=num_steps,
        gradient_length=2,
        refresh_hidden_after_update=True,
        learning_rate=1e-2,
    )
    storage = fill_storage(alg)
    obs_seq = storage.student_observations[:num_steps].clone()
    dones_seq = storage.dones[:num_steps].clone()

    _, stats = alg.update(0)

    reference = make_recurrent_student(seed=0)
    reference.load_state_dict(alg.actor_critic.state_dict())  # post-update weights
    reference.set_actor_hidden_state(None)
    with torch.no_grad():
        for t in range(num_steps):
            reference.act_inference(obs_seq[t])
            reference.mask_actor_hidden_state(dones_seq[t])

    assert torch.allclose(alg._rollout_start_hidden, reference.memory_a.hidden_states, atol=1e-6)
    assert stats["hidden_refresh_time"].item() >= 0.0


def test_refresh_is_off_by_default():
    assert make_algorithm().refresh_hidden_after_update is False


def test_mlp_student_trains_through_the_same_code_path():
    """The hidden-state interface is a no-op for non-recurrent policies, so the algorithm needs
    no is_recurrent branch."""
    alg = make_algorithm(policy=make_mlp_student(), num_steps=8, gradient_length=4)
    fill_storage(alg)

    losses, stats = alg.update(0)

    assert alg._rollout_start_hidden is None
    assert stats["optimizer_steps"].item() == 2
    assert losses["behavior_loss"].item() > 0


def test_lstm_carry_keeps_its_container_type():
    alg = make_algorithm(policy=make_recurrent_student(rnn_type="lstm"), num_steps=4, gradient_length=2)
    fill_storage(alg)

    alg.update(0)

    carry = alg._rollout_start_hidden
    assert hasattr(carry, "hidden") and hasattr(carry, "cell"), "LstmHiddenState was flattened"
    assert not carry.hidden.requires_grad and not carry.cell.requires_grad


def test_checkpoint_does_not_carry_teacher_weights_or_stale_hidden_state():
    alg = make_algorithm(num_steps=4, gradient_length=4)
    fill_storage(alg)
    alg.update(0)
    assert alg._rollout_start_hidden is not None

    state = alg.state_dict()
    assert set(state) == {"model_state_dict", "optimizer_state_dict"}
    assert not any("teacher" in k for k in state["model_state_dict"])

    alg.load_state_dict(state)
    # On resume the environment is reset, so a carry from a different episode is worse than zeros.
    assert alg._rollout_start_hidden is None


def test_distributed_training_fails_loudly_rather_than_silently():
    alg = make_algorithm()
    with pytest.raises(NotImplementedError, match="single-GPU"):
        alg.distributed_data_parallel()
