"""Behaviour-cloning loss semantics and the student-only rollout invariant."""

import pytest
import torch

from helpers import NUM_ACTIONS, NUM_ENVS, STUDENT_OBS_DIM, TEACHER_OBS_DIM, make_algorithm


def test_mse_sum_sums_over_action_dim_and_means_over_envs():
    alg = make_algorithm(loss_type="mse_sum")
    student = torch.zeros(NUM_ENVS, NUM_ACTIONS)
    teacher = torch.ones(NUM_ENVS, NUM_ACTIONS) * 2.0

    # per env: sum_j (0 - 2)^2 = 4 * NUM_ACTIONS ; mean over envs leaves it unchanged
    assert alg._behavior_loss(student, teacher).item() == pytest.approx(4.0 * NUM_ACTIONS)


def test_mse_sum_differs_from_mse_by_exactly_the_action_dimension():
    student = torch.randn(NUM_ENVS, NUM_ACTIONS)
    teacher = torch.randn(NUM_ENVS, NUM_ACTIONS)

    summed = make_algorithm(loss_type="mse_sum")._behavior_loss(student, teacher)
    meaned = make_algorithm(loss_type="mse")._behavior_loss(student, teacher)

    assert summed.item() == pytest.approx(meaned.item() * NUM_ACTIONS, rel=1e-6)


def test_unknown_loss_type_is_rejected():
    alg = make_algorithm(loss_type="not_a_loss")
    with pytest.raises(ValueError, match="Unknown loss_type"):
        alg._behavior_loss(torch.zeros(2, NUM_ACTIONS), torch.zeros(2, NUM_ACTIONS))


def test_environment_always_executes_the_student_action():
    """The core invariant: the teacher labels, it never drives the environment."""
    alg = make_algorithm()
    alg.actor_critic.std.data.fill_(0.0)  # make sampling deterministic so we can compare exactly

    obs = torch.randn(NUM_ENVS, STUDENT_OBS_DIM)
    critic_obs = torch.randn(NUM_ENVS, TEACHER_OBS_DIM)

    executed = alg.act(obs, critic_obs)
    teacher_label = alg._pending_teacher_mean

    assert executed.shape == (NUM_ENVS, NUM_ACTIONS)
    assert not torch.allclose(executed, teacher_label), "executed action must not be the teacher's"
    assert alg.teacher.act_calls == 1


def test_stored_label_is_the_teacher_mean_not_the_executed_action():
    alg = make_algorithm()
    obs = torch.randn(NUM_ENVS, STUDENT_OBS_DIM)
    critic_obs = torch.randn(NUM_ENVS, TEACHER_OBS_DIM)

    executed = alg.act(obs, critic_obs)
    expected_label = alg._pending_teacher_mean.clone()
    alg.process_env_step(
        rewards=torch.zeros(NUM_ENVS, 1),
        dones=torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
        infos={},
        next_obs=obs,
        next_critic_obs=critic_obs,
    )

    assert torch.equal(alg.storage.teacher_action_means[0], expected_label)
    assert torch.equal(alg.storage.student_observations[0], obs)
    assert not torch.allclose(alg.storage.teacher_action_means[0], executed)


def test_missing_critic_obs_fails_loudly():
    alg = make_algorithm(teacher_obs_source="critic")
    with pytest.raises(ValueError, match="no critic observations"):
        alg.act(torch.randn(NUM_ENVS, STUDENT_OBS_DIM), None)
