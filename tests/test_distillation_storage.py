"""DistillationStorage: time ordering, capacity, and the reserved auxiliary fields."""

import pytest
import torch

from instinct_rl.storage.distillation_storage import DistillationStorage

NUM_ENVS = 3
NUM_STEPS = 4
OBS_DIM = 5
ACTION_DIM = 2


def make_storage():
    return DistillationStorage(
        NUM_ENVS,
        NUM_STEPS,
        student_obs_shape=(OBS_DIM,),
        actions_shape=(ACTION_DIM,),
        device="cpu",
    )


def test_iter_timesteps_preserves_rollout_order():
    storage = make_storage()
    for t in range(NUM_STEPS):
        storage.add(
            torch.full((NUM_ENVS, OBS_DIM), float(t)),
            torch.full((NUM_ENVS, ACTION_DIM), float(-t)),
            torch.zeros(NUM_ENVS, 1, dtype=torch.bool),
        )

    seen = [(obs[0, 0].item(), label[0, 0].item()) for obs, label, _ in storage.iter_timesteps()]

    assert seen == [(0.0, 0.0), (1.0, -1.0), (2.0, -2.0), (3.0, -3.0)]


def test_iter_timesteps_only_yields_filled_steps():
    storage = make_storage()
    storage.add(
        torch.zeros(NUM_ENVS, OBS_DIM), torch.zeros(NUM_ENVS, ACTION_DIM), torch.zeros(NUM_ENVS, 1)
    )

    assert len(list(storage.iter_timesteps())) == 1
    assert len(storage) == 1
    assert not storage.is_full


def test_overflow_is_an_error_not_a_silent_wrap():
    storage = make_storage()
    for _ in range(NUM_STEPS):
        storage.add(
            torch.zeros(NUM_ENVS, OBS_DIM), torch.zeros(NUM_ENVS, ACTION_DIM), torch.zeros(NUM_ENVS, 1)
        )
    assert storage.is_full

    with pytest.raises(AssertionError, match="overflow"):
        storage.add(
            torch.zeros(NUM_ENVS, OBS_DIM), torch.zeros(NUM_ENVS, ACTION_DIM), torch.zeros(NUM_ENVS, 1)
        )


def test_clear_resets_the_write_cursor():
    storage = make_storage()
    storage.add(
        torch.zeros(NUM_ENVS, OBS_DIM), torch.zeros(NUM_ENVS, ACTION_DIM), torch.zeros(NUM_ENVS, 1)
    )
    storage.clear()

    assert len(storage) == 0
    assert list(storage.iter_timesteps()) == []


def test_dones_are_stored_as_bool_regardless_of_input_dtype():
    storage = make_storage()
    storage.add(
        torch.zeros(NUM_ENVS, OBS_DIM),
        torch.zeros(NUM_ENVS, ACTION_DIM),
        torch.tensor([1.0, 0.0, 1.0]),
    )

    _, _, dones = next(storage.iter_timesteps())
    assert dones.dtype == torch.bool
    assert dones.reshape(-1).tolist() == [True, False, True]


def test_auxiliary_depth_fields_are_reserved_but_not_wired():
    storage = make_storage()
    assert storage.clean_depth is None
    assert storage.augmented_depth is None

    with pytest.raises(NotImplementedError, match="denoising"):
        storage.add(
            torch.zeros(NUM_ENVS, OBS_DIM),
            torch.zeros(NUM_ENVS, ACTION_DIM),
            torch.zeros(NUM_ENVS, 1),
            clean_depth=torch.zeros(NUM_ENVS, 1),
        )
