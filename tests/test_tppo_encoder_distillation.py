from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from instinct_rl.algorithms.tppo import TPPO
from instinct_rl.modules.parallel_layer import ParallelLayer
from instinct_rl.utils import unpad_trajectories


class FakeEncoderPolicy(nn.Module):
    is_recurrent = False

    def __init__(self, input_size, latent_size):
        super().__init__()
        self.encoders = nn.Linear(input_size, latent_size, bias=False)
        self.encoder_latents_buf = dict()
        self.action_mean = None

    def act(self, observations, **kwargs):
        self.encoder_latents_buf["actor"] = self.encoders(observations)
        self.action_mean = self.encoder_latents_buf["actor"]
        return self.action_mean


def make_algorithm(student, teacher, encoder_loss_coef):
    algorithm = TPPO.__new__(TPPO)
    algorithm.actor_critic = student
    algorithm.teacher_actor_critic = teacher
    algorithm.teacher_policy_normalizer = None
    algorithm.label_action_with_critic_obs = True
    algorithm.action_labels_from_sample = False
    algorithm.using_ppo = False
    algorithm.hidden_state_resample_prob = 0.0
    algorithm.distill_target = "mse_sum"
    algorithm.encoder_distillation_loss_coef = encoder_loss_coef
    algorithm.encoder_distillation_student_component = None
    algorithm.encoder_distillation_teacher_component = None
    algorithm._TPPO__distillation_loss_coef = 1.0
    return algorithm


def make_minibatch(student_obs, teacher_obs):
    return SimpleNamespace(
        obs=student_obs,
        critic_obs=teacher_obs,
        masks=None,
        hidden_states=SimpleNamespace(actor=None),
        action_labels=torch.zeros(student_obs.shape[0], 3),
    )


def test_compute_losses_adds_encoder_mse_and_only_backpropagates_to_student_encoder():
    student = FakeEncoderPolicy(input_size=2, latent_size=3)
    teacher = FakeEncoderPolicy(input_size=2, latent_size=3)
    algorithm = make_algorithm(student, teacher, encoder_loss_coef=0.1)
    student_obs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    teacher_obs = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
    student_encoder_calls = []
    student.encoders.register_forward_hook(lambda *_: student_encoder_calls.append(None))

    losses, _, _ = algorithm.compute_losses(make_minibatch(student_obs, teacher_obs))

    assert len(student_encoder_calls) == 1
    expected = F.mse_loss(student.encoders(student_obs), teacher.encoders(teacher_obs))
    assert losses["encoder_distillation_loss"].item() == pytest.approx(expected.item())
    losses["encoder_distillation_loss"].backward()
    assert student.encoders.weight.grad is not None
    assert teacher.encoders.weight.grad is None


def test_compute_losses_skips_encoder_mse_when_disabled():
    student = FakeEncoderPolicy(input_size=2, latent_size=3)
    teacher = FakeEncoderPolicy(input_size=2, latent_size=3)
    algorithm = make_algorithm(student, teacher, encoder_loss_coef=0.0)
    student_obs = torch.ones(2, 2)

    losses, _, _ = algorithm.compute_losses(make_minibatch(student_obs, torch.ones(2, 2)))

    assert "encoder_distillation_loss" not in losses
    expected_distillation_loss = student.encoders(student_obs).pow(2).sum(dim=-1).mean()
    assert losses["distillation_loss"].item() == pytest.approx(expected_distillation_loss.item())


class ComponentEncoderPolicy(nn.Module):
    is_recurrent = True

    def __init__(self, input_segments, block_configs):
        super().__init__()
        self.encoders = ParallelLayer(input_segments, block_configs)
        self.encoder_latents_buf = dict()


def make_component_algorithm(student, teacher):
    algorithm = TPPO.__new__(TPPO)
    algorithm.actor_critic = student
    algorithm.teacher_actor_critic = teacher
    algorithm.teacher_policy_normalizer = None
    algorithm.label_action_with_critic_obs = True
    algorithm.encoder_distillation_student_component = ("depth_image_encoder",)
    algorithm.encoder_distillation_teacher_component = ("height_scan_encoder",)
    return algorithm


def test_component_encoder_loss_preserves_sequence_shape_and_ignores_padding():
    student = ComponentEncoderPolicy(
        OrderedDict(depth_image=(1, 4, 4)),
        {
            "depth_image_encoder": {
                "class_name": "Conv2dHeadModel",
                "component_names": ["depth_image"],
                "output_size": 2,
                "takeout_input_components": True,
                "channels": [1],
                "kernel_sizes": [1],
                "strides": [1],
                "hidden_sizes": [],
                "paddings": [0],
                "nonlinearity": "ReLU",
                "use_maxpool": False,
                "final_nonlinearity": False,
            }
        },
    )
    teacher = ComponentEncoderPolicy(
        OrderedDict(height_scan=(16,)),
        {
            "height_scan_encoder": {
                "class_name": "MlpModel",
                "component_names": ["height_scan"],
                "output_size": 2,
                "takeout_input_components": True,
                "hidden_sizes": [],
                "nonlinearity": "ReLU",
            }
        },
    )
    algorithm = make_component_algorithm(student, teacher)
    student_obs = torch.randn(3, 3, 16)
    teacher_obs = torch.randn(3, 3, 16)
    masks = torch.tensor([[True, True, True], [True, True, False], [True, False, False]])

    student_features = student.encoders.get_block_outputs(student_obs, "depth_image_encoder")
    teacher_features = teacher.encoders.get_block_outputs(teacher_obs, "height_scan_encoder")
    assert student_features.shape == teacher_features.shape == (3, 3, 2)

    loss = algorithm.compute_encoder_distill_loss(student_obs, teacher_obs, masks)
    expected = F.mse_loss(
        unpad_trajectories(student_features, masks),
        unpad_trajectories(teacher_features, masks),
    )
    assert loss.item() == pytest.approx(expected.item())

    padded_student_obs = student_obs.clone()
    padded_teacher_obs = teacher_obs.clone()
    padded_student_obs[~masks] += 100.0
    padded_teacher_obs[~masks] -= 100.0
    padded_loss = algorithm.compute_encoder_distill_loss(padded_student_obs, padded_teacher_obs, masks)
    assert padded_loss.item() == pytest.approx(loss.item())

    loss.backward()
    assert any(parameter.grad is not None for parameter in student.encoders.parameters())
    assert all(parameter.grad is None for parameter in teacher.encoders.parameters())


def test_compute_losses_rejects_mismatched_encoder_output_shapes():
    student = FakeEncoderPolicy(input_size=2, latent_size=3)
    teacher = FakeEncoderPolicy(input_size=2, latent_size=4)
    algorithm = make_algorithm(student, teacher, encoder_loss_coef=0.1)

    with pytest.raises(RuntimeError, match="identical shapes"):
        algorithm.compute_losses(make_minibatch(torch.ones(2, 2), torch.ones(2, 2)))
