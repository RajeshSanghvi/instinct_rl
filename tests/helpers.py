"""Shared builders for the distillation tests.

Everything here runs on CPU with tiny networks and no simulator, so the whole suite is a
few seconds. The one thing deliberately *not* stubbed is the recurrent policy: the properties
under test (carry restoration, done masking, BPTT depth) live in
``ActorCriticRecurrent``/``Memory``, so tests use the real classes.
"""

from collections import OrderedDict

import torch

from instinct_rl.algorithms.distillation import Distillation
from instinct_rl.modules.actor_critic import ActorCritic
from instinct_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
from instinct_rl.storage.distillation_storage import DistillationStorage

STUDENT_OBS_DIM = 6
TEACHER_OBS_DIM = 9
NUM_ACTIONS = 4
NUM_ENVS = 5


def obs_format(student_dim=STUDENT_OBS_DIM, teacher_dim=TEACHER_OBS_DIM):
    return {
        "policy": OrderedDict(proprio=(student_dim,)),
        "critic": OrderedDict(privileged=(teacher_dim,)),
    }


def make_recurrent_student(rnn_type="gru", rnn_hidden_size=8, rnn_num_layers=1, seed=0):
    torch.manual_seed(seed)
    return ActorCriticRecurrent(
        obs_format=obs_format(),
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        rnn_type=rnn_type,
        rnn_hidden_size=rnn_hidden_size,
        rnn_num_layers=rnn_num_layers,
        init_noise_std=0.1,
    )


def make_mlp_student(seed=0):
    torch.manual_seed(seed)
    return ActorCritic(
        obs_format=obs_format(),
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        init_noise_std=0.1,
    )


class StubTeacher:
    """Stands in for TeacherPolicy: deterministic labels plus a carry we can watch.

    ``reset_calls`` and ``hidden`` exist so tests can assert the invariant that ``update()``
    never touches the teacher -- only rollout and episode ends may move it.
    """

    def __init__(self, num_actions=NUM_ACTIONS, num_envs=NUM_ENVS):
        self.num_actions = num_actions
        self.hidden = torch.zeros(1, num_envs, 3)
        self.reset_calls = 0
        self.act_calls = 0
        self.policy = torch.nn.Linear(1, 1)  # only needs .eval() to exist

    def act_inference(self, obs):
        self.act_calls += 1
        self.hidden = self.hidden + 1.0  # a carry that advances on every label
        return torch.arange(obs.shape[0] * self.num_actions, dtype=torch.float32).reshape(
            obs.shape[0], self.num_actions
        ) * 0.01

    def reset(self, dones):
        self.reset_calls += 1


def make_algorithm(policy=None, num_envs=NUM_ENVS, num_steps=8, **kwargs):
    """A Distillation wired to a stub teacher and a pre-built storage (no checkpoint needed)."""
    policy = policy if policy is not None else make_recurrent_student()
    cfg = dict(
        gradient_length=4,
        learning_rate=1e-2,
        max_grad_norm=1.0,
        loss_type="mse_sum",
    )
    cfg.update(kwargs)
    alg = Distillation(policy, **cfg)
    alg.teacher = StubTeacher(num_envs=num_envs)
    alg.storage = DistillationStorage(
        num_envs,
        num_steps,
        student_obs_shape=(STUDENT_OBS_DIM,),
        actions_shape=(NUM_ACTIONS,),
        device="cpu",
    )
    return alg


def fill_storage(alg, num_steps=None, done_at=(), seed=1):
    """Fill the algorithm's storage with reproducible random data.

    ``done_at`` lists timesteps at which env 0 terminates, so tests can exercise the episode
    boundary handling without needing an environment.
    """
    torch.manual_seed(seed)
    storage = alg.storage
    num_steps = num_steps if num_steps is not None else storage.num_transitions_per_env
    for t in range(num_steps):
        obs = torch.randn(storage.num_envs, STUDENT_OBS_DIM)
        label = torch.randn(storage.num_envs, NUM_ACTIONS)
        dones = torch.zeros(storage.num_envs, 1, dtype=torch.bool)
        if t in done_at:
            dones[0] = True
        storage.add(obs, label, dones)
    return storage
