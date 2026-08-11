"""Sequential rollout buffer for teacher-student distillation.

Deliberately *not* built on :class:`~instinct_rl.storage.rollout_storage.RolloutStorage`.
Pure behaviour cloning needs none of what that class provides -- values, returns, advantages,
action log probabilities, saved hidden states, trajectory splitting and zero padding -- and
inheriting them would force this buffer to keep them consistent for no benefit.

What distillation *does* need is the one thing the PPO buffer cannot give: the rollout in its
original ``(T, N)`` time order, so the student's RNN can be replayed forward through it while
its carry is recomputed with the current weights.
"""

import torch


class DistillationStorage:
    """Stores one rollout of student observations and teacher action labels, in time order.

    Shapes follow the repo convention of ``(num_transitions_per_env, num_envs, ...)``.
    """

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        student_obs_shape,
        actions_shape,
        device="cpu",
    ):
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.device = device

        self.student_observations = torch.zeros(
            num_transitions_per_env, num_envs, *student_obs_shape, device=device
        )
        self.teacher_action_means = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=device)

        # Reserved for the denoising / feature-KL auxiliary losses. Intentionally left
        # unallocated and unwired: allocating (T, N, H, W) depth buffers costs real memory and
        # nothing reads them yet. `add()` accepts them so the plumbing exists the day a loss
        # needs it.
        self.clean_depth = None
        self.augmented_depth = None

        self.step = 0

    def add(self, student_obs, teacher_action_mean, dones, clean_depth=None, augmented_depth=None):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("DistillationStorage overflow: call clear() before refilling the buffer.")
        self.student_observations[self.step].copy_(student_obs)
        self.teacher_action_means[self.step].copy_(teacher_action_mean)
        self.dones[self.step].copy_(dones.reshape(self.num_envs, 1).to(torch.bool))
        if clean_depth is not None or augmented_depth is not None:
            raise NotImplementedError(
                "clean_depth / augmented_depth are reserved for the denoising and feature-KL"
                " losses and are not wired up yet."
            )
        self.step += 1

    def iter_timesteps(self):
        """Yield ``(student_obs, teacher_action_mean, dones)`` in rollout order.

        Time order is the whole point of this class: the student RNN is replayed through these
        steps sequentially so that its carry is produced by the weights being trained, rather
        than read back from whatever the rollout happened to save.
        """
        for t in range(self.step):
            yield self.student_observations[t], self.teacher_action_means[t], self.dones[t]

    def clear(self):
        self.step = 0

    def __len__(self):
        return self.step

    @property
    def is_full(self):
        return self.step == self.num_transitions_per_env
