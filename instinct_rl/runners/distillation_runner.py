"""Runner for teacher-student distillation.

Thin on purpose. :class:`~instinct_rl.runners.on_policy_runner.OnPolicyRunner` already does
everything distillation needs from a runner -- env setup, normalizers, the rollout loop and its
episode bookkeeping, logging, checkpointing, ONNX/JIT export -- and all of it goes through
``self.alg.actor_critic``, which for distillation is the student. Re-implementing ``learn()``
just to drop the ``compute_returns`` call would duplicate ~90 lines of bookkeeping to remove
one no-op.

What this subclass adds is the checks that are specific to distillation and would otherwise
fail late and confusingly.
"""

from instinct_rl.runners.on_policy_runner import OnPolicyRunner


class DistillationRunner(OnPolicyRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        alg = self.alg
        if not hasattr(alg, "teacher"):
            raise TypeError(
                f"DistillationRunner requires a distillation algorithm, got {type(alg).__name__}."
                " Set `algorithm.class_name: Distillation` in the runner config."
            )

        # The teacher normalizes its own observations with the normalizer stored in its
        # checkpoint (see TeacherPolicy). A runner-level normalizer on the same group would
        # normalize them a second time, which produces a plausible-looking but quietly wrong
        # label stream -- exactly the kind of bug that only shows up as "distillation just
        # doesn't converge".
        teacher_group = alg.teacher_obs_source
        if teacher_group in self.normalizers:
            raise ValueError(
                f"A runner-level normalizer is configured for the '{teacher_group}' observation"
                " group, but that group feeds the teacher, which already applies its own frozen"
                " normalizer from the teacher checkpoint. Remove"
                f" `normalizers.{teacher_group}` from the runner config."
            )

        if alg.storage.num_transitions_per_env % alg.gradient_length != 0 and not alg.flush_tail:
            raise ValueError(
                f"num_steps_per_env ({alg.storage.num_transitions_per_env}) is not a multiple of"
                f" gradient_length ({alg.gradient_length}) and flush_tail is disabled, so"
                f" {alg.storage.num_transitions_per_env % alg.gradient_length} steps of every"
                " rollout would contribute to the logged loss but produce no gradient."
            )

    def train_mode(self):
        super().train_mode()
        self.alg.teacher.policy.eval()

    def eval_mode(self):
        super().eval_mode()
        self.alg.teacher.policy.eval()
