"""End-to-end integration: the real ``OnPolicyRunner.learn()`` loop driving ``Distillation``.

The unit tests drive ``Distillation`` directly with a stub teacher and a hand-filled storage.
That leaves exactly the two things HANDOFF.md flags as unverified:

* **Teacher checkpoint loading** (§7.3) -- ``TeacherPolicy._load`` / ``_load_normalizer`` are
  bypassed entirely by ``helpers.StubTeacher``.
* **``DistillationRunner`` and the real rollout loop** (§8) -- in particular the rollout runs
  under ``torch.inference_mode``, so the carry the student accumulates during collection is an
  *inference tensor*. Nothing below the runner can reproduce that condition, and it is the one
  failure mode that no amount of direct-``update()`` testing can catch.

Everything here runs on GPU because ``OnPolicyRunner.log`` calls ``torch.cuda.mem_get_info``
unconditionally.
"""

import os.path as osp
from collections import OrderedDict

import pytest
import torch
import yaml

from instinct_rl.env import VecEnv
from instinct_rl.modules import build_actor_critic
from instinct_rl.modules.normalizer import EmpiricalNormalization
from instinct_rl.runners.distillation_runner import DistillationRunner

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="OnPolicyRunner.log() calls torch.cuda.mem_get_info unconditionally"
)

DEVICE = "cuda:0"
NUM_ENVS = 64
NUM_ACTIONS = 4
PROPRIO_DIM = 6
PRIV_DIM = 3
TEACHER_OBS_DIM = PROPRIO_DIM + PRIV_DIM
TEACHER_HIDDEN = [32, 32]

STUDENT_GROUP = OrderedDict(proprio=(PROPRIO_DIM,))
TEACHER_GROUP = OrderedDict(proprio=(PROPRIO_DIM,), privileged=(PRIV_DIM,))


class StubVecEnv(VecEnv):
    """Minimal VecEnv: deterministic-ish dynamics, staggered episode ends, no simulator.

    The privileged half of the critic observation is held at zero so the teacher's label is a
    pure function of the *student's* observation. That makes the behaviour-cloning target
    genuinely learnable, which is what lets this test assert that the loss actually falls
    rather than merely that the loop does not crash.
    """

    def __init__(self, num_envs=NUM_ENVS, episode_length=13, device=DEVICE, seed=0):
        self.num_envs = num_envs
        self.num_actions = NUM_ACTIONS
        self.num_rewards = 1
        self.device = device
        self.cfg = {}
        self.max_episode_length = episode_length
        self._gen = torch.Generator(device=device).manual_seed(seed)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Stagger the episode phase so dones fire on different steps for different envs --
        # a uniform done would not distinguish per-env masking from a global reset.
        self._deadline = torch.randint(
            3, episode_length, (num_envs,), generator=self._gen, device=device
        )
        self._state = torch.randn(num_envs, PROPRIO_DIM, generator=self._gen, device=device)

    def _obs(self):
        proprio = torch.tanh(self._state)
        privileged = torch.zeros(self.num_envs, PRIV_DIM, device=self.device)
        return proprio, torch.cat([proprio, privileged], dim=-1)

    def get_obs_format(self):
        return {"policy": STUDENT_GROUP, "critic": TEACHER_GROUP}

    def get_observations(self):
        obs, critic_obs = self._obs()
        return obs, {"observations": {"critic": critic_obs}}

    def reset(self):
        self._state = torch.randn(self.num_envs, PROPRIO_DIM, generator=self._gen, device=self.device)
        self.episode_length_buf.zero_()
        return self.get_observations()

    def step(self, actions):
        noise = torch.randn(self.num_envs, PROPRIO_DIM, generator=self._gen, device=self.device)
        drive = torch.nn.functional.pad(actions.detach(), (0, PROPRIO_DIM - NUM_ACTIONS))
        self._state = 0.9 * self._state + 0.1 * drive + 0.1 * noise

        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self._deadline).to(torch.float32)
        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self._state[reset_ids] = torch.randn(
                reset_ids.numel(), PROPRIO_DIM, generator=self._gen, device=self.device
            )
            self.episode_length_buf[reset_ids] = 0

        obs, critic_obs = self._obs()
        rewards = torch.zeros(self.num_envs, self.num_rewards, device=self.device)
        return obs, rewards, dones, {"observations": {"critic": critic_obs}}


def write_teacher_run(tmp_path, seed=7):
    """Save a realistic teacher run: checkpoint + params/agent.yaml + a *non-trivial* normalizer.

    The normalizer statistics are deliberately far from identity. If the runner were to
    normalize the teacher's observation group a second time, or if ``_load_normalizer`` silently
    did nothing, the labels would be visibly wrong rather than subtly so.
    """
    torch.manual_seed(seed)
    logdir = tmp_path / "teacher_run"
    (logdir / "params").mkdir(parents=True, exist_ok=True)

    teacher_net = build_actor_critic(
        "ActorCritic",
        dict(actor_hidden_dims=TEACHER_HIDDEN, critic_hidden_dims=TEACHER_HIDDEN, init_noise_std=0.5),
        {"policy": TEACHER_GROUP, "critic": TEACHER_GROUP},
        num_actions=NUM_ACTIONS,
        num_rewards=1,
    )

    normalizer = EmpiricalNormalization(shape=(TEACHER_OBS_DIM,))
    with torch.no_grad():
        normalizer._mean.copy_(torch.linspace(-1.5, 1.5, TEACHER_OBS_DIM).unsqueeze(0))
        normalizer._var.copy_(torch.linspace(0.25, 4.0, TEACHER_OBS_DIM).unsqueeze(0))
        normalizer._std.copy_(normalizer._var.sqrt())
        normalizer.count.fill_(10_000)

    torch.save(
        {
            "model_state_dict": teacher_net.state_dict(),
            "policy_normalizer_state_dict": normalizer.state_dict(),
            "iter": 100,
            "infos": None,
        },
        logdir / "model_100.pt",
    )
    with open(logdir / "params" / "agent.yaml", "w") as f:
        yaml.dump({"normalizers": {"policy": {"class_name": "EmpiricalNormalization", "until": None}}}, f)

    return logdir, teacher_net.to(DEVICE).eval(), normalizer.to(DEVICE).eval()


def make_train_cfg(teacher_logdir, **alg_overrides):
    """Fresh dicts every call: OnPolicyRunner pops 'class_name' out of the cfg it is handed."""
    algorithm = dict(
        class_name="Distillation",
        teacher_logdir=str(teacher_logdir),
        teacher_checkpoint="model_100.pt",
        teacher_policy_class_name="ActorCritic",
        teacher_policy=dict(
            actor_hidden_dims=TEACHER_HIDDEN, critic_hidden_dims=TEACHER_HIDDEN, init_noise_std=0.5
        ),
        teacher_obs_source="critic",
        loss_type="mse_sum",
        gradient_length=8,
        normalize_accumulated_loss=True,
        flush_tail=True,
        learning_rate=3.0e-3,
        max_grad_norm=1.0,
        freeze_action_std=True,
    )
    algorithm.update(alg_overrides)
    return dict(
        runner_class_name="DistillationRunner",
        num_steps_per_env=20,
        save_interval=1000,
        log_interval=1,
        inference_mode_rollout=True,
        policy=dict(
            class_name="ActorCriticRecurrent",
            rnn_type="gru",
            rnn_hidden_size=32,
            rnn_num_layers=1,
            actor_hidden_dims=[32],
            critic_hidden_dims=[32],
            init_noise_std=0.1,
        ),
        normalizers={"policy": {"class_name": "EmpiricalNormalization", "until": None}},
        algorithm=algorithm,
    )


@pytest.fixture(scope="module")
def teacher_run(tmp_path_factory):
    return write_teacher_run(tmp_path_factory.mktemp("teacher"))


def build_runner(teacher_run, tmp_path, **alg_overrides):
    logdir, _, _ = teacher_run
    torch.manual_seed(0)
    env = StubVecEnv()
    runner = DistillationRunner(
        env, make_train_cfg(logdir, **alg_overrides), log_dir=str(tmp_path), device=DEVICE
    )
    return env, runner


"""
Teacher checkpoint loading (HANDOFF §7.3)
"""


def test_teacher_labels_match_the_teacher_run_normalized_exactly_once(teacher_run, tmp_path):
    """HANDOFF §7.3 step 4. A mis-normalized teacher yields plausible labels and a healthy-looking
    loss curve, so this is checked against the teacher's own forward pass rather than eyeballed."""
    _, ref_net, ref_normalizer = teacher_run
    _, runner = build_runner(teacher_run, tmp_path)

    privileged = torch.randn(NUM_ENVS, TEACHER_OBS_DIM, device=DEVICE)
    expected = ref_net.act_inference(ref_normalizer(privileged))
    actual = runner.alg.teacher.act_inference(privileged)

    assert torch.allclose(actual, expected, atol=1e-6)
    # And the normalizer is genuinely doing something -- otherwise the assertion above would
    # also pass for an implementation that dropped it entirely.
    assert not torch.allclose(actual, ref_net.act_inference(privileged), atol=1e-4)


def test_teacher_normalizer_is_loaded_and_frozen(teacher_run, tmp_path):
    _, runner = build_runner(teacher_run, tmp_path)
    normalizer = runner.alg.teacher.normalizer

    assert normalizer is not None, "teacher checkpoint carried a normalizer but none was loaded"
    assert not normalizer.training
    before = normalizer._mean.clone()
    runner.alg.teacher.act_inference(torch.randn(NUM_ENVS, TEACHER_OBS_DIM, device=DEVICE) * 10.0)
    assert torch.equal(normalizer._mean, before), "teacher normalizer statistics drifted"


def test_teacher_is_frozen_and_absent_from_the_student_parameters(teacher_run, tmp_path):
    _, runner = build_runner(teacher_run, tmp_path)
    assert all(not p.requires_grad for p in runner.alg.teacher.policy.parameters())
    student_params = {id(p) for p in runner.alg.actor_critic.parameters()}
    assert not any(id(p) in student_params for p in runner.alg.teacher.policy.parameters())


def test_architecture_mismatch_on_the_actor_path_is_rejected(teacher_run, tmp_path):
    logdir, _, _ = teacher_run
    env = StubVecEnv()
    cfg = make_train_cfg(logdir)
    cfg["algorithm"]["teacher_policy"]["actor_hidden_dims"] = [8, 8]  # wrong shape on purpose
    with pytest.raises(RuntimeError, match="does not match the configured teacher architecture"):
        DistillationRunner(env, cfg, log_dir=str(tmp_path), device=DEVICE)


def test_teacher_with_a_wider_critic_obs_group_still_loads(tmp_path):
    """A privileged teacher whose *value function* saw more than its actor -- e.g. a height scan
    that the distillation env does not expose -- is the normal case, not an exotic one.

    ``load_state_dict(strict=False)`` does not cover it: it forgives missing and unexpected keys
    but still raises on a size mismatch for a key that *is* present, so the mismatched critic
    tensors have to be dropped before loading.
    """
    torch.manual_seed(11)
    logdir = tmp_path / "wide_critic_teacher"
    (logdir / "params").mkdir(parents=True, exist_ok=True)
    wide_critic_group = OrderedDict(proprio=(PROPRIO_DIM,), privileged=(PRIV_DIM,), height_scan=(20,))

    teacher_net = build_actor_critic(
        "ActorCritic",
        dict(actor_hidden_dims=TEACHER_HIDDEN, critic_hidden_dims=TEACHER_HIDDEN, init_noise_std=0.5),
        {"policy": TEACHER_GROUP, "critic": wide_critic_group},
        num_actions=NUM_ACTIONS,
        num_rewards=1,
    )
    torch.save({"model_state_dict": teacher_net.state_dict()}, logdir / "model_100.pt")

    env = StubVecEnv()
    runner = DistillationRunner(
        env, make_train_cfg(logdir), log_dir=str(tmp_path / "run"), device=DEVICE
    )

    # The actor -- the only half that produces labels -- must still have loaded exactly.
    reference = teacher_net.to(DEVICE).eval()
    privileged = torch.randn(NUM_ENVS, TEACHER_OBS_DIM, device=DEVICE)
    assert torch.allclose(
        runner.alg.teacher.act_inference(privileged), reference.act_inference(privileged), atol=1e-6
    )


"""
The full learn() loop
"""


def test_learn_runs_under_inference_mode_and_the_loss_falls(teacher_run, tmp_path):
    """The load-bearing test: the real rollout collects the carry under torch.inference_mode,
    which no direct-update() test can reproduce."""
    env, runner = build_runner(teacher_run, tmp_path)
    assert runner.cfg["inference_mode_rollout"] is True

    losses = []
    original_update = runner.alg.update

    def spy(iteration):
        loss_dict, stats = original_update(iteration)
        losses.append(loss_dict["behavior_loss"].item())
        return loss_dict, stats

    runner.alg.update = spy
    runner.learn(60, init_at_random_ep_len=True)

    assert len(losses) == 60
    first = sum(losses[:5]) / 5
    last = sum(losses[-5:]) / 5
    assert last < 0.5 * first, f"behaviour loss did not fall: {first:.4f} -> {last:.4f}"


def test_optimizer_step_count_and_tail_match_the_configuration(teacher_run, tmp_path):
    # T=20, G=8 -> two full chunks plus a 4-step tail.
    env, runner = build_runner(teacher_run, tmp_path, gradient_length=8)
    seen = {}
    original_update = runner.alg.update

    def spy(iteration):
        loss_dict, stats = original_update(iteration)
        seen.update({k: v.item() for k, v in stats.items()})
        return loss_dict, stats

    runner.alg.update = spy
    runner.learn(2)

    assert seen["optimizer_steps"] == 3
    assert seen["tail_chunk_size"] == 4
    assert seen["teacher_state_ratio"] == 0.0


def test_carry_survives_the_rollout_as_a_trainable_tensor(teacher_run, tmp_path):
    """The inference-tensor trap: after an update the stored carry must be a normal tensor, or
    the *next* update fails deep inside the RNN rather than here."""
    env, runner = build_runner(teacher_run, tmp_path)
    runner.learn(3)

    carry = runner.alg._rollout_start_hidden
    assert carry is not None
    assert not carry.is_inference(), "carry leaked out of the rollout as an inference tensor"
    assert not carry.requires_grad
    assert carry.shape == (1, NUM_ENVS, 32)


def test_student_is_what_drives_the_environment(teacher_run, tmp_path):
    """No teacher-action mixing: every action handed to env.step comes from the student."""
    env, runner = build_runner(teacher_run, tmp_path)
    executed = []
    original_step = env.step

    def spy(actions):
        executed.append(actions.clone())
        return original_step(actions)

    env.step = spy
    runner.learn(2)

    assert len(executed) == 2 * runner.num_steps_per_env
    # A teacher label would sit in the teacher's output range; more directly, the student's
    # action std is frozen at 0.1, so executed actions must vary run-to-run around its mean
    # while the teacher's deterministic label would not appear at all in the buffer.
    stored_labels = runner.alg.storage.teacher_action_means
    assert not any(torch.allclose(a, stored_labels[0], atol=1e-5) for a in executed[:5])


def test_refresh_hidden_ablation_runs_in_the_real_loop(teacher_run, tmp_path):
    """The documented ablation knob. It replays the rollout a second time under no_grad, which
    touches the same carry that the rollout produced under inference_mode."""
    env, runner = build_runner(teacher_run, tmp_path, refresh_hidden_after_update=True)
    seen = {}
    original_update = runner.alg.update

    def spy(iteration):
        loss_dict, stats = original_update(iteration)
        seen.update({k: v.item() for k, v in stats.items()})
        return loss_dict, stats

    runner.alg.update = spy
    runner.learn(3)

    assert seen["hidden_refresh_time"] > 0.0
    assert not runner.alg._rollout_start_hidden.is_inference()


def test_resume_from_a_distillation_checkpoint(teacher_run, tmp_path):
    """Resume is a real workflow and it goes through OnPolicyRunner.load, which reads the
    checkpoint back with weights_only=True."""
    env, runner = build_runner(teacher_run, tmp_path)
    runner.learn(2)
    path = osp.join(str(tmp_path), "model_2.pt")

    env2, runner2 = build_runner(teacher_run, tmp_path / "resumed")
    runner2.load(path)

    assert runner2.current_learning_iteration == 2
    for (k, a), (_, b) in zip(
        runner2.alg.actor_critic.state_dict().items(), runner.alg.actor_critic.state_dict().items()
    ):
        assert torch.equal(a, b), f"{k} did not survive the resume"
    # The carry is deliberately not restored: on resume the env is reset, so a carry from a
    # different episode is worse than zeros.
    assert runner2.alg._rollout_start_hidden is None
    runner2.learn(2)  # and it keeps training


def test_checkpoint_is_a_plain_student_actor_critic(teacher_run, tmp_path):
    env, runner = build_runner(teacher_run, tmp_path)
    runner.learn(2)

    path = osp.join(str(tmp_path), "model_2.pt")
    assert osp.isfile(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)

    assert set(saved["model_state_dict"]) == set(runner.alg.actor_critic.state_dict())
    assert "policy_normalizer_state_dict" in saved, "runner-level normalizer must be checkpointed"
    assert not any("teacher" in k for k in saved)
    # No carry is persisted: on resume the env is reset, so a stale carry is worse than zeros.
    assert not any("hidden" in k for k in saved)

    # And it loads back into a freshly built student.
    fresh = build_actor_critic(
        "ActorCriticRecurrent",
        dict(
            rnn_type="gru",
            rnn_hidden_size=32,
            rnn_num_layers=1,
            actor_hidden_dims=[32],
            critic_hidden_dims=[32],
            init_noise_std=0.1,
        ),
        {"policy": STUDENT_GROUP, "critic": TEACHER_GROUP},
        num_actions=NUM_ACTIONS,
        num_rewards=1,
    )
    fresh.load_state_dict(saved["model_state_dict"])


"""
DistillationRunner's own guards (HANDOFF §8: previously unexercised)
"""


def test_double_normalizing_the_teacher_group_is_rejected(teacher_run, tmp_path):
    logdir, _, _ = teacher_run
    env = StubVecEnv()
    cfg = make_train_cfg(logdir)
    cfg["normalizers"]["critic"] = {"class_name": "EmpiricalNormalization", "until": None}
    with pytest.raises(ValueError, match="feeds the teacher"):
        DistillationRunner(env, cfg, log_dir=str(tmp_path), device=DEVICE)


def test_silently_dropped_tail_steps_are_rejected(teacher_run, tmp_path):
    logdir, _, _ = teacher_run
    env = StubVecEnv()
    cfg = make_train_cfg(logdir, gradient_length=8, flush_tail=False)  # 20 % 8 == 4
    with pytest.raises(ValueError, match="produce no gradient"):
        DistillationRunner(env, cfg, log_dir=str(tmp_path), device=DEVICE)


def test_non_distillation_algorithm_is_rejected(teacher_run, tmp_path):
    logdir, _, _ = teacher_run
    env = StubVecEnv()
    cfg = make_train_cfg(logdir)
    cfg["algorithm"] = dict(
        class_name="PPO", num_learning_epochs=1, num_mini_batches=1, learning_rate=1e-3
    )
    with pytest.raises(TypeError, match="requires a distillation algorithm"):
        DistillationRunner(env, cfg, log_dir=str(tmp_path), device=DEVICE)
