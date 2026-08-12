# Distillation module — handoff

Teacher-student distillation (privileged teacher → deployable student, behaviour cloning only,
no RL fine-tuning stage) with stateful truncated BPTT for recurrent students.

**Status: executed and green as of 2026-08-11.** 52 tests pass on `env_isaaclab` (torch 2.7.0,
RTX 4090); imports and `ruff` are clean. One real bug was found and fixed in the process
(teacher checkpoint loading, §1a). The remaining unverified step is a real training run (§7.4).

---

## 1. What was added

| File | Role |
| --- | --- |
| `instinct_rl/algorithms/distillation.py` | `Distillation` algorithm + `TeacherPolicy` adapter |
| `instinct_rl/storage/distillation_storage.py` | `DistillationStorage` — sequential, no padding |
| `instinct_rl/runners/distillation_runner.py` | `DistillationRunner` — thin `OnPolicyRunner` subclass |
| `tests/helpers.py` | test builders (tiny CPU networks, stub teacher) |
| `tests/test_distillation_loss.py` | loss semantics + student-only rollout invariant |
| `tests/test_distillation_accumulation.py` | chunking, tail flush, step-size invariance, lr schedule |
| `tests/test_distillation_recurrent.py` | carry handling, BPTT depth, episode boundaries |
| `tests/test_distillation_storage.py` | storage ordering / capacity |
| `tests/test_distillation_accumulation_ema.py` | gradient accumulation + weight EMA (the *Now You See That* recipe) |
| `tests/test_distillation_integration.py` | **GPU**: real `learn()` loop, teacher checkpoint loading, runner guards, resume |
| `pytest.ini` | test discovery config |

Modified:

| File | Change |
| --- | --- |
| `instinct_rl/modules/actor_critic.py` | added the actor hidden-state interface as no-ops |
| `instinct_rl/modules/actor_critic_recurrent.py` | real implementations + `map_hidden_state` helper |
| `instinct_rl/runners/on_policy_runner.py` | removed two hard-coded PPO loss keys from a log branch |
| `algorithms/`, `storage/`, `runners/` `__init__.py` | exports |

Nothing existing was rewired. `PPO`, `TPPO`, `VaeDistill`, `DaggerSaver`, `TwoStageRunner` are
untouched apart from the one `on_policy_runner.py` log fix described in §6.

### Not redundant with what was already there

- `DaggerSaver` is an **offline** demonstration collector (subclass of `DemonstrationSaver`): it
  writes labelled trajectories to disk for a separate training process, and it *does* mix teacher
  actions into the rollout. Different lifecycle, different concern.
- `TwoStageRunner` is PPO with a pretrain stage fed from an on-disk `RolloutDataset`.
- `VaeDistill` is a `TPPO` subclass adding a KL term.

None of them do online, in-process, student-only distillation with stateful TBPTT.

### 1a. Bug found and fixed on first execution

`TeacherPolicy._load` used `load_state_dict(..., strict=False)` and assumed that made critic-path
mismatches non-fatal. It does not: `strict=False` forgives *missing* and *unexpected keys* but
still raises on a **size mismatch for a key that is present**. So the case the code explicitly
documented as supported — a privileged teacher whose value function saw more than its actor, e.g.
a height scan the distillation env does not expose — failed to load at all:

```
RuntimeError: size mismatch for critic.0.weight: copying a param with shape
torch.Size([32, 29]) ... the shape in current model is torch.Size([32, 9]).
```

`_load` now partitions the checkpoint by shape before loading: mismatched **critic** tensors are
dropped (with a printed summary), mismatched **actor** tensors are collected and re-raised in the
existing diagnostic. Regression test:
`test_distillation_integration.py::test_teacher_with_a_wider_critic_obs_group_still_loads`.

The actor/critic split itself (`_is_critic_key`) was checked against a real 106-tensor checkpoint
(`logs/instinct_rl/remote/20260807_153321`): `actor`, `encoders`, `memory_a`, `std` require an
exact load; `critic`, `critic_encoders`, `memory_c` are tolerated. Nothing on the actor path
leaks into the tolerated set — which is the direction that would silently produce wrong labels.

---

## 2. Design decisions worth knowing before you change anything

### The student's carry is recomputed, never replayed from the buffer

PPO's recurrent minibatch generator starts every BPTT segment from the hidden state saved during
rollout; after a few update epochs those were produced by stale weights. Here `update()` replays
the rollout forward in time from `_rollout_start_hidden`, so the carry is produced by the weights
currently being trained.

Residual staleness: at each TBPTT chunk boundary the incoming carry was computed with weights one
optimizer step older. Bounded and small — **not zero**, which is why
`refresh_hidden_after_update` exists as an ablation and defaults to `False` (§4).

### `mask_actor_hidden_state` exists because `Memory.reset` detaches

`instinct_rl/modules/actor_critic_recurrent.py:Memory._reset_tensor_hidden_state` calls
`.detach()` unconditionally. Using `reset(dones)` inside a gradient-carrying replay would
therefore truncate BPTT to a single step for **every** environment, silently — the loss curve
would look completely normal. `mask_actor_hidden_state` multiplies by a 0/1 mask instead, which
zeroes the terminated environments while keeping the graph alive for the rest.

`tests/test_distillation_recurrent.py::test_reset_would_truncate_bptt_which_is_why_mask_exists`
pins this down. Do not "simplify" the mask into a reset.

### The teacher is a separate object

It is never a submodule of the student and `update()` never touches it, so its own recurrent
carry stays continuous across iterations and its architecture is fully decoupled. This is the
structural fix for two bugs that the composed-policy approach (rsl_rl's `StudentTeacher`) has:
the teacher's carry being wiped every update, and teacher/student RNNs being forced to share
`rnn_type`/`hidden_size`/`num_layers`.

### The environment always executes the student's action

`Distillation.act` returns a student sample; the teacher only ever produces labels. No
`teacher_act_prob` mixing. For a recurrent student this matters more than for an MLP: the carry
is a cumulative quantity, so training the memory on state sequences generated by a mixture policy
and then deploying student-only compounds the mismatch over time.

`Train/teacher_state_ratio` is logged as a constant 0 so a future regression shows up on the
dashboard rather than in silence.

### Chunk-mean, not chunk-sum

`normalize_accumulated_loss=True` averages per-step losses over a chunk, so the effective step
size does not scale with `gradient_length`. rsl_rl sums, which couples lr to `gradient_length`.

### The tail is always flushed

`flush_tail=True` runs a final optimizer step on the leftover steps. Without it, with
`num_steps_per_env=24` and `gradient_length=15`, **37.5 % of every rollout contributes to the
logged loss but produces no gradient at all**. Correctness must not depend on `T % G == 0`.

---

## 3. Interface

```python
alg.init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1)
alg.act(obs, critic_obs)                      # -> student action for the env to execute
alg.process_env_step(rewards, dones, infos, next_obs, next_critic_obs)
alg.compute_returns(last_critic_obs)          # no-op, see below
alg.update(current_learning_iteration)        # -> (losses: dict, stats: dict)
alg.state_dict() / alg.load_state_dict(sd)
alg.distributed_data_parallel()               # raises NotImplementedError
alg.train_mode() / alg.test_mode()
```

`compute_returns` is a documented two-line no-op. Removing it would have required a bespoke
`learn()` loop, i.e. duplicating ~90 lines of `OnPolicyRunner` rollout and episode bookkeeping to
delete one call. That trade was judged the wrong way round — if you disagree, the change is
contained to `DistillationRunner`.

Checkpoints hold **student weights only** (no teacher, no carry), so a distillation checkpoint is
a plain actor-critic checkpoint that `OnPolicyRunner` and the exporters consume unchanged. The
carry is deliberately not saved: on resume the env is reset, so a carry from a different episode
is worse than zeros.

---

## 4. Configuration

```yaml
runner_class_name: DistillationRunner

num_steps_per_env: 48          # 0.96 s at 50 Hz
init_at_random_ep_len: true    # decorrelate episode phase across envs
inference_mode_rollout: true   # default; see §5 for why it matters here

policy:                        # the student
  class_name: EncoderActorCriticRecurrent
  rnn_type: gru
  rnn_hidden_size: 128
  rnn_num_layers: 1
  init_noise_std: 0.1

normalizers:                   # see §5a -- omit this block and the actor sees raw observations
  policy:
    class_name: EmpiricalNormalization
    until: 2.0e7
  # no entry for the teacher's group; TeacherPolicy normalizes its own input

algorithm:
  class_name: Distillation

  teacher_logdir: /path/to/teacher/run
  teacher_checkpoint: model_15000.pt      # or null for the latest model_*.pt
  teacher_policy_class_name: EncoderActorCriticRecurrent
  teacher_policy: {...}                   # the teacher's own policy cfg
  teacher_obs_source: critic              # which group of THIS env feeds the teacher's actor

  loss_type: mse_sum
  gradient_length: 24                     # 0.48 s TBPTT; 2 optimizer steps per rollout
  normalize_accumulated_loss: true
  flush_tail: true

  learning_rate: 3.0e-4
  max_grad_norm: 1.0
  optimizer_class_name: Adam
  freeze_action_std: true

  accumulate_gradients: false             # see below
  ema_decay: null                         # e.g. 0.997

  refresh_hidden_after_update: false

  lr_scheduler_class_name: CosineAnnealingLR
  lr_scheduler:
    T_max: 8000                           # see the scheduler note below
    eta_min: 1.0e-5
  lr_scheduler_step_unit: optimizer_step
```

**Scheduler units.** `lr_scheduler_step_unit: optimizer_step` (the default) steps the scheduler
once per optimizer step, i.e. `ceil(num_steps_per_env / gradient_length)` times per iteration
when `flush_tail` is on — the tail flush is an optimizer step and steps the scheduler too. With
`T=48`, `G=24` and 4000 iterations there is no tail, so that is **8000** scheduler steps, not
4000; setting `T_max: 4000` would halve the cosine period. But with e.g. `T=20, G=8` it is 3 per
iteration, not 2.5. Use `lr_scheduler_step_unit: update` if you want one step per iteration.

**`accumulate_gradients`.** Off by default. On, every TBPTT chunk's gradient is accumulated and
the rollout produces **one** clipped optimizer step instead of `ceil(T/G)`. Two consequences:

- The weights do not change during the replay, so the carry entering each chunk comes from
  exactly the weights being trained. The residual one-step staleness described in §2 disappears,
  and `refresh_hidden_after_update` becomes pointless.
- `max_grad_norm` now clips the whole rollout's gradient, and the lr schedule advances once per
  iteration. Both change what a given hyper-parameter means, hence opt-in.

Chunk losses are weighted by chunk *length* (`sum / graded_steps`), not by `1/num_chunks`, so a
short tail chunk does not pull as hard as a full one.

**Reproducing *Now You See That* (Table XII).** `num_steps_per_env: 800`, `gradient_length: 80`
(= 10 accumulation steps), `accumulate_gradients: true`, `max_grad_norm: 1.0`, `ema_decay: 0.997`,
and:

```yaml
  learning_rate: 1.0e-3
  lr_scheduler_class_name: OneCycleLR
  lr_scheduler: {max_lr: 1.0e-2, total_steps: 4000, div_factor: 10.0, final_div_factor: 50.0}
  lr_scheduler_step_unit: optimizer_step   # == one step per iteration under accumulation
```

Verified over a full 4000-iteration schedule: 4000 optimizer steps, lr starts at 1.0e-3, peaks at
1.0e-2 (iteration 1198, the default `pct_start=0.3`), ends at 2.0e-5.

**`ema_decay`.** `None` disables it. When set, an EMA of the student weights is updated after
every optimizer step and **checkpointed as `model_state_dict`** — the slot `OnPolicyRunner` and
the exporters read — while the raw weights ride along under `raw_model_state_dict` so a resume
continues the trajectory the optimizer state describes. `export_as_jit` / `export_as_onnx` trace
`alg.actor_critic` directly, so call `alg.load_ema_into_model()` first if the exported artefact
should match the checkpoint. Checkpoints written before EMA existed still load.

**Normalizer ownership.** The teacher normalizes its own observations using the normalizer stored
in its checkpoint. Do **not** configure `normalizers.critic` (or whichever group
`teacher_obs_source` names) in the runner config — the observations would be normalized twice.
`DistillationRunner.__init__` raises on this rather than letting it become a silent
"distillation just doesn't converge" bug.

**`refresh_hidden_after_update`.** Off by default. It replays the rollout once more under
`no_grad` with the final weights and uses that as the next carry. The staleness it removes is one
optimizer step; the cost is a full extra sequential forward pass over the rollout (student
forward cost of `update` roughly +100%, and sequential small-batch forwards are kernel-launch
bound). With `T/G = 2` I would not expect a measurable difference. Treat it as an ablation.

---

## 5. Traps that are already handled — don't undo them

**Inference-mode tensors.** `on_policy_runner.py:155` runs the rollout under
`torch.inference_mode(self.cfg.get("inference_mode_rollout", True))`. Any carry produced during
rollout is an *inference tensor* and can never take part in autograd — `detach()` and `clone()`
do not rescue it. `set_actor_hidden_state` raises a specific error if you hand it one. The carry
must always come from a replay or a `torch.no_grad()` refresh. `inference_mode_rollout: false` is
a usable debugging escape hatch.

**Done ordering.** Replay applies `mask_actor_hidden_state(dones[t])` *after* computing the loss
at step `t`, mirroring the rollout's `act(obs_t) → env.step → reset(dones_t)`. Reversing this
would let the last step of an episode leak gradient into the next episode's carry, and the loss
curve would not show it.

**`clip_grad_norm_` scope.** Clipping covers `alg.trainable_parameters`, which includes the RNN.
(rsl_rl clips `policy.student.parameters()` — the MLP head only — leaving the recurrent weights,
the actual explosion risk, unprotected.)

---

## 5a. Observation normalization

**It is opt-in.** `OnPolicyRunner` builds `self.normalizers` from `cfg.get("normalizers", {})`.
With no `normalizers` block in the runner config, raw observations go straight into the actor.

**When enabled, the buffer stores post-normalization values.** `rollout_step` normalizes the
observation returned by `env.step`, and the next `alg.act` receives that normalized tensor, which
is what `_pending_student_obs` records. So the replay feeds back exactly the numbers the policy
saw during rollout, even though the running statistics keep drifting in between.

> **Invariant — do not "improve" this.** Storing raw observations and normalizing inside the
> replay instead would use different statistics in rollout and replay, manufacturing a
> distribution shift out of nothing. Store post-normalization values.

**A recurrent student wants this on**, for a reason that does not apply to an MLP. A carry is a
cumulative function of the input sequence: if input channels differ by orders of magnitude,
`W_ih @ x` drives the GRU/LSTM gate pre-activations into saturation, the quiet channels stop
contributing to the carry at all, and the gradient along the cross-time `weight_hh` path
vanishes. That is the exact path the truncated BPTT machinery exists to preserve — with badly
scaled inputs, `gradient_length=24` and `gradient_length=1` may train indistinguishably.

**Freeze the statistics with `until`.** The student's normalizer collects statistics over the
*student's own* state distribution, which early in training (falling over constantly) is nothing
like the converged one. For a recurrent policy that is input non-stationarity the memory has to
absorb on top of everything else. `count` grows by `num_envs` per environment step, so one
iteration at `T=48, N=4096` is ~196k samples:

```yaml
normalizers:
  policy:
    class_name: EmpiricalNormalization
    until: 2.0e7      # ~100 iterations, then frozen
  # Do NOT configure the group named by teacher_obs_source (usually `critic`): TeacherPolicy
  # applies the teacher's own frozen normalizer. DistillationRunner raises on this.
```

Freezing also keeps training and deployment consistent — `export_as_jit` / `export_as_onnx` bake
whatever statistics are current into `policy_normalizer.npz`.

**Known cosmetic gap:** the very first observation of a `learn()` call
(`on_policy_runner.py:124`) bypasses the normalizer. One step per run; not worth fixing.

---

## 5b. GPU notes

Training is single-GPU by default and the code is device-clean: storage buffers, the teacher,
its normalizer and every logged statistic are allocated on `self.device`, and
`mask_actor_hidden_state` moves `dones` to the carry's device itself. There is nothing to
configure. What is worth knowing:

**`gradient_length` is the memory knob, not `num_envs`.** Peak activation memory during
`update()` scales as `gradient_length x num_envs x per_step_activation_size` — the whole chunk's
graph is retained until its `backward()`. If you OOM, halve `gradient_length` before touching
`num_envs`; it costs you BPTT depth but leaves the data throughput intact.

**Storage is lighter than PPO's.** `DistillationStorage` keeps student observations, teacher
action means and dones. It does *not* keep the privileged/critic observations — the teacher is
run at rollout time and only its `(T, N, num_actions)` output is stored. For a privileged
observation space with a height scan that is the difference between a few tens of MB and several
hundred.

**The replay is kernel-launch bound.** `update()` runs `num_steps_per_env` sequential forward
passes; the time dimension cannot be batched, which is the price of recomputing the carry. Each
launch is small, so the GPU is often idle between them. If `Perf/learning_time` looks worse than
expected, that is why — measure before optimising, and measure with `torch.cuda.synchronize()`
around the region or the number will be meaningless. `_refresh_hidden` already synchronises
internally for exactly this reason.

**Per-timestep host syncs are avoided.** All in-loop statistics accumulate as device tensors;
nothing calls `.item()` until `update()` returns. Keep it that way — a `.item()` inside the
replay loop adds one full pipeline stall per environment step.

**`torch.load` in `TeacherPolicy._load`** uses `map_location="cpu"` with no `weights_only`
argument, matching `TPPO`. On torch >= 2.6 the default flipped to `weights_only=True`.
**Verified fine on torch 2.7.0** against a real 30k-iteration checkpoint
(`logs/instinct_rl/remote/20260807_153321/model_20000.pt`), which loaded cleanly. This is not
luck: `OnPolicyRunner.load` itself passes `weights_only=True`, so any checkpoint this repo can
resume from is loadable here too. If a teacher were ever written with a richer `infos` payload
than the runner can read back, the fix is an explicit `weights_only=False`.

**Multi-GPU still raises** (§8). `OnPolicyRunner.learn` calls `distributed_data_parallel()` when
`dist.is_initialized()`, so a distributed launch fails immediately and loudly instead of training
with unsynchronised gradients.

---

## 6. `on_policy_runner.py` change

The console-logging fallback branch (taken whenever `rewbuffer[0]` is empty, i.e. before any
episode has finished) printed `locs["losses"]['value_loss']` and `['surrogate_loss']` by name.
Two problems: the loop directly above already prints every loss, and any algorithm without PPO
loss keys — including this one — raises `KeyError` there during the first iterations. The two
lines were removed. This is a strict bug fix for PPO too (it removes a duplicate print).

---

## 7. What to do on the remote host, in order

### 7.1 Import and static checks

```bash
cd instinct_rl
python -c "import instinct_rl.algorithms, instinct_rl.storage, instinct_rl.runners; print('ok')"
ruff check instinct_rl tests     # or whatever this repo uses
```

**Done — both clean.** No import cycle; `ruff check` passes on the new modules and `tests/`.

### 7.2 Test suite

```bash
source parkour/env_isaaclab/bin/activate
python -m pytest tests -q          # 52 passed
```

**Done — 52 passed.** The 37 unit tests are CPU-only, second-scale and simulator-free; the 15
integration tests need a GPU (they are skipped without one, because `OnPolicyRunner.log` calls
`torch.cuda.mem_get_info` unconditionally).

Only one unit test needed fixing, and it was the test, not the algorithm:
`test_environment_always_executes_the_student_action` set `std` to exactly `0.0` to make sampling
deterministic. `Normal` rejects a zero scale — and this repo's attempt to turn that validation off
is itself broken (see the note below) — so the test now uses `1e-8` and additionally asserts the
executed action *is* the student's `action_mean`, which is a stronger statement than the original.

> **Pre-existing repo bug, untouched, worth a separate fix.** `actor_critic.py:98` reads
> `Normal.set_default_validate_args = False`. That **assigns to** the classmethod instead of
> **calling** it, so distribution validation has never actually been disabled anywhere in this
> repo — every `update_distribution` call in PPO/TPPO pays for it. The one-character fix is
> `Normal.set_default_validate_args(False)`. Left alone here because it changes behaviour for
> every existing algorithm (it also removes a NaN canary on the action mean), which is not
> something a distillation review should decide unilaterally. Note the practical consequence
> meanwhile: **`init_noise_std: 0` crashes**, which is a tempting setting for behaviour cloning
> since the std receives no gradient anyway. Use a small positive value.

Notes on the tests I flagged as fragile — all held up:

- `test_stepwise_replay_matches_a_batched_rnn_over_the_whole_sequence` assumes
  `rnn_highway=False` (the default). If the helper ever enables highway, the reference path must
  fuse too.
- `test_gradient_clipping_covers_the_recurrent_parameters` assumes the un-clipped grad norm
  exceeds `1e-6` on random init. Almost certain, but it is a statistical assumption.
- `test_logged_behavior_loss_covers_every_timestep_including_the_tail` uses a loose `rel=0.5`
  because later chunks are computed with updated weights. Tighten only if you also freeze the lr.
- `helpers.StubTeacher` sidesteps `TeacherPolicy` entirely. That gap is now closed by
  `test_distillation_integration.py`, which writes a real teacher run to disk (checkpoint +
  `params/agent.yaml` + a non-identity normalizer) and loads it through `TeacherPolicy`.

### 7.3 Teacher loading — the first real-data step

`TeacherPolicy._load` is the least testable and most likely-to-bite part — and it is where the
§1a bug was. Steps 1-4 below are now automated in `test_distillation_integration.py`
(`test_teacher_labels_match_the_teacher_run_normalized_exactly_once`,
`test_teacher_normalizer_is_loaded_and_frozen`), against a synthetic-but-real teacher run.
Still walk through them once against **your** teacher before starting a long job:

1. Point `teacher_logdir` at a finished teacher run and construct the runner.
2. Confirm the printed "loading teacher policy from ..." path is the checkpoint you expect.
3. Confirm "loaded and froze the teacher's own observation normalizer" appears — if the teacher
   was trained with a normalizer and this line is missing, labels will be wrong and nothing else
   will complain.
4. Sanity-check the labels: feed one batch of privileged observations and compare
   `alg.teacher.act_inference(obs)` against the teacher's own `play` output on the same input.
   **Do this.** A silently mis-normalized teacher produces plausible-looking labels and an
   apparently healthy loss curve.

The loader accepts missing/unexpected keys **and shape mismatches** on the *critic* path (the
teacher's value function is never used and its observation group may not exist in the
distillation env) but raises on any mismatch on the actor path. The shape half of that was
broken until §1a — if you are reading a copy of this file from before 2026-08-11, a privileged
teacher would not load at all.

If `teacher_policy` has no `obs_format`, one is derived as
`{"policy": env_obs_format[teacher_obs_source], "critic": <same>}`. Pass an explicit
`teacher_policy.obs_format` if your teacher needs something else.

### 7.4 Short smoke run

Few hundred iterations, small `num_envs`. Check:

- `Loss/behavior` decreases
- `Train/optimizer_steps` == `num_steps_per_env / gradient_length` (+1 if a tail exists)
- `Train/tail_chunk_size` is what you expect (0 for `T=48, G=24`)
- `Train/grad_norm_before_clip` is not pinned at the clip threshold every step — if it is,
  lower the lr rather than raising `max_grad_norm`
- `Train/mean_episode_length` moves. **This is the real metric.** `Loss/behavior` is an
  in-distribution regression error; it falls smoothly whether or not the student can walk.
  Because the rollout is student-only, mean episode length is a direct measure of standalone
  student performance.

### 7.5 Then

Only after the smoke run looks sane: full run at `num_steps_per_env: 48`, `gradient_length: 24`.

---

## 8. Known gaps / deliberate omissions

- **Multi-GPU is not implemented.** `distributed_data_parallel()` raises. Note for whoever adds
  it: gradients must be all-reduced before **each** of the `T/gradient_length` optimizer steps in
  an update, not once per update. Getting that wrong is invisible in single-GPU testing.
- **`clean_depth` / `augmented_depth`** are declared on `DistillationStorage` as `None` and
  `add()` raises `NotImplementedError` if you pass them. The plumbing exists for the denoising
  and feature-KL losses; nothing is allocated or wired.
- **No `num_learning_epochs`.** V1 makes one pass per rollout. Teacher labels are free, so extra
  epochs mostly buy carry staleness. Adding it means deciding how chunk grouping interacts with
  epoch boundaries.
- **`TeacherPolicy` normalizer lookup is hard-coded to the `policy` group** of the teacher's
  `params/agent.yaml`, matching `TPPO`. A teacher trained with a differently-named group will
  need a small change in `_load_normalizer`.
- ~~**No test covers `DistillationRunner`**~~ — closed. `tests/test_distillation_integration.py`
  defines a `StubVecEnv` and drives the real `OnPolicyRunner.learn()` loop, so all three runner
  guards, the rollout under `torch.inference_mode`, checkpointing and resume are exercised. That
  loop is also the only place the inference-tensor carry trap (§5) can actually occur.
