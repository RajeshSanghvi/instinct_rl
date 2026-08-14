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
| `tests/test_distillation_config_guards.py` | fail-fast guards: teacher checkpoint, unknown keys, obs aliasing, weight selection |
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
  warm_start_from_teacher: false          # see below -- large effect when applicable

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

- The weights do not change *during* the replay, so every chunk is differentiated at the same
  parameter version. This removes the inter-chunk weight skew but **not** all staleness: the
  optimizer step happens after the replay, so the carry handed to the next rollout was still
  produced by the pre-step weights. What changes is that the staleness becomes uniform and
  exactly one optimizer step instead of a mixture spanning the update.
  `refresh_hidden_after_update` is what removes that last step, and it stays useful here.
  (Even with refresh on, only the *current* rollout is replayed at the final weights — the
  carry entering that rollout still came from earlier history. Stateful TBPTT is bounded
  staleness, never full self-consistency.)
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
continues the trajectory the optimizer state describes. Checkpoints written before EMA existed
still load.

`get_inference_policy`, `export_as_jit` and `export_as_onnx` all read `alg.actor_critic`
directly, so an export would otherwise ship weights the checkpoint does not contain. Select
explicitly, and non-destructively:

```python
with alg.use_student_weights("ema"):
    runner.export_as_onnx(...)
```

It restores on exit (including on exception), so it is safe mid-training — leaving the average
installed would corrupt the next step, since the optimizer state belongs to the raw weights.

**The env group feeding the teacher must match the layout the teacher was built for.** When
`teacher_policy.obs_format` is given explicitly, the teacher is built from *that* declaration
rather than from the env, so the two can silently disagree: `ParallelLayer` slices the incoming
tensor by the declared segment sizes, so a longer observation is truncated and a shorter one
reinterprets neighbouring components. The teacher keeps producing confident, meaningless labels.

This happened. The env's `critic` group was switched to history-stacked observations (proprio
x8, height scan x4 = **3540** values) while the teacher was still declared on the single-frame
layout (**789**). The run trained for **8.3 hours**: initial behaviour loss 72.9 against the
warm-started 5.9, every episode ending in base contact within ~15 steps (`base_contact` 0.80-0.99,
`time_out` exactly 0), and `terrain_levels` pinned at 0 from iteration 100 onward. Nothing raised.

`init_storage` now compares the two and refuses, printing both layouts term by term so the
offending observation is obvious. Dropping `teacher_policy.obs_format` entirely also avoids the
class of bug, since the teacher is then built from the env group by construction.

> A related invariant in the env config, easy to undo: the group feeding the teacher must stay
> **uncorrupted**. `StudentObservationsCfg.CriticCfg` carries a comment saying its noise is
> intentionally omitted because the teacher checkpoint was trained without it. The same run above
> also flipped `enable_corruption` to `True` there. That is the asymmetry of the paper's Eq. 8 —
> the student receives noisy proprioception, the teacher clean.

**Fail-fast configuration.** Three misconfigurations used to produce a healthy-looking loss curve
and a wrong result; all three now raise at construction:

- **No teacher checkpoint.** A randomly initialized teacher gives a perfectly smooth loss — the
  student imitates noise. Pass `allow_random_teacher: true` if that is genuinely intended.
- **An absolute `teacher_checkpoint` with no `teacher_logdir`** used to be silently ignored and
  fall through to a random teacher. It is now honoured, with the run directory derived from the
  checkpoint so the normalizer lookup still resolves.
- **Unknown config keys.** A config half-ported from PPO/TPPO (`num_learning_epochs`,
  `num_mini_batches`, `teacher_act_prob`, `distill_target`, `buffer_dilation_ratio`,
  `denoise_loss_coef`, …) or a typo like `gradient_lenght` used to warn and carry on, so the run
  trained fine and its recorded config did not describe what actually ran.

`gradient_length`, `learning_rate` and `max_grad_norm` are range-checked too. `learning_rate: 0`
stays legal — it freezes the weights while still populating `.grad`.

**`warm_start_from_teacher`.** Off by default. On, every teacher weight the student can accept is
copied in — memory, actor head, MoE gate — leaving only the exteroceptive encoder (`encoders.*`)
and the action std randomly initialized. Nothing is frozen; the inherited weights still train.

It only applies when teacher and student are identical apart from that encoder, which is the
setup *Now You See That* describes ("the teacher and student policies share identical
architectures except for their exteroceptive encoders … the depth encoder output replaces the
height scan embedding"). The check is that both encoders emit the same latent width, since the
GRU input is `proprio + latent` — a differing latent size shifts every recurrent weight. If any
`memory_a.*` or `actor.*` tensor fails to transfer, construction raises rather than training on
from a half-seeded network that the log claims was warm-started.

**Measured on the parkour G1 task**, teacher and student both `EncoderMoEActorCriticRecurrent`
(GRU 256, actor `[512, 256, 128]`, 4 MoE experts, 128-wide latent):

| | cold start, **465 iterations** | warm start, **20 iterations** |
| --- | --- | --- |
| `Episode/Curriculum/terrain_levels` | 0.0000 | **5.45** |
| `Train/mean_episode_length` | 69 steps | **800 steps** |
| `Loss/behavior` | 14.31 | **1.16** |
| `Episode_Termination/time_out` | 0.0000 | dominant |

Cold-started, *every* episode ended in `bad_orientation` (67%) or `root_height` (34%) — a
student-only rollout from random weights produces nothing but falling, so the only states it ever
labels are states the teacher would never visit. Warm-started, `mean_episode_length` was already
746 at **iteration 0**, before a single gradient step.

> **Warm start changes what the learning rate should be.** The inherited policy is already near
> optimal, so One Cycle's premise — a large ramp for from-scratch super-convergence — no longer
> holds. On the run above, `max_lr: 1e-2` degraded the policy *during the ramp*, at an actual lr
> of only ~2e-3: between iterations 40 and 67 the behaviour loss went 0.797 → 2.183, mean episode
> length 880 → 477, terrain level 5.03 → 3.65 and `root_height` terminations 0.086 → 0.296. The
> damage threshold sat somewhere around 1.5–2e-3, i.e. a fifth of the configured peak. Start an
> order of magnitude lower (`max_lr: 1e-3`) and keep `save_interval` small enough to retain the
> best checkpoint.
>
> A second, warm-start-specific mechanism to be aware of: the inherited GRU was trained on the
> *teacher's* height-scan latent distribution, while the student's depth encoder starts random
> and its output distribution keeps moving as it learns — the downstream network is standing on
> ground that shifts under it. This is a concrete reason for the paper's `L_kl` (pin the encoder's
> batch-wise output distribution to `N(0, I)`) beyond "prevent representation collapse". If a low
> lr is not enough, freezing everything but the encoder for the first iterations is the standard
> remedy and directly targets this.

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
`update()` scales as `gradient_length x (num_envs / num_env_minibatches) x
per_step_activation_size` — the whole chunk's graph is retained until its `backward()`.

> An earlier version of this file said "if you OOM, halve `gradient_length` before touching
> `num_envs`". That was the wrong advice: it spends the BPTT depth this module exists to provide.
> Raise `num_env_minibatches` instead (§4) — it is exact, and costs only speed.

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

**`num_env_minibatches` is how you raise `num_envs`.** Peak activation memory during `update()`
is `gradient_length x num_envs x per_step_activation`, and the replay used to hold all
environments at once, so the only way to fit more of them was to shorten the BPTT window. That
is a bad trade — the BPTT machinery is the reason this module exists. Splitting the *environment*
dimension instead costs nothing semantically: environments are independent (each owns its RNN
carry, and no operator in the student mixes across the batch), so the accumulated gradient equals
the unsplit one to numerical precision. `tests/test_distillation_env_minibatch.py` pins that
equality for GRU and LSTM carries, uneven slices, tail chunks and the refresh pass.

Measured on the parkour G1 task (31 GB card, `G=80`): ~10.4 MB per environment, of which ~6.4 MB
is `gradient_length`-proportional activation and ~4 MB is simulation. `k=4` therefore buys
roughly 4x the environments at the same BPTT depth. The cost is speed — k sequential passes over
`num_envs / k` environments launch smaller kernels.

**Why more environments matter here, even though the terrain curriculum saturates at level 6.0
for every value tried (768, 1024, 4096).** Most of this env's domain randomization is
`mode="startup"`: friction, restitution, body mass, COM offset, actuator gains and joint armature
are each sampled **once per environment for the entire run**. The number of distinct dynamics the
policy ever experiences is exactly `num_envs`, and training longer does not add one. That is a
sim-to-real coverage property, invisible to every training metric — which is why the saturated
curriculum is not evidence against raising it.

Requires `accumulate_gradients: true`; stepping inside a slice would let later slices
differentiate at weights the earlier ones moved, which is what the exactness rests on.

**`num_steps_per_env` is a latency knob, not a throughput one.** Collection dominates: on the
parkour G1 task at `num_envs: 1024` it is ~85 ms per environment step (physics + depth render +
the 693-ray height scan the teacher needs), so `T=800` costs ~68 s of collection against ~4 s of
learning — 94 % of the iteration, and no log line until all of it finishes.

With `accumulate_gradients: false`, `T` does **not** change the optimizer-step rate: a rollout
yields `ceil(T/G)` steps and takes time proportional to `T`, so steps-per-second is `1/(G x
per-step-cost)` either way. Nor does it change the effective batch, which is `G x num_envs` per
step. What shrinking `T` toward `G` buys is feedback latency, checkpoint granularity, and less
storage; what it costs is nothing, with one thing in its favour: at `T=G` every chunk is trained
on data the *current* weights collected, whereas at `T=10G` nine tenths of each rollout was
collected by weights up to nine optimizer steps older — which is off-policy for a DAgger method
whose whole premise is the student's own current state distribution.

Getting below `T=G` requires cutting `G` (BPTT depth) or `num_envs`, both of which do cost
something.

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
- ~~**The replay holds every environment at once**~~ — closed by `num_env_minibatches` (§4).
  Peak activation is now `gradient_length x (num_envs / k)`, so `num_envs` is no longer bounded
  by BPTT depth. Note the earlier advice in §5b — "if you OOM, halve `gradient_length`" — was
  wrong, and is the remedy this option replaces.
- ~~**No test covers `DistillationRunner`**~~ — closed. `tests/test_distillation_integration.py`
  defines a `StubVecEnv` and drives the real `OnPolicyRunner.learn()` loop, so all three runner
  guards, the rollout under `torch.inference_mode`, checkpointing and resume are exercised. That
  loop is also the only place the inference-tensor carry trap (§5) can actually occur.
