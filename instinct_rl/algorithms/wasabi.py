import importlib
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import instinct_rl.modules as instinct_modules
from instinct_rl.algorithms.ppo import PPO
from instinct_rl.storage.amp_storage import AmpStorage
from instinct_rl.utils.buffer import buffer_func
from instinct_rl.utils.utils import get_subobs_size


class WasabiAlgoMixin:
    """A plugin algorithm for [WASABI](https://github.com/martius-lab/wasabi), which also includes traditional AMP algorithm
    (Set `discriminator_loss_func` to "BCEWithLogitsLoss" for AMP).
    """

    def __init__(
        self,
        *args,
        actor_state_key="amp_policy",  # the key of getting policy's state sequence
        reference_state_key="amp_reference",  # the key of getting expert's reference sequence
        num_styles=1,  # number of discriminators / styles. 1 -> vanilla single-discriminator AMP.
        style_key="amp_style",  # obs key giving each env's active style index (e.g. terrain group)
        data_free_styles=(),  # style indices without motion data (zero style reward, no discriminator)
        discriminator_class_name="Discriminator",
        discriminator_kwargs={},
        discriminator_optimizer_class_name="AdamW",
        discriminator_optimizer_kwargs={},
        discriminator_reward_coef=1.0,
        discriminator_reward_type: Literal[
            "log", "quad", "wasserstein"
        ] = "log",  # check more on `compute_discriminator_reward`
        discriminator_loss_func: Literal[
            "WassersteinLoss", "BCEWithLogitsLoss", "MSELoss"
        ] = "BCEWithLogitsLoss",  # by default, lead to AMP
        discriminator_loss_coef=1.0,
        discriminator_gradient_penalty_coef=10.0,
        discriminator_weight_decay_coef=0.0,
        discriminator_logit_weight_decay_coef=0.0,  # loss for last layer weight
        discriminator_gradient_torlerance=0.0,  # If the computed gradient is smaller than this value, the gradient will not be penalized.
        discriminator_backbone_gradient_only=False,  # If True, the discriminator must support encoders and backbone_run function.
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.actor_state_key = actor_state_key
        self.reference_state_key = reference_state_key
        self.num_styles = num_styles
        self.style_key = style_key
        self.data_free_styles = data_free_styles
        self.discriminator_class_name = discriminator_class_name
        self.discriminator_kwargs = discriminator_kwargs
        self.discriminator_optimizer_class_name = discriminator_optimizer_class_name
        self.discriminator_optimizer_kwargs = discriminator_optimizer_kwargs
        self.discriminator_reward_coef = discriminator_reward_coef
        self.discriminator_reward_type = discriminator_reward_type
        self.discriminator_loss_func = discriminator_loss_func
        self.discriminator_loss_coef = discriminator_loss_coef
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_weight_decay_coef = discriminator_weight_decay_coef
        self.discriminator_logit_weight_decay_coef = discriminator_logit_weight_decay_coef
        self.discriminator_gradient_torlerance = discriminator_gradient_torlerance
        self.discriminator_backbone_gradient_only = discriminator_backbone_gradient_only

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        # build discriminator
        if ":" in self.discriminator_class_name:
            DiscriminatorClass = getattr(
                importlib.import_module(self.discriminator_class_name.split(":")[0]),
                self.discriminator_class_name.split(":")[1],
            )
        else:
            DiscriminatorClass = getattr(instinct_modules, self.discriminator_class_name)
        # One discriminator per style (num_styles == 1 reproduces vanilla single-discriminator AMP).
        self.discriminator = instinct_modules.MultiDiscriminator(
            discriminator_class=DiscriminatorClass,
            num_styles=self.num_styles,
            input_segment=obs_format[self.actor_state_key],
            data_free_styles=self.data_free_styles,
            **self.discriminator_kwargs,
        ).to(self.device)
        if not "lr" in self.discriminator_optimizer_kwargs:
            self.discriminator_optimizer_kwargs = dict(lr=self.learning_rate, **self.discriminator_optimizer_kwargs)
        self.discriminator_optimizer = getattr(optim, self.discriminator_optimizer_class_name)(
            self.discriminator.parameters(),
            **self.discriminator_optimizer_kwargs,
        )

        self.amp_transition = AmpStorage.Transition()
        reference_state_size = get_subobs_size(obs_format[self.actor_state_key])
        actor_state_size = get_subobs_size(obs_format[self.reference_state_key])
        assert actor_state_size == reference_state_size, "The shape of robot state and reference state must be the same"
        self.amp_storage = AmpStorage(
            num_envs, num_transitions_per_env, [actor_state_size], [reference_state_size], device=self.device
        )

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        if not (self.actor_state_key in infos["observations"] and self.reference_state_key in infos["observations"]):
            raise ValueError(
                "The key of trajectory observations ({}) or reference observations ({}) is not found in the observation"
                " dictionary".format(self.actor_state_key, self.reference_state_key)
            )

        actor_state = infos["observations"][self.actor_state_key]
        reference_state = infos["observations"][self.reference_state_key]
        self.amp_transition.actor_states = actor_state
        self.amp_transition.reference_states = reference_state
        self.amp_transition.style_ids = self._get_style_ids(infos["observations"], actor_state.shape[0])
        if self.discriminator.is_recurrent:
            self.amp_transition.hidden_states = self.discriminator.get_hidden_states()
        self.amp_transition.dones = dones
        self.amp_storage.add_transitions(self.amp_transition)
        self.amp_transition.clear()

        # do not call compute_auxilary_reward here, because it is called in the baseclass function
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

    def _get_style_ids(self, obs_pack: dict[str, torch.Tensor], num_envs: int) -> torch.Tensor:
        """Per-env active style index (num_envs, 1) long.

        When `num_styles == 1` a missing `style_key` falls back to all-zeros so vanilla AMP configs
        keep working. With multiple styles a missing key is a hard error -- silently defaulting to
        style 0 would leave every other discriminator untrained while training looks healthy.
        """
        style_ids = obs_pack.get(self.style_key)
        if style_ids is None:
            if self.num_styles == 1:
                return torch.zeros(num_envs, 1, device=self.device, dtype=torch.long)
            raise KeyError(
                f"num_styles={self.num_styles} but style key '{self.style_key}' is missing from the "
                "observations. Provide a per-env style observation (e.g. amp_terrain_style) or set "
                "num_styles=1."
            )
        style_ids = style_ids.view(-1, 1).to(device=self.device, dtype=torch.long)
        if style_ids.shape[0] != num_envs:
            raise ValueError(f"style ids batch size {style_ids.shape[0]} != num_envs {num_envs}")
        # Validate whenever the key is present, regardless of num_styles: an out-of-range id (e.g.
        # amp_style=1 while num_styles=1) would otherwise silently match no discriminator, giving
        # zero reward and no discriminator update with no error.
        if (style_ids < 0).any() or (style_ids >= self.num_styles).any():
            raise ValueError(f"style id out of range [0, {self.num_styles}): got {style_ids.unique().tolist()}")
        return style_ids

    def _discriminator_reward(self, disc: torch.Tensor) -> torch.Tensor:
        """Map a raw discriminator output to a (positive) style reward."""
        if self.discriminator_reward_type == "log":
            # Typically discimination is the output of a direct linear layer. This is the default AMP implementation.
            return -torch.log(1 - torch.clamp(torch.sigmoid(disc), 1e-6, 1 - 1e-6))
        elif self.discriminator_reward_type == "quad":
            # Copied from WASABI, not sure if this is correct. This assumes disc is the output of a direct linear layer.
            return torch.clamp(1 - (1 / 4) * torch.square(disc - 1), min=0)
        elif self.discriminator_reward_type == "wasserstein":
            # Copied from WASABI, not sure if this is correct. This assumes disc is the output of a direct linear layer.
            return disc
        raise NotImplementedError(f"discriminator reward type {self.discriminator_reward_type} not implemented")

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        super_auxilary_reward = super().compute_auxiliary_reward(obs_pack)
        actor_state = obs_pack[self.actor_state_key]
        num_envs = actor_state.shape[0]
        style_ids = self._get_style_ids(obs_pack, num_envs).view(-1)

        # Each env is scored only by the discriminator of its active style; data-free styles get 0.
        reward = torch.zeros(num_envs, 1, device=self.device)
        for style in range(self.num_styles):
            if not self.discriminator.has_data(style):
                continue
            mask = style_ids == style
            if mask.any():
                # Frozen stats: reward inference must not perturb the normalizer running mean/std.
                disc = self.discriminator.discriminators[style](actor_state[mask], update=False).detach()
                reward[mask] = self._discriminator_reward(disc)
        super_auxilary_reward["discriminator_reward"] = reward
        return super_auxilary_reward

    def _styles_runnable_on_all_ranks(self, active_styles):
        """Filter `active_styles` to those with enough samples to form minibatches on every rank.

        `style_mini_batch_generator` yields nothing when a style has fewer than `num_mini_batches`
        transitions; running it on some ranks but not others would desync the gradient all-reduce.
        """
        style_flat = self.amp_storage.style_ids.reshape(-1)
        runnable = torch.tensor(
            [int((style_flat == s).sum().item()) >= self.num_mini_batches for s in active_styles],
            device=self.device,
            dtype=torch.float,
        )
        if dist.is_initialized():
            dist.all_reduce(runnable, op=dist.ReduceOp.MIN)
        result = [s for s, ok in zip(active_styles, runnable.tolist()) if ok > 0.5]
        dropped = [s for s in active_styles if s not in result]
        if dropped:
            print(f"[Warning] Styles {dropped} have too few samples this iteration and are skipped.")
        return result

    @torch.no_grad()
    def _update_style_normalizers(self, active_styles):
        """Update each active style's discriminator normalizer once, from a balanced batch of
        this rollout's actor and reference states for that style.

        `AmpStorage.actor_states`/`reference_states` are 1:1 per transition (same shape, same
        style_ids), so concatenating them already gives an equal-count actor/reference batch --
        no separate sampling is needed. This is the only place normalizer statistics change during
        `update()`; everywhere else `update=False` is passed explicitly.

        Under DDP each rank only sees its own envs' transitions, so a plain local `update()`
        would fold in a different batch per rank and desync the running stats (see
        `_ddp_update_normalizer`). `active_styles` is already the DDP-synchronized set from
        `_styles_runnable_on_all_ranks`, so every rank enters the all_reduce for the same styles.
        """
        if not active_styles:
            return
        style_flat = self.amp_storage.style_ids.reshape(-1)
        actor_flat = self.amp_storage.actor_states.reshape(-1, *self.amp_storage.actor_states.shape[2:])
        reference_flat = self.amp_storage.reference_states.reshape(-1, *self.amp_storage.reference_states.shape[2:])
        for style in active_styles:
            normalizer = self.discriminator.discriminators[style].normalizer
            if normalizer is None:
                continue
            mask = style_flat == style
            balanced_states = torch.cat([actor_flat[mask], reference_flat[mask]], dim=0)
            if dist.is_initialized():
                self._ddp_update_normalizer(normalizer, balanced_states)
            else:
                normalizer.update(balanced_states)

    def _ddp_update_normalizer(self, normalizer, local_states: torch.Tensor):
        """All-reduce this rank's local batch into one global (mean, var, count) and fold that
        single combined batch into `normalizer`, so every rank ends up byte-identical.

        Averaging each rank's *already-updated* local mean/var post-hoc (the old
        `_sync_discriminator_buffers` approach) is not the same as the true pooled statistic when
        ranks see different sample counts, and averaging `_var` and `_std` independently breaks
        the `_std == sqrt(_var)` invariant. Combining raw sum(x)/sum(x**2)/count via a single
        all_reduce and deriving mean/var from the *global* totals avoids both problems.
        """
        local_n = torch.tensor([float(local_states.shape[0])], device=self.device)
        local_s1 = local_states.sum(dim=0)
        local_s2 = (local_states**2).sum(dim=0)
        packed = torch.cat([local_n, local_s1, local_s2])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        global_n = packed[0]
        global_s1, global_s2 = packed[1:].chunk(2)
        global_mean = (global_s1 / global_n).unsqueeze(0)
        global_var = (global_s2 / global_n - global_mean.squeeze(0) ** 2).unsqueeze(0)
        normalizer.update_from_moments(global_mean, global_var, global_n)

    def update(self, *args, **kwargs):
        mean_losses, average_stats = super().update(*args, **kwargs)

        if self.discriminator.is_recurrent:
            raise NotImplementedError("Recurrent multi-discriminator update is not supported yet.")

        # Each style's discriminator is updated only on its own roll-out buffer (Multi-AMP).
        active_styles = [s for s in range(self.num_styles) if self.discriminator.has_data(s)]
        # A style yields a fixed number of minibatches only if it has enough samples. Under DDP every
        # rank must run the same styles, otherwise the per-step grad all-reduce collectives desync and
        # hang. So drop any style that lacks samples on *any* rank (synchronized via all-reduce MIN).
        active_styles = self._styles_runnable_on_all_ranks(active_styles)

        # Fold this rollout's actor+reference states into each active style's normalizer exactly
        # once, before any minibatch forward pass reads it. Every forward call in the minibatch
        # loop below uses `update=False`, so the statistics stay fixed for the rest of this
        # update() -- otherwise 5 epochs x 4 mini-batches would each nudge the running mean/std,
        # letting minibatch order and epoch count leak into what should be a fixed scoring rule.
        self._update_style_normalizers(active_styles)

        updates_per_style = self.num_learning_epochs * self.num_mini_batches
        num_updates = max(len(active_styles) * updates_per_style, 1)
        for style in active_styles:
            amp_generator = self.amp_storage.style_mini_batch_generator(
                style, self.num_mini_batches, self.num_learning_epochs
            )
            for amp_minibatch in amp_generator:
                losses, _, stats = self.compute_amp_losses(amp_minibatch, style)

                loss = 0.0
                for k, v in losses.items():
                    loss += getattr(self, k + "_coef", 1.0) * v
                    mean_losses[k] = mean_losses[k] + v.detach() / num_updates
                mean_losses["amp_total_loss"] = mean_losses["amp_total_loss"] + loss.detach() / num_updates
                # stats are per-style keyed, so average them over that style's own update count.
                for k, v in stats.items():
                    average_stats[k] = average_stats[k] + v.detach() / updates_per_style

                self.wasabi_gradient_step(loss, average_stats)

        self.amp_storage.clear()

        return mean_losses, average_stats

    def compute_amp_losses(self, amp_minibatch: AmpStorage.MiniBatch, style: int = 0):
        losses, stats, inter_vars = dict(), dict(), dict()
        discriminator = self.discriminator.discriminators[style]

        # discriminator must compute the discriminator during act()
        # TODO: run recurrent discriminator.
        # update=False: normalizer stats for this style were already folded in once by
        # `_update_style_normalizers` before the minibatch loop started; must stay frozen here.
        actor_d = discriminator(
            amp_minibatch.actor_states,
            masks=amp_minibatch.masks,
            update=False,
        )
        reference_d = discriminator(
            amp_minibatch.reference_states,
            masks=amp_minibatch.masks,
            update=False,
        )

        # compute losses of the distriminator
        if self.discriminator_loss_func == "WassersteinLoss":
            actor_d_loss = actor_d.mean()
            reference_d_loss = -reference_d.mean()
        elif self.discriminator_loss_func == "BCEWithLogitsLoss":
            actor_d_loss = torch.nn.BCEWithLogitsLoss()(actor_d, torch.zeros_like(actor_d))
            reference_d_loss = torch.nn.BCEWithLogitsLoss()(reference_d, torch.ones_like(reference_d))
        elif self.discriminator_loss_func == "MSELoss":
            actor_d_loss = torch.nn.MSELoss()(actor_d, -torch.ones_like(actor_d))
            reference_d_loss = torch.nn.MSELoss()(reference_d, torch.ones_like(reference_d))
        else:
            raise NotImplementedError(f"discriminator loss function {self.discriminator_loss_func} not implemented")
        discriminator_loss = (actor_d_loss + reference_d_loss) * 0.5
        if self.discriminator_gradient_penalty_coef > 0:
            if self.discriminator_backbone_gradient_only:
                discriminator_gradient_penalty = self.compute_discriminator_backbone_gradient(amp_minibatch, style)
            else:
                discriminator_gradient_penalty = self.compute_discriminator_gradient(amp_minibatch, style)
        else:
            discriminator_gradient_penalty = torch.zeros(1, device=self.device)[0]

        # compute weight decay loss
        weight_decay_loss = torch.zeros(1, device=self.device)[0]
        for param in discriminator.parameters():
            weight_decay_loss += torch.sum(param**2)

        # Following ProtoMotions, compute the weight decay loss for the last layer of the discriminator.
        logit_weight_decay_loss = torch.zeros(1, device=self.device)[0]
        if self.discriminator_logit_weight_decay_coef > 0:
            logit_weight = discriminator.logit_layer_weights()
            logit_weight_decay_loss = torch.sum(logit_weight**2)

        losses["discriminator_loss"] = discriminator_loss
        losses["discriminator_gradient_penalty"] = discriminator_gradient_penalty
        losses["discriminator_weight_decay"] = weight_decay_loss
        # Key must be "<attr-prefix>" of `discriminator_logit_weight_decay_coef`: `update()` looks up
        # the coefficient via getattr(self, key + "_coef", 1.0). A mismatched key silently applies 1.0.
        losses["discriminator_logit_weight_decay"] = logit_weight_decay_loss
        stats[f"discriminator_actor_{style}"] = actor_d.mean()
        stats[f"discriminator_reference_{style}"] = reference_d.mean()

        return losses, inter_vars, stats

    def compute_discriminator_backbone_gradient(self, amp_minibatch: AmpStorage.MiniBatch, style: int = 0):
        """Compute the gradient w.r.t discriminator input.
        In WASABI algorithm, it is used as a penalty to satisfy the condition of Lipschitz continuity
        """
        discriminator = self.discriminator.discriminators[style]
        reference_states = amp_minibatch.reference_states
        reference_states = buffer_func(reference_states, torch.clone)

        actor_states = amp_minibatch.actor_states
        actor_states = buffer_func(actor_states, torch.clone)

        combined_states = torch.cat([actor_states, reference_states], dim=0)

        # NOTE: assumeing discriminator has encoders and backbone_run function
        latent = discriminator.encoders(combined_states).detach()
        latent.requires_grad = True

        disc = discriminator.backbone_run(latent, update=False)

        ones = torch.ones_like(disc)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=latent,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return torch.clamp(grad.norm(2, dim=1) - self.discriminator_gradient_torlerance, 0).pow(2).mean()

    def compute_discriminator_gradient(self, amp_minibatch: AmpStorage.MiniBatch, style: int = 0):
        """Compute the gradient w.r.t expert input.
        In WASABI algorithm, it is used as a penalty to satisfy the condition of Lipschitz continuity
        """
        discriminator = self.discriminator.discriminators[style]
        reference_states = amp_minibatch.reference_states
        reference_states = buffer_func(reference_states, torch.clone)
        buffer_func(reference_states, setattr, "requires_grad", True)

        actor_states = amp_minibatch.actor_states
        actor_states = buffer_func(actor_states, torch.clone)
        buffer_func(actor_states, setattr, "requires_grad", True)

        combined_states = torch.cat([actor_states, reference_states], dim=0)

        disc = discriminator(
            combined_states,
            masks=amp_minibatch.masks,  # The mask is designed as if the discriminator is recurrent. But it is typically not.
            update=False,
        )

        ones = torch.ones_like(disc)
        grad = torch.autograd.grad(
            outputs=disc,
            inputs=combined_states,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return torch.clamp(grad.norm(2, dim=1) - self.discriminator_gradient_torlerance, 0).pow(2).mean()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["discriminator"] = self.discriminator.state_dict()
        state_dict["discriminator_optimizer"] = self.discriminator_optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "discriminator" in state_dict:
            disc_sd = state_dict["discriminator"]
            # Backward compat: a single-discriminator checkpoint has top-level keys (e.g. `model.*`,
            # `normalizer.*`), whereas MultiDiscriminator nests them under `discriminators.<i>.`.
            is_legacy_single = not any(k.startswith("discriminators.") for k in disc_sd)
            if is_legacy_single:
                if self.num_styles != 1:
                    print(
                        "[Warning] Loading a single-discriminator checkpoint into a multi-style model;"
                        " only style 0 is initialized from it, the others stay randomly initialized."
                    )
                self.discriminator.discriminators[0].load_state_dict(disc_sd)
            else:
                self.discriminator.load_state_dict(disc_sd)
        else:
            print("[Warning] The discriminator state_dict is not found in the checkpoint")
        if "discriminator_optimizer" in state_dict:
            try:
                self.discriminator_optimizer.load_state_dict(state_dict["discriminator_optimizer"])
            except (ValueError, KeyError) as e:
                # Param groups differ (e.g. legacy single-disc ckpt loaded into multi-style); skip
                # rather than crash so the discriminator weights still resume.
                print(f"[Warning] Could not load discriminator optimizer state ({e}); starting fresh.")
        else:
            print("[Warning] The discriminator_optimizer state_dict is not found in the checkpoint")

    def distributed_data_parallel(self):
        """Broadcast actor-critic AND discriminator params from rank 0 so all ranks start identical.

        The base class only broadcasts the actor-critic; without this the per-style discriminators
        would diverge across ranks (gradient all-reduce cannot recover differing initial weights).
        Buffers (e.g. input-normalizer running stats) are broadcast too: they are not parameters,
        so neither the param broadcast nor the gradient all-reduce covers them.
        """
        super().distributed_data_parallel()
        if dist.is_initialized():
            for param in self.discriminator.parameters():
                dist.broadcast(param.data, src=0)
            for buf in self.discriminator.buffers():
                dist.broadcast(buf.data, src=0)

    def wasabi_gradient_step(self, loss: torch.Tensor, average_stats: dict):
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            for param in self.discriminator.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        # TODO: add clip grad norm
        self.discriminator_optimizer.step()


class WasabiPPO(WasabiAlgoMixin, PPO):
    pass
