from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import get_activation  # Assuming get_activation is needed


class MoeLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_experts,
        output_dim=None,
        activation="elu",
        expert_hidden_dims=[],
        gate_hidden_dims=[],
    ):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = get_activation(activation)
        self.gate = self._build_gate(input_dim, num_experts, gate_hidden_dims)
        self.experts = nn.ModuleList(
            [self._build_expert(input_dim, output_dim, expert_hidden_dims) for _ in range(num_experts)]
        )

        # Running statistics for gate routing (not part of model state, not saved in checkpoints)
        self.register_buffer("_gate_score_sum", torch.zeros(num_experts), persistent=False)
        self.register_buffer("_gate_score_sq_sum", torch.zeros(num_experts), persistent=False)
        self.register_buffer("_gate_entropy_sum", torch.zeros(1), persistent=False)
        self.register_buffer("_gate_sample_count", torch.zeros(1), persistent=False)

    def _build_gate(self, input_dim, num_experts, hidden_dims):
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(self.act_fn)
            curr_dim = h
        layers.append(nn.Linear(curr_dim, num_experts))
        return nn.Sequential(*layers)

    def _build_expert(self, input_dim, output_dim, hidden_dims):
        layers = []
        curr_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(self.act_fn)
            curr_dim = h
        if output_dim is not None:
            layers.append(nn.Linear(curr_dim, output_dim))  # no activation for the last layer
        return nn.Sequential(*layers)

    def forward(self, x):
        gate_scores = F.softmax(self.gate(x), dim=-1)  # [batch, num_experts] # gate the expert outputs

        with torch.no_grad():
            flat = gate_scores.detach().reshape(-1, self.num_experts)
            self._gate_score_sum += flat.sum(dim=0)
            self._gate_score_sq_sum += (flat * flat).sum(dim=0)
            sample_entropy = -(flat * flat.clamp_min(1e-10).log()).sum(dim=-1)
            self._gate_entropy_sum += sample_entropy.sum()
            self._gate_sample_count += flat.shape[0]

        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, output_dim]
        output = torch.einsum("be,beo->bo", gate_scores, expert_outputs)  # mix the expert outputs
        return output

    def get_gate_stats(self, reset=True):
        """Return aggregated gate routing statistics since the last reset.

        Returned keys:
          expert_{i}_mean / _std: per-expert average weight and std across samples
          max_mean / min_mean:    largest / smallest per-expert average weight
          mean_entropy:           average per-sample entropy of the gate distribution
                                  (≈log(N) means uniform, 0 means hard one-hot)
          dead_experts:           # of experts with mean weight < 1/(2N)
        """
        count = self._gate_sample_count.item()
        if count == 0:
            return {}
        mean = (self._gate_score_sum / count).detach().cpu()
        sq_mean = (self._gate_score_sq_sum / count).detach().cpu()
        std = (sq_mean - mean * mean).clamp_min_(0).sqrt()
        avg_entropy = (self._gate_entropy_sum / count).item()

        stats = {}
        for i in range(self.num_experts):
            stats[f"expert_{i}_mean"] = mean[i].item()
            stats[f"expert_{i}_std"] = std[i].item()
        stats["max_mean"] = mean.max().item()
        stats["min_mean"] = mean.min().item()
        stats["mean_entropy"] = avg_entropy
        threshold = 1.0 / (2 * self.num_experts)
        stats["dead_experts"] = int((mean < threshold).sum().item())

        if reset:
            self._gate_score_sum.zero_()
            self._gate_score_sq_sum.zero_()
            self._gate_entropy_sum.zero_()
            self._gate_sample_count.zero_()
        return stats


def collect_moe_gate_stats(model, reset=True):
    """Walk a model and collect gate stats from every MoeLayer.

    Returns a flat dict keyed by '<module_path>/<stat_name>' suitable for
    logging directly to tensorboard / wandb.
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, MoeLayer):
            for k, v in module.get_gate_stats(reset=reset).items():
                prefix = name if name else "moe"
                stats[f"{prefix}/{k}"] = v
    return stats
