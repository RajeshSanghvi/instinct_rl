# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x, update: bool | None = None):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values
            update: Whether to fold `x` into the running statistics before normalizing.
                Defaults to `self.training` (existing implicit behavior) when not given.
                Pass explicitly to decouple "update statistics" from "use statistics",
                e.g. to normalize with frozen stats even while the module is in train mode.

        Returns:
            ndarray or Variable: Normalized output values
        """
        if update is None:
            update = self.training
        if update:
            with torch.no_grad():
                self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        self.update_from_moments(mean_x, var_x, x.shape[0])

    @torch.jit.unused
    def update_from_moments(self, mean_x, var_x, count_x):
        """Fold a batch's precomputed (mean, var, count) into the running statistics.

        Splitting this out from `update` lets callers combine several partial batches (e.g. one
        per DDP rank, via all_reduce'd sum(x)/sum(x**2)/count) into a single global batch first,
        then apply one identical update everywhere -- instead of every rank updating once from
        its own divergent local data. `mean_x`/`var_x` must have the same leading-1 shape as
        `_mean`/`_var` (i.e. computed with `keepdim=True`).
        """
        if self.until is not None and self.count >= self.until:
            return

        # `count_x` may arrive as a float tensor (e.g. the DDP path sums it inside a float
        # all_reduce buffer alongside sum(x)/sum(x**2)); `count` is a long buffer, so an in-place
        # add needs an explicit cast rather than relying on implicit tensor-dtype promotion.
        self.count += int(count_x)
        rate = count_x / self.count

        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean

    def export(self, path):
        np.savez(
            path,
            mean=self._mean.cpu().numpy(),
            std=self._std.cpu().numpy(),
            eps=self.eps,
            until=self.until,
        )


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew, update: bool | None = None):
        if update is None:
            update = self.training
        if update:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
