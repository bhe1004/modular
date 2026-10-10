# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Empirical observation normalization (Welford running mean/variance).

Mirrors the reference custom PPO (``reference/ppo/src/ryan_ppo/normalization.py``). Buffers
are registered so they are saved/loaded with the policy ``state_dict`` and frozen during
evaluation/play.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ObsNormalizer(nn.Module):
    def __init__(self, num_obs: int, epsilon: float = 1e-5, clip: float = 10.0) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.clip = clip
        self.register_buffer("mean", torch.zeros(num_obs))
        self.register_buffer("var", torch.ones(num_obs))
        self.register_buffer("count", torch.tensor(epsilon))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics from a batch ``x: (B, num_obs)``."""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean += delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * (self.count * batch_count / total_count)
        self.var = m2 / total_count
        self.count = total_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = (x - self.mean) / torch.sqrt(self.var + self.epsilon)
        if self.clip is not None:
            norm = norm.clamp(-self.clip, self.clip)
        return norm
