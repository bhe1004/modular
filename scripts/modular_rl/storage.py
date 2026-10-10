# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-size rollout buffer for on-policy training.

All buffers are shaped ``(num_steps, num_envs, ...)`` following the reference custom PPO
(``reference/ppo/src/ryan_ppo/storage.py``). Rewards/values/returns/advantages carry a
trailing ``num_value_components`` axis to support HAMNet-style multi-head (vector) critics;
``num_value_components == 1`` recovers the scalar case.
"""

from __future__ import annotations

import torch


class RolloutStorage:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        num_value_components: int = 1,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_value_components = num_value_components
        self.device = device

        C = num_value_components
        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, C, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, C, device=device)
        self.mus = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.sigmas = torch.zeros(num_steps, num_envs, action_dim, device=device)

        # filled by the PPO algorithm after the rollout
        self.returns = torch.zeros(num_steps, num_envs, C, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, C, device=device)

        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps:
            raise RuntimeError("RolloutStorage overflow: call clear() before re-filling.")
        i = self.step
        self.observations[i].copy_(obs)
        self.actions[i].copy_(action)
        self.log_probs[i].copy_(log_prob)
        self.rewards[i].copy_(reward)
        self.dones[i].copy_(done)
        self.values[i].copy_(value)
        self.mus[i].copy_(mu)
        self.sigmas[i].copy_(sigma)
        self.step += 1

    def clear(self) -> None:
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        """Yield flattened mini-batches over the whole rollout for ``num_epochs`` epochs.

        Per item: ``(obs, actions, log_probs, values, returns, advantages, mus, sigmas)``
        where ``values``/``returns``/``advantages`` are ``(mini_batch, num_value_components)``.
        """
        batch_size = self.num_steps * self.num_envs
        mini_batch_size = batch_size // num_mini_batches
        C = self.num_value_components

        obs = self.observations.reshape(batch_size, -1)
        actions = self.actions.reshape(batch_size, -1)
        log_probs = self.log_probs.reshape(batch_size)
        values = self.values.reshape(batch_size, C)
        returns = self.returns.reshape(batch_size, C)
        advantages = self.advantages.reshape(batch_size, C)
        mus = self.mus.reshape(batch_size, -1)
        sigmas = self.sigmas.reshape(batch_size, -1)

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                idx = indices[start : start + mini_batch_size]
                yield (
                    obs[idx],
                    actions[idx],
                    log_probs[idx],
                    values[idx],
                    returns[idx],
                    advantages[idx],
                    mus[idx],
                    sigmas[idx],
                )
