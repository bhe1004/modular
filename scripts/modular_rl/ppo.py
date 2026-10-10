# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO algorithm for the modular policy.

Mirrors the math of the reference custom PPO (``reference/ppo/src/ryan_ppo/ppo.py``) and
the hyper-parameter conventions of rsl_rl: clipped surrogate objective, clipped value
loss, entropy bonus, GAE, an adaptive-KL learning-rate schedule and gradient clipping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from modular_net import LOG_STD_MAX, LOG_STD_MIN, ModularActorCritic
from storage import RolloutStorage

_LR_MIN = 1e-5
_LR_MAX = 1e-2


class ModularPPO:
    def __init__(
        self,
        policy: ModularActorCritic,
        device: torch.device,
        learning_rate: float = 5.0e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_param: float = 0.2,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.0,
        desired_kl: float = 0.01,
        max_grad_norm: float = 1.0,
        schedule: str = "adaptive",
        use_clipped_value_loss: bool = True,
    ) -> None:
        self.policy = policy
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.desired_kl = desired_kl
        self.max_grad_norm = max_grad_norm
        self.schedule = schedule
        self.use_clipped_value_loss = use_clipped_value_loss

    # -- GAE ------------------------------------------------------------------

    @torch.no_grad()
    def compute_returns(self, storage: RolloutStorage, last_values: torch.Tensor) -> None:
        """Fill ``storage.returns`` / ``storage.advantages`` with GAE estimates.

        GAE is computed independently per value component (HAMNet-style): rewards, values,
        returns and advantages all carry a trailing component axis. ``dones`` is shared across
        components and broadcast over it.
        """
        advantage = torch.zeros_like(last_values)  # (num_envs, C)
        for step in reversed(range(storage.num_steps)):
            if step == storage.num_steps - 1:
                next_values = last_values
            else:
                next_values = storage.values[step + 1]
            # (num_envs,) -> (num_envs, 1) so it broadcasts over components
            next_non_terminal = (1.0 - storage.dones[step]).unsqueeze(-1)
            delta = storage.rewards[step] + self.gamma * next_values * next_non_terminal - storage.values[step]
            advantage = delta + self.gamma * self.lam * next_non_terminal * advantage
            storage.returns[step] = advantage + storage.values[step]
        storage.advantages = storage.returns - storage.values

    # -- update ---------------------------------------------------------------

    def update(self, storage: RolloutStorage) -> dict[str, float]:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        num_updates = 0

        generator = storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs, actions, old_log_probs, old_values, returns, advantages, old_mu, old_sigma in generator:
            # HAMNet-style: sum per-component advantages into a single policy advantage,
            # then normalize it per mini-batch. (old_values/returns/advantages: (mb, C))
            policy_adv = advantages.sum(dim=-1)
            policy_adv = (policy_adv - policy_adv.mean()) / (policy_adv.std() + 1e-8)

            log_probs, entropy, values, mu, sigma = self.policy.evaluate(obs, actions)

            # adaptive learning-rate via KL divergence between old and new policy
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(sigma / (old_sigma + 1e-8) + 1e-8)
                        + (old_sigma.square() + (old_mu - mu).square()) / (2.0 * sigma.square() + 1e-8)
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = kl.mean().item()
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(_LR_MIN, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(_LR_MAX, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
                    mean_kl += kl_mean

            # clipped surrogate (policy) loss, using the summed-advantage signal
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate = -policy_adv * ratio
            surrogate_clipped = -policy_adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # clipped value loss, averaged over all value heads (mb, C)
            if self.use_clipped_value_loss:
                value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - returns).square()
                value_losses_clipped = (value_clipped - returns).square()
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns - values).square().mean()

            entropy_mean = entropy.mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_mean

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # keep the state-independent log-std within sane bounds
            self.policy.log_std.data.clamp_(LOG_STD_MIN, LOG_STD_MAX)

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_mean.item()
            num_updates += 1

        num_updates = max(num_updates, 1)
        return {
            "value_loss": mean_value_loss / num_updates,
            "surrogate_loss": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "kl": mean_kl / num_updates,
            "learning_rate": self.learning_rate,
        }
