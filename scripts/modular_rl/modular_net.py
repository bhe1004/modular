# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular actor-critic policy built on the soft mixture-of-experts modules.

Both the actor and the critic are :class:`ModularMLP` networks: a context-conditioned
modulation network routes per-layer over ``num_modules`` expert weight matrices. The
routing context is the observation vector itself. The action distribution is a diagonal
Gaussian with a state-independent ``log_std`` parameter (as in the reference custom PPO).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import Normal

from modular_modules import GateModMLP, ModulationNetwork
from normalization import ObsNormalizer

# bounds for the state-independent log-std (std in roughly [0.018, 2.72])
LOG_STD_MIN = -4.0
LOG_STD_MAX = 1.0


class ModularMLP(nn.Module):
    """A single modular network: routing modulation net + expert GateModMLP.

    The same observation vector serves as both the network input and the routing context.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_modules: int,
        hidden_dims: list[int],
        modulation_hidden_dims: list[int],
        film_gate: bool = True,
        use_layernorm: bool = True,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.experts = GateModMLP(
            in_dim,
            hidden_dims,
            out_dim,
            num_modules=num_modules,
            gate=film_gate,
            layernorm=use_layernorm,
        )
        num_layers = len(hidden_dims) + 1
        gate_dims = list(hidden_dims) if film_gate else None
        self.modnet = ModulationNetwork(
            ctx_dim=in_dim,
            num_layers=num_layers,
            num_modules=num_modules,
            hidden_dims=modulation_hidden_dims,
            gate_dims=gate_dims,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ps, gates = self.modnet(x)
        return self.experts(x, ps, gates)


class ModularActorCritic(nn.Module):
    """Diagonal-Gaussian actor-critic with modular actor and critic networks."""

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        num_modules: int = 4,
        hidden_dims: list[int] | None = None,
        modulation_hidden_dims: list[int] | None = None,
        film_gate: bool = True,
        use_layernorm: bool = True,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        normalize_obs: bool = True,
        num_value_components: int = 1,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]
        modulation_hidden_dims = modulation_hidden_dims or [128, 128]
        self.num_obs = num_obs
        self.num_actions = num_actions
        # number of critic value heads (one per reward component, HAMNet-style)
        self.num_value_components = num_value_components

        self.obs_normalizer = ObsNormalizer(num_obs) if normalize_obs else nn.Identity()

        self.actor = ModularMLP(
            num_obs,
            num_actions,
            num_modules,
            hidden_dims,
            modulation_hidden_dims,
            film_gate=film_gate,
            use_layernorm=use_layernorm,
            activation=activation,
        )
        self.critic = ModularMLP(
            num_obs,
            num_value_components,
            num_modules,
            hidden_dims,
            modulation_hidden_dims,
            film_gate=film_gate,
            use_layernorm=use_layernorm,
            activation=activation,
        )
        self.log_std = nn.Parameter(torch.ones(num_actions) * math.log(init_noise_std))

    # -- distribution helpers -------------------------------------------------

    def _clamped_std(self) -> torch.Tensor:
        return self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def _distribution(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        obs = self.obs_normalizer(obs)
        mean = self.actor(obs)
        std = self._clamped_std().expand_as(mean)
        return Normal(mean, std), mean

    @torch.no_grad()
    def update_normalization(self, obs: torch.Tensor) -> None:
        """Update the running observation statistics (training only)."""
        if isinstance(self.obs_normalizer, ObsNormalizer):
            self.obs_normalizer.update(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns ``(action, log_prob, mean, std)``."""
        dist, mean = self._distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        std = self._clamped_std().expand_as(mean)
        return action, log_prob, mean, std

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic (mean) action for evaluation."""
        return self.actor(self.obs_normalizer(obs))

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate stored transitions during the update.

        Returns ``(log_prob, entropy, value, mean, std)``.
        """
        dist, mean = self._distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(self.obs_normalizer(obs))  # (B, num_value_components)
        std = self._clamped_std().expand_as(mean)
        return log_prob, entropy, value, mean, std

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns per-component values, shape ``(B, num_value_components)``."""
        return self.critic(self.obs_normalizer(obs))
