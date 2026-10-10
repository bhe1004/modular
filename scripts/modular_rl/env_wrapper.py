# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone vectorized-environment wrapper for the modular_rl stack.

This is a self-contained replacement for ``isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`` so the
modular_rl stack has **no dependency on the rsl_rl package** (the original wrapper subclasses
``rsl_rl.env.VecEnv``). It exposes exactly the contract the :class:`ModularOnPolicyRunner`
needs: ``num_envs``, ``num_actions``, ``device``, ``max_episode_length``,
``episode_length_buf`` (get/set), ``unwrapped``, ``reset``, ``get_observations``, ``step``,
``close`` and ``seed``.

Observations are returned as the raw IsaacLab observation dict (e.g. ``{"policy": tensor}``);
the runner extracts the ``policy`` group.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv


class ModularVecEnvWrapper:
    """Minimal IsaacLab vec-env adapter (no rsl_rl dependency)."""

    def __init__(self, env: ManagerBasedRLEnv | DirectRLEnv, clip_actions: float | None = None) -> None:
        if not isinstance(env.unwrapped, (ManagerBasedRLEnv, DirectRLEnv)):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. "
                f"Environment type: {type(env)}"
            )

        self.env = env
        self.clip_actions = clip_actions

        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length

        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)

        # per-term reward components (for HAMNet-style multi-head critics)
        if hasattr(self.unwrapped, "reward_manager"):
            self.reward_component_names = list(self.unwrapped.reward_manager.active_terms)
        else:
            self.reward_component_names = ["reward"]
        self.num_reward_components = len(self.reward_component_names)

        self._modify_action_space()

        # reset at the start since the runner does not call reset itself
        self.env.reset()

    # -- properties -----------------------------------------------------------

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:
        return self.env.unwrapped

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    # -- MDP ------------------------------------------------------------------

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[dict, dict]:
        obs_dict, extras = self.env.reset()
        return obs_dict, extras

    def get_observations(self) -> dict:
        if hasattr(self.unwrapped, "observation_manager"):
            return self.unwrapped.observation_manager.compute()
        return self.unwrapped._get_observations()

    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # time-out info for bootstrapping on infinite-horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        # per-term reward vector whose sum equals `rew`
        # (_step_reward holds value/dt per term; multiply by step_dt to recover contributions)
        if hasattr(self.unwrapped, "reward_manager"):
            extras["reward_components"] = (
                self.unwrapped.reward_manager._step_reward * self.unwrapped.step_dt
            ).clone()
        return obs_dict, rew, dones, extras

    def close(self):
        return self.env.close()

    # -- helpers --------------------------------------------------------------

    def _modify_action_space(self) -> None:
        if self.clip_actions is None:
            return
        self.env.unwrapped.single_action_space = gym.spaces.Box(
            low=-self.clip_actions, high=self.clip_actions, shape=(self.num_actions,)
        )
        self.env.unwrapped.action_space = gym.vector.utils.batch_space(
            self.env.unwrapped.single_action_space, self.num_envs
        )
