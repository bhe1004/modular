# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner for the modular PPO agent.

Plays the role of rsl_rl's ``OnPolicyRunner`` for this custom stack: it owns the policy,
the PPO algorithm and the rollout storage, drives the rollout -> GAE -> update loop, logs
to TensorBoard + console, and saves/loads checkpoints. It consumes the observation
``policy`` group from a :class:`ModularVecEnvWrapper`-wrapped environment.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from modular_net import ModularActorCritic
from ppo import ModularPPO
from storage import RolloutStorage


def _extract_policy_obs(obs) -> torch.Tensor:
    """Extract the ``policy`` observation group tensor from a (Tensor)Dict or tensor."""
    if hasattr(obs, "keys"):
        keys = list(obs.keys())
        if "policy" in keys:
            return obs["policy"]
        return obs[keys[0]]
    return obs


class ModularOnPolicyRunner:
    def __init__(self, env, agent_cfg, log_dir: str | None = None, device: str = "cuda:0") -> None:
        self.env = env
        self.cfg = agent_cfg
        self.device = device
        self.log_dir = log_dir

        self.num_steps_per_env = agent_cfg.num_steps_per_env
        self.save_interval = agent_cfg.save_interval

        # infer observation / action dimensions from the environment
        obs = _extract_policy_obs(env.get_observations()).to(device)
        self.num_obs = obs.shape[-1]
        self.num_actions = env.num_actions
        self.num_envs = env.num_envs

        # number of critic value heads: one per reward component if multi_critic is on
        pcfg = agent_cfg.policy
        env_components = getattr(env, "num_reward_components", 1)
        self.multi_critic = getattr(pcfg, "multi_critic", False) and env_components > 1
        self.num_value_components = env_components if self.multi_critic else 1
        self.reward_component_names = (
            list(getattr(env, "reward_component_names", ["reward"]))
            if self.multi_critic
            else ["reward"]
        )

        # build the modular policy
        self.policy = ModularActorCritic(
            self.num_obs,
            self.num_actions,
            num_modules=pcfg.num_modules,
            hidden_dims=list(pcfg.hidden_dims),
            modulation_hidden_dims=list(pcfg.modulation_hidden_dims),
            film_gate=pcfg.film_gate,
            use_layernorm=pcfg.use_layernorm,
            activation=pcfg.activation,
            init_noise_std=pcfg.init_noise_std,
            normalize_obs=pcfg.normalize_obs,
            num_value_components=self.num_value_components,
        ).to(device)

        # build PPO
        acfg = agent_cfg.algorithm
        self.alg = ModularPPO(
            self.policy,
            device=device,
            learning_rate=acfg.learning_rate,
            gamma=acfg.gamma,
            lam=acfg.lam,
            clip_param=acfg.clip_param,
            num_learning_epochs=acfg.num_learning_epochs,
            num_mini_batches=acfg.num_mini_batches,
            value_loss_coef=acfg.value_loss_coef,
            entropy_coef=acfg.entropy_coef,
            desired_kl=acfg.desired_kl,
            max_grad_norm=acfg.max_grad_norm,
            schedule=acfg.schedule,
        )

        self.storage = RolloutStorage(
            self.num_steps_per_env,
            self.num_envs,
            self.num_obs,
            self.num_actions,
            device,
            num_value_components=self.num_value_components,
        )

        self.writer = SummaryWriter(log_dir) if log_dir is not None else None
        self.current_learning_iteration = 0
        self.tot_timesteps = 0

        if self.multi_critic:
            print(
                f"[modular_rl] multi-head critic with {self.num_value_components} value heads "
                f"(reward terms: {self.reward_component_names})"
            )
        else:
            print("[modular_rl] scalar critic (single value head)")

    # -- training -------------------------------------------------------------

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = _extract_policy_obs(self.env.get_observations()).to(self.device)

        reward_buffer: deque = deque(maxlen=100)
        length_buffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.num_envs, device=self.device)
        cur_episode_length = torch.zeros(self.num_envs, device=self.device)

        gamma = self.alg.gamma
        start_iter = self.current_learning_iteration
        total_iter = start_iter + num_learning_iterations

        for iteration in range(start_iter, total_iter):
            start = time.time()
            self.storage.clear()

            # ---- rollout ----
            for _ in range(self.num_steps_per_env):
                # update running observation statistics (no-op if disabled)
                self.policy.update_normalization(obs)
                action, log_prob, mu, sigma = self.policy.act(obs)
                value = self.policy.get_value(obs)  # (N, C)

                next_obs, reward, dones, extras = self.env.step(action)
                next_obs = _extract_policy_obs(next_obs).to(self.device)
                reward = reward.to(self.device)
                dones = dones.to(self.device).float()

                # per-component reward vector (N, C); scalar env reward -> single component
                if self.multi_critic and "reward_components" in extras:
                    reward_vec = extras["reward_components"].to(self.device)
                else:
                    reward_vec = reward.unsqueeze(-1)

                # bootstrap reward on time-out (truncation), not on true termination
                if "time_outs" in extras:
                    time_outs = extras["time_outs"].to(self.device).float().unsqueeze(-1)
                    reward_vec = reward_vec + gamma * value * time_outs

                self.storage.add(obs, action, log_prob, reward_vec, dones, value, mu, sigma)
                obs = next_obs

                # episode bookkeeping (track the scalar total return for logging)
                cur_reward_sum += reward
                cur_episode_length += 1
                done_idx = dones.bool()
                if done_idx.any():
                    reward_buffer.extend(cur_reward_sum[done_idx].cpu().tolist())
                    length_buffer.extend(cur_episode_length[done_idx].cpu().tolist())
                    cur_reward_sum[done_idx] = 0.0
                    cur_episode_length[done_idx] = 0.0

            # ---- learning step ----
            last_values = self.policy.get_value(obs)
            self.alg.compute_returns(self.storage, last_values)
            losses = self.alg.update(self.storage)

            self.current_learning_iteration = iteration + 1
            self.tot_timesteps += self.num_steps_per_env * self.num_envs
            elapsed = time.time() - start
            fps = int(self.num_steps_per_env * self.num_envs / max(elapsed, 1e-9))
            mean_reward = statistics.mean(reward_buffer) if len(reward_buffer) > 0 else 0.0
            mean_length = statistics.mean(length_buffer) if len(length_buffer) > 0 else 0.0

            self._log(iteration, total_iter, losses, mean_reward, mean_length, fps)

            if self.log_dir is not None and (
                iteration % self.save_interval == 0 or iteration == total_iter - 1
            ):
                self.save(os.path.join(self.log_dir, f"model_{iteration}.pt"))

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def _log(self, iteration, total_iter, losses, mean_reward, mean_length, fps) -> None:
        if self.writer is not None:
            self.writer.add_scalar("Train/mean_reward", mean_reward, iteration)
            self.writer.add_scalar("Train/mean_episode_length", mean_length, iteration)
            self.writer.add_scalar("Loss/value", losses["value_loss"], iteration)
            self.writer.add_scalar("Loss/surrogate", losses["surrogate_loss"], iteration)
            self.writer.add_scalar("Loss/entropy", losses["entropy"], iteration)
            self.writer.add_scalar("Policy/kl", losses["kl"], iteration)
            self.writer.add_scalar("Policy/learning_rate", losses["learning_rate"], iteration)
            self.writer.add_scalar("Perf/fps", fps, iteration)
        print(
            f"[modular_rl] it {iteration + 1}/{total_iter} | "
            f"reward {mean_reward:8.3f} | len {mean_length:6.1f} | "
            f"v_loss {losses['value_loss']:.4f} | surr {losses['surrogate_loss']:.4f} | "
            f"kl {losses['kl']:.4f} | lr {losses['learning_rate']:.2e} | fps {fps}"
        )

    # -- checkpoint / inference ----------------------------------------------

    def save(self, path: str, infos=None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )
        print(f"[modular_rl] saved checkpoint: {path}")

    def load(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_learning_iteration = checkpoint.get("iter", 0)
        return checkpoint.get("infos")

    def get_inference_policy(self, device=None):
        self.policy.eval()
        if device is not None:
            self.policy.to(device)
            self.device = device

        def policy(obs):
            obs_tensor = _extract_policy_obs(obs).to(self.device)
            return self.policy.act_inference(obs_tensor)

        return policy
