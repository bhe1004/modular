# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Agent configuration for the custom modular-network PPO stack (``scripts/modular_rl``).

Mirrors the structure of :mod:`rsl_rl_ppo_cfg` so that ``train.py --task=<name>`` resolves
this through ``@hydra_task_config`` exactly like rsl_rl. The network/algorithm code lives in
``scripts/modular_rl``; this file only holds the hyper-parameters.
"""

from isaaclab.utils import configclass


@configclass
class ModularPolicyCfg:
    """Configuration of the modular (soft mixture-of-experts) actor-critic network."""

    num_modules: int = 4
    """Number of expert modules blended per layer."""
    hidden_dims: list[int] = [256, 128, 64]
    """Hidden layer sizes of the expert MLPs (actor and critic)."""
    modulation_hidden_dims: list[int] = [128, 128]
    """Hidden layer sizes of the context->routing modulation network."""
    film_gate: bool = True
    """Enable FiLM-style per-layer output gating from the modulation network."""
    use_layernorm: bool = True
    """Apply affine-less LayerNorm inside the expert hidden layers."""
    activation: str = "elu"
    """Activation used by the modulation network backbone."""
    init_noise_std: float = 1.0
    """Initial standard deviation of the diagonal Gaussian action distribution."""
    normalize_obs: bool = True
    """Apply empirical (running mean/var) observation normalization."""
    multi_critic: bool = True
    """HAMNet-style multi-head critic: one value head per reward term, with per-component GAE
    and summed advantages for the policy gradient. Falls back to a scalar critic if the
    environment exposes no reward manager."""


@configclass
class ModularAlgorithmCfg:
    """Configuration of the PPO algorithm."""

    learning_rate: float = 5.0e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    schedule: str = "adaptive"


@configclass
class ModularRlPpoRunnerCfg:
    """Top-level runner configuration consumed by ``ModularOnPolicyRunner``."""

    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 32
    max_iterations: int = 1000
    save_interval: int = 50
    experiment_name: str = "modular_humanoid"
    run_name: str = ""
    logger: str = "tensorboard"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    clip_actions: float | None = 1.0
    policy: ModularPolicyCfg = ModularPolicyCfg()
    algorithm: ModularAlgorithmCfg = ModularAlgorithmCfg()
