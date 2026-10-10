# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random


def add_modular_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add modular_rl CLI arguments to the parser."""
    arg_group = parser.add_argument_group("modular_rl", description="Arguments for the modular RL agent.")
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"tensorboard"}, help="Logger module to use."
    )


def update_modular_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Override the agent configuration with non-hydra CLI arguments."""
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    return agent_cfg
