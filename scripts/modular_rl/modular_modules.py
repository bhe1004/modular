# Copyright (c) 2025. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core building blocks of the modular (soft mixture-of-experts) network.

Reimplemented in plain PyTorch (``torch.einsum``) from the HAMNet reference
(``reference/HAMNet/ham/src/ham/models/rl/net/gate_mod_mlp.py``). Only the core
soft-modular mechanism is kept: every layer holds ``M`` expert weight matrices and a
context-conditioned modulation network produces per-layer softmax weights that softly
blend them. The point-cloud cross-attention / geometry encoder / hypernetwork pieces of
HAMNet are intentionally dropped; the routing context here is simply the observation
vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name in ("none", "identity"):
        return nn.Identity()
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(ACTIVATIONS) + ['none']}")
    return ACTIVATIONS[name]()


class GateModLinear(nn.Module):
    """A single soft-modular linear layer.

    Holds ``num_modules`` expert weight matrices ``W: (M, d_out, d_in)`` and biases
    ``b: (M, d_out)``. Given per-sample routing weights ``p: (B, M)`` the effective layer
    is the soft blend ``y = act(LayerNorm(sum_m p_m (W_m x) + sum_m p_m b_m))`` with an
    optional FiLM output gate ``Wx *= (1 + gate)``.
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_modules: int,
        activation: str = "tanh",
        layernorm: bool = True,
        gate: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_modules = num_modules
        self.gate = gate

        self.W = nn.Parameter(torch.empty(num_modules, d_out, d_in))
        self.b = nn.Parameter(torch.zeros(num_modules, d_out))
        self.reset_parameters()

        self.act = _make_activation(activation)
        # Affine-less LayerNorm: the FiLM gate (when enabled) provides the modulation.
        self.norm = nn.LayerNorm(d_out, elementwise_affine=False) if layernorm else nn.Identity()

    def reset_parameters(self) -> None:
        # Initialize each expert independently with orthogonal weights.
        for m in range(self.num_modules):
            nn.init.orthogonal_(self.W[m], gain=1.0)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor, p: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: layer input, shape ``(B, d_in)``.
            p: per-sample module routing weights (softmax), shape ``(B, M)``.
            gate: optional FiLM gate, shape ``(B, d_out)``.
        Returns:
            Layer output, shape ``(B, d_out)``.
        """
        # soft-selected weight matmul and bias
        wx = torch.einsum("bm,moi,bi->bo", p, self.W, x)
        bias = torch.einsum("bm,mo->bo", p, self.b)
        if self.gate and gate is not None:
            wx = wx * (1.0 + gate)
        return self.act(self.norm(wx + bias))

    def extra_repr(self) -> str:
        return f"{self.d_in}->{self.d_out}, num_modules={self.num_modules}, gate={self.gate}"


class GateModMLP(nn.Module):
    """Stack of :class:`GateModLinear` layers forming the modular MLP.

    Hidden layers use ``tanh`` + affine-less LayerNorm (and optional FiLM gating); the final
    layer is a plain soft-modular linear map (no activation, no norm, no gate).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        out_dim: int,
        num_modules: int,
        gate: bool = True,
        layernorm: bool = True,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        last_idx = len(dims) - 2
        layers = []
        for layer_idx, (d_i, d_o) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = layer_idx == last_idx
            layers.append(
                GateModLinear(
                    d_i,
                    d_o,
                    num_modules,
                    activation="tanh" if not is_last else "none",
                    layernorm=layernorm and not is_last,
                    gate=gate and not is_last,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.num_layers = len(self.layers)
        # number of hidden layers that receive a FiLM gate
        self.num_gated_layers = self.num_layers - 1 if gate else 0

    def forward(self, x: torch.Tensor, ps: torch.Tensor, gates: list[torch.Tensor] | None = None) -> torch.Tensor:
        """
        Args:
            x: network input, shape ``(B, in_dim)``.
            ps: per-layer routing weights, shape ``(B, num_layers, num_modules)``.
            gates: optional list (len == num_gated_layers) of FiLM gates ``(B, d_out_l)``.
        Returns:
            Network output, shape ``(B, out_dim)``.
        """
        for layer_idx, layer in enumerate(self.layers):
            gate = gates[layer_idx] if (gates is not None and layer_idx < len(gates)) else None
            x = layer(x, ps[:, layer_idx, :], gate)
        return x


class ModulationNetwork(nn.Module):
    """Maps the routing context to per-layer softmax module weights (and optional gates).

    A small shared backbone MLP feeds: (1) a routing head producing ``(B, L, M)`` logits
    that are softmaxed over modules per layer, and (2) optional per-layer FiLM gate heads
    producing ``(B, d_out_l)`` for the hidden layers.
    """

    def __init__(
        self,
        ctx_dim: int,
        num_layers: int,
        num_modules: int,
        hidden_dims: list[int],
        gate_dims: list[int] | None = None,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules

        backbone = []
        prev = ctx_dim
        for h in hidden_dims:
            backbone.append(nn.Linear(prev, h))
            backbone.append(_make_activation(activation))
            prev = h
        self.backbone = nn.Sequential(*backbone) if backbone else nn.Identity()
        self.feat_dim = prev

        self.route_head = nn.Linear(self.feat_dim, num_layers * num_modules)

        # Optional FiLM gate heads, one per gated (hidden) layer.
        self.gate_dims = gate_dims
        if gate_dims:
            self.gate_heads = nn.ModuleList([nn.Linear(self.feat_dim, d) for d in gate_dims])
        else:
            self.gate_heads = None

    def forward(self, ctx: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Args:
            ctx: routing context, shape ``(B, ctx_dim)``.
        Returns:
            ps: per-layer routing weights, shape ``(B, num_layers, num_modules)``.
            gates: optional list of FiLM gates, each ``(B, d_out_l)``.
        """
        feat = self.backbone(ctx)
        logits = self.route_head(feat).view(-1, self.num_layers, self.num_modules)
        ps = F.softmax(logits, dim=-1)
        gates = None
        if self.gate_heads is not None:
            gates = [head(feat) for head in self.gate_heads]
        return ps, gates
