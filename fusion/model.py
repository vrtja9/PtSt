"""fusion/model.py -- r(t,y) = theta(y) + alpha[t] (docs/math_fixed.md §B Step 7/9).

# DECISION (notes/decisions.md, "Uncertainty raised mid-Phase-2"): ThetaNet is 1->H->1
# (a single hidden layer), matching fusion_numpy.py's init_params/theta_of exactly, NOT the
# "1->H->H->1" text in CLAUDE.md §3 -- T4 requires loading fusion_numpy.init_params's shapes
# (W1:(H,1), b1:(H,), w2:(H,), b2:scalar) bit-for-bit, which only a single hidden layer permits.
# User confirmed this resolution before any of this file was written.

C/C++ -> Python -> R: a hand-written `matmul`+`relu` in a struct-of-weights -> `nn.Module` with
`nn.Parameter` tensors -> an S4/R6 object holding weight matrices, methods calling `%*%`/`pmax`.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ThetaNet(nn.Module):
    """theta(y): MLP(1 -> H -> 1, ReLU). Optional `bounded=B` caps |theta| via B*tanh(./B)
    (the "bounded head" CP3 reply-template option; identity when bounded=None)."""

    def __init__(self, H: int, bounded: Optional[float] = None):
        super().__init__()
        self.W1 = nn.Parameter(torch.zeros(H, 1))
        self.b1 = nn.Parameter(torch.zeros(H))
        self.w2 = nn.Parameter(torch.zeros(H))
        self.b2 = nn.Parameter(torch.zeros(()))
        self.bounded = bounded
        nn.init.normal_(self.W1, std=1.0)
        nn.init.normal_(self.w2, std=1.0 / (H ** 0.5))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """C/C++ -> Python -> R: scalar loop over H units -> `W1 @ y + b1` (H,B) -> `W1 %*% y`."""
        pre = self.W1 @ y[None, :] + self.b1[:, None]   # (H, B)
        h = torch.relu(pre)                              # Step 9: ReLU(pre)
        theta = self.w2 @ h + self.b2                     # (B,)
        if self.bounded is not None:
            theta = self.bounded * torch.tanh(theta / self.bounded)
        return theta

    def load_from_numpy(self, p: dict) -> None:
        """Copy fusion_numpy.init_params(rng, H) values in verbatim, for test T4."""
        with torch.no_grad():
            self.W1.copy_(torch.as_tensor(p["W1"], dtype=self.W1.dtype))
            self.b1.copy_(torch.as_tensor(p["b1"], dtype=self.b1.dtype))
            self.w2.copy_(torch.as_tensor(p["w2"], dtype=self.w2.dtype))
            self.b2.copy_(torch.as_tensor(p["b2"], dtype=self.b2.dtype))


class AlphaTable(nn.Module):
    """alpha in R^m with alpha[m] frozen at 0 (Step 7: the identifiability anchor).

    # DECISION (kickoff PROMPT_kickoff.md Phase 2 step 1): implemented as concatenation with a
    # constant zero buffer, not a boolean mask, so alpha[m] is structurally always exactly 0
    # (test T8) -- it is a `register_buffer`, never a `nn.Parameter`, so no optimizer can touch it.
    """

    def __init__(self, m: int):
        super().__init__()
        self.m = m
        self.free = nn.Parameter(torch.zeros(m - 1))   # alpha(1..m-1); alpha(m) is the buffer below
        self.register_buffer("zero", torch.zeros(1))

    def forward(self) -> torch.Tensor:
        """Returns the full length-m alpha vector, 0-indexed: alpha_vec[t-1] = alpha(t)."""
        return torch.cat([self.free, self.zero])

    def at(self, t: torch.Tensor) -> torch.Tensor:
        """alpha(t) for 1-based year indices t (int64 tensor)."""
        return self.forward()[t.long() - 1]


def ratio_model(theta_net: ThetaNet, alpha_table: AlphaTable, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """r(t,y) = theta(y) + alpha(t) (Step 9's r_i, before the loss is applied)."""
    return theta_net(y) + alpha_table.at(t)
