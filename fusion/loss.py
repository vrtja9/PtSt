"""fusion/loss.py -- challenge eq. (2) / docs/math_fixed.md §B Step 9.

C/C++ -> Python -> R: `for(i<N) sum += (1-z[i])*exp(r[i]) - z[i]*r[i]; sum/N` -> a vectorised
torch expression + `.mean()` -> `mean((1-z)*exp(r) - z*r)`.
"""
from __future__ import annotations

import torch


def fusion_loss(r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """L = mean( (1-z)*exp(r) - z*r ), averaged over the batch (challenge eq. 2)."""
    return torch.mean((1 - z) * torch.exp(r) - z * r)


def dL_dr(r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Per-row derivative (Step 9 / docs/math_fixed.md §C): d/dr_i[(1-z_i)e^{r_i} - z_i r_i].

    This is the derivative of the UNAVERAGED per-row term, matching the identity as stated in
    docs/math_fixed.md §C verbatim -- it is NOT dL/dr_i for the batch-mean `fusion_loss` (that
    would carry an extra 1/N factor, which torch autograd applies automatically via
    `fusion_loss(...).backward()`; this function is the diagnostic/FOC building block, not a
    manual backprop step).
    """
    return (1 - z) * torch.exp(r) - z
