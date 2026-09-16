"""fusion/config.py -- single source of truth for every run setting.

C/C++ -> Python -> R: `struct Config {...}` passed by pointer everywhere ->
a frozen `@dataclass` passed by reference -> a named `list()` used as an
environment. Every saved figure/table is written together with `to_json()`
of the Config that produced it plus the current git commit hash (CLAUDE.md
§2.4), so a figure can always be traced back to the exact settings and code.

# DECISION (design choice, not locked by CLAUDE.md): every fusion/*.py function
# that depends on run settings takes `cfg: Config` as its first positional
# argument (e.g. `dgp.m_t(cfg, t)`) rather than reading a module-level global,
# unlike fusion_numpy.py's global CFG dict. Consequence of changing this: every
# call site in dgp.py/data.py/model.py/train.py/estimate.py/evaluate.py would
# need updating together; keep it consistent if revisited.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    # --- DECISION (CP1, notes/decisions.md): DGP = default, docs/math_fixed.md §D. ---
    # Downstream if these change: fusion/dgp.py closed forms (a_t, ES_closed, theta_star,
    # alpha_star, density_grid) are pure functions of these fields, so they update
    # automatically; but tests/test_dgp.py T1-T3 tolerances, the Phase 1 figures, and every
    # later-phase comparison against theta*/alpha* (Phase 3 training diagnostics, Phase 4
    # stress tests) were tuned by eye against these particular numbers and must be re-run
    # and re-read, not just re-executed.
    m: int = 6                    # last year with population (P_t) data
    M: int = 9                    # last year with survey (S_t) data, m < M
    n: int = 2000                 # i.i.d. draws per (t, z) cell
    sigma: float = 1.0            # P_{t,Y} = N(m_t, sigma^2)
    beta: float = 1.5             # selection steepness in pi_t(y) = c_t * Phi(beta*y)
    m_t_intercept: float = -0.5   # m_t(t) = m_t_intercept + m_t_slope * t
    m_t_slope: float = 0.25
    c_t_intercept: float = 0.9    # c_t(t) = c_t_intercept + c_t_slope * t, c_t in (0,1]
    c_t_slope: float = -0.06

    # --- DECISION (Phase 3 default, docs/math_fixed.md §F "already run" settings): these
    # gave FOC 0.98-1.03 in the numpy twin *when the twin itself used them* (fusion_numpy.py's
    # own CFG differs: lr=1e-2, wd=1e-4, epochs=150 -- see LOG.md Phase 0 entry). Treated here
    # as the first-run proposal for CP3, not yet re-validated with the torch pipeline; CP3 may
    # revise any of the five fields below based on the FOC table and theta~ vs theta* figure. ---
    H: int = 32                   # ThetaNet hidden width (1 -> H -> H -> 1)
    lr: float = 3e-3
    wd: float = 1e-5
    epochs: int = 400
    batch: int = 256
    val_frac: float = 0.2         # stratified 80/20 split inside every (t,z) cell

    # --- DECISION: two independent RNG streams (CLAUDE.md §2.4), never shared, so that
    # re-drawing bootstrap samples or re-shuffling batches cannot silently perturb the DGP. ---
    seed_data: int = 0            # DGP draws, splits, bootstrap
    seed_model: int = 1           # parameter init, batch shuffling

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def hash(self) -> str:
        # C/C++ -> Python -> R: CRC32 of a serialized struct -> sha256 of the sorted-key JSON
        # string -> digest(serialize(list(...))) -- all "canonicalize, then hash" the same way.
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:8]


def git_commit_hash() -> str:
    """Best-effort short commit hash for tagging saved figures/tables (CLAUDE.md §2.4)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"
