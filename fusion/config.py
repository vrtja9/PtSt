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
import math
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
    beta: float = 0.6             # selection steepness in pi_t(y) = c_t * Phi(beta*y); (R3): beta*sigma < 1/sqrt(2); tail index kappa = 3.78
    m_t_intercept: float = -0.5   # m_t(t) = m_t_intercept + m_t_slope * t
    m_t_slope: float = 0.25
    c_t_intercept: float = 0.9    # c_t(t) = c_t_intercept + c_t_slope * t, c_t in (0,1]
    c_t_slope: float = -0.06

    # --- DECISION (CP3, notes/decisions.md, 2026-09-16): the docs/math_fixed.md §F first-run
    # proposal (lr=3e-3, unbounded ThetaNet) gave a noisy/spiky val curve still descending at
    # epoch ~400 (Step 9's "R_hat_n unbounded below" warning made concrete) and theta~ blowing
    # up to ~380 at y=-6 vs theta*'s <50 there -- an unstable tail extrapolation. User adopted
    # the proposed fix: lr lowered to 1e-3, and ThetaNet given a bounded head (theta_bounded).
    # Downstream if revisited: re-run fusion.run --phase 3 and re-check the val curve, FOC table,
    # and theta~ vs theta* figure exactly as this decision was evidenced (LOG.md, "Phase 3"). ---
    H: int = 32                   # ThetaNet hidden width (1 -> H -> 1)
    lr: float = 1e-3
    wd: float = 1e-5
    epochs: int = 400
    batch: int = 256
    theta_bounded: float = 20.0   # ThetaNet(H, bounded=theta_bounded); None = unbounded (pre-CP3)
    val_frac: float = 0.2         # stratified 80/20 split inside every (t,z) cell

    # --- DECISION: two independent RNG streams (CLAUDE.md §2.4), never shared, so that
    # re-drawing bootstrap samples or re-shuffling batches cannot silently perturb the DGP. ---
    seed_data: int = 0            # DGP draws, splits, bootstrap
    seed_model: int = 1           # parameter init, batch shuffling

    # --- DECISION (2026-09-18, authorized correction to the frozen spec, docs/math_fixed.md §G):
    # beta*sigma >= 1/sqrt(2) makes the importance weight's third moment infinite (R3), silently
    # invalidating the delta-method SE and the Hajek estimator's O(1/n) bias expansion -- so it is
    # rejected by default. allow_heavy_tails=True is the explicit opt-in for a deliberate
    # heavy-tailed stress-test config; no existing code path (Phase 1-5) sets it, since none of
    # them construct a Config with beta*sigma >= 1/sqrt(2). ---
    allow_heavy_tails: bool = False

    def __post_init__(self) -> None:
        if self.beta * self.sigma >= 1 / math.sqrt(2) and not self.allow_heavy_tails:
            kappa = 1 + 1 / (self.beta ** 2 * self.sigma ** 2)
            raise ValueError(
                f"beta*sigma={self.beta * self.sigma:.4f} >= 1/sqrt(2)~=0.7071 violates (R3): "
                f"tail index kappa={kappa:.2f} <= 3, so the importance weight's third moment is infinite "
                f"and the Hajek estimator's O(1/n) bias expansion does not apply "
                f"(docs/math_fixed.md §G). Pass allow_heavy_tails=True for a deliberate "
                f"heavy-tailed stress-test config."
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def hash(self) -> str:
        # C/C++ -> Python -> R: CRC32 of a serialized struct -> sha256 of the sorted-key JSON
        # string -> digest(serialize(list(...))) -- all "canonicalize, then hash" the same way.
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:8]


def git_commit_hash() -> str:
    """Best-effort short commit hash for tagging saved figures/tables (CLAUDE.md §2.4).

    # INVARIANT (2026-09-18, corrected per C1): the recorded hash is the commit containing the
    # CODE AND CONFIG that produced the artifact -- the commit from which re-running the
    # generating command reproduces the PNG. It is NOT the commit containing the artifact file
    # itself (a commit cannot embed its own hash), and it is NOT necessarily the commit that
    # ADDED the file: a regenerated figure must cite the commit of the code that regenerated it,
    # which can be an EARLIER commit than the one that (re-)added the PNG, if no relevant code
    # changed between them. Because this function reads HEAD at generation time (necessarily the
    # last real commit, since the one now being prepared doesn't exist yet), a figure generated
    # from code with uncommitted changes will show a stale value until corrected in a follow-up
    # commit once the reproducing commit exists (see notes/decisions.md's C1 audit for the
    # per-figure correction and, before it, the same fix applied via first-add instead of this
    # invariant, which was itself wrong for two figures -- corrected in the follow-up commit).
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"
